# microlp

[![Crates.io](https://img.shields.io/crates/v/microlp.svg)](https://crates.io/crates/microlp)
[![Documentation](https://docs.rs/microlp/badge.svg)](https://docs.rs/microlp/)

microlp is a Rust library for solving linear programs (LPs) and mixed-integer
linear programs (MILPs). It finds the minimum or maximum of a linear objective
subject to equality and inequality constraints. Variables can be real, integer
or boolean.

### Disclaimer

I cannot guarantee that the solver always gives optimal solutions (nor that it is bug free), but I'm trying to expand the test suite to cover most cases and catch any bugs that come with it. If you find a bug, want to contribute new testcases or new features, consider reporting it or contributing and sending a PR.

## Getting started

You can use [microlp](https://crates.io/crates/microlp) directly. For easier
model construction, use it with the [rooc modeling
language](https://github.com/specy/rooc) or [good_lp](https://github.com/rust-or/good_lp).

The examples below use microlp directly.

microlp requires Rust 1.86 or later. The latest releases of its `sprs`
dependency need Rust 1.88, so on Rust 1.86 and 1.87 resolve dependencies with
the MSRV-aware resolver (`resolver = "3"`, the default of edition 2024), or
run `cargo update -p sprs --precise 0.11.4`.

## Features

* Pure Rust. Runs on WebAssembly.
* Real, integer and boolean variables.
* Time limits and MIP gaps. You can edit a solved problem and resume an
  interrupted solve.
* Warm starts from a known solution.
* Dual values (shadow prices) and reduced costs for linear programs.
* Handles problems with hundreds of thousands of variables and constraints.

Problems with integer or boolean variables are solved with branch & bound. On
hard or ill-conditioned problems, a solve can fail with `Error::InternalError`
instead of returning an answer. Please report bugs and contribute code.

## Basic usage

```rust
use microlp::{Problem, OptimizationDirection, ComparisonOp};

// Maximize x + 2y, where x is real with x >= 0 and y is an integer
// with 0 <= y <= 3.
let mut problem = Problem::new(OptimizationDirection::Maximize);
let x = problem.add_var(1.0, (0.0, f64::INFINITY));
let y = problem.add_integer_var(2.0, (0, 3));

// Subject to x + y <= 4 and 2x + y >= 2.
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

// The optimum is 7, at x = 1, y = 3. An integer variable's value is
// reported within `int_tol` of its integer: round it for an exact one.
let solution = problem.solve().unwrap().into_solution().unwrap();
assert!((solution.objective() - 7.0).abs() < 1e-6);
assert!((solution.var_value(x) - 1.0).abs() < 1e-6);
assert_eq!(solution.var_value(y).round(), 3.0);
```

## Defining a problem

Create a `Problem` with an optimization direction, then add variables and
constraints. Each variable has an objective coefficient and bounds. Its
`Variable` handle lets you refer to it later. A handle belongs to the problem
that created it, so use it only with that problem and its solutions. A problem
does not detect a handle from another problem; the handle refers to the
variable with the same index. If that index is absent, accessors and
`Problem::add_constraint` panic, while `Solution::add_constraint`, and a
solve whose warm-start hint names that index, return
`Error::InvalidOperation`.

```rust
use microlp::{ComparisonOp, LinearExpr, OptimizationDirection, Problem};

let mut problem = Problem::new(OptimizationDirection::Minimize);

let x = problem.add_var(1.5, (0.0, f64::INFINITY)); // real, x >= 0
let y = problem.add_integer_var(1.0, (-10, 10));    // integer, -10 <= y <= 10
let z = problem.add_binary_var(2.0);                // boolean: 0 or 1
```

A bound with magnitude 2^52 (about 4.5e15) or more on its side of infinity is
treated as infinite. An upper bound of `f64::MAX` or `i64::MAX` therefore means
there is no upper bound. A real bound of that magnitude cannot be enforced, so
a solution can lie beyond it. microlp checks each bound separately:
`(5e15, f64::MAX)` means `x >= 5e15`. When `min >= max`, microlp keeps the
bounds as given, so a variable fixed at such a value stays fixed and an empty
range stays infeasible. microlp uses a constraint's right-hand side as given,
even when its magnitude is that large. You can express a bound such as
`x <= 5e15` as a constraint.

The left-hand side of a constraint can be given as a slice of
`(variable, coefficient)` pairs, as any iterator of such pairs, or as a
`LinearExpr` built term by term:

```rust
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);

let vars = [x, y];
problem.add_constraint(vars.iter().map(|&v| (v, 2.0)), ComparisonOp::Ge, 2.0);

let mut lhs = LinearExpr::empty();
lhs.add(x, 1.0);
lhs.add(z, -3.0);
problem.add_constraint(lhs, ComparisonOp::Eq, 0.0);
```

`add_constraint` returns a `Constraint` handle. Use it to read the constraint's
dual value from a solution (see [Duals](#duals)). Like a `Variable`, the handle
belongs to the problem that created it.

## Solving and reading the solution

`solve()` searches for an optimal solution. Every solution meets each
constraint and variable bound within an absolute feasibility tolerance: `1e-7`
by default (`Tolerances::feasibility`), or the round-off of evaluating the
constraint or bound in `f64` where that is larger. microlp checks each solution
against the problem as you wrote it. If the solve cannot return a verified
solution, it returns an error:

* `Error::Infeasible`: the solver proved, from the problem's data, that no
  assignment with every integer variable at an integer within its bounds, and
  every variable whose bounds are equal at its value, meets the constraints
  and the other bounds within that tolerance (see `Tolerances::feasibility`).
* `Error::Unbounded`: the objective can improve forever.
* `Error::InternalError`: a numerical failure the solver could not recover
  from. It returns this rather than an answer it could not verify.
* `Error::InvalidOptions` reports an invalid numeric option.
  `Error::InvalidOperation` reports invalid input. Both are checked before
  solving. Every model number must be finite, except that a bound or a
  constraint's right-hand side may be infinite on the side it leaves open:
  `f64::NEG_INFINITY` as a lower bound or on a `Ge` constraint, and
  `f64::INFINITY` as an upper bound or on a `Le` constraint. Those values
  leave that side unconstrained. A warm start hint must give finite values to
  variables in this problem.

The solver proves optimality for the point where it ends, within the
tolerance. It does not prove optimality across every point the tolerance
admits. On an ill-conditioned problem, where a small coefficient sits beside a
large one, the tolerance can move the objective far from the exact optimum.

Keep nonzero coefficients within six orders of magnitude in each constraint
and across all constraints for each variable. In other words, the largest
magnitude should be at most `1e6` times the smallest. microlp does not reject
or change coefficients outside this range, and it still checks every solution
against the model as you wrote it, but a solve can then fail with
`Error::InternalError`. Its message names the constraint (in the order added)
and the variable (by index) whose coefficients span the most, those of them
that lie outside the range. Changing the units of a variable or
constraint usually brings a model back into range. Within the recommended
range, an `Error::InternalError` is a bug or one of the known limitations
below. Please report it.

Nearly dependent constraints can exceed floating-point resolution even within
the recommended coefficient range. A solve can then fail with
`Error::InternalError`. In extreme cases, the solver reports a feasible model
as `Infeasible` when every feasible assignment requires very large variable
values. If the final basis is singular to `f64`, the solver can report an
unbounded model as solved, or return a solution far from a bounded model's
optimum.

A common source is a group of free variables that can all shift together, such
as every bus angle in a power-flow model left free. Rounding in the
coefficients can make that shift have only a tiny effect on the constraints,
so a model meant to have no solution can have one as written at very large
variable values. A solve can return that solution. Fix one of those variables,
such as a reference angle, to remove the shift. Outside the recommended range,
a bounded model can also be reported as `Unbounded`.

If a product of an objective coefficient and variable value, or the sum of
those terms, overflows `f64`, microlp cannot evaluate the objective and the
solve fails with `Error::InternalError`, even when the exact objective is
representable. See the
[known numerical limitations](src/ARCHITECTURE.md#79-known-limitations).

A successful solve returns a `SolveOutcome`, which is one of:

* `SolveOutcome::Solution` contains a validated assignment. Its status is
  `SolutionStatus::Optimal` when the solver proved optimality with no MIP gap,
  or `SolutionStatus::Feasible` when an incumbent (the best solution found so
  far) is available but optimality is unproved, for example because a time or
  node limit was reached.
* `SolveOutcome::Interrupted` means a time or node limit was reached before
  the solver found a usable incumbent. It provides the termination reason and
  statistics, but no objective or variable-value accessors because no solution
  is available. It doesn't mean the problem has no solution: resume the solve
  with `resume()` to continue the search.

`termination_reason()` distinguishes `ProvenOptimal`, `MipGap`, `TimeLimit`,
and `NodeLimit`.
```rust
use microlp::{SolutionStatus, SolveOutcome};

match problem.solve()? {
    SolveOutcome::Solution(solution) => match solution.status() {
        SolutionStatus::Optimal => println!("optimum: {}", solution.objective()),
        SolutionStatus::Feasible => println!(
            "best so far: {} (gap: {:?}, reason: {:?})",
            solution.objective(),
            solution.gap(),
            solution.termination_reason()
        ),
    },
    SolveOutcome::Interrupted(interrupted) => {
        println!("no solution yet: {:?}", interrupted.termination_reason());
    }
}
```

Read a variable's value with `var_value` or by iterating over a `Solution`;
these are the values the solver computed. The reported objective uses exactly
those values. An integer or boolean variable's value is within `int_tol`
(default `1e-6`) of an integer, but is not necessarily that integer. Values
like `0.9999999` are normal. A variable fixed at an integer by its bounds, the
search or an edit lies on that integer. Use `.round()` when you need an exact
integer (`var_domain` tells you which variables are integer or boolean), but
remember that rounded values need not meet the constraints. Under a big-M
constraint such as `x <= 1e7 * y`, a `y` value of `1e-6` allows `x` up to 10,
which rounding `y` to zero forbids. Keep big-M coefficients no larger than the
model needs.

```rust
use microlp::VarDomain;

let value = solution.var_value(x);           // as computed
let count = solution.var_value(y).round();   // an exact integer
let same = solution[x];                      // indexing is equivalent to var_value()

for (var, value) in &solution {
    // Exact integers for the integer and boolean variables (`+ 0.0` turns
    // the `-0` that rounds a value just below zero into `0`).
    let value = match solution.var_domain(var) {
        VarDomain::Integer | VarDomain::Boolean => value.round() + 0.0,
        VarDomain::Real => value,
    };
    println!("{}: {}", var.idx(), value);
}
```

`stats()` returns counters for the solve so far: branch & bound nodes, simplex
iterations, elapsed time, the best proven bound and the current gap.

### Duals

For a problem without integer or boolean variables, a `Solution` also reports
dual values. `dual(constraint)` is the constraint's shadow price: the rate at
which the optimal objective changes as its right-hand side increases.
`duals()` returns the dual value for each constraint, in the order they were
added. `reduced_cost(var)` returns a variable's objective coefficient minus
the sum of its constraint coefficients times the corresponding dual values.
All three methods return `None` for a problem with integer or boolean
variables.

```rust
use microlp::{ComparisonOp, OptimizationDirection, Problem};

// Meet a demand of 10 from two generators, the cheaper one limited to 6.
let mut problem = Problem::new(OptimizationDirection::Minimize);
let cheap = problem.add_var(20.0, (0.0, f64::INFINITY));
let dear = problem.add_var(50.0, (0.0, f64::INFINITY));
let demand = problem.add_constraint(&[(cheap, 1.0), (dear, 1.0)], ComparisonOp::Eq, 10.0);
let limit = problem.add_constraint(&[(cheap, 1.0)], ComparisonOp::Le, 6.0);
let solution = problem.solve()?.into_solution().expect("an LP solves to optimality");

// One more unit of demand comes from the dearer generator (50), and one more
// unit of the cheap one's limit saves 50 - 20.
println!("demand: {}", solution.dual(demand).unwrap()); // 50
println!("limit: {}", solution.dual(limit).unwrap()); // -30
```

The dual values come from the final simplex basis. When several sets of dual
values are optimal, this is one of them, and a different optimal basis can give
different values.
`Solution::add_constraint` returns a handle for the constraint it adds,
numbered after the problem's constraints and any constraints added before it.

## Time limits and resuming

Set a time limit with `set_time_limit`. When the limit expires, the outcome
contains either a feasible incumbent or an interrupted search. `resume()`
continues the same search with the per-call options from the immediately
preceding solve, resume or edit.

Use `resume_with` to resume with different settings.

```rust
use microlp::{ResumeOptions, TerminationReason};
use std::time::Duration;

problem.set_time_limit(Duration::from_millis(100));
let mut outcome = problem.solve()?;

while matches!(
    outcome.termination_reason(),
    TerminationReason::TimeLimit | TerminationReason::NodeLimit
) {
    // Do something between slices: log progress, check for shutdown, ...
    let mut options = ResumeOptions::default();
    options.time_limit = Some(Duration::from_millis(100));
    outcome = outcome.resume_with(options)?;
}
```

The solver checks time limits periodically, so a solve can run slightly past
its limit. If the limit is too long to add to the clock, such as
`Duration::MAX`, the solve treats it as no limit.

## Solve options

Use `solve_with` to set options beyond a time limit. Start with the defaults
and change the options you need:

```rust
use microlp::SolveOptions;
use std::time::Duration;

let mut options = SolveOptions::default();
options.time_limit = Some(Duration::from_secs(10));
options.node_limit = Some(50_000); // deterministic alternative to a time limit
options.mip_gap = 0.01;            // accept anything within 1% of optimal
options.warm_start = Some(vec![(x, 1.0), (y, 3.0)]);

let outcome = problem.solve_with(options)?;
```

* `time_limit`, `node_limit`: execution budgets for this solve call.
* `mip_gap`: stop as soon as the solution is proven within this relative
  distance of the optimum, and report a feasible solution with termination
  reason `MipGap`. The default `0.0` requires exact optimality.
* `warm_start`: a starting assignment (it may cover only some variables).
  A good one gives the solver an immediate solution to improve on. It is
  advisory: if it isn't usable it is ignored, and a problem with only real
  variables does not use it. A value that is NaN or infinite, or a variable
  outside the problem, returns `Error::InvalidOperation`.
* `int_tol`, `tolerances`: numeric tolerances. For acceptance, a value within
  `int_tol` of an integer counts as integral (see `presolve` below). Reported
  integer values are within `int_tol` of an integer; a fixed integer variable
  is on its integer. The feasibility tolerance is absolute and applies to
  every variable bound and constraint.
  Values must be finite and non-negative, the feasibility tolerance at least
  `1e-10`, and `int_tol` below `0.5`.
* `presolve` (on by default): simplify the problem before the search. It never
  removes a variable or changes a real variable's bounds, and it keeps, up to
  round-off, every solution whose integer variables are exactly integral, or,
  where `dual_fixing` fixes a variable because of the objective, an optimal
  one; every solution is still checked against the problem as you wrote it. A
  solution relying on `int_tol` (an integer variable just off its integer
  under a big-M coefficient) is never sought: it is returned where the search
  lands on one, and where the objective can improve forever such a landing
  answers `Unbounded`. A problem with no solution whose integer variables are
  integral is answered `Infeasible` unless that happens, with presolve on or
  off.
* `dual_fixing` (on by default): let presolve fix an integer variable at a
  bound that neither the objective nor any constraint gives it a reason to
  leave. That keeps an optimal solution, not every solution: turn it off when
  every integral solution must stay reachable, for example to enumerate the
  solutions within an objective cutoff. A `warm_start` hint turns it off.

## Changing a solved problem

You can edit a `Solution` and re-solve the problem. `add_constraint` adds a
constraint and returns its `Constraint` handle. `set_rhs` changes a
constraint's right-hand side, `set_obj_coeff` changes a variable's objective
coefficient, `fix_var` pins a variable to a value, and `unfix_var` releases a
previous fix. Each edit consumes the solution and re-solves the model. The
calls return a new `SolveOutcome`; `add_constraint` also returns the new
constraint handle, and `unfix_var` reports whether it released a fix. The
solver reuses as much earlier work as possible, including the previous
solution as a starting point whenever it remains valid for the changed
problem.

For a problem without integer or boolean variables, an edit re-solves from
the previous basis: a new right-hand side or a fix keeps it dual feasible, a
new objective coefficient keeps it primal feasible, so the re-solve usually
takes a few pivots. This is the cheap way to price a scenario or sweep a
parameter. An infinite right-hand side on the side a constraint leaves open
switches it off, and a finite one switches it back on.

```rust
use microlp::{ComparisonOp, OptimizationDirection, Problem};

let mut problem = Problem::new(OptimizationDirection::Minimize);
let cheap = problem.add_var(20.0, (0.0, 6.0));
let dear = problem.add_var(50.0, (0.0, f64::INFINITY));
let demand = problem.add_constraint(&[(cheap, 1.0), (dear, 1.0)], ComparisonOp::Ge, 0.0);
let mut solution = problem.solve().unwrap().into_solution().unwrap();
// The cost of each demand level, each solve warm from the one before.
for level in [2.0, 4.0, 6.0, 8.0, 10.0] {
    solution = solution.set_rhs(demand, level).unwrap().into_solution().unwrap();
    println!("demand {level}: cost {}, price {}", solution.objective(), solution.dual(demand).unwrap());
}
```

Only a `Solution` can be edited. An interrupted outcome has no edit methods
because it has no validated assignment. A feasible-but-unproven `Solution`
can be edited. Each edit returns a `SolveOutcome`, so you can check whether
the re-solve found a solution or was interrupted.

```rust
use microlp::{
    ComparisonOp, Error, OptimizationDirection, Problem, SolveOutcome,
};
use std::time::Duration;

fn resume_until_answer(mut outcome: SolveOutcome) -> Result<SolveOutcome, Error> {
    while outcome.solution().is_none() {
        outcome = outcome.resume()?;
    }
    Ok(outcome)
}

fn main() -> Result<(), Error> {
    // How many units of each product to make (0 to 10 each), maximizing
    // profit under a machine-hours budget.
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let profit = [12.0, 9.0, 7.0, 15.0];
    let hours = [4.0, 3.0, 2.0, 6.0];
    let units: Vec<_> = profit
        .iter()
        .map(|&p| problem.add_integer_var(p, (0, 10)))
        .collect();
    problem.add_constraint(
        units.iter().zip(&hours).map(|(&v, &h)| (v, h)),
        ComparisonOp::Le,
        100.0,
    );

    // First pass, with a time budget. (This toy instance solves instantly;
    // on a real model this may first return Interrupted.)
    problem.set_time_limit(Duration::from_secs(1));
    let outcome = resume_until_answer(problem.solve()?)?;
    let solution = outcome
        .into_solution()
        .expect("resume_until_answer guarantees a solution");
    println!("plan is worth {}", solution.objective());

    // Requirements change after seeing the plan: a customer needs at least
    // 3 units of product 0, and product 3 is out of stock. Edits apply to
    // the problem as you originally defined it, plus the other edits.
    let (outcome, _at_least_3) =
        solution.add_constraint(&[(units[0], 1.0)], ComparisonOp::Ge, 3.0)?;
    let solution = resume_until_answer(outcome)?
        .into_solution()
        .expect("resume_until_answer guarantees a solution");
    let outcome = solution.fix_var(units[3], 0.0)?;
    let solution = resume_until_answer(outcome)?
        .into_solution()
        .expect("resume_until_answer guarantees a solution");
    println!("revised plan is worth {}", solution.objective());

    // Product 3 is available again: release it and read the new solution.
    let (outcome, _was_fixed) = solution.unfix_var(units[3])?;
    let solution = resume_until_answer(outcome)?
        .into_solution()
        .expect("resume_until_answer guarantees a solution");
    for (i, &unit) in units.iter().enumerate() {
        // Rounded: an integer variable's value is reported as computed.
        let count = solution.var_value(unit).round() + 0.0;
        println!("product {}: {} units", i, count);
    }
    Ok(())
}
```

An edit that makes the problem infeasible returns `Err(Error::Infeasible)`
from the edit call itself.

## Testing

Besides unit tests, the repository ships a correctness suite of 200+ LP/MILP
problems with independently known answers:

```bash
cargo test --release --test suite            # default: easy + medium
cargo test --release --test suite -- --hard  # plus the long-running ones
```

See [tests/suite/README.md](tests/suite/README.md) for how it works, and
[src/ARCHITECTURE.md](src/ARCHITECTURE.md) if you want to know how the solver
itself works.

The origin-feasible property test runs 50,000 cases by default. For a deeper
run, use `HEGEL_TEST_CASES=500000 cargo test --release --lib origin`, which
takes about a minute and some 4 GB of memory.

The `xhard` tier gives larger instances a 10-minute budget. It accepts clean
interruptions and checks returned incumbents against externally certified results:

```
cargo test --release --test suite -- --xhard
```

## Benchmarks

[BENCHMARK.md](https://github.com/Specy/microlp/blob/master/BENCHMARK.md)
records a comparison of an earlier revision against other open-source solvers
on the suite's benchmark instances. Its date and revision identify the code
measured; it does not describe the current release. Regenerate it from a clone
of this repository (the crates.io package does not include the benchmark data)
with:

```bash
cargo run -p microlp-benchmark --release
```

The rival solvers set themselves up during the build: HiGHS is compiled from
source, SCIP is downloaded prebuilt (the first build needs internet access)
and Clarabel is pure Rust. What must already be on the machine is a C++
toolchain, CMake and libclang:

* **Linux** (Debian/Ubuntu): `sudo apt install cmake build-essential libclang-dev libgfortran5`
* **macOS**: `xcode-select --install`, then `brew install cmake`
* **Windows**: Visual Studio Build Tools (C++ workload), [CMake](https://cmake.org)
  and [LLVM](https://github.com/llvm/llvm-project/releases) for
  `libclang.dll` — or `pip install libclang` and point `LIBCLANG_PATH` at
  its `native` folder

Runs are timed, so use an otherwise idle machine.
A solver you cannot build
can be left out, e.g.
```bash
cargo run -p microlp-benchmark --release --no-default-features --features highs,clarabel
```

## License

This project is licensed under the [Apache License, Version 2.0](./LICENSE).
