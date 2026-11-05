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

microlp requires Rust 1.88 or later.

## Features

* Pure Rust. Runs on WebAssembly.
* Real, integer and boolean variables.
* Special ordered sets of type 1 and 2, for example for piecewise-linear
  functions that are not convex.
* Time limits and MIP gaps. You can edit a solved problem and resume an
  interrupted solve.
* Warm starts from a known solution.
* Dual values (shadow prices) and reduced costs for linear programs.
* Handles problems with hundreds of thousands of variables and constraints.

Problems with integer or boolean variables, or special ordered sets, are
solved with branch & bound. On hard or ill-conditioned problems, a solve can
fail with `Error::InternalError` instead of returning an answer. Please report
bugs and contribute code.

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
that created it, so use it only with that problem and its solutions.

If a handle's index is absent from the problem, accessors and
`Problem::add_constraint` panic. `Solution::add_constraint` and a solve with
that index in its warm-start hint return `Error::InvalidOperation`.

```rust
use microlp::{ComparisonOp, LinearExpr, OptimizationDirection, Problem};

let mut problem = Problem::new(OptimizationDirection::Minimize);

let x = problem.add_var(1.5, (0.0, f64::INFINITY)); // real, x >= 0
let y = problem.add_integer_var(1.0, (-10, 10));    // integer, -10 <= y <= 10
let z = problem.add_binary_var(2.0);                // boolean: 0 or 1
```

A bound with magnitude 2^52 (about 4.5e15) or more on its side of infinity is
treated as infinite. An upper bound of `f64::MAX` or `i64::MAX` therefore means
there is no upper bound. When `min >= max`, microlp keeps the
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

`add_constraint` returns a `Constraint` handle which you can use to read the
constraint's dual value from a solution (see [Duals](#duals)). Like a
`Variable`, the handle belongs to the problem that created it.

## Solving and reading the solution

`solve()` searches for an optimal solution. Every solution meets each
constraint and variable bound within an absolute feasibility tolerance of
`1e-7` by default (`Tolerances::feasibility`), or the round-off of evaluating
the constraint or bound in `f64` where that is larger. If the solve cannot
return a verified solution, it returns an error:

* `Error::Infeasible`: the problem has no solution. No assignment meets every
  constraint and bound within the feasibility tolerance while every integer
  variable takes a whole number. See [`Error::Infeasible`] for the exact
  conditions.
* `Error::Unbounded`: the objective can improve forever.
* `Error::InternalError`: a numerical failure the solver could not recover
  from.
* `Error::InvalidOptions` reports an invalid numeric option.
  `Error::InvalidOperation` reports invalid input. Both are checked before
  solving.

An optimal solution is optimal up to numerical tolerances. This does not
guarantee the best objective over every assignment the tolerances allow.
When a very large coefficient sits beside small ones, a tolerance-sized
difference in one variable can make the reported objective differ noticeably
from the exact optimum.

Keep the largest nonzero coefficient magnitude at most `1e6` times the
smallest in each constraint and for each variable across constraints.
Changing the units of a variable or of a constraint usually brings a model
into this range. Outside it, microlp still accepts the model and tries to
solve it, but `Error::InternalError` becomes more likely. The error message
names the constraint and variable with the widest ranges, reporting each
only if its range exceeds the recommendation. Please report an
`Error::InternalError` that occurs inside the recommended range.

Constraints that nearly repeat one another can make a solve fail with
`Error::InternalError`. In rare cases, a feasible model can be reported
`Infeasible`, or the returned solution can be far from the optimum. A common
cause is free variables that can all shift together, such as every bus angle
in a power-flow model. Fix one of them, such as a reference angle, to remove
the shared shift. See the crate's [known limitations] for the full list.

A successful solve returns a `SolveOutcome`, which is one of:

* `SolveOutcome::Solution` contains a validated assignment. Its status is
  `SolutionStatus::Optimal` when the solver proved optimality with no MIP gap,
  or `SolutionStatus::Feasible` when an incumbent (the best solution found so
  far) is available but optimality is unproved, for example because a time or
  node limit was reached.
* `SolveOutcome::Interrupted` means a time or node limit was reached before
  the solver found a usable incumbent. It provides the termination reason and
  statistics, but no objective or variable-value accessors because no solution
  is available. It doesn't mean the problem has no solution, you can resume
  the solve with `resume()` to continue the search.

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

Read a variable's value with `var_value` or by iterating over a `Solution`.
An integer or boolean variable's value is within `int_tol` (default `1e-6`)
of an integer, but is not necessarily that integer. Values like `0.9999999`
are normal. A variable fixed at an integer by its bounds, the solve or an
edit lies on that integer. Use `.round()` when you need an exact integer
(`var_domain` identifies integer and boolean variables), but remember that
rounded values need not meet the constraints. Keep big-M coefficients no
larger than the model needs.

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

[`stats()`] reports LP iterations, elapsed time, the best proven bound and
the current gap. For MILPs, it also reports the number of nodes solved.

### Duals

For a problem without integer or boolean variables, a `Solution` also reports
dual values. [`dual(constraint)`] is the constraint's shadow price, the rate
at which the optimal objective changes as its right-hand side increases.
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

## Time limits and resuming

Set a time limit with `set_time_limit`. When the limit expires, the outcome
contains either the best solution found so far or an interrupted solve.
[`resume()`] continues with fresh time and node budgets using the preceding
call's settings. [`resume_with`] replaces those settings with the options you
supply.

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
its limit. A duration too large to represent as a time limit, such as
`Duration::MAX`, means no limit.

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
* `mip_gap`: stop when the best solution found so far is proven within this
  relative distance of the optimum and report a feasible solution with
  termination reason `MipGap`. The default `0.0` keeps solving until
  optimality is proven.
* `warm_start`: a starting assignment (it may cover only some variables).
  A good one gives the solver an immediate solution to improve on. It is
  advisory, if it isn't usable it is ignored, and a problem with only real
  variables does not use it. A value that is NaN or infinite, or a variable
  outside the problem, returns `Error::InvalidOperation`.
* `int_tol`, `tolerances`: `int_tol` (default `1e-6`) is how far a reported
  integer or boolean value may be from a whole number.
  `tolerances.feasibility` (default `1e-7`) is how far a solution may lie
  outside a constraint or variable bound, as an absolute amount. A model
  feasible only at slightly nonintegral integer values is normally reported
  `Infeasible`.
* `presolve` (on by default): simplifies the problem before solving, making
  most solves faster. Every variable is still reported, and the solution
  returned is one of the problem as you wrote it. Turn it off only to
  compare behavior or to rule it out when investigating a numerical problem.
* `dual_fixing` (on by default): a presolve shortcut for integer variables.
  It keeps an optimal solution but can rule out other solutions, including
  equally good ones. Turn it off when every integer solution must stay
  reachable, for example when you enumerate solutions by adding constraints
  and solving again. A `warm_start` hint turns it off.

See [`SolveOptions`] and [`Tolerances`] for accepted values and exact
conditions.

## Changing a solved problem

You can edit a `Solution` and solve the changed problem. `add_constraint`
adds a constraint, `set_rhs` changes a right-hand side, and `set_obj_coeff`
changes an objective coefficient. `fix_var` fixes a variable to a value, and
`unfix_var` restores its original bounds.

Each edit consumes the solution, even if it returns an error. Clone it first
if you need to keep the preceding solution. The calls return a new
`SolveOutcome`. `add_constraint` also returns the new constraint handle, and
`unfix_var` reports whether it released a fix.

Edits to LPs continue from the preceding solve, which is usually much faster
than starting over. An infinite right-hand side on a constraint's open side
switches it off, and a finite one switches it back on. See [`Solution`] for
edit conditions, errors, and the settings each call uses.

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

You can edit a `Solution` whose optimality is unproved. An interrupted
outcome has no validated assignment to edit.

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

Edits return the same errors as solving. `Error::Infeasible` means the changed
problem has no solution. `Error::Unbounded` means its objective can improve
forever, which can happen after `set_rhs` switches off the constraint that
limited it, after `set_obj_coeff`, or after `unfix_var`.

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
interruptions and checks the solutions it returns against externally certified
results:

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

The rival solvers set themselves up during the build so you must have a C++
toolchain, CMake and libclang:

* **Linux** (Debian/Ubuntu): `sudo apt install cmake build-essential libclang-dev libgfortran5`
* **macOS**: `xcode-select --install`, then `brew install cmake`
* **Windows**: Visual Studio Build Tools (C++ workload), [CMake](https://cmake.org)
  and [LLVM](https://github.com/llvm/llvm-project/releases) for
  `libclang.dll` — or `pip install libclang` and point `LIBCLANG_PATH` at
  its `native` folder

A solver you cannot build can be left out, e.g.
```bash
cargo run -p microlp-benchmark --release --no-default-features --features highs,clarabel
```

## License

This project is licensed under the [Apache License, Version 2.0](./LICENSE).

[`Error::Infeasible`]: https://docs.rs/microlp/latest/microlp/enum.Error.html#variant.Infeasible
[known limitations]: https://docs.rs/microlp/latest/microlp/#known-limitations
[`stats()`]: https://docs.rs/microlp/latest/microlp/struct.Solution.html#method.stats
[`dual(constraint)`]: https://docs.rs/microlp/latest/microlp/struct.Solution.html#method.dual
[`resume()`]: https://docs.rs/microlp/latest/microlp/enum.SolveOutcome.html#method.resume
[`resume_with`]: https://docs.rs/microlp/latest/microlp/enum.SolveOutcome.html#method.resume_with
[`SolveOptions`]: https://docs.rs/microlp/latest/microlp/struct.SolveOptions.html
[`Tolerances`]: https://docs.rs/microlp/latest/microlp/struct.Tolerances.html
[`Solution`]: https://docs.rs/microlp/latest/microlp/struct.Solution.html
