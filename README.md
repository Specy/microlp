# microlp

[![Crates.io](https://img.shields.io/crates/v/microlp.svg)](https://crates.io/crates/microlp)
[![Documentation](https://docs.rs/microlp/badge.svg)](https://docs.rs/microlp/)

A linear programming solver: it finds the minimum (or maximum) of a linear
function of a set of variables subject to linear equality and inequality
constraints. Variables can be real, integer or boolean.

This is a fork of the archived [minilp](https://github.com/ztlpn/minilp) crate, which was made to fix some bugs, add MILP solving, new features and allow the community to make issues and PRs.

### Disclaimer

I cannot guarantee that the solver always gives optimal solutions (nor that it is bug free), but I'm trying to expand the test suite to cover most cases and catch any bugs that come with it. If you find a bug, want to contribute new testcases or new features, consider reporting it or contributing and sending a PR.

## Getting started

You can use [microlp](https://crates.io/crates/microlp) on its own, but it's
recommended to use it with [good_lp](https://github.com/rust-or/good_lp) or
with the [rooc modeling language](https://github.com/specy/rooc), as they make
it easier to write models. The sections below show how to use microlp
directly.

## Features

* Pure Rust, no dependencies on native code. Runs on WebAssembly.
* Real, integer and boolean variables.
* Time limits with possibility to edit and resume the solve.
* Edit problems after solving and resume the solve from scratch.
* Warm starts from a known solution.
* Handles problems with hundreds of thousands of variables and constraints.

Integer and boolean variables are handled with branch & bound. The library is
already quite powerful and fast, but it may still cycle or lose precision on
some hard problems — please report bugs and contribute code!

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

// The optimum is 7, at x = 1, y = 3.
let solution = problem.solve().unwrap();
assert_eq!(solution.objective(), 7.0);
assert_eq!(solution.var_value(x), 1.0);
assert_eq!(solution.var_value(y), 3.0);
```

## Defining a problem

A `Problem` is created with an optimization direction and filled with
variables and constraints. Each variable is defined by its objective
coefficient and its bounds, and is represented by a `Variable` handle that you
keep to reference it later.

```rust
use microlp::{ComparisonOp, LinearExpr, OptimizationDirection, Problem};

let mut problem = Problem::new(OptimizationDirection::Minimize);

let x = problem.add_var(1.5, (0.0, f64::INFINITY)); // real, x >= 0
let y = problem.add_integer_var(1.0, (-10, 10));    // integer, -10 <= y <= 10
let z = problem.add_binary_var(2.0);                // boolean: 0 or 1
```

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

## Solving and reading the solution

`solve()` returns `Error::Infeasible` when the constraints contradict each
other and `Error::Unbounded` when the objective can grow forever. Invalid
numeric options return `Error::InvalidOptions`, a call that is illegal for the
solution's state (e.g. editing an interrupted solution before `resume()`)
returns `Error::InvalidOperation`, and unrecoverable numerical failures return
`Error::InternalError`. Running out of time is not an error: it returns an
`Ok(Solution)`, and `status()` tells you what you got:

* `Status::Optimal` — the proven optimum.
* `Status::Feasible` — a limit was hit first; this is the best solution found
  so far, valid but not proven optimal. `gap()` reports how close the proof
  got.
* `Status::Interrupted` — a limit was hit before any solution was found.
  The accessors return the solver's in-progress values, which can be useful
  for logging but are not an answer; check `status()` before using them.

```rust
use microlp::Status;

let solution = problem.solve()?;
match solution.status() {
    Status::Optimal => println!("optimum: {}", solution.objective()),
    Status::Feasible => println!(
        "best so far: {} (gap: {:?})",
        solution.objective(),
        solution.gap()
    ),
    Status::Interrupted => println!("no solution yet, keep solving"),
}
```

Values are read per variable or by iterating. For integer and boolean
variables `var_value` returns an exact integer, and the reported objective is
computed from exactly the values you read:

```rust
let value = solution.var_value(x);   // rounded to an exact integer for
                                     // integer/boolean variables
let raw = solution.var_value_raw(x); // without the rounding
let same = solution[x];              // indexing works too, this is equivalent to var_value()

for (var, value) in &solution {
    println!("{}: {}", var.idx(), value);
}
```

`stats()` returns counters for the solve so far: branch & bound nodes, simplex
iterations, elapsed time, the best proven bound and the current gap.

## Time limits and resuming

Set a wall-clock budget with `set_time_limit`. When it runs out the solve
stops and returns whatever it has (`Feasible` or `Interrupted`); `resume`
continues the same search from where it stopped, with a fresh budget.
`resume(None)` means "run to completion".

```rust
use microlp::Status;
use std::time::Duration;

problem.set_time_limit(Duration::from_millis(100));
let mut solution = problem.solve()?;

while solution.status() != Status::Optimal {
    // Do something between slices: log progress, check for shutdown, ...
    solution = solution.resume(Some(Duration::from_millis(100)))?;
}
```

Two things to know about limits:

* They are checked periodically, so a solve can run slightly past its budget —
  on very large problems, noticeably past it. The clock starts when `solve()`
  is called; building the problem is not counted.
* Solves are deterministic: the same problem with the same options always
  gives the same result, and a solve interrupted and resumed any number of
  times ends with exactly the same solution as an uninterrupted one.

## Solve options

`solve_with` accepts a `SolveOptions` for everything beyond a plain time
limit. Start from the default and change what you need:

```rust
use microlp::SolveOptions;
use std::time::Duration;

let mut options = SolveOptions::default();
options.time_limit = Some(Duration::from_secs(10));
options.node_limit = Some(50_000); // deterministic alternative to a time limit
options.mip_gap = 0.01;            // accept anything within 1% of optimal
options.warm_start = Some(vec![(x, 1.0), (y, 3.0)]);

let solution = problem.solve_with(options)?;
```

* `time_limit`, `node_limit` — budgets for this call; each `resume` gets a
  fresh one. `node_limit` counts branch & bound nodes, so it is reproducible
  across machines.
* `mip_gap` — stop as soon as the solution is proven within this relative
  distance of the optimum, and report it as `Optimal`. The default `0.0`
  proves exact optimality.
* `warm_start` — a starting assignment (it may cover only some variables).
  A good one gives the solver an immediate solution to improve on. It is
  advisory: if it isn't usable it is ignored, and it does not carry over any
  proof — to continue an earlier search, use `resume` instead.
* `int_tol`, `tolerances` — numeric tolerances. The defaults are the safe
  choice; values must be finite and non-negative, and the two integrality
  tolerances must be below `0.5`. Invalid settings are rejected instead of
  being allowed to corrupt branching or pruning. See the
  [documentation](https://docs.rs/microlp/) before changing them.

## Changing a solved problem

A `Solution` can be edited and re-solved: `add_constraint` adds a new
constraint, `fix_var` pins a variable to a value, `unfix_var` releases a
previous fix. Each call consumes the solution and returns a new, re-solved
one, keeping as much earlier work as possible — in particular, the previous
solution is used as a starting point whenever it is still valid for the
changed problem.

Problems with integer variables can be edited whatever the status, including
a paused (`Feasible`/`Interrupted`) solve. Pure-LP solutions must reach
`Optimal` before they can be edited — `resume` them first.

A complete example — solve, react to the result, tighten the problem, and
keep solving:

```rust
use microlp::{ComparisonOp, Error, OptimizationDirection, Problem, Status};
use std::time::Duration;

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
    // on a real model this is where you might get Feasible instead.)
    problem.set_time_limit(Duration::from_secs(1));
    let mut solution = problem.solve()?;
    println!("plan is worth {}", solution.objective());

    // Requirements change after seeing the plan: a customer needs at least
    // 3 units of product 0, and product 3 is out of stock. Edits apply to
    // the problem as you originally defined it, plus the other edits.
    solution = solution.add_constraint(&[(units[0], 1.0)], ComparisonOp::Ge, 3.0)?;
    solution = solution.fix_var(units[3], 0.0)?;
    println!("revised plan is worth {}", solution.objective());

    // Product 3 is available again: release it and solve to proven
    // optimality, resuming in one-second slices.
    let (mut solution, _was_fixed) = solution.unfix_var(units[3])?;
    while solution.status() != Status::Optimal {
        solution = solution.resume(Some(Duration::from_secs(1)))?;
    }
    for (i, &unit) in units.iter().enumerate() {
        println!("product {}: {} units", i, solution.var_value(unit));
    }
    Ok(())
}
```

An edit that makes the problem unsolvable returns `Err(Error::Infeasible)`
from the edit call itself.

On pure-LP solutions there is one more operation, `add_gomory_cut(var)`,
which adds a cutting plane that pushes the given variable's fractional value
toward an integer — a building block for experimenting with cutting-plane
methods:

```rust
let mut solution = lp_problem.solve()?;
while solution.var_value(x).fract() != 0.0 {
    solution = solution.add_gomory_cut(x)?;
}
```

## Testing

Besides unit tests, the repository ships a correctness suite of 200+ LP/MILP
problems with independently known answers:

```bash
cargo test --release --test suite            # default set, a few seconds
cargo test --release --test suite -- --hard  # plus the long-running ones
```

See [tests/suite/README.md](tests/suite/README.md) for how it works, and
[src/ARCHITECTURE.md](src/ARCHITECTURE.md) if you want to know how the solver
itself works.

There is also a harder tier called `xhard` which microlp currently cannot solve in reasonable time. (under 10 minutes) which you can run with. This tier is being used to compare with other solvers.
```
cargo test --release --test suite -- --xhard
```

## Benchmarks

[BENCHMARK.md](https://github.com/Specy/microlp/blob/master/BENCHMARK.md)
compares microlp against other open-source solvers on the suite's benchmark
instances, and tracks how close it gets on the instances it cannot yet solve
within a time budget. Regenerate it from a clone of this repository (the
crates.io package does not include the benchmark data) with:

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
