# microlp
This is a fork of the archived [minilp](https://github.com/ztlpn/minilp) crate, which was made to fix some bugs, add features and allow the community to make issues and PRs.

### ⚠️ Warning ⚠️
If you were using the library prior to 0.2.11, please use the latest version of the library as there was a major bug for integer variables.

[![Crates.io](https://img.shields.io/crates/v/microlp.svg)](https://crates.io/crates/microlp)
[![Documentation](https://docs.rs/microlp/badge.svg)](https://docs.rs/microlp/)

A fast linear programming solver library.

[Linear programming](https://en.wikipedia.org/wiki/Linear_programming) is a technique for
finding the minimum (or maximum) of a linear function of a set of variables
subject to linear equality and inequality constraints.

## Getting started
You can use [microlp](https://crates.io/crates/microlp) on its own, but it's recommended to use it with [goodlp](https://github.com/rust-or/good_lp) or with [rooc modeling language](https://github.com/specy/rooc) as it makes it easier to create models. Look at the examples below on how to use microlp on its own.

## Features

* Pure Rust implementation, WebAssembly-friendly (single-threaded, `web-time` clock).
* Able to solve problems with hundreds of thousands of variables and constraints.
* Continuous, integer and boolean variables; MILPs are solved with branch & bound
  (warm-started dual simplex per node, pseudocost branching, best-bound search).
* Time and node limits that stop **cleanly** — a hit limit is a status, never an
  error or a panic — and solves that `resume()` where they left off.
* Post-solve editing: add constraints, fix/unfix variables on an existing
  `Solution` and re-solve.
* Warm starts from a known solution, MIP gap reporting, solve statistics.
* Problems can be defined via an API or parsed from an
  [MPS](https://en.wikipedia.org/wiki/MPS_(format)) file.

Warning: although the library is already quite powerful and fast, it may cycle
or lose precision on some hard problems. Please report bugs and contribute code!

## Basic usage

```rust
use microlp::{Problem, OptimizationDirection, ComparisonOp};

// Maximize an objective function x + 2 * y of a continuous variable x >= 0
// and an integer variable 0 <= y <= 3.
let mut problem = Problem::new(OptimizationDirection::Maximize);
let x = problem.add_var(1.0, (0.0, f64::INFINITY));
let y = problem.add_integer_var(2.0, (0, 3));

// subject to constraints: x + y <= 4 and 2 * x + y >= 2.
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

// Optimal value is 7, achieved at x = 1 and y = 3.
let solution = problem.solve().unwrap();
assert_eq!(solution.objective(), 7.0);
assert_eq!(solution.var_value(x), 1.0);
assert_eq!(solution.var_value(y), 3.0);
```

For a more involved example, see [examples/tsp](examples#tsp), a solver for the travelling
salesman problem.

## Reading a solution: statuses

`solve()` returns `Err` only when the *problem* has no answer (`Infeasible`,
`Unbounded`). Everything else — including hitting a limit — is an
`Ok(Solution)` whose `status()` tells you what you got:

| `Status` | Meaning | Values |
|---|---|---|
| `Optimal` | proven optimum | trustworthy |
| `Feasible` | a limit was hit; the best incumbent found so far, not proven optimal | a valid feasible point; `gap()` says how far the proof got |
| `Interrupted` | a limit was hit before any usable solution existed | the search's *current working point* — inspection only |

Reading a solution never panics, whatever its status. On `Interrupted` the
accessors expose whatever point the search was at when the limit fired: values
may be fractional for integer variables and may not satisfy all constraints.
That is useful for progress reporting and debugging, but it is not the answer
to your problem — **checking `status()` before treating values as the solution
is the caller's job.**

```rust,ignore
let solution = problem.solve().unwrap();
match solution.status() {
    Status::Optimal => println!("optimum: {}", solution.objective()),
    Status::Feasible => println!(
        "best so far: {} (gap {:?})", solution.objective(), solution.gap()
    ),
    Status::Interrupted => println!("no solution yet — resume the search"),
}
```

Integer/boolean variables come back from `var_value()` as exact integers, and
the reported objective is computed from exactly those values — what you read
is always self-consistent. `var_value_raw()` skips the integer rounding.

## Time limits, resuming, solve options

```rust,ignore
use std::time::Duration;
use microlp::{SolveOptions, Status};

// Convenience: a plain wall-clock budget.
problem.set_time_limit(Duration::from_secs(5));
let mut solution = problem.solve().unwrap();

// Continue where the search left off (the branch & bound tree, incumbent and
// branching statistics are all retained) until proven optimal:
while solution.status() != Status::Optimal {
    solution = solution.resume(Some(Duration::from_secs(5))).unwrap();
}
```

`resume(None)` means "run to completion". Resuming is *the* way to continue an
interrupted solve: it carries the full search state, and a sequence of small
slices reaches the same answer as one unlimited solve, value for value.

For finer control, `solve_with(SolveOptions { .. })` exposes:

* `time_limit` / `node_limit` — wall-clock or branch-and-bound node budgets;
* `mip_gap` — stop early once the relative optimality gap is proven at or
  below this (e.g. `0.01` for "within 1%"); the result is `Optimal`;
* `warm_start` — a known solution's values used as a starting incumbent. The
  hint is validated and silently dropped if it isn't usable; a good hint
  prunes the search from the first node. Note that a hint helps *finding*
  solutions, not *proving* optimality — to carry proof progress across calls,
  use `resume()`;
* `int_tol`, `tolerances` — integrality/feasibility tolerances for expert
  tuning (the defaults are calibrated by the correctness suite).

The time limit is enforced cooperatively (checked between branch-and-bound
nodes and every ~1000 simplex iterations), so a solve can overshoot its
budget by a bounded amount of work — negligible on small problems, seconds on
huge ones. Model construction happens before the clock starts.

## Editing a solved problem

All edits live on `Solution`, consume it, and return a new, re-solved one:

```rust,ignore
let solution = problem.solve().unwrap();

// Tighten the search space and re-solve.
let solution = solution.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 3.0)?;
let solution = solution.fix_var(y, 2.0)?;
let (solution, was_fixed) = solution.unfix_var(y)?;
```

For MILP solutions the edit is applied to the *original problem* (plus all
prior edits) — never to internal branch-and-bound state — and this is allowed
from any status, including a paused (`Feasible`/`Interrupted`) solve: the open
search tree is discarded, the previous incumbent is kept as a warm start when
it survives the edit, and the search re-runs with a fresh budget from the
original options. Pure-LP solutions re-solve incrementally from the live
simplex basis (and must be resumed to completion before they can be edited);
`add_gomory_cut` is available on pure-LP solutions.

`Err(Infeasible)` from an edit means the *edited* problem has no solution.

## Determinism

The solver is single-threaded and deterministic: the same problem with the
same options produces the same result, and interrupted-and-resumed solves
reach exactly the same solution as unlimited ones.

## Testing

Besides the unit tests, the repo ships a correctness suite of 230+ LP/MILP
problems whose answers are independently known (netlib and MIPLIB 3
benchmarks, MILPBench instances certified with HiGHS, constructed classics,
instances verified against exact DP/brute-force oracles, and incremental-API
scenarios). Every returned solution is re-validated against a shadow model —
feasibility, bounds, integrality, objective consistency — before being
compared with the expected answer.

Cases are tiered `easy < medium < hard < xhard`; a tier flag runs everything
up to that tier:

```bash
cargo test --release --test suite                          # default: easy + medium, ~2 s
cargo test --release --test suite -- --hard                # + long-running MILPs (what CI runs)
cargo test --release --test suite -- --xhard               # + 10-minute-budget instances
cargo test --release --test suite -- --limit 25 --seed 42  # stable random subset
```

See [tests/suite/README.md](tests/suite/README.md) for how the suite works
and [src/ARCHITECTURE.md](src/ARCHITECTURE.md) for how the solver itself
works.

## License

This project is licensed under the [Apache License, Version 2.0](./LICENSE).
