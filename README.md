# microlp
This is a fork of the archived [minilp](https://github.com/ztlpn/minilp) crate, which was made to fix some bugs, add features and allow the community to make issues and PRs.

[![Crates.io](https://img.shields.io/crates/v/microlp.svg)](https://crates.io/crates/microlp)
[![Documentation](https://docs.rs/microlp/badge.svg)](https://docs.rs/microlp/)

A fast linear programming solver library.

[Linear programming](https://en.wikipedia.org/wiki/Linear_programming) is a technique for
finding the minimum (or maximum) of a linear function of a set of variables
subject to linear equality and inequality constraints.

## Features

* Pure Rust implementation.
* Able to solve problems with hundreds of thousands of variables and constraints.
* Incremental: add constraints to an existing solution without solving it from scratch.
* Problems can be defined via an API or parsed from an
  [MPS](https://en.wikipedia.org/wiki/MPS_(format)) file.
* Allows for continuous, integer and boolean variables

Warning: this is an early-stage project. Although the library is already quite powerful and fast,
it will probably cycle, lose precision or panic on some harder problems. Please report
bugs and contribute code! 
Models with integer or binary variables are solved using a simple branch & bound method.

## Examples

Basic usage

```rust
use microlp::{Problem, OptimizationDirection, ComparisonOp};

// Maximize an objective function x + 2 * y of two continuous variables x >= 0 and 0 <= y <= 3
let mut problem = Problem::new(OptimizationDirection::Maximize);
let x = problem.add_var(1.0, (0.0, f64::INFINITY));
let y = problem.add_var(2.0, (0.0, 3.0));

// subject to constraints: x + y <= 4 and 2 * x + y >= 2.
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

// Optimal value is 7, achieved at x = 1 and y = 3.
let solution = problem.solve().unwrap();
assert_eq!(solution.objective(), 7.0);
assert_eq!(solution[x], 1.0);
assert_eq!(solution[y], 3.0);
```

For a more involved example, see [examples/tsp](examples#tsp), a solver for the travelling
salesman problem.

## License

This project is licensed under the [Apache License, Version 2.0](./LICENSE).
