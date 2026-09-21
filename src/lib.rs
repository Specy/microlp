/*!
A linear programming solver: it finds the minimum (or maximum) of a linear
function of a set of variables subject to linear equality and inequality
constraints. Variables can be real, integer, or boolean.

# Getting started

You can use microlp directly, but the
[rooc modeling language](https://github.com/specy/rooc) and
[good_lp](https://github.com/rust-or/good_lp) provide
higher-level ways to write models.

# Features

* Pure Rust. Runs on WebAssembly.
* Real, integer, and boolean variables.
* Time limits and MIP gap, with the possibility to edit and resume a solve.
* Warm starts from a known solution.
* Handles problems with hundreds of thousands of variables and constraints.

Integer and boolean variables are handled with branch & bound. The solver may
still cycle or lose precision on some hard problems.

# Example

```
use microlp::{ComparisonOp, OptimizationDirection, Problem};

// Maximize x + 2y, where x is real with x >= 0 and y is an integer
// with 0 <= y <= 3.
let mut problem = Problem::new(OptimizationDirection::Maximize);
let x = problem.add_var(1.0, (0.0, f64::INFINITY));
let y = problem.add_integer_var(2.0, (0, 3));

// Subject to x + y <= 4 and 2x + y >= 2.
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

// The optimum is 7, at x = 1, y = 3.
let solution = problem.solve().unwrap().into_solution().unwrap();
assert_eq!(solution.objective(), 7.0);
assert_eq!(solution.var_value(x), 1.0);
assert_eq!(solution.var_value(y), 3.0);
```

# Solving and reading the solution

[`Problem::solve`] tries to find the optimal solution. Contradictory
constraints produce [`Error::Infeasible`], while an objective that can improve
forever produces [`Error::Unbounded`]. Invalid explicit numeric options produce
[`Error::InvalidOptions`], and unrecoverable numerical failures produce
[`Error::InternalError`].

When a solve call returns successfully:

* [`SolveOutcome::Solution`] contains a validated assignment. Its status is
  [`SolutionStatus::Optimal`] when exact optimality was proved, or
  [`SolutionStatus::Feasible`] when a valid assignment is available without an
  exact proof, for example after reaching a time limit, node limit, or MIP gap.
* [`SolveOutcome::Interrupted`] means a time or node limit fired before a usable
  assignment was found. It exposes the [`TerminationReason`] and [`Stats`], but
  no objective or variable values because no validated assignment is available.
  This does not mean the problem is impossible to solve. Use
  [`SolveOutcome::resume`] to continue the search.

[`SolveOutcome::termination_reason`] distinguishes
[`TerminationReason::ProvenOptimal`], [`TerminationReason::MipGap`],
[`TerminationReason::TimeLimit`], and [`TerminationReason::NodeLimit`].

# Time limits, resuming, and editing

[`Problem::set_time_limit`] sets a time budget. [`SolveOutcome::resume`] uses
the per-call options from the immediately preceding solve or resume call.
[`SolveOutcome::resume_with`] instead uses only the supplied [`ResumeOptions`]:
every field replaces the previous setting, and values are not merged.

A [`Solution`] can be edited and re-solved. [`Solution::add_constraint`] adds a
constraint, [`Solution::fix_var`] pins a variable to a value, and
[`Solution::unfix_var`] releases a previous fix. Each edit consumes the
solution and returns a new [`SolveOutcome`] for the edited problem.


*/

#![deny(missing_debug_implementations, missing_docs)]

#[macro_use]
extern crate log;

mod error;
mod expr;
mod helpers;
mod lu;
mod mip;
mod ordering;
mod outcome;
mod presolve;
mod problem;
// Problem solvers built on top of the microlp library (not part of the
// public API — exists only to exercise the crate's own tests).
#[cfg(test)]
mod problems_solvers;
mod solution;
mod solver;
mod sparse;
mod tests;

pub use error::Error;
pub use expr::{ComparisonOp, LinearExpr, LinearTerm, OptimizationDirection, Variable};
pub use mip::{ResumeOptions, SolutionStatus, SolveOptions, Stats, TerminationReason, Tolerances};
pub use outcome::{InterruptedSolve, SolveOutcome};
pub use problem::{Problem, VarDomain};
pub use solution::{Solution, SolutionIter};

pub(crate) use problem::{timed_lp_call, StopReason};
pub(crate) use solution::SolveState;

/// The crate's internal sparse vector type: one constraint row, or one column
/// of the constraint matrix, in the user's variable indices.
type CsVec = sprs::CsVecI<f64, usize>;
