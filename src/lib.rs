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

Integer and boolean variables are handled with branch & bound. On some hard or
ill-conditioned problems a solve can still fail with [`Error::InternalError`]
rather than answer.

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

// The optimum is 7, at x = 1, y = 3. An integer variable's value is
// reported within `int_tol` of its integer: round it for an exact one.
let solution = problem.solve().unwrap().into_solution().unwrap();
assert!((solution.objective() - 7.0).abs() < 1e-6);
assert!((solution.var_value(x) - 1.0).abs() < 1e-6);
assert_eq!(solution.var_value(y).round(), 3.0);
```

# Solving and reading the solution

[`Problem::solve`] tries to find the optimal solution. Every solution meets
each constraint and each variable bound within an absolute tolerance (`1e-7`
by default, [`Tolerances::feasibility`], or the round-off of evaluating it in
`f64` where that is larger), checked against the problem as written. Otherwise
the solve returns an error:

* [`Error::Infeasible`]: the solver proved, from the problem's data, that no
  assignment with every integer variable at an integer within its bounds, and
  every variable whose bounds are equal at its value, meets the constraints
  and the other bounds within that tolerance (see [`Tolerances::feasibility`]).
* [`Error::Unbounded`]: the objective can improve forever.
* [`Error::InternalError`]: a numerical failure the solver could not recover
  from. It returns this rather than an answer it could not verify.
* [`Error::InvalidOptions`] for an invalid explicit numeric option, and
  [`Error::InvalidOperation`] for invalid input, both checked before anything
  is solved. Every number of the problem must be finite, but a bound or a
  constraint's right-hand side may be infinite on the side it leaves open:
  `f64::NEG_INFINITY` as a lower bound or on a [`ComparisonOp::Ge`]
  constraint, `f64::INFINITY` as an upper bound or on a [`ComparisonOp::Le`]
  constraint, where it constrains nothing. A warm-start hint must give finite
  values to variables of the problem.

An optimal solution is optimal at the point the solver ends at, within the
tolerance, not over every point the tolerance admits: on an ill-conditioned
problem, where a small coefficient sits beside a large one, the tolerance can
move the objective far from the exact optimum.

Keep the coefficients of each constraint, and those of each variable across
the constraints, within six orders of magnitude: the largest magnitude at most
`1e6` times the smallest. Nothing outside that range is refused or changed,
and every solution is still checked against the model as written, but a solve
can then fail with [`Error::InternalError`], whose message names the
constraint (in the order added) and the variable (its index) whose
coefficients span the most, those of them outside the range. Changing the
units of a variable or of a constraint usually brings a model back into
range. Within the range, an [`Error::InternalError`] is a bug or one of the
[known limitations](#known-limitations): please report it.

[`SolveOptions::presolve`], on by default, simplifies the problem before the
search. It never removes a variable or changes a real variable's bounds, and
it keeps, up to round-off, every solution whose integer variables are exactly
integral, or, where [`SolveOptions::dual_fixing`] fixes a variable because of
the objective, an optimal one. A solution relying on [`SolveOptions::int_tol`]
(an integer variable just off its integer under a big-M coefficient) is never
sought: it is returned where the search lands on one, and where the objective
can improve forever such a landing answers [`Error::Unbounded`]. A problem with
no solution whose integer variables are integral is answered
[`Error::Infeasible`] unless that happens, with presolve on or off.

When a solve call returns successfully:

* [`SolveOutcome::Solution`] contains a validated assignment. Its status is
  [`SolutionStatus::Optimal`] when optimality was proved, with no MIP gap, or
  [`SolutionStatus::Feasible`] when a valid assignment is available without
  that proof, for example after reaching a time limit, node limit, or MIP gap.
* [`SolveOutcome::Interrupted`] means a time or node limit fired before a usable
  assignment was found. It exposes the [`TerminationReason`] and [`Stats`], but
  no objective or variable values because no validated assignment is available.
  This does not mean the problem is impossible to solve. Use
  [`SolveOutcome::resume`] to continue the search.

[`SolveOutcome::termination_reason`] distinguishes
[`TerminationReason::ProvenOptimal`], [`TerminationReason::MipGap`],
[`TerminationReason::TimeLimit`], and [`TerminationReason::NodeLimit`].

[`Solution::var_value`] reports a value as the solver computed it. An integer
or boolean variable's value lies within [`SolveOptions::int_tol`] of an
integer, not necessarily on it: values like `0.9999999` are normal; one fixed
at an integer lies on it. Round it where an exact integer is needed
([`Solution::var_domain`] tells which variables are integer or boolean),
bearing in mind that rounded values need not meet the constraints: under
`x <= 1e7 * y`, a `y` of `1e-6` allows `x` up to 10, which `y` rounded to zero
forbids.

# Known limitations

The solver computes in floating point. Nearly dependent constraints can exceed
its resolution even within the recommended coefficient range: a solve can then
fail with [`Error::InternalError`]; in extreme cases a feasible model whose
feasible points need very large variable values is reported
[`Error::Infeasible`], and a solve whose last basis is singular to `f64` reports
a solution for an unbounded model, or one far from the optimum of a bounded
model. A common source is a group of free variables that can all shift
together, such as every bus angle of a power-flow network left free: where
rounding in the coefficients leaves that shift a tiny effect on the
constraints, a model meant to have no solution can have one as written, at very
large variable values, which a solve can return. Fix one of those variables,
such as a reference angle, to remove the shift. Outside the range, a bounded
model can also be reported [`Error::Unbounded`]. An objective whose terms
(costs times values) or their sum overflow `f64` cannot be evaluated: a solve
that meets one fails with [`Error::InternalError`], even where the exact
objective is representable. The known cases are listed in the
[architecture notes](https://github.com/Specy/microlp/blob/v0.7.0/src/ARCHITECTURE.md#79-known-limitations).

# Time limits, resuming, and editing

[`Problem::set_time_limit`] sets a time budget. [`SolveOutcome::resume`] uses
the per-call options from the immediately preceding solve, resume, or
post-solve edit call.
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
mod range;
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

/// A sparse constraint row, indexed by the user's variable indices.
type CsVec = sprs::CsVecI<f64, usize>;
