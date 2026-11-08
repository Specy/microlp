/*!
microlp solves linear programs (LPs) and mixed-integer linear programs (MILPs).
It minimizes or maximizes a linear function of variables subject to equality
and inequality constraints. Variables can be real, integer, or boolean.

# Getting started

Use microlp directly, or write models at a higher level with the
[rooc modeling language](https://github.com/specy/rooc) or
[good_lp](https://github.com/rust-or/good_lp).

# Features

* Pure Rust. Runs on WebAssembly.
* Real, integer, and boolean variables.
* Time limits and MIP gap, with the possibility to edit and resume a solve.
* Warm starts from a known solution.
* Handles problems with hundreds of thousands of variables and constraints.

Problems with integer or boolean variables are solved with branch & bound. A
solve on a hard or ill-conditioned problem can still return
[`Error::InternalError`] instead of a solution.

# Example

```
use microlp::{ComparisonOp, OptimizationDirection, Problem};

// Maximize x + 2y, where x is a real variable with x >= 0 and y is an integer
// variable with 0 <= y <= 3.
let mut problem = Problem::new(OptimizationDirection::Maximize);
let x = problem.add_var(1.0, (0.0, f64::INFINITY));
let y = problem.add_integer_var(2.0, (0, 3));

// Subject to x + y <= 4 and 2x + y >= 2.
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

// The optimum is 7, at x = 1, y = 3. An integer variable's value is within
// `int_tol` of an integer. Round it when you need an exact integer.
let solution = problem.solve().unwrap().into_solution().unwrap();
assert!((solution.objective() - 7.0).abs() < 1e-6);
assert!((solution.var_value(x) - 1.0).abs() < 1e-6);
assert_eq!(solution.var_value(y).round(), 3.0);
```

# Solving and reading the solution

[`Problem::solve`] tries to find an optimal solution. The solver checks each
solution against the problem as written. Every constraint and variable bound
must hold within the absolute feasibility tolerance: `1e-7` by default
([`Tolerances::feasibility`]), or the round-off from evaluating it in `f64` if
that is larger. Otherwise, the solve returns an error:

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

An optimal solution is optimal at the point where the solver ends, within the
tolerance. It is not necessarily optimal over every point that the tolerance
admits. On an ill-conditioned problem, where a small coefficient sits beside
a large one, the tolerance can move the objective far from the exact optimum.

Keep the nonzero coefficients in each constraint, and the nonzero
coefficients for each variable across the constraints, within six orders of
magnitude: the largest magnitude should be at most `1e6` times the smallest.
The solver does not reject or change coefficients outside that range, and it
still checks every solution against the model as written, but a solve can then
return [`Error::InternalError`]. Its message names the constraint (in the order
added) and the variable (by index) whose coefficients span the most, those of
them that lie outside the range.
Changing a variable's or constraint's units usually brings a model back into
range. Within the recommended range, an [`Error::InternalError`] is a bug or
one of the [known limitations](#known-limitations): please report it.

[`SolveOptions::presolve`] simplifies the problem before the search and is on
by default. It never removes a variable or changes a real variable's bounds.
Up to round-off, it keeps every solution whose integer variables are exactly
integral. If [`SolveOptions::dual_fixing`] fixes a variable because of the
objective, presolve keeps an optimal solution. The search never seeks a
solution that relies on [`SolveOptions::int_tol`], such as an integer variable
just off its integer under a big-M coefficient. It can still return such a
solution if it lands on one; if the objective can improve forever from that
point, the solve returns [`Error::Unbounded`].
A problem with no solution whose integer variables are integral returns
[`Error::Infeasible`] unless the search lands on a solution that relies on
`int_tol`, whether presolve is on or off.

When a solve call returns successfully:

* [`SolveOutcome::Solution`] contains a validated solution. Its status is
  [`SolutionStatus::Optimal`] when optimality was proved, with no MIP gap, or
  [`SolutionStatus::Feasible`] when a valid solution is available without
  that proof, for example after reaching a time limit, node limit, or MIP gap.
* [`SolveOutcome::Interrupted`] means a time limit or node limit fired before
  the solver found a usable solution. It exposes the [`TerminationReason`]
  and [`Stats`], but no objective or variable values because no validated
  solution is available. This does not mean the problem is impossible to
  solve. Use [`SolveOutcome::resume`] to continue the search.

[`SolveOutcome::termination_reason`] distinguishes
[`TerminationReason::ProvenOptimal`], [`TerminationReason::MipGap`],
[`TerminationReason::TimeLimit`], and [`TerminationReason::NodeLimit`].

[`Solution::var_value`] reports the value computed by the solver. An integer
or boolean variable's value is within [`SolveOptions::int_tol`] of an integer,
but is not necessarily equal to one. Values such as `0.9999999` are normal; a
variable fixed at an integer has that exact value. Round the value when you
need an exact integer. [`Solution::var_domain`] identifies integer and boolean
variables. Rounded values need not meet the constraints: under
`x <= 1e7 * y`, a `y` of `1e-6` allows `x` up to 10, but rounding `y` to zero
forbids that value of `x`.

For a problem without integer or boolean variables, [`Solution::dual`] returns
a constraint's dual value, also called its shadow price. Pass the [`Constraint`]
handle returned by [`Problem::add_constraint`]. The dual value is the rate at
which the optimal objective changes as the constraint's right-hand side
increases. [`Solution::reduced_cost`] returns a variable's reduced cost.

# Known limitations

The solver computes in floating point. Nearly dependent constraints can exceed
its resolution even within the recommended coefficient range, causing a solve
to return [`Error::InternalError`]. In extreme cases, the solver reports
[`Error::Infeasible`] for a feasible model whose feasible points require very
large variable values. If the last basis is singular to `f64`, a solve can
return a solution for an unbounded model or one far from the optimum of a
bounded model. A common source is a group of free variables that can all shift
together, such as every bus angle of a power-flow network left free. Rounding
in the coefficients can make that shift have only a tiny effect on the
constraints, so a model meant to have no solution can have one as written at
very large variable values, which a solve can return. Fix one of those
variables, such as a reference angle, to remove the shift. Outside the
recommended range, a bounded model can also be reported as
[`Error::Unbounded`]. If objective terms (costs times values) or their sum
overflow `f64`, the solver cannot evaluate the objective and returns
[`Error::InternalError`], even when the exact objective is representable. The
known cases are listed in the
[architecture notes](https://github.com/Specy/microlp/blob/v0.7.0/src/ARCHITECTURE.md#79-known-limitations).

# Time limits, resuming, and editing

[`Problem::set_time_limit`] sets a time limit. [`SolveOutcome::resume`] uses
the per-call options from the immediately preceding solve, resume, or
post-solve edit. [`SolveOutcome::resume_with`] uses only the supplied
[`ResumeOptions`]. Each field replaces the previous setting; the options are
not merged.

A [`Solution`] can be edited and re-solved. [`Solution::add_constraint`] adds a
constraint and returns its [`Constraint`] handle. [`Solution::set_rhs`]
changes a constraint's right-hand side, and [`Solution::set_obj_coeff`]
changes a variable's objective coefficient. [`Solution::fix_var`] fixes a
variable to a value, and [`Solution::unfix_var`] releases a previous fix. Each
edit consumes the solution and returns a new [`SolveOutcome`] for the edited
problem. For a problem without integer or boolean variables, the solver
re-solves from the previous basis.
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
pub use expr::{ComparisonOp, Constraint, LinearExpr, LinearTerm, OptimizationDirection, Variable};
pub use mip::{ResumeOptions, SolutionStatus, SolveOptions, Stats, TerminationReason, Tolerances};
pub use outcome::{InterruptedSolve, SolveOutcome};
pub use problem::{Problem, VarDomain};
pub use solution::{Solution, SolutionIter};

pub(crate) use problem::{timed_lp_call, StopReason};
pub(crate) use solution::SolveState;

/// A sparse constraint row, indexed by the user's variable indices.
type CsVec = sprs::CsVecI<f64, usize>;
