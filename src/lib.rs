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
* Dual values (shadow prices) and reduced costs for linear programs.
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
must hold within the absolute feasibility tolerance of `1e-7` by default
([`Tolerances::feasibility`]), or the round-off from evaluating it in `f64` if
that is larger. Otherwise, the solve returns an error:

* [`Error::Infeasible`]: the problem has no solution. No assignment meets every
  constraint and bound within the feasibility tolerance while every integer
  variable takes a whole number. See [`Error::Infeasible`] for the exact
  conditions.
* [`Error::Unbounded`]: the objective can improve forever.
* [`Error::InternalError`]: a numerical failure the solver could not recover
  from. It returns this rather than an answer it could not verify.
* [`Error::InvalidOptions`] for an invalid explicit numeric option, and
  [`Error::InvalidOperation`] for invalid input. Both are checked before
  solving. See [`Problem::solve_with`] for accepted model values and
  [`SolveOptions::warm_start`] for valid hints.

[`SolutionStatus::Optimal`] means optimal up to numerical tolerances. It
does not guarantee the best objective over every assignment the tolerances
allow. A large coefficient can multiply a tolerance-sized difference in a
variable's value into substantial objective error. See
[`Tolerances::feasibility`] for an example.

Keep the nonzero coefficients in each constraint, and the nonzero
coefficients for each variable across the constraints, within six orders of
magnitude. The largest magnitude should be at most `1e6` times the smallest.
The solver accepts coefficients outside that range without changing them.
It still checks every returned solution against the problem as written, but
[`Error::InternalError`] becomes more likely. Its message identifies the
constraint with the widest coefficient range, by order added, and the
variable with the widest range, by index, when those ranges exceed the
recommendation. Changing a variable's or constraint's units usually brings a
model back into range. Within the recommended range, an
[`Error::InternalError`] is a bug or one of the
[known limitations](#known-limitations). Please report it.

[`SolveOptions::presolve`] is on by default and simplifies the problem before
solving. It makes most solves faster. Every variable is still reported, and
returned solutions satisfy the original problem within the documented
tolerances. See [`SolveOptions::dual_fixing`] before enumerating solutions
through edits.

[`SolveOptions::int_tol`] permits slightly nonintegral returned values for
integer and boolean variables. Increasing it does not make the solver seek
all assignments within that tolerance of an integer. A model feasible only
at such assignments normally returns [`Error::Infeasible`], though the solver
can return one if it happens to find it. This applies with presolve on or off.

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
but is not necessarily equal to one. Values such as `0.9999999` are normal.
A variable fixed at an integer has that exact value. Round the value when you
need an exact integer. [`Solution::var_domain`] identifies integer and boolean
variables. Rounded values need not meet the constraints. For example, under
`x <= 1e7 * y`, a `y` of `1e-6` allows `x` up to 10, but rounding `y` to zero
forbids that value of `x`.

For a problem without integer or boolean variables, [`Solution::dual`] returns
a constraint's dual value, also called its shadow price. Pass the [`Constraint`]
handle returned by [`Problem::add_constraint`]. The dual value is the rate at
which the optimal objective changes as the constraint's right-hand side
increases. [`Solution::reduced_cost`] returns a variable's reduced cost.

# Known limitations

The solver computes in floating point. Nearly repeating constraints, or
constraints nearly implied by others, can exceed its numerical resolution
even within the recommended coefficient range. A solve can then return
[`Error::InternalError`]. In extreme cases, a feasible model requiring very
large variable values can be reported as [`Error::Infeasible`]. Rarely, a
solve can return a solution for an unbounded model or a solution far from a
bounded model's optimum. Outside the recommended range, a bounded model can
also be reported as [`Error::Unbounded`].

Free variables that can all shift together need particular care. For example,
leaving every bus angle free in a power-flow model can allow very large
variable values with almost no effect on the constraints. Coefficient
round-off can then make a model intended to be infeasible admit a solution
at those values, and the solver can return it. Fix a reference variable,
such as one bus angle, to remove the shared shift.

If an objective coefficient times its variable value, or the sum of those
terms, overflows `f64`, the solve returns [`Error::InternalError`]. This can
happen even when the exact objective is representable. See the
[known numerical cases](https://github.com/Specy/microlp/blob/v0.8.0/src/ARCHITECTURE.md#79-known-limitations).

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
edit consumes the solution even if it returns an error, and returns a new
[`SolveOutcome`] on success. Clone the solution first if you need to keep it.
For LPs, edits continue from the preceding solve and are usually much faster
than starting over.

Edits return the same errors as solving. [`Error::Infeasible`] means the
changed problem has no solution. [`Error::Unbounded`] means its objective can
improve forever, for example after disabling a limiting constraint with
[`Solution::set_rhs`], changing an objective coefficient with
[`Solution::set_obj_coeff`], or releasing a fix with [`Solution::unfix_var`].
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
