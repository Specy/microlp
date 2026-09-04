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

mod helpers;
mod lu;
mod mip;
mod ordering;
// Problem solvers built on top of the microlp library (not part of the
// public API — exists only to exercise the crate's own tests).
#[cfg(test)]
mod problems_solvers;
mod solver;
mod sparse;
mod tests;

use solver::Solver;
use sprs::errors::StructureError;

use core::time::Duration;
use web_time::Instant;

/// Selects whether a problem's objective is minimized or maximized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizationDirection {
    /// Minimize the objective value.
    Minimize,
    /// Maximize the objective value.
    Maximize,
}

/// Identifies a variable created by a [`Problem`].
///
/// A variable should only be used with the problem that created it and with
/// solutions obtained from that problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Variable(pub(crate) usize);

impl Variable {
    /// Returns the variable's zero-based creation order.
    ///
    /// The first variable added to a problem has index `0`.
    pub fn idx(&self) -> usize {
        self.0
    }
}

/// A weighted sum of variables used on the left-hand side of a constraint.
#[derive(Clone, Debug)]
pub struct LinearExpr {
    vars: Vec<usize>,
    coeffs: Vec<f64>,
}

impl LinearExpr {
    /// Creates a linear expression containing no terms.
    pub fn empty() -> Self {
        Self {
            vars: vec![],
            coeffs: vec![],
        }
    }

    /// Adds the term `coeff * var` to the expression.
    ///
    /// Terms may be added in any order, but each variable may appear only once.
    /// Passing an expression with repeated variables to
    /// [`Problem::add_constraint`] will panic.
    pub fn add(&mut self, var: Variable, coeff: f64) {
        self.vars.push(var.0);
        self.coeffs.push(coeff);
    }
}

/// A single `variable * constant` term in a linear expression.
/// This is an auxiliary struct for specifying conversions.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct LinearTerm(Variable, f64);

impl From<(Variable, f64)> for LinearTerm {
    fn from(term: (Variable, f64)) -> Self {
        LinearTerm(term.0, term.1)
    }
}

impl<'a> From<&'a (Variable, f64)> for LinearTerm {
    fn from(term: &'a (Variable, f64)) -> Self {
        LinearTerm(term.0, term.1)
    }
}

impl<I: IntoIterator<Item = impl Into<LinearTerm>>> From<I> for LinearExpr {
    fn from(iter: I) -> Self {
        let mut expr = LinearExpr::empty();
        for term in iter {
            let LinearTerm(var, coeff) = term.into();
            expr.add(var, coeff);
        }
        expr
    }
}

impl std::iter::FromIterator<(Variable, f64)> for LinearExpr {
    fn from_iter<I: IntoIterator<Item = (Variable, f64)>>(iter: I) -> Self {
        let mut expr = LinearExpr::empty();
        for term in iter {
            expr.add(term.0, term.1)
        }
        expr
    }
}

impl std::iter::Extend<(Variable, f64)> for LinearExpr {
    fn extend<I: IntoIterator<Item = (Variable, f64)>>(&mut self, iter: I) {
        for term in iter {
            self.add(term.0, term.1)
        }
    }
}

/// Specifies how a constraint's left-hand expression is compared with its
/// right-hand value.
#[derive(Clone, Copy, Debug)]
pub enum ComparisonOp {
    /// The left-hand side must equal the right-hand side (`==`).
    Eq,
    /// The left-hand side must be less than or equal to the right-hand side
    /// (`<=`).
    Le,
    /// The left-hand side must be greater than or equal to the right-hand side
    /// (`>=`).
    Ge,
}

/// An error returned while validating, solving, or editing a problem.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// No assignment satisfies all variable bounds and constraints.
    Infeasible,
    /// The objective can improve without a finite limit.
    Unbounded,
    /// A solve or resume option is non-finite or outside its accepted range.
    ///
    /// The message identifies the invalid field.
    InvalidOptions(String),
    /// The requested operation cannot be applied to the current problem or
    /// outcome.
    InvalidOperation(String),
    /// The solve could not continue because of an unexpected numerical or
    /// structural failure.
    InternalError(String),
}
impl From<StructureError> for Error {
    fn from(err: StructureError) -> Self {
        Error::InternalError(err.to_string())
    }
}

impl From<sparse::Error> for Error {
    fn from(value: sparse::Error) -> Self {
        Error::InternalError(value.to_string())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let msg = match self {
            Error::Infeasible => "problem is infeasible",
            Error::Unbounded => "problem is unbounded",
            Error::InvalidOptions(msg)
            | Error::InvalidOperation(msg)
            | Error::InternalError(msg) => msg,
        };
        msg.fmt(f)
    }
}

impl std::error::Error for Error {}

/// A linear optimization model that can be populated and solved.
///
/// Add variables and constraints, then call [`Problem::solve`] or
/// [`Problem::solve_with`].
#[derive(Clone)]
pub struct Problem {
    direction: OptimizationDirection,
    obj_coeffs: Vec<f64>,
    var_mins: Vec<f64>,
    var_maxs: Vec<f64>,
    var_domains: Vec<VarDomain>,
    constraints: Vec<(CsVec, ComparisonOp, f64)>,
    time_limit: Option<Duration>,
}

impl std::fmt::Debug for Problem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Only printing lengths here because actual data is probably huge.
        f.debug_struct("Problem")
            .field("direction", &self.direction)
            .field("num_vars", &self.obj_coeffs.len())
            .field("num_constraints", &self.constraints.len())
            .finish()
    }
}

type CsVec = sprs::CsVecI<f64, usize>;

#[derive(Clone, Debug, PartialEq)]
/// The values a variable is allowed to take.
pub enum VarDomain {
    /// Any integer within the variable's bounds.
    Integer,
    /// Any real value within the variable's bounds.
    Real,
    /// Either `0` or `1`.
    Boolean,
}

impl Problem {
    /// Creates an empty problem with the selected optimization direction.
    pub fn new(direction: OptimizationDirection) -> Self {
        Problem {
            direction,
            obj_coeffs: vec![],
            var_mins: vec![],
            var_maxs: vec![],
            var_domains: vec![],
            constraints: vec![],
            time_limit: None,
        }
    }

    /// Sets the time budget used by [`Problem::solve`].
    ///
    /// If the budget expires, the outcome contains a feasible [`Solution`] when
    /// one is available, or [`SolveOutcome::Interrupted`] when no usable
    /// assignment has been found. The outcome can be continued with
    /// [`SolveOutcome::resume`].
    ///
    /// [`Problem::solve_with`] uses the `time_limit` in its supplied
    /// [`SolveOptions`] instead of this setting.
    pub fn set_time_limit(&mut self, duration: Duration) {
        self.time_limit = Some(duration);
    }

    /// Adds a real-valued variable to the problem.
    ///
    /// `obj_coeff` is the variable's coefficient in the objective. `min` and
    /// `max` are inclusive bounds; use [`f64::NEG_INFINITY`] or
    /// [`f64::INFINITY`] for an unbounded side.
    pub fn add_var(&mut self, obj_coeff: f64, (min, max): (f64, f64)) -> Variable {
        self.internal_add_var(obj_coeff, (min, max), VarDomain::Real)
    }

    /// Adds an integer-valued variable to the problem.
    ///
    /// `obj_coeff` is the variable's coefficient in the objective. `min` and
    /// `max` are inclusive bounds. Use [`i32::MIN`] or [`i32::MAX`] when no
    /// tighter integer bound is required on that side.
    pub fn add_integer_var(&mut self, obj_coeff: f64, (min, max): (i32, i32)) -> Variable {
        self.internal_add_var(obj_coeff, (min as f64, max as f64), VarDomain::Integer)
    }

    /// Returns whether the problem contains any integer or boolean variables.
    pub fn has_integer_vars(&self) -> bool {
        self.var_domains
            .iter()
            .any(|v| *v == VarDomain::Integer || *v == VarDomain::Boolean)
    }

    /// Adds a variable restricted to `0` or `1`.
    ///
    /// `obj_coeff` is the variable's coefficient in the objective.
    pub fn add_binary_var(&mut self, obj_coeff: f64) -> Variable {
        self.internal_add_var(obj_coeff, (0.0, 1.0), VarDomain::Boolean)
    }

    pub(crate) fn internal_add_var(
        &mut self,
        obj_coeff: f64,
        (min, max): (f64, f64),
        var_type: VarDomain,
    ) -> Variable {
        let var = Variable(self.obj_coeffs.len());
        let obj_coeff = match self.direction {
            OptimizationDirection::Minimize => obj_coeff,
            OptimizationDirection::Maximize => -obj_coeff,
        };
        self.obj_coeffs.push(obj_coeff);
        self.var_mins.push(min);
        self.var_maxs.push(max);
        self.var_domains.push(var_type);
        var
    }

    /// Adds the linear constraint `expr cmp_op rhs` to the problem.
    ///
    /// The expression may be a [`LinearExpr`] or any supported collection or
    /// iterator of `(Variable, coefficient)` pairs. Use variables created by
    /// this problem, with each variable appearing at most once.
    ///
    /// # Panics
    ///
    /// Panics if the expression repeats a variable or contains a variable index
    /// outside this problem.
    ///
    /// # Examples
    ///
    /// The left-hand side can be specified in several ways:
    /// ```
    /// # use microlp::*;
    /// let mut problem = Problem::new(OptimizationDirection::Minimize);
    /// let x = problem.add_var(1.0, (0.0, f64::INFINITY));
    /// let y = problem.add_var(1.0, (0.0, f64::INFINITY));
    ///
    /// // Add the constraint x + y >= 2:
    ///
    /// // * with a slice of variable-coefficient pairs
    /// problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 2.0);
    ///
    /// // * with an iterator of variable-coefficient pairs
    /// let vars = [x, y];
    /// problem.add_constraint(vars.iter().map(|&v| (v, 1.0)), ComparisonOp::Ge, 2.0);
    ///
    /// // * with a LinearExpr built term by term
    /// let mut lhs = LinearExpr::empty();
    /// for &v in &vars {
    ///     lhs.add(v, 1.0);
    /// }
    /// problem.add_constraint(lhs, ComparisonOp::Ge, 2.0);
    /// ```
    pub fn add_constraint(&mut self, expr: impl Into<LinearExpr>, cmp_op: ComparisonOp, rhs: f64) {
        let expr = expr.into();
        self.constraints.push((
            CsVec::new_from_unsorted(self.obj_coeffs.len(), expr.vars, expr.coeffs).unwrap(),
            cmp_op,
            rhs,
        ));
    }
}

pub use mip::{ResumeOptions, SolutionStatus, SolveOptions, Stats, TerminationReason, Tolerances};

/// Internal signal for whether a simplex operation finished or hit its deadline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopReason {
    Limit,
    Finished,
}

fn timed_lp_call<T>(
    solver: &mut Solver,
    time_limit: Option<Duration>,
    call: impl FnOnce(&mut Solver) -> Result<T, Error>,
) -> Result<T, Error> {
    let started = Instant::now();
    solver.deadline = time_limit.map(|duration| started + duration);
    let result = call(solver);
    solver.elapsed += started.elapsed();
    result
}

impl Problem {
    pub(crate) fn build_solver(&self, deadline: solver::Deadline) -> Result<Solver, Error> {
        Solver::try_new(
            &self.obj_coeffs,
            &self.var_mins,
            &self.var_maxs,
            &self.constraints,
            &self.var_domains,
            deadline,
        )
    }

    /// Tries to solve the problem using the default options.
    ///
    /// A time limit configured with [`Problem::set_time_limit`] is applied to
    /// this call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Infeasible`] when no feasible assignment exists,
    /// [`Error::Unbounded`] when the objective has no finite optimum, or
    /// [`Error::InternalError`] when the solve cannot continue.
    pub fn solve(&self) -> Result<SolveOutcome, Error> {
        let options = SolveOptions {
            time_limit: self.time_limit,
            ..SolveOptions::default()
        };
        self.solve_with(options)
    }

    /// Tries to solve the problem using the supplied [`SolveOptions`].
    ///
    /// These options control this call directly and override any previously set
    /// options.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOptions`] when an option is non-finite or outside
    /// its accepted range. See [`Problem::solve`] for errors that can be
    /// reported while solving.
    pub fn solve_with(&self, options: SolveOptions) -> Result<SolveOutcome, Error> {
        options.validate()?;
        let num_vars = self.obj_coeffs.len();
        if self.has_integer_vars() {
            let run = mip::run(self, options.clone())?;
            let resume_options = ResumeOptions::from(&options);
            Ok(SolveOutcome::from_mip_run(
                self.direction,
                num_vars,
                run,
                resume_options,
            ))
        } else {
            let started = Instant::now();
            let deadline = options.time_limit.map(|duration| started + duration);
            let mut solver = self.build_solver(deadline)?;
            solver.operation_time_limit = options.time_limit;
            let stop = solver.initial_solve()?;
            solver.elapsed += started.elapsed();
            let resume_options = ResumeOptions::from(&options);
            Ok(SolveOutcome::from_lp_stop(
                self.direction,
                num_vars,
                stop,
                Box::new(solver),
                resume_options,
            ))
        }
    }
}

#[derive(Clone)]
enum SolveState {
    Lp(Box<Solver>),
    Mip(Box<mip::MipState>),
}

/// The result of a successful solve.
///
/// The outcome either contains a validated [`Solution`] or reports that a
/// configured limit interrupted the call before a usable assignment was found.
/// An interrupted outcome has no objective or variable values, but it can be
/// resumed to keep searching for an incumbent.
#[derive(Clone)]
pub enum SolveOutcome {
    /// A validated assignment that is optimal or feasible without a proof of
    /// optimality.
    Solution(Solution),
    /// A time or node limit was reached before a usable assignment was found.
    Interrupted(InterruptedSolve),
}

impl std::fmt::Debug for SolveOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Solution(solution) => f.debug_tuple("Solution").field(solution).finish(),
            Self::Interrupted(interrupted) => {
                f.debug_tuple("Interrupted").field(interrupted).finish()
            }
        }
    }
}

impl SolveOutcome {
    fn from_lp_stop(
        direction: OptimizationDirection,
        num_vars: usize,
        stop: StopReason,
        solver: Box<Solver>,
        last_options: ResumeOptions,
    ) -> Self {
        match stop {
            StopReason::Finished => Self::Solution(Solution {
                direction,
                num_vars,
                status: SolutionStatus::Optimal,
                termination_reason: TerminationReason::ProvenOptimal,
                state: SolveState::Lp(solver),
                last_options,
            }),
            StopReason::Limit => Self::Interrupted(InterruptedSolve {
                direction,
                num_vars,
                termination_reason: TerminationReason::TimeLimit,
                state: SolveState::Lp(solver),
                last_options,
            }),
        }
    }

    fn from_mip_run(
        direction: OptimizationDirection,
        num_vars: usize,
        run: mip::MipRun,
        last_options: ResumeOptions,
    ) -> Self {
        let mip::MipRun { reason, state } = run;
        let has_incumbent = state.incumbent.is_some() && !state.classifying_unbounded;
        match reason {
            TerminationReason::ProvenOptimal => {
                debug_assert!(has_incumbent);
                Self::Solution(Solution {
                    direction,
                    num_vars,
                    status: SolutionStatus::Optimal,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                })
            }
            TerminationReason::MipGap => {
                debug_assert!(has_incumbent);
                Self::Solution(Solution {
                    direction,
                    num_vars,
                    status: SolutionStatus::Feasible,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                })
            }
            TerminationReason::TimeLimit | TerminationReason::NodeLimit if has_incumbent => {
                Self::Solution(Solution {
                    direction,
                    num_vars,
                    status: SolutionStatus::Feasible,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                })
            }
            TerminationReason::TimeLimit | TerminationReason::NodeLimit => {
                Self::Interrupted(InterruptedSolve {
                    direction,
                    num_vars,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                })
            }
        }
    }

    /// Borrows the solution contained in this outcome, if one is available.
    ///
    /// Returns `None` for [`SolveOutcome::Interrupted`].
    pub fn solution(&self) -> Option<&Solution> {
        match self {
            Self::Solution(solution) => Some(solution),
            Self::Interrupted(_) => None,
        }
    }

    /// Consumes this outcome and returns its solution.
    ///
    /// If the outcome was interrupted, the [`InterruptedSolve`] value is
    /// returned as the error so it can still be inspected or resumed.
    pub fn into_solution(self) -> Result<Solution, InterruptedSolve> {
        match self {
            Self::Solution(solution) => Ok(solution),
            Self::Interrupted(interrupted) => Err(interrupted),
        }
    }

    /// Returns why the call that produced this outcome stopped.
    pub fn termination_reason(&self) -> TerminationReason {
        match self {
            Self::Solution(solution) => solution.termination_reason(),
            Self::Interrupted(interrupted) => interrupted.termination_reason(),
        }
    }

    /// Returns statistics for the solve history represented by this outcome.
    ///
    /// The values include any resumes performed before this outcome was
    /// produced.
    pub fn stats(&self) -> Stats {
        match self {
            Self::Solution(solution) => solution.stats(),
            Self::Interrupted(interrupted) => interrupted.stats(),
        }
    }

    /// Returns whether this outcome contains a solution with proven optimality.
    pub fn is_optimal(&self) -> bool {
        matches!(
            self,
            Self::Solution(Solution {
                status: SolutionStatus::Optimal,
                ..
            })
        )
    }

    /// Returns the settings that [`SolveOutcome::resume`] will use.
    ///
    /// These are the time limit, node limit, and MIP gap from the immediately
    /// preceding solve, resume, or post-solve edit call.
    pub fn last_resume_options(&self) -> ResumeOptions {
        match self {
            Self::Solution(solution) => solution.last_options.clone(),
            Self::Interrupted(interrupted) => interrupted.last_options.clone(),
        }
    }

    /// Consumes this outcome and continues using the preceding call's settings.
    ///
    /// Time and node limits are applied as fresh budgets for this call. An
    /// already optimal outcome is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns errors encountered while continuing the solve, such as
    /// [`Error::Infeasible`], [`Error::Unbounded`], or
    /// [`Error::InternalError`].
    pub fn resume(self) -> Result<Self, Error> {
        let options = self.last_resume_options();
        self.resume_with(options)
    }

    /// Consumes this outcome and continues using the supplied settings.
    ///
    /// The fields replace, rather than merge with, the preceding call's
    /// settings. Time and node limits are fresh budgets for this call, and
    /// `None` makes either budget unlimited. A `mip_gap` of `None` requests
    /// exact optimality.
    ///
    /// An already optimal outcome is returned unchanged after validating
    /// `options`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOptions`] when a supplied value is non-finite or
    /// outside its accepted range. Errors encountered while continuing the
    /// solve are returned unchanged.
    pub fn resume_with(self, options: ResumeOptions) -> Result<Self, Error> {
        options.validate()?;
        if self.is_optimal() {
            return Ok(self);
        }
        let (direction, num_vars, state) = match self {
            Self::Solution(solution) => {
                debug_assert_eq!(solution.status, SolutionStatus::Feasible);
                (solution.direction, solution.num_vars, solution.state)
            }
            Self::Interrupted(interrupted) => (
                interrupted.direction,
                interrupted.num_vars,
                interrupted.state,
            ),
        };
        match state {
            SolveState::Lp(mut solver) => {
                solver.operation_time_limit = options.time_limit;
                let stop = timed_lp_call(&mut solver, options.time_limit, Solver::initial_solve)?;
                Ok(Self::from_lp_stop(
                    direction, num_vars, stop, solver, options,
                ))
            }
            SolveState::Mip(mut state) => {
                let reason = mip::resume_run(&mut state, options.clone())?;
                Ok(Self::from_mip_run(
                    direction,
                    num_vars,
                    mip::MipRun {
                        reason,
                        state: *state,
                    },
                    options,
                ))
            }
        }
    }
}

/// A validated feasible assignment returned by a solve-like call.
///
/// Use [`Solution::status`] to distinguish a proven optimum from an assignment
/// that is feasible without a proof of optimality.
#[derive(Clone)]
pub struct Solution {
    direction: OptimizationDirection,
    num_vars: usize,
    status: SolutionStatus,
    termination_reason: TerminationReason,
    state: SolveState,
    last_options: ResumeOptions,
}

impl From<&SolveOptions> for ResumeOptions {
    fn from(options: &SolveOptions) -> Self {
        Self {
            time_limit: options.time_limit,
            node_limit: options.node_limit,
            mip_gap: if options.mip_gap == 0.0 {
                None
            } else {
                Some(options.mip_gap)
            },
        }
    }
}

impl std::fmt::Debug for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solution")
            .field("direction", &self.direction)
            .field("num_vars", &self.num_vars)
            .field("status", &self.status)
            .field("termination_reason", &self.termination_reason)
            .field("objective", &self.objective())
            .finish()
    }
}

impl Solution {
    /// Returns whether the assignment is proven optimal or only known to be
    /// feasible.
    pub fn status(&self) -> SolutionStatus {
        self.status
    }

    /// Returns why the call that produced this solution stopped.
    pub fn termination_reason(&self) -> TerminationReason {
        self.termination_reason
    }

    /// Returns the objective value in the problem's original optimization
    /// direction.
    pub fn objective(&self) -> f64 {
        let internal = match &self.state {
            SolveState::Lp(solver) => solver.cur_obj_val,
            SolveState::Mip(state) => state.current_objective(),
        };
        match self.direction {
            OptimizationDirection::Minimize => internal,
            OptimizationDirection::Maximize => -internal,
        }
    }

    /// Returns a variable's value without integer or boolean rounding.
    ///
    /// For an integer or boolean variable, prefer [`Solution::var_value`] when
    /// an exact domain value is desired.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    pub fn var_value_raw(&self, var: Variable) -> f64 {
        assert!(var.0 < self.num_vars);
        match &self.state {
            SolveState::Lp(solver) => *solver.get_value(var.0),
            SolveState::Mip(state) => {
                state
                    .incumbent
                    .as_ref()
                    .expect("a public MIP Solution must have an incumbent")
                    .values[var.0]
            }
        }
    }

    /// Returns a variable's value, rounding integer and boolean variables.
    ///
    /// Real-valued variables are returned unchanged. Integer and boolean values
    /// are rounded to the nearest integer.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range or an integer value is farther than
    /// [`Tolerances::integrality_rounding`] from its nearest integer.
    pub fn var_value(&self, var: Variable) -> f64 {
        let value = self.var_value_raw(var);
        let domain = match &self.state {
            SolveState::Lp(solver) => &solver.orig_var_domains[var.0],
            SolveState::Mip(state) => &state.solver.orig_var_domains[var.0],
        };
        if matches!(domain, VarDomain::Integer | VarDomain::Boolean) {
            let rounded = value.round();
            let tolerance = Tolerances::default().integrality_rounding;
            assert!(
                (rounded - value).abs() < tolerance,
                "Variable was expected to be an integer, got {}",
                value
            );
            rounded
        } else {
            value
        }
    }

    /// Returns the relative gap between this solution and the best proven bound.
    ///
    /// Returns `None` when no proven bound is available yet. An optimal linear
    /// program reports `Some(0.0)`.
    pub fn gap(&self) -> Option<f64> {
        match &self.state {
            SolveState::Lp(_) => Some(0.0),
            SolveState::Mip(state) => state.stats.gap,
        }
    }

    /// Returns solve statistics associated with this solution.
    ///
    /// The values include any resumes that led to this solution.
    pub fn stats(&self) -> Stats {
        match &self.state {
            SolveState::Lp(solver) => Stats {
                lp_iterations: solver.lp_iterations,
                elapsed: solver.elapsed,
                best_bound: Some(self.objective()),
                gap: Some(0.0),
                ..Stats::default()
            },
            SolveState::Mip(state) => state.stats,
        }
    }

    /// Iterates over all variables and their values in creation order.
    ///
    /// Values use the same integer and boolean rounding as
    /// [`Solution::var_value`].
    pub fn iter(&self) -> SolutionIter<'_> {
        SolutionIter {
            solution: self,
            var_idx: 0,
        }
    }

    /// Returns the settings used by the immediately preceding solve, resume, or
    /// post-solve edit call.
    ///
    /// These are the settings that resuming this solution would reuse.
    pub fn last_resume_options(&self) -> ResumeOptions {
        self.last_options.clone()
    }

    /// Consumes this solution, adds a constraint, and solves the edited problem.
    ///
    /// The returned [`SolveOutcome`] may contain an optimal solution, a feasible
    /// solution, or an interrupted call when a configured limit is reached.
    /// The edit uses the per-call limits and MIP gap associated with this
    /// solution.
    ///
    /// Use variables from the problem that produced this solution, with each
    /// variable appearing at most once in `expr`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Infeasible`] if the new constraint makes the problem
    /// infeasible. A malformed expression or a failure while solving is
    /// returned as [`Error::InternalError`].
    pub fn add_constraint(
        self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<SolveOutcome, Error> {
        let last_options = self.last_options.clone();
        let Self {
            direction,
            num_vars,
            state,
            ..
        } = self;
        match state {
            SolveState::Lp(mut solver) => {
                let expr = expr.into();
                let time_limit = solver.operation_time_limit;
                let stop = timed_lp_call(&mut solver, time_limit, move |solver| {
                    solver.add_constraint(
                        CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs)
                            .map_err(|error| Error::InternalError(error.2.to_string()))?,
                        cmp_op,
                        rhs,
                    )
                })?;
                Ok(SolveOutcome::from_lp_stop(
                    direction,
                    num_vars,
                    stop,
                    solver,
                    last_options,
                ))
            }
            SolveState::Mip(mut state) => {
                let expr = expr.into();
                let coefficients = CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs)
                    .map_err(|error| Error::InternalError(error.2.to_string()))?;
                state.base.constraints.push((coefficients, cmp_op, rhs));
                let run = mip::reedit_and_resolve(state)?;
                Ok(SolveOutcome::from_mip_run(
                    direction,
                    num_vars,
                    run,
                    last_options,
                ))
            }
        }
    }

    /// Consumes this solution, fixes a variable to `val`, and solves the edited
    /// problem.
    ///
    /// Fixing the same variable again replaces its previous fixed value. A later
    /// [`Solution::unfix_var`] restores the variable's original bounds. The
    /// returned [`SolveOutcome`] uses the per-call limits and MIP gap associated
    /// with this solution.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Infeasible`] if `val` is not finite, is outside the
    /// variable's original bounds, is incompatible with its integer or boolean
    /// domain, or leaves the edited problem without a feasible assignment. A
    /// numerical failure while re-solving is returned as
    /// [`Error::InternalError`].
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this solution.
    pub fn fix_var(self, var: Variable, val: f64) -> Result<SolveOutcome, Error> {
        let last_options = self.last_options.clone();
        let Self {
            direction,
            num_vars,
            state,
            ..
        } = self;
        assert!(var.0 < num_vars);
        if !val.is_finite() {
            return Err(Error::Infeasible);
        }
        match state {
            SolveState::Lp(mut solver) => {
                let time_limit = solver.operation_time_limit;
                let stop =
                    timed_lp_call(&mut solver, time_limit, |solver| solver.fix_var(var.0, val))?;
                Ok(SolveOutcome::from_lp_stop(
                    direction,
                    num_vars,
                    stop,
                    solver,
                    last_options,
                ))
            }
            SolveState::Mip(mut state) => {
                if val < state.base.var_mins[var.0] || val > state.base.var_maxs[var.0] {
                    return Err(Error::Infeasible);
                }
                state.fixed.insert(var.0, val);
                let run = mip::reedit_and_resolve(state)?;
                Ok(SolveOutcome::from_mip_run(
                    direction,
                    num_vars,
                    run,
                    last_options,
                ))
            }
        }
    }

    /// Consumes this solution and releases a fix created by
    /// [`Solution::fix_var`].
    ///
    /// When a fix exists, the variable's original bounds are restored and the
    /// edited problem is solved using the settings associated with this
    /// solution. The returned boolean is `true` when a fix was released. If the
    /// variable was not fixed, no edit is applied and the boolean is `false`.
    ///
    /// # Errors
    ///
    /// Returns errors encountered while solving after the fix is released.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this solution.
    pub fn unfix_var(self, var: Variable) -> Result<(SolveOutcome, bool), Error> {
        let last_options = self.last_options.clone();
        let Self {
            direction,
            num_vars,
            status,
            termination_reason,
            state,
            ..
        } = self;
        assert!(var.0 < num_vars);
        match state {
            SolveState::Lp(mut solver) => {
                let time_limit = solver.operation_time_limit;
                let (was_fixed, stop) =
                    timed_lp_call(&mut solver, time_limit, |solver| solver.unfix_var(var.0))?;
                Ok((
                    SolveOutcome::from_lp_stop(direction, num_vars, stop, solver, last_options),
                    was_fixed,
                ))
            }
            SolveState::Mip(mut state) => {
                if state.fixed.remove(&var.0).is_none() {
                    return Ok((
                        SolveOutcome::Solution(Solution {
                            direction,
                            num_vars,
                            status,
                            termination_reason,
                            state: SolveState::Mip(state),
                            last_options,
                        }),
                        false,
                    ));
                }
                let run = mip::reedit_and_resolve(state)?;
                Ok((
                    SolveOutcome::from_mip_run(direction, num_vars, run, last_options),
                    true,
                ))
            }
        }
    }
}

/// An outcome returned when a limit is reached before a usable solution exists.
///
/// It has a termination reason and statistics, but no objective or variable
/// values because no validated assignment is available. It does not mean that
/// the problem is infeasible. Continue it through [`SolveOutcome::resume`] or
/// [`SolveOutcome::resume_with`].
#[derive(Clone)]
pub struct InterruptedSolve {
    direction: OptimizationDirection,
    num_vars: usize,
    termination_reason: TerminationReason,
    state: SolveState,
    last_options: ResumeOptions,
}

impl std::fmt::Debug for InterruptedSolve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterruptedSolve")
            .field("direction", &self.direction)
            .field("num_vars", &self.num_vars)
            .field("termination_reason", &self.termination_reason)
            .field("stats", &self.stats())
            .finish()
    }
}

impl InterruptedSolve {
    /// Returns the limit that interrupted the most recent call.
    pub fn termination_reason(&self) -> TerminationReason {
        self.termination_reason
    }

    /// Returns statistics accumulated before the interruption.
    pub fn stats(&self) -> Stats {
        match &self.state {
            SolveState::Lp(solver) => Stats {
                lp_iterations: solver.lp_iterations,
                elapsed: solver.elapsed,
                ..Stats::default()
            },
            SolveState::Mip(state) => state.stats,
        }
    }

    /// Returns the settings used by the immediately preceding solve, resume, or
    /// post-solve edit call.
    pub fn last_resume_options(&self) -> ResumeOptions {
        self.last_options.clone()
    }
}

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    /// Returns a variable's unrounded value with `solution[var]` syntax.
    ///
    /// For an integer or boolean variable, use [`Solution::var_value`] when an
    /// exact domain value is desired.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    fn index(&self, var: Variable) -> &Self::Output {
        assert!(var.0 < self.num_vars);
        match &self.state {
            SolveState::Lp(solver) => solver.get_value(var.0),
            SolveState::Mip(state) => {
                &state
                    .incumbent
                    .as_ref()
                    .expect("a public MIP Solution must have an incumbent")
                    .values[var.0]
            }
        }
    }
}

/// Iterates over a [`Solution`]'s variables in creation order.
///
/// Each item contains the variable and the same domain-aware value returned by
/// [`Solution::var_value`].
#[derive(Debug, Clone)]
pub struct SolutionIter<'a> {
    solution: &'a Solution,
    var_idx: usize,
}

impl<'a> Iterator for SolutionIter<'a> {
    type Item = (Variable, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.var_idx < self.solution.num_vars {
            let var = Variable(self.var_idx);
            self.var_idx += 1;
            Some((var, self.solution.var_value(var)))
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Solution {
    type Item = (Variable, f64);
    type IntoIter = SolutionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
