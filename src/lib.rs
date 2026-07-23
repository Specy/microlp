/*!
A fast linear programming solver library.

[Linear programming](https://en.wikipedia.org/wiki/Linear_programming) is a technique for
finding the minimum (or maximum) of a linear function of a set of continuous variables
subject to linear equality and inequality constraints.

# Features

* Pure Rust implementation.
* Able to solve problems with hundreds of thousands of variables and constraints.
* Incremental: add constraints to an existing solution without solving it from scratch.
* Interruptible: time/node limits with clean resume, warm starts from known solutions,
  and MIP gap reporting for integer problems.

# Entry points

Begin by creating a [`Problem`](struct.Problem.html) instance, declaring variables and adding
constraints. Solving it produces a [`SolveOutcome`]. A
[`SolveOutcome::Solution`] contains a validated optimal or feasible assignment;
[`SolveOutcome::Interrupted`] contains resumable search state but no answer
values.

# Example

```
use microlp::{Problem, OptimizationDirection, ComparisonOp};

// Maximize an objective function x + 2 * y of two variables x >= 0 and 0 <= y <= 3
let mut problem = Problem::new(OptimizationDirection::Maximize);
let x = problem.add_var(1.0, (0.0, f64::INFINITY));
let y = problem.add_var(2.0, (0.0, 3.0));

// subject to constraints: x + y <= 4 and 2 * x + y >= 2.
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

// Optimal value is 7, achieved at x = 1 and y = 3.
let solution = problem.solve().unwrap().into_solution().unwrap();
assert_eq!(solution.objective(), 7.0);
assert_eq!(solution[x], 1.0);
assert_eq!(solution[y], 3.0);
```
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

/// An enum indicating whether to minimize or maximize objective function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizationDirection {
    /// Minimize the objective function.
    Minimize,
    /// Maximize the objective function.
    Maximize,
}

/// A reference to a variable in a linear programming problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Variable(pub(crate) usize);

impl Variable {
    /// Sequence number of the variable.
    ///
    /// Variables are referenced by their number in the addition sequence. The method returns
    /// this number.
    pub fn idx(&self) -> usize {
        self.0
    }
}

/// A sum of variables multiplied by constant coefficients used as a left-hand side
/// when defining constraints.
#[derive(Clone, Debug)]
pub struct LinearExpr {
    vars: Vec<usize>,
    coeffs: Vec<f64>,
}

impl LinearExpr {
    /// Creates an empty linear expression.
    pub fn empty() -> Self {
        Self {
            vars: vec![],
            coeffs: vec![],
        }
    }

    /// Add a single term to the linear expression.
    ///
    /// Variables can be added to an expression in any order, but adding the same variable
    /// several times is forbidden (the [`Problem::add_constraint`] method will panic).
    ///
    /// [`Problem::add_constraint`]: struct.Problem.html#method.add_constraint
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

/// An operator specifying the relation between left-hand and right-hand sides of the constraint.
#[derive(Clone, Copy, Debug)]
pub enum ComparisonOp {
    /// The == operator (equal to)
    Eq,
    /// The <= operator (less than or equal to)
    Le,
    /// The >= operator (greater than or equal to)
    Ge,
}

/// An error encountered while solving a problem.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Constrains can't simultaneously be satisfied.
    Infeasible,
    /// The objective function is unbounded.
    Unbounded,
    /// A [`SolveOptions`] value is out of range: a non-finite or negative gap or
    /// tolerance, or an integrality tolerance not in `[0, 0.5)`. The message
    /// names the offending field. This is a caller error — fix the option value
    /// and re-solve.
    InvalidOptions(String),
    /// The requested operation is not valid in the current solver state.
    InvalidOperation(String),
    /// An internal error occurred.
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

/// A specification of a linear programming problem.
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
/// The domain of a variable.
pub enum VarDomain {
    /// The variable is integer.
    Integer,
    /// The variable is real.
    Real,
    /// The variable is boolean T/F.
    Boolean,
}

impl Problem {
    /// Create a new problem instance.
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

    /// Set a time limit for the solver.
    ///
    /// If the budget expires, [`Problem::solve`] returns a feasible
    /// [`Solution`] when an incumbent exists, or
    /// [`SolveOutcome::Interrupted`] otherwise. Both outcomes can be
    /// continued with [`SolveOutcome::resume`].
    ///
    /// The implementation uses [`web_time::Instant`] under the hood, which works
    /// on both native and WebAssembly targets.
    pub fn set_time_limit(&mut self, duration: Duration) {
        self.time_limit = Some(duration);
    }

    /// Add a new real variable to the problem.
    ///
    /// `obj_coeff` is a coefficient of the term in the objective function corresponding to this
    /// variable, `min` and `max` are the minimum and maximum (inclusive) bounds of this
    /// variable. If one of the bounds is absent, use `f64::NEG_INFINITY` for minimum and
    /// `f64::INFINITY` for maximum.
    pub fn add_var(&mut self, obj_coeff: f64, (min, max): (f64, f64)) -> Variable {
        self.internal_add_var(obj_coeff, (min, max), VarDomain::Real)
    }

    /// Add a new integer variable to the problem.
    ///
    /// `obj_coeff` is a coefficient of the term in the objective function corresponding to this
    /// variable, `min` and `max` are the minimum and maximum (inclusive) bounds of this
    /// variable. If one of the bounds is absent, use `i32::MIN` for minimum and `i32::MAX` for
    /// maximum.
    pub fn add_integer_var(&mut self, obj_coeff: f64, (min, max): (i32, i32)) -> Variable {
        self.internal_add_var(obj_coeff, (min as f64, max as f64), VarDomain::Integer)
    }

    /// Check if the problem has any integer variables.
    pub fn has_integer_vars(&self) -> bool {
        self.var_domains
            .iter()
            .any(|v| *v == VarDomain::Integer || *v == VarDomain::Boolean)
    }

    /// Add a new binary variable to the problem.
    ///
    /// `obj_coeff` is a coefficient of the term in the objective function corresponding to this variable.
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

    /// Add a linear constraint to the problem.
    ///
    /// # Panics
    ///
    /// Will panic if a variable was added more than once to the left-hand side expression.
    ///
    /// # Examples
    ///
    /// Left-hand side of the constraint can be specified in several ways:
    /// ```
    /// # use microlp::*;
    /// let mut problem = Problem::new(OptimizationDirection::Minimize);
    /// let x = problem.add_var(1.0, (0.0, f64::INFINITY));
    /// let y = problem.add_var(1.0, (0.0, f64::INFINITY));
    ///
    /// // Add an x + y >= 2 constraint, specifying the left-hand side expression:
    ///
    /// // * by passing a slice of pairs (useful when explicitly enumerating variables)
    /// problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 2.0);
    ///
    /// // * by passing an iterator of variable-coefficient pairs.
    /// let vars = [x, y];
    /// problem.add_constraint(vars.iter().map(|&v| (v, 1.0)), ComparisonOp::Ge, 2.0);
    ///
    /// // * by manually constructing a LinearExpr.
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

    /// Solve with default options (respecting [`Problem::set_time_limit`] if set).
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if no feasible (integer) point exists,
    /// [`Error::Unbounded`] if the objective is unbounded.
    pub fn solve(&self) -> Result<SolveOutcome, Error> {
        let options = SolveOptions {
            time_limit: self.time_limit,
            ..SolveOptions::default()
        };
        self.solve_with(options)
    }

    /// Solve with explicit [`SolveOptions`].
    ///
    /// A limit is an outcome rather than an error. If a valid incumbent exists,
    /// the result is [`SolveOutcome::Solution`] with
    /// [`SolutionStatus::Feasible`]. Otherwise it is
    /// [`SolveOutcome::Interrupted`], which exposes no answer values.
    ///
    /// # Errors
    ///
    /// See [`Problem::solve`]. Returns [`Error::InvalidOptions`] when a
    /// numeric solve option is non-finite or outside its documented range.
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

/// The result of a bounded solve call.
///
/// A [`Solution`] always contains a validated feasible assignment.
/// [`InterruptedSolve`] contains resumable search state but deliberately has no
/// objective or variable-value accessors.
#[derive(Clone)]
pub enum SolveOutcome {
    /// A usable optimal or feasible solution.
    Solution(Solution),
    /// A limit fired before any usable solution was available.
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

    /// Returns the usable solution, if this outcome contains one.
    pub fn solution(&self) -> Option<&Solution> {
        match self {
            Self::Solution(solution) => Some(solution),
            Self::Interrupted(_) => None,
        }
    }

    /// Extracts the usable solution or returns the interrupted search state.
    pub fn into_solution(self) -> Result<Solution, InterruptedSolve> {
        match self {
            Self::Solution(solution) => Ok(solution),
            Self::Interrupted(interrupted) => Err(interrupted),
        }
    }

    /// Why this solve call returned.
    pub fn termination_reason(&self) -> TerminationReason {
        match self {
            Self::Solution(solution) => solution.termination_reason(),
            Self::Interrupted(interrupted) => interrupted.termination_reason(),
        }
    }

    /// Statistics accumulated by this solve.
    pub fn stats(&self) -> Stats {
        match self {
            Self::Solution(solution) => solution.stats(),
            Self::Interrupted(interrupted) => interrupted.stats(),
        }
    }

    /// Whether this outcome contains a proof-complete optimal solution.
    pub fn is_optimal(&self) -> bool {
        matches!(
            self,
            Self::Solution(Solution {
                status: SolutionStatus::Optimal,
                ..
            })
        )
    }

    /// Options to re-apply the budgets and target used in the previous solve call.
    pub fn last_resume_options(&self) -> ResumeOptions {
        match self {
            Self::Solution(solution) => solution.last_options.clone(),
            Self::Interrupted(interrupted) => interrupted.last_options.clone(),
        }
    }

    /// Continue search, passing back the same options that were used in the last call.
    pub fn resume(self) -> Result<Self, Error> {
        let options = self.last_resume_options();
        self.resume_with(options)
    }

    /// Continue with explicit fresh budgets and an optional new MIP gap.
    ///
    /// Time and node limits are fresh per-call budgets (`None` = unlimited).
    /// `options.mip_gap == None` means no MIP gap target (exact optimality `0.0`).
    pub fn resume_with(self, options: ResumeOptions) -> Result<Self, Error> {
        if self.is_optimal() {
            return Ok(self);
        }
        options.validate()?;
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

/// A validated feasible assignment returned by a solve.
///
/// For integer problems this stores the incumbent plus opaque resumable search
/// state. For pure LPs it keeps the live simplex basis for incremental edits.
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
    /// Whether this usable solution is proof-complete or feasible-but-unproven.
    pub fn status(&self) -> SolutionStatus {
        self.status
    }

    /// Why the solve call that produced this solution returned.
    pub fn termination_reason(&self) -> TerminationReason {
        self.termination_reason
    }

    /// Objective value in the problem's original optimization direction.
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

    /// The variable's value as stored in the validated solution.
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

    /// Value of the variable, rounded to an exact integer for integer/boolean
    /// variables.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range, or if an accepted integer value is
    /// further than the default [`Tolerances::integrality_rounding`] from an
    /// integer, which indicates a solver bug.
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

    /// Relative MIP gap of this solution.
    ///
    /// Returns `None` until both an incumbent and a proven bound exist. An
    /// optimal LP always reports `Some(0.0)`.
    pub fn gap(&self) -> Option<f64> {
        match &self.state {
            SolveState::Lp(_) => Some(0.0),
            SolveState::Mip(state) => state.stats.gap,
        }
    }

    /// Solve statistics accumulated across resumes.
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

    /// Iterate over variable/value pairs.
    pub fn iter(&self) -> SolutionIter<'_> {
        SolutionIter {
            solution: self,
            var_idx: 0,
        }
    }

    /// Options used in the previous solve call.
    pub fn last_resume_options(&self) -> ResumeOptions {
        self.last_options.clone()
    }

    /// Add a constraint and re-solve, returning a typed outcome.
    ///
    /// LP solutions re-solve incrementally from the live basis. MILP solutions
    /// restart from the edited base model and retain the incumbent as a warm
    /// start only when it remains feasible.
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
                    direction, num_vars, stop, solver, last_options,
                ))
            }
            SolveState::Mip(mut state) => {
                let expr = expr.into();
                let coefficients = CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs)
                    .map_err(|error| Error::InternalError(error.2.to_string()))?;
                state.base.constraints.push((coefficients, cmp_op, rhs));
                let run = mip::reedit_and_resolve(state)?;
                Ok(SolveOutcome::from_mip_run(
                    direction, num_vars, run, last_options,
                ))
            }
        }
    }

    /// Fix a variable and re-solve, returning a typed outcome.
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the fix is outside the original bounds or
    /// `val` is not finite.
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
                    direction, num_vars, stop, solver, last_options,
                ))
            }
            SolveState::Mip(mut state) => {
                if val < state.base.var_mins[var.0] || val > state.base.var_maxs[var.0] {
                    return Err(Error::Infeasible);
                }
                state.fixed.insert(var.0, val);
                let run = mip::reedit_and_resolve(state)?;
                Ok(SolveOutcome::from_mip_run(
                    direction, num_vars, run, last_options,
                ))
            }
        }
    }

    /// Undo a previous [`Solution::fix_var`].
    ///
    /// The boolean reports whether the variable was actually fixed.
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

/// A resumable search that has not produced a usable solution.
///
/// ```compile_fail
/// # use microlp::InterruptedSolve;
/// fn invalid(interrupted: &InterruptedSolve) {
///     let _ = interrupted.objective();
/// }
/// ```
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
    /// Why the search was interrupted.
    pub fn termination_reason(&self) -> TerminationReason {
        self.termination_reason
    }

    /// Solver statistics accumulated before interruption.
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

    /// Options used in the previous solve call.
    pub fn last_resume_options(&self) -> ResumeOptions {
        self.last_options.clone()
    }
}

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    /// Raw value access for a validated solution.
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

/// An iterator over the variable-value pairs of a [`Solution`].
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
