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
* Problems can be defined via an API or parsed from an
  [MPS](https://en.wikipedia.org/wiki/MPS_(format)) file.

# Entry points

Begin by creating a [`Problem`](struct.Problem.html) instance, declaring variables and adding
constraints. Solving it will produce a [`Solution`](struct.Solution.html) that can be used to
get the optimal objective value, corresponding variable values and to add more constraints
to the problem.

Alternatively, create an [`MpsFile`](mps/struct.MpsFile.html) by parsing a file in the MPS format.

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
let solution = problem.solve().unwrap();
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
mod mps;
mod ordering;
/// Problem solvers built on top of the microlp library.
pub mod problems_solvers;
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
            Error::InternalError(msg) => msg,
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

    /// Set a time limit for the solver. If the solver exceeds this duration,
    /// the returned solution's status will be [`Status::Feasible`] or
    /// [`Status::Interrupted`] and can be continued with [`Solution::resume`].
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

pub use mip::{SolveOptions, Stats, Status, Tolerances};

/// Internal signal for "a limit fired mid-simplex" (public API uses [`Status`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopReason {
    Limit,
    Finished,
}

impl Problem {
    /// Solve with default options (respecting [`Problem::set_time_limit`] if set).
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if no feasible (integer) point exists,
    /// [`Error::Unbounded`] if the objective is unbounded.
    pub fn solve(&self) -> Result<Solution, Error> {
        let options = SolveOptions {
            time_limit: self.time_limit,
            ..SolveOptions::default()
        };
        self.solve_with(options)
    }

    /// Solve with explicit [`SolveOptions`].
    ///
    /// Hitting a limit is NOT an error: the returned [`Solution`] has status
    /// [`Status::Feasible`] (an incumbent exists) or [`Status::Interrupted`]
    /// (none yet) and can be continued with [`Solution::resume`].
    ///
    /// # Errors
    ///
    /// See [`Problem::solve`].
    pub fn solve_with(&self, options: SolveOptions) -> Result<Solution, Error> {
        let num_vars = self.obj_coeffs.len();
        if self.has_integer_vars() {
            let run = mip::run(self, options)?;
            Ok(Solution {
                direction: self.direction,
                num_vars,
                status: mip::status_of(run.outcome, &run.state),
                kind: SolutionKind::Mip(Box::new(run.state)),
            })
        } else {
            let deadline = options.time_limit.map(|d| Instant::now() + d);
            let mut solver = Solver::try_new(
                &self.obj_coeffs,
                &self.var_mins,
                &self.var_maxs,
                &self.constraints,
                &self.var_domains,
                deadline,
            )?;
            let status = match solver.initial_solve()? {
                StopReason::Finished => Status::Optimal,
                StopReason::Limit => Status::Interrupted,
            };
            Ok(Solution {
                direction: self.direction,
                num_vars,
                status,
                kind: SolutionKind::Lp(Box::new(solver)),
            })
        }
    }
}

/// A solution of a problem.
///
/// For problems with integer variables this is plain data (values + status) plus
/// an opaque resumable search state; for pure-LP problems it keeps the live
/// simplex basis so constraints can be added incrementally.
#[derive(Clone)]
pub struct Solution {
    direction: OptimizationDirection,
    num_vars: usize,
    status: Status,
    kind: SolutionKind,
}

#[derive(Clone)]
enum SolutionKind {
    Lp(Box<solver::Solver>),
    Mip(Box<mip::MipState>),
}

impl std::fmt::Debug for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solution")
            .field("direction", &self.direction)
            .field("num_vars", &self.num_vars)
            .field("status", &self.status)
            .field("objective", &self.objective())
            .finish()
    }
}

impl Solution {
    /// The outcome of the solve: proven optimal, feasible-but-unproven, or interrupted.
    ///
    /// This is the field that decides whether the value accessors mean
    /// anything — see [`Solution::objective`].
    pub fn status(&self) -> Status {
        self.status
    }

    /// Objective value of the best solution known to this `Solution`, in the
    /// problem's original optimization direction.
    ///
    /// **What this number means depends on [`status()`](Self::status), and
    /// checking it is the caller's responsibility:**
    ///
    /// * [`Status::Optimal`] — the proven optimum.
    /// * [`Status::Feasible`] — the best incumbent found so far; a valid
    ///   feasible value, but not proven optimal ([`Solution::gap`] says how
    ///   far the proof got).
    /// * [`Status::Interrupted`] — no usable solution exists yet. Reading is
    ///   still allowed and returns the search's *current working value* (the
    ///   objective of the simplex point the solver was at when the limit
    ///   fired). Useful for inspection and progress reporting, but it is not
    ///   the answer to your problem — it may correspond to an infeasible or
    ///   fractional point. [`Solution::resume`] the search to make progress.
    pub fn objective(&self) -> f64 {
        let internal = match &self.kind {
            SolutionKind::Lp(solver) => solver.cur_obj_val,
            SolutionKind::Mip(state) => match &state.incumbent {
                Some(incumbent) => incumbent.objective,
                // No incumbent yet (interrupted early): the freshest number
                // the search has is the current node LP's objective.
                None => state.solver.cur_obj_val,
            },
        };
        match self.direction {
            OptimizationDirection::Minimize => internal,
            OptimizationDirection::Maximize => -internal,
        }
    }

    /// The variable's value, as recorded by the solve.
    ///
    /// For an LP solution this is the live value read straight from the
    /// simplex basis. For a MILP solution this is the value stored in the
    /// accepted incumbent, where integer/boolean variables are already
    /// rounded to an exact integer — despite the "raw" name, this is not an
    /// unrounded branch & bound leaf value.
    ///
    /// On [`Status::Interrupted`] there is no incumbent yet, and this returns
    /// the search's *current working point* instead (the value in the node LP
    /// the solver was at when the limit fired) — possibly fractional for
    /// integer variables and possibly infeasible. Only [`Status::Optimal`] /
    /// [`Status::Feasible`] values answer the problem; checking
    /// [`status()`](Self::status) is the caller's responsibility (see
    /// [`Solution::objective`]).
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    pub fn var_value_raw(&self, var: Variable) -> f64 {
        assert!(var.0 < self.num_vars);
        match &self.kind {
            SolutionKind::Lp(solver) => *solver.get_value(var.0),
            SolutionKind::Mip(state) => match &state.incumbent {
                Some(incumbent) => incumbent.values[var.0],
                None => *state.solver.get_value(var.0),
            },
        }
    }

    /// Value of the variable, rounded to an exact integer for integer/boolean vars.
    ///
    /// On [`Status::Interrupted`] there is no incumbent yet, and this returns
    /// the search's current working value *unrounded* — a mid-search point may
    /// legitimately be fractional for integer variables, and rounding it here
    /// would fabricate an answer. Checking [`status()`](Self::status) before
    /// treating values as the solution is the caller's responsibility (see
    /// [`Solution::objective`]).
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range, or if an *accepted* solution's integer
    /// variable is further than [`Tolerances::integrality_rounding`]'s
    /// *default* (`1e-5`) from an integer (indicates a solver bug). This
    /// sanity check intentionally always uses the default, never the
    /// `tolerances.integrality_rounding` configured for the solve that
    /// produced this solution: it exists to catch a solver bug — an accepted
    /// incumbent must already be integral-clean — not to reflect a caller's
    /// own (possibly loosened) preference.
    pub fn var_value(&self, var: Variable) -> f64 {
        let val = self.var_value_raw(var);
        if self.status == Status::Interrupted {
            // Mid-search working point: fractional integer values are
            // legitimate here, so no rounding and no integrality check.
            return val;
        }
        let domain = match &self.kind {
            SolutionKind::Lp(solver) => &solver.orig_var_domains[var.0],
            SolutionKind::Mip(state) => &state.solver.orig_var_domains[var.0],
        };
        if *domain == VarDomain::Integer || *domain == VarDomain::Boolean {
            let rounded = val.round();
            let tol = Tolerances::default().integrality_rounding;
            assert!(
                f64::abs(rounded - val) < tol,
                "Variable was expected to be an integer, got {}",
                val
            );
            rounded
        } else {
            val
        }
    }

    /// Relative MIP gap of the best known solution (`None` until an incumbent and
    /// a bound exist; always `Some(0.0)` for optimal LP solutions).
    pub fn gap(&self) -> Option<f64> {
        match &self.kind {
            SolutionKind::Lp(_) => (self.status == Status::Optimal).then_some(0.0),
            SolutionKind::Mip(state) => state.stats.gap,
        }
    }

    /// Solve statistics (nodes, simplex iterations, elapsed time).
    pub fn stats(&self) -> Stats {
        match &self.kind {
            SolutionKind::Lp(solver) => Stats {
                lp_iterations: solver.lp_iterations,
                ..Stats::default()
            },
            SolutionKind::Mip(state) => state.stats,
        }
    }

    /// Iterate over variable/value pairs (raw values, like [`Solution::var_value_raw`]).
    ///
    /// On [`Status::Interrupted`] the yielded values are the search's current
    /// working point, not a usable solution — see [`Solution::objective`] for
    /// the status contract.
    pub fn iter(&self) -> SolutionIter<'_> {
        SolutionIter {
            solution: self,
            var_idx: 0,
        }
    }

    /// Continue an interrupted solve with a fresh time budget (`None` = unlimited).
    /// A no-op on solutions whose status is already [`Status::Optimal`].
    ///
    /// # Errors
    ///
    /// Same as [`Problem::solve`].
    pub fn resume(mut self, time_limit: Option<Duration>) -> Result<Self, Error> {
        if self.status == Status::Optimal {
            return Ok(self);
        }
        match &mut self.kind {
            SolutionKind::Lp(solver) => {
                solver.deadline = time_limit.map(|d| Instant::now() + d);
                self.status = match solver.initial_solve()? {
                    StopReason::Finished => Status::Optimal,
                    StopReason::Limit => Status::Interrupted,
                };
            }
            SolutionKind::Mip(state) => {
                let outcome = mip::resume_run(state, time_limit)?;
                self.status = mip::status_of(outcome, state);
            }
        }
        Ok(self)
    }

    /// Add a constraint to the solved problem and re-solve.
    ///
    /// LP solutions re-solve incrementally from the live basis (dual simplex).
    /// MILP solutions are re-solved on the original problem plus all edits; the
    /// previous incumbent is kept as a warm start when it remains feasible.
    /// Editing a paused (`Feasible`/`Interrupted`) MILP solution is allowed — the
    /// open search tree is discarded and the search restarts from the edited
    /// problem.
    ///
    /// The MILP re-solve runs with a fresh budget taken from the original
    /// [`SolveOptions`] (each edit gets the full `time_limit`/`node_limit` again),
    /// whereas the LP re-solve inherits the original absolute deadline; the two may
    /// be aligned in a later release.
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the edit makes the problem infeasible.
    pub fn add_constraint(
        self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<Self, Error> {
        let Solution {
            direction,
            num_vars,
            status,
            kind,
        } = self;
        match kind {
            SolutionKind::Lp(mut solver) => {
                if status == Status::Interrupted {
                    return Err(Error::InternalError(
                        "cannot edit an interrupted solution; resume() it first".to_string(),
                    ));
                }
                let expr = expr.into();
                let sr = solver.add_constraint(
                    CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs)
                        .map_err(|v| Error::InternalError(v.2.to_string()))?,
                    cmp_op,
                    rhs,
                )?;
                Ok(Solution {
                    direction,
                    num_vars,
                    status: match sr {
                        StopReason::Finished => Status::Optimal,
                        StopReason::Limit => Status::Interrupted,
                    },
                    kind: SolutionKind::Lp(solver),
                })
            }
            SolutionKind::Mip(mut state) => {
                // Edit-after-pause is allowed by design: any status is editable here.
                let expr = expr.into();
                let coeffs = CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs)
                    .map_err(|v| Error::InternalError(v.2.to_string()))?;
                state.base.constraints.push((coeffs, cmp_op, rhs));
                let run = mip::reedit_and_resolve(state)?;
                Ok(Solution {
                    direction,
                    num_vars,
                    status: mip::status_of(run.outcome, &run.state),
                    kind: SolutionKind::Mip(Box::new(run.state)),
                })
            }
        }
    }

    /// Fix the variable to `val` and re-solve. See [`Solution::add_constraint`]
    /// for LP/MILP semantics.
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the fix makes the problem infeasible.
    pub fn fix_var(self, var: Variable, val: f64) -> Result<Self, Error> {
        let Solution {
            direction,
            num_vars,
            status,
            kind,
        } = self;
        assert!(var.0 < num_vars);
        match kind {
            SolutionKind::Lp(mut solver) => {
                if status == Status::Interrupted {
                    return Err(Error::InternalError(
                        "cannot edit an interrupted solution; resume() it first".to_string(),
                    ));
                }
                let sr = solver.fix_var(var.0, val)?;
                Ok(Solution {
                    direction,
                    num_vars,
                    status: match sr {
                        StopReason::Finished => Status::Optimal,
                        StopReason::Limit => Status::Interrupted,
                    },
                    kind: SolutionKind::Lp(solver),
                })
            }
            SolutionKind::Mip(mut state) => {
                if val < state.base.var_mins[var.0] || val > state.base.var_maxs[var.0] {
                    return Err(Error::Infeasible);
                }
                state.fixed.insert(var.0, val);
                let run = mip::reedit_and_resolve(state)?;
                Ok(Solution {
                    direction,
                    num_vars,
                    status: mip::status_of(run.outcome, &run.state),
                    kind: SolutionKind::Mip(Box::new(run.state)),
                })
            }
        }
    }

    /// Undo a previous [`Solution::fix_var`]. The boolean reports whether the
    /// variable was actually fixed.
    ///
    /// MILP solutions re-solve on the original problem plus the remaining edits
    /// (see [`Solution::add_constraint`]); the previous incumbent is kept as a
    /// warm start when it survives.
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the problem without this fix is infeasible — this
    /// is possible when a prior edit was interrupted before proving infeasibility
    /// and the unfix re-solve completes that proof. [`Error::InternalError`] on an
    /// internal re-solve failure. The pure-LP path never errors.
    pub fn unfix_var(self, var: Variable) -> Result<(Self, bool), Error> {
        let Solution {
            direction,
            num_vars,
            status,
            kind,
        } = self;
        assert!(var.0 < num_vars);
        match kind {
            SolutionKind::Lp(mut solver) => {
                let res = solver.unfix_var(var.0);
                Ok((
                    Solution {
                        direction,
                        num_vars,
                        status,
                        kind: SolutionKind::Lp(solver),
                    },
                    res,
                ))
            }
            SolutionKind::Mip(mut state) => {
                if state.fixed.remove(&var.0).is_none() {
                    return Ok((
                        Solution {
                            direction,
                            num_vars,
                            status,
                            kind: SolutionKind::Mip(state),
                        },
                        false,
                    ));
                }
                // Relaxing a fix cannot make a proven-feasible problem infeasible,
                // but base+fixes may still be infeasible if an earlier edit was
                // interrupted before proving it (the unfix re-solve can complete
                // that proof), and a node LP can fail internally. Propagate both
                // rather than converting reachable errors into panics.
                let run = mip::reedit_and_resolve(state)?;
                Ok((
                    Solution {
                        direction,
                        num_vars,
                        status: mip::status_of(run.outcome, &run.state),
                        kind: SolutionKind::Mip(Box::new(run.state)),
                    },
                    true,
                ))
            }
        }
    }

    /// Add a Gomory cut for `var`. Only available on pure-LP solutions.
    ///
    /// # Errors
    ///
    /// [`Error::Infeasible`] if the cut makes the problem infeasible;
    /// [`Error::InternalError`] if the solution is [`Status::Interrupted`]
    /// (`resume()` it first) or if it is a MILP solution (the cut reads the live
    /// simplex tableau).
    pub fn add_gomory_cut(mut self, var: Variable) -> Result<Self, Error> {
        assert!(var.0 < self.num_vars);
        match &mut self.kind {
            SolutionKind::Lp(solver) => {
                // Same guard as add_constraint/fix_var: the cut reaches solver
                // internals that assume a completed solve, so a half-solved
                // (interrupted) LP must be resumed before it can be edited.
                if self.status == Status::Interrupted {
                    return Err(Error::InternalError(
                        "cannot edit an interrupted solution; resume() it first".to_string(),
                    ));
                }
                let sr = solver.add_gomory_cut(var.0)?;
                self.status = match sr {
                    StopReason::Finished => Status::Optimal,
                    StopReason::Limit => Status::Interrupted,
                };
                Ok(self)
            }
            SolutionKind::Mip(_) => Err(Error::InternalError(
                "Gomory cuts require a pure-LP solution".to_string(),
            )),
        }
    }
}

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    /// Raw value access, like [`Solution::var_value_raw`]: on
    /// [`Status::Interrupted`] this is the search's current working point,
    /// not a usable solution (see [`Solution::objective`] for the status
    /// contract).
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    fn index(&self, var: Variable) -> &Self::Output {
        assert!(var.0 < self.num_vars);
        match &self.kind {
            SolutionKind::Lp(solver) => solver.get_value(var.0),
            SolutionKind::Mip(state) => match &state.incumbent {
                Some(incumbent) => &incumbent.values[var.0],
                None => state.solver.get_value(var.0),
            },
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
            Some((var, self.solution.var_value_raw(var)))
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

pub use mps::MpsFile;
