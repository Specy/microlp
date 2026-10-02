//! [`Solution`]: a validated assignment, everything that can be read off it,
//! and the post-solve edits that continue from it.

use crate::range::CoefficientRange;
use crate::solver::Solver;
use crate::{
    mip, timed_lp_call, ComparisonOp, CsVec, Error, LinearExpr, OptimizationDirection,
    ResumeOptions, SolutionStatus, SolveOptions, SolveOutcome, Stats, TerminationReason, VarDomain,
    Variable,
};

#[derive(Clone)]
pub(crate) enum SolveState {
    Lp(Box<Solver>),
    Mip(Box<mip::MipState>),
}

impl SolveState {
    /// The spread of the rows of the problem as edited so far, which a loud
    /// error of the next call names ([`CoefficientRange`]).
    pub(crate) fn coefficient_range(&self) -> CoefficientRange {
        match self {
            Self::Lp(solver) => solver.range.clone(),
            Self::Mip(state) => CoefficientRange::of(&state.base.constraints),
        }
    }
}

/// A validated feasible assignment returned by a solve-like call.
///
/// Use [`Solution::status`] to distinguish a proven optimum from an assignment
/// that is feasible without a proof of optimality.
#[derive(Clone)]
pub struct Solution {
    pub(crate) direction: OptimizationDirection,
    pub(crate) num_vars: usize,
    pub(crate) status: SolutionStatus,
    pub(crate) termination_reason: TerminationReason,
    pub(crate) state: SolveState,
    pub(crate) last_options: ResumeOptions,
    /// For an LP solution, the verified values in user units
    /// (`Solver::reported_values`), kept because the engine holds them scaled
    /// and `Index` must borrow them. Empty for a MIP, which uses its incumbent.
    pub(crate) lp_values: Vec<f64>,
    /// For an LP solution, the objective of `lp_values` in internal (minimize)
    /// space, so the reported objective is exactly that of the reported values.
    pub(crate) lp_objective: f64,
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
            SolveState::Lp(_) => self.lp_objective,
            SolveState::Mip(state) => state.current_objective(),
        };
        match self.direction {
            OptimizationDirection::Minimize => internal,
            OptimizationDirection::Maximize => -internal,
        }
    }

    /// Returns a variable's value, as the solver computed it.
    ///
    /// The values meet the variable bounds and the constraints within
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility). An integer
    /// or boolean variable's value is within [`SolveOptions::int_tol`] of an
    /// integer, not necessarily on it: values like `0.9999999` are normal. One
    /// fixed at an integer (by its bounds, the search or an edit) lies on it.
    /// Use `.round()` where an exact integer is needed, bearing in mind that
    /// the rounded values need not meet the constraints: under
    /// `x <= 1e9 * y`, rounding a `y` of `1e-9` to zero moves the constraint
    /// by `1`.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    pub fn var_value(&self, var: Variable) -> f64 {
        assert!(var.0 < self.num_vars);
        match &self.state {
            SolveState::Lp(_) => self.lp_values[var.0],
            SolveState::Mip(state) => {
                state
                    .incumbent
                    .as_ref()
                    .expect("a public MIP Solution must have an incumbent")
                    .values[var.0]
            }
        }
    }

    /// Returns a variable's value: the same as [`Solution::var_value`], which
    /// no longer rounds.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    #[deprecated(note = "values are reported as computed: use `Solution::var_value`")]
    pub fn var_value_raw(&self, var: Variable) -> f64 {
        self.var_value(var)
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

    /// Iterates over all variables and their values in creation order, as
    /// [`Solution::var_value`] returns them.
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
        let expr = expr.into();
        let coefficients = CsVec::new_from_unsorted(self.num_vars, expr.vars, expr.coeffs)
            .map_err(|error| Error::InternalError(error.2.to_string()))?;
        let mut range = self.state.coefficient_range();
        range.add_row(&coefficients);
        self.add_row(coefficients, cmp_op, rhs, range.clone())
            .map_err(|error| range.annotate(error))
    }

    /// [`Solution::add_constraint`] of the row `coefficients cmp_op rhs`;
    /// `range` is the edited problem's.
    fn add_row(
        self,
        coefficients: CsVec,
        cmp_op: ComparisonOp,
        rhs: f64,
        range: CoefficientRange,
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
                solver.range = range;
                let time_limit = solver.operation_time_limit;
                let stop = timed_lp_call(&mut solver, time_limit, move |solver| {
                    solver.add_constraint(coefficients, cmp_op, rhs)
                })?;
                SolveOutcome::from_lp_stop(direction, num_vars, stop, solver, last_options)
            }
            SolveState::Mip(mut state) => {
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
    /// domain, or leaves the edited problem without a feasible assignment.
    /// A `min <= -2^52` or `max >= 2^52` counts as infinite here too (see
    /// [`Problem::add_var`](crate::Problem::add_var)), and the bound and domain
    /// checks allow the solve's tolerances, so any value this solution reports
    /// is accepted (for an integer variable, with
    /// [`SolveOptions::int_tol`] at most
    /// [`Tolerances::integrality_rounding`](crate::Tolerances::integrality_rounding),
    /// as by default; one fixed at an integer lies on it): a value up to
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility) outside a
    /// bound is fixed at that bound, and a value within `integrality_rounding`
    /// of an integer is fixed at that integer.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this solution.
    pub fn fix_var(self, var: Variable, val: f64) -> Result<SolveOutcome, Error> {
        let range = self.state.coefficient_range();
        self.fix(var, val).map_err(|error| range.annotate(error))
    }

    /// [`Solution::fix_var`], its loud errors not yet naming the range.
    fn fix(self, var: Variable, val: f64) -> Result<SolveOutcome, Error> {
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
                SolveOutcome::from_lp_stop(direction, num_vars, stop, solver, last_options)
            }
            SolveState::Mip(mut state) => {
                let (min, max) = (state.base.var_mins[var.0], state.base.var_maxs[var.0]);
                let tolerances = &state.options.tolerances;
                if crate::solver::outside_bounds(val, min, max, tolerances.feasibility) {
                    return Err(Error::Infeasible);
                }
                let mut val = val.clamp(min, max);
                if matches!(
                    state.base.var_domains[var.0],
                    VarDomain::Integer | VarDomain::Boolean
                ) {
                    let integer = val.round();
                    if (val - integer).abs() > tolerances.integrality_rounding {
                        return Err(Error::Infeasible);
                    }
                    val = integer;
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
        let range = self.state.coefficient_range();
        self.unfix(var).map_err(|error| range.annotate(error))
    }

    /// [`Solution::unfix_var`], its loud errors not yet naming the range.
    fn unfix(self, var: Variable) -> Result<(SolveOutcome, bool), Error> {
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
                    SolveOutcome::from_lp_stop(direction, num_vars, stop, solver, last_options)?,
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
                            lp_values: Vec::new(),
                            lp_objective: 0.0,
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

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    /// Returns a variable's value with `solution[var]` syntax, as
    /// [`Solution::var_value`] returns it.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    fn index(&self, var: Variable) -> &Self::Output {
        assert!(var.0 < self.num_vars);
        match &self.state {
            SolveState::Lp(_) => &self.lp_values[var.0],
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
/// Each item contains the variable and the value [`Solution::var_value`]
/// returns.
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
