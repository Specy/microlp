//! [`Solution`]: a validated assignment, everything that can be read off it,
//! and the post-solve edits that continue from it.

use crate::problem::{check_row, row_of};
use crate::range::CoefficientRange;
use crate::solver::Solver;
use crate::{
    mip, timed_lp_call, ComparisonOp, Constraint, CsVec, Error, LinearExpr, OptimizationDirection,
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
    /// For an LP solution, the dual of each of the user's rows and the
    /// reduced cost of each var, in user units and internal (minimize) space
    /// (`Solver::duals`). Empty for a MIP.
    pub(crate) lp_duals: Vec<f64>,
    pub(crate) lp_reduced_costs: Vec<f64>,
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
    /// Round it where an exact integer is needed: [`Solution::var_domain`]
    /// tells which variables are integer or boolean, and its documentation
    /// why the rounded values need not meet the constraints.
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

    /// Returns a variable's domain: whether it is real, integer or boolean.
    ///
    /// This is how to round the values of the integer and boolean variables
    /// where exact integers are needed: [`Solution::var_value`] reports such a
    /// value as the solver computed it, within [`SolveOptions::int_tol`] of an
    /// integer, not necessarily on it. Bear in mind that the rounded values
    /// need not meet the constraints: under `x <= 1e7 * y`, a `y` of `1e-6`
    /// allows `x` up to 10, which `y` rounded to zero forbids.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    ///
    /// # Examples
    ///
    /// ```
    /// use microlp::{ComparisonOp, OptimizationDirection, Problem, VarDomain};
    ///
    /// let mut problem = Problem::new(OptimizationDirection::Maximize);
    /// let x = problem.add_var(1.0, (0.0, 2.5));
    /// let n = problem.add_integer_var(2.0, (0, 10));
    /// problem.add_constraint(&[(x, 1.0), (n, 1.0)], ComparisonOp::Le, 7.5);
    /// let solution = problem.solve().unwrap().into_solution().unwrap();
    ///
    /// // Exact integers for the integer and boolean variables, the others as
    /// // computed.
    /// let values: Vec<f64> = solution
    ///     .iter()
    ///     .map(|(var, value)| match solution.var_domain(var) {
    ///         VarDomain::Integer | VarDomain::Boolean => value.round(),
    ///         VarDomain::Real => value,
    ///     })
    ///     .collect();
    /// assert_eq!(values[n.idx()], 7.0);
    /// assert!((values[x.idx()] - 0.5).abs() < 1e-6);
    /// ```
    pub fn var_domain(&self, var: Variable) -> VarDomain {
        assert!(var.0 < self.num_vars);
        match &self.state {
            SolveState::Lp(solver) => solver.orig_var_domains[var.0],
            SolveState::Mip(state) => state.base.var_domains[var.0],
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

    /// Returns a constraint's dual value (shadow price): the rate at which the
    /// optimal objective changes as the constraint's right-hand side
    /// increases, in the objective's units per unit of the right-hand side.
    /// `None` for a problem with integer or boolean variables, whose
    /// solutions carry no duals.
    ///
    /// The sign follows from that definition. When minimizing, a
    /// [`ComparisonOp::Le`] constraint's dual is at most zero and a
    /// [`ComparisonOp::Ge`] constraint's at least zero (raising the
    /// right-hand side loosens the first and tightens the second); when
    /// maximizing, the other way round. A [`ComparisonOp::Eq`] constraint's
    /// may have either sign. A constraint that does not bind at the solution
    /// has dual zero, within the tolerances, as has one that every point
    /// satisfies, or that [`SolveOptions::presolve`] found implied by the
    /// variable bounds (the bounds' reduced costs then carry its price).
    ///
    /// The duals are those of the final simplex basis. Where several sets of
    /// duals are optimal (a degenerate optimum), this is one of them, which
    /// another solver may not pick. They meet dual feasibility within the
    /// solve's tolerances. After [`Solution::fix_var`], [`Solution::unfix_var`]
    /// or [`Solution::add_constraint`], the new solution's duals are those of
    /// the edited problem.
    ///
    /// # Panics
    ///
    /// Panics if `constraint` is out of range for this solution.
    ///
    /// # Examples
    ///
    /// ```
    /// use microlp::{ComparisonOp, OptimizationDirection, Problem};
    ///
    /// // Meet a demand of 10 from two generators, the cheaper one limited
    /// // to 6.
    /// let mut problem = Problem::new(OptimizationDirection::Minimize);
    /// let cheap = problem.add_var(20.0, (0.0, f64::INFINITY));
    /// let dear = problem.add_var(50.0, (0.0, f64::INFINITY));
    /// let demand = problem.add_constraint(&[(cheap, 1.0), (dear, 1.0)], ComparisonOp::Eq, 10.0);
    /// let limit = problem.add_constraint(&[(cheap, 1.0)], ComparisonOp::Le, 6.0);
    /// let solution = problem.solve().unwrap().into_solution().unwrap();
    ///
    /// // One more unit of demand comes from the dearer generator, and one
    /// // more unit of the cheap one's limit saves 50 - 20.
    /// assert!((solution.dual(demand).unwrap() - 50.0).abs() < 1e-9);
    /// assert!((solution.dual(limit).unwrap() + 30.0).abs() < 1e-9);
    /// ```
    pub fn dual(&self, constraint: Constraint) -> Option<f64> {
        let duals = self.duals()?;
        assert!(
            constraint.0 < duals.len(),
            "constraint {} is out of range 0..{}",
            constraint.0,
            duals.len()
        );
        Some(duals[constraint.0])
    }

    /// Returns the dual of every constraint, indexed by [`Constraint::idx`]:
    /// the problem's constraints in the order added, then those
    /// [`Solution::add_constraint`] added. `None` for a problem with integer
    /// or boolean variables. See [`Solution::dual`].
    pub fn duals(&self) -> Option<&[f64]> {
        match &self.state {
            SolveState::Lp(_) => Some(&self.lp_duals),
            SolveState::Mip(_) => None,
        }
    }

    /// Returns a variable's reduced cost: the rate at which the objective
    /// changes as the variable moves away from its value with the other
    /// variables at their bounds held there, in the objective's units per
    /// unit of the variable. It equals the variable's objective coefficient
    /// less the sum, over the constraints, of its coefficient times the
    /// constraint's [dual](Solution::dual). `None` for a problem with integer
    /// or boolean variables.
    ///
    /// A variable strictly between its bounds has reduced cost zero, within
    /// the tolerances; one at a bound, the marginal value of that bound (when
    /// minimizing, at least zero at a lower bound and at most zero at an upper
    /// bound; the other way round when maximizing). For a variable fixed by
    /// its bounds or by [`Solution::fix_var`], it is the marginal value of the
    /// fix.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this solution.
    pub fn reduced_cost(&self, var: Variable) -> Option<f64> {
        assert!(var.0 < self.num_vars);
        match &self.state {
            SolveState::Lp(_) => Some(self.lp_reduced_costs[var.0]),
            SolveState::Mip(_) => None,
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
    /// Returns [`Error::InvalidOperation`], before anything is solved, when
    /// `expr` repeats a variable or contains a variable index outside the
    /// problem, or when a coefficient or `rhs` is a number
    /// [`Problem::solve_with`](crate::Problem::solve_with) refuses; the
    /// message numbers the constraint after the problem's and those added
    /// before it. Returns [`Error::Infeasible`] when the solver proves the
    /// edited problem infeasible (see [`Error::Infeasible`]), and
    /// [`Error::InternalError`] for a failure while solving.
    pub fn add_constraint(
        self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<SolveOutcome, Error> {
        let mut range = self.state.coefficient_range();
        let row = range.rows();
        let coefficients =
            row_of(expr.into(), self.num_vars, row).map_err(Error::InvalidOperation)?;
        check_row(&coefficients, cmp_op, rhs, row)?;
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
    /// Returns [`Error::InvalidOperation`] if `val` is NaN or infinite. Returns
    /// [`Error::Infeasible`] if `val` is outside the variable's original
    /// bounds or is incompatible with its integer or boolean domain, and when
    /// the solver proves the edited problem infeasible (see
    /// [`Error::Infeasible`]). A `min <= -2^52` or `max >= 2^52` counts as
    /// infinite here too (see [`Problem::add_var`](crate::Problem::add_var)),
    /// and the bound and domain checks allow the solve's tolerances, so any
    /// value this solution reports is accepted (an integer variable's within
    /// [`SolveOptions::int_tol`] of an integer, one fixed at an integer on
    /// it): `val` may lie up to
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility), or the
    /// round-off floor at a large bound, outside a bound. A continuous variable
    /// is fixed at `val` itself, and an integer variable at the integer within
    /// `int_tol` of `val` (of the bound, where `val` lies outside it). A MILP
    /// edit searches anew: where this solution has an integer variable past a
    /// bound or off its integer, a fix at one of its values can leave a
    /// problem feasible only that way, and the edit can then answer
    /// `Infeasible` (see
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility)). The same
    /// holds for any edit where this solution used the tolerances of several
    /// bounds and rows at once: with the variable fixed exactly there, the
    /// others can be left less room than the infeasibility proof grants.
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
            return Err(Error::InvalidOperation(format!(
                "invalid fix_var value {val} of variable {}: expected a finite value",
                var.0
            )));
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
                // A continuous var is fixed at `val` itself, as on the LP path
                // (`Solver::fix_var` says why); an integer var its bounds fix
                // has no tolerance, so it is fixed at an integer within them.
                let mut val = val;
                if matches!(
                    state.base.var_domains[var.0],
                    VarDomain::Integer | VarDomain::Boolean
                ) {
                    let clamped = val.clamp(min, max);
                    let integer = clamped.round();
                    if (clamped - integer).abs() > state.options.int_tol {
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
                            lp_duals: Vec::new(),
                            lp_reduced_costs: Vec::new(),
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
