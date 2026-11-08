//! [`Solution`]: a validated assignment, everything that can be read off it,
//! and the post-solve edits that continue from it.

use crate::problem::{check_rhs, check_row, row_of};
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

/// A validated feasible assignment returned by a solve, resume, or edit.
/// For a problem with integer or boolean variables, it is the incumbent: the
/// best feasible solution found so far.
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
    /// Returns whether this solution is proven optimal or is only known to be
    /// feasible.
    pub fn status(&self) -> SolutionStatus {
        self.status
    }

    /// Returns why the solve, resume, or edit that produced this solution
    /// stopped.
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

    /// Returns a variable's value as the solver computed it.
    ///
    /// The value meets the variable bounds and constraints within the
    /// feasibility tolerance
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility). An integer
    /// or boolean variable's value is within the integrality tolerance
    /// [`SolveOptions::int_tol`] of an integer, but it need not equal that
    /// integer. Values like `0.9999999` are normal. A variable fixed at an
    /// integer by its bounds, branch & bound, or an edit has that exact value.
    /// Round integer and boolean values where you need exact integers.
    /// [`Solution::var_domain`] tells you which variables to round, and its
    /// documentation explains why the rounded values need not meet the
    /// constraints.
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

    /// Returns whether a variable is real, integer or boolean.
    ///
    /// Use this to decide which values to round when you need exact integers.
    /// [`Solution::var_value`] reports integer and boolean values as the solver
    /// computed them, within the integrality tolerance
    /// [`SolveOptions::int_tol`] of an integer, but not necessarily equal to
    /// it. The rounded values need not meet the constraints. For example,
    /// under `x <= 1e7 * y`, a `y` of `1e-6` allows `x` up to 10, which `y`
    /// rounded to zero forbids.
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

    /// Returns the relative gap between this solution and the best proven
    /// bound.
    ///
    /// Returns `None` when no proven bound is available yet. An optimal linear
    /// program reports `Some(0.0)`.
    pub fn gap(&self) -> Option<f64> {
        match &self.state {
            SolveState::Lp(_) => Some(0.0),
            SolveState::Mip(state) => state.stats.gap,
        }
    }

    /// Returns a constraint's dual value (shadow price): the change in the
    /// optimal objective per unit increase in the constraint's right-hand
    /// side, in objective units per right-hand-side unit. Returns `None` for a
    /// problem with integer or boolean variables, whose solutions have no
    /// dual values.
    ///
    /// The sign follows from this definition. When minimizing, a
    /// [`ComparisonOp::Le`] constraint's dual is at most zero because raising
    /// its right-hand side loosens the constraint. A [`ComparisonOp::Ge`]
    /// constraint's dual is at least zero because raising its right-hand side
    /// tightens the constraint. When maximizing, these signs are reversed. A
    /// [`ComparisonOp::Eq`] constraint's dual may have either sign. A
    /// constraint that does not bind at the solution has dual value zero
    /// within the tolerances. The same is true for a constraint that every
    /// point satisfies or that [`SolveOptions::presolve`] found implied by
    /// the variable bounds; the bounds' reduced costs then carry its price.
    ///
    /// The dual values come from the final simplex basis. If a degenerate
    /// optimum has several optimal sets of dual values, this is one of them;
    /// a different optimal basis may give different values. They meet dual
    /// feasibility within the solve's tolerances. After
    /// [`Solution::fix_var`], [`Solution::unfix_var`],
    /// [`Solution::add_constraint`], [`Solution::set_rhs`] or
    /// [`Solution::set_obj_coeff`], the new solution's dual values are those
    /// of the edited problem.
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

    /// Returns the dual value for every constraint, indexed by
    /// [`Constraint::idx`]. The order is the problem's constraints as added,
    /// followed by constraints added with [`Solution::add_constraint`].
    /// Returns `None` for a problem with integer or boolean variables. See
    /// [`Solution::dual`].
    pub fn duals(&self) -> Option<&[f64]> {
        match &self.state {
            SolveState::Lp(_) => Some(&self.lp_duals),
            SolveState::Mip(_) => None,
        }
    }

    /// Returns a variable's reduced cost: the rate at which the objective
    /// changes as the variable moves away from its value, in objective units
    /// per unit of the variable, while the other variables at a bound stay
    /// there. It equals the variable's objective coefficient minus the sum,
    /// over the constraints, of its coefficient times the constraint's
    /// [dual value](Solution::dual). Returns `None` for a problem with integer
    /// or boolean variables.
    ///
    /// A variable strictly between its bounds has reduced cost zero, within
    /// the tolerances. For a variable at a bound, the reduced cost is the
    /// marginal value of that bound: when minimizing, at least zero at a lower
    /// bound and at most zero at an upper bound, and the other way round when
    /// maximizing. For a variable fixed by its bounds or by
    /// [`Solution::fix_var`], it is the marginal value of the fix.
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

    /// Adds a constraint to this solution's problem and solves the edited
    /// problem, consuming this solution.
    ///
    /// The returned [`SolveOutcome`] can contain an optimal solution, a
    /// feasible solution without a proof of optimality, or an interrupted
    /// solve if a time limit or node limit is reached before a usable solution
    /// is found. The edit uses this solution's per-call time limit, node limit,
    /// and MIP gap. The returned [`Constraint`] identifies the new constraint.
    /// Use this handle with [`Solution::dual`] or [`Solution::set_rhs`] on
    /// later solutions. Its index follows the problem's constraints and
    /// constraints added earlier with [`Solution::add_constraint`].
    ///
    /// Use variables from the problem that produced this solution. Each
    /// variable may appear at most once in `expr`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOperation`] before solving if `expr` repeats a
    /// variable, contains a variable index outside the problem, or has a
    /// coefficient or `rhs` that
    /// [`Problem::solve_with`](crate::Problem::solve_with) refuses. The
    /// error message numbers the new constraint after the problem's
    /// constraints and those added earlier. Returns
    /// [`Error::Infeasible`] if the solver proves the edited problem
    /// infeasible (see [`Error::Infeasible`]), and [`Error::InternalError`]
    /// if solving fails.
    pub fn add_constraint(
        self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<(SolveOutcome, Constraint), Error> {
        let mut range = self.state.coefficient_range();
        let row = range.rows();
        let coefficients =
            row_of(expr.into(), self.num_vars, row).map_err(Error::InvalidOperation)?;
        check_row(&coefficients, cmp_op, rhs, row)?;
        range.add_row(&coefficients);
        let outcome = self
            .add_row(coefficients, cmp_op, rhs, range.clone())
            .map_err(|error| range.annotate(error))?;
        Ok((outcome, Constraint(row)))
    }

    /// Sets a constraint's right-hand side to `rhs` and solves the edited
    /// problem, consuming this solution.
    ///
    /// `constraint` must be one added with [`Problem::add_constraint`] or
    /// [`Solution::add_constraint`]. Its comparison and terms stay the same,
    /// and a later call can set its right-hand side again. The returned
    /// [`SolveOutcome`] uses this solution's per-call time limit, node limit,
    /// and MIP gap.
    ///
    /// A problem without integer or boolean variables re-solves from this
    /// solution's basis. The basis stays dual feasible, so the dual simplex
    /// usually needs only a few pivots. The new solution's
    /// [dual values](Solution::dual) are those of the edited problem. For a
    /// problem with integer or boolean variables, the solver starts a new
    /// branch & bound search and uses this solution as a warm start if it
    /// remains feasible for the edited problem.
    ///
    /// Set `rhs` to `inf` for [`ComparisonOp::Le`] or `-inf` for
    /// [`ComparisonOp::Ge`] to switch the constraint off. Set it to a finite
    /// value to switch the constraint on again.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOperation`] before solving if `rhs` is NaN or
    /// infinite with the wrong sign for the comparison: `-inf` for
    /// [`ComparisonOp::Le`], `inf` for [`ComparisonOp::Ge`], or either
    /// infinity for [`ComparisonOp::Eq`]. Returns [`Error::Infeasible`] if
    /// the solver proves the edited problem infeasible (see
    /// [`Error::Infeasible`]), and [`Error::InternalError`] if solving fails.
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
    /// let mut problem = Problem::new(OptimizationDirection::Minimize);
    /// let cheap = problem.add_var(20.0, (0.0, 6.0));
    /// let dear = problem.add_var(50.0, (0.0, f64::INFINITY));
    /// let demand = problem.add_constraint(&[(cheap, 1.0), (dear, 1.0)], ComparisonOp::Ge, 10.0);
    /// let solution = problem.solve().unwrap().into_solution().unwrap();
    /// assert!((solution.objective() - 320.0).abs() < 1e-9);
    ///
    /// // Demand falls to 5: the cheap generator alone meets it.
    /// let solution = solution.set_rhs(demand, 5.0).unwrap().into_solution().unwrap();
    /// assert!((solution.objective() - 100.0).abs() < 1e-9);
    /// assert!((solution.dual(demand).unwrap() - 20.0).abs() < 1e-9);
    /// ```
    ///
    /// [`Problem::add_constraint`]: crate::Problem::add_constraint
    pub fn set_rhs(self, constraint: Constraint, rhs: f64) -> Result<SolveOutcome, Error> {
        let range = self.state.coefficient_range();
        self.edit_rhs(constraint.0, rhs)
            .map_err(|error| range.annotate(error))
    }

    /// [`Solution::set_rhs`] of the user's row `row`, its loud errors not yet
    /// naming the range.
    fn edit_rhs(self, row: usize, rhs: f64) -> Result<SolveOutcome, Error> {
        let last_options = self.last_options.clone();
        let Self {
            direction,
            num_vars,
            state,
            ..
        } = self;
        match state {
            SolveState::Lp(mut solver) => {
                let rows = solver.num_user_rows();
                assert!(row < rows, "constraint {row} is out of range 0..{rows}");
                check_rhs(solver.user_row_op(row), rhs, row)?;
                let time_limit = solver.operation_time_limit;
                let stop =
                    timed_lp_call(&mut solver, time_limit, |solver| solver.set_rhs(row, rhs))?;
                SolveOutcome::from_lp_stop(direction, num_vars, stop, solver, last_options)
            }
            SolveState::Mip(mut state) => {
                let rows = state.base.constraints.len();
                assert!(row < rows, "constraint {row} is out of range 0..{rows}");
                let (_, cmp_op, old_rhs) = &mut state.base.constraints[row];
                check_rhs(*cmp_op, rhs, row)?;
                *old_rhs = rhs;
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

    /// Sets a variable's objective coefficient to `obj_coeff` and solves the
    /// edited problem, consuming this solution.
    ///
    /// Give `obj_coeff` in the problem's optimization direction, as for
    /// [`Problem::add_var`](crate::Problem::add_var). A later call can set it
    /// again. The returned [`SolveOutcome`] uses this solution's per-call time
    /// limit, node limit, and MIP gap.
    ///
    /// A problem without integer or boolean variables re-solves from this
    /// solution's basis. The basis stays primal feasible, so the primal
    /// simplex usually needs only a few pivots. For a problem with integer or
    /// boolean variables, the solver starts a new branch & bound search and
    /// uses this solution as a warm start. Changing the objective does not
    /// make this solution infeasible.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOperation`] before solving if `obj_coeff` is
    /// NaN or infinite. Returns [`Error::Unbounded`] if the edited objective
    /// has no finite optimum, and [`Error::InternalError`] if solving fails.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this solution.
    ///
    /// # Examples
    ///
    /// ```
    /// use microlp::{ComparisonOp, OptimizationDirection, Problem};
    ///
    /// let mut problem = Problem::new(OptimizationDirection::Minimize);
    /// let a = problem.add_var(20.0, (0.0, f64::INFINITY));
    /// let b = problem.add_var(50.0, (0.0, f64::INFINITY));
    /// problem.add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 10.0);
    /// let solution = problem.solve().unwrap().into_solution().unwrap();
    /// assert!((solution[a] - 10.0).abs() < 1e-9);
    ///
    /// // `a` becomes the dearer one: the demand moves to `b`.
    /// let solution = solution.set_obj_coeff(a, 60.0).unwrap().into_solution().unwrap();
    /// assert!((solution[b] - 10.0).abs() < 1e-9);
    /// assert!((solution.objective() - 500.0).abs() < 1e-9);
    /// ```
    pub fn set_obj_coeff(self, var: Variable, obj_coeff: f64) -> Result<SolveOutcome, Error> {
        let range = self.state.coefficient_range();
        self.edit_obj_coeff(var, obj_coeff)
            .map_err(|error| range.annotate(error))
    }

    /// [`Solution::set_obj_coeff`], its loud errors not yet naming the range.
    fn edit_obj_coeff(self, var: Variable, obj_coeff: f64) -> Result<SolveOutcome, Error> {
        let last_options = self.last_options.clone();
        let Self {
            direction,
            num_vars,
            state,
            ..
        } = self;
        assert!(var.0 < num_vars);
        if !obj_coeff.is_finite() {
            return Err(Error::InvalidOperation(format!(
                "invalid objective coefficient {obj_coeff} of variable {}: expected a finite value",
                var.0
            )));
        }
        // Stored in internal (minimize) space, as `Problem` stores it.
        let internal = match direction {
            OptimizationDirection::Minimize => obj_coeff,
            OptimizationDirection::Maximize => -obj_coeff,
        };
        match state {
            SolveState::Lp(mut solver) => {
                let time_limit = solver.operation_time_limit;
                let stop = timed_lp_call(&mut solver, time_limit, |solver| {
                    solver.set_obj_coeff(var.0, internal)
                })?;
                SolveOutcome::from_lp_stop(direction, num_vars, stop, solver, last_options)
            }
            SolveState::Mip(mut state) => {
                state.base.obj_coeffs[var.0] = internal;
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

    /// Fixes a variable at `val` and solves the edited problem, consuming this
    /// solution.
    ///
    /// Fixing the same variable again replaces its previous fixed value. A
    /// later [`Solution::unfix_var`] restores the variable's original bounds.
    /// The returned [`SolveOutcome`] uses this solution's per-call time limit,
    /// node limit, and MIP gap.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOperation`] if `val` is NaN or infinite.
    /// Returns [`Error::Infeasible`] if `val` violates an original bound
    /// beyond that bound's tolerance, is incompatible with the variable's
    /// integer or boolean domain, or the solver proves the edited problem
    /// infeasible (see [`Error::Infeasible`]). A `min <= -2^52` or
    /// `max >= 2^52` counts as infinite here too (see
    /// [`Problem::add_var`](crate::Problem::add_var)).
    /// The bound and domain checks allow the solve's tolerances, so any value
    /// this solution reports is accepted: `val` may lie outside a bound by up
    /// to [`Tolerances::feasibility`](crate::Tolerances::feasibility), or by
    /// the round-off floor at a large bound, and an integer or boolean
    /// variable's `val` may be within the integrality tolerance
    /// [`SolveOptions::int_tol`] of an integer without equaling it (one fixed
    /// at an integer lies on it).
    ///
    /// A real variable is fixed at `val` itself. An integer or boolean
    /// variable is fixed at the integer within `int_tol` of `val`, or of the
    /// bound if `val` lies outside it.
    ///
    /// A MILP edit starts a new branch & bound search. If a reported integer
    /// value is just beyond a bound or is not exactly integral, fixing the
    /// variable can leave the edited problem feasible only within the solve's
    /// tolerances, so the solver can answer `Infeasible` (see
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility)). The same
    /// can happen with any fix where this solution used the tolerances of
    /// several bounds and constraints at once: with the variable fixed exactly
    /// at that value, the others can be left less room than the infeasibility
    /// proof grants.
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

    /// Releases a fix created by [`Solution::fix_var`] and solves the edited
    /// problem, consuming this solution.
    ///
    /// If the variable is fixed, this method restores its original bounds and
    /// solves the edited problem using this solution's settings. It returns
    /// `true` when it releases a fix. If the variable was not fixed, it makes
    /// no edit and returns `false`.
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

    /// Indexes the solution by variable and returns its value, as
    /// [`Solution::var_value`] does.
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

/// Iterates over a [`Solution`]'s variables and values in creation order.
///
/// Each item is `(variable, value)`, with `value` from
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
