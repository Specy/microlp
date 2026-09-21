//! The model itself: [`Problem`], the variable domains it accepts, and the
//! entry points that hand it to the LP engine or the branch & bound driver.

use core::time::Duration;
use web_time::Instant;

use crate::solver::Solver;
use crate::{
    mip, presolve, ComparisonOp, CsVec, Error, LinearExpr, OptimizationDirection, ResumeOptions,
    SolveOptions, SolveOutcome, Variable,
};

/// A linear optimization model that can be populated and solved.
///
/// Add variables and constraints, then call [`Problem::solve`] or
/// [`Problem::solve_with`].
#[derive(Clone)]
pub struct Problem {
    pub(crate) direction: OptimizationDirection,
    pub(crate) obj_coeffs: Vec<f64>,
    pub(crate) var_mins: Vec<f64>,
    pub(crate) var_maxs: Vec<f64>,
    pub(crate) var_domains: Vec<VarDomain>,
    pub(crate) constraints: Vec<(CsVec, ComparisonOp, f64)>,
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
    /// If the budget expires, the outcome contains a feasible [`Solution`](crate::Solution) when
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

    fn internal_add_var(
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

/// Internal signal for whether a simplex operation finished or hit its deadline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StopReason {
    Limit,
    Finished,
}

pub(crate) fn timed_lp_call<T>(
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
            // Lp mode applies only feasible-set-exact reductions, so the live
            // solver stays sound under every later Solution edit.
            let presolved = if options.presolve {
                Some(presolve::presolve(
                    &self.obj_coeffs,
                    &self.var_mins,
                    &self.var_maxs,
                    &self.constraints,
                    &self.var_domains,
                    presolve::Mode::Lp,
                    options.tolerances.feasibility,
                    options.int_tol,
                    false,
                )?)
            } else {
                None
            };
            let (var_mins, var_maxs, constraints) = match &presolved {
                Some(p) => (&p.var_mins[..], &p.var_maxs[..], &p.constraints[..]),
                None => (
                    &self.var_mins[..],
                    &self.var_maxs[..],
                    &self.constraints[..],
                ),
            };
            let mut solver = Solver::try_new(
                &self.obj_coeffs,
                var_mins,
                var_maxs,
                constraints,
                &self.var_domains,
                deadline,
                options.tolerances.feasibility,
            )?;
            solver.operation_time_limit = options.time_limit;
            let stop = solver.initial_solve()?;
            solver.elapsed += started.elapsed();
            let resume_options = ResumeOptions::from(&options);
            let outcome = SolveOutcome::from_lp_stop(
                self.direction,
                num_vars,
                stop,
                Box::new(solver),
                resume_options,
            )?;
            // The engine validated its own (presolved) rows; the contract is
            // on the rows as written, so validate those too.
            if let SolveOutcome::Solution(solution) = &outcome {
                if let Some(violation) = mip::first_violation(
                    self,
                    &std::collections::BTreeMap::new(),
                    &solution.lp_values,
                    &options.tolerances,
                ) {
                    return Err(Error::InternalError(format!(
                        "LP solution violates the original problem: {violation}"
                    )));
                }
            }
            Ok(outcome)
        }
    }
}
