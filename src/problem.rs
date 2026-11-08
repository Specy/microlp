//! The model itself: [`Problem`], the variable domains it accepts, and the
//! entry points that hand it to the LP engine or the branch & bound driver.

use core::time::Duration;
use web_time::Instant;

use crate::range::CoefficientRange;
use crate::solver::{deadline_after, valid_rhs, Solver};
use crate::{
    mip, presolve, ComparisonOp, Constraint, CsVec, Error, LinearExpr, OptimizationDirection,
    ResumeOptions, SolveOptions, SolveOutcome, Variable,
};

/// A linear optimization problem that you can populate and solve.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A `VarDomain` says whether a variable is real, integer or boolean.
/// [`Problem::var_domain`] and
/// [`Solution::var_domain`](crate::Solution::var_domain) report this domain.
pub enum VarDomain {
    /// An integer variable can take any integer within its bounds.
    Integer,
    /// A real variable can take any real value within its bounds.
    Real,
    /// A boolean variable can take `0` or `1`.
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

    /// Sets the time limit used by [`Problem::solve`].
    ///
    /// If the time limit expires, the outcome contains a feasible
    /// [`Solution`](crate::Solution) when one is available, or
    /// [`SolveOutcome::Interrupted`] when no usable assignment has been found.
    /// Resume the outcome with [`SolveOutcome::resume`] to continue searching.
    /// A duration too long to add to the current time, such as
    /// [`Duration::MAX`], counts as no limit.
    ///
    /// [`Problem::solve_with`] uses the `time_limit` in its supplied
    /// [`SolveOptions`] instead of this setting.
    pub fn set_time_limit(&mut self, duration: Duration) {
        self.time_limit = Some(duration);
    }

    /// Adds a real variable to the problem.
    ///
    /// `obj_coeff` is the variable's objective coefficient. If it is NaN or
    /// infinite, the solve returns [`Error::InvalidOperation`]. `min` and
    /// `max` are inclusive bounds. Use [`f64::NEG_INFINITY`] or
    /// [`f64::INFINITY`] for an unbounded side. A NaN bound, a `min` of
    /// [`f64::INFINITY`], or a `max` of [`f64::NEG_INFINITY`] also makes the
    /// solve return [`Error::InvalidOperation`]. A `min <= -2^52` or a
    /// `max >= 2^52` (2^52 is about 4.5e15) is treated as infinite. So
    /// `f64::MAX` works like [`f64::INFINITY`], and a solution may lie beyond
    /// such a bound. Each bound is judged on its own: `(5e15, f64::MAX)` means
    /// `x >= 5e15`. When `min >= max`, the bounds are kept as given: a variable
    /// fixed at a huge value stays fixed, and an empty range stays infeasible.
    pub fn add_var(&mut self, obj_coeff: f64, (min, max): (f64, f64)) -> Variable {
        self.internal_add_var(obj_coeff, (min, max), VarDomain::Real)
    }

    /// Adds an integer variable to the problem.
    ///
    /// `obj_coeff` is the variable's objective coefficient. If it is NaN or
    /// infinite, the solve returns [`Error::InvalidOperation`]. `min` and
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

    /// Returns whether `var` is a real, integer or boolean variable, as it was
    /// added to the problem.
    /// [`Solution::var_domain`](crate::Solution::var_domain) reports the same
    /// domain. For an integer or boolean variable, a solution value lies within
    /// [`SolveOptions::int_tol`] of an integer, but may not equal it. Rounding
    /// values can make a solution violate constraints. For example, with
    /// `x <= 1e7 * y`, a `y` of `1e-6` allows `x` up to 10, but rounding `y` to
    /// zero forbids that value of `x`.
    ///
    /// # Panics
    ///
    /// Panics if `var` is out of range for this problem.
    pub fn var_domain(&self, var: Variable) -> VarDomain {
        assert!(var.0 < self.var_domains.len());
        self.var_domains[var.0]
    }

    /// Adds a boolean variable restricted to `0` or `1`.
    ///
    /// `obj_coeff` is the variable's objective coefficient. If it is NaN or
    /// infinite, the solve returns [`Error::InvalidOperation`].
    pub fn add_binary_var(&mut self, obj_coeff: f64) -> Variable {
        self.internal_add_var(obj_coeff, (0.0, 1.0), VarDomain::Boolean)
    }

    /// Records a var, storing its bounds as the solve takes them
    /// ([`crate::solver::entry_bounds`]), so that every later reader (presolve,
    /// the engine, the MIP and its post-edit incumbent check,
    /// `Solution::fix_var`) sees the same bounds, not the raw input.
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
        let (min, max) = crate::solver::entry_bounds(min, max);
        self.obj_coeffs.push(obj_coeff);
        self.var_mins.push(min);
        self.var_maxs.push(max);
        self.var_domains.push(var_type);
        var
    }

    /// Adds the linear constraint `expr cmp_op rhs` to the problem and returns
    /// its handle. For a problem without integer or boolean variables,
    /// [`Solution::dual`](crate::Solution::dual) uses it to report the
    /// constraint's dual value.
    ///
    /// `expr` may be a [`LinearExpr`] or any supported collection or iterator
    /// of `(Variable, coefficient)` pairs. Use variables created by this
    /// problem, with each variable appearing at most once. Keep the magnitudes
    /// of the nonzero coefficients in each constraint, and of the nonzero
    /// coefficients for each variable across constraints, within six orders
    /// of magnitude. This is the range recommended in the crate documentation.
    ///
    /// A NaN or infinite coefficient, or a NaN `rhs`, makes the solve return
    /// [`Error::InvalidOperation`]. An infinite `rhs` also makes the solve
    /// return [`Error::InvalidOperation`] if it is [`f64::NEG_INFINITY`] with
    /// [`ComparisonOp::Le`], [`f64::INFINITY`] with [`ComparisonOp::Ge`], or
    /// either infinity with [`ComparisonOp::Eq`]. The opposite infinity with
    /// [`ComparisonOp::Le`] or [`ComparisonOp::Ge`] imposes no restriction.
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
    pub fn add_constraint(
        &mut self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Constraint {
        let row = self.constraints.len();
        let coeffs = row_of(expr.into(), self.obj_coeffs.len(), row)
            .unwrap_or_else(|message| panic!("{message}"));
        self.constraints.push((coeffs, cmp_op, rhs));
        Constraint(row)
    }
}

/// The coefficients `expr` gives constraint `row` (in the order added) of a
/// problem of `num_vars` variables, or what makes the expression malformed:
/// a variable index outside the problem, or a variable it repeats.
pub(crate) fn row_of(expr: LinearExpr, num_vars: usize, row: usize) -> Result<CsVec, String> {
    if let Some(var) = expr.vars.iter().find(|&&var| var >= num_vars) {
        return Err(format!(
            "invalid constraint {row}: variable index {var} is out of range 0..{num_vars}"
        ));
    }
    CsVec::new_from_unsorted(num_vars, expr.vars, expr.coeffs).map_err(|(mut vars, _, _)| {
        vars.sort_unstable();
        let repeated = vars
            .windows(2)
            .find(|pair| pair[0] == pair[1])
            .expect("an expression of distinct variables in range is a valid row")[0];
        format!("invalid constraint {row}: variable {repeated} appears more than once")
    })
}

/// `InvalidOperation` for the first number of constraint `row` (in the order
/// added) no solve can take: a coefficient, by variable index, that is NaN or
/// infinite, then a right-hand side that is NaN or infinite on the side the
/// constraint bounds ([`valid_rhs`]).
pub(crate) fn check_row(
    coeffs: &CsVec,
    cmp_op: ComparisonOp,
    rhs: f64,
    row: usize,
) -> Result<(), Error> {
    if let Some((var, coeff)) = coeffs.iter().find(|(_, coeff)| !coeff.is_finite()) {
        return Err(Error::InvalidOperation(format!(
            "invalid coefficient {coeff} of variable {var} in constraint {row}: expected a finite \
             value"
        )));
    }
    check_rhs(cmp_op, rhs, row)
}

/// `InvalidOperation` for a right-hand side of constraint `row` (in the order
/// added) no solve can take: NaN, or infinite on the side the constraint
/// bounds ([`valid_rhs`]).
pub(crate) fn check_rhs(cmp_op: ComparisonOp, rhs: f64, row: usize) -> Result<(), Error> {
    if !valid_rhs(cmp_op, rhs) {
        let expected = match cmp_op {
            ComparisonOp::Le => "a finite value or inf",
            ComparisonOp::Ge => "a finite value or -inf",
            ComparisonOp::Eq => "a finite value",
        };
        return Err(Error::InvalidOperation(format!(
            "invalid right-hand side {rhs} of constraint {row} ({cmp_op:?}): expected {expected}"
        )));
    }
    Ok(())
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
    solver.deadline = deadline_after(started, time_limit);
    let result = call(solver);
    solver.elapsed += started.elapsed();
    result
}

impl Problem {
    /// Solves the problem with the default options.
    ///
    /// A time limit configured with [`Problem::set_time_limit`] is applied to
    /// this call.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Infeasible`] when the solver proves the problem
    /// infeasible (see [`Error::Infeasible`]), [`Error::Unbounded`] when the
    /// objective has no finite optimum, [`Error::InvalidOperation`] when the
    /// problem holds a number no solve can take (see [`Problem::solve_with`]),
    /// or [`Error::InternalError`] when the solve cannot reach an answer it can
    /// verify.
    pub fn solve(&self) -> Result<SolveOutcome, Error> {
        let options = SolveOptions {
            time_limit: self.time_limit,
            ..SolveOptions::default()
        };
        self.solve_with(options)
    }

    /// Solves the problem with the supplied [`SolveOptions`].
    ///
    /// These options apply to this call and override any previously set
    /// options.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOptions`] when an option is non-finite or outside
    /// its accepted range. Returns [`Error::InvalidOperation`] for the first
    /// value of the problem that no solve can take: an objective or constraint
    /// coefficient that is NaN or infinite, a bound or right-hand side that is
    /// NaN, a lower bound of [`f64::INFINITY`], an upper bound of
    /// [`f64::NEG_INFINITY`], or a right-hand side infinite on the side its
    /// constraint bounds (see [`Problem::add_constraint`]); then for a
    /// [`SolveOptions::warm_start`] entry that names a variable outside the
    /// problem or gives it a NaN or infinite value. All of these are checked
    /// before anything is solved. The error message gives the value as written
    /// and its variable (by index) or constraint (in the order added). See
    /// [`Problem::solve`] for errors that can be reported while solving.
    pub fn solve_with(&self, options: SolveOptions) -> Result<SolveOutcome, Error> {
        options.validate()?;
        self.validate()?;
        if let Some(hint) = &options.warm_start {
            self.validate_hint(hint)?;
        }
        let range = CoefficientRange::of(&self.constraints);
        self.solve_validated(options, &range)
            .map_err(|error| range.annotate(error))
    }

    /// [`Problem::solve_with`] once the options and the model are validated;
    /// `range` is this problem's ([`CoefficientRange`]).
    fn solve_validated(
        &self,
        options: SolveOptions,
        range: &CoefficientRange,
    ) -> Result<SolveOutcome, Error> {
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
            let deadline = deadline_after(started, options.time_limit);
            // Lp mode applies only feasible-set-exact reductions, so the live
            // solver stays sound under every later Solution edit.
            let presolved = if options.presolve {
                presolve::reduced_model(presolve::presolve(
                    &self.obj_coeffs,
                    &self.var_mins,
                    &self.var_maxs,
                    &self.constraints,
                    &self.var_domains,
                    presolve::Mode::Lp,
                    options.tolerances.feasibility,
                    options.int_tol,
                    false,
                ))?
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
            // The engine also holds the answer to the rows presolve dropped or
            // rewrote, as written, here and on every later edit or resume.
            if let Some(presolved) = presolved {
                solver.set_presolved_rows(
                    &presolved.kept_rows,
                    presolved.validation_rows,
                    self.constraints.len(),
                );
            }
            solver.range = range.clone();
            solver.operation_time_limit = options.time_limit;
            let stop = solver.initial_solve()?;
            solver.elapsed += started.elapsed();
            let resume_options = ResumeOptions::from(&options);
            SolveOutcome::from_lp_stop(
                self.direction,
                num_vars,
                stop,
                Box::new(solver),
                resume_options,
            )
        }
    }

    /// `InvalidOperation` for the first number of the model no solve can
    /// take, variable by variable (its objective coefficient, then its
    /// bounds), then constraint by constraint ([`check_row`]): every number
    /// must be finite, but a bound or a right-hand side may be infinite on the
    /// side it leaves open. A value is reported as the user gave it: a
    /// `Maximize` cost is stored negated, and the bounds
    /// [`entry_bounds`](crate::solver::entry_bounds) stores differ from the
    /// given ones only by making a valid bound infinite on its open side.
    fn validate(&self) -> Result<(), Error> {
        for var in 0..self.obj_coeffs.len() {
            let given = match self.direction {
                OptimizationDirection::Minimize => self.obj_coeffs[var],
                OptimizationDirection::Maximize => -self.obj_coeffs[var],
            };
            if !given.is_finite() {
                return Err(Error::InvalidOperation(format!(
                    "invalid objective coefficient {given} of variable {var}: expected a finite \
                     value"
                )));
            }
            let (min, max) = (self.var_mins[var], self.var_maxs[var]);
            if min.is_nan() || min == f64::INFINITY {
                return Err(Error::InvalidOperation(format!(
                    "invalid lower bound {min} of variable {var}: expected a finite value or -inf"
                )));
            }
            if max.is_nan() || max == f64::NEG_INFINITY {
                return Err(Error::InvalidOperation(format!(
                    "invalid upper bound {max} of variable {var}: expected a finite value or inf"
                )));
            }
        }
        for (row, (coeffs, cmp_op, rhs)) in self.constraints.iter().enumerate() {
            check_row(coeffs, *cmp_op, *rhs, row)?;
        }
        Ok(())
    }

    /// `InvalidOperation` for the first entry of a warm-start hint no solve
    /// can take: a variable outside the problem, or a value that is NaN or
    /// infinite. Any finite value is a hint, which the search ignores where
    /// it cannot use it.
    fn validate_hint(&self, hint: &[(Variable, f64)]) -> Result<(), Error> {
        let num_vars = self.obj_coeffs.len();
        for &(var, value) in hint {
            if var.0 >= num_vars {
                return Err(Error::InvalidOperation(format!(
                    "invalid warm-start hint: variable index {} is out of range 0..{num_vars}",
                    var.0
                )));
            }
            if !value.is_finite() {
                return Err(Error::InvalidOperation(format!(
                    "invalid warm-start value {value} of variable {}: expected a finite value",
                    var.0
                )));
            }
        }
        Ok(())
    }
}
