/*!
A fast linear programming solver library.

[Linear programming](https://en.wikipedia.org/wiki/Linear_programming) is a technique for
finding the minimum (or maximum) of a linear function of a set of continuous variables
subject to linear equality and inequality constraints.

# Features

* Pure Rust implementation.
* Able to solve problems with hundreds of thousands of variables and constraints.
* Incremental: add constraints to an existing solution without solving it from scratch.
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
let mut problem = problem;
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
    /// the solution will have [`StopReason::Limit`] as its stop reason.
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

    /// Solve the problem, finding the optimal objective function value and variable values.
    ///
    /// If a time limit was set and exceeded, the returned [`Solution`] will have
    /// [`StopReason::Limit`] as its stop reason, containing the best solution found so far.
    ///
    /// # Errors
    ///
    /// Will return an error if the problem is infeasible (constraints can't be satisfied)
    /// or if the objective value is unbounded.
    pub fn solve(&self) -> Result<Solution, Error> {
        let deadline = self.time_limit.map(|d| Instant::now() + d);

        let mut solver = Solver::try_new(
            &self.obj_coeffs,
            &self.var_mins,
            &self.var_maxs,
            &self.constraints,
            &self.var_domains,
            deadline,
        )?;
        let mut stop_reason = solver.initial_solve()?;

        if stop_reason == StopReason::Finished && self.has_integer_vars() {
            let non_integer_solution = Solution {
                num_vars: self.obj_coeffs.len(),
                direction: self.direction,
                solver: solver.clone(),
                stop_reason,
            };
            stop_reason = solver.solve_integer(non_integer_solution, self.direction)?;
            let solution = Solution {
                num_vars: self.obj_coeffs.len(),
                direction: self.direction,
                solver,
                stop_reason,
            };
            Ok(solution)
        } else {
            Ok(Solution {
                num_vars: self.obj_coeffs.len(),
                direction: self.direction,
                solver,
                stop_reason,
            })
        }
    }
}

/// The reason why the solver stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// The solver reached the time limit before finding an optimal solution.
    Limit,
    /// The solver finished normally.
    Finished,
}

/// A solution of a problem: optimal objective function value and variable values.
///
/// Note that a `Solution` instance contains the whole solver machinery which can require
/// a lot of memory for larger problems. Thus saving the `Solution` instance (as opposed
/// to getting the values of interest and discarding the solution) is mainly useful if you
/// want to add more constraints to it later.
#[derive(Clone)]
pub struct Solution {
    direction: OptimizationDirection,
    num_vars: usize,
    solver: solver::Solver,
    stop_reason: StopReason,
}

impl std::fmt::Debug for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Only printing lengths here because actual data is probably huge.
        f.debug_struct("Solution")
            .field("direction", &self.direction)
            .field("num_vars", &self.num_vars)
            .field("num_constraints", &self.solver.num_constraints())
            .field("objective", &self.objective())
            .finish()
    }
}

impl Solution {
    /// Returns the reason why the solver stopped.
    pub fn stop_reason(&self) -> &StopReason {
        &self.stop_reason
    }

    /// Optimal value of the objective function.
    pub fn objective(&self) -> f64 {
        match self.direction {
            OptimizationDirection::Minimize => self.solver.cur_obj_val,
            OptimizationDirection::Maximize => -self.solver.cur_obj_val,
        }
    }

    /// WARNING: It is recommended to create a new solver with a longer time limit instead of resuming it,
    /// as there might be subtle bugs in the stopping/resuming process that were not caught by tests.
    ///
    /// Resume solving after the solver stopped due to a time limit.
    ///
    /// If the solver previously stopped with [`StopReason::Limit`], this method continues
    /// solving with the given `time_limit`. Pass `None` to run without a time limit.
    ///
    /// If the solver already finished ([`StopReason::Finished`]), this is a no-op and
    /// returns the solution as-is.
    ///
    /// # Errors
    ///
    /// Will return an error if the problem is infeasible or unbounded.
    pub fn resume(mut self, time_limit: Option<Duration>) -> Result<Self, Error> {
        if self.stop_reason == StopReason::Finished {
            return Ok(self);
        }

        self.solver.deadline = time_limit.map(|d| Instant::now() + d);

        let has_bb_state = self.solver.bb_state.is_some();

        let stop_reason = if has_bb_state {
            // We are resuming a branch-and-bound search. The solver is already
            // at the best integer solution found so far (or the LP relaxation if
            // none was found yet). Skip initial_solve and go straight to B&B.
            //
            // Take bb_state out before cloning so we don't deep-clone the
            // entire DFS stack (which contains Solution/Solver snapshots for
            // every pending branch). solve_integer will .take() it from
            // self.solver anyway.
            let bb_state = self.solver.bb_state.take();
            let cur_solution = Solution {
                num_vars: self.num_vars,
                direction: self.direction,
                solver: self.solver.clone(),
                stop_reason: StopReason::Finished,
            };
            self.solver.bb_state = bb_state;
            self.solver.solve_integer(cur_solution, self.direction)?
        } else {
            // No persisted B&B state — resume the LP solve first.
            let mut sr = self.solver.initial_solve()?;

            if sr == StopReason::Finished && self.solver.has_integer_vars() {
                let cur_solution = Solution {
                    num_vars: self.num_vars,
                    direction: self.direction,
                    solver: self.solver.clone(),
                    stop_reason: StopReason::Finished,
                };
                sr = self.solver.solve_integer(cur_solution, self.direction)?;
            }
            sr
        };

        self.stop_reason = stop_reason;
        Ok(self)
    }

    /// Value of the variable at optimum.
    ///
    /// Note that you can use indexing operations to get variable values.
    /// # Warning
    /// If the variable is an integer, there might be rounding errors.
    /// For example you could see 0.999999999999 instead of 1.0.
    pub fn var_value_raw(&self, var: Variable) -> &f64 {
        assert!(var.0 < self.num_vars);
        self.solver.get_value(var.0)
    }

    /// Value of the variable at optimum.
    ///
    /// If the variable was defined as an integer or boolean, it rounds it.
    /// it removes precision errors
    pub fn var_value(&self, var: Variable) -> f64 {
        let val = self.var_value_raw(var);
        let domain = &self.solver.orig_var_domains[var.0];
        if *domain == VarDomain::Integer || *domain == VarDomain::Boolean {
            let rounded = val.round();
            assert!(
                f64::abs(rounded - val) < 1e-5,
                "Variable was expected to be an integer, got {}",
                val
            );
            rounded
        } else {
            *val
        }
    }

    /// Iterate over the variable-value pairs of the solution.
    ///
    /// # Warning
    /// If you used integer variables, there might be rounding errors in the variable results
    /// for example you could see 0.999999999999 instead of 1.0.
    pub fn iter(&self) -> SolutionIter<'_> {
        SolutionIter {
            solution: self,
            var_idx: 0,
        }
    }

    /// Add another constraint and return the solution to the updated problem.
    ///
    /// This method will consume the solution and not return it in case of error. See also
    /// examples of specifying the left-hand side in the docs for the [`Problem::add_constraint`]
    /// method.
    ///
    /// [`Problem::add_constraint`]: struct.Problem.html#method.add_constraint
    ///
    /// # Errors
    ///
    /// Will return an error if the problem becomes infeasible with the additional constraint.
    pub fn add_constraint(
        mut self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<Self, Error> {
        let expr = expr.into();
        let stop_reason = self.solver.add_constraint(
            CsVec::new_from_unsorted(self.num_vars, expr.vars, expr.coeffs)
                .map_err(|v| Error::InternalError(v.2.to_string()))?,
            cmp_op,
            rhs,
        )?;
        self.stop_reason = stop_reason;
        Ok(self)
    }

    /// Fix the variable to the specified value and return the solution to the updated problem.
    ///
    /// This method will consume the solution and not return it in case of error.
    ///
    /// # Errors
    ///
    /// Will return an error if the problem becomes infeasible with the additional constraint.
    pub fn fix_var(mut self, var: Variable, val: f64) -> Result<Self, Error> {
        assert!(var.0 < self.num_vars);
        let stop_reason = self.solver.fix_var(var.0, val)?;
        self.stop_reason = stop_reason;
        Ok(self)
    }

    /// If the variable was fixed with [`fix_var`](#method.fix_var) before, remove that constraint
    /// and return the solution to the updated problem and a boolean indicating if the variable was
    /// really fixed before.
    pub fn unfix_var(mut self, var: Variable) -> (Self, bool) {
        assert!(var.0 < self.num_vars);
        let res = self.solver.unfix_var(var.0);
        (self, res)
    }

    // TODO: remove_constraint

    /// Add a [Gomory cut] constraint to the problem and return the solution.
    ///
    /// [Gomory cut]: https://en.wikipedia.org/wiki/Cutting-plane_method#Gomory's_cut
    ///
    /// # Errors
    ///
    /// Will return an error if the problem becomes infeasible with the additional constraint.
    ///
    /// # Panics
    ///
    /// Will panic if the variable is not basic (variable is basic if it has value other than
    /// its bounds).
    pub fn add_gomory_cut(mut self, var: Variable) -> Result<Self, Error> {
        assert!(var.0 < self.num_vars);
        let stop_reason = self.solver.add_gomory_cut(var.0)?;
        self.stop_reason = stop_reason;
        Ok(self)
    }
}

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    fn index(&self, var: Variable) -> &Self::Output {
        self.var_value_raw(var)
    }
}

/// An iterator over the variable-value pairs of a [`Solution`].
#[derive(Debug, Clone)]
pub struct SolutionIter<'a> {
    solution: &'a Solution,
    var_idx: usize,
}

impl<'a> Iterator for SolutionIter<'a> {
    type Item = (Variable, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.var_idx < self.solution.num_vars {
            let var_idx = self.var_idx;
            self.var_idx += 1;
            Some((Variable(var_idx), self.solution.solver.get_value(var_idx)))
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Solution {
    type Item = (Variable, &'a f64);
    type IntoIter = SolutionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub use mps::MpsFile;
