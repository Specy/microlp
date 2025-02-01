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
let solution = problem.solve().unwrap();
assert_eq!(solution.objective(), 7.0);
assert_eq!(solution[x], 1.0);
assert_eq!(solution[y], 3.0);
```
*/

#![deny(missing_debug_implementations, missing_docs)]

#[macro_use]
extern crate log;

mod broken_tests;
mod helpers;
mod lu;
mod mps;
mod ordering;
mod solver;
mod sparse;

use solver::Solver;
use sprs::errors::StructureError;

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
        }
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
    /// variable. If one of the bounds is absent, use `f64::NEG_INFINITY` for minimum and
    /// `f64::INFINITY` for maximum.
    pub fn add_integer_var(&mut self, obj_coeff: f64, (min, max): (i32, i32)) -> Variable {
        self.internal_add_var(obj_coeff, (min as f64, max as f64), VarDomain::Integer)
    }

    /// Check if the problem has any integer variables.
    pub fn has_integer_vars(&self) -> bool {
        self.var_domains.iter().any(|v| *v == VarDomain::Integer)
    }

    /// Add a new binary variable to the problem.
    ///
    /// `obj_coeff` is a coefficient of the term in the objective function corresponding to this variable.
    pub fn add_binary_var(&mut self, obj_coeff: f64) -> Variable {
        self.internal_add_var(obj_coeff, (0.0, 1.0), VarDomain::Integer)
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
    /// # Errors
    ///
    /// Will return an error, if the problem is infeasible (constraints can't be satisfied)
    /// or if the objective value is unbounded.
    pub fn solve(&self) -> Result<Solution, Error> {
        let mut solver = Solver::try_new(
            &self.obj_coeffs,
            &self.var_mins,
            &self.var_maxs,
            &self.constraints,
            &self.var_domains,
        )?;
        solver.initial_solve()?;

        if self.has_integer_vars() {
            let non_integer_solution = Solution {
                num_vars: self.obj_coeffs.len(),
                direction: self.direction,
                solver: solver.clone(),
            };
            solver.solve_integer(non_integer_solution, self.direction)?;
            let solution = Solution {
                num_vars: self.obj_coeffs.len(),
                direction: self.direction,
                solver,
            };
            Ok(solution)
        } else {
            Ok(Solution {
                num_vars: self.obj_coeffs.len(),
                direction: self.direction,
                solver,
            })
        }
    }
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
    /// Optimal value of the objective function.
    pub fn objective(&self) -> f64 {
        match self.direction {
            OptimizationDirection::Minimize => self.solver.cur_obj_val,
            OptimizationDirection::Maximize => -self.solver.cur_obj_val,
        }
    }

    /// Value of the variable at optimum.
    ///
    /// Note that you can use indexing operations to get variable values.
    /// # Warning
    /// If the variable is an integer, there might be rounding errors.
    /// For example you could see 0.999999999999 instead of 1.0.
    pub fn var_value(&self, var: Variable) -> &f64 {
        assert!(var.0 < self.num_vars);
        self.solver.get_value(var.0)
    }

    /// Value of the variable at optimum.
    ///
    /// If the variable was defined as an integer, it rounds
    /// it remove precision errors
    pub fn var_value_rounded(&self, var: Variable) -> f64 {
        let val = self.var_value(var);
        if self.solver.orig_var_domains[var.0] == VarDomain::Integer {
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
    pub fn iter(&self) -> SolutionIter {
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
        self.solver.add_constraint(
            CsVec::new_from_unsorted(self.num_vars, expr.vars, expr.coeffs)
                .map_err(|v| Error::InternalError(v.2.to_string()))?,
            cmp_op,
            rhs,
        )?;
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
        self.solver.fix_var(var.0, val)?;
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
        self.solver.add_gomory_cut(var.0)?;
        Ok(self)
    }
}

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    fn index(&self, var: Variable) -> &Self::Output {
        self.var_value(var)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn optimize() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(3.0, (12.0, f64::INFINITY));
        let v2 = problem.add_var(4.0, (5.0, f64::INFINITY));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 20.0);
        problem.add_constraint([(v2, -4.0), (v1, 1.0)], ComparisonOp::Ge, -20.0);

        let sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 12.0);
        assert_eq!(sol[v2], 8.0);
        assert_eq!(sol.objective(), 68.0);
    }

    #[test]
    fn empty_expr_constraints() {
        init();
        let trivial = [
            (LinearExpr::empty(), ComparisonOp::Eq, 0.0),
            (LinearExpr::empty(), ComparisonOp::Ge, -1.0),
            (LinearExpr::empty(), ComparisonOp::Le, 1.0),
        ];

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let _ = problem.add_var(1.0, (0.0, f64::INFINITY));
        for (expr, op, b) in trivial.iter().cloned() {
            problem.add_constraint(expr, op, b);
        }
        assert_eq!(problem.solve().map(|s| s.objective()), Ok(0.0));

        {
            let mut sol = problem.solve().unwrap();
            for (expr, op, b) in trivial.iter().cloned() {
                sol = sol.add_constraint(expr, op, b).unwrap();
            }
            assert_eq!(sol.objective(), 0.0);
        }

        let infeasible = [
            (LinearExpr::empty(), ComparisonOp::Eq, 12.0),
            (LinearExpr::empty(), ComparisonOp::Ge, 34.0),
            (LinearExpr::empty(), ComparisonOp::Le, -56.0),
        ];

        for (expr, op, b) in infeasible.iter().cloned() {
            let mut cloned = problem.clone();
            cloned.add_constraint(expr, op, b);
            assert_eq!(cloned.solve().map(|_| "solved"), Err(Error::Infeasible));
        }

        for (expr, op, b) in infeasible.iter().cloned() {
            let sol = problem.solve().unwrap().add_constraint(expr, op, b);
            assert_eq!(sol.map(|_| "solved"), Err(Error::Infeasible));
        }

        let _ = problem.add_var(-1.0, (0.0, f64::INFINITY));
        assert_eq!(problem.solve().map(|_| "solved"), Err(Error::Unbounded));
    }

    #[test]
    fn free_variables() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(1.0, (0.0, f64::INFINITY));
        let v2 = problem.add_var(2.0, (f64::NEG_INFINITY, f64::INFINITY));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 2.0);
        problem.add_constraint([(v1, 1.0), (v2, -1.0)], ComparisonOp::Ge, 0.0);

        let sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 2.0);
        assert_eq!(sol[v2], 2.0);
        assert_eq!(sol.objective(), 6.0);
    }

    #[test]
    fn fix_unfix_var() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(1.0, (0.0, 3.0));
        let v2 = problem.add_var(2.0, (0.0, 3.0));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 1.0);

        let orig_sol = problem.solve().unwrap();

        {
            let mut sol = orig_sol.clone().fix_var(v1, 0.5).unwrap();
            assert_eq!(sol[v1], 0.5);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 6.5);

            sol = sol.unfix_var(v1).0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }

        {
            let mut sol = orig_sol.clone().fix_var(v2, 2.5).unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 2.5);
            assert_eq!(sol.objective(), 6.5);

            sol = sol.unfix_var(v2).0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }
    }

    #[test]
    fn add_constraint() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let v1 = problem.add_var(2.0, (0.0, f64::INFINITY));
        let v2 = problem.add_var(1.0, (0.0, f64::INFINITY));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 2.0);

        let orig_sol = problem.solve().unwrap();

        {
            let sol = orig_sol
                .clone()
                .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                .unwrap();

            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 1.0);
            assert_eq!(sol.objective(), 3.0);
        }

        {
            let sol = orig_sol
                .clone()
                .fix_var(v2, 1.5)
                .unwrap()
                .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                .unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 1.5);
            assert_eq!(sol.objective(), 4.5);
        }

        {
            let sol = orig_sol
                .clone()
                .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Ge, 3.0)
                .unwrap();

            assert_eq!(sol[v1], 0.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 3.0);
        }
    }

    #[test]
    fn gomory_cut() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let v1 = problem.add_var(0.0, (0.0, f64::INFINITY));
        let v2 = problem.add_var(-1.0, (0.0, f64::INFINITY));
        problem.add_constraint([(v1, 3.0), (v2, 2.0)], ComparisonOp::Le, 6.0);
        problem.add_constraint([(v1, -3.0), (v2, 2.0)], ComparisonOp::Le, 0.0);

        let mut sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 1.0);
        assert_eq!(sol[v2], 1.5);
        assert_eq!(sol.objective(), -1.5);

        sol = sol.add_gomory_cut(v2).unwrap();
        assert!(f64::abs(sol[v1] - 2.0 / 3.0) < 1e-8);
        assert_eq!(sol[v2], 1.0);
        assert_eq!(sol.objective(), -1.0);

        sol = sol.add_gomory_cut(v1).unwrap();
        assert!(f64::abs(sol[v1] - 1.0) < 1e-8);
        assert_eq!(sol[v2], 1.0);
        assert_eq!(sol.objective(), -1.0);
    }

    fn cast_result_to_integers(vec: Vec<f64>) -> Vec<i64> {
        vec.into_iter()
            .map(|x| {
                let val = x.round() as i64;
                assert!(f64::abs(x - val as f64) < 1e-5);
                val
            })
            .collect()
    }

    #[test]
    fn knapsack_solve() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let weights = [10, 60, 30, 40, 30, 20, 20, 2];
        let values = [1, 10, 15, 40, 60, 90, 100, 15];
        let capacity = 102;
        let mut vars = vec![];
        for i in 0..weights.len() {
            let var = problem.add_binary_var(values[i] as f64);
            vars.push(var);
        }
        let entries = vars
            .iter()
            .map(|v| (*v, weights[v.0] as f64))
            .collect::<Vec<_>>();
        problem.add_constraint(&entries, ComparisonOp::Le, capacity as f64);
        let sol = problem.solve().unwrap();

        let values = vars
            .iter()
            .map(|v| sol.var_value_rounded(*v))
            .collect::<Vec<_>>();
        assert_eq!(
            cast_result_to_integers(values),
            vec![0, 0, 1, 0, 1, 1, 1, 1]
        );
        assert_eq!(sol.objective(), 280.0);
    }

    #[test]
    fn dominating_set_solve() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let vars = [
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
        ];
        let rows = vec![
            vec![1, 1, 0, 1, 1, 0],
            vec![1, 1, 1, 1, 0, 0],
            vec![0, 1, 1, 1, 0, 0],
            vec![1, 1, 1, 1, 0, 0],
            vec![1, 0, 0, 0, 1, 0],
            vec![1, 0, 0, 0, 0, 1],
        ];
        for row in rows {
            problem.add_constraint(
                row.iter()
                    .enumerate()
                    .map(|(i, v)| (vars[i], *v as f64))
                    .collect::<Vec<_>>(),
                ComparisonOp::Ge,
                1.0,
            );
        }
        let sol = problem.solve().unwrap();
        let values = vars
            .iter()
            .map(|v| sol.var_value_rounded(*v))
            .collect::<Vec<_>>();
        assert_eq!(cast_result_to_integers(values), vec![1, 0, 1, 0, 0, 0]);
        assert_eq!(sol.objective(), 2.0);
    }

    #[test]
    fn solve_milp() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);

        // Define variables with their objective coefficients
        let x = problem.add_var(50.0, (2.0, f64::INFINITY)); // x ≥ 0
        let y = problem.add_var(40.0, (0.0, 7.0)); // y ≥ 0
        let z = problem.add_integer_var(45.0, (0, i32::MAX)); // z ≥ 0 and integer
                                                              // Machine time constraint: 3x + 2y + z ≤ 20
        problem.add_constraint(&[(x, 3.0), (y, 2.0), (z, 1.0)], ComparisonOp::Le, 20.0);

        // Labor time constraint: 2x + y + 3z ≤ 15
        problem.add_constraint(&[(x, 2.0), (y, 1.0), (z, 3.0)], ComparisonOp::Le, 15.0);

        let sol = problem.solve().unwrap();

        assert_eq!(
            [
                sol.var_value_rounded(x),
                sol.var_value_rounded(y),
                sol.var_value_rounded(z)
            ],
            [2.0, 6.5, 1.0]
        );
        assert_eq!(sol.objective(), 405.0);
    }

    #[test]
    fn solve_big_m() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);

        let m = 1.0e9;

        // Define variables with their objective coefficients
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        let b = problem.add_binary_var(-1.0);

        problem.add_constraint([(x, 1.0)], ComparisonOp::Ge, 5.0);
        problem.add_constraint([(b, -m), (x, 1.0)], ComparisonOp::Le, 10.0);
        problem.add_constraint([(b, -m), (x, 1.0)], ComparisonOp::Ge, -m + 10.0);

        let sol = problem.solve().unwrap();

        assert_eq!([*sol.var_value(x), *sol.var_value(b)], [5.0, 0.0]);
        assert_eq!(sol.objective().round(), 5.0);
    }
}
