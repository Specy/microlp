//! The vocabulary a model is written in: the optimization direction, variable
//! handles, linear expressions, and the comparison operators constraints use.

/// Selects whether a problem's objective is minimized or maximized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizationDirection {
    /// Minimize the objective value.
    Minimize,
    /// Maximize the objective value.
    Maximize,
}

/// Identifies a variable created by a [`Problem`](crate::Problem).
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

/// Identifies a constraint created by
/// [`Problem::add_constraint`](crate::Problem::add_constraint), to read its
/// dual with [`Solution::dual`](crate::Solution::dual).
///
/// A constraint should only be used with the problem that created it and with
/// solutions obtained from that problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Constraint(pub(crate) usize);

impl Constraint {
    /// Returns the constraint's zero-based creation order.
    ///
    /// The first constraint added to a problem has index `0`. A constraint
    /// added by [`Solution::add_constraint`](crate::Solution::add_constraint)
    /// is numbered after the problem's and those added before it, which is
    /// its index in [`Solution::duals`](crate::Solution::duals).
    pub fn idx(&self) -> usize {
        self.0
    }
}

/// A weighted sum of variables used on the left-hand side of a constraint.
#[derive(Clone, Debug)]
pub struct LinearExpr {
    pub(crate) vars: Vec<usize>,
    pub(crate) coeffs: Vec<f64>,
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
    /// [`Problem::add_constraint`](crate::Problem::add_constraint) will panic;
    /// [`Solution::add_constraint`](crate::Solution::add_constraint) returns
    /// [`Error::InvalidOperation`](crate::Error::InvalidOperation) for it.
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
