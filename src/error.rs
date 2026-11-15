//! The single error type every fallible call in the public API returns.

/// An error returned while validating, solving, or editing a problem.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The problem has no solution.
    ///
    /// No assignment satisfies the constraints and bounds within
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility) while each
    /// integer variable is at an integer within its bounds. Variables with
    /// equal bounds must be exactly at that value. Increasing
    /// [`SolveOptions::int_tol`](crate::SolveOptions::int_tol) does not make
    /// a model with no integer solution feasible.
    ///
    /// In extreme cases, the solver can report a feasible model as infeasible
    /// when it has nearly dependent constraints and its feasible points need
    /// very large variable values. See the crate's
    /// [known limitations](crate#known-limitations).
    Infeasible,
    /// The objective can improve without a finite limit.
    ///
    /// Outside the recommended coefficient range, a bounded model can be
    /// reported unbounded. See the crate's
    /// [known limitations](crate#known-limitations).
    Unbounded,
    /// A solve or resume option is non-finite or outside its accepted range.
    ///
    /// The message identifies the invalid field.
    InvalidOptions(String),
    /// The input is invalid, so the requested solve or edit did not run. This
    /// can mean a model number is outside its accepted range. Every objective
    /// coefficient, constraint coefficient, bound, and right-hand side must
    /// be finite, except that a bound or right-hand side may be infinite on
    /// the side it leaves open (see
    /// [`Problem::solve_with`](crate::Problem::solve_with)). It can also mean
    /// a warm-start hint names a variable outside the problem or gives it a
    /// NaN or infinite value, a constraint added to a solution repeats a
    /// variable or contains a variable index outside the problem, or
    /// [`Solution::fix_var`](crate::Solution::fix_var) receives a NaN or
    /// infinite value.
    ///
    /// The message identifies the invalid input. For an invalid number, it
    /// gives the value as provided and identifies the variable by index or
    /// the constraint by its order of addition.
    InvalidOperation(String),
    /// An unexpected numerical or structural failure stopped the solve. The
    /// solver returns this error instead of a solution it could not verify.
    ///
    /// If the model is outside the recommended coefficient range (see the
    /// crate documentation), the message names the constraint (in the order
    /// added) and the variable (by index) with the widest coefficient ranges,
    /// when those ranges exceed the recommendation. Within the range, it is
    /// a bug or one of the crate's
    /// [known limitations](crate#known-limitations). Please report it.
    InternalError(String),
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
