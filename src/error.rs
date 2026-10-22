//! The single error type every fallible call in the public API returns.

/// An error returned while validating, solving, or editing a problem.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The solver proved, from the problem's data, that no assignment with
    /// every integer variable at an integer within its bounds, and every
    /// variable whose bounds are equal at its value, meets the constraints and
    /// the other bounds within the feasibility tolerance (see
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility)).
    ///
    /// In extreme cases, a feasible model of nearly dependent constraints,
    /// whose feasible points need very large variable values, is reported
    /// infeasible: see the crate's [known limitations](crate#known-limitations).
    Infeasible,
    /// The objective can improve without a finite limit.
    ///
    /// Outside the recommended coefficient range, a bounded model can be
    /// reported unbounded: see the crate's
    /// [known limitations](crate#known-limitations).
    Unbounded,
    /// A solve or resume option is non-finite or outside its accepted range.
    ///
    /// The message identifies the invalid field.
    InvalidOptions(String),
    /// The input is invalid, and nothing was solved: a model number outside
    /// its accepted range (every objective coefficient, constraint
    /// coefficient, bound and right-hand side must be finite, but a bound or a
    /// right-hand side may be infinite on the side it leaves open; see
    /// [`Problem::solve_with`](crate::Problem::solve_with)), a warm-start hint
    /// that names a variable outside the problem or gives a NaN or infinite
    /// value, a constraint added to a solution that repeats a variable or
    /// contains a variable index outside the problem, or a NaN or infinite
    /// value given to [`Solution::fix_var`](crate::Solution::fix_var).
    ///
    /// The message names what is invalid: the value as given, and the
    /// variable (its index) or the constraint (in the order added) it belongs
    /// to.
    InvalidOperation(String),
    /// The solve could not continue because of an unexpected numerical or
    /// structural failure. The solver returns this rather than an answer it
    /// could not verify.
    ///
    /// When the model lies outside the recommended coefficient range (see the
    /// crate documentation), the message names the constraint and the
    /// variable whose coefficients span the most, those of them outside it.
    /// Within the range, it is a bug or one of the crate's
    /// [known limitations](crate#known-limitations): please report it.
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
