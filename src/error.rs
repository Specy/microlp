//! The single error type every fallible call in the public API returns.

use sprs::errors::StructureError;

/// An error returned while validating, solving, or editing a problem.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// The solver proved, from the problem's data, that no assignment with
    /// every integer variable at an integer within its bounds, and every
    /// variable whose bounds are equal at its value, meets the constraints and
    /// the other bounds within the feasibility tolerance (see
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility)).
    Infeasible,
    /// The objective can improve without a finite limit.
    Unbounded,
    /// A solve or resume option is non-finite or outside its accepted range.
    ///
    /// The message identifies the invalid field.
    InvalidOptions(String),
    /// The requested operation cannot be applied to the current problem or
    /// outcome.
    InvalidOperation(String),
    /// The solve could not continue because of an unexpected numerical or
    /// structural failure. The solver returns this rather than an answer it
    /// could not verify.
    ///
    /// When the model lies outside the recommended coefficient range (see the
    /// crate documentation), the message names the constraint and the
    /// variable whose coefficients span the most, those of them outside it.
    /// Within the range, please report it as a bug.
    InternalError(String),
}
impl From<StructureError> for Error {
    fn from(err: StructureError) -> Self {
        Error::InternalError(err.to_string())
    }
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
