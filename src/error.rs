//! The single error type every fallible call in the public API returns.

use sprs::errors::StructureError;

/// An error returned while validating, solving, or editing a problem.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// No assignment satisfies all variable bounds and constraints.
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
    /// structural failure.
    InternalError(String),
}
impl From<StructureError> for Error {
    fn from(err: StructureError) -> Self {
        Error::InternalError(err.to_string())
    }
}

impl From<crate::sparse::Error> for Error {
    fn from(value: crate::sparse::Error) -> Self {
        Error::InternalError(value.to_string())
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
