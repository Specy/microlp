//! Tests for the public solve API, grouped by topic.
//!
//! Every group drives the crate exactly as a user would — through [`Problem`],
//! [`SolveOutcome`] and [`Solution`] — so these are the tests that pin the
//! behaviour the API promises.

#[cfg(test)]
mod common;
#[cfg(test)]
mod edits;
#[cfg(test)]
mod lp;
#[cfg(test)]
mod options;
#[cfg(test)]
mod proofs;
#[cfg(test)]
mod solve;
#[cfg(test)]
mod warm_start;
