//! The knobs a MILP solve is configured with and the facts it reports back.
//!
//! These are the public vocabulary of a search: what it was asked to do
//! ([`SolveOptions`], [`ResumeOptions`], [`Tolerances`]), how it ended
//! ([`SolutionStatus`], [`TerminationReason`]) and what it cost ([`Stats`]).

use crate::{Error, Variable};
use core::time::Duration;

/// Whether a usable solution is proven optimal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionStatus {
    /// The solver completed the optimality proof.
    Optimal,
    /// A valid incumbent is available, but exact optimality was not proven.
    Feasible,
}

/// Why a solve or resume call returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TerminationReason {
    /// The LP or branch-and-bound optimality proof completed.
    ProvenOptimal,
    /// The configured relative MIP gap was reached before exact proof completed.
    MipGap,
    /// The wall-clock budget for this call was exhausted.
    TimeLimit,
    /// The branch-and-bound node budget for this call was exhausted.
    NodeLimit,
}

/// Options controlling a solve. Construct with [`SolveOptions::default`] and
/// mutate the fields you need.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolveOptions {
    /// Wall-clock budget for this call (`None` = unlimited). On expiry the search
    /// and can be resumed.
    pub time_limit: Option<Duration>,
    /// Maximum number of branch & bound nodes to solve in this call
    /// (`None` = unlimited). Deterministic alternative to `time_limit`; the
    /// budget applies per call, so [`crate::SolveOutcome::resume`] reapplies it
    /// as a fresh budget. The root relaxation does not count as a node.
    pub node_limit: Option<u64>,
    /// Relative MIP gap at which the search may stop with a feasible incumbent.
    /// Such a stop reports [`SolutionStatus::Feasible`] and
    /// [`TerminationReason::MipGap`]. Must be finite and non-negative. Default
    /// `0.0` (prove exact optimality).
    pub mip_gap: f64,
    /// Integrality tolerance: a value within this distance of an integer counts
    /// as integral. Default `1e-6`. A very loose `int_tol` mainly causes extra
    /// exact-fixing branching rather than admitting an infeasible point.
    /// Must be finite and in the half-open range `[0, 0.5)`.
    pub int_tol: f64,
    /// Optional (partial) starting assignment used to seed the incumbent.
    /// An infeasible or incomplete hint is ignored. Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
    /// Enable presolve: reductions applied to the problem before the search
    /// starts (bound tightening, redundant-row elimination, variable fixing;
    /// for integer problems also coefficient tightening and dual fixing).
    /// The problem you observe through [`crate::Solution`] is unchanged —
    /// reductions never remove variables and preserve at least one optimum.
    /// Disable only to compare raw solver behavior or to rule presolve out
    /// while investigating a suspected numerical issue. Default `true`.
    pub presolve: bool,
    /// Edit the tolerances used by the solver, most callers
    /// should leave this at [`Tolerances::default`]; override an individual
    /// field only once you understand the correctness/permissiveness
    /// trade-off documented on it.
    pub tolerances: Tolerances,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            time_limit: None,
            node_limit: None,
            mip_gap: 0.0,
            int_tol: 1e-6,
            warm_start: None,
            presolve: true,
            tolerances: Tolerances::default(),
        }
    }
}

impl SolveOptions {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if !self.mip_gap.is_finite() || self.mip_gap < 0.0 {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.mip_gap: expected a finite non-negative value".to_string(),
            ));
        }
        if !self.int_tol.is_finite() || !(0.0..0.5).contains(&self.int_tol) {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.int_tol: expected a finite value in [0, 0.5)".to_string(),
            ));
        }
        if !self.tolerances.feasibility.is_finite() || self.tolerances.feasibility < 0.0 {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.tolerances.feasibility: expected a finite non-negative value"
                    .to_string(),
            ));
        }
        if !self.tolerances.integrality_rounding.is_finite()
            || !(0.0..0.5).contains(&self.tolerances.integrality_rounding)
        {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.tolerances.integrality_rounding: expected a finite value in [0, 0.5)"
                    .to_string(),
            ));
        }
        if !self.tolerances.prune_epsilon.is_finite() || self.tolerances.prune_epsilon < 0.0 {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.tolerances.prune_epsilon: expected a finite non-negative value"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Overrides for the solver settings used for a subsequent search/resume call.
///
/// These fields override the ones defined in the previous call to
/// [`Problem::solve`] or [`Problem::resume`].
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ResumeOptions {
    /// New wall-clock budget (`None` = unlimited).
    pub time_limit: Option<Duration>,
    /// New branch-and-bound node budget (`None` = unlimited).
    pub node_limit: Option<u64>,
    /// New relative MIP gap (`None` = no MIP gap / exact optimality `0.0`).
    pub mip_gap: Option<f64>,
}

impl ResumeOptions {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if let Some(mip_gap) = self.mip_gap {
            if !mip_gap.is_finite() || mip_gap < 0.0 {
                return Err(Error::InvalidOptions(
                    "invalid ResumeOptions.mip_gap: expected a finite non-negative value"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// Numeric tolerances for a solve (see [`SolveOptions::tolerances`]).
///
/// This options override the solver's tolerances when solving the problem.
/// Only edit those if you are sure of the impact of changing those values.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Tolerances {
    /// The absolute feasibility tolerance of a solution: each variable's
    /// distance outside its bounds and each row's distance outside its
    /// feasible range may be at most this, or the round-off of evaluating
    /// the row or bound in `f64` if that is larger. The simplex engine
    /// holds every row to it, and it validates a rounded-to-integer
    /// candidate before it is accepted as the incumbent, a pure-LP solution
    /// before it is returned, and, in the post-edit warm-start pre-filter,
    /// whether a previous incumbent survives a [`crate::Solution`] edit.
    /// Must be finite and non-negative. Default `1e-7`.
    pub feasibility: f64,
    /// Distance from the nearest integer within which an integer/boolean
    /// variable's value is still treated as exactly that integer. Used by
    /// the post-edit warm-start pre-filter's integrality check.
    /// Must be finite and in the half-open range `[0, 0.5)`. Default `1e-5`.
    pub integrality_rounding: f64,
    /// Relative slack subtracted from the incumbent objective to form the
    /// branch & bound pruning cutoff: a node whose bound is not strictly
    /// better than `incumbent - max(prune_epsilon, prune_epsilon *
    /// |incumbent|)` is pruned. Guards against continuing to explore or
    /// retain nodes that could only ever match the incumbent to within
    /// float noise.
    ///
    /// Must be finite and non-negative. Default `1e-9`.
    pub prune_epsilon: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            feasibility: 1e-7,
            integrality_rounding: 1e-5,
            prune_epsilon: 1e-9,
        }
    }
}

/// Statistics of a solve, available via [`crate::Solution::stats`].
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct Stats {
    /// Branch & bound nodes whose LP was solved (0 for pure-LP problems).
    pub nodes_solved: u64,
    /// Total simplex pivots across the whole solve (including the root LP).
    pub lp_iterations: u64,
    /// Wall-clock time spent inside the solver, accumulated across resumes.
    pub elapsed: Duration,
    /// Best proven bound on the objective, in user space. `None` until an
    /// incumbent or an open node exists to derive one from.
    pub best_bound: Option<f64>,
    /// Relative gap between incumbent and best bound. `None` until both are
    /// known; `Some(0.0)` once optimality is proven.
    pub gap: Option<f64>,
}
