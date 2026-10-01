//! The public options of a solve and what it reports back: what it was asked
//! to do ([`SolveOptions`], [`ResumeOptions`], [`Tolerances`]), how it ended
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
    /// stops and can be resumed.
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
    /// as integral. A solution reports an integer or boolean variable's value
    /// as the solver computed it, within this distance of an integer, not
    /// necessarily on it: values like `0.9999999` are normal (one fixed at
    /// an integer, by its bounds, the search or an edit, lies on it). Use
    /// `.round()` where an exact integer is needed, bearing in mind that the
    /// rounded values need not meet the constraints: under `x <= 1e7 * y`, a
    /// `y` of `1e-6` allows `x` up to 10, which `y` rounded to zero forbids.
    /// Keep a big-M coefficient times this tolerance small against the
    /// quantities it bounds. Default `1e-6`. Must be finite and in the
    /// half-open range `[0, 0.5)`.
    pub int_tol: f64,
    /// Optional (partial) starting assignment used to seed the incumbent. The
    /// variables it leaves out are completed by an LP; a hint that is invalid or
    /// infeasible, or whose completion is fractional or fails, is ignored.
    /// Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
    /// Enable presolve, which reduces the problem before the search starts
    /// (dropping redundant rows and, for integer variables, tightening bounds,
    /// fixing variables and tightening big-M coefficients). It never removes
    /// variables or changes a continuous variable's bounds, and it keeps, up
    /// to round-off, every point with each integer variable exactly integral
    /// that meets each constraint within [`Tolerances::feasibility`] less the
    /// round-off of evaluating it (the tolerance the simplex holds it to) and
    /// each bound within its tolerance, or, for a variable it fixes because of
    /// the objective, at least one optimum. The problem you observe through
    /// [`crate::Solution`] is the one you wrote: answers are checked against
    /// it. A point beyond that tolerance can be cut, and so can a point within
    /// it only with an integer variable off its integer, within `int_tol`:
    /// without presolve the MIP search can reach one through a
    /// variable that presolve would fix or rewrite. So on a problem whose
    /// every point needs that, presolve on can answer `Infeasible`, or a worse
    /// objective, where presolve off finds a point. If presolve finds the
    /// problem infeasible, the problem is solved without it. Disable it only to
    /// compare raw solver behavior or to rule it out when investigating a
    /// numerical issue. Default `true`.
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
        if !self.tolerances.feasibility.is_finite()
            || self.tolerances.feasibility < MIN_FEASIBILITY_TOLERANCE
        {
            return Err(Error::InvalidOptions(format!(
                "invalid SolveOptions.tolerances.feasibility: expected a finite value of at least \
                 {MIN_FEASIBILITY_TOLERANCE:e}"
            )));
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
/// [`Problem::solve_with`](crate::Problem::solve_with) or
/// [`SolveOutcome::resume_with`](crate::SolveOutcome::resume_with).
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

/// The smallest accepted [`Tolerances::feasibility`]. Below it the tolerance
/// stops being one: at zero, every bound at zero is held exactly and every
/// row to the bare round-off of its evaluation, so a basic value computed a
/// few ulps off a bound is infeasible and the simplex pivots on round-off.
const MIN_FEASIBILITY_TOLERANCE: f64 = 1e-10;

/// Numeric tolerances for a solve (see [`SolveOptions::tolerances`]).
///
/// This options override the solver's tolerances when solving the problem.
/// Only edit those if you are sure of the impact of changing those values.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Tolerances {
    /// The absolute feasibility tolerance of a solution: each variable may lie
    /// outside its bounds, and each row outside its feasible range, by at most
    /// this, or by the round-off of evaluating the bound or row in `f64` if
    /// that is larger. The simplex holds every bound to it (an integer
    /// variable its bounds fix to its value exactly) and every row to it less
    /// the round-off of evaluating the row, and it is checked on a candidate,
    /// as computed, before it becomes the incumbent, on a pure-LP solution
    /// before it is returned, and on a previous incumbent to decide whether it
    /// survives a [`crate::Solution`] edit. Being absolute, it lets a variable
    /// with a large coefficient move its rows far: a `y` `1e-7` above its
    /// upper bound `u` lets `x <= 1e8 * y` hold at `x = 1e8 * u + 10`.
    /// Must be finite and at least `1e-10`. Default `1e-7`.
    pub feasibility: f64,
    /// Distance from the nearest integer within which an integer/boolean
    /// variable's value is still treated as exactly that integer: by the
    /// post-edit warm-start pre-filter's integrality check, and by
    /// [`crate::Solution::fix_var`], which fixes such a value at that integer.
    /// A solution's integer values lie within [`SolveOptions::int_tol`] of an
    /// integer (a fixed integer variable's on it), so with `int_tol` at most
    /// this, as by default, `fix_var` accepts every value a solution reports.
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
