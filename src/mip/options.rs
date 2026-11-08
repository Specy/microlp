//! The public options of a solve and what it reports back: what it was asked
//! to do ([`SolveOptions`], [`ResumeOptions`], [`Tolerances`]), how it ended
//! ([`SolutionStatus`], [`TerminationReason`]) and what it cost ([`Stats`]).

use crate::{Error, Variable};
use core::time::Duration;

/// Says whether a solution is proven optimal or only feasible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionStatus {
    /// The solver proved this solution optimal. For what that means on an
    /// ill-conditioned problem, see [`Tolerances::feasibility`].
    Optimal,
    /// A valid incumbent, the best solution found so far, is available, but
    /// the solver has not proved it optimal.
    Feasible,
}

/// Says why a solve or resume call stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TerminationReason {
    /// The LP or branch & bound search proved optimality.
    ProvenOptimal,
    /// The search reached the configured relative MIP gap before proving
    /// optimality.
    MipGap,
    /// The solve or resume call reached its wall-clock time limit.
    TimeLimit,
    /// The solve or resume call reached its branch & bound node limit.
    NodeLimit,
}

/// Settings for a solve call. Start with [`SolveOptions::default`] and change
/// the fields you need.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolveOptions {
    /// Wall-clock time limit for one call. `None` means unlimited. A time limit
    /// too long to add to the current time, such as [`Duration::MAX`], is also
    /// unlimited. When it expires, the search stops and can be resumed.
    pub time_limit: Option<Duration>,
    /// Maximum number of branch & bound nodes to solve in one call. `None`
    /// means unlimited. This is a deterministic alternative to
    /// `time_limit`. The limit applies per call, so
    /// [`crate::SolveOutcome::resume`] reapplies it as a fresh limit. The root
    /// LP relaxation does not count as a node.
    pub node_limit: Option<u64>,
    /// Relative MIP gap at which the search may stop with its incumbent, the
    /// best solution found so far. Stopping at this gap reports [`SolutionStatus::Feasible`] and
    /// [`TerminationReason::MipGap`]. Must be finite and non-negative. Default
    /// `0.0` (prove exact optimality).
    pub mip_gap: f64,
    /// Integrality tolerance for accepting integer and boolean variable values.
    /// A value within this distance of an integer counts as integral, though
    /// the search seeks solutions whose integer variables are integral (see
    /// [`SolveOptions::presolve`]). Returned values are the values the solver
    /// computed and may be within this distance of an integer without lying
    /// exactly on one; values like `0.9999999` are normal. A variable fixed at
    /// an integer by its bounds, the search or an edit lies on that integer.
    /// Round values when you need exact integers.
    /// [`crate::Solution::var_domain`] identifies integer and boolean
    /// variables; see its documentation for why rounded values need not meet
    /// the constraints. Keep a big-M coefficient
    /// times this tolerance small relative to the quantities it bounds. This
    /// tolerance also determines integrality in [`crate::Solution::fix_var`]
    /// and when checking whether an incumbent survives an edit. Default
    /// `1e-6`. Must be finite and in the half-open range `[0, 0.5)`.
    pub int_tol: f64,
    /// Optional partial assignment used to seed an incumbent for a problem
    /// with integer or boolean variables. The solver completes omitted
    /// variables with an LP. It ignores a hint that is invalid or infeasible
    /// for the model (a value is outside its variable's bounds, or a constraint
    /// excludes it), or whose LP completion is fractional or fails. A NaN or
    /// infinite value, or a variable outside the problem, makes the solve
    /// return [`Error::InvalidOperation`] instead. For a problem with only
    /// real variables, the solver validates the hint the same way but does not
    /// use it. Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
    /// Enable presolve. It reduces a problem before the solve starts by
    /// removing redundant constraints and, for integer variables, tightening
    /// bounds, fixing variables and tightening big-M coefficients. It never
    /// removes variables or changes a real variable's bounds.
    ///
    /// Up to round-off, presolve keeps every point whose integer variables are
    /// exactly integral and that meets every constraint within the tolerance
    /// the simplex holds it to (see [`Tolerances::feasibility`]), and every
    /// bound within its tolerance. If presolve fixes a variable because of the objective, it
    /// keeps an optimum. Every solution is checked against the problem as
    /// written.
    ///
    /// The search does not target solutions that rely on `int_tol`, such as an
    /// integer variable slightly off an integer under a coefficient that makes
    /// that offset matter. It may return such a solution if it lands on one. If
    /// the objective can improve forever from there, the result is
    /// `Unbounded`. If no solution has every integer variable exactly integral,
    /// the result is `Infeasible` unless the search lands on such a
    /// tolerance-based point. This applies whether presolve is on or off. If
    /// presolve finds the problem infeasible, microlp solves it without
    /// presolve. Disable presolve only to compare solver behavior without it or
    /// to rule it out when investigating a numerical issue. Default `true`.
    pub presolve: bool,
    /// Let presolve fix an integer variable at a bound when neither the
    /// objective nor any of its constraints gives a reason to move away from
    /// that bound (dual fixing). This keeps an optimum, not every feasible point. It can remove
    /// alternative optima and feasible points the objective does not prefer.
    ///
    /// Turn this off when every integral point must stay reachable. For
    /// example, turn it off to enumerate solutions within an objective cutoff
    /// by adding constraints and solving again, or to rule it out when
    /// investigating a result. It takes effect only when
    /// [`SolveOptions::presolve`] is on and the problem has integer or boolean
    /// variables. Supplying a [`SolveOptions::warm_start`] hint turns it off so
    /// that it cannot exclude a feasible hint. Default `true`.
    pub dual_fixing: bool,
    /// Numeric tolerances used by the solve. Defaults suit most problems.
    /// Change a field only after you understand the correctness and
    /// permissiveness trade-off documented for it.
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
            dual_fixing: true,
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
        if !self.tolerances.prune_epsilon.is_finite() || self.tolerances.prune_epsilon < 0.0 {
            return Err(Error::InvalidOptions(
                "invalid SolveOptions.tolerances.prune_epsilon: expected a finite non-negative value"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Settings to override for the next search or resume call.
///
/// Each field replaces the corresponding setting from the preceding call to
/// [`Problem::solve_with`](crate::Problem::solve_with) or
/// [`SolveOutcome::resume_with`](crate::SolveOutcome::resume_with).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct ResumeOptions {
    /// New wall-clock time limit. `None` means unlimited. A time limit too long
    /// to add to the current time, such as [`Duration::MAX`], is also unlimited.
    pub time_limit: Option<Duration>,
    /// New branch & bound node limit. `None` means unlimited.
    pub node_limit: Option<u64>,
    /// New relative MIP gap. `None` removes gap-based stopping and sets the gap
    /// to `0.0`, so the search must prove exact optimality.
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

/// Numeric tolerances used by a solve (see [`SolveOptions::tolerances`]).
///
/// Defaults suit most problems. Change a field only after you understand the
/// trade-off documented for it.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Tolerances {
    /// Absolute feasibility tolerance for variable bounds and constraint
    /// ranges. The solver checks every returned solution against the problem
    /// as written. A variable may lie outside its bounds, and a constraint
    /// outside its range, by at most this tolerance, or by the round-off of
    /// evaluating the bound or constraint in `f64` where that is larger. The
    /// solver applies the same check to each candidate before it becomes the
    /// incumbent (the best solution found so far), and to an incumbent before
    /// it survives a [`crate::Solution`] edit.
    ///
    /// The simplex keeps each variable within the full tolerance of its
    /// bounds. An integer variable fixed by its bounds stays at that value
    /// exactly. For a constraint, the simplex allows this tolerance minus the
    /// round-off from evaluating the constraint, so every solution it finds
    /// passes the check.
    ///
    /// [`Error::Infeasible`] means the solver proved that no point meets this
    /// tolerance with every integer variable at an integer within its bounds
    /// and every fixed variable at its fixed value. A problem feasible only
    /// because round-off lets a constraint pass is reported infeasible. So is
    /// a problem feasible only by using [`SolveOptions::int_tol`], unless the
    /// search lands on such a point (see [`SolveOptions::presolve`]).
    ///
    /// [`SolutionStatus::Optimal`] means optimal at the vertices the simplex
    /// reaches, not over every point admitted by this tolerance. Because the
    /// tolerance is absolute, a variable with a large coefficient can move its
    /// constraints far within it, and the objective with them. For example, `y` `1e-7` above upper bound `u` lets `x <= 1e8 * y` hold at
    /// `x = 1e8 * u + 10`. The coefficient range recommended in the crate
    /// documentation keeps these effects small. Must be finite and at least
    /// `1e-10`. Default `1e-7`.
    pub feasibility: f64,
    /// Relative slack in the branch & bound pruning cutoff. The solver prunes
    /// a node whose bound is not strictly better than
    /// `incumbent - max(prune_epsilon, prune_epsilon * |incumbent|)`. This
    /// avoids searching nodes that could only match the incumbent within
    /// floating-point noise.
    ///
    /// When every variable with a nonzero objective coefficient is integer or
    /// fixed, objective values at points with integral integer variables are
    /// separated by fixed-size steps. The solver also accounts for the room
    /// that tolerances and round-off could leave for improvement, then rounds
    /// the node's bound up to the next objective value on those steps. If this
    /// value is not strictly better than the incumbent, the solver prunes the
    /// node. No point in that node with integral integer variables and every
    /// constraint and bound met exactly can then beat the incumbent. A point
    /// that relies on tolerances can still be pruned even if its objective is
    /// better.
    ///
    /// Must be finite and non-negative. Default `1e-9`.
    pub prune_epsilon: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            feasibility: 1e-7,
            prune_epsilon: 1e-9,
        }
    }
}

/// Statistics for a solve, available via [`crate::Solution::stats`],
/// [`crate::SolveOutcome::stats`] and [`crate::InterruptedSolve::stats`].
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct Stats {
    /// Branch & bound nodes whose LP relaxation was solved. This is `0` for
    /// pure-LP problems.
    pub nodes_solved: u64,
    /// Total simplex pivots across the whole solve, including the root LP.
    pub lp_iterations: u64,
    /// Wall-clock time spent inside the solver, accumulated across resumes.
    pub elapsed: Duration,
    /// Best proven bound on the objective, expressed in the problem's
    /// optimization direction. `None` until an incumbent or an open node
    /// provides a bound.
    pub best_bound: Option<f64>,
    /// Relative gap between the incumbent (the best solution found so far) and
    /// the best bound. `None` until both are known; `Some(0.0)` once
    /// optimality is proven.
    pub gap: Option<f64>,
}
