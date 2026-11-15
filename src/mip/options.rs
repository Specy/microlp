//! The public options of a solve and what it reports back: what it was asked
//! to do ([`SolveOptions`], [`ResumeOptions`], [`Tolerances`]), how it ended
//! ([`SolutionStatus`], [`TerminationReason`]) and what it cost ([`Stats`]).

use crate::{Error, Variable};
use core::time::Duration;

/// Says whether a solution is proven optimal or only feasible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionStatus {
    /// The solution is optimal up to numerical tolerances. This does not
    /// guarantee the best objective over every assignment those tolerances
    /// allow. See [`Tolerances::feasibility`] for possible objective error on
    /// models with large differences in coefficient magnitude.
    Optimal,
    /// A valid incumbent, the best solution found so far, is available, but
    /// the solver has not proved it optimal.
    Feasible,
}

/// Says why a solve or resume call stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TerminationReason {
    /// The solver proved optimality within the numerical tolerances.
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
    /// Wall-clock time limit for one call. `None` means unlimited. A duration
    /// too large to represent as a time limit, such as [`Duration::MAX`], also
    /// means unlimited. Time limits are checked periodically, so the call
    /// can run slightly past its limit. You can resume the resulting outcome.
    pub time_limit: Option<Duration>,
    /// Maximum number of branch & bound nodes to solve in one call. `None`
    /// means unlimited. This is a deterministic alternative to
    /// `time_limit`. The limit applies per call, so
    /// [`crate::SolveOutcome::resume`] reapplies it as a fresh limit. The
    /// initial LP solve does not count as a node.
    pub node_limit: Option<u64>,
    /// Relative MIP gap at which the solve may stop with its incumbent, the
    /// best solution found so far. Stopping at this gap reports
    /// [`SolutionStatus::Feasible`] and [`TerminationReason::MipGap`]. Must be
    /// finite and non-negative. Default `0.0` disables gap-based stopping.
    /// Numerical tolerances still apply, including
    /// [`Tolerances::prune_epsilon`].
    pub mip_gap: f64,
    /// Integrality tolerance for accepting integer and boolean values.
    /// Reported values are within this distance of an integer, but need not
    /// equal it. Values such as `0.9999999` are normal. A variable fixed at an
    /// integer by its bounds, the solve, or an edit has that exact value.
    /// Round values when you need exact integers.
    /// [`crate::Solution::var_domain`] identifies integer and boolean
    /// variables and explains why rounding can violate constraints. Keep a
    /// big-M coefficient times this tolerance small relative to the
    /// quantities it bounds.
    ///
    /// Integer and boolean variables still define an integer model.
    /// Increasing this tolerance does not make the solver seek every
    /// assignment it would accept. A model feasible only at slightly
    /// nonintegral values normally returns [`Error::Infeasible`]. The solver
    /// can still return such a solution if it happens to find one. If the
    /// objective can improve forever from that assignment, the solve returns
    /// [`Error::Unbounded`]. This applies with presolve on or off. See
    /// [`Error::Infeasible`] for the exact infeasibility conditions.
    ///
    /// This tolerance also determines integrality in
    /// [`crate::Solution::fix_var`] and when checking whether the best
    /// solution found so far remains valid after an edit. Default `1e-6`.
    /// Must be finite and in the half-open range `[0, 0.5)`.
    pub int_tol: f64,
    /// Optional starting assignment for a problem with integer or boolean
    /// variables. You may supply values for some or all variables. A usable
    /// hint provides an incumbent, the best solution found so far, to improve
    /// on. Finite hints outside variable bounds, excluded by constraints, or
    /// otherwise unusable may be ignored.
    ///
    /// A NaN or infinite value, or a variable index absent from the problem,
    /// returns [`Error::InvalidOperation`]. A problem with only real
    /// variables validates the hint the same way but does not use it.
    /// Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
    /// Simplify the problem before solving. This makes most solves faster.
    /// Every variable is still reported, and every returned solution is
    /// checked against the original problem.
    ///
    /// Up to floating-point round-off, presolve preserves assignments that
    /// satisfy every constraint and bound exactly and give integer variables
    /// whole-number values. [`SolveOptions::dual_fixing`] can exclude some
    /// of these assignments while preserving an optimum.
    ///
    /// Disable presolve only to compare behavior or rule it out when
    /// investigating a numerical issue. Default `true`.
    pub presolve: bool,
    /// Allow presolve to fix an integer variable at a bound when doing so
    /// cannot worsen the objective or violate a constraint. This preserves
    /// an optimum but may exclude other feasible solutions, including other
    /// optima.
    ///
    /// Disable this when every integer solution must remain reachable, for
    /// example when enumerating solutions within an objective cutoff by
    /// adding constraints and solving again. You can also disable it to
    /// investigate a result. It takes effect only with
    /// [`SolveOptions::presolve`] on and integer or boolean variables.
    /// Supplying any [`SolveOptions::warm_start`] hint disables it, so that
    /// it cannot exclude a usable hint. Default `true`.
    pub dual_fixing: bool,
    /// Numeric tolerances used by the solve. Defaults suit most problems.
    /// Read each field's conditions before changing it.
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
    /// New wall-clock time limit. `None` means unlimited. A duration too large
    /// to represent as a time limit, such as [`Duration::MAX`], also means
    /// unlimited.
    pub time_limit: Option<Duration>,
    /// New branch & bound node limit. `None` means unlimited.
    pub node_limit: Option<u64>,
    /// New relative MIP gap. `None` removes gap-based stopping and sets the
    /// gap to `0.0`. Numerical tolerances still apply as documented in
    /// [`SolveOptions::mip_gap`].
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
/// Defaults suit most problems. Read each field's conditions before changing
/// it.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Tolerances {
    /// Absolute permitted violation of a variable bound or constraint. It is
    /// not relative to the magnitude of the bound or constraint.
    /// Every returned solution is checked against the problem as written.
    /// The permitted violation is this tolerance or the round-off of
    /// evaluating the bound or constraint in `f64`, whichever is larger.
    /// The same checks apply before accepting a candidate as the incumbent
    /// (the best solution found so far) or retaining it after a
    /// [`crate::Solution`] edit. An integer variable fixed by its bounds has
    /// that exact value.
    ///
    /// [`Error::Infeasible`] means no assignment satisfies the constraints
    /// and bounds within this tolerance while every integer variable is at
    /// an integer within its bounds and every fixed variable is at its fixed
    /// value. A problem that is feasible only because round-off lets a
    /// constraint pass is reported infeasible. See
    /// [`SolveOptions::int_tol`] for the distinction between the integer model
    /// and slightly nonintegral returned values.
    ///
    /// [`SolutionStatus::Optimal`] means optimal up to numerical tolerances.
    /// It does not guarantee the best objective over every assignment the
    /// tolerances allow. A large coefficient can multiply a tolerance-sized
    /// variable difference into a substantial objective difference from the
    /// exact optimum. For example, `y` `1e-7` above upper bound `u` allows
    /// `x <= 1e8 * y` at `x = 1e8 * u + 10`. Follow the coefficient-range
    /// recommendation in the crate documentation to limit these effects.
    /// Must be finite and at least `1e-10`. Default `1e-7`.
    pub feasibility: f64,
    /// Minimum objective improvement considered when skipping a branch.
    /// A branch must be able to beat the incumbent (the best solution found
    /// so far) by more than
    /// `max(prune_epsilon, prune_epsilon * |objective|)` to be considered.
    /// Here, `objective` is the incumbent's objective value. This applies to
    /// both minimization and maximization. A solution better by less than
    /// this amount can be missed.
    ///
    /// A solution relying on tolerances may also be missed even when its
    /// reported objective would improve on the incumbent.
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
    /// Branch & bound nodes solved. This is `0` for problems with only real
    /// variables.
    pub nodes_solved: u64,
    /// Total LP iterations across the whole solve, including the initial LP
    /// solve.
    pub lp_iterations: u64,
    /// Wall-clock time spent inside the solver, accumulated across resumes.
    pub elapsed: Duration,
    /// Best proven bound on the objective, expressed in the problem's
    /// optimization direction. `None` until the solve establishes a bound.
    pub best_bound: Option<f64>,
    /// Relative gap between the incumbent (the best solution found so far) and
    /// the best bound. `None` until both are known. Reports `Some(0.0)` once
    /// optimality is proven.
    pub gap: Option<f64>,
}
