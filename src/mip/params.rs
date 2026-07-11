//! Internal, non-configurable numeric constants for the branch & bound driver.
//!
//! These differ from [`crate::Tolerances`] in kind, not just location: every
//! constant here is an implementation-detail knob that keeps one specific
//! piece of solver machinery numerically well-behaved — a denominator that
//! must not be zero, a floor that keeps a score function totally ordered, a
//! slack against float round-trip noise in an already-validated input. None
//! of them carries user-facing semantic meaning, none is expected to need
//! tuning from outside the crate, and changing one is a source edit plus a
//! rebuild, never a per-solve option. If a number here ever needs to become
//! something a caller can reasonably want to tune, promote it to
//! [`crate::Tolerances`] rather than adding a separate top-level
//! [`crate::SolveOptions`] field for it.

/// Floor applied to each side of the pseudocost product-score
/// `max(est_down * f_down, SCORE_EPS) * max(est_up * f_up, SCORE_EPS)`
/// computed by `choose_branch_var`. Without it, a variable whose estimated
/// degradation on one side is still exactly (or numerically near) zero —
/// always true before any observation on a zero-objective-coefficient
/// variable — would score exactly zero and be indistinguishable from a
/// variable that is genuinely useless to branch on; the floor keeps the
/// product rule a well-defined total order from the very first branching
/// decision onward.
pub(crate) const SCORE_EPS: f64 = 1e-6;

/// Offset added to `|obj_coeff|` when seeding a variable's initial
/// (pre-observation) pseudocost estimate in `PseudoCosts::new`. Keeps the
/// seed estimate strictly positive even for a variable with a zero
/// objective coefficient, so [`SCORE_EPS`]'s floor above is never the only
/// thing standing between an unobserved variable and a hard-zero score.
pub(crate) const PSEUDOCOST_INIT_EPS: f64 = 1e-6;

/// Denominator guard applied to `branch_frac` when a node's actual
/// objective degradation is folded back into the pseudocost that predicted
/// it (`PseudoCosts::record`, called from the search loop after each node
/// solve). `branch_frac` is a fractional part in `(0.0, 1.0]` by
/// construction, but the guard is kept defensively so a value that rounds
/// to (numerically) zero can never divide by zero or blow up the recorded
/// per-unit degradation.
pub(crate) const BRANCH_FRAC_GUARD: f64 = 1e-6;

/// Denominator guard in `relative_gap`'s `incumbent_obj.abs().max(..)`: an
/// incumbent objective of exactly (or numerically near) zero must not make
/// the reported relative gap divide by zero or explode to `inf`.
pub(crate) const GAP_DENOM_GUARD: f64 = 1e-10;

/// Slack allowed on each side of a warm-start hint's bound check in
/// `try_warm_start`. A hint that misses its variable's bounds by more than
/// float round-trip noise is a genuinely invalid hint and is dropped — the
/// whole mechanism is advisory, per [`crate::SolveOptions::warm_start`] —
/// this constant only forgives noise in the hint value itself, never a real
/// out-of-bounds input.
pub(crate) const HINT_BOUNDS_SLACK: f64 = 1e-9;

/// Node propagation self-throttle: after this many propagation calls the
/// driver checks the hit rate (calls that pruned the node or deduced at
/// least one bound) and disables propagation for the rest of the search
/// when it falls below `1 / PROP_HIT_DIVISOR`. On structures propagation
/// cannot deduce from (covering rows, loose dense rows) it is pure per-node
/// overhead — measured 2× on BIP_easy before this throttle existed — while
/// effective instances (fixed-charge, bounded general integers) hit far
/// above the bar from the very first nodes.
pub(crate) const PROP_SAMPLE_CALLS: u32 = 64;
pub(crate) const PROP_HIT_DIVISOR: u32 = 8;

/// Reduced costs below this magnitude are treated as zero by the
/// reduced-cost fixing pass: dividing the cutoff slack by a noise-level
/// reduced cost would produce a garbage (astronomical) movement bound, and
/// a genuinely-zero reduced cost carries no fixing information at all.
pub(crate) const RC_EPS: f64 = 1e-9;
