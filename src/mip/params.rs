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

/// Step cap for the root diving heuristic: each step fixes one integer var
/// and re-solves the LP, so this bounds the dive's LP work regardless of
/// problem size. The dive is advisory (its only product is a possible first
/// incumbent), so abandoning at the cap is always sound.
pub(crate) const DIVE_MAX_STEPS: usize = 256;

/// Simplex-pivot budget for the whole dive, as a multiple of what the ROOT
/// LP itself took (with an absolute floor for trivial roots). Keeps the
/// dive's LP work a bounded fraction of the solve regardless of size.
pub(crate) const DIVE_PIVOT_FACTOR: u64 = 2;
pub(crate) const DIVE_PIVOT_MIN: u64 = 500;

/// The dive fires only after this many nodes have been solved in a run with
/// NO incumbent found. Instances that find incumbents naturally never dive
/// (a root-time dive measurably perturbed their search trajectory: gt2 went
/// 75 ms → 315 ms from a weaker-cutoff tree, not from dive cost); the dive
/// exists purely as a rescue for searches that would otherwise run their
/// whole budget empty-handed. 512 was tuned on the corpus: at 128 the dive
/// still fired inside BIP_easy's natural incumbent-discovery window and its
/// weaker cutoff cost ~25%; at 512 BIP never dives while mod008 (incumbent-
/// less past 512 nodes) keeps a ~30% win from the rescue.
pub(crate) const DIVE_TRIGGER_NODES: u64 = 512;

/// Cap on root cut-loop rounds (each round separates on the re-solved
/// optimum and adds a batch of cuts). The loop usually exits earlier on the
/// tailing-off test below; the cap bounds root work on instances where
/// separation keeps finding shallow cuts that barely move the bound.
pub(crate) const CUT_MAX_ROUNDS: u32 = 8;

/// Most-violated cuts admitted per round. Each added row costs a basis
/// refactorization plus a short dual-simplex re-solve, so unbounded batches
/// would let one cut-rich round dominate the root; the next round separates
/// against the re-solved point anyway, which supersedes anything dropped.
pub(crate) const CUTS_PER_ROUND: usize = 64;

/// Minimum violation (LHS − rhs at the separation point) for a cut to be
/// admitted. Cover cuts have ±1 coefficients, so this is on a unit scale:
/// below it a cut barely touches the current optimum and mostly adds rows.
pub(crate) const CUT_MIN_VIOLATION: f64 = 1e-4;

/// Tailing-off exit for the root cut loop: a round that improves the root
/// bound by less than this (relative to `1 + |bound|`) ends the loop — the
/// remaining gap belongs to the tree search, not to more shallow cuts.
pub(crate) const CUT_TAILOFF_REL: f64 = 1e-6;

/// A candidate cover is only trusted when its weight exceeds the knapsack
/// capacity by this margin, relative to `max(1, |capacity|)`. The emitted
/// inequality's validity rests on "all cover members at 1 violates the
/// row", which must hold beyond float round-off AND beyond the feasibility
/// tolerance the LP enforces rows at (1e-7): a cover that overshoots by
/// less than the row's enforcement slack proves nothing.
pub(crate) const COVER_MARGIN_REL: f64 = 1e-6;

/// Keep-or-rollback threshold for the WHOLE root cut loop: if the final
/// bound gain over the pre-loop bound (relative to `1 + |bound|`) does not
/// clear this, the pre-loop solver snapshot is restored and every cut row
/// vanishes. Cuts exist to move the bound; zero-gain cuts still perturb the
/// search trajectory through a different optimal vertex — measured +32% on
/// BIP_easy, whose root bound already equals its optimum (its search time
/// is incumbent DISCOVERY, which extra rows and a reshuffled vertex only
/// disturb). Trajectory effects can also land lucky (enigma measured −39%
/// from zero-gain cuts), but a coin-flip is not a mechanism; the rollback
/// keeps only what provably paid.
pub(crate) const CUT_KEEP_MIN_GAIN_REL: f64 = 1e-6;
