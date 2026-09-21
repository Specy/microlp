//! Presolve's internal constants: the thresholds and caps the reductions in
//! [`super::work`] are gated on. Each one's comment says what it buys; none of
//! them is part of the soundness argument, which rests on the tolerance model
//! in the module docs.

/// Minimum relative improvement for recording a working continuous bound.
/// This only filters ulp-level churn between passes; a bound's validity
/// comes from its tolerance relaxation (module docs), not from this.
pub(crate) const IMPROVE_REL: f64 = 1e-9;

/// Margin required to drop a row as redundant, relative to the row's
/// magnitude. Keeping a truly redundant row costs nothing but speed, so
/// this errs toward keeping.
pub(crate) const REDUNDANT_MARGIN: f64 = 1e-9;

/// Minimum RELATIVE improvement for a binary coefficient tightening: the
/// coefficient must shrink by at least this fraction of its own magnitude.
pub(crate) const COEFF_MIN_IMPROVE: f64 = 0.1;

/// A LONG row is coefficient-tightened only when AT MOST this many of its
/// terms are eligible. The reduction's excess `U − b` is invariant under
/// each application, so on a loose dense binary row EVERY coefficient
/// cascades into eligibility and the row collapses toward a cardinality
/// constraint — a "tighter" relaxation that is massively degenerate in
/// practice (BIP_easy: 2 900 rows × ~290 terms, 367 409 rewrites, seconds
/// slower). The reduction that pays on long rows is the fixed-charge /
/// big-M signature: ONE (occasionally two) oversized coefficient per row
/// (`x − M·y ≤ 0`), which this cap isolates.
pub(crate) const COEFF_TIGHTEN_ROW_CAP: usize = 2;

/// Rows with at most this many terms are exempt from
/// [`COEFF_TIGHTEN_ROW_CAP`]: a handful of rewritten coefficients cannot
/// manufacture the degeneracy mass that dense rows can, and on small binary
/// MILPs (lseu-class knapsacks) the full row rewrite measurably shrinks the
/// search tree.
pub(crate) const COEFF_TIGHTEN_SHORT_ROW: usize = 16;

/// Primal fixpoint pass cap (per round) and outer primal+dual round cap.
pub(crate) const MAX_PASSES: usize = 10;
pub(crate) const MAX_ROUNDS: usize = 3;
