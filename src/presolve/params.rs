//! Presolve's thresholds and caps. None of them is part of the soundness
//! argument, which rests on the tolerance model in the module docs.

/// Minimum relative improvement for recording a working continuous bound.
/// It only filters ulp-level churn between passes; a bound's validity comes
/// from its tolerance relaxation, not from this.
pub(crate) const IMPROVE_REL: f64 = 1e-9;

/// Margin required to drop a row as redundant, relative to the row's
/// magnitude. Keeping a redundant row costs only speed, so this errs toward
/// keeping.
pub(crate) const REDUNDANT_MARGIN: f64 = 1e-9;

/// Minimum relative improvement for a binary coefficient tightening: the
/// coefficient must shrink by at least this fraction of its magnitude.
pub(crate) const COEFF_MIN_IMPROVE: f64 = 0.1;

/// A long row is coefficient-tightened only when at most this many of its
/// terms are eligible. The excess `U − b` is invariant under each
/// application, so on a loose dense binary row every coefficient becomes
/// eligible and the row collapses toward a cardinality constraint, a tighter
/// but massively degenerate relaxation. What pays on long rows is the big-M
/// signature, one or two oversized coefficients per row (`x − M·y ≤ 0`).
pub(crate) const COEFF_TIGHTEN_ROW_CAP: usize = 2;

/// Rows with at most this many terms are exempt from
/// [`COEFF_TIGHTEN_ROW_CAP`]: a few rewritten coefficients cannot create the
/// degeneracy dense rows can, and on small binary knapsacks the full rewrite
/// shrinks the search tree.
pub(crate) const COEFF_TIGHTEN_SHORT_ROW: usize = 16;

/// Primal fixpoint pass cap (per round) and outer primal+dual round cap.
pub(crate) const MAX_PASSES: usize = 10;
pub(crate) const MAX_ROUNDS: usize = 3;
