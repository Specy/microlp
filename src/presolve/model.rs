//! The data presolve works on and hands back: the reduction [`Mode`], the
//! in-flight [`Row`] and [`Activity`] views, and the [`Presolved`] result.

use crate::{ComparisonOp, CsVec, VarDomain};

/// Which reduction families are sound for this solve path (see module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Lp,
    Mip,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PresolveStats {
    pub rows_dropped: usize,
    pub bounds_tightened: usize,
    pub vars_fixed: usize,
    pub coeffs_tightened: usize,
    pub passes: usize,
}

#[derive(Debug)]
pub(crate) struct Presolved {
    pub var_mins: Vec<f64>,
    pub var_maxs: Vec<f64>,
    pub constraints: Vec<(CsVec, ComparisonOp, f64)>,
    /// Reduction counters; consumed by the unit tests and the debug log
    /// (the lib itself logs from the working copy before this is built,
    /// hence the dead-code allowance).
    #[allow(dead_code)]
    pub stats: PresolveStats,
}

/// One row in the uniform two-sided working form `lo <= a·x <= hi` (exactly
/// one side finite for Le/Ge, both equal for Eq — MPS ranged rows arrive as
/// two one-sided rows, so this is lossless and no transformation below ever
/// creates a genuinely two-sided row).
pub(crate) struct Row {
    pub(crate) vars: Vec<usize>,
    pub(crate) coeffs: Vec<f64>,
    pub(crate) lo: f64,
    pub(crate) hi: f64,
    /// Magnitudes folded into `lo`/`hi` by fixed-var substitution: part of
    /// the row's magnitude at every point (the folded terms are still terms
    /// of the original row), so every tolerance of this row includes it.
    pub(crate) fold_scale: f64,
    /// The `(var, coefficient)` terms substituted out of `vars`/`coeffs`.
    /// Folding only changes presolve's view of the row: the engine gets the
    /// fixed terms back (see the emission in [`presolve`]), because a row
    /// keeps the tolerance of its full magnitude only with them in it.
    pub(crate) folded: Vec<(usize, f64)>,
    pub(crate) alive: bool,
    /// Index of this row in the input `constraints` slice.
    pub(crate) orig: u32,
    /// Whether coefficients/bounds were rewritten (coefficient tightening).
    /// Every other surviving row is emitted as a clone of the input row —
    /// byte-identical and cheaper than rebuilding a CsVec.
    pub(crate) coeffs_touched: bool,
}

/// Row activity bounds computed from a bound set, with the standard
/// infinite-contribution counters (a bound is derivable for a term exactly
/// when the rest of the row has no infinite contribution on that side).
pub(crate) struct Activity {
    pub(crate) l: f64,
    pub(crate) u: f64,
    pub(crate) ninf_l: u32,
    pub(crate) ninf_u: u32,
    /// Sum of |finite minimum contributions|: with the row bound and the
    /// fold scale, the row's magnitude at its minimum-activity corner.
    pub(crate) abs_l: f64,
    /// Same for the maximum-activity corner.
    pub(crate) abs_u: f64,
    /// `Σ|a_j| · bound_tol_j`: how far beyond `[l, u]` the activity of a
    /// point the engine may return can reach (module docs).
    pub(crate) slack: f64,
}

pub(crate) fn is_int_domain(d: &VarDomain) -> bool {
    matches!(d, VarDomain::Integer | VarDomain::Boolean)
}

/// Contribution range of one term given the var's bounds.
pub(crate) fn term_range(a: f64, lo: f64, hi: f64) -> (f64, f64) {
    if a > 0.0 {
        (a * lo, a * hi)
    } else {
        (a * hi, a * lo)
    }
}

/// The larger finite magnitude of two bounds (zero when both are infinite).
pub(crate) fn bound_magnitude(lo: f64, hi: f64) -> f64 {
    [lo.abs(), hi.abs()]
        .into_iter()
        .filter(|m| m.is_finite())
        .fold(0.0, f64::max)
}
