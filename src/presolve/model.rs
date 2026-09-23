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
    /// The input rows `constraints` does not carry as written (dropped as
    /// implied, or rewritten for integer points), unchanged: the engine
    /// checks the answer against them as against its own rows
    /// (`Solver::set_validation_rows`), so a reduction that cut or let
    /// through a point it should not have shows at the boundary.
    pub validation_rows: Vec<(CsVec, ComparisonOp, f64)>,
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
    /// The signed sum of the fixed terms folded into `lo`/`hi`: the bounds
    /// as written are `lo + shift` and `hi + shift` (see [`Row::rhs_mag`]).
    pub(crate) shift: f64,
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

impl Row {
    /// The magnitude of the row bound `b` (one of `lo`, `hi`) as the engine
    /// sees it: the right-hand side as written, fixed terms not substituted
    /// out. The engine evaluates the row with those terms in it, so its
    /// magnitude, and every tolerance derived from it, counts this and not
    /// what is left of `b` after substitution (at `x = y = 1e9`, `x + y =
    /// 2e9 + 1.5e-5` leaves `1.5e-5`, but its magnitude is 4e9).
    pub(crate) fn rhs_mag(&self, b: f64) -> f64 {
        (b + self.shift).abs()
    }
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
    /// `Σ|a_j| · bound_tol_j`, each var's tolerance taken at the bound it
    /// sits at in the minimum-activity corner: how far below `l` the
    /// activity of a point the engine may return can reach (module docs).
    pub(crate) slack_l: f64,
    /// The same at the maximum-activity corner, beyond `u`.
    pub(crate) slack_u: f64,
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
