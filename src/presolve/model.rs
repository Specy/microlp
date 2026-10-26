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
    /// Per row of `constraints`, the index of the input row it came from.
    pub kept_rows: Vec<usize>,
    /// The input rows that `constraints` does not carry as written (dropped,
    /// or rewritten for integer points), unchanged. The engine checks the
    /// answer against them too (`Solver::set_validation_rows`), so a wrong
    /// reduction shows at the boundary, where an LP takes the row over.
    pub validation_rows: Vec<(CsVec, ComparisonOp, f64)>,
    /// Per row of `validation_rows`, its index among the input rows.
    pub validation_origs: Vec<usize>,
    /// Whether this differs from the model as written: an integer bound
    /// tightened or rounded, or a row dropped or rewritten. A model presolve
    /// did not reduce is solved as written
    /// ([`reduced_model`](super::reduced_model)).
    pub reduced: bool,
    /// Reduction counters, read only by the unit tests (the debug log reads
    /// the working copy), hence the dead-code allowance.
    #[allow(dead_code)]
    pub stats: PresolveStats,
}

/// One row in the two-sided working form `lo <= a·x <= hi`: exactly one side
/// finite for Le/Ge, both equal for Eq. No reduction ever creates a genuinely
/// two-sided row.
pub(crate) struct Row {
    pub(crate) vars: Vec<usize>,
    pub(crate) coeffs: Vec<f64>,
    pub(crate) lo: f64,
    pub(crate) hi: f64,
    /// Sum of the magnitudes folded into `lo`/`hi` by fixed-var substitution.
    /// The folded terms are still terms of the row, so every tolerance of the
    /// row includes it.
    pub(crate) fold_scale: f64,
    /// The signed sum of the fixed terms folded into `lo`/`hi`: the bounds
    /// as written are `lo + shift` and `hi + shift` (see [`Row::rhs_mag`]).
    pub(crate) shift: f64,
    /// The `(var, coefficient)` terms substituted out of `vars`/`coeffs`.
    /// Folding changes only presolve's view of the row: the engine gets the
    /// fixed terms back, since only with them does the row keep the
    /// tolerance of its full magnitude.
    pub(crate) folded: Vec<(usize, f64)>,
    pub(crate) alive: bool,
    /// Index of this row in the input `constraints` slice.
    pub(crate) orig: u32,
    /// Whether coefficient tightening rewrote the row. Every other surviving
    /// row is emitted as its input row, byte-identical.
    pub(crate) coeffs_touched: bool,
    /// The binding branch of the row's rewrites that shrink its bound, if
    /// any.
    pub(crate) binding: Option<Binding>,
}

/// The binding branch of a row's coefficient-tightening rewrites that shrink
/// its bound, their binaries at one (`Work::rewrite_room`): its magnitude over
/// the bound, counted by [`Row::rhs_mag`] (which substitution leaves
/// unchanged), and the binaries' terms.
pub(crate) struct Binding {
    /// The rewritten binaries.
    pub(crate) binaries: Vec<usize>,
    /// `|b| + Σ|a_k|` over the bound and the binaries' coefficients as
    /// written, and as rewritten (the bound before its relaxation).
    pub(crate) written: f64,
    pub(crate) rewritten: f64,
    /// The relaxation of the bound, at the fixed terms folded so far.
    pub(crate) kept: f64,
}

impl Row {
    /// Input row `orig`, `coeffs op rhs`, in the working form, nothing folded
    /// or rewritten yet.
    pub(crate) fn as_written(orig: usize, coeffs: &CsVec, op: ComparisonOp, rhs: f64) -> Self {
        let (lo, hi) = match op {
            ComparisonOp::Le => (f64::NEG_INFINITY, rhs),
            ComparisonOp::Ge => (rhs, f64::INFINITY),
            ComparisonOp::Eq => (rhs, rhs),
        };
        Row {
            vars: coeffs.indices().to_vec(),
            coeffs: coeffs.data().to_vec(),
            lo,
            hi,
            fold_scale: 0.0,
            shift: 0.0,
            folded: Vec::new(),
            alive: true,
            orig: orig as u32,
            coeffs_touched: false,
            binding: None,
        }
    }

    /// The magnitude of row bound `b` (`lo` or `hi`) as the engine sees it:
    /// the right-hand side as written, fixed terms not substituted out, since
    /// the engine evaluates the row with those terms in it.
    pub(crate) fn rhs_mag(&self, b: f64) -> f64 {
        (b + self.shift).abs()
    }

    /// Var `v`'s coefficient, among the terms or the fixed terms folded out.
    pub(crate) fn coeff(&self, v: usize) -> Option<f64> {
        let terms = self.vars.iter().zip(&self.coeffs);
        let folded = self.folded.iter().map(|(u, a)| (u, a));
        terms.chain(folded).find(|&(&u, _)| u == v).map(|(_, &a)| a)
    }
}

/// Row activity bounds over a bound set, with counts of infinite
/// contributions (a term's implied bound exists exactly when the rest of the
/// row has no infinite contribution on that side).
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
    /// `Σ|a_j| · bound_tol_j` at the minimum-activity corner: how far below
    /// `l` the activity of a point the engine may return can reach.
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
