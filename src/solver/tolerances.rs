//! The engine's tolerance model: the one contract every layer holds a point
//! to, and the few constants that describe what the arithmetic can resolve.
//!
//! The contract (ARCHITECTURE §7): a value may leave its range by the user's
//! absolute `feasibility` tolerance, or by the round-off of the quantities it
//! was computed from if that is larger. [`row_tolerance`] is that rule,
//! [`bound_tolerance`] its instance for a var's bounds, and [`outside`] the
//! one comparison every check makes. The engine holds each row to
//! [`ROW_BUDGET_SHARE`] of the contract ([`slack_tol`]) and caps a var's
//! bound tolerance by what its rows can absorb ([`structural_tol`]), so that
//! the point it reports passes the same evaluation at the boundary (the MIP
//! candidate guard, the pure-LP validation) with the other half to spare for
//! a snapped integer or fixed value and for evaluating the row in another
//! order. Everything else here is a property of `f64` and of the
//! factorization, not of the contract.

/// The absolute part of the reduced-cost tolerance ([`dual_tol`]) and of
/// [`float_eq`], in the engine's scaled units, where quantities are of order
/// one: far above round-off, far below any decision the contract makes.
pub const EPS: f64 = 1e-10;

/// Relative round-off allowance: about a hundred ulps of `f64`. A quantity
/// computed from terms of magnitude `m` cannot be trusted below
/// `ROUNDOFF_FLOOR * m`, so no tolerance on it is ever tighter than that.
/// This is the same allowance the crate's own contract tests grant a row.
pub(crate) const ROUNDOFF_FLOOR: f64 = 1e-14;

/// Round-off of a single tableau entry computed through the factorization,
/// relative to the largest entry of its row or column: about ten ulps of
/// `f64`. An entry below this fraction of its neighbours is indistinguishable
/// from the round-off of computing it, and two computations of the same
/// entry (through the row and through the column) agree to about this much.
/// Tighter than [`ROUNDOFF_FLOOR`], which allows for a sum over a row's
/// many terms.
pub(crate) const ENTRY_ROUNDOFF: f64 = 1e-15;

/// Relative pivot tolerance of both ratio tests: an entry of the pivot row
/// (dual simplex) or pivot column (primal simplex) is a candidate pivot only
/// if its magnitude is at least this fraction of the largest entry scanned.
/// A pivot seven orders of magnitude below its neighbours amplifies the
/// round-off of the update by the same factor; with the matrix scaled, an
/// entry that small relative to its row or column is noise or as good as
/// noise. The largest entry is always a candidate, so this can never turn a
/// feasible model infeasible or a bounded one unbounded on its own.
pub(crate) const PIVOT_REL_TOL: f64 = 1e-7;

/// Relative disagreement between the pivot element computed through the
/// pivot row (`row_coeffs[col]`) and through the pivot column
/// (`col_coeffs[row]`) above which the factorization is not trusted for the
/// pivot: the two are the same number in exact arithmetic and agree to
/// ~1e-12 at every legitimate pivot ever measured, while a pivot on round-off
/// noise disagrees by tens of percent.
pub(crate) const PIVOT_AGREEMENT_TOL: f64 = 1e-7;

/// Iterative refinement steps [`super::Solver::verify_primal`] may apply to
/// the basic values before giving up on the current factorization.
pub(crate) const REFINEMENT_STEPS: usize = 3;

/// Upper bound on the number of primal/dual phase alternations
/// [`super::Solver::run_phases`] performs before giving up loudly. A phase ends on
/// values verified against the rows, so a second round is already rare;
/// hitting this bound means the basis is numerically unusable, which must
/// not be reported as an optimum.
pub(crate) const MAX_PHASE_ROUNDS: usize = 8;

/// The share of a row's contract tolerance the engine uses for the row
/// itself ([`slack_tol`]). The rest is reserved for what happens to the
/// point after the engine has verified it: an integer var is rounded and a
/// fixed var reported at its value, each moving the row by its coefficient
/// times the snap ([`snap_budget`] keeps that within the reserve), and the
/// row is re-evaluated in another order of operations (by the user, by the
/// contract tests, by presolve for the rows it dropped), which costs
/// round-off the reserve also covers. Presolve grants itself the same share
/// for the violations its own reductions may introduce.
pub(crate) const ROW_BUDGET_SHARE: f64 = 0.5;

/// The tolerance a quantity of magnitude `magnitude` is held to under an
/// absolute tolerance `tol`: `tol` itself, but never finer than the
/// round-off of computing the quantity. This is the contract: every row and
/// bound check in the crate is this rule, applied through [`outside`].
pub(crate) fn row_tolerance(tol: f64, magnitude: f64) -> f64 {
    tol.max(ROUNDOFF_FLOOR * magnitude)
}

/// The larger finite magnitude of two bounds (zero when both are infinite).
pub(crate) fn bound_magnitude(lo: f64, hi: f64) -> f64 {
    [lo.abs(), hi.abs()]
        .into_iter()
        .filter(|m| m.is_finite())
        .fold(0.0, f64::max)
}

/// The contract on a var's bounds: [`row_tolerance`] at the magnitude of
/// the bound.
pub(crate) fn bound_tolerance(feasibility: f64, lo: f64, hi: f64) -> f64 {
    row_tolerance(feasibility, bound_magnitude(lo, hi))
}

/// Whether `value` lies outside `[lo, hi]` by more than `tol`: the one
/// comparison of the contract, written the way the engine tests a basic var
/// against its bounds, so that every check computes the same bits from the
/// same numbers (`value - hi > tol` is not the same test in `f64`). A NaN is
/// outside.
#[inline]
pub(crate) fn outside(value: f64, lo: f64, hi: f64, tol: f64) -> bool {
    value.is_nan() || value < lo - tol || value > hi + tol
}

/// How far `value` lies outside `[lo, hi]` (zero inside, infinite for a
/// NaN), for reports.
pub(crate) fn excursion(value: f64, lo: f64, hi: f64) -> f64 {
    if value.is_nan() {
        f64::INFINITY
    } else {
        (lo - value).max(value - hi).max(0.0)
    }
}

/// Tolerance within which a row's slack counts as at or within its bounds:
/// the engine's share of the contract in the row's equilibrated units, at
/// the row's `magnitude` (`|b| + Σ|a_j x_j|`) at the current values. The
/// row's factor is a power of two, so this is exactly the user's tolerance
/// on the unscaled row. Refreshed whenever the rows are checked, since the
/// round-off floor follows the values.
pub(crate) fn slack_tol(feasibility: f64, row_scale: f64, magnitude: f64) -> f64 {
    ROW_BUDGET_SHARE * row_tolerance(feasibility * row_scale, magnitude)
}

/// How far a var with scaled coefficient `coeff` in a row of factor
/// `row_scale` may sit off a bound before snapping it there (rounding an
/// integer var, reporting a fixed var at its value) would move the row by
/// more than the reserve of its contract: the reserve over the coefficient.
pub(crate) fn snap_budget(feasibility: f64, row_scale: f64, coeff: f64) -> f64 {
    (1.0 - ROW_BUDGET_SHARE) * feasibility * row_scale / coeff.abs()
}

/// Tolerance within which a structural var (column factor `scale`, scaled
/// bounds `min..max`) counts as at or within its bounds: the user's
/// tolerance in user units, `feasibility / scale` in the engine's units, but
/// never more than any of its rows can absorb when the value is snapped to
/// the bound (`row_budget`, the smallest [`snap_budget`] over the var's
/// rows; infinite for a var in no row), and never finer than the round-off
/// of the bound. One rule for every var: the cap is what keeps a big-M row
/// within contract when its binary is rounded, and a fixed var's report at
/// its value within the rows presolve substituted it out of.
pub(crate) fn structural_tol(
    scale: f64,
    min: f64,
    max: f64,
    feasibility: f64,
    row_budget: f64,
) -> f64 {
    row_tolerance(
        (feasibility / scale).min(row_budget),
        bound_magnitude(min, max),
    )
}

/// Tolerance within which a reduced cost counts as zero, given the magnitude
/// `|c_j| + sum |a_ij y_i|` of the terms it was computed from.
pub(crate) fn dual_tol(magnitude: f64) -> f64 {
    EPS.max(ROUNDOFF_FLOOR * magnitude)
}

/// How often (in simplex iterations) the primal/dual loops in `optimize` and
/// `restore_feasibility` check the deadline and emit a progress `debug!` log.
/// Checking every iteration would make the deadline check itself a
/// significant fraction of the per-iteration cost on easy problems; checking
/// too rarely would make a time limit overshoot by a visible amount on hard
/// ones. 1000 keeps the check overhead negligible while still bounding the
/// worst-case overshoot to about a thousand pivots.
pub(crate) const DEADLINE_CHECK_INTERVAL: u64 = 1000;

/// Threshold-pivoting stability coefficient passed to [`crate::lu::lu_factorize`] for
/// every LU (re)factorization the simplex performs: a candidate pivot is
/// accepted only if its magnitude is at least this fraction of the column's
/// largest eligible entry. 0.1 is the standard textbook default for
/// Gilbert-Peierls sparse LU (see `lu_factorize`'s doc reference) — it
/// balances numerical stability (higher would refuse more marginal pivots,
/// at the cost of extra fill-in) against sparsity (lower risks amplifying
/// rounding error through a poorly-conditioned pivot).
pub(crate) const LU_STABILITY_THRESHOLD: f64 = 0.1;

pub(crate) fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

/// Whether `val` sits at `bound` within `tol` (the var's own tolerance, see
/// [`super::Solver::var_tols`]).
#[inline]
pub(crate) fn at_bound(val: f64, bound: f64, tol: f64) -> bool {
    (val - bound).abs() <= tol
}
