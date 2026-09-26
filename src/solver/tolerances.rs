//! The engine's tolerance model: the one contract every layer holds a point
//! to, and the few constants that describe what the arithmetic can resolve.
//!
//! The contract (ARCHITECTURE §7): a value may leave its range by the user's
//! absolute `feasibility` tolerance, or by the round-off of the quantities it
//! was computed from if that is larger. [`row_tolerance`] is that rule,
//! [`bound_tolerance`] its instance for a var's bound, and [`outside`] the
//! one comparison every check makes. The engine holds each row to
//! [`ROW_BUDGET_SHARE`] of the contract ([`slack_tol`]) and caps a var's
//! bound tolerance by what its rows can absorb ([`structural_tol`]), so that
//! the point it reports passes the same evaluation at the boundary (the MIP
//! candidate guard, the pure-LP validation) with the other half to spare for
//! a snapped integer or fixed value and for evaluating the row in another
//! order. Everything else here is a property of `f64` and of the
//! factorization, not of the contract.

use crate::{ComparisonOp, CsVec};

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
/// entry (through the row and through the column) agree to about this much. Tighter than [`ROUNDOFF_FLOOR`], which allows
/// for a sum over a row's many terms.
pub(crate) const ENTRY_ROUNDOFF: f64 = 1e-15;

/// A column of the basis whose largest eligible pivot, in the LU
/// factorization, is below this fraction of the column's largest
/// transformed entry is treated as singular: the eligible part is then
/// within the round-off of the elimination that produced it, i.e. the column
/// is dependent on the previous ones as far as `f64` can tell. Relative to
/// the column's own scale rather than absolute, so the test does not depend
/// on the units the column happens to be written in; an ill-conditioned but
/// non-singular basis is judged by the verification of the values it
/// produces, not by a threshold on its pivots.
///
/// The ratio tests hold a pivot element within this fraction of the scale
/// of its row and column to more than the factorization can resolve: its
/// two computations must agree relatively, not just within their round-off
/// (see `Solver::pivots_agree`), since at this size a round-off entry agrees
/// with anything of its own size.
pub(crate) const LU_SINGULAR_REL: f64 = ROUNDOFF_FLOOR;

/// Relative pivot tolerance of both ratio tests: an entry of the pivot row
/// (dual simplex) or pivot column (primal simplex) is a preferred pivot only
/// if its magnitude is at least this fraction of the largest entry scanned.
/// A pivot seven orders of magnitude below its neighbours amplifies the
/// round-off of the update by the same factor, so it is taken only when no
/// larger entry will do: the largest entry of a dual row is always a
/// candidate, and in the primal test a smaller entry still bounds the step
/// (its basic var is held to its tolerance like any other) and is the pivot
/// when it is what blocks the step. So this never turns a feasible model
/// infeasible or a bounded one unbounded, and never lets a step leave a
/// basic var outside its tolerance.
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
/// values verified against the rows, so a second round is already rare, and
/// each round after the first cuts the primal ratio test's tolerance
/// ([`LOOP_HARRIS_REDUCTION`]); hitting this bound means the phases cannot
/// agree on a verified state, which must not be reported as an optimum.
pub(crate) const MAX_PHASE_ROUNDS: usize = 8;

/// The factor by which each round of [`super::Solver::run_phases`] after
/// the first cuts the tolerance the primal ratio test lets a basic var pass
/// its bound by (the Harris allowance, `0.01^round` of the var's tolerance):
/// SoPlex's "to avoid shifts we restrict tolerances in the ratio test",
/// which cuts its ratio tester's delta a hundredfold each time its
/// algorithms loop (`SPxSolverBase::solve`, src/soplex/spxsolve.hpp). A
/// round ends in a verification that recomputes the values from the data,
/// and a Harris step may leave a var exactly at its tolerance, where the
/// recomputation finds it an ulp past: the dual phase then undid the step,
/// the primal phase redid it, and the rounds ran out (an equality
/// `x2 + 1e-7 x3 = 1.0000001` ending its step at its tolerance and
/// recomputed `1.15e-12` over it). The first round keeps the whole
/// tolerance, so the Harris protection stays where scaling is worst; the
/// allowance was once cut on every step by the round-off of recomputing the
/// var instead, which left rows above about `5e6` in magnitude with none at
/// all (C3-10 of the solver review). SoPlex cuts at every pass of either
/// algorithm once one has ended unverified (its `loopCount`); a round here, a
/// dual and a primal pass, cuts once. A var the earlier rounds left past its
/// bound by more than the cut allowance (within its whole tolerance) admits
/// no step toward it: the ratio test's limit is then zero, never negative
/// (see `choose_pivot`).
pub(crate) const LOOP_HARRIS_REDUCTION: f64 = 0.01;

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

/// The contract on one bound of a var: [`row_tolerance`] at the magnitude of
/// that bound. Each bound has its own, since comparing a value with a bound
/// resolves it to the round-off of that bound, not of the other one (the
/// larger of the two gave a var in `[0, f64::MAX]` a tolerance of 1e294 at
/// zero, and a var in `[0, 1e10]` one of 1e-4). An infinite bound compares
/// with nothing; its tolerance is the absolute part alone.
pub(crate) fn bound_tolerance(feasibility: f64, bound: f64) -> f64 {
    if bound.is_finite() {
        row_tolerance(feasibility, bound.abs())
    } else {
        feasibility
    }
}

/// Whether `value` violates the var bounds `[lo, hi]` beyond the contract:
/// [`outside`] against each bound with that bound's own tolerance
/// ([`bound_tolerance`]).
pub(crate) fn outside_bounds(value: f64, lo: f64, hi: f64, feasibility: f64) -> bool {
    outside(value, lo, f64::INFINITY, bound_tolerance(feasibility, lo))
        || outside(
            value,
            f64::NEG_INFINITY,
            hi,
            bound_tolerance(feasibility, hi),
        )
}

/// A var's tolerances at its lower and upper bound, in the engine's units
/// (see [`structural_tol`]; a slack's two are equal).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BoundTols {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

impl BoundTols {
    pub(crate) fn uniform(tol: f64) -> Self {
        Self { lo: tol, hi: tol }
    }

    /// The tolerance at the bound a value moving up (`true`) or down meets.
    #[inline]
    pub(crate) fn toward(&self, up: bool) -> f64 {
        if up {
            self.hi
        } else {
            self.lo
        }
    }
}

/// Whether `value` lies outside `[lo, hi]` by more than `tol`: the one
/// comparison of the contract, written the way the engine tests a basic var
/// against its bounds, so that every check computes the same bits from the
/// same numbers (`value - hi > tol` is not the same test in `f64`). A value
/// that is not finite is outside whatever the range: the contract is about
/// points, and a point is finite (an infinite value would otherwise pass an
/// infinite bound, and its row an infinite tolerance).
#[inline]
pub(crate) fn outside(value: f64, lo: f64, hi: f64, tol: f64) -> bool {
    !value.is_finite() || value < lo - tol || value > hi + tol
}

/// The range a row of sense `op` allows for `rhs - activity`, i.e. the
/// bounds of its slack var: `[0, ∞)` for `≤`, `(-∞, 0]` for `≥`, `[0, 0]`
/// for `=`.
pub(crate) fn slack_bounds(op: ComparisonOp) -> (f64, f64) {
    match op {
        ComparisonOp::Le => (0.0, f64::INFINITY),
        ComparisonOp::Ge => (f64::NEG_INFINITY, 0.0),
        ComparisonOp::Eq => (0.0, 0.0),
    }
}

/// How far the user-unit row `coeffs op rhs` is from holding at the point
/// `value` (a var's value by index), if it violates the contract there
/// (`row_tolerance(feasibility, m)` at the row's magnitude
/// `m = |rhs| + Σ|a_j x_j|`); `None` if it holds. This is the check of a row
/// as the user wrote it, for rows the engine does not hold itself (see
/// `Solver::first_violated_row`), and presolve's verdict on a row whose vars
/// it fixed, so that the two are the same evaluation.
pub(crate) fn row_violation(
    coeffs: &CsVec,
    op: ComparisonOp,
    rhs: f64,
    value: impl Fn(usize) -> f64,
    feasibility: f64,
) -> Option<f64> {
    let mut lhs = 0.0;
    let mut magnitude = rhs.abs();
    for (v, &coeff) in coeffs.iter() {
        let term = coeff * value(v);
        lhs += term;
        magnitude += term.abs();
    }
    let (smin, smax) = slack_bounds(op);
    let implied = rhs - lhs;
    outside(implied, smin, smax, row_tolerance(feasibility, magnitude))
        .then(|| excursion(implied, smin, smax))
}

/// What a row says before any point is looked at: `Some(false)` when no
/// point can satisfy it (a coefficient or right-hand side that is not a
/// number, an infinite coefficient, an infinite right-hand side on the
/// side the row constrains, an empty row whose right-hand side the contract
/// does not allow at activity zero), `Some(true)` when every point does (an
/// infinite right-hand side on the side the row leaves open, an empty row
/// within the contract), and `None` when its terms decide. An empty row is
/// judged by the same comparison as every other row: its activity is
/// exactly zero, and its magnitude is that of its right-hand side.
pub(crate) fn trivial_row_verdict(
    coeffs: &CsVec,
    op: ComparisonOp,
    rhs: f64,
    feasibility: f64,
) -> Option<bool> {
    if rhs.is_nan() || coeffs.data().iter().any(|c| !c.is_finite()) {
        return Some(false);
    }
    let (smin, smax) = slack_bounds(op);
    if rhs.is_infinite() {
        // `rhs - activity` is infinite for every finite point: inside the
        // slack range only if that range is unbounded on rhs's side.
        return Some(if rhs > 0.0 {
            smax == f64::INFINITY
        } else {
            smin == f64::NEG_INFINITY
        });
    }
    if coeffs.indices().is_empty() {
        return Some(!outside(
            rhs,
            smin,
            smax,
            row_tolerance(feasibility, rhs.abs()),
        ));
    }
    None
}

/// How far `value` lies outside `[lo, hi]` (zero inside, infinite for a
/// value that is not finite, as [`outside`] has it), for reports.
pub(crate) fn excursion(value: f64, lo: f64, hi: f64) -> f64 {
    if !value.is_finite() {
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

/// Tolerances within which a structural var (column factor `scale`, scaled
/// bounds `min..max`) counts as at or within each bound: the user's
/// tolerance in user units, `feasibility / scale` in the engine's units, but
/// never more than any of its rows can absorb when the value is snapped to
/// the bound (`row_budget`, the smallest [`snap_budget`] over the var's
/// rows; infinite for a var in no row), and never finer than the round-off
/// of that bound ([`bound_tolerance`]). One rule for every var: the cap is
/// what keeps a big-M row within contract when its binary is rounded, and a
/// fixed var's report at its value within the rows presolve substituted it
/// out of.
pub(crate) fn structural_tol(
    scale: f64,
    min: f64,
    max: f64,
    feasibility: f64,
    row_budget: f64,
) -> BoundTols {
    let base = (feasibility / scale).min(row_budget);
    BoundTols {
        lo: bound_tolerance(base, min),
        hi: bound_tolerance(base, max),
    }
}

/// Tolerance within which a reduced cost counts as zero, given the magnitude
/// `|c_j| + sum |a_ij y_i|` of the terms it was computed from.
pub(crate) fn dual_tol(magnitude: f64) -> f64 {
    EPS.max(ROUNDOFF_FLOOR * magnitude)
}

/// The smallest value a dual steepest-edge weight `||e_r^T B^-1||^2` is
/// updated to (HiGHS's `kMinDualSteepestEdgeWeight`,
/// highs/simplex/SimplexConst.h, applied in
/// `HEkk::updateDualSteepestEdgeWeights`). The weights are only a pricing
/// measure, so the floor decides nothing about feasibility; it keeps a weight
/// the update has driven to zero or below (from the underestimate a basis
/// load resets every weight to) from making its row dominate the choice or
/// never be chosen.
pub(crate) const MIN_DUAL_EDGE_WEIGHT: f64 = 1e-4;

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
