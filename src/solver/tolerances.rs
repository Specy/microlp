//! The engine's tolerance model: the one contract every layer holds a point
//! to, and the few constants that describe what the arithmetic can resolve.
//!
//! The contract (ARCHITECTURE §7): a value may leave its range by the user's
//! absolute `feasibility` tolerance, or by the round-off of the quantities it
//! was computed from if that is larger. [`row_tolerance`] is that rule,
//! [`bound_tolerance`] its instance for a var's bound, and [`outside`] the
//! one comparison every check makes. The engine holds each row to its
//! contract less the round-off of evaluating it ([`slack_tol`]) and each var
//! to its whole contract, an integer var its bounds fix to its value exactly
//! ([`structural_tol`]), and reports the values as it computed them, so the
//! point it reports passes the boundary's evaluation (the MIP candidate
//! guard, the pure-LP validation) and one in another order. Everything else
//! here is a property of `f64` and of the factorization, not of the
//! contract.

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
/// from the round-off of computing it, and two computations of the same entry
/// (through the row and through the column) agree to about this much. Tighter
/// than [`ROUNDOFF_FLOOR`], which allows for a sum over a row's many terms.
pub(crate) const ENTRY_ROUNDOFF: f64 = 1e-15;

/// A column of the basis whose largest eligible pivot, in the LU
/// factorization, is below this fraction of the column's largest transformed
/// entry is treated as singular: the eligible part is then within the
/// round-off of the elimination, so the column is dependent on the previous
/// ones as far as `f64` can tell. Relative, so that the test does not depend
/// on the units the column is written in; an ill-conditioned but non-singular
/// basis is judged by the verification of the values it produces, not by its
/// pivots.
///
/// A pivot element of the ratio tests within this fraction of the scale of
/// its row and column must have its two computations agree relatively, not
/// just within their round-off (see `Solver::pivots_agree`), since at this
/// size a round-off entry agrees with anything of its own size.
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

/// The smallest pivot element the dual simplex takes as its last resort, in
/// the engine's scaled units, where rows and columns are equilibrated to
/// entries near one. The regular floors ([`ENTRY_ROUNDOFF`],
/// [`PIVOT_REL_TOL`]) are relative to the row's largest entry, so a huge
/// entry elsewhere in the row (a fixed var's `1e17` beside a genuine `1`) can
/// make every entry that could repair it "round-off". When such a row does
/// not prove the model infeasible and no other row is left (see
/// `Solver::dual_phase`), the dual simplex pivots on an entry the regular
/// test refused if the entry is at least this; below it an entry is
/// indistinguishable from the round-off of a solve through a basis of any
/// condition, so no pivot is taken on it at all.
pub(crate) const INSTABLE_PIVOT_TOL: f64 = 1e-10;

/// Relative disagreement between the pivot element computed through the
/// pivot row (`row_coeffs[col]`) and through the pivot column
/// (`col_coeffs[row]`) above which the element is in doubt (see
/// `PivotChoice::Inconsistent`): the two are the same number in exact
/// arithmetic and agree to about `1e-12` at a legitimate pivot, while a pivot
/// on round-off noise disagrees by tens of percent.
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

/// The factor by which each round of [`super::Solver::run_phases`] after the
/// first cuts the tolerance the primal ratio test lets a basic var pass its
/// bound by (the Harris allowance, `0.01^round` of the var's tolerance). A
/// round ends in a verification that recomputes the values from the data, and
/// a Harris step may leave a var exactly at its tolerance, where the
/// recomputation finds it an ulp past: without the cut, the dual phase undoes
/// the step, the primal phase redoes it, and the rounds run out. The first
/// round keeps the whole tolerance, so the Harris protection stays where
/// scaling is worst. A var the earlier rounds left past its bound by more
/// than the cut allowance (within its whole tolerance) admits no step toward
/// it: the ratio test's limit is then zero, never negative (see
/// `choose_pivot`).
pub(crate) const LOOP_HARRIS_REDUCTION: f64 = 0.01;

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
/// larger of the two would give a var in `[0, 1e10]` a tolerance of `1e-4` at
/// zero). An infinite bound compares with nothing; its tolerance is the
/// absolute part alone.
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

/// The unit round-off of `f64`, `2^-53`.
const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// Tolerance within which a row counts as satisfied: its contract,
/// [`row_tolerance`] of the absolute `tol` at the row's `magnitude`
/// (`|b| + Σ|a_j x_j|` at the current values), less the round-off of
/// evaluating it, `(terms + 2) 2^-53 magnitude` for a row of `terms`
/// nonzeros: the plain `f64` sum the engine evaluates the row with, and one
/// more evaluation in another order, the boundary's or a user's. The margin
/// is at most half the contract. The engine passes the user's tolerance times
/// the row's factor, a power of two, so this is exactly the tolerance on the
/// unscaled row; presolve passes the user's. The engine refreshes a slack's
/// tolerance whenever it checks the rows, since the round-off follows the
/// values.
pub(crate) fn slack_tol(tol: f64, magnitude: f64, terms: usize) -> f64 {
    let contract = row_tolerance(tol, magnitude);
    let margin = ((terms as f64 + 2.0) * UNIT_ROUNDOFF * magnitude).min(0.5 * contract);
    contract - margin
}

/// Tolerances within which a structural var (column factor `scale`, scaled
/// bounds `min..max`) counts as within each bound: the user's tolerance in
/// user units, `feasibility / scale` in the engine's units, floored at the
/// round-off of that bound ([`bound_tolerance`]). The reported value is the
/// engine's times `scale`, a power of two, so it meets the contract exactly
/// when the engine's does.
///
/// An `integer` var its bounds fix (`min == max`: a branch, an edit,
/// presolve) gets none: the engine holds it exactly at its value, pivoting
/// it out of the basis onto that value when it is basic off it. A row
/// presolve rewrote holds as written only at an exactly integral binary, and
/// the search cannot branch on a var its node fixes when that row rejects
/// the point.
pub(crate) fn structural_tol(
    scale: f64,
    min: f64,
    max: f64,
    feasibility: f64,
    integer: bool,
) -> BoundTols {
    if integer && min == max {
        return BoundTols::uniform(0.0);
    }
    let base = feasibility / scale;
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
/// updated to. The weights are only a pricing measure, so the floor decides
/// nothing about feasibility; it keeps a weight the update has driven to zero
/// or below (from the underestimate a basis load resets every weight to) from
/// making its row dominate the choice or never be chosen.
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
