//! The engine's tolerance model: the one contract every layer holds a point
//! to, and the few constants that describe what the arithmetic can resolve.
//!
//! The contract (ARCHITECTURE §7.1): a value may leave its range by the user's
//! absolute `feasibility` tolerance, or by the round-off of the quantities it
//! was computed from if that is larger. [`row_tolerance`] is that rule,
//! [`bound_tolerance`] its instance for a var's bound, and [`outside`] the
//! one comparison every check makes. The engine holds each row to its
//! contract less the round-off of evaluating it ([`slack_tol`]) and each var
//! to its whole contract, an integer var its bounds fix to its value exactly
//! ([`structural_tol`]), and reports the values as it computed them, so the
//! point it reports passes the boundary's evaluation (the MIP candidate
//! guard, the pure-LP check), which sums each row in compensated arithmetic
//! ([`implied_slack`]) and so does not depend on the engine's order. Where
//! the dual simplex's widening moves a bound by its var's tolerance, the
//! bound holds that tolerance ([`band_edge`]) and the side keeps none, or for
//! a slack only the evaluation margin ([`slack_margin`]). Everything else
//! here is a property of `f64` and of the factorization, not of the contract.

use super::CDouble;
use crate::{ComparisonOp, CsVec};

/// The absolute part of the reduced-cost tolerance ([`dual_tol`]), in the
/// engine's scaled units, where quantities are of order one and it is far
/// above round-off, and in the user's, where the round-off floor of the
/// tolerance and the refinement of the multipliers cover what a solve gets
/// wrong.
pub const EPS: f64 = 1e-10;

/// Relative round-off allowance: about a hundred ulps of `f64`. A quantity
/// computed from terms of magnitude `m` cannot be trusted below
/// `ROUNDOFF_FLOOR * m`, so no tolerance on it is tighter than that, but the
/// bound tolerance of an integer var its bounds fix, which is zero: the
/// engine holds that var exactly at its value ([`structural_tol`]). This is
/// the same allowance the crate's own contract tests grant a row.
pub(crate) const ROUNDOFF_FLOOR: f64 = 1e-14;

/// A column of the basis whose largest eligible pivot, in the LU
/// factorization, is below this fraction of the column's largest transformed
/// entry is treated as singular: the eligible part is then within the
/// round-off of the elimination, so the column is dependent on the previous
/// ones as far as `f64` can tell. Relative, so that the test does not depend
/// on the units the column is written in; an ill-conditioned but non-singular
/// basis is judged by the verification of the values it produces, not by its
/// pivots.
pub(crate) const LU_SINGULAR_REL: f64 = ROUNDOFF_FLOOR;

/// The pivot tolerance of the dual ratio test after `updates` basis changes
/// on the current factorization (the eta file's length), in the engine's
/// scaled units: an entry of the pivot row bounds the dual step and is a
/// candidate only if its magnitude exceeds this. The row is computed through
/// the factorization and every update since, each adding its round-off to
/// what a small entry can be made of, so the older the factorization, the
/// larger an entry must be to be trusted. Absolute, since rows and columns are
/// scaled to entries near one; what a relative floor could not tell from
/// round-off, an entry small beside the rest of its row, the growth test on
/// the chosen pivot catches ([`PIVOT_GROWTH_TOL`]). A genuine entry below it
/// enters only as a last resort, once a refinement shows it genuine
/// (`Solver::genuine_entry_step`).
pub(crate) fn dual_pivot_tolerance(updates: usize) -> f64 {
    if updates < 10 {
        1e-9
    } else if updates < 20 {
        3e-8
    } else {
        1e-6
    }
}

/// The pivot tolerance of the primal ratio test after `updates` basis
/// changes on the current factorization, in the engine's scaled units, for
/// the reasons of [`dual_pivot_tolerance`]: an entry of the pivot column
/// bounds the step and can be the pivot only if its magnitude exceeds this.
/// The step passes the row of a smaller entry, and a basic var it leaves
/// outside its bounds beyond its tolerance has its working bound shifted
/// (`Solver::correct_primal`). A step nothing above it bounds, toward an
/// infinite bound, refines its column, whose entries beyond their measured
/// error bound it (`Solver::refine_pivot_col`).
pub(crate) fn primal_pivot_tolerance(updates: usize) -> f64 {
    if updates < 10 {
        1e-9
    } else if updates < 20 {
        1e-8
    } else {
        1e-7
    }
}

/// The growth test of the dual ratio test's chosen pivot: a pivot whose
/// magnitude is at most this fraction of the pivot row's largest entry over
/// the vars whose bounds do not fix them (that entry rounded up to a power of
/// two) is small, however far above the pivot tolerance it lies. Every entry
/// of the row carries the round-off of the one solve it comes from, of the
/// order of the largest, so the pivot may be made of it, and dividing by it
/// grows the basis inverse and the reduced costs a billionfold. The row is
/// refined and its entries summed again in compensated arithmetic before such
/// a pivot is passed over (`Solver::choose_entering_col_dual`), and it enters
/// only as the first of the dual simplex's last resorts, genuine on the
/// refined row (`Solver::genuine_entry_step`).
pub(crate) const PIVOT_GROWTH_TOL: f64 = 1e-9;

/// Relative disagreement between the pivot element computed through the
/// pivot row (`row_coeffs[col]`) and through the pivot column
/// (`col_coeffs[row]`) above which the factorization is in doubt (see
/// `PivotChoice::Inconsistent`): the two are the same number in exact
/// arithmetic and agree to about `1e-12` at a legitimate pivot, while a pivot
/// computed through an inaccurate factorization disagrees by far more.
pub(crate) const PIVOT_AGREEMENT_TOL: f64 = 1e-7;

/// Upper bound on the number of primal/dual phase alternations
/// [`super::Solver::run_phases`] performs before giving up loudly. A round
/// verifies the values before its primal phase, so a solve whose primal
/// phase pivots takes two rounds, and a third is rare, but for an excursion
/// on refined values, which takes a round each (`Solver::refined_first`);
/// each round after the first cuts the primal ratio test's tolerance
/// ([`LOOP_HARRIS_REDUCTION`]). Hitting this bound means the phases cannot
/// agree on a verified state, which must not be reported as an optimum; an
/// excursion that hits it ends at the answer it started from.
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
/// `m = |rhs| + Σ|a_j x_j|`); `None` if it holds, as a row every point
/// satisfies does ([`trivial_row_verdict`]). This is the check of a row
/// as the user wrote it, for rows the engine does not hold itself (see
/// `Solver::first_violated_row`), and presolve's verdict on a row whose vars
/// it fixed, so that the two are the same evaluation: in compensated
/// arithmetic ([`implied_slack`]), whatever the order of the terms.
pub(crate) fn row_violation(
    coeffs: &CsVec,
    op: ComparisonOp,
    rhs: f64,
    value: impl Fn(usize) -> f64,
    feasibility: f64,
) -> Option<f64> {
    if trivial_row_verdict(coeffs, op, rhs, feasibility) == Some(true) {
        return None;
    }
    let (implied, magnitude) =
        implied_slack(rhs, coeffs.iter().map(|(v, &coeff)| (coeff, value(v))));
    let (smin, smax) = slack_bounds(op);
    outside(implied, smin, smax, row_tolerance(feasibility, magnitude))
        .then(|| excursion(implied, smin, smax))
}

/// `2^996`: a factor below it never overflows Dekker's split in
/// [`CDouble::add_product`], which multiplies it by `2^27 + 1`.
const SPLIT_LIMIT: f64 = 6.696928794914171e299;

/// The slack `rhs - Σ a_j x_j` a row implies over its `terms` `(a_j, x_j)`,
/// with the magnitude `|rhs| + Σ|a_j x_j|` it was computed from: the products
/// exact and summed in compensated arithmetic ([`CDouble`]), so that the
/// result is the exact one to about an ulp of itself, whatever the order of
/// the terms (the low parts add some `n^2 2^-106` of the magnitude). A product
/// with a factor beyond [`SPLIT_LIMIT`] is rounded instead, as in a plain sum.
pub(crate) fn implied_slack(rhs: f64, terms: impl Iterator<Item = (f64, f64)>) -> (f64, f64) {
    let mut implied = CDouble::from(rhs);
    let mut magnitude = rhs.abs();
    for (coeff, value) in terms {
        if coeff.abs() < SPLIT_LIMIT && value.abs() < SPLIT_LIMIT {
            implied.add_product(-coeff, value);
        } else {
            implied.add(-coeff * value);
        }
        magnitude += (coeff * value).abs();
    }
    (implied.value(), magnitude)
}

/// A bound on the error of [`implied_slack`] over `terms` products at
/// `magnitude`, at a slack of magnitude `slack`: `4 u s + 4 (n + 2)^2 u^2 m`
/// (`u = 2^-53`). Its last rounding errs by `u |s|`, and the `2n` low parts it
/// sums in `f64`, two per product, by `(2n - 1)(n + 1) u^2 m` to first order
/// (the `Dot2` bound of Ogita, Rump and Oishi 2005, for that count); the bound
/// holds both at least twice over, which covers the higher-order terms and the
/// round-off of `m` itself. The error grows with the slack, but slower, so a
/// caller may take the slack at an edge it lies past
/// (`Solver::leaving_slack_cut`). Products it rounds (a factor past
/// [`SPLIT_LIMIT`]) aside.
pub(crate) fn implied_slack_error(slack: f64, magnitude: f64, terms: usize) -> f64 {
    let n = terms as f64 + 2.0;
    4.0 * UNIT_ROUNDOFF * slack + 4.0 * n * n * UNIT_ROUNDOFF * UNIT_ROUNDOFF * magnitude
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
pub(crate) const UNIT_ROUNDOFF: f64 = f64::EPSILON / 2.0;

/// Tolerance within which a row counts as satisfied: its contract,
/// [`row_tolerance`] of the absolute `tol` at the row's `magnitude`
/// (`|b| + Σ|a_j x_j|` at the current values), less the round-off of
/// evaluating it, `(terms + 2) 2^-53 magnitude` for a row of `terms`
/// nonzeros: the plain `f64` sum the engine evaluates the row with, and the
/// boundary's compensated one ([`implied_slack`]). The margin is at most half
/// the contract; where that caps it, the engine sums the row in compensated
/// arithmetic too ([`plain_sum_resolves`]). The engine passes the user's
/// tolerance times the row's factor, a power of two, so this is exactly the
/// tolerance on the unscaled row; presolve passes the user's. The engine
/// refreshes a slack's tolerance whenever it checks the rows, since the
/// round-off follows the values.
pub(crate) fn slack_tol(tol: f64, magnitude: f64, terms: usize) -> f64 {
    row_tolerance(tol, magnitude) - slack_margin(tol, magnitude, terms)
}

/// The evaluation margin [`slack_tol`] takes off a row's contract.
pub(crate) fn slack_margin(tol: f64, magnitude: f64, terms: usize) -> f64 {
    ((terms as f64 + 2.0) * UNIT_ROUNDOFF * magnitude).min(0.5 * row_tolerance(tol, magnitude))
}

/// Whether the round-off of the engine's plain `f64` sum of a row fits the
/// evaluation margin [`slack_margin`] leaves it, which is at most half the
/// contract. A row of more than 43 terms at its round-off floor does not,
/// and the engine sums it in compensated arithmetic ([`implied_slack`]).
pub(crate) fn plain_sum_resolves(tol: f64, magnitude: f64, terms: usize) -> bool {
    (terms as f64 + 2.0) * UNIT_ROUNDOFF * magnitude <= 0.5 * row_tolerance(tol, magnitude)
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

/// The farthest value past `bound`, above it (`up`) or below, within `tol`
/// of it: `bound ± tol`, less the round-off of that sum.
pub(crate) fn band_edge(bound: f64, tol: f64, up: bool) -> f64 {
    let mut edge = if up { bound + tol } else { bound - tol };
    while (edge - bound).abs() > tol {
        edge = if up { edge.next_down() } else { edge.next_up() };
    }
    edge
}

/// Tolerance within which a reduced cost counts as zero: [`EPS`] in the
/// engine's units and in the user's, the engine's reduced cost being `unit`
/// times the user's (`s_j` for a structural var; for a slack, per unit of its
/// row's activity divided by the row's largest coefficient as written,
/// [`super::slack_dual_unit`]), floored at the round-off of the terms
/// `|c_j| + sum |a_ij y_i|` it was computed from (`magnitude`, engine units).
/// In the engine's units alone, a cost scaling shrinks below `EPS` would not
/// count: `max x` s.t. `2e10 x <= 2e10` would end at `x = 0`.
pub(crate) fn dual_tol(magnitude: f64, unit: f64) -> f64 {
    (EPS * unit.min(1.0)).max(ROUNDOFF_FLOOR * magnitude)
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
/// rounding error through a poorly-conditioned pivot). A solver starts with
/// it and raises it once the two computations of a pivot element disagree
/// (see `Solver::raise_lu_threshold`).
pub(crate) const LU_STABILITY_THRESHOLD: f64 = 0.1;

/// The highest threshold of the factorization: the one a pivot disagreement
/// raises the solver's to, and the one the verification falls back on when
/// the values recomputed through a fresh factorization at the solver's still
/// miss the rows (see `Solver::verify_primal`). A pivot at least half its
/// column's largest entry bounds the growth of the factors, which is what a
/// solve through them errs by.
pub(crate) const LU_STABLE_THRESHOLD: f64 = 0.5;

/// Whether two values agree to [`EPS`]: a comparison for the tests.
#[cfg(test)]
pub(crate) fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}
