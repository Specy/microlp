//! Turning a user problem into the engine's scaled representation: power-of-two
//! row and column scaling, per-row preparation, and the rule that picks a
//! non-basic variable's starting bound.
//!
//! Every factor is a power of two, so scaling and unscaling change only an
//! exponent and no user datum is ever rounded.

use crate::{ComparisonOp, CsVec, Error};

use super::tolerances::{slack_bounds, trivial_row_verdict, valid_rhs};

/// A variable bound at or beyond this magnitude on its side of infinity (a
/// lower bound at or below `-INFINITE_BOUND`, an upper bound at or above it)
/// stands for infinity, and is made infinite when the var is added to a
/// problem ([`entry_bounds`], called by `Problem`'s `internal_add_var`, which
/// stores the bounds so): the one data-level infinity of the crate. It is
/// made infinite there, not only kept out of the starting vertex, since a
/// basis reload or a ratio test would otherwise put a var at such a bound,
/// and one of `f64::MAX` overflows the var's rows.
///
/// The value follows from the contract: a row is held to the round-off of its
/// magnitude (`row_tolerance`, `1e-14` of `|b| + Σ|a_j x_j|`), so beside a
/// var at a bound `B` with a coefficient `a` it is held to about
/// `1e-14 |a| B` only, and with `|a|` near one, from 2^52 on the spacing of
/// `f64` is one or more, so no fraction of the row's data survives. A huge
/// bound written for a missing one, as `f64::MAX`, `f32::MAX` or `i64::MAX`
/// commonly is, would then change the answer as soon as a var sits at it;
/// bounds from about `1e15` up to the threshold still can. The price: a bound
/// of 2^52 or more that is meant as a bound is none here, so `max x` with
/// `x ≤ 5e15` is `Unbounded`, and a solution can lie beyond such a bound
/// (both pinned in `src/tests/regressions.rs`). The threshold is on the bound
/// as given, not on `|a| B`, so a var whose rows could resolve a bound beyond
/// it loses the bound all the same (`max x` s.t. `1e-8 x ≤ 1e9`, `x ≤ 1e16`
/// gives `x = 1e17`). It also ends the step over which an entry the primal
/// ratio test cannot resolve must be negligible for a ray
/// (`Solver::unresolved_entries_leave_a_ray`).
pub(crate) const INFINITE_BOUND: f64 = 4_503_599_627_370_496.0; // 2^52

/// Power-of-two row equilibration keeps the largest structural coefficient
/// near one without rounding its mantissa. If scaling would overflow the RHS,
/// leave the row unchanged and let the solver report any resulting numerical
/// failure explicitly.
fn equilibration_scale(coeffs: &CsVec, rhs: f64, is_fixed: impl Fn(usize) -> bool) -> f64 {
    // A fixed var contributes a constant to the row; its coefficient says
    // nothing about the row's scale and, left in, would let a large constant
    // term shrink the entries of the vars that can actually move.
    let max_coeff = coeffs
        .iter()
        .filter(|&(var, _)| !is_fixed(var))
        .map(|(_, coeff)| coeff.abs())
        .fold(0.0, f64::max);
    if max_coeff == 0.0 || !max_coeff.is_finite() {
        return 1.0;
    }

    let exponent = (max_coeff.log2().floor() as i32).clamp(-1023, 1023);
    let scale = 2.0_f64.powi(-exponent);
    if scale.is_finite() && (rhs * scale).is_finite() {
        scale
    } else {
        1.0
    }
}

/// The factor between the reduced cost of a row's slack in the engine's units
/// and in the user's (see [`super::dual_tol`]): `1 / (r m)`, for the row
/// factor `r` and `m` the row's largest coefficient as written on a var its
/// bounds do not fix (the vars [`equilibration_scale`] reads), so
/// that the slack's reduced cost is held per unit of the row's activity
/// divided by `m`, whatever factor the row was written at; one for a row with
/// no such coefficient.
pub(crate) fn slack_dual_unit(
    coeffs: &CsVec,
    row_scale: f64,
    is_fixed: impl Fn(usize) -> bool,
) -> f64 {
    let max_coeff = coeffs
        .iter()
        .filter(|&(var, _)| !is_fixed(var))
        .map(|(_, coeff)| coeff.abs())
        .fold(0.0, f64::max);
    let span = row_scale * max_coeff;
    if span > 0.0 && span.is_finite() {
        1.0 / span
    } else {
        1.0
    }
}

/// Number of alternating geometric-mean column/row passes used to derive the
/// column scales; four are enough for the scales to settle on the coefficient
/// ranges this solver sees.
const SCALING_PASSES: usize = 4;

/// If every structural coefficient magnitude lies within
/// `[1 / NO_SCALING_RANGE, NO_SCALING_RANGE]`, the matrix is already well
/// scaled and column scaling is skipped: every column factor is one and the
/// engine works on the user's own units.
const NO_SCALING_RANGE: f64 = 8.0;

/// Power-of-two column scales `s_j` for the structural columns: the engine
/// works on `x'_j = x_j / s_j`, i.e. on columns `a'_j = s_j a_j`, bounds
/// `[lo_j / s_j, hi_j / s_j]` and objective coefficients `s_j c_j`.
///
/// The factors are the rounded geometric means of each column's extreme
/// coefficient magnitudes, iterated against the corresponding row factors so
/// that a coefficient that is tiny only because its row or column is written
/// in awkward units ends up near one. Only the column factors are returned:
/// rows are then equilibrated by [`equilibration_scale`], which makes the row
/// factor part of the same power-of-two family.
///
/// Powers of two are exact: scaling and unscaling a value is a change of
/// exponent, so no user datum is rounded, integrality of `s_j x'_j` is
/// decidable exactly, and a bound change expressed in user units maps to
/// exactly one internal bound.
///
/// The factors are not bounded but for keeping the exponent within ±1000,
/// inside the range of `f64`: the engine takes any finite coefficient, and
/// its ratio tests take an entry at or below an absolute pivot tolerance, or
/// small beside the largest of its row, only as a last resort, where a
/// refinement shows it genuine, so a row that bounded factors leave wide
/// loses the stable pivots on its genuine entries. With the factors bounded
/// to `2^±20`, `min -g + y` s.t. `1e30 g + y >= 2e30`, `g ∈ [0, 1]`,
/// `y >= 0` (a test in `src/tests/magnitudes.rs`) would leave `y`'s entry at
/// `1.1e-18` beside one.
///
/// A column is left at one when scaling it would leave the finite range of
/// `f64` for any of its coefficients, its bounds or its objective coefficient.
pub(crate) fn column_scales(
    obj_coeffs: &[f64],
    var_mins: &[f64],
    var_maxs: &[f64],
    constraints: &[(CsVec, ComparisonOp, f64)],
) -> Vec<f64> {
    let num_vars = obj_coeffs.len();
    // A fixed var contributes a constant to every row it is in: its column
    // is neither scaled nor allowed to influence the row factors.
    let fixed: Vec<bool> = (0..num_vars).map(|j| var_mins[j] == var_maxs[j]).collect();
    let mut col_min = vec![f64::INFINITY; num_vars];
    let mut col_max = vec![0.0f64; num_vars];
    for (coeffs, _, _) in constraints {
        for (j, &a) in coeffs.iter() {
            let m = a.abs();
            if m > 0.0 && m.is_finite() && !fixed[j] {
                col_min[j] = col_min[j].min(m);
                col_max[j] = col_max[j].max(m);
            }
        }
    }
    let matrix_min = col_min.iter().copied().fold(f64::INFINITY, f64::min);
    let matrix_max = col_max.iter().copied().fold(0.0, f64::max);
    if matrix_max == 0.0 || (matrix_min >= 1.0 / NO_SCALING_RANGE && matrix_max <= NO_SCALING_RANGE)
    {
        return vec![1.0; num_vars];
    }

    // Log2 of the column and row factors, iterated to convergence.
    let mut log_col = vec![0.0f64; num_vars];
    let mut log_row = vec![0.0f64; constraints.len()];
    for _ in 0..SCALING_PASSES {
        let mut lo = vec![f64::INFINITY; num_vars];
        let mut hi = vec![f64::NEG_INFINITY; num_vars];
        for (i, (coeffs, _, _)) in constraints.iter().enumerate() {
            for (j, &a) in coeffs.iter() {
                let m = a.abs();
                if m > 0.0 && m.is_finite() && !fixed[j] {
                    let l = m.log2() + log_row[i];
                    lo[j] = lo[j].min(l);
                    hi[j] = hi[j].max(l);
                }
            }
        }
        for j in 0..num_vars {
            if lo[j].is_finite() {
                log_col[j] = -(lo[j] + hi[j]) / 2.0;
            }
        }
        for (i, (coeffs, _, _)) in constraints.iter().enumerate() {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for (j, &a) in coeffs.iter() {
                let m = a.abs();
                if m > 0.0 && m.is_finite() && !fixed[j] {
                    let l = m.log2() + log_col[j];
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
            }
            if lo.is_finite() {
                log_row[i] = -(lo + hi) / 2.0;
            }
        }
    }

    (0..num_vars)
        .map(|j| {
            let scale = 2.0_f64.powi(log_col[j].round().clamp(-1000.0, 1000.0) as i32);
            let bound_ok = |b: f64| b.is_infinite() || (b / scale).is_finite();
            let representable = col_max[j] == 0.0
                || ((col_max[j] * scale).is_finite()
                    && (col_min[j] * scale).is_normal()
                    && (obj_coeffs[j] * scale).is_finite()
                    && bound_ok(var_mins[j])
                    && bound_ok(var_maxs[j]));
            if representable {
                scale
            } else {
                1.0
            }
        })
        .collect()
}

/// A non-empty constraint row in the exact representation consumed by the
/// simplex engine. Structural coefficients and the right-hand side share the
/// same power-of-two scale; the slack coefficient remains one.
pub(crate) struct PreparedRow {
    pub(crate) coeffs: CsVec,
    pub(crate) rhs: f64,
    pub(crate) row_scale: f64,
    pub(crate) slack_var_min: f64,
    pub(crate) slack_var_max: f64,
}

/// Judge the rows that need no slack var and prepare one retained row for
/// storage. A row that every point satisfies (an empty row within the
/// contract, an infinite right-hand side on the open side) is `None`; one
/// that no point satisfies (see [`trivial_row_verdict`]) is `Infeasible`.
///
/// Every row enters the engine here, its coefficients scaled by their
/// columns' factors. A right-hand side that is NaN or infinite on the side
/// the row bounds, or a coefficient the scaling makes infinite, is an
/// `InternalError`: the model's validation refuses the first, so only
/// arithmetic on magnitudes far outside the recommended range makes either
/// (a row added with columns scaled for the rows before it, a fixed var's
/// term beside tiny ones).
pub(crate) fn prepare_row(
    mut coeffs: CsVec,
    cmp_op: ComparisonOp,
    rhs: f64,
    feasibility: f64,
    is_fixed: impl Fn(usize) -> bool,
) -> Result<Option<PreparedRow>, Error> {
    if !valid_rhs(cmp_op, rhs) {
        return Err(Error::InternalError(format!(
            "a {cmp_op:?} row reached the engine with the right-hand side {rhs}"
        )));
    }
    match trivial_row_verdict(&coeffs, cmp_op, rhs, feasibility) {
        Some(true) => return Ok(None),
        Some(false) => return Err(Error::Infeasible),
        None => {}
    }

    let row_scale = equilibration_scale(&coeffs, rhs, is_fixed);
    if row_scale != 1.0 {
        coeffs.map_inplace(|coeff| coeff * row_scale);
    }
    if let Some((var, coeff)) = coeffs.iter().find(|(_, coeff)| !coeff.is_finite()) {
        return Err(Error::InternalError(format!(
            "a row's coefficient of variable {var} is not finite once scaled: {coeff}"
        )));
    }
    let (slack_var_min, slack_var_max) = slack_bounds(cmp_op);
    Ok(Some(PreparedRow {
        coeffs,
        rhs: rhs * row_scale,
        row_scale,
        slack_var_min,
        slack_var_max,
    }))
}

/// The bounds a var enters the solve with ([`INFINITE_BOUND`]): each bound at
/// or beyond the threshold on its side of infinity made infinite, each side
/// on its own. A bound on the other side is data wherever it lies, as
/// `fix_var` may fix a var at a value the solve reported out there:
/// `x ∈ [5e15, f64::MAX]` enters as `x ≥ 5e15`, and so does
/// `x ∈ [5e15, 1e16]`, whose upper bound is beyond the threshold too.
///
/// A range that is a point or empty (`min >= max`) keeps its bounds as
/// written: its far bound does not lie beyond its near one, so it is the
/// var's value or a contradiction, not a stand-in for infinity, and a var
/// fixed out there stays fixed, an empty range out there stays infeasible. A
/// bound the solve refuses (a NaN one, a lower bound of `+∞`, an upper bound
/// of `-∞`; `Problem::solve_with`) stays as written, so it is reported as
/// given: only a lower bound is ever made `-∞`, an upper bound `+∞`, and a
/// NaN fails every comparison.
///
/// Applied once, where a var's bounds enter the crate (`Problem`'s
/// `internal_add_var` stores them so), so that presolve, the engine and every
/// check of a value against a bound see the same bounds; not applied to the
/// bounds presolve derives.
pub(crate) fn entry_bounds(min: f64, max: f64) -> (f64, f64) {
    if min >= max {
        return (min, max);
    }
    let min = if min <= -INFINITE_BOUND {
        f64::NEG_INFINITY
    } else {
        min
    };
    let max = if max >= INFINITE_BOUND {
        f64::INFINITY
    } else {
        max
    };
    (min, max)
}

/// Initial value for a non-basic structural variable, preferring the bound that
/// keeps its reduced cost dual-feasible. Returns `(value, dual_feasible)`, where
/// `dual_feasible` is false when no finite bound can satisfy dual feasibility (a
/// free variable, or a variable unbounded on the side its objective coefficient
/// pushes toward). `dual_tol` is the var's reduced-cost tolerance. Callers
/// handle a fixed variable (`min == max`) separately.
pub(crate) fn initial_nonbasic_value(
    obj_coeff: f64,
    min: f64,
    max: f64,
    dual_tol: f64,
) -> (f64, bool) {
    if min.is_infinite() && max.is_infinite() {
        // Free variable: dual-feasible only if the objective coefficient is zero.
        (0.0, obj_coeff.abs() < dual_tol)
    } else if obj_coeff > 0.0 {
        // Prefer the lower bound; fall back to the upper if the lower is infinite.
        if min.is_finite() {
            (min, true)
        } else {
            (max, false)
        }
    } else if obj_coeff < 0.0 {
        // Prefer the upper bound; fall back to the lower if the upper is infinite.
        if max.is_finite() {
            (max, true)
        } else {
            (min, false)
        }
    } else if min.is_finite() {
        // Zero objective coefficient: any finite bound is dual-feasible.
        (min, true)
    } else {
        (max, true)
    }
}

/// Multiply each structural coefficient of a user row by its column factor.
pub(crate) fn scale_columns(coeffs: CsVec, col_scales: &[f64]) -> CsVec {
    let len = coeffs.dim();
    let (indices, mut data) = coeffs.into_raw_storage();
    for (value, &var) in data.iter_mut().zip(&indices) {
        *value *= col_scales[var];
    }
    CsVec::new(len, indices, data)
}
