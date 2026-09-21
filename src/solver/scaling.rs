//! Turning a user problem into the engine's scaled representation: power-of-two
//! row and column scaling, per-row preparation, and the rule that picks a
//! non-basic variable's starting bound.
//!
//! Every factor is a power of two, so scaling and unscaling change only an
//! exponent and no user datum is ever rounded.

use crate::{ComparisonOp, CsVec, Error};

use super::tolerances::float_eq;

/// A variable bound whose magnitude is at least this large is treated as
/// infinite when choosing a non-basic variable's INITIAL value (see
/// [`initial_nonbasic_value`]). Seeding a non-basic variable *at* such a bound
/// floods the tableau with a value that swamps the actual problem data — the
/// rhs and structural coefficients lose all significance against it and the
/// solve converges to a wrong vertex or NaN. This is issue #3: `f64::MAX`,
/// `f32::MAX` and `i64::MAX` upper bounds produced non-optimal answers where
/// `f64::INFINITY` did not, because only the latter skipped the seed-at-bound
/// step. 2^52 is the largest f64 whose unit (`1.0`) is still exactly
/// representable; beyond it a finite bound is numerically a stand-in for
/// infinity, so we seed as if it were infinite. The true bound is left
/// untouched in `orig_var_mins`/`orig_var_maxs`, so the ratio test still
/// honours it exactly — only the starting vertex changes.
const SEED_AS_INFINITE: f64 = 4_503_599_627_370_496.0; // 2^52

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

/// Number of alternating geometric column/row passes used to derive the
/// column scales (Curtis–Reid-style iteration; four passes are enough for the
/// scales to settle on the coefficient ranges this solver sees).
const SCALING_PASSES: usize = 4;

/// If every structural coefficient magnitude lies within
/// `[1 / NO_SCALING_RANGE, NO_SCALING_RANGE]`, the matrix is already well
/// scaled and column scaling is skipped: every column factor is one and the
/// engine works on the user's own units, so ordinary models take exactly the
/// path they always took.
const NO_SCALING_RANGE: f64 = 8.0;

/// Power-of-two column scales `s_j` for the structural columns: the engine
/// works on `x'_j = x_j / s_j`, i.e. on columns `a'_j = s_j a_j`, bounds
/// `[lo_j / s_j, hi_j / s_j]` and objective coefficients `s_j c_j`.
///
/// The factors are the rounded geometric means of each column's extreme
/// coefficient magnitudes, iterated against the corresponding row factors so
/// that a coefficient that is tiny only because its row or column is written
/// in awkward units ends up near one. Only the column factors are returned:
/// rows are then equilibrated by [`equilibration_scale`] as before, which
/// makes the row factor part of the same power-of-two family.
///
/// Powers of two are exact: scaling and unscaling a value is a change of
/// exponent, so no user datum is rounded, integrality of `s_j x'_j` is
/// decidable exactly, and a bound change expressed in user units maps to
/// exactly one internal bound.
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

/// Validate empty-row semantics and prepare one retained row for storage.
/// `None` denotes a tautology that does not need a slack variable.
pub(crate) fn prepare_row(
    mut coeffs: CsVec,
    cmp_op: ComparisonOp,
    rhs: f64,
    is_fixed: impl Fn(usize) -> bool,
) -> Result<Option<PreparedRow>, Error> {
    if coeffs.indices().is_empty() {
        let tautological = match cmp_op {
            ComparisonOp::Eq => float_eq(rhs, 0.0),
            ComparisonOp::Le => 0.0 <= rhs,
            ComparisonOp::Ge => 0.0 >= rhs,
        };
        return if tautological {
            Ok(None)
        } else {
            Err(Error::Infeasible)
        };
    }

    let row_scale = equilibration_scale(&coeffs, rhs, is_fixed);
    if row_scale != 1.0 {
        coeffs.map_inplace(|coeff| coeff * row_scale);
    }
    let (slack_var_min, slack_var_max) = match cmp_op {
        ComparisonOp::Le => (0.0, f64::INFINITY),
        ComparisonOp::Ge => (f64::NEG_INFINITY, 0.0),
        ComparisonOp::Eq => (0.0, 0.0),
    };
    Ok(Some(PreparedRow {
        coeffs,
        rhs: rhs * row_scale,
        row_scale,
        slack_var_min,
        slack_var_max,
    }))
}

/// Initial value for a non-basic structural variable, preferring the bound that
/// keeps its reduced cost dual-feasible. Returns `(value, dual_feasible)`, where
/// `dual_feasible` is false when no finite bound can satisfy dual feasibility (a
/// free variable, or a variable unbounded on the side its objective coefficient
/// pushes toward). Callers handle a fixed variable (`min == max`) separately.
///
/// A bound at or beyond [`SEED_AS_INFINITE`] is treated as infinite here so that
/// a huge finite bound is never used as the seed value (issue #3); the caller's
/// stored bounds are left untouched, so the ratio test still honours them.
pub(crate) fn initial_nonbasic_value(obj_coeff: f64, min: f64, max: f64) -> (f64, bool) {
    let min = if min <= -SEED_AS_INFINITE {
        f64::NEG_INFINITY
    } else {
        min
    };
    let max = if max >= SEED_AS_INFINITE {
        f64::INFINITY
    } else {
        max
    };

    if min.is_infinite() && max.is_infinite() {
        // Free variable: dual-feasible only if the objective coefficient is zero.
        (0.0, float_eq(obj_coeff, 0.0))
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
