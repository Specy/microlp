//! Root cutting planes: separation of valid inequalities that tighten the
//! LP relaxation without excluding any integer-feasible point. Design doc:
//! docs/superpowers/specs/2026-07-12-root-cuts-design.md.
//!
//! Cuts are a ROOT-ONLY device in this solver: rows are added by
//! `mip::run_root_cuts` after the root LP solves and before the root node
//! snapshots its basis, so the search-time invariant that rows never grow
//! once nodes exist stays intact (every node basis is sized for the final,
//! cut-extended row set).
//!
//! This module is pure separation math over plain slices — it never touches
//! the `Solver` — so its validity can be tested by brute-force enumeration
//! of 0-1 points, which is exactly what its test module does. An invalid
//! cut is the worst class of solver bug (it silently removes optima), so
//! everything here leans conservative: rows that are not purely binary are
//! skipped, covers must exceed the capacity by a real margin (not float
//! noise), and emitted coefficients are exactly ±1 with an integer rhs.
//!
//! Phase 1 implements knapsack cover cuts. For a row brought into the form
//! `Σ aʹ_j y_j ≤ bʹ` with `aʹ_j > 0` over binary `y_j` (negative
//! coefficients complemented via `x = 1 − y`, fixed variables folded into
//! the rhs), any cover `C` with `Σ_{j∈C} aʹ_j > bʹ` yields the valid
//! inequality `Σ_{j∈C} y_j ≤ |C| − 1`, strengthened for free by the
//! extension set `E = {k ∉ C : aʹ_k ≥ max_{j∈C} aʹ_j}` (any `|C|`-subset of
//! `C ∪ E` still covers, so the rhs stays `|C| − 1`).

use std::collections::HashSet;

use crate::mip::params;
use crate::VarDomain;

/// A separated cut over STRUCTURAL variables, `Σ coeffs·x ≤ rhs`, with
/// `vars` sorted ascending. Cover cuts have ±1 coefficients and an integer
/// rhs by construction.
#[derive(Clone, Debug)]
pub(crate) struct Cut {
    pub vars: Vec<usize>,
    pub coeffs: Vec<f64>,
    pub rhs: f64,
    /// LHS − rhs at the separation point (how deeply the cut is violated).
    pub violation: f64,
}

/// Dedup set over a whole cut phase: two covers found from different rows
/// (or the two forms of an `Eq` row) can be the identical inequality, and a
/// row already in the LP must not be added again.
#[derive(Default)]
pub(crate) struct CutDedup(HashSet<(Vec<(u32, i8)>, i64)>);

impl CutDedup {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// True if `cut` was not seen before (and records it).
    fn insert(&mut self, cut: &Cut) -> bool {
        let key: Vec<(u32, i8)> = cut
            .vars
            .iter()
            .zip(&cut.coeffs)
            .map(|(&v, &c)| (v as u32, if c > 0.0 { 1 } else { -1 }))
            .collect();
        self.0.insert((key, cut.rhs.round() as i64))
    }
}

/// How a variable participates in knapsack forms, derived once per
/// separation call from its bounds and domain.
#[derive(Clone, Copy)]
enum VarKind {
    /// Fixed (`lo == hi`): folds into the rhs as a constant.
    Fixed(f64),
    /// Binary: integer domain with bounds exactly [0, 1].
    Binary,
    /// Anything else: a row containing it (with a nonzero coefficient)
    /// cannot be covered in v1.
    Other,
}

/// One item of a knapsack form `Σ a·y ≤ bʹ`: `a > 0`, `y` is `x` itself or
/// its complement `1 − x`, and `w = 1 − y*` is the separation weight (how
/// far the LP point is from including the item in a cover violation).
struct Item {
    var: usize,
    a: f64,
    comp: bool,
    w: f64,
}

/// Separate cover cuts from `rows` (activity form `lo ≤ Σ coeffs·x ≤ hi`,
/// the propagator's view of the search's rows) at the LP point `values`.
/// Returns at most `max_cuts` cuts, most violated first; `dedup` persists
/// across rounds so no inequality is ever emitted twice in one phase.
pub(crate) fn separate_cover_cuts<'a>(
    rows: impl Iterator<Item = (&'a [usize], &'a [f64], f64, f64)>,
    bounds: &[(f64, f64)],
    domains: &[VarDomain],
    values: &[f64],
    max_cuts: usize,
    dedup: &mut CutDedup,
) -> Vec<Cut> {
    let kind = |v: usize| -> VarKind {
        let (lo, hi) = bounds[v];
        if lo == hi {
            VarKind::Fixed(lo)
        } else if matches!(domains[v], VarDomain::Integer | VarDomain::Boolean)
            && lo == 0.0
            && hi == 1.0
        {
            VarKind::Binary
        } else {
            VarKind::Other
        }
    };

    let mut cuts: Vec<Cut> = Vec::new();
    let mut items: Vec<Item> = Vec::new();
    for (vars, coeffs, lo, hi) in rows {
        // Each finite side is one `≤`-form knapsack: `Σ s·coeffs·x ≤ s·side`
        // with s = +1 for the hi side and s = −1 for the lo side (Eq rows
        // yield both).
        for (sign, side) in [(1.0, hi), (-1.0, lo)] {
            if !(side * sign).is_finite() {
                continue;
            }
            let mut bprime = side * sign;
            items.clear();
            let mut coverable = true;
            for (&v, &c) in vars.iter().zip(coeffs) {
                let a = sign * c;
                if a == 0.0 {
                    continue;
                }
                match kind(v) {
                    VarKind::Fixed(val) => bprime -= a * val,
                    VarKind::Binary => {
                        // y* clamped to [0,1]: LP values can sit a hair
                        // outside their bounds by feasibility tolerance.
                        let xstar = values[v].clamp(0.0, 1.0);
                        if a > 0.0 {
                            items.push(Item {
                                var: v,
                                a,
                                comp: false,
                                w: 1.0 - xstar,
                            });
                        } else {
                            // Complement x = 1 − y: coefficient |a| on y,
                            // constant |a| moves to the rhs.
                            bprime += -a;
                            items.push(Item {
                                var: v,
                                a: -a,
                                comp: true,
                                w: xstar,
                            });
                        }
                    }
                    VarKind::Other => {
                        coverable = false;
                        break;
                    }
                }
            }
            if !coverable {
                continue;
            }
            if let Some(cut) = separate_form(&mut items, bprime) {
                if dedup.insert(&cut) {
                    cuts.push(cut);
                }
            }
        }
    }

    cuts.sort_by(|a, b| b.violation.total_cmp(&a.violation));
    cuts.truncate(max_cuts);
    cuts
}

/// Greedy minimal-cover separation on one knapsack form, plus the extension
/// set. Returns the cut mapped back to x-space, or `None` when no cover is
/// violated at the separation point.
fn separate_form(items: &mut [Item], bprime: f64) -> Option<Cut> {
    // A cover is only trusted when its weight clears the capacity by a REAL
    // margin: the emitted inequality's validity rests on "all cover members
    // at 1 violates the row", which must hold beyond both float error and
    // the feasibility tolerance the LP enforces the row at.
    let margin = params::COVER_MARGIN_REL * 1.0f64.max(bprime.abs());
    if bprime < 0.0 {
        // All aʹ > 0, so even y ≡ 0 exceeds the capacity — the LP could not
        // have solved this row to feasibility. Unreachable defensively.
        return None;
    }
    let total: f64 = items.iter().map(|it| it.a).sum();
    if total <= bprime + margin {
        return None; // no subset covers: the row admits all-ones
    }

    // Greedy: fill with the items costing the least violation-weight per
    // unit of capacity, i.e. the most fractional-toward-1 and heaviest.
    items.sort_by(|x, y| (x.w / x.a).total_cmp(&(y.w / y.a)));
    let mut member = vec![false; items.len()];
    let mut sum_a = 0.0;
    for (i, it) in items.iter().enumerate() {
        member[i] = true;
        sum_a += it.a;
        if sum_a > bprime + margin {
            break;
        }
    }
    if sum_a <= bprime + margin {
        return None; // defensive: total > bprime + margin ensures coverage
    }

    // Minimalize by dropping the highest-weight members first: dropping an
    // item removes its w from the cover weight, so violation only grows.
    let mut order: Vec<usize> = (0..items.len()).filter(|&i| member[i]).collect();
    order.sort_by(|&x, &y| items[y].w.total_cmp(&items[x].w));
    for i in order {
        if sum_a - items[i].a > bprime + margin {
            member[i] = false;
            sum_a -= items[i].a;
        }
    }

    let cover: Vec<usize> = (0..items.len()).filter(|&i| member[i]).collect();
    // A size-1 cover (aʹ_j > bʹ) is a bound deduction, not a row: presolve's
    // activity math already forces such a y to 0 — and forces every would-be
    // extension member the same way, since each has aʹ_k ≥ aʹ_j > bʹ.
    if cover.len() < 2 {
        return None;
    }
    let cover_weight: f64 = cover.iter().map(|&i| items[i].w).sum();
    let violation = 1.0 - cover_weight;
    if violation <= params::CUT_MIN_VIOLATION {
        return None;
    }

    // Extension set: every non-member at least as heavy as the heaviest
    // cover member joins the LHS with the rhs unchanged at |C| − 1 — any
    // |C|-subset of C ∪ E out-weighs C itself, so it still covers. Plain
    // `>=` is exactly the swap argument's requirement; no tolerance.
    let a_max = cover
        .iter()
        .map(|&i| items[i].a)
        .fold(f64::NEG_INFINITY, f64::max);
    let rhs_y = cover.len() as f64 - 1.0;
    let members: Vec<usize> = (0..items.len())
        .filter(|&i| member[i] || items[i].a >= a_max)
        .collect();

    // Back to x-space: y = x keeps +1, y = 1 − x turns into −1 on x and
    // shifts the rhs down by one. Coefficients ±1, integer rhs — pristine.
    let mut xs: Vec<(usize, f64)> = members
        .iter()
        .map(|&i| (items[i].var, if items[i].comp { -1.0 } else { 1.0 }))
        .collect();
    xs.sort_by_key(|&(v, _)| v);
    let n_comp = members.iter().filter(|&&i| items[i].comp).count();
    let rhs = rhs_y - n_comp as f64;
    let lhs: f64 = members
        .iter()
        .map(|&i| 1.0 - items[i].w) // y* of the member
        .sum::<f64>();
    Some(Cut {
        vars: xs.iter().map(|&(v, _)| v).collect(),
        coeffs: xs.iter().map(|&(_, c)| c).collect(),
        rhs,
        // In y-space LHS − rhs_y; the x-space mapping shifts LHS and rhs by
        // the same n_comp, so the violation is identical in both spaces.
        violation: lhs - rhs_y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin_bounds(n: usize) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); n]
    }

    fn int_domains(n: usize) -> Vec<VarDomain> {
        vec![VarDomain::Boolean; n]
    }

    /// One row in activity form, hi side only.
    fn le_row(vars: Vec<usize>, coeffs: Vec<f64>, hi: f64) -> (Vec<usize>, Vec<f64>, f64, f64) {
        (vars, coeffs, f64::NEG_INFINITY, hi)
    }

    fn separate(
        rows: &[(Vec<usize>, Vec<f64>, f64, f64)],
        bounds: &[(f64, f64)],
        domains: &[VarDomain],
        values: &[f64],
    ) -> Vec<Cut> {
        let mut dedup = CutDedup::new();
        separate_cover_cuts(
            rows.iter()
                .map(|(v, c, lo, hi)| (v.as_slice(), c.as_slice(), *lo, *hi)),
            bounds,
            domains,
            values,
            usize::MAX,
            &mut dedup,
        )
    }

    #[test]
    fn uniform_knapsack_finds_extended_cover() {
        // 3x + 3y + 3z <= 5: any two items cover; x* makes {x, y} the greedy
        // pick and z joins as the extension (equal weight).
        let rows = [le_row(vec![0, 1, 2], vec![3.0, 3.0, 3.0], 5.0)];
        let cuts = separate(&rows, &bin_bounds(3), &int_domains(3), &[0.9, 0.9, 0.05]);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert_eq!(cut.vars, vec![0, 1, 2]);
        assert_eq!(cut.coeffs, vec![1.0, 1.0, 1.0]);
        assert_eq!(cut.rhs, 1.0);
        assert!((cut.violation - 0.85).abs() < 1e-9, "{}", cut.violation);
    }

    #[test]
    fn negative_coefficient_is_complemented() {
        // 3x − 3y <= 2 complements y: 3x + 3yʹ <= 5, cover {x, yʹ} maps back
        // to x − y <= 0 (x = 1 forces y = 1).
        let rows = [le_row(vec![0, 1], vec![3.0, -3.0], 2.0)];
        let cuts = separate(&rows, &bin_bounds(2), &int_domains(2), &[0.9, 0.1]);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert_eq!(cut.vars, vec![0, 1]);
        assert_eq!(cut.coeffs, vec![1.0, -1.0]);
        assert_eq!(cut.rhs, 0.0);
    }

    #[test]
    fn ge_row_covers_through_the_lo_side() {
        // 2x + 3y + 4z >= 6 (activity lo = 6): complementing all three gives
        // the knapsack 2xʹ + 3yʹ + 4zʹ <= 3; C = {xʹ, yʹ}, E = {zʹ} — the
        // cut is x + y + z >= 2 in x-space.
        let rows = [(vec![0, 1, 2], vec![2.0, 3.0, 4.0], 6.0, f64::INFINITY)];
        let cuts = separate(&rows, &bin_bounds(3), &int_domains(3), &[0.05, 0.1, 0.9]);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert_eq!(cut.coeffs, vec![-1.0, -1.0, -1.0]);
        assert_eq!(cut.rhs, -2.0);
    }

    #[test]
    fn redundant_row_yields_nothing() {
        // Sum of all coefficients is under the capacity: no cover exists.
        let rows = [le_row(vec![0, 1], vec![1.0, 1.0], 3.0)];
        let cuts = separate(&rows, &bin_bounds(2), &int_domains(2), &[0.9, 0.9]);
        assert!(cuts.is_empty());
    }

    #[test]
    fn size_one_cover_is_left_to_presolve() {
        // 5x + 1y <= 4: {x} alone covers, which is a bound deduction
        // (presolve fixes x = 0), not a row.
        let rows = [le_row(vec![0, 1], vec![5.0, 1.0], 4.0)];
        let cuts = separate(&rows, &bin_bounds(2), &int_domains(2), &[0.79, 0.9]);
        assert!(cuts.is_empty());
    }

    #[test]
    fn satisfied_covers_are_not_emitted() {
        // Same knapsack as the uniform test but an LP point that violates no
        // cover: every 2-subset has weight sum >= 1.
        let rows = [le_row(vec![0, 1, 2], vec![3.0, 3.0, 3.0], 5.0)];
        let cuts = separate(&rows, &bin_bounds(3), &int_domains(3), &[0.5, 0.5, 0.5]);
        assert!(cuts.is_empty());
    }

    #[test]
    fn fixed_vars_fold_into_the_capacity() {
        // z fixed at 1 consumes 3 of the capacity 8, leaving 3x + 3y <= 5.
        let rows = [le_row(vec![0, 1, 2], vec![3.0, 3.0, 3.0], 8.0)];
        let bounds = [(0.0, 1.0), (0.0, 1.0), (1.0, 1.0)];
        let cuts = separate(&rows, &bounds, &int_domains(3), &[0.9, 0.9, 1.0]);
        assert_eq!(cuts.len(), 1);
        let cut = &cuts[0];
        assert_eq!(cut.vars, vec![0, 1]);
        assert_eq!(cut.rhs, 1.0);
    }

    #[test]
    fn non_binary_var_disqualifies_the_row() {
        let rows = [le_row(vec![0, 1, 2], vec![3.0, 3.0, 3.0], 5.0)];
        let mut bounds = bin_bounds(3);
        bounds[2] = (0.0, 2.0); // integer but not binary
        let cuts = separate(&rows, &bounds, &int_domains(3), &[0.9, 0.9, 0.4]);
        assert!(cuts.is_empty());
    }

    #[test]
    fn duplicate_inequalities_are_emitted_once() {
        // The same knapsack twice: identical cover, one cut.
        let rows = [
            le_row(vec![0, 1, 2], vec![3.0, 3.0, 3.0], 5.0),
            le_row(vec![0, 1, 2], vec![3.0, 3.0, 3.0], 5.0),
        ];
        let cuts = separate(&rows, &bin_bounds(3), &int_domains(3), &[0.9, 0.9, 0.05]);
        assert_eq!(cuts.len(), 1);
    }

    /// The killer test: on seeded random knapsacks, every emitted cut must
    /// be satisfied by EVERY row-feasible 0-1 point (validity by brute
    /// force) and genuinely violated at the separation point. An invalid
    /// cut silently deletes optima in the wild; it fails loudly here.
    #[test]
    fn random_knapsack_cuts_never_cut_feasible_points() {
        let mut rng_state: u64 = 0x9e3779b97f4a7c15;
        let mut rng = move || {
            // splitmix64, same generator the suite uses for seeded cases.
            rng_state = rng_state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let mut emitted = 0usize;
        for _ in 0..200 {
            let n = 4 + (rng() % 6) as usize; // 4..=9 vars
            let coeffs: Vec<f64> = (0..n)
                .map(|_| {
                    let mag = 1 + (rng() % 8) as i64;
                    (if rng() % 3 == 0 { -mag } else { mag }) as f64
                })
                .collect();
            let pos_sum: f64 = coeffs.iter().filter(|&&c| c > 0.0).sum();
            let neg_sum: f64 = coeffs.iter().filter(|&&c| c < 0.0).sum();
            // Capacity strictly between the row's min and max activity so
            // both feasible points and covers exist.
            let span = pos_sum - neg_sum;
            let rhs = neg_sum + ((rng() % 1000) as f64 / 1000.0) * span;
            let values: Vec<f64> = (0..n).map(|_| (rng() % 1000) as f64 / 999.0).collect();

            let rows = [le_row((0..n).collect(), coeffs.clone(), rhs)];
            let cuts = separate(&rows, &bin_bounds(n), &int_domains(n), &values);
            emitted += cuts.len();

            for cut in &cuts {
                // Violated at the separation point, beyond the threshold.
                let lhs_star: f64 = cut
                    .vars
                    .iter()
                    .zip(&cut.coeffs)
                    .map(|(&v, &c)| c * values[v])
                    .sum();
                assert!(
                    lhs_star - cut.rhs > params::CUT_MIN_VIOLATION,
                    "cut not violated at its own separation point"
                );
                // Valid on every feasible 0-1 point.
                for point in 0u32..(1 << n) {
                    let x = |v: usize| ((point >> v) & 1) as f64;
                    let row_lhs: f64 = (0..n).map(|v| coeffs[v] * x(v)).sum();
                    if row_lhs > rhs {
                        continue; // infeasible for the row: free to cut
                    }
                    let cut_lhs: f64 = cut
                        .vars
                        .iter()
                        .zip(&cut.coeffs)
                        .map(|(&v, &c)| c * x(v))
                        .sum();
                    assert!(
                        cut_lhs <= cut.rhs + 1e-9,
                        "cut {:?} cuts off feasible point {:b} of row {:?} <= {}",
                        cut,
                        point,
                        coeffs,
                        rhs,
                    );
                }
            }
        }
        // The generator must actually exercise the machinery.
        assert!(emitted > 50, "only {emitted} cuts emitted across 200 draws");
    }
}
