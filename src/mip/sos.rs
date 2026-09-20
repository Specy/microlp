//! Special ordered sets: feasibility test and branching rule.
//!
//! A [`SosSet`] is a list of structural variables ordered by weight. SOS1
//! allows at most one member to be nonzero; SOS2 allows at most two, and they
//! must be adjacent in the weight order. Members may be continuous or integer.
//!
//! # Zero threshold
//!
//! A member counts as zero when `|x| <= zero_tol` (the driver passes
//! `Tolerances::feasibility`) OR when its current bounds are `[0, 0]`. The
//! second rule is what makes SOS branching terminate independently of
//! tolerances: a member already fixed to zero is never counted again, so every
//! child of a violated set fixes at least one member that was still free, and
//! the number of free members strictly decreases down any path of the tree.
//!
//! # Branching (Beale–Tomlin)
//!
//! Let the counted nonzero members sit at positions `a..=b` (`a < b`, since a
//! satisfied set never reaches the split). The weighted mean of their weights
//! picks a split position `r`:
//!
//! * SOS1: `a <= r < b`; the left child zeroes positions `> r`, the right
//!   child zeroes positions `<= r`.
//! * SOS2: `a < r < b`; the left child zeroes positions `> r`, the right child
//!   zeroes positions `< r`, and position `r` stays free in both.
//!
//! Either way the left child zeroes position `b` and the right child zeroes
//! position `a`, so both children exclude the current point.

use crate::solver::Solver;

/// Which special-ordered-set rule a set enforces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SosKind {
    /// At most one nonzero member.
    Sos1,
    /// At most two nonzero members, adjacent in weight order.
    Sos2,
}

/// One special ordered set: members sorted by ascending weight.
#[derive(Clone, Debug)]
pub(crate) struct SosSet {
    pub kind: SosKind,
    /// Structural variable indices, in ascending weight order.
    pub members: Vec<usize>,
    /// The members' weights, in the same order.
    pub weights: Vec<f64>,
}

/// A branching decision on one set: which members each child fixes to zero.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Split {
    /// Members the left child fixes to zero (positions above the split).
    pub left_zero: Vec<usize>,
    /// Members the right child fixes to zero (positions below the split).
    pub right_zero: Vec<usize>,
    /// Sum of `|x|` over the counted nonzero members the left child keeps
    /// free, used to decide which child to dive into first. For SOS2 the
    /// split position stays free in both children and counts on both sides.
    pub left_mass: f64,
    /// Same for the right child.
    pub right_mass: f64,
}

impl SosSet {
    /// The counted nonzero members: `(position, |x|)` for every member whose
    /// candidate value is nonzero and whose current bounds are not `[0, 0]`.
    fn nonzero_positions(
        &self,
        value: &dyn Fn(usize) -> f64,
        fixed_to_zero: &dyn Fn(usize) -> bool,
    ) -> Vec<(usize, f64)> {
        self.members
            .iter()
            .enumerate()
            .filter_map(|(pos, &v)| {
                let x = value(v);
                if x != 0.0 && !fixed_to_zero(v) {
                    Some((pos, x.abs()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Whether the set holds for the given candidate values. `value` must
    /// already be snapped (see [`snap_members`]): a value counts as nonzero
    /// exactly when it is not `0.0`.
    pub(crate) fn is_satisfied(
        &self,
        value: &dyn Fn(usize) -> f64,
        fixed_to_zero: &dyn Fn(usize) -> bool,
    ) -> bool {
        let nz = self.nonzero_positions(value, fixed_to_zero);
        match self.kind {
            SosKind::Sos1 => nz.len() <= 1,
            SosKind::Sos2 => match (nz.first(), nz.last()) {
                (Some(&(a, _)), Some(&(b, _))) => b - a <= 1,
                _ => true,
            },
        }
    }

    /// The branching split for a violated set, or `None` when the set holds.
    pub(crate) fn split(
        &self,
        value: &dyn Fn(usize) -> f64,
        fixed_to_zero: &dyn Fn(usize) -> bool,
    ) -> Option<Split> {
        let nz = self.nonzero_positions(value, fixed_to_zero);
        let (a, b) = match (nz.first(), nz.last()) {
            (Some(&(a, _)), Some(&(b, _))) => (a, b),
            _ => return None,
        };
        let violated = match self.kind {
            SosKind::Sos1 => nz.len() >= 2,
            SosKind::Sos2 => b - a >= 2,
        };
        if !violated {
            return None;
        }

        // Weighted mean of the nonzero members' weights, then the last
        // position (zero members between `a` and `b` included) whose weight
        // does not exceed it.
        let mass: f64 = nz.iter().map(|&(_, m)| m).sum();
        let mean = nz.iter().map(|&(p, m)| self.weights[p] * m).sum::<f64>() / mass;
        let mut r = a;
        for p in a..=b {
            if self.weights[p] <= mean {
                r = p;
            }
        }
        let r = match self.kind {
            SosKind::Sos1 => r.clamp(a, b - 1),
            SosKind::Sos2 => r.clamp(a + 1, b - 1),
        };

        // Children carry only members that are still free: a member already
        // at [0, 0] needs no new bound change.
        let free = |slice: &[usize]| -> Vec<usize> {
            slice
                .iter()
                .copied()
                .filter(|&v| !fixed_to_zero(v))
                .collect()
        };
        let (left_zero, right_zero) = match self.kind {
            SosKind::Sos1 => (free(&self.members[r + 1..]), free(&self.members[..=r])),
            SosKind::Sos2 => (free(&self.members[r + 1..]), free(&self.members[..r])),
        };
        let left_mass: f64 = nz.iter().filter(|&&(p, _)| p <= r).map(|&(_, m)| m).sum();
        let right_mass: f64 = nz
            .iter()
            .filter(|&&(p, _)| match self.kind {
                SosKind::Sos1 => p > r,
                SosKind::Sos2 => p >= r,
            })
            .map(|&(_, m)| m)
            .sum();
        Some(Split {
            left_zero,
            right_zero,
            left_mass,
            right_mass,
        })
    }
}

/// Snap the SOS members of a candidate vector: a member with `|x| <= zero_tol`
/// becomes exactly `0.0`, as does a member whose current bounds are `[0, 0]`
/// (a fixed variable that is basic can carry round-off). Non-members are left
/// untouched, so models without SOS sets are unaffected.
pub(crate) fn snap_members(
    sets: &[SosSet],
    values: &mut [f64],
    zero_tol: f64,
    fixed_to_zero: &dyn Fn(usize) -> bool,
) {
    for set in sets {
        for &v in &set.members {
            if values[v].abs() <= zero_tol || fixed_to_zero(v) {
                values[v] = 0.0;
            }
        }
    }
}

/// Whether every set holds for a snapped candidate vector.
pub(crate) fn all_satisfied(
    sets: &[SosSet],
    value: &dyn Fn(usize) -> f64,
    fixed_to_zero: &dyn Fn(usize) -> bool,
) -> bool {
    sets.iter().all(|s| s.is_satisfied(value, fixed_to_zero))
}

/// The split of the first violated set, in declaration order.
pub(crate) fn first_violated(
    sets: &[SosSet],
    value: &dyn Fn(usize) -> f64,
    fixed_to_zero: &dyn Fn(usize) -> bool,
) -> Option<Split> {
    sets.iter().find_map(|s| s.split(value, fixed_to_zero))
}

/// `[0, 0]` test against the solver's current bounds.
pub(crate) fn solver_fixed_to_zero(solver: &Solver) -> impl Fn(usize) -> bool + '_ {
    move |v| solver.get_var_bounds(v) == (0.0, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(kind: SosKind, n: usize) -> SosSet {
        SosSet {
            kind,
            members: (10..10 + n).collect(),
            weights: (0..n).map(|i| i as f64 + 1.0).collect(),
        }
    }

    fn vals(x: &[f64]) -> impl Fn(usize) -> f64 + '_ {
        move |v| x[v - 10]
    }

    fn none_fixed(_: usize) -> bool {
        false
    }

    #[test]
    fn sos1_feasibility() {
        let s = set(SosKind::Sos1, 4);
        assert!(s.is_satisfied(&vals(&[0.0, 0.0, 0.0, 0.0]), &none_fixed));
        assert!(s.is_satisfied(&vals(&[0.0, 3.0, 0.0, 0.0]), &none_fixed));
        assert!(!s.is_satisfied(&vals(&[1.0, 0.0, 0.0, -2.0]), &none_fixed));
        // A member fixed to [0, 0] is zero whatever its value says.
        assert!(s.is_satisfied(&vals(&[1e-12, 0.0, 0.0, -2.0]), &|v| v == 10));
    }

    #[test]
    fn sos2_feasibility() {
        let s = set(SosKind::Sos2, 5);
        assert!(s.is_satisfied(&vals(&[0.0; 5]), &none_fixed));
        assert!(s.is_satisfied(&vals(&[0.0, 0.0, 0.5, 0.0, 0.0]), &none_fixed));
        assert!(s.is_satisfied(&vals(&[0.0, 0.3, 0.7, 0.0, 0.0]), &none_fixed));
        assert!(s.is_satisfied(&vals(&[0.0, 0.0, 0.0, 0.7, 0.3]), &none_fixed));
        // Two non-adjacent, or three consecutive: violated.
        assert!(!s.is_satisfied(&vals(&[0.5, 0.0, 0.5, 0.0, 0.0]), &none_fixed));
        assert!(!s.is_satisfied(&vals(&[0.0, 0.2, 0.3, 0.5, 0.0]), &none_fixed));
    }

    /// For every nonzero pattern of a small set, both children must exclude
    /// the current point: the left child zeroes the last nonzero, the right
    /// child zeroes the first, and every fixed member is a counted nonzero
    /// or lies beyond the split (never a member the point already has at 0
    /// on the "wrong" side that would leave the point feasible).
    #[test]
    fn split_children_exclude_the_point() {
        for kind in [SosKind::Sos1, SosKind::Sos2] {
            let n = 6;
            let s = set(kind, n);
            for pattern in 1u32..(1 << n) {
                let x: Vec<f64> = (0..n)
                    .map(|i| {
                        if pattern & (1 << i) != 0 {
                            0.5 + i as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let split = s.split(&vals(&x), &none_fixed);
                let satisfied = s.is_satisfied(&vals(&x), &none_fixed);
                assert_eq!(split.is_none(), satisfied, "{kind:?} {x:?}");
                let Some(split) = split else { continue };
                let nz: Vec<usize> = (0..n).filter(|&i| x[i] != 0.0).collect();
                let first = 10 + nz[0];
                let last = 10 + nz[nz.len() - 1];
                assert!(split.left_zero.contains(&last), "{kind:?} {x:?} {split:?}");
                assert!(
                    split.right_zero.contains(&first),
                    "{kind:?} {x:?} {split:?}"
                );
                assert!(!split.left_zero.contains(&first));
                assert!(!split.right_zero.contains(&last));
                assert!(split.left_mass > 0.0 && split.right_mass > 0.0);
                // Left and right zero sets are disjoint for SOS1 and share
                // no member for SOS2 either (position r is free in both).
                assert!(split
                    .left_zero
                    .iter()
                    .all(|v| !split.right_zero.contains(v)));
                // Sorted ascending: deterministic bound-change order.
                assert!(split.left_zero.windows(2).all(|w| w[0] < w[1]));
                assert!(split.right_zero.windows(2).all(|w| w[0] < w[1]));
            }
        }
    }

    #[test]
    fn sos2_split_keeps_position_r_free_in_both_children() {
        let s = set(SosKind::Sos2, 5);
        // Nonzeros at 0 and 4, equal mass: mean weight 3 → r = position 2.
        let split = s
            .split(&vals(&[1.0, 0.0, 0.0, 0.0, 1.0]), &none_fixed)
            .unwrap();
        assert_eq!(split.left_zero, vec![13, 14]);
        assert_eq!(split.right_zero, vec![10, 11]);
    }

    #[test]
    fn sos1_split_uses_weighted_mean() {
        let s = set(SosKind::Sos1, 4);
        // Mass concentrated on the last member pulls the split up: mean
        // weight (1·1 + 4·9)/10 = 3.7 → r = position 2 (weight 3).
        let split = s.split(&vals(&[1.0, 0.0, 0.0, 9.0]), &none_fixed).unwrap();
        assert_eq!(split.left_zero, vec![13]);
        assert_eq!(split.right_zero, vec![10, 11, 12]);
    }

    #[test]
    fn snap_zeroes_members_within_tolerance_and_fixed_ones_only() {
        let s = set(SosKind::Sos1, 3);
        let mut x = vec![0.0; 14];
        x[9] = 1e-9; // not a member: untouched
        x[10] = 1e-9;
        x[11] = 0.5;
        x[12] = 1e-9;
        snap_members(&[s], &mut x, 1e-7, &|v| v == 12);
        assert_eq!(x[9], 1e-9);
        assert_eq!(x[10], 0.0);
        assert_eq!(x[11], 0.5);
        assert_eq!(x[12], 0.0);
    }

    #[test]
    fn violated_set_always_has_a_split_and_every_child_fixes_a_free_member() {
        // Members 10 and 11 already fixed to zero but carrying noise: only 12
        // and 13 count, so the split must fix one of THEM on each side.
        let s = set(SosKind::Sos1, 4);
        let fixed = |v: usize| v == 10 || v == 11;
        let split = s.split(&vals(&[1e-12, -1e-12, 2.0, 3.0]), &fixed).unwrap();
        assert!(split.left_zero.contains(&13));
        assert!(split.right_zero.contains(&12));
    }
}
