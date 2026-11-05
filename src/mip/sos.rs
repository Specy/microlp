//! Special ordered sets: when a set holds, and how a violated one is split.
//!
//! A [`SosSet`] is a list of structural vars ordered by weight. SOS1 allows at
//! most one member to be nonzero; SOS2 allows at most two, and they must be
//! adjacent in the weight order. Members may be continuous or integer. Sets
//! never reach the engine: the search enforces them by branching, as it does
//! integrality.
//!
//! # When a member counts as zero
//!
//! A member counts as zero when its value meets the bound `[0, 0]` within the
//! contract every bound is held to (`crate::solver::outside_bounds` at
//! [`crate::Tolerances::feasibility`]), or when its bounds in force are
//! `[0, 0]`. Values are reported as the engine computed them, as an integer
//! var's are within `int_tol` of its integer: nothing is snapped, since a
//! snapped value would no longer be the point the rows were verified at.
//!
//! The second rule makes the search terminate whatever the tolerances: a
//! member its bounds hold at zero is never counted again, so each child of a
//! violated set fixes at least one member still free, and the number of free
//! members strictly decreases down any path of the tree.
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
//! Either way the left child zeroes position `b` and the right child position
//! `a`, so both children exclude the current point.
//!
//! Adapted from Adel Kaleche's SOS branching (Specy/microlp#50).

use crate::solver::{outside_bounds, Solver};

/// Which special-ordered-set rule a set enforces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SosKind {
    /// At most one nonzero member.
    Sos1,
    /// At most two nonzero members, adjacent in weight order.
    Sos2,
}

impl SosKind {
    /// The fewest members a set of this kind needs to constrain anything.
    pub(crate) fn min_members(self) -> usize {
        match self {
            SosKind::Sos1 => 2,
            SosKind::Sos2 => 3,
        }
    }
}

/// One special ordered set: members sorted by ascending weight.
#[derive(Clone, Debug)]
pub(crate) struct SosSet {
    pub kind: SosKind,
    /// Structural var indices, in ascending weight order.
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
    /// free, which decides the child the search dives into first. For SOS2
    /// the split position stays free in both children and counts on both
    /// sides.
    pub left_mass: f64,
    /// Same for the right child.
    pub right_mass: f64,
}

/// How a point's members are read: each one's value, and whether its bounds
/// in force are `[0, 0]`.
pub(crate) struct Point<'a> {
    pub value: &'a dyn Fn(usize) -> f64,
    pub fixed_to_zero: &'a dyn Fn(usize) -> bool,
    /// A member whose value meets `[0, 0]` within this absolute tolerance
    /// counts as zero; `None` counts every nonzero value (the strict test of
    /// a candidate the guard rejected).
    pub zero_tol: Option<f64>,
}

impl Point<'_> {
    /// `|x|` of member `v` if it counts as nonzero, else `None`.
    fn nonzero(&self, v: usize) -> Option<f64> {
        if (self.fixed_to_zero)(v) {
            return None;
        }
        let x = (self.value)(v);
        let counts = match self.zero_tol {
            Some(tol) => outside_bounds(x, 0.0, 0.0, tol),
            None => x != 0.0,
        };
        counts.then_some(x.abs())
    }
}

impl SosSet {
    /// The counted nonzero members as `(position, |x|)`.
    fn nonzero_positions(&self, point: &Point) -> Vec<(usize, f64)> {
        self.members
            .iter()
            .enumerate()
            .filter_map(|(pos, &v)| point.nonzero(v).map(|m| (pos, m)))
            .collect()
    }

    /// Whether the set holds at `point`.
    pub(crate) fn is_satisfied(&self, point: &Point) -> bool {
        let nz = self.nonzero_positions(point);
        match self.kind {
            SosKind::Sos1 => nz.len() <= 1,
            SosKind::Sos2 => match (nz.first(), nz.last()) {
                (Some(&(a, _)), Some(&(b, _))) => b - a <= 1,
                _ => true,
            },
        }
    }

    /// The branching split of a set violated at `point`, or `None` when it
    /// holds there.
    pub(crate) fn split(&self, point: &Point) -> Option<Split> {
        let nz = self.nonzero_positions(point);
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

        // Children carry only members that are still free: one already at
        // [0, 0] needs no new bound change.
        let free = |slice: &[usize]| -> Vec<usize> {
            slice
                .iter()
                .copied()
                .filter(|&v| !(point.fixed_to_zero)(v))
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

/// Whether every set holds at `point`.
pub(crate) fn all_satisfied(sets: &[SosSet], point: &Point) -> bool {
    sets.iter().all(|s| s.is_satisfied(point))
}

/// The split of the first set violated at `point`, in declaration order.
pub(crate) fn first_violated(sets: &[SosSet], point: &Point) -> Option<Split> {
    sets.iter().find_map(|s| s.split(point))
}

/// Whether `sets` hold at `values` (user units), a member counting as zero
/// within `zero_tol` or where `bounds` are `[0, 0]`: the check of an
/// incumbent that survives an edit (`super::incumbent_feasible`).
pub(crate) fn values_satisfy(
    sets: &[SosSet],
    values: &[f64],
    zero_tol: f64,
    bounds: impl Fn(usize) -> (f64, f64),
) -> bool {
    all_satisfied(
        sets,
        &Point {
            value: &|v| values[v],
            fixed_to_zero: &|v| bounds(v) == (0.0, 0.0),
            zero_tol: Some(zero_tol),
        },
    )
}

/// The solver's current point, a member counting as zero within `zero_tol`
/// (strictly, for `None`), and its split of the first violated set.
pub(crate) fn solver_split(
    sets: &[SosSet],
    solver: &Solver,
    zero_tol: Option<f64>,
) -> Option<Split> {
    first_violated(
        sets,
        &Point {
            value: &|v| solver.get_value(v),
            fixed_to_zero: &|v| solver.get_var_bounds(v) == (0.0, 0.0),
            zero_tol,
        },
    )
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

    /// `x` as a point (members 10 up), nothing fixed, the strict test.
    fn satisfied(s: &SosSet, x: &[f64], fixed: &dyn Fn(usize) -> bool) -> bool {
        s.is_satisfied(&Point {
            value: &|v| x[v - 10],
            fixed_to_zero: fixed,
            zero_tol: None,
        })
    }

    fn split(s: &SosSet, x: &[f64], fixed: &dyn Fn(usize) -> bool) -> Option<Split> {
        s.split(&Point {
            value: &|v| x[v - 10],
            fixed_to_zero: fixed,
            zero_tol: None,
        })
    }

    fn none_fixed(_: usize) -> bool {
        false
    }

    #[test]
    fn sos1_feasibility() {
        let s = set(SosKind::Sos1, 4);
        assert!(satisfied(&s, &[0.0, 0.0, 0.0, 0.0], &none_fixed));
        assert!(satisfied(&s, &[0.0, 3.0, 0.0, 0.0], &none_fixed));
        assert!(!satisfied(&s, &[1.0, 0.0, 0.0, -2.0], &none_fixed));
        // A member fixed to [0, 0] is zero whatever its value says.
        assert!(satisfied(&s, &[1e-12, 0.0, 0.0, -2.0], &|v| v == 10));
    }

    #[test]
    fn sos2_feasibility() {
        let s = set(SosKind::Sos2, 5);
        assert!(satisfied(&s, &[0.0; 5], &none_fixed));
        assert!(satisfied(&s, &[0.0, 0.0, 0.5, 0.0, 0.0], &none_fixed));
        assert!(satisfied(&s, &[0.0, 0.3, 0.7, 0.0, 0.0], &none_fixed));
        assert!(satisfied(&s, &[0.0, 0.0, 0.0, 0.7, 0.3], &none_fixed));
        // Two non-adjacent, or three consecutive: violated.
        assert!(!satisfied(&s, &[0.5, 0.0, 0.5, 0.0, 0.0], &none_fixed));
        assert!(!satisfied(&s, &[0.0, 0.2, 0.3, 0.5, 0.0], &none_fixed));
    }

    /// A member counts as zero within the bound contract at `[0, 0]`: the
    /// absolute tolerance, not a share of the member's range.
    #[test]
    fn a_member_within_the_bound_tolerance_of_zero_counts_as_zero() {
        let s = set(SosKind::Sos1, 3);
        let x = [5e-8, 2.0, 0.0];
        let value = |v: usize| x[v - 10];
        let point = |zero_tol| Point {
            value: &value,
            fixed_to_zero: &none_fixed,
            zero_tol,
        };
        assert!(s.is_satisfied(&point(Some(1e-7))));
        assert!(!s.is_satisfied(&point(Some(1e-8))));
        assert!(!s.is_satisfied(&point(None)));
    }

    /// For every nonzero pattern of a small set, both children must exclude
    /// the current point: the left child zeroes the last nonzero, the right
    /// child zeroes the first.
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
                let split = split(&s, &x, &none_fixed);
                assert_eq!(
                    split.is_none(),
                    satisfied(&s, &x, &none_fixed),
                    "{kind:?} {x:?}"
                );
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
                // The two zero sets share no member (SOS2 keeps r free in both).
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
        let split = split(&s, &[1.0, 0.0, 0.0, 0.0, 1.0], &none_fixed).unwrap();
        assert_eq!(split.left_zero, vec![13, 14]);
        assert_eq!(split.right_zero, vec![10, 11]);
    }

    #[test]
    fn sos1_split_uses_weighted_mean() {
        let s = set(SosKind::Sos1, 4);
        // Mass concentrated on the last member pulls the split up: mean
        // weight (1·1 + 4·9)/10 = 3.7 → r = position 2 (weight 3).
        let split = split(&s, &[1.0, 0.0, 0.0, 9.0], &none_fixed).unwrap();
        assert_eq!(split.left_zero, vec![13]);
        assert_eq!(split.right_zero, vec![10, 11, 12]);
    }

    #[test]
    fn violated_set_always_has_a_split_and_every_child_fixes_a_free_member() {
        // Members 10 and 11 already fixed to zero but carrying noise: only 12
        // and 13 count, so the split must fix one of THEM on each side.
        let s = set(SosKind::Sos1, 4);
        let fixed = |v: usize| v == 10 || v == 11;
        let split = split(&s, &[1e-12, -1e-12, 2.0, 3.0], &fixed).unwrap();
        assert!(split.left_zero.contains(&13));
        assert!(split.right_zero.contains(&12));
    }
}
