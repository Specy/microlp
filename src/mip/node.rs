//! Plain-data branch & bound tree nodes.

use crate::solver::Basis;
use std::collections::BTreeMap;

/// One open node of the search tree. Contains no solver machinery — applying
/// `bound_changes` on top of the root bounds plus loading `basis` fully
/// reconstructs the node's starting state.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    /// Cumulative bound changes from the root, in creation order; for a var that
    /// appears multiple times the LAST entry is its current bounds.
    pub bound_changes: Vec<(usize, f64, f64)>,
    /// Optimal basis of the parent node (dual-feasible warm start after tightening).
    pub basis: Basis,
    /// Parent's LP objective in internal (minimize) space — a valid lower bound
    /// for this node, used for pruning before any LP work.
    pub lp_bound: f64,
    pub depth: u32,
    /// Sequence number of the branching that created this node; used to detect
    /// "the solver is already at my parent's optimum" (warm dive).
    pub parent_id: u64,
}

/// Collapse a bound-change list to one entry per var (later entries win),
/// sorted by var index for deterministic application order.
pub(crate) fn effective_bounds(changes: &[(usize, f64, f64)]) -> Vec<(usize, f64, f64)> {
    let mut map: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for &(v, lo, hi) in changes {
        map.insert(v, (lo, hi));
    }
    map.into_iter().map(|(v, (lo, hi))| (v, lo, hi)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_bounds_later_entries_win() {
        let changes = vec![(3, 0.0, 7.0), (1, 0.0, 1.0), (3, 2.0, 5.0)];
        assert_eq!(
            effective_bounds(&changes),
            vec![(1, 0.0, 1.0), (3, 2.0, 5.0)]
        );
        assert_eq!(effective_bounds(&[]), vec![]);
    }
}
