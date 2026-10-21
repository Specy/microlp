//! Plain-data branch & bound tree nodes.

use crate::solver::Basis;

/// One open node of the search tree. Contains no solver machinery — applying
/// `bound_changes` on top of the root bounds plus loading `basis` fully
/// reconstructs the node's starting state.
#[derive(Clone, Debug)]
pub(crate) struct Node {
    /// Bound changes from the root: one entry per branched var, its bounds
    /// from the last branching on it, sorted by var index.
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
    /// Which variable this node's creating branch changed. Root-like nodes that
    /// do not represent a variable branch carry `None` and cannot update a
    /// variable pseudocost.
    pub branch_var: Option<usize>,
    /// Direction and parent fractionality for a variable branch.
    pub branch_up: bool,
    pub branch_frac: f64,
}

/// `changes` with `var`'s bounds set to `(lo, hi)`, replacing its entry or
/// inserting one in var order: a child's bound changes from its parent's, so
/// a node holds one entry per var however deep it lies. `changes` must be as
/// a node stores them, sorted by var with one entry per var.
pub(crate) fn with_var_bounds(
    changes: &[(usize, f64, f64)],
    var: usize,
    lo: f64,
    hi: f64,
) -> Vec<(usize, f64, f64)> {
    debug_assert!(
        changes.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "bound changes out of var order or with a var twice"
    );
    let entry = (var, lo, hi);
    match changes.binary_search_by_key(&var, |&(v, _, _)| v) {
        Ok(i) => [&changes[..i], &[entry], &changes[i + 1..]].concat(),
        Err(i) => [&changes[..i], &[entry], &changes[i..]].concat(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_var_bounds_replaces_a_vars_entry_or_inserts_it_in_var_order() {
        let changes = [(1, 0.0, 1.0), (3, 0.0, 7.0)];
        assert_eq!(
            with_var_bounds(&changes, 3, 2.0, 5.0),
            vec![(1, 0.0, 1.0), (3, 2.0, 5.0)]
        );
        assert_eq!(
            with_var_bounds(&changes, 1, 1.0, 1.0),
            vec![(1, 1.0, 1.0), (3, 0.0, 7.0)]
        );
        assert_eq!(
            with_var_bounds(&changes, 0, -1.0, 0.0),
            vec![(0, -1.0, 0.0), (1, 0.0, 1.0), (3, 0.0, 7.0)]
        );
        assert_eq!(
            with_var_bounds(&changes, 2, 4.0, 4.0),
            vec![(1, 0.0, 1.0), (2, 4.0, 4.0), (3, 0.0, 7.0)]
        );
        assert_eq!(
            with_var_bounds(&changes, 5, 6.0, 9.0),
            vec![(1, 0.0, 1.0), (3, 0.0, 7.0), (5, 6.0, 9.0)]
        );
        assert_eq!(with_var_bounds(&[], 4, 0.0, 1.0), vec![(4, 0.0, 1.0)]);
    }
}
