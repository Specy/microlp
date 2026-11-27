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
    /// Parent's LP objective in internal (minimize) space: a lower bound on
    /// the objective of this node's points up to the vertex freedom of the
    /// LP (ARCHITECTURE §7.1), used for pruning before any LP work.
    pub lp_bound: f64,
    /// `lp_bound` less the parent's dual slack (`MipState::lattice_bound`): a
    /// lower bound on the objective of every point of this node that meets
    /// its rows and bounds exactly, which the objective lattice rounds up.
    pub lattice_bound: f64,
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

/// The open nodes of a search: a list in push order, which a dive pops from
/// the end, with an index of `(lp_bound, position)` kept in step with it, so
/// that the best bound, and the first node holding it, cost a logarithm of
/// the list's length instead of a pass over it. Both were a pass per node
/// (the gap check, `super::global_bound_internal`, and every jump of
/// `super::search::pop_node`): 60 to 90% of the search's time on models
/// whose open list reaches 10^5 nodes.
///
/// The index orders bounds as `<` does, `-0.0` with `0.0`, and ties by
/// position, so the first of the best bounds is the one a scan with a
/// strict `<` would pick.
#[derive(Clone, Debug, Default)]
pub(crate) struct OpenNodes {
    nodes: Vec<Node>,
    index: std::collections::BTreeSet<(BoundKey, usize)>,
}

/// A node bound as the index orders it: `f64::total_cmp`, with `-0.0` read
/// as `0.0` so that the two compare equal, as under `<`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BoundKey(f64);

impl BoundKey {
    fn of(bound: f64) -> Self {
        BoundKey(if bound == 0.0 { 0.0 } else { bound })
    }
}

impl Eq for BoundKey {}

impl PartialOrd for BoundKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BoundKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl OpenNodes {
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Node> {
        self.nodes.iter()
    }

    pub(crate) fn push(&mut self, node: Node) {
        self.index
            .insert((BoundKey::of(node.lp_bound), self.nodes.len()));
        self.nodes.push(node);
    }

    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.index.clear();
    }

    /// Remove the node at `at`, the last node taking its place, as
    /// `Vec::swap_remove`.
    pub(crate) fn swap_remove(&mut self, at: usize) -> Node {
        let last = self.nodes.len() - 1;
        self.index
            .remove(&(BoundKey::of(self.nodes[at].lp_bound), at));
        if at != last {
            let moved = BoundKey::of(self.nodes[last].lp_bound);
            self.index.remove(&(moved, last));
            self.index.insert((moved, at));
        }
        self.nodes.swap_remove(at)
    }

    /// Swap the nodes at `a` and `b`, as `slice::swap`.
    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let (ka, kb) = (
            BoundKey::of(self.nodes[a].lp_bound),
            BoundKey::of(self.nodes[b].lp_bound),
        );
        self.index.remove(&(ka, a));
        self.index.remove(&(kb, b));
        self.index.insert((ka, b));
        self.index.insert((kb, a));
        self.nodes.swap(a, b);
    }

    /// The position of the first node with the lowest bound.
    pub(crate) fn best(&self) -> Option<usize> {
        self.index.first().map(|&(_, at)| at)
    }

    /// The lowest bound of an open node, `None` when there is none.
    pub(crate) fn min_bound(&self) -> Option<f64> {
        self.best().map(|at| self.nodes[at].lp_bound)
    }
}

impl std::ops::Index<usize> for OpenNodes {
    type Output = Node;

    fn index(&self, at: usize) -> &Node {
        &self.nodes[at]
    }
}

impl<'a> IntoIterator for &'a OpenNodes {
    type Item = &'a Node;
    type IntoIter = std::slice::Iter<'a, Node>;

    fn into_iter(self) -> Self::IntoIter {
        self.nodes.iter()
    }
}

#[cfg(test)]
mod open_tests {
    use super::*;

    fn node(lp_bound: f64, depth: u32) -> Node {
        Node {
            bound_changes: Vec::new(),
            basis: Basis(Vec::new()),
            lp_bound,
            lattice_bound: lp_bound,
            depth,
            parent_id: 0,
            branch_var: None,
            branch_up: false,
            branch_frac: 1.0,
        }
    }

    /// What `pop_node` scanned for before the index: the first position of
    /// the lowest bound, under a strict `<`.
    fn scanned_best(open: &OpenNodes) -> Option<usize> {
        if open.is_empty() {
            return None;
        }
        let mut best = 0;
        for i in 0..open.len() {
            if open[i].lp_bound < open[best].lp_bound {
                best = i;
            }
        }
        Some(best)
    }

    /// Through random pushes, removals, swaps and clears, with ties, zeros
    /// of both signs and infinite bounds, the index names the node the scan
    /// would, and its bound.
    #[test]
    fn the_index_picks_the_node_a_scan_would() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let bounds = [-1.0, 0.0, -0.0, 0.5, 2.0, f64::INFINITY, f64::NEG_INFINITY];
        let mut open = OpenNodes::default();
        for step in 0..20_000u32 {
            match next() % 10 {
                0..=4 => open.push(node(bounds[(next() % 7) as usize], step)),
                5..=6 if !open.is_empty() => {
                    let at = (next() % open.len() as u64) as usize;
                    open.swap_remove(at);
                }
                7 if open.len() > 1 => {
                    let a = (next() % open.len() as u64) as usize;
                    let b = (next() % open.len() as u64) as usize;
                    open.swap(a, b);
                }
                8 if next() % 50 == 0 => open.clear(),
                _ => {}
            }
            assert_eq!(open.best(), scanned_best(&open), "step {step}");
            assert_eq!(open.index.len(), open.len());
            if let Some(at) = open.best() {
                assert_eq!(open.min_bound(), Some(open[at].lp_bound));
            }
        }
    }
}
