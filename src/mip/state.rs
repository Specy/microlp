//! The complete, resumable state of one branch & bound search.
//!
//! [`MipState`] is what a solve carries between nodes and what a resumed or
//! post-solve-edited call picks back up; it owns the single [`Solver`]
//! instance the whole search pivots on.

use super::branching;
use super::node::Node;
use super::options::{SolveOptions, Stats, TerminationReason};
use crate::solver::{Deadline, Solver};
use crate::{OptimizationDirection, Problem};
use std::collections::BTreeMap;

/// A feasible integer assignment, in internal (minimize) objective space.
#[derive(Clone, Debug)]
pub(crate) struct Incumbent {
    /// Values of the structural variables (length = Problem var count).
    pub values: Vec<f64>,
    pub objective: f64,
}

/// The complete, resumable state of a branch & bound search.
#[derive(Clone)]
pub(crate) struct MipState {
    pub solver: Solver,
    /// Original bounds of the structural vars (to reset when jumping between nodes).
    pub root_bounds: Vec<(f64, f64)>,
    /// Bound changes currently applied to `solver` (collapsed, sorted by var).
    pub applied: Vec<(usize, f64, f64)>,
    pub open: Vec<Node>,
    /// Pop policy toggle for `pop_node`: `true` while the most recently processed
    /// node pushed children (keep plunging via LIFO pop); `false` once a dive dies
    /// out with no children pushed, or once a node is requeued unsolved by an
    /// interruption — the next pop then jumps to the open node with the best
    /// (lowest) bound instead of blindly continuing the old dive.
    pub diving: bool,
    pub incumbent: Option<Incumbent>,
    /// Sequence counter for branchings; children carry it as `parent_id`.
    pub node_seq: u64,
    /// `Some(id)` iff `solver` currently holds the optimal basis + bounds of the
    /// branching with that id — its children can skip the basis load (warm dive).
    pub last_solved_id: Option<u64>,
    pub root_solved: bool,
    pub stats: Stats,
    pub options: SolveOptions,
    pub deadline: Deadline,
    /// Consumed by `fill_bound_stats` to report `best_bound`/`gap` in user space.
    pub direction: OptimizationDirection,
    /// Learned per-variable branching degradation estimates, updated after each
    /// node LP solve and consulted by `branching::choose_branch_var`.
    pub pseudocosts: branching::PseudoCosts,
    /// Clean copy of the user's problem, including post-solve edits — never
    /// contains branching artifacts. Post-solve edits re-solve from this.
    pub base: Problem,
    /// User-level fix_var overlay on `base` (var → fixed value).
    pub fixed: BTreeMap<usize, f64>,
    /// True while a zero-objective search classifies an unbounded LP
    /// relaxation as either integer-feasible (the original MILP is unbounded)
    /// or integer-infeasible.
    pub classifying_unbounded: bool,
}

impl std::fmt::Debug for MipState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipState")
            .field("open_nodes", &self.open.len())
            .field("has_incumbent", &self.incumbent.is_some())
            .field("stats", &self.stats)
            .field("diving", &self.diving)
            .field(
                "pseudocost_observations",
                &self.pseudocosts.observation_count(),
            )
            .field("classifying_unbounded", &self.classifying_unbounded)
            .finish()
    }
}

impl MipState {
    /// Objective of the solution exposed through the public value accessors.
    /// Unboundedness classification uses a zero-objective solver, so a working
    /// point without an incumbent is evaluated against the original model.
    pub(crate) fn current_objective(&self) -> f64 {
        if let Some(incumbent) = &self.incumbent {
            return incumbent.objective;
        }
        self.base
            .obj_coeffs
            .iter()
            .enumerate()
            .map(|(v, &coefficient)| coefficient * self.solver.get_value(v))
            .sum()
    }
}

#[derive(Debug)]
pub(crate) struct MipRun {
    pub reason: TerminationReason,
    pub state: MipState,
}
