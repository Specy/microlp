//! The search itself: visiting, branching, warm-starting and pruning nodes,
//! and the loop that drives them until a termination reason is proven.

use super::node::{with_var_bounds, Node};
use super::params;
use super::state::{Incumbent, MipState};
use super::{
    branching, candidate_variables_feasible, dominated, global_bound_internal, relative_gap,
    TerminationReason,
};
use crate::solver::check_deadline;
use crate::{Error, StopReason, VarDomain};

/// Validate and adopt the solver's current point, as the engine computed it:
/// every bound and row of the model (the rows presolve dropped or rewrote
/// included) within the contract, and every integer var within `int_tol` of
/// an integer (one the node fixes lies on its value, where the engine holds
/// it). The bounds are the root's: the node's are not always within them,
/// since a warm-start hint fixes a var up to `params::HINT_BOUNDS_SLACK`
/// outside them, and the var may sit its whole tolerance beyond that.
/// `Ok(false)` means the point fails the check and the caller must branch. A
/// valid candidate completes unboundedness classification. A candidate whose
/// objective cannot be evaluated in `f64`, its terms (costs times values) or
/// their sum overflowing, is an `InternalError`: no bound or incumbent
/// compares with it, and no branching changes it.
pub(crate) fn try_adopt_incumbent(state: &mut MipState) -> Result<bool, Error> {
    let values = state.solver.reported_values();
    let solver = &state.solver;
    let domains = &solver.orig_var_domains;
    let (tolerances, int_tol) = (&state.options.tolerances, state.options.int_tol);
    let root_bounds = &state.root_bounds;
    if !candidate_variables_feasible(&values, domains, tolerances.feasibility, int_tol, |v| {
        root_bounds[v]
    }) {
        debug!("integral-within-tol solution rejected: a value violates its bounds or domain");
        return Ok(false);
    }
    if let Some((row, violation)) = solver.first_violated_row(&values) {
        debug!("integral-within-tol solution rejected: it violates row {row} by {violation:e}");
        return Ok(false);
    }
    let objective = solver.objective_of(&values);
    if !objective.is_finite() {
        return Err(Error::InternalError(
            "the objective of an integral solution cannot be evaluated in f64: its terms \
             (costs times values) or their sum overflow"
                .to_string(),
        ));
    }
    let better = match &state.incumbent {
        Some(inc) => objective < inc.objective,
        None => true,
    };
    if better {
        debug!("new incumbent, internal obj: {:.6}", objective);
        state.incumbent = Some(Incumbent { values, objective });
    }
    if state.classifying_unbounded {
        Err(Error::Unbounded)
    } else {
        Ok(true)
    }
}

enum IntegralCandidate {
    Closed,
    Branch(usize),
}

/// Adopt the node's LP point, integral within `int_tol`, and close the
/// subtree: its objective is the node's bound. The engine verified its own
/// rows and bounds at these very values, within their contract less the
/// round-off of its own evaluation, which the guard's compensated sums leave
/// it, so a point fails the guard only on a row presolve dropped or rewrote. Such
/// a row holds where the integer vars are exactly integral, but an LP value
/// within `int_tol` of an integer, or past a bound within its tolerance, can
/// break it (a rewritten coefficient is exact only at an integral binary):
/// the node then branches at `int_tol = 0` on an integer var off its integer,
/// which the child that fixes it holds exactly at its value. A point that
/// fails with every integer var exactly integral, nothing left to branch on,
/// is an `InternalError`: presolve's reduction did not keep it.
fn process_integral_candidate(
    state: &mut MipState,
    domains: &[VarDomain],
) -> Result<IntegralCandidate, Error> {
    if try_adopt_incumbent(state)? {
        return Ok(IntegralCandidate::Closed);
    }
    match branching::choose_branch_var(&state.solver, domains, 0.0, &state.pseudocosts) {
        Some(var) => Ok(IntegralCandidate::Branch(var)),
        None => Err(Error::InternalError(
            "an integral solution the search cannot branch on failed feasibility validation"
                .to_string(),
        )),
    }
}

/// Apply `node`'s bounds to the solver: reset each var in `applied` that the
/// node has no entry for to its root bounds, then set every entry of the
/// node, also one already in force, which drops any widening a previous
/// solve left on it (`Solver::set_var_bounds`). Returns false (node pruned,
/// solver untouched) if the node's bounds cross.
fn apply_node_bounds(state: &mut MipState, node: &Node) -> bool {
    let target = &node.bound_changes;
    if target.iter().any(|&(_, lo, hi)| lo > hi) {
        return false;
    }
    // Reset vars that are currently changed but absent from the target.
    for &(v, _, _) in &state.applied {
        if target.binary_search_by_key(&v, |t| t.0).is_err() {
            let (rlo, rhi) = state.root_bounds[v];
            state
                .solver
                .set_var_bounds(v, rlo, rhi)
                .expect("root bounds cannot cross");
        }
    }
    // Set every target bound, those in force too (validated above, cannot fail).
    for &(v, lo, hi) in target {
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("validated bounds cannot cross");
    }
    state.applied.clone_from(target);
    true
}

/// Branch on `var` at the solver's current (just solved) optimum: push the two
/// children carrying the parent's basis, objective bound and lattice bound.
fn branch(state: &mut MipState, parent: &Node, var: usize) {
    let z = state.solver.cur_obj_val;
    let lattice_bound = state.lattice_bound(z);
    let val = state.solver.get_value(var);
    let (lo, hi) = state.solver.get_var_bounds(var);
    // The split point k (children: x ≤ k and x ≥ k + 1) must be
    // noise-robust: a raw `val.floor()` of a within-tolerance-integral value
    // is catastrophic — floor(−8e-16) = −1 makes the up child
    // (max(0, lo), hi) reproduce the parent VERBATIM and the search
    // descends forever. Reachable through the rejected-candidate re-branch
    // path (`choose_branch_var` at int_tol = 0) whenever LP noise puts an
    // integer var a hair below an integer. Take near-integral values at
    // their integer first, then clamp k into [lo, hi − 1] so BOTH children
    // strictly tighten the parent's [lo, hi] whenever hi − lo ≥ 1
    // (integral bounds — guaranteed for branchable vars).
    let floor = {
        let near = val.round();
        let k = if (val - near).abs() <= state.options.int_tol {
            near
        } else {
            val.floor()
        };
        k.clamp(lo, (hi - 1.0).max(lo))
    };
    let f_down = (val - floor).clamp(0.0, 1.0);

    state.node_seq += 1;
    let id = state.node_seq;
    state.last_solved_id = Some(id);
    let basis = state.solver.snapshot_basis();

    let down_node = Node {
        bound_changes: with_var_bounds(&parent.bound_changes, var, lo, floor),
        basis: basis.clone(),
        lp_bound: z,
        lattice_bound,
        depth: parent.depth + 1,
        parent_id: id,
        branch_var: Some(var),
        branch_up: false,
        branch_frac: f_down,
    };
    let up_node = Node {
        bound_changes: with_var_bounds(&parent.bound_changes, var, floor + 1.0, hi),
        basis,
        lp_bound: z,
        lattice_bound,
        depth: parent.depth + 1,
        parent_id: id,
        branch_var: Some(var),
        branch_up: true,
        branch_frac: 1.0 - f_down,
    };

    // Estimate-ordered dive: push the child with the LARGER estimated degradation
    // first, so the cheaper (more promising) direction is popped/dived first.
    let est_down = state.pseudocosts.estimate(var, false) * f_down;
    let est_up = state.pseudocosts.estimate(var, true) * (1.0 - f_down);
    if est_down > est_up {
        // up is cheaper → push it last so it is dived first
        state.open.push(down_node);
        state.open.push(up_node);
    } else {
        state.open.push(up_node);
        state.open.push(down_node);
    }
    // Children were pushed: keep plunging (LIFO pop) into this subtree.
    state.diving = true;
}

/// Outcome of solving one branch & bound node's LP relaxation.
pub(super) enum NodeLp {
    /// The solver now holds this node's optimal basis and objective.
    Solved,
    /// The node's LP is infeasible under its current bounds. The engine
    /// returns [`Error::Infeasible`] only on bounds that cross or on a proof
    /// checked on the model's data (a certificate whose gap exceeds the
    /// tolerance, `Solver::prove_infeasible`); a row it can neither repair nor
    /// prove infeasible is an error instead, so an unproven verdict never
    /// prunes a feasible subtree. The proof's one concession: a coefficient on
    /// an infinite bound counts as zero where the error of the multipliers
    /// could make it zero (`Solver::proof_of`). A node is pruned wrongly only
    /// if a genuine coefficient lies within that error, which the arithmetic
    /// cannot tell from zero, and times its var's value makes room at every
    /// point of the node.
    Infeasible,
    /// A limit interrupted the solve (possibly during the slack retry); the
    /// solver's unfinished, non-optimal state must not be used as a node result.
    Limit,
}

/// Solve the current node's LP relaxation. Bounds and (if needed) the warm basis
/// are already loaded into the solver by the caller.
///
/// Robustness valve: if the first `reoptimize` fails with anything but
/// [`Error::Infeasible`] — an internal error, e.g. a row the dual simplex can
/// neither repair nor prove infeasible even after the engine's own start from
/// the slack basis, or an [`Error::Unbounded`], which a node of the search
/// reports only numerically — fall back ONCE to the all-slack basis and
/// re-solve the node from scratch, then take that retry's outcome as final.
/// The retry cannot loop: it is attempted at most once, and its own error is
/// propagated, an `Unbounded` as an internal error. Only [`Error::Infeasible`]
/// prunes the node (see [`NodeLp::Infeasible`]).
pub(super) fn solve_node_lp(state: &mut MipState) -> Result<NodeLp, Error> {
    state.solver.deadline = state.deadline;
    let err = match state.solver.reoptimize() {
        Ok(StopReason::Finished) => return Ok(NodeLp::Solved),
        Ok(StopReason::Limit) => return Ok(NodeLp::Limit),
        Err(Error::Infeasible) => return Ok(NodeLp::Infeasible),
        // Any other outcome, `Unbounded` included: fall through to the
        // one-shot slack retry.
        Err(e) => e,
    };

    debug!(
        "node LP reoptimize failed ({}); retrying from slack basis",
        err
    );
    let slack = state.solver.slack_basis();
    state.solver.load_basis(&slack)?;
    match state.solver.reoptimize() {
        Ok(StopReason::Finished) => Ok(NodeLp::Solved),
        Ok(StopReason::Limit) => Ok(NodeLp::Limit),
        Err(Error::Infeasible) => Ok(NodeLp::Infeasible),
        Err(Error::Unbounded) => Err(Error::InternalError(
            "bounded B&B node reported unbounded".to_string(),
        )),
        // The retry also failed internally: propagate the error this time.
        Err(e) => Err(e),
    }
}

/// Pop policy: keep diving (DFS) while the last processed node produced children;
/// when a dive dies out, jump to the open node with the best (lowest) bound. Ties
/// in `lp_bound` resolve to the first (lowest-index) such node in `open` — the
/// scan uses strict `<`, so `best` only moves for a strictly smaller bound.
/// Returns the node and the index it held, which `swap_remove` refills with the
/// last node ([`put_back`] undoes the pop).
pub(super) fn pop_node(state: &mut MipState) -> Option<(Node, usize)> {
    if state.open.is_empty() {
        return None;
    }
    let at = if state.diving {
        state.open.len() - 1
    } else {
        let mut best = 0;
        for (i, n) in state.open.iter().enumerate() {
            if n.lp_bound < state.open[best].lp_bound {
                best = i;
            }
        }
        best
    };
    Some((state.open.swap_remove(at), at))
}

/// Undo [`pop_node`] of `node` from index `at`: `open` as it was before the
/// pop, so that with `diving` as it was too, a resumed search pops the nodes
/// an uninterrupted one would.
pub(super) fn put_back(state: &mut MipState, node: Node, at: usize) {
    state.open.push(node);
    let last = state.open.len() - 1;
    state.open.swap(at, last);
}

/// Evaluate a warm-start hint: fix hinted vars, LP-complete the rest, and if the
/// completion is feasible and integral adopt it as the initial incumbent.
/// Advisory by design — every failure path just drops the hint, an engine error
/// in the completion included: the search solves the whole model without it.
/// Always restores the solver to the root optimum before returning.
///
/// Returns `Ok(None)` on the normal path (the caller proceeds to build the root
/// node). `Err` is either the `Unbounded` verdict, when the completion is the
/// integer point an unboundedness classification looks for, or an error
/// restoring the root. Returns `Ok(Some(TerminationReason::TimeLimit))` when the
/// deadline strikes during the re-solve of the restored root basis: rather than
/// let the caller read an unfinished `cur_obj_val` as the root bound, it un-sets
/// `root_solved` so a resume re-enters `initial_solve` and continues honestly
/// from the solver's feasibility flags.
pub(crate) fn try_warm_start(
    state: &mut MipState,
    hints: &[(crate::Variable, f64)],
) -> Result<Option<TerminationReason>, Error> {
    let domains = state.solver.orig_var_domains.clone();
    let root_basis = state.solver.snapshot_basis();
    let mut applied: Vec<usize> = Vec::new();
    let mut ok = true;
    // `try_adopt_incumbent`'s `Unbounded` verdict, returned once the root is restored.
    let mut verdict = None;

    for &(var, val) in hints {
        let v = var.idx();
        // The solve refuses a hint naming a var outside the problem or with a
        // value that is not finite (`Problem::solve_with`), and an edit's hint
        // is a validated incumbent.
        debug_assert!(
            v < state.solver.num_vars && val.is_finite(),
            "a hint the solve's validation refuses reached the search"
        );
        let val = if matches!(
            domains.get(v),
            Some(VarDomain::Integer | VarDomain::Boolean)
        ) {
            val.round()
        } else {
            val
        };
        let (lo, hi) = state.root_bounds[v];
        if val < lo - params::HINT_BOUNDS_SLACK || val > hi + params::HINT_BOUNDS_SLACK {
            ok = false;
            break;
        }
        // A value the engine's scaled units cannot hold is no usable hint
        // either.
        if state.solver.set_var_bounds(v, val, val).is_err() {
            ok = false;
            break;
        }
        applied.push(v);
    }

    if ok {
        match state.solver.reoptimize() {
            Ok(StopReason::Finished) => {
                if branching::is_integral(&state.solver, &domains, state.options.int_tol) {
                    // The incumbent's feasibility guard: never bypass it for a hint.
                    // If it rejects the completion, drop the hint — do not branch
                    // below-tolerance vars here, that fallback is only for the main
                    // search loop.
                    match try_adopt_incumbent(state) {
                        Ok(true) => {}
                        Ok(false) => {
                            debug!("warm-start hint rejected by feasibility guard; ignored");
                        }
                        Err(Error::Unbounded) => verdict = Some(Error::Unbounded),
                        // A completion whose objective cannot be evaluated is
                        // no usable hint either.
                        Err(error) => {
                            debug!("warm-start hint rejected ({error}); ignored");
                        }
                    }
                } else {
                    debug!("warm-start hint LP-completed fractionally; ignored");
                }
            }
            Ok(StopReason::Limit) | Err(Error::Infeasible) => {
                debug!("warm-start hint infeasible or out of time; ignored");
            }
            // The engine's own error, even `Unbounded`: the completion only
            // restricts the bounded root LP.
            Err(error) => {
                debug!("warm-start hint LP completion failed ({error}); ignored");
            }
        }
    } else {
        debug!("warm-start hint out of bounds or beyond the engine's units; ignored");
    }

    // Restore the root state: bounds back, then the optimal root basis
    // (load_basis recomputes everything, discarding the hint solve), re-solved
    // in case the load had to repair it, which otherwise only verifies it.
    for v in applied {
        let (lo, hi) = state.root_bounds[v];
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("root bounds cannot cross");
    }
    state.solver.load_basis(&root_basis)?;
    if state.solver.reoptimize()? == StopReason::Limit {
        // A limit during the root re-solve leaves `cur_obj_val` unsuitable
        // as a root bound. Mark the root unsolved and return Interrupted;
        // `initialize_root` continues `initial_solve` from the solver's
        // feasibility flags on resume.
        state.root_solved = false;
        if let Some(verdict) = verdict {
            return Err(verdict);
        }
        return Ok(Some(TerminationReason::TimeLimit));
    }
    if let Some(verdict) = verdict {
        return Err(verdict);
    }
    Ok(None)
}

/// Solve or resume the root relaxation, restore any advisory warm start, and
/// either close the problem or seed the open tree. `None` means node processing
/// can begin; `Some` is a completed or interrupted root outcome.
fn initialize_root(
    state: &mut MipState,
    domains: &[VarDomain],
) -> Result<Option<TerminationReason>, Error> {
    if state.root_solved {
        return Ok(None);
    }

    if state.solver.initial_solve()? == StopReason::Limit {
        return Ok(Some(TerminationReason::TimeLimit));
    }
    state.root_solved = true;

    if let Some(hints) = state.options.warm_start.take() {
        if let Some(outcome) = try_warm_start(state, &hints)? {
            return Ok(Some(outcome));
        }
    }

    let root = Node {
        bound_changes: Vec::new(),
        basis: state.solver.snapshot_basis(),
        lp_bound: state.solver.cur_obj_val,
        lattice_bound: state.lattice_bound(state.solver.cur_obj_val),
        depth: 0,
        parent_id: 0,
        branch_var: None,
        branch_up: false,
        branch_frac: 1.0,
    };
    let int_tol = state.options.int_tol;
    if branching::is_integral(&state.solver, domains, int_tol) {
        match process_integral_candidate(state, domains)? {
            IntegralCandidate::Branch(var) => branch(state, &root, var),
            IntegralCandidate::Closed => {
                return Ok(Some(TerminationReason::ProvenOptimal));
            }
        }
    } else {
        let var = fractional_branch_var(state, domains, int_tol);
        branch(state, &root, var);
    }

    Ok(None)
}

enum NodeVisit {
    /// The node was discarded before an LP solve, so it consumes no node budget.
    Pruned,
    /// One node LP completed, including an infeasible relaxation.
    Solved,
    /// The node must be restored to the frontier. `lp_solved` distinguishes an
    /// interrupted initial LP from an interrupted retry after a completed LP.
    Interrupted { node: Node, lp_solved: bool },
}

/// Reconstruct and process one node selected by the outer search policy.
fn visit_node(state: &mut MipState, node: Node, domains: &[VarDomain]) -> Result<NodeVisit, Error> {
    if !apply_node_bounds(state, &node) {
        state.diving = false;
        return Ok(NodeVisit::Pruned);
    }

    let warm = state.last_solved_id == Some(node.parent_id);
    if !warm {
        state.solver.load_basis(&node.basis)?;
    }

    match solve_node_lp(state)? {
        NodeLp::Solved => {}
        NodeLp::Infeasible => {
            state.last_solved_id = None;
            state.diving = false;
            return Ok(NodeVisit::Solved);
        }
        NodeLp::Limit => {
            return Ok(NodeVisit::Interrupted {
                node,
                lp_solved: false,
            })
        }
    }

    let objective = state.solver.cur_obj_val;
    if let Some(var) = node.branch_var {
        state.pseudocosts.record(
            var,
            node.branch_up,
            (objective - node.lp_bound).max(0.0) / node.branch_frac.max(params::BRANCH_FRAC_GUARD),
        );
    }
    let lattice_bound = state.lattice_bound(objective);
    if dominated(state, objective, lattice_bound) {
        state.last_solved_id = None;
        state.diving = false;
        return Ok(NodeVisit::Solved);
    }

    let int_tol = state.options.int_tol;
    if branching::is_integral(&state.solver, domains, int_tol) {
        match process_integral_candidate(state, domains)? {
            IntegralCandidate::Branch(var) => branch(state, &node, var),
            IntegralCandidate::Closed => {
                state.last_solved_id = None;
                state.diving = false;
            }
        }
        return Ok(NodeVisit::Solved);
    }

    let var = fractional_branch_var(state, domains, int_tol);
    branch(state, &node, var);
    Ok(NodeVisit::Solved)
}

/// The var to branch on at a point that is not integral within `int_tol`:
/// one exists, since `branching::is_integral` weighs only the vars the search
/// can branch on.
fn fractional_branch_var(state: &MipState, domains: &[VarDomain], int_tol: f64) -> usize {
    branching::choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts)
        .expect("a point that is not integral has an integer var the search can branch on")
}

pub(crate) fn search_loop(state: &mut MipState) -> Result<TerminationReason, Error> {
    let domains = state.solver.orig_var_domains.clone();
    state.solver.deadline = state.deadline;

    if let Some(outcome) = initialize_root(state, &domains)? {
        return Ok(outcome);
    }

    let mut nodes_this_run: u64 = 0;

    loop {
        // Tree exhausted → the proof is COMPLETE: fall through to the post-loop
        // incumbent-vs-Infeasible verdict. Checked before the gap/deadline/node
        // limit tests so a limit that lands on the exact iteration the tree empties
        // never masks a finished proof as Interrupted/Feasible. (The loop is only
        // entered with `root_solved` true; the bottom `pop_node → None → break`
        // remains a safety net for any other path that empties `open` mid-body.)
        if state.open.is_empty() {
            break;
        }

        // Proof-quality stops: exact incumbent/bound equality wins; otherwise
        // an incumbent within `mip_gap` is a valid feasible result. Checked
        // first so proof quality takes priority over limit interruptions.
        if state.options.mip_gap > 0.0 {
            if let (Some(inc), Some(bound)) = (&state.incumbent, global_bound_internal(state)) {
                // The open tree may still contain nodes whose stored bounds
                // equal the incumbent. Equality of the incumbent and global
                // bound is nevertheless a complete proof, so it must not be
                // weakened to a gap-satisfied feasible result.
                if bound >= inc.objective {
                    state.open.clear();
                    break;
                }
                if relative_gap(inc.objective, bound) <= state.options.mip_gap {
                    return Ok(TerminationReason::MipGap);
                }
            }
        }

        // Global limits are checked between nodes; no unfinished node result is consulted.
        if check_deadline(&state.deadline) == StopReason::Limit {
            return Ok(TerminationReason::TimeLimit);
        }
        let (node, at) = match pop_node(state) {
            Some(popped) => popped,
            None => break,
        };

        // Prune with the stored parent bound before any LP work.
        if dominated(state, node.lp_bound, node.lattice_bound) {
            state.diving = false;
            continue;
        }

        // A node budget limits LP solves, not free bookkeeping. Pop and apply
        // the stored-bound prune first so hitting the exact solve count can
        // still finish a proof whose remaining nodes are already dominated by
        // the incumbent. The node goes back where it was, the dive as it was,
        // and the solver still sits where the last node left it: a resume
        // continues the very search.
        if let Some(nl) = state.options.node_limit {
            if nodes_this_run >= nl {
                put_back(state, node, at);
                return Ok(TerminationReason::NodeLimit);
            }
        }

        match visit_node(state, node, &domains)? {
            NodeVisit::Pruned => continue,
            NodeVisit::Solved => {
                state.stats.nodes_solved += 1;
                nodes_this_run += 1;
            }
            NodeVisit::Interrupted { node, lp_solved } => {
                if lp_solved {
                    state.stats.nodes_solved += 1;
                }
                // Back where it was, the dive as it was; its LP, which the
                // limit cut short, is solved again from its stored basis.
                put_back(state, node, at);
                state.last_solved_id = None;
                return Ok(TerminationReason::TimeLimit);
            }
        }
    }

    if state.incumbent.is_some() {
        Ok(TerminationReason::ProvenOptimal)
    } else {
        Err(Error::Infeasible)
    }
}
