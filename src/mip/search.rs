//! The search itself: how one node is visited, branched, warm-started and
//! pruned, and the loop that drives those steps until a termination reason
//! is proven.

use super::node::{effective_bounds, Node};
use super::params;
use super::state::{Incumbent, MipState};
use super::{
    branching, candidate_variables_feasible, cutoff, global_bound_internal, relative_gap,
    TerminationReason,
};
use crate::solver::check_deadline;
use crate::{Error, StopReason, VarDomain};

/// Validate and adopt the solver's current solution using integer-rounded
/// values. `Ok(false)` means rounding produced an invalid candidate and the
/// caller must branch. A valid candidate completes unboundedness classification.
pub(crate) fn try_adopt_incumbent(state: &mut MipState) -> Result<bool, Error> {
    let tolerances = &state.options.tolerances;
    let mut values = state.solver.reported_values();
    let solver = &state.solver;
    let domains = &solver.orig_var_domains;
    for (val, dom) in values.iter_mut().zip(domains.iter()) {
        if matches!(dom, VarDomain::Integer | VarDomain::Boolean) {
            *val = val.round();
        }
    }
    if !candidate_variables_feasible(&values, domains, tolerances, |v| state.root_bounds[v]) {
        debug!(
            "integral-within-tol solution rejected: a rounded value violates its bounds or domain"
        );
        return Ok(false);
    }
    if let Some((row, violation)) = solver.first_violated_row(&values) {
        debug!(
            "integral-within-tol solution rejected: rounded values violate row {row} by {violation:e}"
        );
        return Ok(false);
    }
    let objective = solver.objective_of(&values);
    if !objective.is_finite() {
        debug!("integral-within-tol solution rejected: objective is non-finite");
        return Ok(false);
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

/// Adopt a feasible rounded candidate, but close the current subtree only when
/// the LP point itself is exactly integral. An exactly integral point that
/// fails the feasibility guard is a contradiction: the engine reports
/// `Finished` only after verifying the very values it reports through the
/// very evaluation the guard repeats (`Solver::check_rows` and
/// `Solver::first_violated_row` share it), with half the contract to spare,
/// so this cannot be round-off. It is reported loudly rather than papered
/// over.
fn process_integral_candidate(
    state: &mut MipState,
    domains: &[VarDomain],
    _int_tol: f64,
) -> Result<IntegralCandidate, Error> {
    let adopted = try_adopt_incumbent(state)?;
    if let Some(var) = branching::choose_branch_var(&state.solver, domains, 0.0, &state.pseudocosts)
    {
        return Ok(IntegralCandidate::Branch(var));
    }
    if adopted {
        Ok(IntegralCandidate::Closed)
    } else {
        Err(Error::InternalError(
            "exactly integral solution failed feasibility validation".to_string(),
        ))
    }
}

/// Apply `node`'s bounds to the solver, diffing against what is currently applied.
/// Returns false (node pruned, solver untouched) if the node's bounds cross.
fn apply_node_bounds(state: &mut MipState, node: &Node) -> bool {
    let target = effective_bounds(&node.bound_changes);
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
    // Apply the target bounds (validated above, cannot fail).
    for &(v, lo, hi) in &target {
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("validated bounds cannot cross");
    }
    state.applied = target;
    true
}

/// Branch on `var` at the solver's current (just solved) optimum: push the two
/// children carrying the parent's basis and objective bound.
fn branch(state: &mut MipState, parent: &Node, var: usize) {
    let z = state.solver.cur_obj_val;
    let val = state.solver.get_value(var);
    let (lo, hi) = state.solver.get_var_bounds(var);
    // The split point k (children: x ≤ k and x ≥ k + 1) must be
    // noise-robust: a raw `val.floor()` of a within-tolerance-integral value
    // is catastrophic — floor(−8e-16) = −1 makes the up child
    // (max(0, lo), hi) reproduce the parent VERBATIM and the search
    // descends forever. Reachable through the rounding-rejected re-branch
    // path (`choose_branch_var` at int_tol = 0) whenever LP noise puts an
    // integer var a hair below an integer. Snap near-integral values to
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

    let mut down_changes = parent.bound_changes.clone();
    down_changes.push((var, lo, floor));
    let mut up_changes = parent.bound_changes.clone();
    up_changes.push((var, floor + 1.0, hi));

    let down_node = Node {
        bound_changes: down_changes,
        basis: basis.clone(),
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
        branch_var: Some(var),
        branch_up: false,
        branch_frac: f_down,
    };
    let up_node = Node {
        bound_changes: up_changes,
        basis,
        lp_bound: z,
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
enum NodeLp {
    /// The solver now holds this node's optimal basis and objective.
    Solved,
    /// The node's LP is infeasible under its current bounds.
    Infeasible,
    /// A limit interrupted the solve (possibly during the slack retry); the
    /// solver's unfinished, non-optimal state must not be used as a node result.
    Limit,
}

/// Solve the current node's LP relaxation. Bounds and (if needed) the warm basis
/// are already loaded into the solver by the caller.
///
/// Robustness valve: if the first `reoptimize` fails with an internal error class
/// — anything that is neither [`Error::Infeasible`] nor [`Error::Unbounded`], e.g.
/// a singular LU produced by numerical degradation during pivoting — fall back
/// ONCE to the all-slack basis (which is documented to always load) and re-solve
/// the node from scratch, then take that retry's outcome as final. The retry
/// cannot loop: it is attempted at most once and its own internal error is
/// propagated rather than retried again.
fn solve_node_lp(state: &mut MipState) -> Result<NodeLp, Error> {
    state.solver.deadline = state.deadline;
    let err = match state.solver.reoptimize() {
        Ok(StopReason::Finished) => return Ok(NodeLp::Solved),
        Ok(StopReason::Limit) => return Ok(NodeLp::Limit),
        Err(Error::Infeasible) => return Ok(NodeLp::Infeasible),
        Err(Error::Unbounded) => {
            return Err(Error::InternalError(
                "bounded B&B node reported unbounded".to_string(),
            ))
        }
        // Internal/singular error class: fall through to the one-shot slack retry.
        Err(e) => e,
    };

    debug!(
        "node LP reoptimize failed ({}); retrying from slack basis",
        err
    );
    let slack = state.solver.slack_basis();
    state
        .solver
        .load_basis(&slack)
        .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
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
fn pop_node(state: &mut MipState) -> Option<Node> {
    if state.open.is_empty() {
        return None;
    }
    if state.diving {
        state.open.pop()
    } else {
        let mut best = 0;
        for (i, n) in state.open.iter().enumerate() {
            if n.lp_bound < state.open[best].lp_bound {
                best = i;
            }
        }
        Some(state.open.swap_remove(best))
    }
}

/// Evaluate a warm-start hint: fix hinted vars, LP-complete the rest, and if the
/// completion is feasible and integral adopt it as the initial incumbent.
/// Advisory by design — every failure path just drops the hint. Always restores
/// the solver to the root optimum before returning.
///
/// Returns `Ok(None)` on the normal path (the caller proceeds to build the root
/// node). Returns `Ok(Some(TerminationReason::TimeLimit))` only when the restore
/// of the root basis fails AND the deadline strikes mid-restore: rather than let the
/// caller read an unfinished `cur_obj_val` as the root bound, it un-sets
/// `root_solved` so a resume re-enters `initial_solve` and continues honestly from
/// the solver's feasibility flags.
pub(crate) fn try_warm_start(
    state: &mut MipState,
    hints: &[(crate::Variable, f64)],
) -> Result<Option<TerminationReason>, Error> {
    let domains = state.solver.orig_var_domains.clone();
    let root_basis = state.solver.snapshot_basis();
    let mut applied: Vec<usize> = Vec::new();
    let mut ok = true;
    let mut pending_error = None;

    for &(var, val) in hints {
        let v = var.idx();
        if v >= state.solver.num_vars || !val.is_finite() {
            ok = false;
            break;
        }
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
        state
            .solver
            .set_var_bounds(v, val, val)
            .expect("fixing to [val, val] cannot cross");
        applied.push(v);
    }

    if ok {
        match state.solver.reoptimize() {
            Ok(StopReason::Finished) => {
                if branching::is_integral(&state.solver, &domains, state.options.int_tol) {
                    // Rounded-incumbent feasibility guard: never bypass it for a hint.
                    // If it rejects the completion, drop the hint — do not branch
                    // below-tolerance vars here, that fallback is only for the main
                    // search loop.
                    match try_adopt_incumbent(state) {
                        Ok(true) => {}
                        Ok(false) => {
                            debug!("warm-start hint rejected by feasibility guard; ignored");
                        }
                        Err(error) => pending_error = Some(error),
                    }
                } else {
                    debug!("warm-start hint LP-completed fractionally; ignored");
                }
            }
            Ok(StopReason::Limit) | Err(Error::Infeasible) => {
                debug!("warm-start hint infeasible or out of time; ignored");
            }
            Err(error) => pending_error = Some(error),
        }
    } else {
        debug!(
            "warm-start hint invalid (unknown variable, non-finite value, or out of bounds); ignored"
        );
    }

    // Restore the root state exactly: bounds back, then the optimal root basis
    // (load_basis recomputes everything, discarding the hint solve).
    for v in applied {
        let (lo, hi) = state.root_bounds[v];
        state
            .solver
            .set_var_bounds(v, lo, hi)
            .expect("root bounds cannot cross");
    }
    if state.solver.load_basis(&root_basis).is_err() {
        let slack = state.solver.slack_basis();
        state
            .solver
            .load_basis(&slack)
            .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
        if state.solver.reoptimize()? == StopReason::Limit {
            // A limit during the root re-solve leaves `cur_obj_val` unsuitable
            // as a root bound. Mark the root unsolved and return Interrupted;
            // `initialize_root` continues `initial_solve` from the solver's
            // feasibility flags on resume.
            state.root_solved = false;
            if let Some(error) = pending_error {
                return Err(error);
            }
            return Ok(Some(TerminationReason::TimeLimit));
        }
    }
    if let Some(error) = pending_error {
        return Err(error);
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
        depth: 0,
        parent_id: 0,
        branch_var: None,
        branch_up: false,
        branch_frac: 1.0,
    };
    let int_tol = state.options.int_tol;
    if branching::is_integral(&state.solver, domains, int_tol) {
        match process_integral_candidate(state, domains, int_tol)? {
            IntegralCandidate::Branch(var) => branch(state, &root, var),
            IntegralCandidate::Closed => {
                return Ok(Some(TerminationReason::ProvenOptimal));
            }
        }
    } else {
        match branching::choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts) {
            Some(var) => branch(state, &root, var),
            // Fractional integer variables fixed to a non-integer value cannot
            // produce an integer point or a useful branch.
            None => return Err(Error::Infeasible),
        }
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
    if !warm && state.solver.load_basis(&node.basis).is_err() {
        debug!("basis load failed; falling back to slack basis");
        let slack = state.solver.slack_basis();
        state
            .solver
            .load_basis(&slack)
            .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
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
    if let Some(incumbent) = &state.incumbent {
        if objective >= cutoff(incumbent.objective, state.options.tolerances.prune_epsilon) {
            state.last_solved_id = None;
            state.diving = false;
            return Ok(NodeVisit::Solved);
        }
    }

    let int_tol = state.options.int_tol;
    if branching::is_integral(&state.solver, domains, int_tol) {
        match process_integral_candidate(state, domains, int_tol)? {
            IntegralCandidate::Branch(var) => branch(state, &node, var),
            IntegralCandidate::Closed => {
                state.last_solved_id = None;
                state.diving = false;
            }
        }
        return Ok(NodeVisit::Solved);
    }

    match branching::choose_branch_var(&state.solver, domains, int_tol, &state.pseudocosts) {
        Some(var) => branch(state, &node, var),
        None => {
            state.last_solved_id = None;
            state.diving = false;
        }
    }
    Ok(NodeVisit::Solved)
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
        let node = match pop_node(state) {
            Some(n) => n,
            None => break,
        };

        // Prune with the stored parent bound before any LP work.
        if let Some(inc) = &state.incumbent {
            if node.lp_bound >= cutoff(inc.objective, state.options.tolerances.prune_epsilon) {
                state.diving = false;
                continue;
            }
        }

        // A node budget limits LP solves, not free bookkeeping. Pop and apply
        // the stored-bound prune first so hitting the exact solve count can
        // still finish a proof whose remaining nodes are already dominated by
        // the incumbent.
        if let Some(nl) = state.options.node_limit {
            if nodes_this_run >= nl {
                state.open.push(node);
                state.diving = false;
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
                state.open.push(node);
                state.last_solved_id = None;
                state.diving = false;
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
