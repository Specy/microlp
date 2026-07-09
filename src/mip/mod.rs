//! Branch & bound driver for mixed-integer problems.
//!
//! Owns exactly one [`Solver`] per search. Branching changes variable bounds in
//! place (never adds constraint rows), so the LP never grows during the search.

pub(crate) mod branching;
pub(crate) mod node;

use crate::solver::{check_deadline, Deadline, Solver};
use crate::{Error, OptimizationDirection, Problem, StopReason, VarDomain, Variable};
use core::time::Duration;
use node::{effective_bounds, Node};
use web_time::Instant;

/// The outcome class of a finished or interrupted solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Proven optimal (within the configured `mip_gap`, which defaults to exact).
    Optimal,
    /// A limit was hit; a feasible solution is available but optimality is unproven.
    Feasible,
    /// A limit was hit before any usable solution was found. Value accessors
    /// ([`crate::Solution::objective`] etc.) panic on such solutions; call
    /// [`crate::Solution::resume`] to continue the search.
    Interrupted,
}

/// Options controlling a solve. Construct with [`SolveOptions::default`] and
/// mutate the fields you need.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SolveOptions {
    /// Wall-clock budget for this call (`None` = unlimited). On expiry the search
    /// stops cleanly and can be resumed.
    pub time_limit: Option<Duration>,
    /// Maximum number of branch & bound nodes to solve in this call
    /// (`None` = unlimited). Deterministic alternative to `time_limit`; the
    /// budget applies per call, so each `resume` gets a fresh budget.
    /// The root relaxation does not count as a node.
    pub node_limit: Option<u64>,
    /// Relative MIP gap at which the search stops and reports [`Status::Optimal`].
    /// Default `0.0` (prove exact optimality).
    pub mip_gap: f64,
    /// Integrality tolerance: a value within this distance of an integer counts
    /// as integral. Default `1e-6`.
    pub int_tol: f64,
    /// Optional (partial) starting assignment used to seed the incumbent.
    /// Advisory: an infeasible or incomplete hint is ignored. Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            time_limit: None,
            node_limit: None,
            mip_gap: 0.0,
            int_tol: 1e-6,
            warm_start: None,
        }
    }
}

/// Statistics of a solve, available via [`crate::Solution::stats`].
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct Stats {
    /// Branch & bound nodes whose LP was solved (0 for pure-LP problems).
    pub nodes_solved: u64,
    /// Total simplex pivots across the whole solve (including the root LP).
    pub lp_iterations: u64,
    /// Wall-clock time spent inside the solver, accumulated across resumes.
    pub elapsed: Duration,
    /// Best proven bound on the objective, in user space (filled in phase 2).
    pub best_bound: Option<f64>,
    /// Relative gap between incumbent and best bound (filled in phase 2).
    pub gap: Option<f64>,
}

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
    // Read by best-bound / gap reporting starting in Tasks 7-9.
    #[allow(dead_code)]
    pub direction: OptimizationDirection,
}

impl std::fmt::Debug for MipState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MipState")
            .field("open_nodes", &self.open.len())
            .field("has_incumbent", &self.incumbent.is_some())
            .field("stats", &self.stats)
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MipOutcome {
    Optimal,
    Interrupted,
}

#[derive(Debug)]
pub(crate) struct MipRun {
    pub outcome: MipOutcome,
    pub state: MipState,
}

pub(crate) fn status_of(outcome: MipOutcome, state: &MipState) -> Status {
    match outcome {
        MipOutcome::Optimal => Status::Optimal,
        MipOutcome::Interrupted => {
            if state.incumbent.is_some() {
                Status::Feasible
            } else {
                Status::Interrupted
            }
        }
    }
}

/// Build the search state for `problem` and run it under `options`.
pub(crate) fn run(problem: &Problem, options: SolveOptions) -> Result<MipRun, Error> {
    let deadline = options.time_limit.map(|d| Instant::now() + d);
    let solver = Solver::try_new(
        &problem.obj_coeffs,
        &problem.var_mins,
        &problem.var_maxs,
        &problem.constraints,
        &problem.var_domains,
        deadline,
    )?;
    let root_bounds = problem
        .var_mins
        .iter()
        .zip(&problem.var_maxs)
        .map(|(&lo, &hi)| (lo, hi))
        .collect();
    let mut state = MipState {
        solver,
        root_bounds,
        applied: Vec::new(),
        open: Vec::new(),
        incumbent: None,
        node_seq: 0,
        last_solved_id: None,
        root_solved: false,
        stats: Stats::default(),
        options,
        deadline,
        direction: problem.direction,
    };
    let outcome = resume_run_with_deadline(&mut state)?;
    Ok(MipRun { outcome, state })
}

/// Continue a paused search with a fresh time budget.
pub(crate) fn resume_run(
    state: &mut MipState,
    time_limit: Option<Duration>,
) -> Result<MipOutcome, Error> {
    state.deadline = time_limit.map(|d| Instant::now() + d);
    resume_run_with_deadline(state)
}

fn resume_run_with_deadline(state: &mut MipState) -> Result<MipOutcome, Error> {
    let started = Instant::now();
    let res = search_loop(state);
    state.stats.elapsed += started.elapsed();
    state.stats.lp_iterations = state.solver.lp_iterations;
    res
}

/// Prune threshold: a node whose lower bound is ≥ this cannot improve the incumbent.
fn cutoff(incumbent_obj: f64) -> f64 {
    incumbent_obj - f64::max(1e-9, 1e-9 * incumbent_obj.abs())
}

/// Try to adopt the solver's current solution (integral within `int_tol`) as
/// the incumbent, using integer-ROUNDED values. When `require_feasible`, the
/// rounded vector is validated against root bounds and the original rows first
/// — guarding against the big-M trap where sub-tolerance fractionality times a
/// huge coefficient is a real violation. Returns false iff validation failed
/// (caller must branch instead). Rounded values are stored so the incumbent is
/// exactly what the user reads.
fn try_adopt_incumbent(state: &mut MipState, require_feasible: bool) -> bool {
    let solver = &state.solver;
    let n = solver.num_vars;
    let domains = &solver.orig_var_domains;
    let mut values: Vec<f64> = (0..n).map(|v| *solver.get_value(v)).collect();
    for (val, dom) in values.iter_mut().zip(domains.iter()) {
        if matches!(dom, VarDomain::Integer | VarDomain::Boolean) {
            *val = val.round();
        }
    }
    if require_feasible {
        let in_bounds = (0..n).all(|v| {
            let (lo, hi) = state.root_bounds[v];
            values[v] >= lo - 1e-7 && values[v] <= hi + 1e-7
        });
        if !in_bounds || !solver.check_constraints(&values, 1e-7) {
            debug!("integral-within-tol solution rejected: rounded values infeasible");
            return false;
        }
    }
    let objective = solver.objective_of(&values);
    let better = match &state.incumbent {
        Some(inc) => objective < inc.objective,
        None => true,
    };
    if better {
        debug!("new incumbent, internal obj: {:.6}", objective);
        state.incumbent = Some(Incumbent { values, objective });
    }
    true
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
    let val = *state.solver.get_value(var);
    let (lo, hi) = state.solver.get_var_bounds(var);
    let floor = val.floor();

    state.node_seq += 1;
    let id = state.node_seq;
    state.last_solved_id = Some(id);
    let basis = state.solver.snapshot_basis();

    let mut down_changes = parent.bound_changes.clone();
    down_changes.push((var, lo, floor));
    let mut up_changes = parent.bound_changes.clone();
    up_changes.push((var, floor + 1.0, hi));

    // Phase-1 order parity with the old solver: the up (ceil) child is dived first.
    state.open.push(Node {
        bound_changes: down_changes,
        basis: basis.clone(),
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
    });
    state.open.push(Node {
        bound_changes: up_changes,
        basis,
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
    });
}

/// Outcome of solving one branch & bound node's LP relaxation.
enum NodeLp {
    /// The solver now holds this node's optimal basis and objective.
    Solved,
    /// The node's LP is infeasible under its current bounds.
    Infeasible,
    /// A limit interrupted the solve (possibly during the slack retry); the
    /// solver's half-pivoted state must not be consulted.
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

fn search_loop(state: &mut MipState) -> Result<MipOutcome, Error> {
    let domains = state.solver.orig_var_domains.clone();
    let int_tol = state.options.int_tol;
    state.solver.deadline = state.deadline;

    // Root relaxation (also the resume path if the root was interrupted mid-LP:
    // initial_solve continues from the solver's feasibility flags).
    if !state.root_solved {
        match state.solver.initial_solve()? {
            StopReason::Limit => return Ok(MipOutcome::Interrupted),
            StopReason::Finished => {}
        }
        state.root_solved = true;
        let root = Node {
            bound_changes: Vec::new(),
            basis: state.solver.snapshot_basis(),
            lp_bound: state.solver.cur_obj_val,
            depth: 0,
            parent_id: 0,
        };
        if branching::is_integral(&state.solver, &domains, int_tol) {
            if try_adopt_incumbent(state, true) {
                return Ok(MipOutcome::Optimal);
            }
            match branching::choose_branch_var(&state.solver, &domains, 0.0) {
                Some(var) => branch(state, &root, var),
                None => {
                    // All int vars exactly integral yet the check failed: float noise
                    // in the re-computed rows, not in the solution — accept it.
                    try_adopt_incumbent(state, false);
                    return Ok(MipOutcome::Optimal);
                }
            }
        } else {
            let var = branching::choose_branch_var(&state.solver, &domains, int_tol)
                .expect("non-integral relaxation must have a fractional int var");
            branch(state, &root, var);
        }
    }

    let mut nodes_this_run: u64 = 0;

    loop {
        // Limits are checked BETWEEN nodes; nothing half-solved is ever consulted.
        if check_deadline(&state.deadline) == StopReason::Limit {
            return Ok(MipOutcome::Interrupted);
        }
        if let Some(nl) = state.options.node_limit {
            if nodes_this_run >= nl {
                return Ok(MipOutcome::Interrupted);
            }
        }

        let node = match state.open.pop() {
            Some(n) => n,
            None => break,
        };

        // Prune with the stored parent bound before any LP work.
        if let Some(inc) = &state.incumbent {
            if node.lp_bound >= cutoff(inc.objective) {
                continue;
            }
        }

        // Load the node into the solver: bounds first (values derive from them),
        // then the basis if the solver isn't already at this node's parent optimum.
        if !apply_node_bounds(state, &node) {
            continue; // crossing bounds — pruned without touching the solver
        }
        let warm = state.last_solved_id == Some(node.parent_id);
        if !warm && state.solver.load_basis(&node.basis).is_err() {
            // Robustness valve: a numerically singular basis falls back to the
            // all-slack basis and the node is solved from scratch.
            debug!("basis load failed; falling back to slack basis");
            let slack = state.solver.slack_basis();
            state
                .solver
                .load_basis(&slack)
                .map_err(|e| Error::InternalError(format!("slack basis load failed: {}", e)))?;
        }

        // Solve the node's LP. A re-solve that degrades into a singular LU is
        // retried once from scratch on the all-slack basis inside solve_node_lp.
        match solve_node_lp(state)? {
            NodeLp::Solved => {}
            NodeLp::Infeasible => {
                state.stats.nodes_solved += 1;
                nodes_this_run += 1;
                state.last_solved_id = None;
                continue;
            }
            NodeLp::Limit => {
                // Requeue UNSOLVED: the node's plain data is intact, the solver's
                // half-pivoted state will be discarded by the next basis load.
                state.open.push(node);
                state.last_solved_id = None;
                return Ok(MipOutcome::Interrupted);
            }
        }
        state.stats.nodes_solved += 1;
        nodes_this_run += 1;

        let z = state.solver.cur_obj_val;
        if let Some(inc) = &state.incumbent {
            if z >= cutoff(inc.objective) {
                state.last_solved_id = None;
                continue;
            }
        }

        if branching::is_integral(&state.solver, &domains, int_tol) {
            if try_adopt_incumbent(state, true) {
                state.last_solved_id = None;
            } else if let Some(var) = branching::choose_branch_var(&state.solver, &domains, 0.0) {
                // Below-tolerance fractionality with real infeasibility on rounding:
                // branch anyway — the children fix the var exactly (floor/floor+1).
                branch(state, &node, var);
            } else {
                try_adopt_incumbent(state, false);
                state.last_solved_id = None;
            }
            continue;
        }

        let var = branching::choose_branch_var(&state.solver, &domains, int_tol)
            .expect("non-integral node must have a fractional int var");
        branch(state, &node, var);
    }

    if state.incumbent.is_some() {
        Ok(MipOutcome::Optimal)
    } else {
        Err(Error::Infeasible)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComparisonOp, OptimizationDirection, Problem};

    fn int_2var_problem() -> Problem {
        // minimize 3a + 4b s.t. a + 2b >= 5, 3a + b >= 4; a,b integer in [0,10].
        // LP relaxation: a=0.6, b=2.2, obj 10.6. Integer optimum: a=1, b=2, obj 11.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let a = p.add_integer_var(3.0, (0, 10));
        let b = p.add_integer_var(4.0, (0, 10));
        p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
        p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
        p
    }

    fn binary_knapsack() -> Problem {
        // maximize 8x + 11y + 6z + 4w s.t. 5x + 7y + 4z + 3w <= 14, binaries.
        // Optimum: y + z + w = 21 (weight 14).
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_binary_var(8.0);
        let y = p.add_binary_var(11.0);
        let z = p.add_binary_var(6.0);
        let w = p.add_binary_var(4.0);
        p.add_constraint(
            &[(x, 5.0), (y, 7.0), (z, 4.0), (w, 3.0)],
            ComparisonOp::Le,
            14.0,
        );
        p
    }

    fn incumbent_obj(state: &MipState) -> f64 {
        state.incumbent.as_ref().unwrap().objective
    }

    #[test]
    fn driver_finds_integer_optimum() {
        let run = run(&int_2var_problem(), SolveOptions::default()).unwrap();
        assert_eq!(run.outcome, MipOutcome::Optimal);
        // Internal space == user space for Minimize.
        assert!((incumbent_obj(&run.state) - 11.0).abs() < 1e-6);
        let inc = run.state.incumbent.as_ref().unwrap();
        assert!((inc.values[0] - 1.0).abs() < 1e-6);
        assert!((inc.values[1] - 2.0).abs() < 1e-6);
        assert!(run.state.stats.nodes_solved > 0);
    }

    #[test]
    fn driver_binary_knapsack_maximize() {
        let run = run(&binary_knapsack(), SolveOptions::default()).unwrap();
        assert_eq!(run.outcome, MipOutcome::Optimal);
        // Maximize is negated internally: internal optimum is -21.
        assert!((incumbent_obj(&run.state) + 21.0).abs() < 1e-6);
    }

    #[test]
    fn driver_no_integer_point_is_infeasible() {
        // 2x == 1 with x integer in [0,10]: LP-feasible (x=0.5), integer-infeasible.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
        assert_eq!(
            run(&p, SolveOptions::default()).unwrap_err(),
            crate::Error::Infeasible
        );
    }

    #[test]
    fn driver_node_limit_interrupts_and_resumes_to_same_optimum() {
        let mut options = SolveOptions::default();
        options.node_limit = Some(1);
        let mut r = run(&int_2var_problem(), options).unwrap();
        let mut guard = 0;
        while r.outcome == MipOutcome::Interrupted {
            guard += 1;
            assert!(guard < 10_000, "resume loop did not terminate");
            r.outcome = resume_run(&mut r.state, None).unwrap();
        }
        assert!(guard >= 1, "node_limit=1 should interrupt at least once");
        assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn driver_zero_time_limit_interrupts_cleanly_then_resumes() {
        let mut options = SolveOptions::default();
        options.time_limit = Some(Duration::ZERO);
        let mut r = run(&binary_knapsack(), options).unwrap();
        assert_eq!(r.outcome, MipOutcome::Interrupted);
        assert!(r.state.incumbent.is_none());
        assert_eq!(status_of(r.outcome, &r.state), Status::Interrupted);
        let outcome = resume_run(&mut r.state, None).unwrap();
        assert_eq!(outcome, MipOutcome::Optimal);
        assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
    }
}
