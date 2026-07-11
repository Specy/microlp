//! Branch & bound driver for mixed-integer problems.
//!
//! Owns exactly one [`Solver`] per search. Branching changes variable bounds in
//! place (never adds constraint rows), so the LP never grows during the search.

pub(crate) mod branching;
pub(crate) mod node;
pub(crate) mod params;

use crate::solver::{check_deadline, Deadline, Solver};
use crate::{ComparisonOp, Error, OptimizationDirection, Problem, StopReason, VarDomain, Variable};
use core::time::Duration;
use node::{effective_bounds, Node};
use std::collections::BTreeMap;
use web_time::Instant;

/// The outcome class of a finished or interrupted solve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Proven optimal (within the configured `mip_gap`, which defaults to exact).
    Optimal,
    /// A limit was hit; a feasible solution is available but optimality is unproven.
    Feasible,
    /// A limit was hit before any usable solution was found. Value accessors
    /// ([`crate::Solution::objective`] etc.) expose the search's current
    /// working point on such solutions — possibly fractional and infeasible,
    /// useful for inspection only. Checking the status before treating values
    /// as the answer is the caller's responsibility; call
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
    /// as integral. Default `1e-6`. Loosening it does not loosen final
    /// feasibility: a rounded candidate must still pass the absolute
    /// `tolerances.feasibility` per-row/bound check (default `1e-7`) before it
    /// is accepted, so a very loose `int_tol` mainly causes extra exact-fixing
    /// branching rather than admitting an infeasible point.
    pub int_tol: f64,
    /// Optional (partial) starting assignment used to seed the incumbent.
    /// Advisory: an infeasible or incomplete hint is ignored. Default `None`.
    pub warm_start: Option<Vec<(Variable, f64)>>,
    /// Enable presolve: reductions applied to the problem before the search
    /// starts (bound tightening, redundant-row elimination, variable fixing;
    /// for integer problems also coefficient tightening and dual fixing).
    /// The problem you observe through [`crate::Solution`] is unchanged —
    /// reductions never remove variables and preserve at least one optimum.
    /// Disable only to compare raw solver behavior or to rule presolve out
    /// while investigating a suspected numerical issue. Default `true`.
    pub presolve: bool,
    /// Expert-level numeric tolerances (see [`Tolerances`]). Most callers
    /// should leave this at [`Tolerances::default`]; override an individual
    /// field only once you understand the correctness/permissiveness
    /// trade-off documented on it.
    pub tolerances: Tolerances,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            time_limit: None,
            node_limit: None,
            mip_gap: 0.0,
            int_tol: 1e-6,
            warm_start: None,
            presolve: true,
            tolerances: Tolerances::default(),
        }
    }
}

/// Expert-level numeric tolerances for a solve (see [`SolveOptions::tolerances`]).
///
/// These are distinct from the rest of [`SolveOptions`] in kind: each field
/// here trades correctness risk against permissiveness in a way that
/// requires understanding a specific piece of solver behavior to tune
/// safely, so they are grouped separately rather than left as top-level
/// `SolveOptions` fields. Most callers never need to touch this and should
/// start from [`Tolerances::default`].
///
/// Purely internal numeric constants that carry no user-facing meaning (e.g.
/// denominator guards, branching heuristics) live in a separate, undocumented-
/// to-callers internal module instead of here — this struct is reserved for
/// numbers whose value is part of the solver's observable contract.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct Tolerances {
    /// Absolute tolerance, in the same units as the problem's bounds and
    /// constraint right-hand sides, used to validate a rounded-to-integer
    /// candidate solution before it is accepted as the incumbent (the
    /// "rounded-incumbent guard"): applied to each variable's distance
    /// outside its bounds and to each row's distance outside its feasible
    /// range. Also used, identically, by the post-edit warm-start
    /// pre-filter that decides whether a previous incumbent survives a
    /// [`crate::Solution`] edit.
    ///
    /// This is deliberately an ABSOLUTE tolerance, never one scaled by a
    /// row's or bound's magnitude: a relative tolerance is blind to the
    /// "big-M" trap, where a violation that is tiny RELATIVE to a huge row
    /// coefficient (e.g. a slack of 5.0 against a coefficient of 1e9) is
    /// nonetheless decisive in absolute terms — exactly the case this guard
    /// exists to catch. See `Solver::check_constraints` for the full
    /// rationale.
    ///
    /// Default `1e-7`.
    pub feasibility: f64,
    /// Distance from the nearest integer within which an integer/boolean
    /// variable's value is still treated as exactly that integer. Used by
    /// the post-edit warm-start pre-filter's integrality check, mirroring
    /// [`crate::Solution::var_value`]'s own rounding check.
    ///
    /// Note: [`crate::Solution::var_value`]'s internal rounding sanity
    /// assert always uses [`Tolerances::default`]'s value for this field,
    /// never the value configured for the solve that produced the solution.
    /// That assert exists purely to catch a solver bug — an accepted
    /// incumbent must already be integral-clean by the time it reaches the
    /// user — not to reflect a caller's preference, so it intentionally does
    /// not follow a loosened setting here.
    ///
    /// Default `1e-5`.
    pub integrality_rounding: f64,
    /// Relative slack subtracted from the incumbent objective to form the
    /// branch & bound pruning cutoff: a node whose bound is not strictly
    /// better than `incumbent - max(prune_epsilon, prune_epsilon *
    /// |incumbent|)` is pruned. Guards against continuing to explore or
    /// retain nodes that could only ever match the incumbent to within
    /// float noise.
    ///
    /// Default `1e-9`.
    pub prune_epsilon: f64,
}

impl Default for Tolerances {
    fn default() -> Self {
        Self {
            feasibility: 1e-7,
            integrality_rounding: 1e-5,
            prune_epsilon: 1e-9,
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
    /// Best proven bound on the objective, in user space. `None` until an
    /// incumbent or an open node exists to derive one from.
    pub best_bound: Option<f64>,
    /// Relative gap between incumbent and best bound. `None` until both are
    /// known; `Some(0.0)` once optimality is proven.
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
    // Presolve the search problem. `state.base` below stays the user's
    // untouched problem, so post-solve edits keep composing against it; the
    // presolved bounds become the ROOT bounds (branching resets to them).
    // Dual fixing is optimum-preserving but not feasible-point-preserving,
    // so it is disabled when a warm-start hint must be honored.
    let pre = if options.presolve {
        Some(crate::presolve::presolve(
            &problem.obj_coeffs,
            &problem.var_mins,
            &problem.var_maxs,
            &problem.constraints,
            &problem.var_domains,
            crate::presolve::Mode::Mip,
            options.int_tol,
            options.warm_start.is_none(),
        )?)
    } else {
        None
    };
    let (var_mins, var_maxs, constraints) = match &pre {
        Some(p) => (&p.var_mins[..], &p.var_maxs[..], &p.constraints[..]),
        None => (
            &problem.var_mins[..],
            &problem.var_maxs[..],
            &problem.constraints[..],
        ),
    };
    let solver = Solver::try_new(
        &problem.obj_coeffs,
        var_mins,
        var_maxs,
        constraints,
        &problem.var_domains,
        deadline,
    )?;
    let root_bounds = var_mins
        .iter()
        .zip(var_maxs)
        .map(|(&lo, &hi)| (lo, hi))
        .collect();
    let pseudocosts = branching::PseudoCosts::new(&problem.obj_coeffs, problem.obj_coeffs.len());
    let mut state = MipState {
        solver,
        root_bounds,
        applied: Vec::new(),
        open: Vec::new(),
        diving: false,
        incumbent: None,
        node_seq: 0,
        last_solved_id: None,
        root_solved: false,
        stats: Stats::default(),
        options,
        deadline,
        direction: problem.direction,
        pseudocosts,
        base: problem.clone(),
        fixed: BTreeMap::new(),
    };
    let outcome = resume_run_with_deadline(&mut state)?;
    Ok(MipRun { outcome, state })
}

/// `base` with the fix_var overlay applied to the variable bounds.
pub(crate) fn effective_problem(base: &Problem, fixed: &BTreeMap<usize, f64>) -> Problem {
    let mut p = base.clone();
    for (&v, &val) in fixed {
        p.var_mins[v] = val;
        p.var_maxs[v] = val;
    }
    p
}

/// Cheap feasibility check of a value vector against base + fixes: bounds and
/// every constraint to the absolute `tolerances.feasibility`, integrality to
/// `tolerances.integrality_rounding`. Only ever used as a warm-start
/// pre-filter (see [`reedit_and_resolve`]) — an incumbent this accepts is not
/// trusted blindly, it re-enters the search as a hint and is re-validated by
/// the same absolute-tolerance guard [`try_adopt_incumbent`] applies to every
/// candidate, so this function unifying on the ABSOLUTE `feasibility` value
/// (rather than a row-magnitude-relative one) only tightens consistency
/// between the two guards; it cannot admit anything the real guard would not
/// also have to accept.
pub(crate) fn incumbent_feasible(
    base: &Problem,
    fixed: &BTreeMap<usize, f64>,
    values: &[f64],
    tolerances: &Tolerances,
) -> bool {
    for (v, &val) in values.iter().enumerate() {
        let (mut lo, mut hi) = (base.var_mins[v], base.var_maxs[v]);
        if let Some(&f) = fixed.get(&v) {
            lo = f;
            hi = f;
        }
        if val < lo - tolerances.feasibility || val > hi + tolerances.feasibility {
            return false;
        }
        if matches!(base.var_domains[v], VarDomain::Integer | VarDomain::Boolean)
            && (val - val.round()).abs() > tolerances.integrality_rounding
        {
            return false;
        }
    }
    for (coeffs, op, rhs) in &base.constraints {
        let lhs: f64 = coeffs.iter().map(|(i, c)| c * values[i]).sum();
        let tol = tolerances.feasibility;
        let ok = match op {
            ComparisonOp::Eq => (lhs - rhs).abs() <= tol,
            ComparisonOp::Le => lhs <= rhs + tol,
            ComparisonOp::Ge => lhs >= *rhs - tol,
        };
        if !ok {
            return false;
        }
    }
    true
}

/// After a user edit: drop the open tree, carry the incumbent as a warm-start
/// hint when it survives the edit, and re-run the search on base + fixes.
/// The fresh run gets the state's original options (incl. a fresh time budget).
pub(crate) fn reedit_and_resolve(state: Box<MipState>) -> Result<MipRun, Error> {
    let MipState {
        base,
        fixed,
        incumbent,
        mut options,
        ..
    } = *state;

    options.warm_start = incumbent
        .filter(|inc| incumbent_feasible(&base, &fixed, &inc.values, &options.tolerances))
        .map(|inc| {
            inc.values
                .iter()
                .enumerate()
                .map(|(v, &val)| (Variable(v), val))
                .collect()
        });

    let effective = effective_problem(&base, &fixed);
    let mut run = run(&effective, options)?;
    // `run` cloned `effective` as its base; restore the true base/fixed split so
    // later edits keep composing against the user's problem.
    run.state.base = base;
    run.state.fixed = fixed;
    Ok(run)
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
    fill_bound_stats(state);
    res
}

/// Best proven lower bound (internal space) on the optimum: the min over open-node
/// bounds and the incumbent. `None` while nothing is known (no nodes, no incumbent).
/// Only valid BETWEEN nodes (a popped node's subtree is otherwise unaccounted).
fn global_bound_internal(state: &MipState) -> Option<f64> {
    let open_min = state
        .open
        .iter()
        .map(|n| n.lp_bound)
        .fold(f64::INFINITY, f64::min);
    match (&state.incumbent, state.open.is_empty()) {
        (Some(inc), true) => Some(inc.objective), // proof complete
        (Some(inc), false) => Some(open_min.min(inc.objective)),
        (None, false) => Some(open_min),
        (None, true) => None,
    }
}

/// Relative gap between incumbent and bound, internal space (0.0 when they meet).
fn relative_gap(incumbent_obj: f64, bound: f64) -> f64 {
    (incumbent_obj - bound).max(0.0) / incumbent_obj.abs().max(params::GAP_DENOM_GUARD)
}

fn to_user_space(direction: OptimizationDirection, internal: f64) -> f64 {
    match direction {
        OptimizationDirection::Minimize => internal,
        OptimizationDirection::Maximize => -internal,
    }
}

fn fill_bound_stats(state: &mut MipState) {
    let bound = global_bound_internal(state);
    state.stats.best_bound = bound.map(|b| to_user_space(state.direction, b));
    state.stats.gap = match (&state.incumbent, bound) {
        (Some(inc), Some(b)) => Some(relative_gap(inc.objective, b)),
        _ => None,
    };
}

/// Prune threshold: a node whose lower bound is ≥ this cannot improve the incumbent.
fn cutoff(incumbent_obj: f64, prune_epsilon: f64) -> f64 {
    incumbent_obj - f64::max(prune_epsilon, prune_epsilon * incumbent_obj.abs())
}

/// Try to adopt the solver's current solution (integral within `int_tol`) as
/// the incumbent, using integer-ROUNDED values. When `require_feasible`, the
/// rounded vector is validated against root bounds and the original rows first
/// — guarding against the big-M trap where sub-tolerance fractionality times a
/// huge coefficient is a real violation. Returns false iff validation failed
/// (caller must branch instead). Rounded values are stored so the incumbent is
/// exactly what the user reads.
fn try_adopt_incumbent(state: &mut MipState, require_feasible: bool) -> bool {
    let feasibility_tol = state.options.tolerances.feasibility;
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
            values[v] >= lo - feasibility_tol && values[v] <= hi + feasibility_tol
        });
        if !in_bounds || !solver.check_constraints(&values, feasibility_tol) {
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
    let f_down = val - floor;

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
        branch_var: var,
        branch_up: false,
        branch_frac: f_down,
    };
    let up_node = Node {
        bound_changes: up_changes,
        basis,
        lp_bound: z,
        depth: parent.depth + 1,
        parent_id: id,
        branch_var: var,
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
/// node). Returns `Ok(Some(MipOutcome::Interrupted))` only when the restore of the
/// root basis fails AND the deadline strikes mid-restore: rather than let the
/// caller read a half-solved `cur_obj_val` as the root bound, it un-sets
/// `root_solved` so a resume re-enters `initial_solve` and continues honestly from
/// the solver's feasibility flags.
fn try_warm_start(
    state: &mut MipState,
    hints: &[(crate::Variable, f64)],
) -> Result<Option<MipOutcome>, Error> {
    let domains = state.solver.orig_var_domains.clone();
    let root_basis = state.solver.snapshot_basis();
    let mut applied: Vec<usize> = Vec::new();
    let mut ok = true;

    for &(var, val) in hints {
        let v = var.idx();
        if v >= state.solver.num_vars {
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
                    if !try_adopt_incumbent(state, true) {
                        debug!("warm-start hint rejected by feasibility guard; ignored");
                    }
                } else {
                    debug!("warm-start hint LP-completed fractionally; ignored");
                }
            }
            Ok(StopReason::Limit) | Err(Error::Infeasible) => {
                debug!("warm-start hint infeasible or out of time; ignored");
            }
            Err(e) => return Err(e),
        }
    } else {
        debug!("warm-start hint invalid (unknown var or out of bounds); ignored");
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
            // The root re-solve from slack ran out of budget MID-SOLVE. The
            // solver's `cur_obj_val` is now a half-solved value; if we returned
            // `Ok(None)` the caller would seed the root node's `lp_bound` from it
            // — an invalid lower bound that can prune legitimate subtrees (the one
            // breach of the "never consult half-solved state" invariant). Instead
            // un-set `root_solved` and signal an immediate Interrupted: on resume
            // the `!root_solved` block re-enters `initial_solve`, which continues
            // honestly from the solver's feasibility flags. Untestable
            // deterministically — it needs an LU failure on the root basis load
            // AND a deadline strike on the very next re-solve, on the same node —
            // but the invariant must hold structurally.
            state.root_solved = false;
            return Ok(Some(MipOutcome::Interrupted));
        }
    }
    Ok(None)
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
        if let Some(hints) = state.options.warm_start.take() {
            // `Some(outcome)` means the restore failed and the deadline struck
            // mid-re-solve: bail out immediately (root_solved was un-set inside)
            // rather than build the root node from a half-solved bound.
            if let Some(outcome) = try_warm_start(state, &hints)? {
                return Ok(outcome);
            }
        }
        let root = Node {
            bound_changes: Vec::new(),
            basis: state.solver.snapshot_basis(),
            lp_bound: state.solver.cur_obj_val,
            depth: 0,
            parent_id: 0,
            // Placeholder: the root is never popped from `open` (it's consumed here,
            // not pushed), so this is never fed into `pseudocosts.record`.
            branch_var: 0,
            branch_up: false,
            branch_frac: 1.0,
        };
        if branching::is_integral(&state.solver, &domains, int_tol) {
            if try_adopt_incumbent(state, true) {
                return Ok(MipOutcome::Optimal);
            }
            match branching::choose_branch_var(&state.solver, &domains, 0.0, &state.pseudocosts) {
                Some(var) => branch(state, &root, var),
                None => {
                    // All int vars exactly integral yet the check failed: float noise
                    // in the re-computed rows, not in the solution — accept it.
                    try_adopt_incumbent(state, false);
                    return Ok(MipOutcome::Optimal);
                }
            }
        } else {
            let var =
                branching::choose_branch_var(&state.solver, &domains, int_tol, &state.pseudocosts)
                    .expect("non-integral relaxation must have a fractional int var");
            branch(state, &root, var);
        }
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

        // Gap-based stop: an incumbent within `mip_gap` of the proven bound counts
        // as optimal. Checked first so it takes priority over limit interruptions.
        if state.options.mip_gap > 0.0 {
            if let (Some(inc), Some(bound)) = (&state.incumbent, global_bound_internal(state)) {
                if relative_gap(inc.objective, bound) <= state.options.mip_gap {
                    return Ok(MipOutcome::Optimal);
                }
            }
        }

        // Limits are checked BETWEEN nodes; nothing half-solved is ever consulted.
        if check_deadline(&state.deadline) == StopReason::Limit {
            return Ok(MipOutcome::Interrupted);
        }
        if let Some(nl) = state.options.node_limit {
            if nodes_this_run >= nl {
                return Ok(MipOutcome::Interrupted);
            }
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

        // Load the node into the solver: bounds first (values derive from them),
        // then the basis if the solver isn't already at this node's parent optimum.
        if !apply_node_bounds(state, &node) {
            state.diving = false;
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
                state.diving = false;
                continue;
            }
            NodeLp::Limit => {
                // Requeue UNSOLVED: the node's plain data is intact, the solver's
                // half-pivoted state will be discarded by the next basis load. Stop
                // diving too: after a resume the requeued node must be reachable via
                // the best-bound jump, not privileged by a stale LIFO position — and
                // the solver holds no useful warm state for it either way.
                state.open.push(node);
                state.last_solved_id = None;
                state.diving = false;
                return Ok(MipOutcome::Interrupted);
            }
        }
        state.stats.nodes_solved += 1;
        nodes_this_run += 1;

        let z = state.solver.cur_obj_val;
        // Feed this node's actual degradation (vs. its parent's bound, the estimate
        // used at branch time) back into the pseudocost that predicted it.
        state.pseudocosts.record(
            node.branch_var,
            node.branch_up,
            (z - node.lp_bound).max(0.0) / node.branch_frac.max(params::BRANCH_FRAC_GUARD),
        );
        if let Some(inc) = &state.incumbent {
            if z >= cutoff(inc.objective, state.options.tolerances.prune_epsilon) {
                state.last_solved_id = None;
                state.diving = false;
                continue;
            }
        }

        if branching::is_integral(&state.solver, &domains, int_tol) {
            if try_adopt_incumbent(state, true) {
                state.last_solved_id = None;
                state.diving = false;
            } else if let Some(var) =
                branching::choose_branch_var(&state.solver, &domains, 0.0, &state.pseudocosts)
            {
                // Below-tolerance fractionality with real infeasibility on rounding:
                // branch anyway — the children fix the var exactly (floor/floor+1);
                // branch() sets `diving = true`, so the dive continues into them.
                branch(state, &node, var);
            } else {
                try_adopt_incumbent(state, false);
                state.last_solved_id = None;
                state.diving = false;
            }
            continue;
        }

        let var =
            branching::choose_branch_var(&state.solver, &domains, int_tol, &state.pseudocosts)
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
    fn driver_exact_node_exhaustion_reports_infeasible_not_interrupted() {
        // Same infeasible fixture (2x == 1, x int in [0,10]): the root LP is
        // fractional (x=0.5) and branches into x<=0 and x>=1, both LP-infeasible.
        // The tree is therefore exactly two nodes; empirically node_limit=1 yields
        // Interrupted and node_limit=2 exhausts it. With the limit set to that
        // exact exhaustion count, the loop's empty-`open` check (which now runs
        // BEFORE the node-limit check) must report the completed proof as
        // Infeasible rather than a spurious Interrupted.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
        let mut options = SolveOptions::default();
        options.node_limit = Some(2);
        assert_eq!(run(&p, options).unwrap_err(), crate::Error::Infeasible);
    }

    #[test]
    fn driver_node_limit_equal_to_exhaustion_count_reports_optimal() {
        // int_2var_problem is proven optimal in exactly 2 B&B nodes (deterministic:
        // the unlimited solve reports nodes_solved == 2; node_limit=1 -> Interrupted,
        // node_limit=2 -> Optimal). Setting the node limit to that exact count must
        // report Optimal, not Interrupted: when the tree empties on the same
        // iteration the limit would fire, the empty-`open` check at the loop top
        // wins and the finished proof is reported honestly.
        assert_eq!(
            run(&int_2var_problem(), SolveOptions::default())
                .unwrap()
                .state
                .stats
                .nodes_solved,
            2,
            "node count must stay deterministic; update the limit below if it changes"
        );
        let mut options = SolveOptions::default();
        options.node_limit = Some(2);
        let r = run(&int_2var_problem(), options).unwrap();
        assert_eq!(r.outcome, MipOutcome::Optimal);
        assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
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

    #[test]
    fn optimal_solve_reports_zero_gap_and_matching_bound() {
        let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
        assert_eq!(r.outcome, MipOutcome::Optimal);
        assert_eq!(r.state.stats.gap, Some(0.0));
        // User space == internal for Minimize.
        assert!((r.state.stats.best_bound.unwrap() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn maximize_bound_is_in_user_space() {
        let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
        assert_eq!(r.outcome, MipOutcome::Optimal);
        // Internally -21; user-facing bound must be +21.
        assert!((r.state.stats.best_bound.unwrap() - 21.0).abs() < 1e-6);
    }

    #[test]
    fn mip_gap_stops_early_with_consistent_bound() {
        let mut options = SolveOptions::default();
        options.mip_gap = 0.5;
        let r = run(&binary_knapsack(), options).unwrap();
        assert_eq!(r.outcome, MipOutcome::Optimal); // optimal within the configured gap
        let inc = -incumbent_obj(&r.state); // user space (Maximize)
        let bound = r.state.stats.best_bound.unwrap();
        // Incumbent within 50% of the proven bound, and never better than it.
        assert!(inc <= bound + 1e-9);
        assert!((bound - inc) / bound.abs().max(1e-10) <= 0.5 + 1e-9);
    }

    #[test]
    fn feasible_interrupt_reports_gap() {
        let mut options = SolveOptions::default();
        options.node_limit = Some(2);
        let mut r = run(&binary_knapsack(), options).unwrap();
        // Resume with node budget until an incumbent exists but the search isn't done.
        let mut guard = 0;
        while r.outcome == MipOutcome::Interrupted && r.state.incumbent.is_none() {
            guard += 1;
            assert!(guard < 10_000);
            r.outcome = resume_run(&mut r.state, None).unwrap();
        }
        if r.outcome == MipOutcome::Interrupted {
            // Feasible-but-unproven: a gap must be reported.
            assert!(r.state.stats.gap.unwrap() >= 0.0);
            assert!(r.state.stats.best_bound.is_some());
        }
    }

    #[test]
    fn plunge_and_jump_selection_preserves_optima() {
        // Same optima as plain DFS on all driver test problems, plus interrupt/resume.
        let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
        assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);

        let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
        assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);

        let mut options = SolveOptions::default();
        options.node_limit = Some(1);
        let mut r = run(&binary_knapsack(), options).unwrap();
        let mut guard = 0;
        while r.outcome == MipOutcome::Interrupted {
            guard += 1;
            assert!(guard < 10_000);
            r.outcome = resume_run(&mut r.state, None).unwrap();
        }
        assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
        // After a best-bound jump the pop is NOT the last-pushed node at least once
        // on this instance; correctness above is the real assertion.
    }

    #[test]
    fn tolerances_default_matches_documented_values() {
        let t = Tolerances::default();
        assert_eq!(
            t.feasibility, 1e-7,
            "see Tolerances::feasibility's doc default"
        );
        assert_eq!(
            t.integrality_rounding, 1e-5,
            "see Tolerances::integrality_rounding's doc default"
        );
        assert_eq!(
            t.prune_epsilon, 1e-9,
            "see Tolerances::prune_epsilon's doc default"
        );
    }

    #[test]
    fn try_adopt_incumbent_respects_custom_feasibility_tolerance() {
        // Derived from the big-M fixture `tests_general::solve_big_m` (same
        // m = 1e9 shape: `x - m*b == 10`, minimize x). Pin b to a value that
        // is integral-within-`int_tol` (5e-7, well inside the default 1e-6)
        // but not exactly 0; the rounded-incumbent guard then re-checks the
        // ROUNDED point (b -> 0) against the ORIGINAL row, which is off by
        // exactly m * 5e-7 = 500 — precisely the "big-M trap"
        // `tolerances.feasibility` exists to catch, in absolute terms.
        let m = 1.0e9;
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, f64::INFINITY));
        let b = p.add_binary_var(0.0);
        p.add_constraint(&[(x, 1.0), (b, -m)], ComparisonOp::Eq, 10.0);

        let mut solved = run(&p, SolveOptions::default()).unwrap();
        let state = &mut solved.state;

        // Force the relaxation to a specific near-zero fractional b: pin its
        // bounds to [5e-7, 5e-7] and re-solve. The equality row then forces x
        // to exactly 10 + m*5e-7 = 510, deterministically — no dependence on
        // which vertex the simplex would otherwise have picked.
        state.solver.set_var_bounds(b.idx(), 5e-7, 5e-7).unwrap();
        assert_eq!(
            state.solver.reoptimize().unwrap(),
            crate::StopReason::Finished
        );
        assert!((*state.solver.get_value(x.idx()) - 510.0).abs() < 1e-6);

        // Default tolerance (1e-7): the rounded point (x=510, b=0) misses the
        // original `x - m*b == 10` row by 500 — must be rejected.
        state.options.tolerances.feasibility = Tolerances::default().feasibility;
        assert!(
            !try_adopt_incumbent(state, true),
            "a 500-unit rounding-induced violation must be rejected at the default feasibility tolerance"
        );

        // Absurdly loosened tolerance: the same 500-unit violation is now
        // within bounds — the guard must accept it.
        state.options.tolerances.feasibility = 1e6;
        assert!(
            try_adopt_incumbent(state, true),
            "the same violation must be accepted once tolerances.feasibility is loosened past it"
        );
    }

    #[test]
    fn incumbent_feasible_row_tolerance_is_absolute_not_relative_to_rhs() {
        // Post-unification behavior check: `incumbent_feasible`'s row check
        // used to be `1e-7 * rhs.abs().max(1.0)` (relative), which on a large
        // rhs like 1000.0 would forgive violations up to 1e-4. It is now the
        // same ABSOLUTE `tolerances.feasibility` (default 1e-7) the
        // rounded-incumbent guard uses, regardless of the row's magnitude.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, f64::INFINITY));
        p.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1000.0);
        let fixed = std::collections::BTreeMap::new();
        let tolerances = Tolerances::default();

        // Within the absolute tolerance (5e-8 < 1e-7): accepted.
        assert!(incumbent_feasible(
            &p,
            &fixed,
            &[1000.0 + 5e-8],
            &tolerances
        ));

        // Outside the absolute tolerance (5e-5 > 1e-7) but well inside what the
        // old relative-to-rhs formula (1e-4 on this row) would have forgiven:
        // must now be rejected, proving the check no longer scales with rhs.
        assert!(!incumbent_feasible(
            &p,
            &fixed,
            &[1000.0 + 5e-5],
            &tolerances
        ));
    }
}
