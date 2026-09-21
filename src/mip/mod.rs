//! Branch & bound driver for mixed-integer problems.
//!
//! Owns exactly one [`Solver`] per search. Branching changes variable bounds in
//! place.
//!
//! This file is the lifecycle: how a search is built from a [`Problem`], how it
//! is started, resumed and re-edited, and how a candidate is judged against the
//! user's original rows. The pieces beside it:
//!
//! * [`options`] — the public option and report types.
//! * [`state`] — [`MipState`], the resumable state a search carries.
//! * [`search`] — node visiting, branching and the search loop.
//! * [`branching`] / [`node`] / [`params`] — pseudocosts, tree nodes, constants.

pub(crate) mod branching;
pub(crate) mod node;
mod options;
pub(crate) mod params;
mod search;
mod state;

pub use options::{
    ResumeOptions, SolutionStatus, SolveOptions, Stats, TerminationReason, Tolerances,
};
pub(crate) use state::{MipRun, MipState};

use crate::solver::Solver;
use crate::{ComparisonOp, Error, OptimizationDirection, Problem, VarDomain, Variable};
use search::search_loop;
use std::collections::BTreeMap;
use web_time::Instant;

fn build_state(problem: &Problem, options: SolveOptions) -> Result<MipState, Error> {
    let deadline = options.time_limit.map(|d| Instant::now() + d);
    // Presolve the search problem. `state.base` below stays the user's
    // untouched problem, so post-solve edits keep composing against it; the
    // presolved bounds become the ROOT bounds (branching resets to them).
    // Dual fixing is optimum-preserving but not feasible-point-preserving,
    // so it is disabled when a warm-start hint must be honored.
    let presolved = if options.presolve {
        Some(crate::presolve::presolve(
            &problem.obj_coeffs,
            &problem.var_mins,
            &problem.var_maxs,
            &problem.constraints,
            &problem.var_domains,
            crate::presolve::Mode::Mip,
            options.tolerances.feasibility,
            options.int_tol,
            options.warm_start.is_none(),
        )?)
    } else {
        None
    };
    let (var_mins, var_maxs, constraints) = match &presolved {
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
        options.tolerances.feasibility,
    )?;
    let root_bounds = var_mins
        .iter()
        .zip(var_maxs)
        .map(|(&lo, &hi)| (lo, hi))
        .collect();
    let pseudocosts = branching::PseudoCosts::new(&problem.obj_coeffs, problem.obj_coeffs.len());
    Ok(MipState {
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
        classifying_unbounded: false,
    })
}

/// Replace a state whose original relaxation is unbounded with a
/// zero-objective integer-feasibility search. For a rational MILP, one
/// integer-feasible point plus the original relaxation ray proves
/// unboundedness; exhausting this search proves integer infeasibility.
fn begin_unbounded_classification(state: &mut MipState) -> Result<(), Error> {
    let base = state.base.clone();
    let fixed = state.fixed.clone();
    let mut feasibility = effective_problem(&base, &fixed);
    feasibility.obj_coeffs.fill(0.0);

    let deadline = state.deadline;
    let elapsed = state.stats.elapsed;
    let lp_iterations = state.solver.lp_iterations;
    let mut replacement = build_state(&feasibility, state.options.clone())?;
    replacement.deadline = deadline;
    replacement.solver.deadline = deadline;
    replacement.solver.lp_iterations = lp_iterations;
    replacement.stats.elapsed = elapsed;
    replacement.base = base;
    replacement.fixed = fixed;
    replacement.classifying_unbounded = true;
    *state = replacement;
    Ok(())
}

fn resume_or_classify(state: &mut MipState) -> Result<TerminationReason, Error> {
    match resume_run_with_deadline(state) {
        Err(Error::Unbounded) if !state.classifying_unbounded => {
            begin_unbounded_classification(state)?;
            resume_run_with_deadline(state)
        }
        result => result,
    }
}

/// Build the search state for `problem` and run it under `options`.
pub(crate) fn run(problem: &Problem, options: SolveOptions) -> Result<MipRun, Error> {
    let mut state = build_state(problem, options)?;
    let reason = resume_or_classify(&mut state)?;
    Ok(MipRun { reason, state })
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

pub(crate) fn candidate_variables_feasible(
    values: &[f64],
    domains: &[VarDomain],
    tolerances: &Tolerances,
    mut bounds: impl FnMut(usize) -> (f64, f64),
) -> bool {
    if values.len() != domains.len() {
        return false;
    }
    values.iter().enumerate().all(|(v, &value)| {
        let (lo, hi) = bounds(v);
        // The contract on a bound, the same rule and the same comparison as
        // the engine's and the LP path's.
        let tol = crate::solver::bound_tolerance(tolerances.feasibility, lo, hi);
        value.is_finite()
            && !lo.is_nan()
            && !hi.is_nan()
            && lo <= hi
            && !crate::solver::outside(value, lo, hi, tol)
            && (!matches!(domains[v], VarDomain::Integer | VarDomain::Boolean)
                || (value - value.round()).abs() <= tolerances.integrality_rounding)
    })
}

/// Cheap feasibility check of a value vector against base + fixes: bounds,
/// domains, and every user-scale constraint row. This is a warm-start prefilter;
/// adoption re-validates the candidate against the active solver's scaled rows.
pub(crate) fn incumbent_feasible(
    base: &Problem,
    fixed: &BTreeMap<usize, f64>,
    values: &[f64],
    tolerances: &Tolerances,
) -> bool {
    first_violation(base, fixed, values, tolerances).is_none()
}

/// [`incumbent_feasible`] with the first offending bound or row described,
/// for error messages.
pub(crate) fn first_violation(
    base: &Problem,
    fixed: &BTreeMap<usize, f64>,
    values: &[f64],
    tolerances: &Tolerances,
) -> Option<String> {
    if !candidate_variables_feasible(values, &base.var_domains, tolerances, |v| {
        fixed
            .get(&v)
            .map_or((base.var_mins[v], base.var_maxs[v]), |&value| {
                (value, value)
            })
    }) {
        return Some(format!(
            "a variable value violates its bounds or domain: {values:?}"
        ));
    }
    for (row, (coeffs, op, rhs)) in base.constraints.iter().enumerate() {
        let mut lhs = 0.0;
        let mut magnitude = rhs.abs();
        for (i, c) in coeffs.iter() {
            let term = c * values[i];
            lhs += term;
            magnitude += term.abs();
        }
        if !lhs.is_finite() {
            return Some(format!("row {row} has a non-finite activity"));
        }
        // The contract on a row, as `Solver::first_violated_row` applies it:
        // the slack the row implies, against the slack bounds its sense
        // encodes.
        let tol = crate::solver::row_tolerance(tolerances.feasibility, magnitude);
        let (smin, smax) = match op {
            ComparisonOp::Eq => (0.0, 0.0),
            ComparisonOp::Le => (0.0, f64::INFINITY),
            ComparisonOp::Ge => (f64::NEG_INFINITY, 0.0),
        };
        if crate::solver::outside(rhs - lhs, smin, smax, tol) {
            return Some(format!(
                "row {row} ({op:?} {rhs:e}) has activity {lhs:e}, tolerance {tol:e}, at {values:?}"
            ));
        }
    }
    None
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

/// Continue a paused search with fresh per-call budgets and options.
pub(crate) fn resume_run(
    state: &mut MipState,
    options: ResumeOptions,
) -> Result<TerminationReason, Error> {
    options.validate()?;
    state.deadline = options.time_limit.map(|d| Instant::now() + d);
    state.options.time_limit = options.time_limit;
    state.options.node_limit = options.node_limit;
    state.options.mip_gap = options.mip_gap.unwrap_or(0.0);
    resume_or_classify(state)
}

fn resume_run_with_deadline(state: &mut MipState) -> Result<TerminationReason, Error> {
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
pub(crate) fn global_bound_internal(state: &MipState) -> Option<f64> {
    if !state.root_solved {
        return None;
    }
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
pub(crate) fn relative_gap(incumbent_obj: f64, bound: f64) -> f64 {
    (incumbent_obj - bound).max(0.0) / incumbent_obj.abs().max(params::GAP_DENOM_GUARD)
}

fn to_user_space(direction: OptimizationDirection, internal: f64) -> f64 {
    match direction {
        OptimizationDirection::Minimize => internal,
        OptimizationDirection::Maximize => -internal,
    }
}

pub(crate) fn fill_bound_stats(state: &mut MipState) {
    if state.classifying_unbounded {
        state.stats.best_bound = None;
        state.stats.gap = None;
        return;
    }
    let bound = global_bound_internal(state);
    state.stats.best_bound = bound.map(|b| to_user_space(state.direction, b));
    state.stats.gap = match (&state.incumbent, bound) {
        (Some(inc), Some(b)) => Some(relative_gap(inc.objective, b)),
        _ => None,
    };
}

/// Prune threshold: a node whose lower bound is ≥ this cannot improve the incumbent.
pub(crate) fn cutoff(incumbent_obj: f64, prune_epsilon: f64) -> f64 {
    incumbent_obj - f64::max(prune_epsilon, prune_epsilon * incumbent_obj.abs())
}

#[cfg(test)]
mod tests;
