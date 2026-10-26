//! Presolve: problem reductions applied before the simplex / branch & bound
//! starts. Every reduction is exact: it keeps, up to round-off, every point
//! of the model as written within the engine's box whose integer vars are
//! exactly integral, the box holding each row to the budget the engine holds
//! it to (*Tolerances*) and each var to its bounds within their tolerance
//! (dual fixing keeps an optimum instead: *Soundness by mode*):
//!
//! * a row the bounds make redundant is dropped;
//! * an integer var's bounds are rounded and tightened, and an integer var is
//!   fixed where its rows leave it one value, or by dual fixing;
//! * a binary's big-M coefficient is tightened.
//!
//! A continuous var keeps its bounds as written: presolve fixes none and
//! turns no row into bounds. Such a reduction would take from the var the
//! room the model as written leaves it (the row's allowance over the
//! coefficient, or its bound tolerance), and the rooms of several vars can
//! add up in a row presolve keeps, beyond that row's tolerance.
//!
//! No other point is guaranteed. A point beyond the box, within the
//! contract by less than the round-off of evaluating a row, can be cut: a
//! working bound keeps only the budget of the row that implies it, and what
//! is deduced from that bound inherits it. So can a point within the box
//! only with an integer var off its integer, within `int_tol`, under a
//! coefficient that makes the offset count: an LP value `1 - 3e-14` under
//! `5e7` moves its row by `1.6e-6`. Without presolve the MILP search can
//! reach such a point through its LP; fixing the var, tightening its bounds
//! or shrinking its coefficient takes that path away, as a branch on the
//! var does. So where every point of a model needs the integrality
//! tolerance, presolve on can answer `Infeasible`, or a worse optimum,
//! where presolve off finds one: either answer is right (ARCHITECTURE §7.1).
//!
//! The variable set never changes: a var presolve "removes" gets `lo == hi`
//! bounds, which the simplex prices out, and its value is substituted out of
//! every row. Rows may be dropped or rewritten freely, since nothing outside
//! the `Solver` addresses constraints by index. So no postsolve is needed:
//! read-out, resume, warm starts and post-solve edits all work off the
//! original variable indices.
//!
//! # Working vs emitted bounds
//!
//! WORKING bounds collect every valid implication and drive further
//! deductions. EMITTED bounds are what the reduced problem carries: a
//! continuous var's bounds as written, and an integer var's rounded, implied
//! or fixed bounds. A continuous bound implied by a row stays working-only. A
//! row is dropped only when implied by the EMITTED bounds, never by working
//! bounds whose justification might be that very row.
//!
//! # Tolerances
//!
//! Every decision is made in user units against the engine's contract
//! (ARCHITECTURE §7.4, [`row_tolerance`]), so that presolve on and off
//! agree on what is feasible:
//!
//! * The engine holds a row of magnitude `m` (`|b| + Σ|a_j x_j|`, `b` the
//!   right-hand side as written, [`Row::rhs_mag`](model::Row::rhs_mag)) and
//!   `n` terms to a budget of [`slack_tol`]`(feasibility, m, n)`, its
//!   contract less the round-off of evaluating it, and the boundary accepts
//!   the full `row_tolerance`. Presolve never introduces a violation above
//!   the budget in a row the engine holds. `m` is taken at the corner of the
//!   box the decision is about, never over the whole box, so a bound of
//!   `f64::MAX` cannot inflate the tolerance of a row evaluated at small
//!   values.
//! * Every var may leave its EMITTED bounds by its bound tolerance (as
//!   `Solver::first_violated_bound` allows), an integer var too: the engine
//!   reports values as it computed them. An activity range therefore reaches
//!   `slack = Σ|a_j| · bound_tol_j` beyond its bounds, which every verdict
//!   and every dropped row accounts for (an integer var its emitted bounds
//!   fix the engine holds exactly, so for it the slack only keeps more).
//! * A working bound implied by a row is relaxed outward, over the
//!   coefficient, by the row's budget, by the round-off of the activity
//!   ([`ROUNDOFF_FLOOR`]` * m`, including what a cancellation leaves) and by
//!   the slack, so it never cuts a point the engine may return.
//! * An integer var's bound implied by a row keeps every integral value that
//!   meets the row within its budget, the other vars within their working
//!   bounds: a singleton row rounds the var's implied bounds, and is dropped
//!   only where those bounds, with their bound tolerance, meet it within its
//!   whole contract less round-off; a forcing row fixes the var only where no
//!   other integer lies within the room the row leaves it (its budget at its
//!   magnitude, round-off part included, over the coefficient).
//! * A big-M row that coefficient tightening rewrites keeps, up to
//!   round-off, every point of its binding branch within the budget, its
//!   rewritten binaries exactly at one: the branch is relaxed by the contract
//!   the rewrites take out of it at the magnitude of the row's fixed terms,
//!   fixed before the rewrites or after them (`Work::rewrite_room`), and its
//!   bound is derived from the one as written (`Work::rewritten_bound`).
//!   Where that arithmetic overflows `f64`, the row stays as written
//!   (`Work::hold_rewrites`): a rewrite only tightens the relaxation.
//! * A row whose vars are all fixed is judged exactly as the boundary judges
//!   it once dropped ([`row_violation`], full contract): nothing the engine
//!   does can move it, since a var fixed on entry never leaves its value.
//! * Every row presolve drops or rewrites is still checked against the answer
//!   as written ([`Presolved::validation_rows`]), so a wrong rule cannot
//!   return a point the model rejects: an LP takes over a row its answer
//!   misses and resumes, and the MILP guard rejects such a candidate.
//! * Presolve's own `Infeasible` carries no certificate, so it is not final:
//!   the solve then goes on with the model as written ([`reduced_model`]),
//!   where the engine decides on a proof. The engine's `Infeasible` on the
//!   reduced model is final. The reductions keep, up to round-off, a point
//!   within the box with its integer vars exactly integral wherever the model
//!   as written has one.
//!
//! [`row_tolerance`]: crate::solver::row_tolerance
//! [`ROUNDOFF_FLOOR`]: crate::solver::ROUNDOFF_FLOOR
//! [`row_violation`]: crate::solver::row_violation
//! [`slack_tol`]: crate::solver::slack_tol
//!
//! # Soundness by mode
//!
//! * [`Mode::Lp`] applies only reductions that keep the feasible set exactly
//!   (within the tolerances above), so facts baked into the live solver stay
//!   valid under every later edit (`Solution::add_constraint` and `fix_var`
//!   only shrink the region).
//! * [`Mode::Mip`] adds reductions that keep the integer feasible set
//!   (integer bound rounding, binary coefficient tightening) or only some
//!   optimum (dual fixing). This is sound because MILP post-solve edits
//!   re-solve from the untouched `MipState::base` and incumbents are checked
//!   against the original rows. Dual fixing also needs `allow_dual`, off when
//!   a warm-start hint accompanies the solve, so that a feasible hint is
//!   never excluded by an optimality-only argument, and when the caller turns
//!   it off (`SolveOptions::dual_fixing`).

mod model;
mod params;
mod work;

pub(crate) use model::{Mode, PresolveStats, Presolved};

use model::{is_int_domain, Row};
use params::*;
use work::Work;

use crate::solver::trivial_row_verdict;
use crate::{ComparisonOp, CsVec, Error, VarDomain};

/// Run presolve. `obj_coeffs` are the internal (minimize-space) objective
/// coefficients as stored on `Problem`; `feasibility` is the user's absolute
/// feasibility tolerance, the contract every decision is made against.
/// Returns `Err(Infeasible)` when the reductions find no acceptable point,
/// which the solve does not take as final ([`reduced_model`]).
#[allow(clippy::too_many_arguments)] // mirrors Solver::try_new's flat-arrays style
pub(crate) fn presolve(
    obj_coeffs: &[f64],
    var_mins: &[f64],
    var_maxs: &[f64],
    constraints: &[(CsVec, ComparisonOp, f64)],
    var_domains: &[VarDomain],
    mode: Mode,
    feasibility: f64,
    int_tol: f64,
    allow_dual: bool,
) -> Result<Presolved, Error> {
    let started = web_time::Instant::now();
    let num_vars = obj_coeffs.len();
    let mut lo = var_mins.to_vec();
    let mut hi = var_maxs.to_vec();

    // Crossed input bounds: same verdict Solver::try_new gives. The bounds
    // are numbers, each infinite only on its open side: the model's
    // validation (`Problem::solve_with`) takes no others.
    for v in 0..num_vars {
        debug_assert!(
            lo[v] < f64::INFINITY && hi[v] > f64::NEG_INFINITY,
            "bounds the model's validation refuses reached presolve"
        );
        if lo[v] > hi[v] {
            return Err(Error::Infeasible);
        }
    }
    // In Mip mode integer bounds are rounded up front: this keeps the integer
    // feasible set exactly and makes later fixings integral.
    if mode == Mode::Mip {
        for v in 0..num_vars {
            if is_int_domain(&var_domains[v]) {
                if lo[v].is_finite() {
                    lo[v] = (lo[v] - int_tol).ceil();
                }
                if hi[v].is_finite() {
                    hi[v] = (hi[v] + int_tol).floor();
                }
                if lo[v] > hi[v] {
                    return Err(Error::Infeasible);
                }
            }
        }
    }

    let mut dropped_on_input = 0usize;
    let mut rows: Vec<Row> = Vec::with_capacity(constraints.len());
    for (orig, (coeffs, op, rhs)) in constraints.iter().enumerate() {
        // Rows whose data alone decide them (empty, or an infinite
        // right-hand side): the verdict Solver::try_new gives them, under the
        // same contract.
        match trivial_row_verdict(coeffs, *op, *rhs, feasibility) {
            Some(true) => {
                dropped_on_input += 1;
                continue;
            }
            Some(false) => return Err(Error::Infeasible),
            None => {}
        }
        rows.push(Row::as_written(orig, coeffs, *op, *rhs));
    }

    // var -> rows adjacency (flat CSR). Rows only ever lose terms, so this
    // superset stays valid for the whole run.
    let mut adj_start = vec![0u32; num_vars + 1];
    for row in &rows {
        for &v in &row.vars {
            adj_start[v + 1] += 1;
        }
    }
    for v in 0..num_vars {
        adj_start[v + 1] += adj_start[v];
    }
    let mut adj_rows = vec![0u32; adj_start[num_vars] as usize];
    let mut fill = adj_start.clone();
    for (r, row) in rows.iter().enumerate() {
        for &v in &row.vars {
            adj_rows[fill[v] as usize] = r as u32;
            fill[v] += 1;
        }
    }

    let mut work = Work {
        wlo: lo.clone(),
        whi: hi.clone(),
        elo: lo,
        ehi: hi,
        domains: var_domains,
        constraints,
        mode,
        feas: feasibility,
        stats: PresolveStats {
            rows_dropped: dropped_on_input,
            ..PresolveStats::default()
        },
        adj_start,
        adj_rows,
        dirty: vec![true; rows.len()],
        queue: (0..rows.len() as u32).collect(),
        coeff_dirty: vec![true; rows.len()],
    };

    // Fixpoint: drain the primal worklist in waves, then (Mip) run the
    // coefficient and dual reductions, which requeue the rows they touch. The
    // wave cap is a safety net: every reduction is optional, so abandoning
    // leftover work is always sound.
    let mut rounds = 0;
    loop {
        while !work.queue.is_empty() && work.stats.passes < MAX_PASSES * MAX_ROUNDS {
            work.stats.passes += 1;
            work.primal_wave(&mut rows)?;
        }
        if mode != Mode::Mip {
            break;
        }
        let mut dual_changed = work.tighten_binary_coeffs(&mut rows);
        if allow_dual {
            dual_changed |= work.dual_fix(&rows, obj_coeffs);
        }
        rounds += 1;
        if !dual_changed || rounds >= MAX_ROUNDS {
            break;
        }
    }

    // Rebuild the output rows in the Problem's (coeffs, op, rhs) form, and keep
    // every input row the output does not carry as written for the engine to
    // check the answer against (`Presolved::validation_rows`).
    let mut out = Vec::with_capacity(rows.len());
    let mut validation_rows = Vec::new();
    for row in rows.iter_mut() {
        if !row.alive {
            validation_rows.push(constraints[row.orig as usize].clone());
            continue;
        }
        // A row can become all-substituted after the final pass (e.g. by a
        // last dual fixing): judge it here instead of emitting an empty row.
        work.fold_fixed(row);
        if row.vars.is_empty() {
            work.empty_row_check(row)?;
            validation_rows.push(constraints[row.orig as usize].clone());
            continue;
        }
        if !row.coeffs_touched {
            // Not rewritten: emit the input row as written, fixed terms
            // included. Folded, the row would have a smaller magnitude, hence
            // a tighter tolerance than the contract grants it, and the fold's
            // round-off (1e-7 on a shift of 1e9) could make two consistent
            // rows disagree beyond that tolerance.
            out.push(constraints[row.orig as usize].clone());
            continue;
        }
        // Rewritten: the rewritten terms, the folded terms restored so that
        // the row keeps its magnitude, and the bound derived from the one as
        // written (`Work::rewritten_bound`). The rewrite holds for integer
        // points only, so the answer is also checked against the input row.
        // A bound the arithmetic overflows leaves the row as written, which
        // keeps every point the rewrites kept.
        let (op, rhs) = work.rewritten_bound(row);
        if !rhs.is_finite() {
            out.push(constraints[row.orig as usize].clone());
            continue;
        }
        validation_rows.push(constraints[row.orig as usize].clone());
        let mut vars = row.vars.clone();
        let mut coeffs = row.coeffs.clone();
        for &(v, a) in &row.folded {
            vars.push(v);
            coeffs.push(a);
        }
        out.push((
            CsVec::new_from_unsorted(num_vars, vars, coeffs)
                .expect("presolve rows have unique indices"),
            op,
            rhs,
        ));
    }

    debug!(
        "presolve ({:?}): rows {} -> {}, {} bounds tightened, {} vars fixed, {} coeffs tightened, {} passes, {:?}",
        mode,
        constraints.len(),
        out.len(),
        work.stats.bounds_tightened,
        work.stats.vars_fixed,
        work.stats.coeffs_tightened,
        work.stats.passes,
        started.elapsed(),
    );

    // A row kept unchanged is emitted as written, so the rows are the model's
    // unless one was dropped or rewritten (a validation row) or dropped on
    // input. Bounds compare as values: rounding an integer bound of 0 gives -0.
    let reduced = work.elo != var_mins
        || work.ehi != var_maxs
        || !validation_rows.is_empty()
        || out.len() != constraints.len();
    debug_assert!(
        (0..num_vars).all(|v| {
            (mode == Mode::Mip && is_int_domain(&var_domains[v]))
                || (work.elo[v] == var_mins[v] && work.ehi[v] == var_maxs[v])
        }),
        "presolve changed a continuous var's bounds"
    );
    Ok(Presolved {
        var_mins: work.elo,
        var_maxs: work.ehi,
        constraints: out,
        validation_rows,
        reduced,
        stats: work.stats,
    })
}

/// [`presolve`]'s result as the solve takes it: the reduced model, or `None`
/// to solve the model as written. Presolve's `Infeasible` carries no
/// certificate, so it is not final: the engine returns `Infeasible` only on a
/// proof and a MILP's search decides from there. A model presolve did not
/// reduce is solved as written.
pub(crate) fn reduced_model(result: Result<Presolved, Error>) -> Result<Option<Presolved>, Error> {
    match result {
        Ok(presolved) if presolved.reduced => Ok(Some(presolved)),
        Ok(_) => {
            debug!("presolve reduced nothing; solving the model as written");
            Ok(None)
        }
        Err(Error::Infeasible) => {
            debug!("presolve found the model infeasible; solving it as written");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests;
