//! Presolve: problem reductions applied before the simplex / branch & bound
//! starts.
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
//! deductions. EMITTED bounds are what the reduced problem carries: only
//! deductions whose source row is removed (singleton conversions, forcing
//! fixings), integer roundings and variable fixings. A continuous bound
//! implied by a kept row is not emitted, since it is redundant with the row
//! and would only perturb the pivot order. A row is dropped only when
//! implied by the EMITTED bounds, never by working bounds whose
//! justification might be that very row.
//!
//! # Tolerances
//!
//! Every decision is made in user units against the engine's contract
//! (ARCHITECTURE §7, [`row_tolerance`]), so that presolve on and off agree on
//! what is feasible:
//!
//! * The engine holds a row of magnitude `m` (`|b| + Σ|a_j x_j|`, `b` the
//!   right-hand side as written, [`Row::rhs_mag`](model::Row::rhs_mag)) to a
//!   budget of `ROW_BUDGET_SHARE * row_tolerance(feasibility, m)`, and the
//!   boundary accepts the full `row_tolerance`. Presolve never introduces a
//!   violation above the budget in a row the engine holds, and never
//!   declares infeasible what a point within the budget could satisfy. `m`
//!   is taken at the corner of the box the decision is about, never over the
//!   whole box, so a bound of `f64::MAX` cannot inflate the tolerance of a
//!   row evaluated at small values.
//! * A continuous var may leave its EMITTED bounds by its bound tolerance
//!   (as `Solver::first_violated_bound` allows); an integer var's adopted
//!   value is exactly integral. An activity range therefore reaches
//!   `slack = Σ|a_j| · bound_tol_j` beyond its bounds, which every verdict
//!   and every dropped row accounts for.
//! * A working bound implied by a row is relaxed outward, over the
//!   coefficient, by the row's allowance (the budget, or for an integer var
//!   the incumbent guard's full contract), by the round-off of the activity
//!   ([`ROUNDOFF_FLOOR`]` * m`, including what a cancellation leaves) and by
//!   the slack, so it never cuts a point the engine may return.
//! * A row becomes bounds (a singleton conversion, an equality or forcing
//!   fixing) only where the bounds keep what the row does, as the engine and
//!   the boundary would hold it (`Work::singleton_row`,
//!   `Work::forcing_is_exact`).
//! * A row whose vars are all fixed is judged exactly as the boundary judges
//!   it once dropped ([`row_violation`], full contract): nothing the engine
//!   does can move it.
//! * Each worklist wave processes the rows with at most one term first, so a
//!   singleton pins its var before a multi-term row deduces anything about it.
//! * Every row presolve drops or rewrites is still checked against the answer
//!   as written ([`Presolved::validation_rows`]), so a wrong rule fails loudly
//!   at the boundary.
//!
//! [`row_violation`]: crate::solver::row_violation
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
//!   never excluded by an optimality-only argument.

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
/// Returns `Err(Infeasible)` when the reductions prove no acceptable point
/// exists.
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

    // Crossed or NaN input bounds: same verdict Solver::try_new gives.
    for v in 0..num_vars {
        if lo[v].is_nan() || hi[v].is_nan() || lo[v] > hi[v] {
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
        // Rows whose data alone decide them (empty, non-finite): the verdict
        // Solver::try_new gives them, under the same contract.
        match trivial_row_verdict(coeffs, *op, *rhs, feasibility) {
            Some(true) => {
                dropped_on_input += 1;
                continue;
            }
            Some(false) => return Err(Error::Infeasible),
            None => {}
        }
        let (lo, hi) = match op {
            ComparisonOp::Le => (f64::NEG_INFINITY, *rhs),
            ComparisonOp::Ge => (*rhs, f64::INFINITY),
            ComparisonOp::Eq => (*rhs, *rhs),
        };
        rows.push(Row {
            vars: coeffs.indices().to_vec(),
            coeffs: coeffs.data().to_vec(),
            lo,
            hi,
            fold_scale: 0.0,
            shift: 0.0,
            folded: Vec::new(),
            alive: true,
            orig: orig as u32,
            coeffs_touched: false,
        });
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
        // Rewritten: restore the folded terms and shift the bounds back, so
        // the row keeps its magnitude; the shift's round-off is within the
        // relaxation the rewrite applied. The rewrite holds for integer
        // points only, so the answer is also checked against the input row.
        validation_rows.push(constraints[row.orig as usize].clone());
        let shift: f64 = row.folded.iter().map(|&(v, a)| a * work.elo[v]).sum();
        let (op, rhs) = match (row.lo.is_finite(), row.hi.is_finite()) {
            (false, true) => (ComparisonOp::Le, row.hi + shift),
            (true, false) => (ComparisonOp::Ge, row.lo + shift),
            (true, true) => {
                // Substitution shifts both sides identically and every other
                // transformation keeps rows one-sided, so a two-sided row is
                // an Eq row to the last bit.
                assert!(
                    row.lo == row.hi,
                    "presolve produced a two-sided row: [{}, {}] — this is a bug",
                    row.lo,
                    row.hi
                );
                (ComparisonOp::Eq, row.lo + shift)
            }
            (false, false) => {
                unreachable!("a both-sides-infinite row is always dropped as redundant")
            }
        };
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

    Ok(Presolved {
        var_mins: work.elo,
        var_maxs: work.ehi,
        constraints: out,
        validation_rows,
        stats: work.stats,
    })
}

#[cfg(test)]
mod tests;
