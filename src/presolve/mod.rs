//! Presolve: problem reductions applied before the simplex / branch & bound
//! starts.
//!
//! The variable set is never changed: a variable presolve "removes" is fixed
//! by giving it `lo == hi` bounds (the simplex prices such vars out natively
//! — `choose_pivot` skips vars that are at both bounds) and substituting its
//! value out of every row. Rows, by contrast, may be dropped or rewritten
//! freely: nothing outside the `Solver` addresses constraints by index. This
//! is what makes a postsolve layer unnecessary — solution read-out, resume,
//! warm starts and post-solve edits all work off original variable indices.
//!
//! # Working vs emitted bounds
//!
//! Bound deductions live in two layers:
//!
//! * WORKING bounds collect every valid implication (activity-based
//!   tightenings included) and drive further deductions: infeasibility
//!   detection, cascade tightening, coefficient tightening.
//! * EMITTED bounds are what the reduced problem actually carries. Only
//!   deductions whose *source row is removed* (singleton conversions,
//!   forcing fixings), integer roundings and variable fixings persist.
//!
//! An activity-derived continuous bound whose source row is kept is
//! deliberately NOT emitted: it is redundant with that row, so emitting it
//! would only perturb simplex pivot order (ulp-level objective jitter for
//! zero structural gain). Keeping emitted bounds minimal also makes row
//! elimination self-consistent: a row is dropped only when implied by the
//! EMITTED bounds — material that provably stays in the reduced problem —
//! never by working bounds whose own justification might be the very row
//! being dropped.
//!
//! # Tolerances
//!
//! Every decision is made against the engine's contract (ARCHITECTURE §7,
//! [`row_tolerance`]), in user units, so that presolve on and off agree on
//! what is feasible:
//!
//! * A row of magnitude `m` (`|b| + Σ|a_j x_j|` at the point in question)
//!   is held by the engine to a BUDGET of `ROW_BUDGET_SHARE *
//!   row_tolerance(feasibility, m)`; the checks at the boundary accept the
//!   full `row_tolerance`. Presolve never introduces a violation above the
//!   budget and never declares infeasible what a point within the budget
//!   could satisfy. The magnitude is taken at the corner of the box the
//!   decision is about (the minimum-activity corner for an upper row bound,
//!   and so on), never over the whole box: a bound of `f64::MAX` must not
//!   inflate the tolerance of a row evaluated at small values.
//! * A continuous var may sit outside its EMITTED bounds by its bound
//!   tolerance `row_tolerance(feasibility, |bound|)`, exactly as
//!   `Solver::first_violated_bound` allows; an integer var's adopted value
//!   is exactly integral. An activity range therefore reaches
//!   `slack = Σ|a_j| · bound_tol_j` beyond what the bounds say, and every
//!   verdict and every dropped row accounts for that.
//! * Round-off: an activity computed from terms of magnitude `m` is exact
//!   only to [`ROUNDOFF_FLOOR`]` * m`; the one equality test (forcing rows)
//!   uses that.
//! * A working bound implied by a row is relaxed outward by the row's
//!   budget plus slack, divided by the coefficient: the implication then
//!   holds for every point the engine may return, so the bound never cuts
//!   one. This is what makes a tiny coefficient a weak witness (a `1e-7`
//!   coefficient in a row held to `5e-8` determines its var to about `0.5`),
//!   exactly as the engine treats it.
//! * A row becomes bounds only where a bound describes it at least as well
//!   as the engine would hold the row: forcing rows are judged over the
//!   EMITTED bounds, only when every term's range is resolvable above the
//!   row's round-off (a term below it is invisible to the row and cannot be
//!   forced), and fix vars at those exact values; a singleton row is
//!   converted only when the engine's bound tolerance keeps the row within
//!   its budget, or when it fixes the var (a fixed var is exact); and a
//!   conversion is skipped when the round-off it moves into the var,
//!   amplified by the var's coefficient in another row, would exceed that
//!   row's budget — the rows disagree at the level of their own round-off,
//!   and only the simplex, which balances both, resolves that.
//!
//! # Soundness by mode
//!
//! * [`Mode::Lp`] applies only feasible-set-EXACT reductions (within the
//!   tolerances above): the presolved problem has the same set of feasible
//!   points as the input. Facts baked into the live solver therefore stay
//!   valid under every later edit (`Solution::add_constraint` / `fix_var` /
//!   Gomory cuts only shrink the region; implied facts remain implied).
//! * [`Mode::Mip`] adds reductions that preserve the INTEGER feasible set
//!   (integer bound rounding, binary coefficient tightening) or merely some
//!   optimum (dual fixing). Sound because MILP post-solve edits re-solve from
//!   the untouched `MipState::base` and incumbents are validated against the
//!   original rows. Dual fixing is additionally gated on `allow_dual`: it is
//!   disabled when a warm-start hint accompanies the solve, so a feasible
//!   hint is never excluded by an optimality-only argument.

mod model;
mod params;
mod work;

pub(crate) use model::{Mode, PresolveStats, Presolved};

use model::{is_int_domain, Row};
use params::*;
use work::Work;

use crate::{ComparisonOp, CsVec, Error, VarDomain};

/// Run presolve. `obj_coeffs` are the INTERNAL (minimize-space) objective
/// coefficients, exactly as stored on `Problem`; `feasibility` is the user's
/// absolute feasibility tolerance (`Tolerances::feasibility`), the contract
/// every decision is made against (module docs). Returns `Err(Infeasible)`
/// when the reductions prove no acceptable point exists.
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

    // Crossed input bounds: same verdict Solver::try_new gives.
    for v in 0..num_vars {
        if lo[v] > hi[v] {
            return Err(Error::Infeasible);
        }
    }
    // Integer bounds are integral from the start in Mip mode (preserves the
    // integer feasible set exactly; also what makes later fixings integral).
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
        if coeffs.indices().is_empty() {
            // User-authored empty row: replicate Solver::try_new's exact
            // (tolerance-free) semantics.
            let tautological = match op {
                ComparisonOp::Eq => *rhs == 0.0,
                ComparisonOp::Le => 0.0 <= *rhs,
                ComparisonOp::Ge => 0.0 >= *rhs,
            };
            if tautological {
                dropped_on_input += 1;
                continue;
            } else {
                return Err(Error::Infeasible);
            }
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
    // coefficient/dual reductions, which push the rows they touch back into
    // the worklist. The wave cap is a global safety net; every reduction is
    // optional, so abandoning leftover work is always sound.
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

    // Rebuild the output rows in the Problem's (coeffs, op, rhs) form.
    let mut out = Vec::with_capacity(rows.len());
    for row in rows.iter_mut() {
        if !row.alive {
            continue;
        }
        // A row can become all-substituted after the final pass (e.g. by a
        // last dual fixing): judge it here instead of emitting an empty row.
        work.fold_fixed(row);
        if row.vars.is_empty() {
            work.empty_row_check(row)?;
            continue;
        }
        if !row.coeffs_touched {
            // Not rewritten: emit the input row byte-identically, fixed
            // terms included. Folding them out would hand the engine a row
            // of smaller magnitude, held to a tighter tolerance than the
            // contract grants the original row; the fold's own round-off
            // (a shift of magnitude 1e9 is exact only to 1e-7) could then
            // make two consistent rows disagree beyond it. (A CsVec clone is
            // also a memcpy; rebuilding would re-sort for nothing.)
            out.push(constraints[row.orig as usize].clone());
            continue;
        }
        // Rewritten: rebuild with the folded terms restored and the bounds
        // shifted back, so the row keeps its magnitude here too. The shift's
        // round-off is within the relaxation the rewrite already applied.
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
        stats: work.stats,
    })
}

#[cfg(test)]
mod tests;
