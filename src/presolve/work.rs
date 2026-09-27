//! [`Work`]: the bound state the reduction passes mutate, and the passes
//! themselves. The module docs in [`super`] give the working-vs-emitted bound
//! split and the tolerance model every decision here is made against.

use super::model::{is_int_domain, term_range, Activity, Mode, PresolveStats, Row};
use super::params::*;
use crate::solver::{
    bound_tolerance, row_tolerance, row_violation, ROUNDOFF_FLOOR, ROW_BUDGET_SHARE,
};
use crate::{ComparisonOp, CsVec, Error, VarDomain};

/// Bound state and reduction context. Rows live outside this struct so a
/// pass can hold a row borrow while it mutates bounds, which keeps the
/// per-row loops allocation-free.
pub(crate) struct Work<'a> {
    /// Working bounds: every valid deduction, used to deduce further.
    pub(crate) wlo: Vec<f64>,
    pub(crate) whi: Vec<f64>,
    /// Emitted bounds: what the reduced problem carries.
    /// Invariant: `[wlo, whi] ⊆ [elo, ehi]` per var.
    pub(crate) elo: Vec<f64>,
    pub(crate) ehi: Vec<f64>,
    pub(crate) domains: &'a [VarDomain],
    /// The input rows as written (indexed by [`Row::orig`]): what the
    /// boundary checks a dropped row against.
    pub(crate) constraints: &'a [(CsVec, ComparisonOp, f64)],
    pub(crate) mode: Mode,
    /// The user's absolute feasibility tolerance (`Tolerances::feasibility`).
    pub(crate) feas: f64,
    pub(crate) stats: PresolveStats,
    /// var -> rows adjacency (flat CSR), built once. Rows only lose terms, so
    /// it stays a superset.
    pub(crate) adj_start: Vec<u32>,
    pub(crate) adj_rows: Vec<u32>,
    /// Worklist: a row is queued when one of its inputs (a var bound, its own
    /// coefficients or rhs) changed since it was last processed, so the
    /// fixpoint costs O(changed work), not O(passes * nnz).
    pub(crate) dirty: Vec<bool>,
    pub(crate) queue: Vec<u32>,
    /// Rows whose inputs changed since the coefficient pass last saw them.
    pub(crate) coeff_dirty: Vec<bool>,
}

impl Work<'_> {
    fn is_int(&self, v: usize) -> bool {
        self.mode == Mode::Mip && is_int_domain(&self.domains[v])
    }

    /// Fixed means EMITTED-fixed: only then may the value be substituted out
    /// of rows (the reduced problem itself pins the var there).
    fn fixed(&self, v: usize) -> bool {
        self.elo[v] == self.ehi[v] && self.elo[v].is_finite()
    }

    /// The engine's budget for a row of magnitude `magnitude` (module docs).
    fn budget(&self, magnitude: f64) -> f64 {
        ROW_BUDGET_SHARE * row_tolerance(self.feas, magnitude)
    }

    /// How far a row of magnitude `magnitude` may be violated at the points a
    /// bound implied for var `v` must keep: the engine's budget, and for an
    /// integer var, whose values that matter are incumbents', the full
    /// contract the incumbent guard applies (rounding against the budget
    /// alone would cut integer points the guard accepts).
    fn allowance(&self, v: usize, magnitude: f64) -> f64 {
        if self.is_int(v) {
            row_tolerance(self.feas, magnitude)
        } else {
            self.budget(magnitude)
        }
    }

    /// Tolerance, in user units, within which the engine holds var `v` to its
    /// emitted upper (`upper`) or lower bound (`Solver::first_violated_bound`);
    /// zero for an integer var, whose adopted values are exactly integral.
    fn bound_tol(&self, v: usize, upper: bool) -> f64 {
        if self.is_int(v) {
            0.0
        } else {
            bound_tolerance(self.feas, if upper { self.ehi[v] } else { self.elo[v] })
        }
    }

    /// Activity range of `row` over the working or the emitted box.
    fn activity(&self, row: &Row, working: bool) -> Activity {
        let (lo, hi) = if working {
            (&self.wlo, &self.whi)
        } else {
            (&self.elo, &self.ehi)
        };
        let mut act = Activity {
            l: 0.0,
            u: 0.0,
            ninf_l: 0,
            ninf_u: 0,
            abs_l: 0.0,
            abs_u: 0.0,
            slack_l: 0.0,
            slack_u: 0.0,
        };
        for (&v, &a) in row.vars.iter().zip(&row.coeffs) {
            let (cmin, cmax) = term_range(a, lo[v], hi[v]);
            if cmin.is_finite() {
                act.l += cmin;
                act.abs_l += cmin.abs();
            } else {
                act.ninf_l += 1;
            }
            if cmax.is_finite() {
                act.u += cmax;
                act.abs_u += cmax.abs();
            } else {
                act.ninf_u += 1;
            }
            // At the minimum corner the var sits at its lower bound when
            // its coefficient is positive, at its upper bound otherwise.
            act.slack_l += a.abs() * self.bound_tol(v, a < 0.0);
            act.slack_u += a.abs() * self.bound_tol(v, a > 0.0);
        }
        act
    }

    /// Largest |coefficient| of `v` in a live row other than `except`.
    fn other_coeff(&self, rows: &[Row], v: usize, except: usize) -> f64 {
        let (s, e) = (self.adj_start[v] as usize, self.adj_start[v + 1] as usize);
        let mut max = 0.0f64;
        for &k in &self.adj_rows[s..e] {
            let k = k as usize;
            if k == except || !rows[k].alive {
                continue;
            }
            if let Some(i) = rows[k].vars.iter().position(|&u| u == v) {
                max = max.max(rows[k].coeffs[i].abs());
            }
        }
        max
    }

    /// Whether fixing every var of a forcing row at its extreme cuts no point
    /// a var can resolve. With its corner `sliver` short of its bound, the
    /// row leaves each var `room / |a|` off its extreme, `room` being the
    /// sliver plus the engine's share of the absolute tolerance (the
    /// round-off part of the budget is room no evaluation resolves). A
    /// continuous var's room must lie within its bound tolerance, and the
    /// sliver it gives up within what each of its other rows tolerates at its
    /// coefficient there; an integer var's room must be short of a unit.
    fn forcing_is_exact(&self, rows: &[Row], r: usize, sliver: f64, min_corner: bool) -> bool {
        let room = (sliver + ROW_BUDGET_SHARE * self.feas).max(0.0);
        let row = &rows[r];
        row.vars.iter().zip(&row.coeffs).all(|(&v, &a)| {
            if self.is_int(v) {
                room / a.abs() < 1.0
            } else {
                // The extreme the var is fixed at: its lower bound when its
                // coefficient pushes the activity towards the corner.
                let upper = (a < 0.0) == min_corner;
                room / a.abs() <= self.bound_tol(v, upper)
                    && self.other_coeff(rows, v, r) * sliver.max(0.0)
                        <= ROW_BUDGET_SHARE * self.feas * a.abs()
            }
        })
    }

    /// Whether every term of a forcing candidate can be told apart from the
    /// row's round-off. A var whose whole term range `|a|·(hi − lo)` lies
    /// below the sliver plus the round-off of the sum is invisible to the row
    /// (the corner meets the bound with the var anywhere in its range), so
    /// the row forces nothing about it and must not fix it.
    fn terms_resolvable(&self, row: &Row, sliver: f64, magnitude: f64) -> bool {
        row.vars.iter().zip(&row.coeffs).all(|(&v, &a)| {
            a.abs() * (self.ehi[v] - self.elo[v]) > sliver.max(0.0) + ROUNDOFF_FLOOR * magnitude
        })
    }

    /// Requeue one row (bound of one of its vars, or its own data, changed).
    fn mark_row(&mut self, r: usize) {
        if !self.dirty[r] {
            self.dirty[r] = true;
            self.queue.push(r as u32);
        }
        self.coeff_dirty[r] = true;
    }

    /// Requeue every row containing `v`.
    fn mark_var(&mut self, v: usize) {
        let (s, e) = (self.adj_start[v] as usize, self.adj_start[v + 1] as usize);
        for i in s..e {
            let r = self.adj_rows[i] as usize;
            self.mark_row(r);
        }
    }

    fn fix(&mut self, v: usize, val: f64) {
        debug_assert!(val.is_finite(), "presolve only fixes to finite values");
        self.wlo[v] = val;
        self.whi[v] = val;
        self.elo[v] = val;
        self.ehi[v] = val;
        self.stats.vars_fixed += 1;
        self.mark_var(v);
    }

    /// Emit `[lo, hi]` for `v` (a subset of its current emitted bounds) and
    /// shrink the working bounds into it.
    fn emit_bounds(&mut self, v: usize, lo: f64, hi: f64) {
        if lo > self.elo[v] || hi < self.ehi[v] {
            self.stats.bounds_tightened += 1;
            self.mark_var(v);
        }
        self.elo[v] = lo;
        self.ehi[v] = hi;
        // Working bounds may already be tighter; only ever shrink them.
        self.wlo[v] = self.wlo[v].max(lo).min(hi);
        self.whi[v] = self.whi[v].min(hi).max(lo);
    }

    /// Substitute emitted-fixed vars out of `row` (rhs shift) and drop exact
    /// zero coefficients.
    pub(crate) fn fold_fixed(&self, row: &mut Row) {
        let mut i = 0;
        while i < row.vars.len() {
            let v = row.vars[i];
            let a = row.coeffs[i];
            let fixed = self.fixed(v);
            if a == 0.0 || fixed {
                if fixed && a != 0.0 {
                    let shift = a * self.elo[v];
                    row.lo -= shift; // -inf and +inf survive the shift intact
                    row.hi -= shift;
                    row.fold_scale += shift.abs();
                    row.shift += shift;
                    row.folded.push((v, a));
                }
                row.vars.swap_remove(i);
                row.coeffs.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Record the implied upper bound `q` for var `v`, valid for every point
    /// the engine may return once relaxed outward by `relax` (in var units).
    /// Integer bounds round and are emitted; continuous bounds stay
    /// working-only. A crossing within the var's bound tolerance is no
    /// reduction (the solver handles slivers); beyond it, a proof of
    /// infeasibility.
    fn tighten_upper(&mut self, v: usize, q: f64, relax: f64) -> Result<bool, Error> {
        if !q.is_finite() {
            return Ok(false);
        }
        if self.is_int(v) {
            let qi = (q + relax).floor();
            if qi < self.ehi[v] - 0.5 {
                if qi < self.wlo[v] - 0.5 {
                    // Integral bounds crossing: an integer-empty range.
                    debug!(
                        "presolve: var {v} integer upper {qi} < lower {}",
                        self.wlo[v]
                    );
                    return Err(Error::Infeasible);
                }
                self.ehi[v] = qi;
                self.whi[v] = self.whi[v].min(qi);
                self.stats.bounds_tightened += 1;
                self.mark_var(v);
                return Ok(true);
            }
        } else {
            let nq = q + relax;
            if nq < self.whi[v] - IMPROVE_REL * nq.abs().max(1.0) {
                if nq < self.wlo[v] {
                    if self.wlo[v] - nq > self.bound_tol(v, false) {
                        debug!(
                            "presolve: var {v} implied upper {nq:e} < lower {:e}",
                            self.wlo[v]
                        );
                        return Err(Error::Infeasible);
                    }
                    return Ok(false); // sub-tolerance sliver: leave it alone
                }
                self.whi[v] = nq;
                self.stats.bounds_tightened += 1;
                self.mark_var(v);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Mirror of [`Self::tighten_upper`] for the lower bound.
    fn tighten_lower(&mut self, v: usize, q: f64, relax: f64) -> Result<bool, Error> {
        if !q.is_finite() {
            return Ok(false);
        }
        if self.is_int(v) {
            let qi = (q - relax).ceil();
            if qi > self.elo[v] + 0.5 {
                if qi > self.whi[v] + 0.5 {
                    debug!(
                        "presolve: var {v} integer lower {qi} > upper {}",
                        self.whi[v]
                    );
                    return Err(Error::Infeasible);
                }
                self.elo[v] = qi;
                self.wlo[v] = self.wlo[v].max(qi);
                self.stats.bounds_tightened += 1;
                self.mark_var(v);
                return Ok(true);
            }
        } else {
            let nq = q - relax;
            if nq > self.wlo[v] + IMPROVE_REL * nq.abs().max(1.0) {
                if nq > self.whi[v] {
                    if nq - self.whi[v] > self.bound_tol(v, true) {
                        debug!(
                            "presolve: var {v} implied lower {nq:e} > upper {:e}",
                            self.whi[v]
                        );
                        return Err(Error::Infeasible);
                    }
                    return Ok(false);
                }
                self.wlo[v] = nq;
                self.stats.bounds_tightened += 1;
                self.mark_var(v);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Whether an all-substituted row holds at its vars' fixed values, judged
    /// exactly as the boundary (`Solver::first_violated_row`) judges it once
    /// dropped, and as [`trivial_row_verdict`] judges an empty input row: the
    /// row as written, evaluated by [`row_violation`] against the full
    /// contract, with no further allowance for round-off. Nothing the engine
    /// does can move such a row, so the engine's budget has no part in it;
    /// held to the budget, a feasible model could be declared infeasible. A
    /// var the row keeps with a zero coefficient is not fixed and adds a zero
    /// term whatever its value.
    ///
    /// [`trivial_row_verdict`]: crate::solver::trivial_row_verdict
    fn empty_row_consistent(&self, row: &Row) -> bool {
        let (coeffs, op, rhs) = &self.constraints[row.orig as usize];
        let value = |v: usize| if self.fixed(v) { self.elo[v] } else { 0.0 };
        row_violation(coeffs, *op, *rhs, value, self.feas).is_none()
    }

    /// An all-substituted row: consistent (drop) or a proof of infeasibility.
    pub(crate) fn empty_row_check(&mut self, row: &mut Row) -> Result<(), Error> {
        if !self.empty_row_consistent(row) {
            debug!(
                "presolve: all-substituted row {} inconsistent: residual [{:e}, {:e}], folded magnitude {:e}",
                row.orig, row.lo, row.hi, row.fold_scale
            );
            return Err(Error::Infeasible);
        }
        row.alive = false;
        self.stats.rows_dropped += 1;
        Ok(())
    }

    /// Singleton row `a·x ∈ [lo, hi]`: fold it into the var's EMITTED bounds
    /// and drop it, where bounds describe the row exactly as the engine and
    /// the boundary would hold it. `other_coeff` is the largest |coefficient|
    /// of `x` in any other live row. Returns whether the row was consumed.
    fn singleton_row(&mut self, row: &mut Row, other_coeff: f64) -> Result<bool, Error> {
        let v = row.vars[0];
        let a = row.coeffs[0];
        // The row bounds that yield the var's lower and upper implied bound.
        let (blo, bhi) = if a > 0.0 {
            (row.lo, row.hi)
        } else {
            (row.hi, row.lo)
        };
        let (qlo, qhi) = (blo / a, bhi / a);
        // The row's magnitude where the given side is tight: its right-hand
        // side as written, the var's term there (`|a·q| = |b|`), and the fixed
        // terms folded out of it.
        let fold = row.fold_scale;
        let tight_mag = |b: f64| row.rhs_mag(b) + b.abs() + fold;

        if self.is_int(v) {
            // The bounds keep every integral point the incumbent guard
            // accepts (those meeting the row within its contract), widened by
            // the round-off of the folded right-hand side. The row is dropped
            // only where the contract less that round-off gives the same
            // bounds, i.e. no integer point lies within round-off of the
            // contract; otherwise the guard, with its own round-off, might
            // reject a point the bounds admit (an `InternalError` on an
            // exactly integral candidate), so the row stays.
            let range = |widen: f64| {
                let side_tol = |b: f64| {
                    let mag = tight_mag(b);
                    (row_tolerance(self.feas, mag) + widen * ROUNDOFF_FLOOR * mag).max(0.0)
                        / a.abs()
                };
                let mut new_lo = self.elo[v];
                let mut new_hi = self.ehi[v];
                if qlo.is_finite() {
                    new_lo = new_lo.max((qlo - side_tol(blo)).ceil());
                }
                if qhi.is_finite() {
                    new_hi = new_hi.min((qhi + side_tol(bhi)).floor());
                }
                (new_lo, new_hi)
            };
            let (new_lo, new_hi) = range(1.0);
            let unambiguous = range(-1.0) == (new_lo, new_hi);
            if new_lo > new_hi {
                debug!(
                    "presolve: singleton row {} leaves integer var {v} no value: [{new_lo}, {new_hi}]",
                    row.orig
                );
                return Err(Error::Infeasible);
            }
            self.emit_bounds(v, new_lo, new_hi);
            if !unambiguous {
                return Ok(false);
            }
            row.alive = false;
            self.stats.rows_dropped += 1;
            return Ok(true);
        }

        // Converting the row moves its round-off into the var, where the
        // var's other rows amplify it by their coefficients: they must stay
        // within budget, or only the simplex can balance the rows.
        let finite_mag = |b: f64| if b.is_finite() { tight_mag(b) } else { 0.0 };
        let round_off = ROUNDOFF_FLOOR * finite_mag(blo).max(finite_mag(bhi));
        if other_coeff * round_off > ROW_BUDGET_SHARE * self.feas * a.abs() {
            return Ok(false);
        }

        // A dropped row accepts `x` within its room `δ` (below) of its implied
        // bounds `qlo`, `qhi`; the engine holds a bound to within `btol`. The
        // row becomes the bounds `qlo`, `qhi` themselves, which keep the
        // vertices the row makes (a bound relaxed into the row's tolerance
        // would make a vertex there, with a better objective than the model's).
        // A side the var's bound already implies, with the tolerances, needs no
        // bound; a side whose `δ` is below `btol` cannot be a bound, since the
        // engine could leave it by more than the row allows, so the row stays.
        // An equality fixes its var only where it pins it within the var's own
        // tolerance and no other row of the var tells the pinned range apart;
        // otherwise a fixing would cut points the row accepts, or make rows
        // that agree only within their tolerances look inconsistent.
        let btol_lo = bound_tolerance(self.feas, self.elo[v].max(qlo));
        let btol_hi = bound_tolerance(self.feas, self.ehi[v].min(qhi));
        // The row's room around its implied bound as the engine holds it
        // (budget over `|a|`), and as the boundary checks it once dropped:
        // only the var's bound tolerance can move a dropped singleton row, so
        // it has its whole contract, less the round-off of evaluating it.
        let dlo = self.budget(finite_mag(blo)) / a.abs();
        let dhi = self.budget(finite_mag(bhi)) / a.abs();
        let dropped_room = |b: f64| {
            let mag = finite_mag(b);
            (row_tolerance(self.feas, mag) - ROUNDOFF_FLOOR * mag).max(0.0) / a.abs()
        };
        let (drop_lo, drop_hi) = (dropped_room(blo), dropped_room(bhi));
        let (elo, ehi) = (self.elo[v], self.ehi[v]);
        // The row's accepted set misses the var's, both taken with their
        // tolerances: a proof of infeasibility.
        if qlo - dlo > ehi + btol_hi || qhi + dhi < elo - btol_lo {
            debug!(
                "presolve: singleton row {} misses var {v}'s bounds [{elo:e}, {ehi:e}]",
                row.orig
            );
            return Err(Error::Infeasible);
        }
        if qlo == qhi {
            if dlo > btol_lo.min(btol_hi)
                || other_coeff * dlo > ROW_BUDGET_SHARE * self.feas
                || qlo < elo
                || qlo > ehi
            {
                return Ok(false);
            }
            self.fix(v, qlo);
            row.alive = false;
            self.stats.rows_dropped += 1;
            return Ok(true);
        }
        let (mut lo, mut hi) = (elo, ehi);
        if qlo.is_finite() && elo - btol_lo < qlo - drop_lo {
            if drop_lo < btol_lo {
                return Ok(false);
            }
            lo = qlo;
        }
        if qhi.is_finite() && ehi + btol_hi > qhi + drop_hi {
            if drop_hi < btol_hi {
                return Ok(false);
            }
            hi = qhi;
        }
        if lo > hi {
            // Within the tolerances but not within the bounds: only the
            // simplex, holding both, can find the sliver between them.
            return Ok(false);
        }
        if lo == hi {
            self.fix(v, lo);
        } else {
            self.emit_bounds(v, lo, hi);
        }
        row.alive = false;
        self.stats.rows_dropped += 1;
        Ok(true)
    }

    /// One wave of the primal worklist: process every row queued so far;
    /// rows whose inputs change during the wave land in the next wave.
    ///
    /// Rows left with at most one term go first, in queue order, then the
    /// others. A singleton row states its var's range exactly, while a
    /// multi-term row deduces from bounds; processed first, the singleton
    /// pins its var before a forcing row can fix it at another bound that
    /// the singleton allows only within its tolerance. A multi-term row that
    /// loses terms to the singletons' fixings is still processed in the same
    /// wave, as the singleton it may have become.
    pub(crate) fn primal_wave(&mut self, rows: &mut [Row]) -> Result<(), Error> {
        let wave = std::mem::take(&mut self.queue);
        let mut multi_term = Vec::with_capacity(wave.len());
        for &r in &wave {
            let r = r as usize;
            if !rows[r].alive {
                self.dirty[r] = false;
                continue;
            }
            self.fold_fixed(&mut rows[r]);
            if rows[r].vars.len() > 1 {
                // Still queued (dirty) for the second pass of this wave: a
                // change to its vars before then needs no requeue.
                multi_term.push(r);
                continue;
            }
            // Clear before processing: a change made while processing this
            // very row (its own var tightened) legitimately requeues it.
            self.dirty[r] = false;
            if rows[r].vars.is_empty() {
                self.empty_row_check(&mut rows[r])?;
            } else {
                let other = self.other_coeff(rows, rows[r].vars[0], r);
                self.singleton_row(&mut rows[r], other)?;
            }
        }
        for r in multi_term {
            self.dirty[r] = false;
            if !rows[r].alive {
                continue;
            }
            self.fold_fixed(&mut rows[r]);
            if rows[r].vars.is_empty() {
                self.empty_row_check(&mut rows[r])?;
                continue;
            }
            if rows[r].vars.len() == 1 {
                let other = self.other_coeff(rows, rows[r].vars[0], r);
                self.singleton_row(&mut rows[r], other)?;
                continue;
            }

            let row = &rows[r];
            let wact = self.activity(row, true);
            let (row_lo, row_hi) = (row.lo, row.hi);
            let fold = row.fold_scale;
            let hi_side = row_hi.is_finite() && wact.ninf_l == 0;
            let lo_side = row_lo.is_finite() && wact.ninf_u == 0;

            // Infeasibility: the working box, widened by the bound slack, holds
            // every point the engine may return (its bounds are tolerance-valid
            // implications), so a row its best corner cannot meet within the
            // budget at that corner is a proof. The corner decides: moving away
            // from it raises the activity faster than the tolerance.
            let budget_l = if hi_side {
                self.budget(wact.abs_l + row.rhs_mag(row_hi) + fold)
            } else {
                0.0
            };
            let budget_u = if lo_side {
                self.budget(wact.abs_u + row.rhs_mag(row_lo) + fold)
            } else {
                0.0
            };
            if hi_side && wact.l - wact.slack_l > row_hi + budget_l {
                debug!(
                    "presolve: row {} min activity {:e} > hi {row_hi:e}",
                    row.orig, wact.l
                );
                return Err(Error::Infeasible);
            }
            if lo_side && wact.u + wact.slack_u < row_lo - budget_u {
                debug!(
                    "presolve: row {} max activity {:e} < lo {row_lo:e}",
                    row.orig, wact.u
                );
                return Err(Error::Infeasible);
            }

            // Forcing: over the EMITTED bounds, which the reduced problem
            // carries, the corner meets the row bound within the round-off of
            // the sum, so every feasible point has each var at its extreme and
            // fixing it there is exact, if the room left each var is room
            // nothing resolves (`forcing_is_exact`). The corner may pass the
            // bound by the budget (the fixed point is then the best there is).
            // The working activity, which dominates, is a cheap precheck.
            let mut eact: Option<Activity> = None;
            if hi_side && wact.l >= row_hi - budget_l {
                let e = eact.get_or_insert_with(|| self.activity(row, false));
                if e.ninf_l == 0 {
                    let mag = e.abs_l + row.rhs_mag(row_hi) + fold;
                    let sliver = row_hi - e.l;
                    if sliver <= ROUNDOFF_FLOOR * mag
                        && -sliver <= self.budget(mag)
                        && self.forcing_is_exact(rows, r, sliver, true)
                        && self.terms_resolvable(row, sliver, mag)
                    {
                        let at: Vec<(usize, f64)> = row
                            .vars
                            .iter()
                            .zip(&row.coeffs)
                            .map(|(&v, &a)| (v, if a > 0.0 { self.elo[v] } else { self.ehi[v] }))
                            .collect();
                        for (v, val) in at {
                            self.fix(v, val);
                        }
                        rows[r].alive = false;
                        self.stats.rows_dropped += 1;
                        continue;
                    }
                }
            }
            if lo_side && wact.u <= row_lo + budget_u {
                let e = eact.get_or_insert_with(|| self.activity(row, false));
                if e.ninf_u == 0 {
                    let mag = e.abs_u + row.rhs_mag(row_lo) + fold;
                    let sliver = e.u - row_lo;
                    if sliver <= ROUNDOFF_FLOOR * mag
                        && -sliver <= self.budget(mag)
                        && self.forcing_is_exact(rows, r, sliver, false)
                        && self.terms_resolvable(row, sliver, mag)
                    {
                        let at: Vec<(usize, f64)> = row
                            .vars
                            .iter()
                            .zip(&row.coeffs)
                            .map(|(&v, &a)| (v, if a > 0.0 { self.ehi[v] } else { self.elo[v] }))
                            .collect();
                        for (v, val) in at {
                            self.fix(v, val);
                        }
                        rows[r].alive = false;
                        self.stats.rows_dropped += 1;
                        continue;
                    }
                }
            }

            // Redundancy: judged against the EMITTED bounds only, so a drop is
            // never justified by a deduction the drop itself would orphan, and
            // with the bound slack, so a point the engine may return still
            // meets the dropped row. Working activities dominate emitted ones
            // (L_e <= L_w, U_e >= U_w), so the working check is a precheck.
            let w_lo_red =
                !row_lo.is_finite() || (wact.ninf_l == 0 && wact.l - wact.slack_l >= row_lo);
            let w_hi_red =
                !row_hi.is_finite() || (wact.ninf_u == 0 && wact.u + wact.slack_u <= row_hi);
            if w_lo_red && w_hi_red {
                let e = eact.get_or_insert_with(|| self.activity(row, false));
                let magnitude = e.abs_l
                    + e.abs_u
                    + fold
                    + if row_lo.is_finite() {
                        row.rhs_mag(row_lo)
                    } else {
                        0.0
                    }
                    + if row_hi.is_finite() {
                        row.rhs_mag(row_hi)
                    } else {
                        0.0
                    };
                let red = REDUNDANT_MARGIN * magnitude.max(1.0);
                let lo_red =
                    !row_lo.is_finite() || (e.ninf_l == 0 && e.l - e.slack_l >= row_lo + red);
                let hi_red =
                    !row_hi.is_finite() || (e.ninf_u == 0 && e.u + e.slack_u <= row_hi - red);
                if lo_red && hi_red {
                    rows[r].alive = false;
                    self.stats.rows_dropped += 1;
                    continue;
                }
            }

            // Per-term implied bounds from working activities, each relaxed by
            // the row's allowance at the corner where the bound is attained
            // (this term at the bound, the rest at its extreme), by the
            // round-off of computing the rest by cancellation against this
            // term's own range, and by the bound slack, so that it holds for
            // every point the engine may return. Bounds tightened earlier in
            // this loop are not folded into `wact`: the stale activity is
            // looser, hence still valid.
            for i in 0..row.vars.len() {
                let (v, a) = (row.vars[i], row.coeffs[i]);
                let (cmin, cmax) = term_range(a, self.wlo[v], self.whi[v]);
                let rest_l = if cmin.is_finite() {
                    (wact.ninf_l == 0).then_some(wact.l - cmin)
                } else {
                    (wact.ninf_l == 1).then_some(wact.l)
                };
                let rest_u = if cmax.is_finite() {
                    (wact.ninf_u == 0).then_some(wact.u - cmax)
                } else {
                    (wact.ninf_u == 1).then_some(wact.u)
                };
                if row_hi.is_finite() {
                    if let Some(rl) = rest_l {
                        let q = (row_hi - rl) / a;
                        let rest_abs = wact.abs_l - if cmin.is_finite() { cmin.abs() } else { 0.0 };
                        let mag =
                            row.rhs_mag(row_hi) + rest_abs.max(0.0) + (row_hi - rl).abs() + fold;
                        let relax =
                            (self.allowance(v, mag) + ROUNDOFF_FLOOR * wact.abs_l + wact.slack_l)
                                / a.abs();
                        if a > 0.0 {
                            self.tighten_upper(v, q, relax)?;
                        } else {
                            self.tighten_lower(v, q, relax)?;
                        }
                    }
                }
                if row_lo.is_finite() {
                    if let Some(ru) = rest_u {
                        let q = (row_lo - ru) / a;
                        let rest_abs = wact.abs_u - if cmax.is_finite() { cmax.abs() } else { 0.0 };
                        let mag =
                            row.rhs_mag(row_lo) + rest_abs.max(0.0) + (row_lo - ru).abs() + fold;
                        let relax =
                            (self.allowance(v, mag) + ROUNDOFF_FLOOR * wact.abs_u + wact.slack_u)
                                / a.abs();
                        if a > 0.0 {
                            self.tighten_lower(v, q, relax)?;
                        } else {
                            self.tighten_upper(v, q, relax)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Savelsbergh coefficient tightening on one-sided rows for strict
    /// binaries (Mip mode only): rewrite the (coefficient, bound) pair so the
    /// vacuous branch of the implication becomes exactly bound-tight. The LP
    /// relaxation tightens and integer points are kept exactly: the binding
    /// branch is reproduced to the last bit, and the vacuous one is relaxed by
    /// the activity's round-off and the bound slack, so every point the engine
    /// may return for the original row satisfies the rewritten one.
    pub(crate) fn tighten_binary_coeffs(&mut self, rows: &mut [Row]) -> bool {
        let mut changed = false;
        for (r, row) in rows.iter_mut().enumerate() {
            // Only rows whose inputs changed since this pass last saw them.
            if !row.alive || !self.coeff_dirty[r] {
                continue;
            }
            self.coeff_dirty[r] = false;
            let le_form = row.hi.is_finite() && !row.lo.is_finite();
            let ge_form = row.lo.is_finite() && !row.hi.is_finite();
            if !(le_form || ge_form) {
                continue; // Eq rows bind on both sides: nothing is vacuous
            }
            let mut act = self.activity(row, true);
            // The relaxation of the vacuous branch. `act.abs_*` are left
            // untouched by the incremental updates below, which only makes
            // this more conservative.
            let nz = if le_form { act.slack_u } else { act.slack_l }
                + ROUNDOFF_FLOOR
                    * if le_form {
                        act.abs_u + row.rhs_mag(row.hi) + row.fold_scale
                    } else {
                        act.abs_l + row.rhs_mag(row.lo) + row.fold_scale
                    };
            // Count eligible terms first: the excess (activity bound minus
            // row bound) is invariant under applications, so eligibility is
            // decided by this precount exactly. Skip cascade-prone rows.
            let mut eligible = 0usize;
            for i in 0..row.vars.len() {
                let v = row.vars[i];
                let a = row.coeffs[i];
                if !(is_int_domain(&self.domains[v]) && self.elo[v] == 0.0 && self.ehi[v] == 1.0)
                    || a == 0.0
                {
                    continue;
                }
                if le_form && act.ninf_u == 0 {
                    let b = row.hi;
                    let rest_u = act.u - a.max(0.0);
                    if a > 0.0 {
                        let delta = b - (rest_u + nz);
                        if delta > COEFF_MIN_IMPROVE * a && a - delta > 0.0 {
                            eligible += 1;
                        }
                    } else {
                        let na = b - (rest_u + nz);
                        if na - a > COEFF_MIN_IMPROVE * -a && na < 0.0 {
                            eligible += 1;
                        }
                    }
                }
                if ge_form && act.ninf_l == 0 {
                    let b = row.lo;
                    let rest_l = act.l - a.min(0.0);
                    if a < 0.0 {
                        let delta = rest_l - nz - b;
                        if delta > COEFF_MIN_IMPROVE * -a && a + delta < 0.0 {
                            eligible += 1;
                        }
                    } else {
                        let na = b - (rest_l - nz);
                        if a - na > COEFF_MIN_IMPROVE * a && na > 0.0 {
                            eligible += 1;
                        }
                    }
                }
            }
            let capped = row.vars.len() > COEFF_TIGHTEN_SHORT_ROW;
            if eligible == 0 || (capped && eligible > COEFF_TIGHTEN_ROW_CAP) {
                continue;
            }
            for i in 0..row.vars.len() {
                let v = row.vars[i];
                let a = row.coeffs[i];
                // Strict binary in the EMITTED problem (what the solver sees).
                let binary =
                    is_int_domain(&self.domains[v]) && self.elo[v] == 0.0 && self.ehi[v] == 1.0;
                if !binary || a == 0.0 {
                    continue;
                }
                let mut applied = false;
                if le_form && act.ninf_u == 0 {
                    let b = row.hi;
                    let rest_u = act.u - a.max(0.0); // binary term range is [min(a,0), max(a,0)]
                    if a > 0.0 {
                        // x=0 branch vacuous (rest ≤ b): shrink a and b so it
                        // becomes exactly bound-tight; x=1 stays "rest ≤ b - a".
                        let bp = rest_u + nz;
                        let delta = b - bp;
                        let na = a - delta;
                        if delta > COEFF_MIN_IMPROVE * a && na > 0.0 {
                            row.coeffs[i] = na;
                            row.hi = bp;
                            act.u -= delta; // max contribution a -> na
                            applied = true;
                        }
                    } else {
                        // x=1 branch vacuous (rest ≤ b - a): shrink |a|, keep b.
                        let na = b - (rest_u + nz);
                        if na - a > COEFF_MIN_IMPROVE * -a && na < 0.0 {
                            act.l += na - a; // min contribution a -> na
                            row.coeffs[i] = na;
                            applied = true;
                        }
                    }
                }
                if ge_form && act.ninf_l == 0 {
                    let b = row.lo;
                    let rest_l = act.l - a.min(0.0);
                    if a < 0.0 {
                        // x=0 branch vacuous (rest ≥ b): raise a and lo;
                        // x=1 stays "rest ≥ b - a".
                        let bp = rest_l - nz;
                        let delta = bp - b; // ≥ 0 when applicable
                        let na = a + delta;
                        if delta > COEFF_MIN_IMPROVE * -a && na < 0.0 {
                            row.coeffs[i] = na;
                            row.lo = bp;
                            act.l += delta; // min contribution a -> na
                            applied = true;
                        }
                    } else {
                        // x=1 branch vacuous (rest ≥ b - a): shrink a, keep lo.
                        let na = b - (rest_l - nz);
                        if a - na > COEFF_MIN_IMPROVE * a && na > 0.0 {
                            act.u -= a - na; // max contribution a -> na
                            row.coeffs[i] = na;
                            applied = true;
                        }
                    }
                }
                if applied {
                    self.stats.coeffs_tightened += 1;
                    changed = true;
                    row.coeffs_touched = true;
                    // The row's data changed: reprocess it in the primal
                    // worklist (new redundancies/bounds may follow).
                    if !self.dirty[r] {
                        self.dirty[r] = true;
                        self.queue.push(r as u32);
                    }
                }
            }
        }
        changed
    }

    /// Dual fixing: a var whose objective coefficient and row signs make one
    /// direction of movement never helpful is fixed at the opposite (finite)
    /// bound. It keeps an optimum, not the feasible set, so it runs in Mip
    /// mode with `allow_dual` only. An infinite target bound means "possibly
    /// unbounded", a verdict for the simplex, so such vars are skipped.
    ///
    /// The target bound is always an EMITTED bound: a working lower bound on
    /// `v` can only be derived from a row that also blocks decreasing `v` (Ge
    /// with a>0, Le with a<0, or Eq), so `!bad_dec` implies `wlo == elo`
    /// (mirrored for the upper side).
    pub(crate) fn dual_fix(&mut self, rows: &[Row], obj: &[f64]) -> bool {
        let n = self.wlo.len();
        let mut bad_dec = vec![false; n];
        let mut bad_inc = vec![false; n];
        for row in rows.iter().filter(|r| r.alive) {
            let le_side = row.hi.is_finite();
            let ge_side = row.lo.is_finite();
            for (&v, &a) in row.vars.iter().zip(&row.coeffs) {
                if a > 0.0 {
                    // decreasing x lowers the activity: hurts a >= side
                    if ge_side {
                        bad_dec[v] = true;
                    }
                    if le_side {
                        bad_inc[v] = true;
                    }
                } else {
                    if le_side {
                        bad_dec[v] = true;
                    }
                    if ge_side {
                        bad_inc[v] = true;
                    }
                }
            }
        }
        let mut changed = false;
        for v in 0..n {
            if self.fixed(v) {
                continue;
            }
            if obj[v] >= 0.0 && !bad_dec[v] && self.wlo[v].is_finite() {
                debug_assert_eq!(self.wlo[v], self.elo[v]);
                let val = self.wlo[v];
                self.fix(v, val);
                changed = true;
            } else if obj[v] <= 0.0 && !bad_inc[v] && self.whi[v].is_finite() {
                debug_assert_eq!(self.whi[v], self.ehi[v]);
                let val = self.whi[v];
                self.fix(v, val);
                changed = true;
            }
        }
        changed
    }
}
