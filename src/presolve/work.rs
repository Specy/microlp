//! [`Work`]: the bound state the reduction passes mutate, and the passes
//! themselves.
//!
//! Rows live outside this struct so a hot pass can hold a row borrow and
//! mutate bounds at the same time; that is what keeps the per-row inner loops
//! allocation-free. See the module docs in [`super`] for the working-vs-emitted
//! bound split and the tolerance model every decision here is made against.

use super::model::{
    bound_magnitude, is_int_domain, term_range, Activity, Mode, PresolveStats, Row,
};
use super::params::*;
use crate::solver::{bound_tolerance, row_tolerance, ROUNDOFF_FLOOR, ROW_BUDGET_SHARE};
use crate::{Error, VarDomain};

/// Bound state and reduction context. Rows live OUTSIDE this struct so the
/// hot passes can hold a row borrow and mutate bounds at the same time —
/// this is what keeps the per-row inner loops allocation-free.
pub(crate) struct Work<'a> {
    /// Working bounds: every valid deduction, used to deduce further.
    pub(crate) wlo: Vec<f64>,
    pub(crate) whi: Vec<f64>,
    /// Emitted bounds: what the reduced problem carries (see module docs).
    /// Invariant: `[wlo, whi] ⊆ [elo, ehi]` per var.
    pub(crate) elo: Vec<f64>,
    pub(crate) ehi: Vec<f64>,
    pub(crate) domains: &'a [VarDomain],
    pub(crate) mode: Mode,
    /// The user's absolute feasibility tolerance (`Tolerances::feasibility`).
    pub(crate) feas: f64,
    pub(crate) stats: PresolveStats,
    /// var -> rows adjacency in flat CSR form, built once up front. Rows
    /// only ever LOSE terms, so this stays a (cheap-to-check) superset.
    pub(crate) adj_start: Vec<u32>,
    pub(crate) adj_rows: Vec<u32>,
    /// Worklist: a row is queued when one of its inputs (a variable bound,
    /// its own coefficients/rhs) changed since it was last processed. This
    /// is what makes the fixpoint O(changed work), not O(passes * nnz).
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

    /// Tolerance within which the engine holds var `v` to its emitted
    /// bounds, in user units (`Solver::first_violated_bound`); zero for an
    /// integer var, whose adopted values are exactly integral.
    fn bound_tol(&self, v: usize) -> f64 {
        if self.is_int(v) {
            0.0
        } else {
            bound_tolerance(self.feas, self.elo[v], self.ehi[v])
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
            slack: 0.0,
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
            act.slack += a.abs() * self.bound_tol(v);
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

    /// Whether fixing every var of row `r` at its extreme, when the row's
    /// corner falls short of its bound by `sliver` (row units), stays within
    /// the budget of every other row: each var gives up `sliver / |a|` of
    /// range, which another row amplifies by its own coefficient.
    fn sliver_within_budget(&self, rows: &[Row], r: usize, sliver: f64) -> bool {
        if sliver <= 0.0 {
            return true;
        }
        let row = &rows[r];
        row.vars.iter().zip(&row.coeffs).all(|(&v, &a)| {
            self.other_coeff(rows, v, r) * sliver <= ROW_BUDGET_SHARE * self.feas * a.abs()
        })
    }

    /// Whether every term of a forcing candidate can be told apart from the
    /// row's round-off: a var whose whole term range `|a|·(hi − lo)` lies
    /// below the sliver plus the round-off of the sum is invisible to the
    /// row — the corner meets the bound with the var anywhere in its range —
    /// so the row forces nothing about it and must not fix it. (A `1e-7`
    /// coefficient on a row of magnitude `2^28`, whose round-off is `5e-6`,
    /// pinned a var with a range of `0.25` at its extreme and made a
    /// feasible model infeasible.)
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
    /// the engine may return once relaxed outward by `relax` (the row's
    /// budget plus slack over the coefficient, in var units). Integer bounds
    /// round and persist (emitted); continuous bounds stay working-only. A
    /// crossing within the var's bound tolerance is "no reduction" (the
    /// solver handles slivers); beyond it, a proof of infeasibility.
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
                    if self.wlo[v] - nq > self.bound_tol(v) {
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
                    if nq - self.whi[v] > self.bound_tol(v) {
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

    /// Whether an all-substituted row holds at the substituted values within
    /// the engine's budget: what remains of `lo`/`hi` after the shifts is
    /// the residual, judged at the magnitude of what was folded.
    fn empty_row_consistent(&self, row: &Row) -> bool {
        let side = if row.lo.is_finite() {
            row.lo.abs()
        } else {
            row.hi.abs()
        };
        let budget = self.budget(side + row.fold_scale);
        row.lo <= budget && row.hi >= -budget
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

    /// Singleton row `a·x ∈ [lo, hi]`: fold it into the variable's EMITTED
    /// bounds and drop it, where a bound describes the row at least as well
    /// as the engine would hold the row itself (module docs). `other_coeff`
    /// is the largest |coefficient| of `x` in any other live row. Returns
    /// whether the row was consumed.
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
        // Budget of the row where the given side is tight: `|b| + |a·x| = 2|b|`.
        let fold = row.fold_scale;
        let side_budget = |w: &Self, b: f64| w.budget(2.0 * b.abs() + fold);

        if self.is_int(v) {
            // Exact for integers: the integral points the engine could
            // adopt are those meeting the row within its budget.
            let mut new_lo = self.elo[v];
            let mut new_hi = self.ehi[v];
            if qlo.is_finite() {
                new_lo = new_lo.max((qlo - side_budget(self, blo) / a.abs()).ceil());
            }
            if qhi.is_finite() {
                new_hi = new_hi.min((qhi + side_budget(self, bhi) / a.abs()).floor());
            }
            if new_lo > new_hi {
                debug!(
                    "presolve: singleton row {} leaves integer var {v} no value: [{new_lo}, {new_hi}]",
                    row.orig
                );
                return Err(Error::Infeasible);
            }
            self.emit_bounds(v, new_lo, new_hi);
            row.alive = false;
            self.stats.rows_dropped += 1;
            return Ok(true);
        }

        // Converting the row moves its round-off into the var, where the
        // var's other rows amplify it by their coefficients: they must stay
        // within budget, or only the simplex can balance the rows.
        let round_off = ROUNDOFF_FLOOR * (2.0 * bound_magnitude(blo, bhi) + fold);
        if other_coeff * round_off > ROW_BUDGET_SHARE * self.feas * a.abs() {
            return Ok(false);
        }
        let new_lo = self.elo[v].max(qlo);
        let new_hi = self.ehi[v].min(qhi);
        if new_lo > new_hi {
            // The implied interval misses the var's bounds: at the nearest
            // bound the row is off by |a|·gap. Within budget, that bound is
            // the answer; beyond budget plus the engine's bound tolerance,
            // no acceptable point exists; in between, the simplex decides.
            let (at, gap, budget) = if qlo > self.ehi[v] {
                (self.ehi[v], qlo - self.ehi[v], side_budget(self, blo))
            } else {
                (self.elo[v], self.elo[v] - qhi, side_budget(self, bhi))
            };
            if a.abs() * gap <= budget {
                self.fix(v, at);
                row.alive = false;
                self.stats.rows_dropped += 1;
                return Ok(true);
            }
            if a.abs() * (gap - self.bound_tol(v)) > budget {
                debug!(
                    "presolve: singleton row {} crosses var {v}'s bounds by {gap:e}",
                    row.orig
                );
                return Err(Error::Infeasible);
            }
            return Ok(false);
        }
        if new_lo == new_hi {
            // A fixed var is exact in the engine (never basic, never moved).
            self.fix(v, new_lo);
            row.alive = false;
            self.stats.rows_dropped += 1;
            return Ok(true);
        }
        // A genuine bound: the engine may leave it by the bound tolerance,
        // which must keep the row within budget on every side the row
        // provides.
        let btol = bound_tolerance(self.feas, new_lo, new_hi);
        if (new_lo > self.elo[v] && a.abs() * btol > side_budget(self, blo))
            || (new_hi < self.ehi[v] && a.abs() * btol > side_budget(self, bhi))
        {
            return Ok(false);
        }
        self.emit_bounds(v, new_lo, new_hi);
        row.alive = false;
        self.stats.rows_dropped += 1;
        Ok(true)
    }

    /// One wave of the primal worklist: process every row queued so far;
    /// rows whose inputs change during the wave land in the next wave.
    pub(crate) fn primal_wave(&mut self, rows: &mut [Row]) -> Result<(), Error> {
        let wave = std::mem::take(&mut self.queue);
        for &r in &wave {
            let r = r as usize;
            // Clear before processing: a change made while processing this
            // very row (its own var tightened) legitimately requeues it.
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

            // Infeasibility: the working box holds every point the engine
            // may return (its bounds are tolerance-valid implications),
            // widened by the bound slack; a row that its best corner cannot
            // meet within the budget at that corner is a proof. Moving away
            // from the corner raises the activity faster than it raises the
            // tolerance, so the corner is the decisive point.
            let budget_l = if hi_side {
                self.budget(wact.abs_l + row_hi.abs() + fold)
            } else {
                0.0
            };
            let budget_u = if lo_side {
                self.budget(wact.abs_u + row_lo.abs() + fold)
            } else {
                0.0
            };
            if hi_side && wact.l - wact.slack > row_hi + budget_l {
                debug!(
                    "presolve: row {} min activity {:e} > hi {row_hi:e}",
                    row.orig, wact.l
                );
                return Err(Error::Infeasible);
            }
            if lo_side && wact.u + wact.slack < row_lo - budget_u {
                debug!(
                    "presolve: row {} max activity {:e} < lo {row_lo:e}",
                    row.orig, wact.u
                );
                return Err(Error::Infeasible);
            }

            // Forcing: judged over the EMITTED bounds, the exact values the
            // reduced problem carries, equal to the row bound within the
            // round-off of the sum. Every feasible point then has each var
            // at its extreme, and fixing it there is exact. The corner may
            // exceed the bound by the budget (the fixed point is then the
            // best point there is) and fall short of it only by a sliver
            // every other row of every var tolerates. The working activity,
            // which dominates the emitted one, is a cheap precheck.
            let mut eact: Option<Activity> = None;
            if hi_side && wact.l >= row_hi - budget_l {
                let e = eact.get_or_insert_with(|| self.activity(row, false));
                if e.ninf_l == 0 {
                    let mag = e.abs_l + row_hi.abs() + fold;
                    let sliver = row_hi - e.l;
                    if sliver <= ROUNDOFF_FLOOR * mag
                        && -sliver <= self.budget(mag)
                        && self.sliver_within_budget(rows, r, sliver)
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
                    let mag = e.abs_u + row_lo.abs() + fold;
                    let sliver = e.u - row_lo;
                    if sliver <= ROUNDOFF_FLOOR * mag
                        && -sliver <= self.budget(mag)
                        && self.sliver_within_budget(rows, r, sliver)
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

            // Redundancy: judged against EMITTED bounds only — material that
            // provably remains in the reduced problem — so dropping is never
            // justified by a deduction that the drop itself would orphan,
            // and against the bound slack, so that a point the engine may
            // return still meets the dropped row. Cheap precheck first:
            // working activities dominate emitted ones (L_e <= L_w,
            // U_e >= U_w), so if the working side already fails there is no
            // need to compute the emitted activity at all.
            let w_lo_red =
                !row_lo.is_finite() || (wact.ninf_l == 0 && wact.l - wact.slack >= row_lo);
            let w_hi_red =
                !row_hi.is_finite() || (wact.ninf_u == 0 && wact.u + wact.slack <= row_hi);
            if w_lo_red && w_hi_red {
                let e = eact.get_or_insert_with(|| self.activity(row, false));
                let magnitude = e.abs_l
                    + e.abs_u
                    + fold
                    + if row_lo.is_finite() {
                        row_lo.abs()
                    } else {
                        0.0
                    }
                    + if row_hi.is_finite() {
                        row_hi.abs()
                    } else {
                        0.0
                    };
                let red = REDUNDANT_MARGIN * magnitude.max(1.0);
                let lo_red =
                    !row_lo.is_finite() || (e.ninf_l == 0 && e.l - e.slack >= row_lo + red);
                let hi_red =
                    !row_hi.is_finite() || (e.ninf_u == 0 && e.u + e.slack <= row_hi - red);
                if lo_red && hi_red {
                    rows[r].alive = false;
                    self.stats.rows_dropped += 1;
                    continue;
                }
            }

            // Per-term implied bounds from working activities, each relaxed
            // by the row's budget at the corner where the bound is attained
            // (this term at the bound, the rest at its extreme) plus the
            // bound slack, so that it holds for every point the engine may
            // return. Bounds tightened earlier in this loop are not folded
            // into `wact`: the stale activity is looser, hence still valid.
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
                        let mag = row_hi.abs() + rest_abs.max(0.0) + (row_hi - rl).abs() + fold;
                        let relax = (self.budget(mag) + wact.slack) / a.abs();
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
                        let mag = row_lo.abs() + rest_abs.max(0.0) + (row_lo - ru).abs() + fold;
                        let relax = (self.budget(mag) + wact.slack) / a.abs();
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

    /// Savelsbergh coefficient improvement on one-sided rows for strict
    /// binaries: rewrite the (coefficient, bound) pair so the vacuous branch
    /// of the implication becomes exactly bound-tight. Integer points are
    /// preserved exactly (the binding branch's content is reproduced to the
    /// last bit and the vacuous branch is relaxed outward by the round-off
    /// of the activity and the bound slack, so every point the engine may
    /// return for the original row satisfies the rewritten one); the LP
    /// relaxation tightens. Mip mode only.
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
            let nz = act.slack
                + ROUNDOFF_FLOOR
                    * if le_form {
                        act.abs_u + row.hi.abs() + row.fold_scale
                    } else {
                        act.abs_l + row.lo.abs() + row.fold_scale
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

    /// Dual fixing: a variable whose objective coefficient and constraint
    /// signs make one direction of movement never-helpful is pinned to the
    /// opposite (finite) bound. Optimum-preserving, not feasible-set-
    /// preserving — Mip mode with `allow_dual` only. An infinite target
    /// bound means "possibly unbounded"; that verdict belongs to the
    /// simplex, so such vars are skipped.
    ///
    /// The target bound is always an EMITTED bound: a working lower bound on
    /// `v` can only ever be derived from a row that also blocks decreasing
    /// `v` (Ge with a>0, Le with a<0, or Eq), so `!bad_dec` implies `wlo ==
    /// elo` (mirrored for the upper side).
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
