//! Presolve: problem reductions applied before the simplex / branch & bound
//! starts. Design doc: docs/superpowers/specs/2026-07-11-presolve-design.md.
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
//!   detection, forcing rows, cascade tightening, coefficient tightening.
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
//! # Soundness by mode
//!
//! * [`Mode::Lp`] applies only feasible-set-EXACT reductions: the presolved
//!   problem has the identical set of feasible points as the input. Facts
//!   baked into the live solver therefore stay valid under every later edit
//!   (`Solution::add_constraint` / `fix_var` / Gomory cuts only shrink the
//!   region; implied facts remain implied).
//! * [`Mode::Mip`] adds reductions that preserve the INTEGER feasible set
//!   (integer bound rounding, binary coefficient tightening) or merely some
//!   optimum (dual fixing). Sound because MILP post-solve edits re-solve from
//!   the untouched `MipState::base` and incumbents are validated against the
//!   transformed rows, which agree with the original rows on integer points
//!   within bounds. Dual fixing is additionally gated on `allow_dual`: it is
//!   disabled when a warm-start hint accompanies the solve, so a feasible
//!   hint is never excluded by an optimality-only argument.

use crate::{ComparisonOp, CsVec, Error, VarDomain};

/// Which reduction families are sound for this solve path (see module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Lp,
    Mip,
}

/// Absolute-per-unit-scale tolerance for declaring a row infeasible or an
/// implied bound crossed. Deliberately matches the default
/// `Tolerances::feasibility` (see `check_constraints` for the absolute-vs-
/// relative rationale): a false "infeasible" is a wrong answer, so the test
/// is generous — real detections exceed it by orders of magnitude.
const FEAS_TOL: f64 = 1e-7;

/// Minimum relative improvement for a working continuous bound tightening,
/// and the amount the computed bound is relaxed OUTWARD before being stored.
/// The relaxation makes the stored bound a weakening of the mathematically
/// implied one, so recording it stays feasible-set-exact under rounding
/// noise; using the same value for both keeps passes idempotent (a
/// re-derived bound never re-applies).
const IMPROVE_REL: f64 = 1e-9;

/// Margin required to drop a row as redundant. Keeping a truly redundant row
/// costs nothing but speed, so this errs toward keeping.
const REDUNDANT_MARGIN: f64 = 1e-9;

/// Rounding-noise model: activity sums are trusted to `NOISE_REL * scale`
/// where `scale` is the sum of |finite contributions| + |row bounds|. This is
/// ~4 orders of magnitude above the worst-case f64 accumulation error for a
/// 100-term row and calibrates every guard that must not cut feasible points
/// (forcing rows, implied-bound relaxation, coefficient tightening). Big-M
/// rows are exactly where the amplified noise matters: 1e9-scale terms make
/// `NOISE_REL * scale` ≈ 1e-3, which correctly disables aggressive integer
/// rounding there instead of cutting a true optimum.
const NOISE_REL: f64 = 1e-12;

/// Minimum RELATIVE improvement for a binary coefficient tightening: the
/// coefficient must shrink by at least this fraction of its own magnitude.
const COEFF_MIN_IMPROVE: f64 = 0.1;

/// A LONG row is coefficient-tightened only when AT MOST this many of its
/// terms are eligible. The reduction's excess `U − b` is invariant under
/// each application, so on a loose dense binary row EVERY coefficient
/// cascades into eligibility and the row collapses toward a cardinality
/// constraint — a "tighter" relaxation that is massively degenerate in
/// practice (BIP_easy: 2 900 rows × ~290 terms, 367 409 rewrites, seconds
/// slower). The reduction that pays on long rows is the fixed-charge /
/// big-M signature: ONE (occasionally two) oversized coefficient per row
/// (`x − M·y ≤ 0`), which this cap isolates.
const COEFF_TIGHTEN_ROW_CAP: usize = 2;

/// Rows with at most this many terms are exempt from
/// [`COEFF_TIGHTEN_ROW_CAP`]: a handful of rewritten coefficients cannot
/// manufacture the degeneracy mass that dense rows can, and on small binary
/// MILPs (lseu-class knapsacks) the full row rewrite measurably shrinks the
/// search tree.
const COEFF_TIGHTEN_SHORT_ROW: usize = 16;

/// Primal fixpoint pass cap (per round) and outer primal+dual round cap.
const MAX_PASSES: usize = 10;
const MAX_ROUNDS: usize = 3;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PresolveStats {
    pub rows_dropped: usize,
    pub bounds_tightened: usize,
    pub vars_fixed: usize,
    pub coeffs_tightened: usize,
    pub passes: usize,
}

#[derive(Debug)]
pub(crate) struct Presolved {
    pub var_mins: Vec<f64>,
    pub var_maxs: Vec<f64>,
    pub constraints: Vec<(CsVec, ComparisonOp, f64)>,
    pub stats: PresolveStats,
}

/// One row in the uniform two-sided working form `lo <= a·x <= hi` (exactly
/// one side finite for Le/Ge, both equal for Eq — MPS ranged rows arrive as
/// two one-sided rows, so this is lossless and no transformation below ever
/// creates a genuinely two-sided row).
struct Row {
    vars: Vec<usize>,
    coeffs: Vec<f64>,
    lo: f64,
    hi: f64,
    /// Magnitudes folded into `lo`/`hi` by fixed-var substitution plus the
    /// original rhs magnitude: the scale that calibrates the "is this
    /// all-substituted row consistent" tolerance. Rows the USER wrote as
    /// empty are judged with the solver's exact semantics instead.
    fold_scale: f64,
    alive: bool,
    /// Index of this row in the input `constraints` slice.
    orig: u32,
    /// Whether terms/bounds were rewritten (fold, coefficient tightening).
    /// Untouched surviving rows are emitted as clones of the input row —
    /// byte-identical and cheaper than rebuilding a CsVec.
    touched: bool,
}

/// Row activity bounds computed from a bound set, with the standard
/// infinite-contribution counters (a bound is derivable for a term exactly
/// when the rest of the row has no infinite contribution on that side).
struct Activity {
    l: f64,
    u: f64,
    ninf_l: u32,
    ninf_u: u32,
    /// Sum of |finite contributions| + |finite row bounds|: the noise scale.
    abs: f64,
}

fn is_int_domain(d: &VarDomain) -> bool {
    matches!(d, VarDomain::Integer | VarDomain::Boolean)
}

/// Contribution range of one term given the var's bounds.
fn term_range(a: f64, lo: f64, hi: f64) -> (f64, f64) {
    if a > 0.0 {
        (a * lo, a * hi)
    } else {
        (a * hi, a * lo)
    }
}

fn activity(row: &Row, lo: &[f64], hi: &[f64]) -> Activity {
    let mut act = Activity {
        l: 0.0,
        u: 0.0,
        ninf_l: 0,
        ninf_u: 0,
        abs: 0.0,
    };
    for (&v, &a) in row.vars.iter().zip(&row.coeffs) {
        let (cmin, cmax) = term_range(a, lo[v], hi[v]);
        if cmin.is_finite() {
            act.l += cmin;
            act.abs += cmin.abs();
        } else {
            act.ninf_l += 1;
        }
        if cmax.is_finite() {
            act.u += cmax;
            act.abs += cmax.abs();
        } else {
            act.ninf_u += 1;
        }
    }
    if row.lo.is_finite() {
        act.abs += row.lo.abs();
    }
    if row.hi.is_finite() {
        act.abs += row.hi.abs();
    }
    act
}

/// Bound state and reduction context. Rows live OUTSIDE this struct so the
/// hot passes can hold a row borrow and mutate bounds at the same time —
/// this is what keeps the per-row inner loops allocation-free.
struct Work<'a> {
    /// Working bounds: every valid deduction, used to deduce further.
    wlo: Vec<f64>,
    whi: Vec<f64>,
    /// Emitted bounds: what the reduced problem carries (see module docs).
    /// Invariant: `[wlo, whi] ⊆ [elo, ehi]` per var.
    elo: Vec<f64>,
    ehi: Vec<f64>,
    domains: &'a [VarDomain],
    mode: Mode,
    int_tol: f64,
    stats: PresolveStats,
    /// var -> rows adjacency in flat CSR form, built once up front. Rows
    /// only ever LOSE terms, so this stays a (cheap-to-check) superset.
    adj_start: Vec<u32>,
    adj_rows: Vec<u32>,
    /// Worklist: a row is queued when one of its inputs (a variable bound,
    /// its own coefficients/rhs) changed since it was last processed. This
    /// is what makes the fixpoint O(changed work), not O(passes * nnz).
    dirty: Vec<bool>,
    queue: Vec<u32>,
    /// Rows whose inputs changed since the coefficient pass last saw them.
    coeff_dirty: Vec<bool>,
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

    /// Substitute emitted-fixed vars out of `row` (rhs shift) and drop exact
    /// zero coefficients.
    fn fold_fixed(&self, row: &mut Row) {
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
                }
                row.vars.swap_remove(i);
                row.coeffs.swap_remove(i);
                row.touched = true;
            } else {
                i += 1;
            }
        }
    }

    /// Record the implied upper bound `q` for var `v`; `amp_noise` is the row
    /// noise amplified by `1/|a|`. Integer bounds round and persist (emitted);
    /// continuous bounds stay working-only. A within-tolerance crossing is
    /// treated as "no reduction" (the solver handles slivers); a beyond-
    /// tolerance crossing is a proof of infeasibility.
    fn tighten_upper(&mut self, v: usize, q: f64, amp_noise: f64) -> Result<bool, Error> {
        if !q.is_finite() {
            return Ok(false);
        }
        if self.is_int(v) {
            let slack = self.int_tol.max(amp_noise);
            if slack > 0.1 {
                return Ok(false); // arithmetic too noisy to round integers safely
            }
            let qi = (q + slack).floor();
            if qi < self.ehi[v] - 0.5 {
                if qi < self.wlo[v] - 0.5 {
                    // Integral bounds crossing: an integer-empty range.
                    return Err(Error::Infeasible);
                }
                self.ehi[v] = qi;
                self.whi[v] = self.whi[v].min(qi);
                self.stats.bounds_tightened += 1;
                self.mark_var(v);
                return Ok(true);
            }
        } else {
            let relax = IMPROVE_REL * q.abs().max(1.0) + amp_noise;
            let nq = q + relax;
            if nq < self.whi[v] {
                if nq < self.wlo[v] {
                    let scale = nq.abs().max(self.wlo[v].abs()).max(1.0);
                    if self.wlo[v] - nq > FEAS_TOL * scale {
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
    fn tighten_lower(&mut self, v: usize, q: f64, amp_noise: f64) -> Result<bool, Error> {
        if !q.is_finite() {
            return Ok(false);
        }
        if self.is_int(v) {
            let slack = self.int_tol.max(amp_noise);
            if slack > 0.1 {
                return Ok(false);
            }
            let qi = (q - slack).ceil();
            if qi > self.elo[v] + 0.5 {
                if qi > self.whi[v] + 0.5 {
                    return Err(Error::Infeasible);
                }
                self.elo[v] = qi;
                self.wlo[v] = self.wlo[v].max(qi);
                self.stats.bounds_tightened += 1;
                self.mark_var(v);
                return Ok(true);
            }
        } else {
            let relax = IMPROVE_REL * q.abs().max(1.0) + amp_noise;
            let nq = q - relax;
            if nq > self.wlo[v] {
                if nq > self.whi[v] {
                    let scale = nq.abs().max(self.whi[v].abs()).max(1.0);
                    if nq - self.whi[v] > FEAS_TOL * scale {
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

    /// An all-substituted row: consistent (drop) or a proof of infeasibility.
    /// The tolerance scales with the magnitudes folded into it.
    fn empty_row_check(&mut self, row: &mut Row) -> Result<(), Error> {
        let tol = FEAS_TOL * (1.0 + row.fold_scale);
        if row.lo > tol || row.hi < -tol {
            return Err(Error::Infeasible);
        }
        row.alive = false;
        self.stats.rows_dropped += 1;
        Ok(())
    }

    /// Singleton row `a·x ∈ [lo, hi]`: fold it into the variable's EMITTED
    /// bounds (the bound replaces the row, exactly) and drop it. Returns
    /// whether the row was consumed; a sub-tolerance crossing keeps the row
    /// for the simplex instead.
    fn singleton_row(&mut self, row: &mut Row) -> Result<bool, Error> {
        let v = row.vars[0];
        let a = row.coeffs[0];
        let (mut qlo, mut qhi) = if a > 0.0 {
            (row.lo / a, row.hi / a)
        } else {
            (row.hi / a, row.lo / a)
        };
        if self.is_int(v) {
            if qlo.is_finite() {
                qlo = (qlo - self.int_tol).ceil();
            }
            if qhi.is_finite() {
                qhi = (qhi + self.int_tol).floor();
            }
        }
        let new_lo = self.elo[v].max(qlo);
        let new_hi = self.ehi[v].min(qhi);
        if new_lo > new_hi {
            let gap = new_lo - new_hi;
            let scale = new_lo.abs().max(new_hi.abs()).max(1.0);
            if self.is_int(v) || gap > FEAS_TOL * scale {
                return Err(Error::Infeasible);
            }
            return Ok(false); // sub-tolerance sliver: keep the row as-is
        }
        if new_lo > self.elo[v] || new_hi < self.ehi[v] {
            self.stats.bounds_tightened += 1;
            self.mark_var(v);
        }
        self.elo[v] = new_lo;
        self.ehi[v] = new_hi;
        // Working bounds may already be tighter; only ever shrink them.
        self.wlo[v] = self.wlo[v].max(new_lo).min(new_hi);
        self.whi[v] = self.whi[v].min(new_hi).max(new_lo);
        row.alive = false;
        self.stats.rows_dropped += 1;
        Ok(true)
    }

    /// One wave of the primal worklist: process every row queued so far;
    /// rows whose inputs change during the wave land in the next wave.
    fn primal_wave(&mut self, rows: &mut [Row]) -> Result<(), Error> {
        let wave = std::mem::take(&mut self.queue);
        for &r in &wave {
            // Clear before processing: a change made while processing this
            // very row (its own var tightened) legitimately requeues it.
            self.dirty[r as usize] = false;
            let row = &mut rows[r as usize];
            if !row.alive {
                continue;
            }
            self.fold_fixed(row);
            if row.vars.is_empty() {
                self.empty_row_check(row)?;
                continue;
            }
            if row.vars.len() == 1 {
                self.singleton_row(row)?;
                continue;
            }

            let wact = activity(row, &self.wlo, &self.whi);
            let nz = NOISE_REL * wact.abs;
            let (row_lo, row_hi) = (row.lo, row.hi);

            // Infeasibility: working bounds are valid implications, so a
            // beyond-tolerance activity violation is a proof. Generous
            // tolerance — a false positive is a wrong answer, a miss just
            // leaves the verdict to the simplex.
            let inf_tol = FEAS_TOL * wact.abs.max(1.0);
            if row_hi.is_finite() && wact.ninf_l == 0 && wact.l > row_hi + inf_tol {
                return Err(Error::Infeasible);
            }
            if row_lo.is_finite() && wact.ninf_u == 0 && wact.u < row_lo - inf_tol {
                return Err(Error::Infeasible);
            }

            // Forcing: the minimum working activity already meets the upper
            // side (or the maximum the lower side) — every feasible point has
            // each var pinned (within noise) at its extreme, so fixing there
            // is exact-within-tolerance. The trigger tolerance is the NOISE
            // model, not the feasibility tolerance: fixing interior-feasible
            // vars would cut true optima.
            if row_hi.is_finite() && wact.ninf_l == 0 && wact.l >= row_hi - nz {
                for i in 0..row.vars.len() {
                    let (v, a) = (row.vars[i], row.coeffs[i]);
                    let val = if a > 0.0 { self.wlo[v] } else { self.whi[v] };
                    self.fix(v, val);
                }
                row.alive = false;
                self.stats.rows_dropped += 1;
                continue;
            }
            if row_lo.is_finite() && wact.ninf_u == 0 && wact.u <= row_lo + nz {
                for i in 0..row.vars.len() {
                    let (v, a) = (row.vars[i], row.coeffs[i]);
                    let val = if a > 0.0 { self.whi[v] } else { self.wlo[v] };
                    self.fix(v, val);
                }
                row.alive = false;
                self.stats.rows_dropped += 1;
                continue;
            }

            // Redundancy: judged against EMITTED bounds only — material that
            // provably remains in the reduced problem — so dropping is never
            // justified by a deduction that the drop itself would orphan.
            // Cheap precheck first: working activities dominate emitted ones
            // (L_e <= L_w, U_e >= U_w), so if the working side already fails
            // there is no need to compute the emitted activity at all.
            let w_lo_red = !row_lo.is_finite()
                || (wact.ninf_l == 0 && wact.l >= row_lo + REDUNDANT_MARGIN * wact.abs.max(1.0));
            let w_hi_red = !row_hi.is_finite()
                || (wact.ninf_u == 0 && wact.u <= row_hi - REDUNDANT_MARGIN * wact.abs.max(1.0));
            if w_lo_red && w_hi_red {
                let eact = activity(row, &self.elo, &self.ehi);
                let red = REDUNDANT_MARGIN * eact.abs.max(1.0);
                let lo_red = !row_lo.is_finite() || (eact.ninf_l == 0 && eact.l >= row_lo + red);
                let hi_red = !row_hi.is_finite() || (eact.ninf_u == 0 && eact.u <= row_hi - red);
                if lo_red && hi_red {
                    row.alive = false;
                    self.stats.rows_dropped += 1;
                    continue;
                }
            }

            // Per-term implied bounds from working activities.
            for i in 0..row.vars.len() {
                let (v, a) = (row.vars[i], row.coeffs[i]);
                let (cmin, cmax) = term_range(a, self.wlo[v], self.whi[v]);
                let rest_l = if cmin.is_finite() {
                    (wact.ninf_l == 0).then(|| wact.l - cmin)
                } else {
                    (wact.ninf_l == 1).then_some(wact.l)
                };
                let rest_u = if cmax.is_finite() {
                    (wact.ninf_u == 0).then(|| wact.u - cmax)
                } else {
                    (wact.ninf_u == 1).then_some(wact.u)
                };
                let amp = nz / a.abs();
                if row_hi.is_finite() {
                    if let Some(rl) = rest_l {
                        let q = (row_hi - rl) / a;
                        if a > 0.0 {
                            self.tighten_upper(v, q, amp)?;
                        } else {
                            self.tighten_lower(v, q, amp)?;
                        }
                    }
                }
                if row_lo.is_finite() {
                    if let Some(ru) = rest_u {
                        let q = (row_lo - ru) / a;
                        if a > 0.0 {
                            self.tighten_lower(v, q, amp)?;
                        } else {
                            self.tighten_upper(v, q, amp)?;
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
    /// last bit and the vacuous branch is relaxed outward by the noise
    /// budget); the LP relaxation tightens. Mip mode only.
    fn tighten_binary_coeffs(&mut self, rows: &mut [Row]) -> bool {
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
            let mut act = activity(row, &self.wlo, &self.whi);
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
                let nz = NOISE_REL * act.abs;
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
                let nz = NOISE_REL * act.abs;
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
                    row.touched = true;
                    // The row's data changed: reprocess it in the primal
                    // worklist (new redundancies/bounds may follow). The
                    // activities were updated incrementally above (act.abs is
                    // left stale-high, which only makes the noise budget more
                    // conservative).
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
    fn dual_fix(&mut self, rows: &[Row], obj: &[f64]) -> bool {
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

/// Run presolve. `obj_coeffs` are the INTERNAL (minimize-space) objective
/// coefficients, exactly as stored on `Problem`. Returns `Err(Infeasible)`
/// when the reductions prove no feasible point exists.
#[allow(clippy::too_many_arguments)] // mirrors Solver::try_new's flat-arrays style
pub(crate) fn presolve(
    obj_coeffs: &[f64],
    var_mins: &[f64],
    var_maxs: &[f64],
    constraints: &[(CsVec, ComparisonOp, f64)],
    var_domains: &[VarDomain],
    mode: Mode,
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
            alive: true,
            orig: orig as u32,
            touched: false,
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
        int_tol,
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
            let tol = FEAS_TOL * (1.0 + row.fold_scale);
            if row.lo > tol || row.hi < -tol {
                return Err(Error::Infeasible);
            }
            work.stats.rows_dropped += 1;
            continue;
        }
        if !row.touched {
            // Never rewritten: emit the input row byte-identically (a CsVec
            // clone is a memcpy; rebuilding would re-sort for nothing).
            out.push(constraints[row.orig as usize].clone());
            continue;
        }
        let (op, rhs) = match (row.lo.is_finite(), row.hi.is_finite()) {
            (false, true) => (ComparisonOp::Le, row.hi),
            (true, false) => (ComparisonOp::Ge, row.lo),
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
                (ComparisonOp::Eq, row.lo)
            }
            (false, false) => {
                unreachable!("a both-sides-infinite row is always dropped as redundant")
            }
        };
        out.push((
            CsVec::new_from_unsorted(num_vars, row.vars.clone(), row.coeffs.clone())
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
mod tests {
    use super::*;

    fn csvec(n: usize, terms: &[(usize, f64)]) -> CsVec {
        CsVec::new_from_unsorted(
            n,
            terms.iter().map(|t| t.0).collect(),
            terms.iter().map(|t| t.1).collect(),
        )
        .unwrap()
    }

    fn real(n: usize) -> Vec<VarDomain> {
        vec![VarDomain::Real; n]
    }

    fn run_lp(
        mins: &[f64],
        maxs: &[f64],
        cons: &[(CsVec, ComparisonOp, f64)],
    ) -> Result<Presolved, Error> {
        let n = mins.len();
        presolve(
            &vec![0.0; n],
            mins,
            maxs,
            cons,
            &real(n),
            Mode::Lp,
            1e-6,
            false,
        )
    }

    #[test]
    fn kept_rows_do_not_leak_working_bounds() {
        // x + y <= 4 with x, y >= 0: presolve DEDUCES x, y <= 4 internally
        // but the row stays, so the emitted bounds must be untouched (an
        // emitted redundant bound would only perturb pivot order).
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 4.0)];
        let pre = run_lp(&[0.0, 0.0], &[f64::INFINITY, f64::INFINITY], &cons).unwrap();
        assert_eq!(pre.constraints.len(), 1, "row is binding, must be kept");
        assert_eq!(pre.var_maxs, vec![f64::INFINITY, f64::INFINITY]);
    }

    #[test]
    fn integer_bounds_are_rounded_in_mip_mode() {
        let cons: Vec<(CsVec, ComparisonOp, f64)> = vec![];
        let pre = presolve(
            &[1.0],
            &[0.3],
            &[2.7],
            &cons,
            &[VarDomain::Integer],
            Mode::Mip,
            1e-6,
            false,
        )
        .unwrap();
        assert_eq!(pre.var_mins[0], 1.0);
        assert_eq!(pre.var_maxs[0], 2.0);
    }

    #[test]
    fn integer_bound_from_kept_row_is_emitted() {
        // 2x + y <= 7 with y in [0, 10]: x <= 3.5 -> x <= 3 must be emitted
        // even though the row stays (integer rounding genuinely tightens).
        let cons = vec![(csvec(2, &[(0, 2.0), (1, 1.0)]), ComparisonOp::Le, 7.0)];
        let pre = presolve(
            &[-1.0, 0.0],
            &[0.0, 0.0],
            &[100.0, 10.0],
            &cons,
            &[VarDomain::Integer, VarDomain::Real],
            Mode::Mip,
            1e-6,
            false,
        )
        .unwrap();
        assert_eq!(pre.var_maxs[0], 3.0);
        assert_eq!(pre.var_maxs[1], 10.0, "continuous bound must not persist");
        assert_eq!(pre.constraints.len(), 1);
    }

    #[test]
    fn fractional_integer_range_is_infeasible() {
        // No integer in [0.4, 0.6].
        let cons: Vec<(CsVec, ComparisonOp, f64)> = vec![];
        let err = presolve(
            &[1.0],
            &[0.4],
            &[0.6],
            &cons,
            &[VarDomain::Integer],
            Mode::Mip,
            1e-6,
            false,
        )
        .unwrap_err();
        assert_eq!(err, Error::Infeasible);
    }

    #[test]
    fn singleton_row_becomes_bound_and_is_dropped() {
        // 2x <= 10  ->  x <= 5, row gone (the bound replaces the row).
        let cons = vec![(csvec(1, &[(0, 2.0)]), ComparisonOp::Le, 10.0)];
        let pre = run_lp(&[0.0], &[f64::INFINITY], &cons).unwrap();
        assert!(pre.constraints.is_empty());
        assert_eq!(pre.var_maxs[0], 5.0);
    }

    #[test]
    fn singleton_eq_fixes_var_and_substitutes() {
        // x == 3; x + y <= 10  ->  y <= 7 (both rows gone: the second becomes
        // a singleton on y after substitution).
        let cons = vec![
            (csvec(2, &[(0, 1.0)]), ComparisonOp::Eq, 3.0),
            (csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 10.0),
        ];
        let pre = run_lp(&[0.0, 0.0], &[f64::INFINITY, f64::INFINITY], &cons).unwrap();
        assert!(pre.constraints.is_empty());
        assert_eq!(pre.var_mins[0], 3.0);
        assert_eq!(pre.var_maxs[0], 3.0);
        assert_eq!(pre.var_maxs[1], 7.0);
    }

    #[test]
    fn redundant_row_is_dropped() {
        // x + y <= 10 with both vars in [0, 3]: implied by bounds, dropped.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 10.0)];
        let pre = run_lp(&[0.0, 0.0], &[3.0, 3.0], &cons).unwrap();
        assert!(pre.constraints.is_empty());
    }

    #[test]
    fn forcing_ge_row_fixes_all_vars() {
        // x + y >= 10 with x, y <= 5: only x = y = 5 works.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 10.0)];
        let pre = run_lp(&[0.0, 0.0], &[5.0, 5.0], &cons).unwrap();
        assert!(pre.constraints.is_empty());
        assert_eq!((pre.var_mins[0], pre.var_maxs[0]), (5.0, 5.0));
        assert_eq!((pre.var_mins[1], pre.var_maxs[1]), (5.0, 5.0));
    }

    #[test]
    fn activity_infeasibility_is_detected() {
        // x + y >= 11 with x, y <= 5: max activity 10 < 11.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Ge, 11.0)];
        assert_eq!(
            run_lp(&[0.0, 0.0], &[5.0, 5.0], &cons).unwrap_err(),
            Error::Infeasible
        );
    }

    #[test]
    fn crossed_input_bounds_are_infeasible() {
        let cons: Vec<(CsVec, ComparisonOp, f64)> = vec![];
        assert_eq!(
            run_lp(&[1.0], &[0.0], &cons).unwrap_err(),
            Error::Infeasible
        );
    }

    #[test]
    fn user_empty_rows_keep_solver_semantics() {
        // 0 <= 1 tautological, dropped; 0 >= 1 infeasible.
        let taut = vec![(csvec(1, &[]), ComparisonOp::Le, 1.0)];
        let pre = run_lp(&[0.0], &[1.0], &taut).unwrap();
        assert!(pre.constraints.is_empty());
        let bad = vec![(csvec(1, &[]), ComparisonOp::Ge, 1.0)];
        assert_eq!(run_lp(&[0.0], &[1.0], &bad).unwrap_err(), Error::Infeasible);
    }

    #[test]
    fn dual_fixing_pins_costly_var_to_lower_bound() {
        // minimize x, x in [2, 9], only occurrence x + y <= 100 (a > 0 in a
        // <=-row never blocks decreasing) -> fixed at 2. Gated on mode+flag.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 100.0)];
        let doms = vec![VarDomain::Integer, VarDomain::Integer];
        let fixed = presolve(
            &[1.0, 0.0],
            &[2.0, 0.0],
            &[9.0, 5.0],
            &cons,
            &doms,
            Mode::Mip,
            1e-6,
            true,
        )
        .unwrap();
        assert_eq!((fixed.var_mins[0], fixed.var_maxs[0]), (2.0, 2.0));
        // y has c == 0 and is only in a <=-row: dual-fixable at its lower bound too.
        assert_eq!((fixed.var_mins[1], fixed.var_maxs[1]), (0.0, 0.0));

        let hinted = presolve(
            &[1.0, 0.0],
            &[2.0, 0.0],
            &[9.0, 5.0],
            &cons,
            &doms,
            Mode::Mip,
            1e-6,
            false, // warm hint present -> no dual fixing
        )
        .unwrap();
        assert_eq!((hinted.var_mins[0], hinted.var_maxs[0]), (2.0, 9.0));
    }

    #[test]
    fn dual_fixing_blocked_by_eq_row_and_infinite_bound() {
        // x in an Eq row: both directions blocked; nothing may be fixed here.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Eq, 5.0)];
        let pre = presolve(
            &[1.0, -1.0],
            &[0.0, 0.0],
            &[9.0, 9.0],
            &cons,
            &real(2),
            Mode::Mip,
            1e-6,
            true,
        )
        .unwrap();
        assert!(pre.var_mins[0] < pre.var_maxs[0]);
        assert!(pre.var_mins[1] < pre.var_maxs[1]);

        // minimize x with lo = -inf and no rows: possibly unbounded, must be
        // left for the simplex (no fixing, no error).
        let none: Vec<(CsVec, ComparisonOp, f64)> = vec![];
        let pre = presolve(
            &[1.0],
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            &none,
            &real(1),
            Mode::Mip,
            1e-6,
            true,
        )
        .unwrap();
        assert_eq!(pre.var_mins[0], f64::NEG_INFINITY);
    }

    #[test]
    fn binary_coefficient_tightening_le_row() {
        // 10 x + y <= 12, x binary, y in [0, 5]: x=1 forces y <= 2 and x=0 is
        // vacuous (5 <= 12) -> becomes (about) 3x + y <= 5. Integer content
        // identical, LP relaxation strictly tighter.
        let cons = vec![(csvec(2, &[(0, 10.0), (1, 1.0)]), ComparisonOp::Le, 12.0)];
        let pre = presolve(
            &[0.0, -1.0],
            &[0.0, 0.0],
            &[1.0, 5.0],
            &cons,
            &[VarDomain::Boolean, VarDomain::Real],
            Mode::Mip,
            1e-6,
            false,
        )
        .unwrap();
        assert_eq!(pre.constraints.len(), 1);
        let (coeffs, op, rhs) = &pre.constraints[0];
        assert!(matches!(op, ComparisonOp::Le));
        let a_x = coeffs.get(0).copied().unwrap();
        let a_y = coeffs.get(1).copied().unwrap();
        assert_eq!(a_y, 1.0);
        assert!((a_x - 3.0).abs() < 1e-6, "10 -> ~3, got {}", a_x);
        assert!((rhs - 5.0).abs() < 1e-6, "12 -> ~5, got {}", rhs);
        // The binding branch is preserved exactly: rhs - a_x == 12 - 10.
        assert!((rhs - a_x - 2.0).abs() < 1e-9);
        assert_eq!(pre.stats.coeffs_tightened, 1);
    }

    #[test]
    fn binary_coefficient_tightening_ge_row() {
        // -10 x + y >= -8, x binary, y in [0, 5]: x=1 forces y >= 2, x=0
        // vacuous (0 >= -8) -> lo' = rest_l = 0, a' = a + (rest_l - lo) =
        // -10 + 8 = -2: becomes -2x + y >= 0.
        let cons = vec![(csvec(2, &[(0, -10.0), (1, 1.0)]), ComparisonOp::Ge, -8.0)];
        let pre = presolve(
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[1.0, 5.0],
            &cons,
            &[VarDomain::Boolean, VarDomain::Real],
            Mode::Mip,
            1e-6,
            false,
        )
        .unwrap();
        assert_eq!(pre.constraints.len(), 1);
        let (coeffs, op, rhs) = &pre.constraints[0];
        assert!(matches!(op, ComparisonOp::Ge));
        let a_x = coeffs.get(0).copied().unwrap();
        assert!((a_x + 2.0).abs() < 1e-6, "-10 -> ~-2, got {}", a_x);
        assert!(rhs.abs() < 1e-6, "-8 -> ~0, got {}", rhs);
        assert!(
            (rhs - a_x - 2.0).abs() < 1e-9,
            "x=1 branch must stay y >= 2"
        );
    }

    #[test]
    fn knapsack_coefficient_tightening_shrinks_loose_coeff() {
        // 5x + 7y + 4z + 3w <= 14, binaries. y's coefficient is loose: when
        // y = 0 the rest can reach at most 12 < 14, so the row is equivalent
        // to 5x + 5y + 4z + 3w <= 12 on binary points (y = 1 still forces
        // rest <= 7). x/z/w after the update: rest_u == b, so they stay.
        let cons = vec![(
            csvec(4, &[(0, 5.0), (1, 7.0), (2, 4.0), (3, 3.0)]),
            ComparisonOp::Le,
            14.0,
        )];
        let pre = presolve(
            &[-8.0, -11.0, -6.0, -4.0],
            &[0.0; 4],
            &[1.0; 4],
            &cons,
            &vec![VarDomain::Boolean; 4],
            Mode::Mip,
            1e-6,
            true,
        )
        .unwrap();
        assert_eq!(pre.stats.coeffs_tightened, 1);
        assert_eq!(pre.constraints.len(), 1);
        let (coeffs, _, rhs) = &pre.constraints[0];
        assert!((rhs - 12.0).abs() < 1e-6, "14 -> ~12, got {}", rhs);
        assert_eq!(coeffs.get(0).copied().unwrap(), 5.0);
        let a_y = coeffs.get(1).copied().unwrap();
        assert!((a_y - 5.0).abs() < 1e-6, "7 -> ~5, got {}", a_y);
        assert_eq!(coeffs.get(2).copied().unwrap(), 4.0);
        assert_eq!(coeffs.get(3).copied().unwrap(), 3.0);
        // y = 1 branch preserved exactly: rhs - a_y == 14 - 7.
        assert!((rhs - a_y - 7.0).abs() < 1e-9);
    }

    #[test]
    fn lp_mode_never_rounds_or_dual_fixes() {
        // Same shape as the dual-fixing test but Mode::Lp: bounds must come
        // out exactly as they went in.
        let cons = vec![(csvec(2, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Le, 100.0)];
        let pre = presolve(
            &[1.0, 0.0],
            &[0.3, 0.0],
            &[9.7, 5.0],
            &cons,
            &real(2),
            Mode::Lp,
            1e-6,
            true, // even with the flag on, Lp mode must not dual-fix
        )
        .unwrap();
        assert_eq!(pre.var_mins, vec![0.3, 0.0]);
        assert_eq!(pre.var_maxs, vec![9.7, 5.0]);
    }

    #[test]
    fn cascading_substitution_reaches_fixpoint() {
        // x == 2 (singleton Eq), x + y == 5 -> y == 3, y + z <= 4 -> z <= 1.
        let cons = vec![
            (csvec(3, &[(0, 1.0)]), ComparisonOp::Eq, 2.0),
            (csvec(3, &[(0, 1.0), (1, 1.0)]), ComparisonOp::Eq, 5.0),
            (csvec(3, &[(1, 1.0), (2, 1.0)]), ComparisonOp::Le, 4.0),
        ];
        let pre = run_lp(&[0.0; 3], &[f64::INFINITY; 3], &cons).unwrap();
        assert!(pre.constraints.is_empty());
        assert_eq!((pre.var_mins[0], pre.var_maxs[0]), (2.0, 2.0));
        assert_eq!((pre.var_mins[1], pre.var_maxs[1]), (3.0, 3.0));
        assert_eq!(pre.var_maxs[2], 1.0);
    }

    #[test]
    fn inconsistent_substitution_is_infeasible() {
        // x == 2 and x == 3 via two singleton Eq rows.
        let cons = vec![
            (csvec(1, &[(0, 1.0)]), ComparisonOp::Eq, 2.0),
            (csvec(1, &[(0, 1.0)]), ComparisonOp::Eq, 3.0),
        ];
        assert_eq!(
            run_lp(&[0.0], &[f64::INFINITY], &cons).unwrap_err(),
            Error::Infeasible
        );
    }

    #[test]
    fn infinite_rhs_le_row_is_dropped_not_crashed() {
        let cons = vec![(
            csvec(2, &[(0, 1.0), (1, 1.0)]),
            ComparisonOp::Le,
            f64::INFINITY,
        )];
        let pre = run_lp(&[0.0, 0.0], &[1.0, 1.0], &cons).unwrap();
        assert!(pre.constraints.is_empty());
    }
}
