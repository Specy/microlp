//! The dual simplex's working costs: the cost perturbation a dual phase
//! starts with, the cost shifts that stand in for backward dual steps and
//! correct the dual infeasibilities a phase starts from, and the cleanup
//! that removes all of them when the phase ends. This is HiGHS's
//! anti-degeneracy package for its dual simplex (`HEkk::initialiseCost`,
//! highs/simplex/HEkk.cpp; `HEkkDual::solve`, `shiftCost`, `shiftBack`,
//! `correctDualInfeasibilities`, `cleanup`, highs/simplex/HEkkDual.cpp).
//! SoPlex does the same with its dual bounds: it shifts every violated one
//! by a random 10 to 100 times its tolerance when its dual simplex starts,
//! and removes the shifts before it accepts a result
//! (`SPxSolverBase::shiftPvec`, `unShift`, src/soplex/spxshift.hpp).
//!
//! The working cost of a var is its cost in force (the model's, or the
//! phase-1 artificial objective while that is in force) plus two terms:
//!
//! * [`Solver::cost_perturbation`] — the perturbation drawn when the phase
//!   starts, plus the shifts that correct a reduced cost found dual
//!   infeasible (HiGHS adds both into `workCost_`);
//! * [`Solver::cost_shifts`] — the shift that made the reduced cost of an
//!   entering var zero at a zero dual step (HiGHS's `workShift_`); it is
//!   given back when the var leaves the basis again (`shiftBack`), so it
//!   lives only while the var is basic.
//!
//! The reduced costs, the multipliers and the objective the dual phase
//! works with are those of the working costs. Why each term:
//!
//! * **Perturbation.** A dual degenerate vertex (reduced costs at zero)
//!   gives dual steps of zero, on which the dual simplex can stall, or
//!   cycle through bases of one objective value. Costs perturbed by random
//!   amounts far above the dual tolerance make ties between breakpoints, and
//!   so zero steps, the exception. (The four-pivot cycle the property stress
//!   found was of another kind, driven by absorbing pivots, and survived
//!   the perturbation; the cycling detection breaks it: see
//!   [`super::cycling`].)
//! * **Shifting.** The Harris pass lets a reduced cost cross zero by its
//!   tolerance, so the entering var's own reduced cost can lie on the wrong
//!   side of zero. The step `d_q / a_q` is then backwards: the dual
//!   objective falls, the leaving var ends dual infeasible, and a sequence
//!   of such steps can repeat a basis. The step is taken as zero instead
//!   and the entering var's cost shifted by `-d_q`, which is exactly what a
//!   zero step needs: every other reduced cost stays as it is.
//! * **Correction.** A warm start (a loaded basis, a changed bound, an
//!   added row) can reach the dual phase dual infeasible. Before the first
//!   step, a boxed var with a reduced cost of the wrong sign moves to its
//!   other bound (where the sign is right) and any other var has its cost
//!   shifted until its reduced cost is just feasible, so the phase runs on
//!   dual feasible working costs, as the dual simplex requires. A phase that
//!   cleans up after another shifts the boxed var's cost too, as HiGHS's
//!   cleanup calls force phase 2.
//! * **Cleanup.** When the phase ends, however it ends, the working costs
//!   return to the costs in force; `run_phases` recomputes the reduced costs
//!   from them once it has verified the values, and what dual infeasibility
//!   that uncovers is the primal simplex's to remove, as HiGHS follows its
//!   cleanup with one primal simplex pass.

use super::{
    random_fraction, NonBasicVarState, Solver, Stream, EPS, NEAR_OPTIMAL_MAX_INFEASIBILITIES,
    NEAR_OPTIMAL_MAX_INFEASIBILITY,
};
use crate::Error;

impl NonBasicVarState {
    /// Whether the var sits at its upper bound and not at its lower one: the
    /// part of a non-basic var's state the vertex hash of the cycling
    /// detection covers (a fixed var sits at both, a free one at neither).
    pub(super) fn at_upper(&self) -> bool {
        self.at_max && !self.at_min
    }

    /// How far the reduced cost `d` of a non-basic var in this state lies on
    /// its infeasible side (HiGHS's `-nonbasicMove * dual`, `|dual|` for a
    /// free var): `-d` at a lower bound, `d` at an upper bound, `|d|` for a
    /// var at neither (free, or left between its bounds), and zero for a
    /// fixed var, which is dual feasible whatever its reduced cost. Positive
    /// means infeasible by that much; the var counts as dual infeasible
    /// when this reaches its tolerance. A reduced cost that is not a number
    /// gives not a number in every state, a fixed var's too: it compares
    /// false with every tolerance, so the var counts as dual infeasible and
    /// the primal pricing reports the breakdown (`price_primal`).
    pub(super) fn dual_infeasibility(&self, d: f64) -> f64 {
        match (self.at_min, self.at_max) {
            (true, true) if d.is_nan() => d,
            (true, true) => 0.0,
            (true, false) => -d,
            (false, true) => d,
            (false, false) => d.abs(),
        }
    }

    /// Whether the reduced cost `d` is a dual infeasibility to correct: on
    /// its infeasible side by at least `tol`. A reduced cost that is not a
    /// number is not one; it is left to the pricing, which reports it.
    fn needs_correction(&self, d: f64, tol: f64) -> bool {
        self.dual_infeasibility(d) >= tol
    }
}

impl Solver {
    /// Start a dual phase, as HiGHS's `HEkkDual::solve` does up to the first
    /// rebuild of its phase-2 loop.
    ///
    /// The reduced costs of the costs in force are made exact and their dual
    /// infeasibilities measured. A phase that cleans up after the one before
    /// it (`cleanup`, see `run_phases`) runs as HiGHS's cleanup calls of its
    /// dual simplex do (`HEkkPrimal::solve` calls `HEkkDual::solve` with
    /// `pass_force_phase2` and the cost perturbation off): unperturbed, and
    /// forcing phase 2, so that its dual infeasibilities are all removed by
    /// shifts. Any other phase forces phase 2 when its dual infeasibilities
    /// are small (HiGHS's `force_phase2`, the largest infeasibility squared
    /// below the dual tolerance), and is perturbed ([`Self::perturb_costs`])
    /// unless it starts near optimal: with phase 2 forced or dual feasible,
    /// with fewer than a thousand rows violated and none by `1e-3` or more,
    /// since a perturbation of costs whose optimum is a few pivots away
    /// costs more pivots in the cleanup than it saves. Then the dual
    /// infeasibilities are corrected ([`Self::correct_dual_infeasibilities`]),
    /// by shifts alone when phase 2 is forced (flipping a boxed var for an
    /// infeasibility the perturbation created, or one a cleanup starts with,
    /// moves the primal point for nothing: in a cleanup the var is an
    /// integer at a branching bound, say, which the phase before moved the
    /// point off, and flipping it to its other bound took the phase to a
    /// row it could not repair, back through a zero step, round after round
    /// until the rounds ran out), and the cycling detection starts.
    pub(super) fn begin_dual_phase(&mut self, cleanup: bool) -> Result<(), Error> {
        debug_assert!(!self.costs_modified, "a dual phase is already running");
        if self.reduced_costs_dirty {
            self.recalc_obj_coeffs()?;
        }
        let mut num_dual_infeasible = 0usize;
        let mut max_dual_infeasibility = 0.0f64;
        for (c, (state, &d)) in self
            .nb_var_states
            .iter()
            .zip(&self.nb_var_obj_coeffs)
            .enumerate()
        {
            let infeasibility = state.dual_infeasibility(d);
            if infeasibility > 0.0 {
                max_dual_infeasibility = max_dual_infeasibility.max(infeasibility);
                if infeasibility >= self.dual_tols[self.nb_vars[c]] {
                    num_dual_infeasible += 1;
                }
            }
        }
        let force_phase2 = cleanup || max_dual_infeasibility * max_dual_infeasibility < EPS;
        let (num_primal_infeasible, max_primal_infeasibility) = self.primal_infeasibility_extent();
        let near_optimal = (num_dual_infeasible == 0 || force_phase2)
            && num_primal_infeasible < NEAR_OPTIMAL_MAX_INFEASIBILITIES
            && max_primal_infeasibility < NEAR_OPTIMAL_MAX_INFEASIBILITY;
        let perturbed = !cleanup && !near_optimal;
        if perturbed {
            self.perturb_costs();
            self.recalc_obj_coeffs()?;
        }
        trace!(
            "dual phase start: {num_dual_infeasible} dual infeasibilities (largest \
             {max_dual_infeasibility:e}), {num_primal_infeasible} rows violated (largest \
             {max_primal_infeasibility:e}); costs {}",
            if perturbed {
                "perturbed"
            } else if cleanup {
                "not perturbed (cleanup)"
            } else {
                "not perturbed (near optimal)"
            }
        );
        self.correct_dual_infeasibilities(force_phase2);
        self.start_cycle_detection();
        Ok(())
    }

    /// The number of basic vars outside their bounds beyond their
    /// tolerance, and the largest amount by which such a var is outside
    /// (HiGHS's `computeSimplexPrimalInfeasible`).
    pub(super) fn primal_infeasibility_extent(&self) -> (usize, f64) {
        let mut num = 0;
        let mut max = 0.0f64;
        for (r, ((&val, &min), &max_bound)) in self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
            .enumerate()
        {
            let tol = self.var_tols[self.basic_vars[r]];
            let excess = if val < min - tol.lo {
                min - val
            } else if val > max_bound + tol.hi {
                val - max_bound
            } else {
                continue;
            };
            num += 1;
            max = max.max(excess);
        }
        (num, max)
    }

    /// Perturb the working costs, as HiGHS's `HEkk::initialiseCost` does
    /// for its dual simplex with its default multiplier of one.
    ///
    /// The base is `5e-7` times the largest structural cost in force, taken
    /// as its fourth root above 100 (so a huge objective does not swamp the
    /// perturbation of the small costs), as at most one when fewer than 1%
    /// of the vars are boxed, and as one when every cost is zero (HiGHS's
    /// default path, which perturbs a zero objective as if its largest cost
    /// were one). A structural var's cost moves by
    /// `(1 + r_j) (|c_j| + 1) base`, `r_j` a pseudo-random fraction of the
    /// var, in the direction that keeps its reduced cost feasible at a
    /// bound it can sit at: up for a var bounded below only, down for one
    /// bounded above only, and in the direction of its cost's sign for a
    /// boxed var; a free or fixed var is left alone. A slack's cost moves by
    /// `(0.5 - r_i) 1e-12`, either way, to break ties between rows.
    ///
    /// A cost of [`INFINITE_COST`] or more (or one that is not a number)
    /// takes no part: it neither sets the base nor moves, as HiGHS's
    /// perturbation never sees such a cost (see the constant). The formula
    /// grows like `1e-6 |c|^(5/4)` for the largest cost, so without that
    /// limit a cost above about `2e251` perturbed to infinity, and the
    /// reduced costs computed from it were not numbers (the dual ratio test
    /// then found no candidate and reported a feasible model infeasible);
    /// already a cost of `1e30` moved by sixteen to thirty-two times itself.
    /// With it the base is at most `5e-7 * 1e20^(1/4) = 0.05`, so every
    /// working cost is finite and moves by less than a tenth of `|c_j| + 1`;
    /// when every cost is of order one, by about `1e-6`. That is far above
    /// the dual tolerance for any but tiny costs, so the breakpoints of the
    /// dual ratio test no longer tie where the costs did, and the vertex the
    /// perturbed phase ends at is optimal for the real costs, or the primal
    /// pivots of the cleanup reach an optimal one from it.
    pub(super) fn perturb_costs(&mut self) {
        let num_vars = self.num_vars;
        let num_total_vars = self.num_total_vars();
        let costs = self
            .artificial_obj
            .as_deref()
            .unwrap_or(&self.orig_obj_coeffs);
        let takes_part = |cost: f64| cost.abs() < INFINITE_COST;
        let mut max_abs_cost = costs[..num_vars]
            .iter()
            .filter(|&&c| takes_part(c))
            .fold(0.0f64, |max, c| max.max(c.abs()));
        if max_abs_cost == 0.0 {
            max_abs_cost = 1.0;
        }
        if max_abs_cost > 100.0 {
            max_abs_cost = max_abs_cost.sqrt().sqrt();
        }
        let num_boxed = (0..num_total_vars)
            .filter(|&v| self.orig_var_maxs[v] - self.orig_var_mins[v] < 1e30)
            .count();
        if (num_boxed as f64) < 0.01 * num_total_vars as f64 {
            max_abs_cost = max_abs_cost.min(1.0);
        }
        let base = 5e-7 * max_abs_cost;
        for (var, &cost) in costs.iter().enumerate().take(num_vars) {
            let (min, max) = (self.orig_var_mins[var], self.orig_var_maxs[var]);
            let amount =
                (1.0 + random_fraction(var, Stream::Perturbation)) * (cost.abs() + 1.0) * base;
            self.cost_perturbation[var] =
                if !takes_part(cost) || min == max || (min.is_infinite() && max.is_infinite()) {
                    0.0
                } else if max.is_infinite() {
                    amount
                } else if min.is_infinite() || cost < 0.0 {
                    -amount
                } else {
                    amount
                };
        }
        for (var, perturbation) in self.cost_perturbation.iter_mut().enumerate().skip(num_vars) {
            *perturbation =
                (0.5 - random_fraction(var, Stream::Perturbation)) * ROW_COST_PERTURBATION;
        }
        self.costs_modified = true;
    }

    /// Remove every dual infeasibility of the current reduced costs, as
    /// HiGHS's `HEkkDual::correctDualInfeasibilities` does at the rebuilds of
    /// its dual simplex: a var whose reduced cost is on its infeasible side
    /// by at least its tolerance is, if boxed, moved to the bound where that
    /// sign is feasible (all such moves carried into the basic values by one
    /// solve, [`Self::apply_flips`]), and otherwise has its working cost
    /// shifted until its reduced cost is just feasible
    /// ([`Self::shift_to_dual_feasible`]). With `shift_only`, boxed vars are
    /// shifted too: at the start of a phase that forces phase 2 (a cleanup,
    /// or one whose dual infeasibilities were small before the perturbation:
    /// HiGHS's `force_phase2`), and at every rebuild within a phase (see
    /// `rebuild_dual`). A fixed var is always dual feasible.
    pub(super) fn correct_dual_infeasibilities(&mut self, shift_only: bool) {
        let mut flips = Vec::new();
        for col in 0..self.nb_vars.len() {
            let var = self.nb_vars[col];
            let d = self.nb_var_obj_coeffs[col];
            if !self.nb_var_states[col].needs_correction(d, self.dual_tols[var]) {
                continue;
            }
            let (min, max) = (self.orig_var_mins[var], self.orig_var_maxs[var]);
            if !shift_only && min.is_finite() && max.is_finite() {
                flips.push((col, if d > 0.0 { min } else { max }));
            } else {
                self.shift_to_dual_feasible(col);
            }
        }
        if !flips.is_empty() {
            trace!(
                "dual phase: {} boxed vars moved to their dual feasible bound",
                flips.len()
            );
            self.apply_flips(&flips);
        }
    }

    /// Shift the working cost of the non-basic var at `col` so that its
    /// reduced cost is feasible for its state, if it is not: `(1 + r)` times
    /// its tolerance on the feasible side at a bound (HiGHS's
    /// `correctDualInfeasibilities`, `r` a pseudo-random fraction of the
    /// var), zero at neither bound (HiGHS's `exitPhase1ResetDuals` for a free
    /// var). The working objective moves by the shift times the var's value.
    pub(super) fn shift_to_dual_feasible(&mut self, col: usize) {
        let var = self.nb_vars[col];
        let state = &self.nb_var_states[col];
        let d = self.nb_var_obj_coeffs[col];
        let tol = self.dual_tols[var];
        if !state.needs_correction(d, tol) {
            return;
        }
        let margin = (1.0 + random_fraction(var, Stream::Shift)) * tol;
        let new_d = match (state.at_min, state.at_max) {
            (true, false) => margin,
            (false, true) => -margin,
            _ => 0.0,
        };
        let shift = new_d - d;
        self.cost_perturbation[var] += shift;
        self.nb_var_obj_coeffs[col] = new_d;
        self.cur_obj_val += shift * self.nb_var_vals[col];
        self.costs_modified = true;
    }

    /// Shift the working cost of the entering non-basic var at `col` so its
    /// reduced cost is zero, for a zero dual step (HiGHS's
    /// `HEkkDual::shiftCost`, `workShift_[q] = -workDual[q]`). The shift is
    /// given back when the var leaves the basis ([`Self::pivot`], HiGHS's
    /// `shiftBack`).
    pub(super) fn shift_entering_cost(&mut self, col: usize) {
        let var = self.nb_vars[col];
        let d = self.nb_var_obj_coeffs[col];
        if d == 0.0 {
            return;
        }
        debug_assert_eq!(self.cost_shifts[var], 0.0, "a non-basic var keeps a shift");
        self.cost_shifts[var] = -d;
        self.nb_var_obj_coeffs[col] = 0.0;
        self.cur_obj_val -= d * self.nb_var_vals[col];
        self.costs_modified = true;
    }

    /// End a dual phase, however it ends (HiGHS's `HEkkDual::cleanup`): the
    /// working costs return to the costs in force. The reduced costs, which
    /// belonged to the working costs, are left to be recomputed from the
    /// costs in force where `run_phases` verifies a phase's result: after the
    /// basic values have been verified against the rows, it recomputes the
    /// reduced costs exactly, and if they show a dual infeasibility (the
    /// perturbation or a shift left the vertex short of optimal for the real
    /// costs), the primal simplex removes it, as HiGHS follows its cleanup
    /// with one primal simplex pass. The objective becomes that of the
    /// current values under the costs in force. A cleanup that recomputed
    /// the reduced costs itself sent `run_phases` to its primal phase before
    /// the values were verified, and a model of the differential fuzz ended
    /// above its optimum (`wide_lp_558_reaches_the_optimum` in
    /// `src/tests/magnitudes.rs`).
    pub(super) fn end_dual_phase(&mut self) {
        self.cycle_guard.stop();
        if self.costs_modified {
            self.clear_working_costs();
            self.reduced_costs_dirty = true;
            self.cur_obj_val = self.exact_obj_val();
        }
    }

    fn clear_working_costs(&mut self) {
        self.cost_perturbation.fill(0.0);
        self.cost_shifts.fill(0.0);
        self.costs_modified = false;
    }
}

/// A structural cost this large or larger takes no part in the cost
/// perturbation ([`Solver::perturb_costs`]): HiGHS's `infinite_cost`
/// (default `1e20`, highs/lp_data/HighsOptions.h). HiGHS's perturbation never
/// sees such a cost: `assessCosts` (highs/lp_data/HighsLpUtils.cpp) makes it
/// infinite when the model is passed, and `Highs::handleInfCost`
/// (highs/lp_data/HighsInterface.cpp) fixes its var at the bound the cost
/// pushes it to and zeroes the cost before any solve, so `initialiseCost`
/// neither counts it in the largest cost nor perturbs the (fixed) column. The
/// engine keeps such a var and its cost as the model gives them and leaves
/// it out of the perturbation the same way; applied to the engine's scaled
/// costs, which are what the perturbation formula sees, the limit holds for
/// every cost it perturbs.
const INFINITE_COST: f64 = 1e20;

/// The scale of the perturbation of a slack's cost (HiGHS's row cost
/// perturbation base, `dual_simplex_cost_perturbation_multiplier * 1e-12`):
/// below any dual tolerance, it only breaks ties between rows.
const ROW_COST_PERTURBATION: f64 = 1e-12;
