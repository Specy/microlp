//! The dual simplex's working costs: perturbation, shifting and cleanup.
//!
//! A var's working cost is its cost in force (the model's, or the phase-1
//! artificial objective) plus [`Solver::cost_perturbation`] (the
//! perturbation and the correcting shifts) and [`Solver::cost_shifts`] (the
//! shift of a zero dual step, which lives only while its var is basic). A
//! dual phase's reduced costs, multipliers and objective are those of the
//! working costs.
//!
//! * **Perturbation.** At a dual degenerate vertex the dual steps are zero,
//!   and the dual simplex can stall or cycle. Costs perturbed by random
//!   amounts far above the dual tolerance make ties between breakpoints, and
//!   so zero steps, the exception; [`super::cycling`] breaks the cycles
//!   that remain.
//! * **Shifting.** The Harris pass lets a reduced cost cross zero by its
//!   tolerance, so the step `d_q / a_q` can run backwards, leave the leaving
//!   var dual infeasible and repeat a basis. The step is taken as zero
//!   instead and the entering var's cost shifted by `-d_q`, which leaves
//!   every other reduced cost as it is.
//! * **Correction.** The dual simplex needs dual feasible costs, and a warm
//!   start (a loaded basis, a changed bound, an added row) can reach the
//!   dual phase without them: before the first step, a dual infeasible
//!   boxed var moves to its other bound (unless phase 2 is forced), and any
//!   other has its cost shifted.
//! * **Cleanup.** However the phase ends, the working costs return to the
//!   costs in force; `run_phases` recomputes the reduced costs once it has
//!   verified the values, and leaves any dual infeasibility they show to the
//!   primal simplex.

use super::{
    random_fraction, NonBasicVarState, Solver, Stream, EPS, NEAR_OPTIMAL_MAX_INFEASIBILITIES,
    NEAR_OPTIMAL_MAX_INFEASIBILITY,
};
use crate::Error;

impl NonBasicVarState {
    /// Whether the var sits at its upper bound and not at its lower one (a
    /// fixed var sits at both): the part of its state the vertex hash of the
    /// cycling detection covers.
    pub(super) fn at_upper(&self) -> bool {
        self.at_max && !self.at_min
    }

    /// How far the reduced cost `d` of a non-basic var in this state lies on
    /// its infeasible side; zero for a fixed var, which is dual feasible
    /// whatever its reduced cost. The var is dual infeasible when this
    /// reaches its tolerance. A `d` that is not a number gives not a number
    /// in every state, a fixed var's too, so that it compares false with
    /// every tolerance and the primal pricing reports the breakdown
    /// (`price_primal`).
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
    /// Start a dual phase: make the reduced costs of the costs in force exact,
    /// perturb the costs ([`Self::perturb_costs`]) unless the phase is a
    /// cleanup (`cleanup`, see `run_phases`) or starts near optimal, correct
    /// the dual infeasibilities ([`Self::correct_dual_infeasibilities`]), and
    /// start the cycling detection.
    ///
    /// A perturbation whose optimum is a few pivots away costs more pivots in
    /// the cleanup than it saves, hence the near-optimal test. With phase 2
    /// forced (a cleanup, or dual infeasibilities that are all small), a boxed
    /// var is shifted rather than flipped: a flip for an infeasibility the
    /// perturbation created, or one a cleanup starts with (an integer var at
    /// a branching bound, say), moves the primal point for nothing, and in a
    /// cleanup can take the phase to a row it cannot repair and back, round
    /// after round.
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
    /// tolerance, and the largest amount by which such a var is outside.
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

    /// Perturb the working costs.
    ///
    /// A structural var's cost moves by `(1 + r_j) (|c_j| + 1) base`, `r_j` a
    /// pseudo-random fraction of the var, in the direction that keeps its
    /// reduced cost feasible at a bound it can sit at. The largest cost enters
    /// the base by its fourth root above 100, so that a huge objective does
    /// not swamp the perturbation of the small costs; a slack's cost moves
    /// only to break ties between rows.
    ///
    /// A cost of [`INFINITE_COST`] or more (or one that is not a number)
    /// neither sets the base nor moves: the perturbation of the largest cost
    /// grows like `1e-6 |c|^(5/4)` and would reach infinity above about
    /// `2e251`. With that limit the base is at most
    /// `5e-7 * 1e20^(1/4) = 0.05`, so every working cost moves by less than a
    /// tenth of `|c_j| + 1`; with costs of order one, by about `1e-6`, far
    /// above the dual tolerance.
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
        for var in num_vars..num_total_vars {
            self.cost_perturbation[var] = (0.5 - random_fraction(var, Stream::Perturbation))
                * ROW_COST_PERTURBATION
                * self.dual_unit(var).min(1.0);
        }
        self.costs_modified = true;
    }

    /// Remove every dual infeasibility of the current reduced costs: move a
    /// boxed var to the bound where its reduced cost's sign is feasible (one
    /// solve for all, [`Self::apply_flips`]), and shift the cost of any other
    /// ([`Self::shift_to_dual_feasible`]). With `shift_only` (a phase that
    /// forces phase 2, and every rebuild within a phase, see `rebuild_dual`),
    /// boxed vars are shifted too. A fixed var is always dual feasible.
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
    /// its tolerance on the feasible side at a bound (`r` a pseudo-random
    /// fraction of the var), zero at neither bound. The working objective
    /// moves by the shift times the var's value.
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
    /// reduced cost is zero, for a zero dual step. The shift is given back
    /// when the var leaves the basis ([`Self::pivot`]).
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

    /// End a dual phase, however it ends: the working costs return to the
    /// costs in force, and the objective becomes that of the current values
    /// under them. The reduced costs are only marked stale: `run_phases`
    /// recomputes them once it has verified the basic values against the
    /// rows, and the primal simplex removes any dual infeasibility they show.
    /// Recomputing them here would send `run_phases` to its primal phase
    /// before the values are verified, which can end above the optimum
    /// (`wide_lp_558_reaches_the_optimum` in `src/tests/magnitudes.rs`).
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

/// A structural cost this large or larger, in the engine's scaled units
/// (which the perturbation formula sees), takes no part in the cost
/// perturbation ([`Solver::perturb_costs`]); the var and its cost stay as
/// the model gives them. This caps the perturbation base at `0.05`.
const INFINITE_COST: f64 = 1e20;

/// The scale of the perturbation of a slack's cost, bounding it in the
/// user's units of its reduced cost and in the engine's (the engine's
/// perturbation is scaled by the slack's [`super::dual_tol`] unit where below
/// one): below its dual tolerance in both, it only breaks ties between rows.
const ROW_COST_PERTURBATION: f64 = 1e-12;
