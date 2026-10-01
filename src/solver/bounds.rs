//! The primal simplex's working bounds: perturbation, shifting and cleanup,
//! the counterpart of the dual simplex's working costs ([`super::costs`]).
//!
//! During a primal phase every bound the simplex looks at is a working
//! bound ([`Solver::working_bounds`]): the basic vars' (`basic_var_mins`,
//! `basic_var_maxs`), the entering var's other bound, and the bound a
//! non-basic var sits at. They start as the true bounds and move:
//!
//! * **Perturbation.** At a primal degenerate vertex the steps are zero, and
//!   the primal simplex can stall or cycle. Bounds moved outward by random
//!   amounts well above the tolerances make ties in the ratio test, and so
//!   zero steps, the exception.
//! * **Shifting.** A step can leave a basic var outside its bounds beyond
//!   its tolerance (the Harris pass lets the var that bounds the step leave
//!   its bound by up to its tolerance, an entry below the column's round-off
//!   floor bounds no step, and a recomputation of the values can land a var
//!   anywhere). Its bound is then shifted to just beyond where it sits, so
//!   the phase goes on over bounds its point satisfies instead of leaving
//!   the violation to a later phase.
//! * **Leaving where it sits.** A leaving var past the bound it leaves at
//!   (within its tolerance) would give a step backwards. The step is zero
//!   instead, and the var leaves where it sits (see `choose_pivot`): past a
//!   bound the phase moved, the bound moves on to it; past its true bound it
//!   stays there, at that bound.
//! * **Cleanup.** However the phase ends, the working bounds return to the
//!   true ones (see `end_primal_phase`), and `run_phases` leaves the
//!   violations this uncovers to the dual simplex.

use super::{
    random_fraction, BoundTols, Solver, Stream, VarState, NEAR_OPTIMAL_MAX_INFEASIBILITIES,
    NEAR_OPTIMAL_MAX_INFEASIBILITY,
};
use crate::Error;

impl Solver {
    /// The working bounds of `var`: its bounds as a running primal phase has
    /// moved them, its true bounds otherwise.
    #[inline]
    pub(super) fn working_bounds(&self, var: usize) -> (f64, f64) {
        if self.bounds_modified {
            (self.working_mins[var], self.working_maxs[var])
        } else {
            (self.orig_var_mins[var], self.orig_var_maxs[var])
        }
    }

    /// Set the working bounds of `var`, and of its row if it is basic. The
    /// first move of a phase takes the working bounds from the true ones.
    fn set_working_bounds(&mut self, var: usize, min: f64, max: f64) {
        if !self.bounds_modified {
            self.working_mins.clone_from(&self.orig_var_mins);
            self.working_maxs.clone_from(&self.orig_var_maxs);
            self.bounds_modified = true;
        }
        self.working_mins[var] = min;
        self.working_maxs[var] = max;
        if let VarState::Basic(row) = self.var_states[var] {
            self.basic_var_mins[row] = min;
            self.basic_var_maxs[row] = max;
        }
    }

    /// Start a primal phase: perturb the bounds ([`Self::perturb_bounds`])
    /// unless the phase is a cleanup (`cleanup`, see `run_phases`) or starts
    /// near optimal, shift the bound of every basic var outside its working
    /// bounds ([`Self::correct_primal`]), and start the cycling detection. A
    /// perturbation whose optimum is a few pivots away costs more in the
    /// cleanup than it saves, hence the near-optimal test.
    pub(super) fn begin_primal_phase(&mut self, cleanup: bool) {
        debug_assert!(!self.bounds_modified, "a primal phase is already running");
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
        let (num_primal_infeasible, _) = self.primal_infeasibility_extent();
        let near_optimal = num_dual_infeasible < NEAR_OPTIMAL_MAX_INFEASIBILITIES
            && max_dual_infeasibility < NEAR_OPTIMAL_MAX_INFEASIBILITY
            && num_primal_infeasible == 0;
        let perturbed = !cleanup && !near_optimal;
        if perturbed {
            self.perturb_bounds();
        }
        trace!(
            "primal phase start: {num_dual_infeasible} dual infeasibilities (largest \
             {max_dual_infeasibility:e}), {num_primal_infeasible} rows violated; bounds {}",
            if perturbed {
                "perturbed"
            } else if cleanup {
                "not perturbed (cleanup)"
            } else {
                "not perturbed (near optimal)"
            }
        );
        self.correct_primal();
        self.start_cycle_detection();
    }

    /// Perturb the working bounds: every finite bound moves outward by
    /// `r 5e-7 max(1, |b|)`, `r` a pseudo-random fraction of the var and `b`
    /// the bound in user units (for a slack, its row's right-hand side), and a
    /// non-basic var at a bound moves with it ([`Self::apply_flips`]). A fixed
    /// var is never perturbed, so that an equality ends on its right-hand
    /// side: its slack is at its value whenever it is at a bound, and leaves
    /// the basis at its true value even when a shift has moved its bounds
    /// (see `choose_pivot`).
    ///
    /// The amount is in user units, those of the engine's tolerances, mapped
    /// into scaled units: a user unit is `1 / s_j` of a structural var's
    /// scaled units and `r_i` of a slack's. Perturbed in scaled units
    /// instead, a slack whose row the equilibration scaled by `2^-20` would
    /// have its bound moved by millions of times its tolerance.
    pub(super) fn perturb_bounds(&mut self) {
        for var in 0..self.num_total_vars() {
            let (min, max) = (self.orig_var_mins[var], self.orig_var_maxs[var]);
            if min == max {
                continue;
            }
            // One user unit, and the magnitude of each bound, in scaled units.
            let (unit, min_mag, max_mag) = if var < self.num_vars {
                (1.0 / self.col_scales[var], min.abs(), max.abs())
            } else {
                let row = var - self.num_vars;
                let rhs = self.orig_rhs[row].abs();
                (self.row_scales[row], rhs, rhs)
            };
            let r = random_fraction(var, Stream::Perturbation);
            let amount = |mag: f64| r * BOUND_PERTURBATION * unit.max(mag);
            let perturbed_min = if min.is_finite() {
                min - amount(min_mag)
            } else {
                min
            };
            let perturbed_max = if max.is_finite() {
                max + amount(max_mag)
            } else {
                max
            };
            self.set_working_bounds(var, perturbed_min, perturbed_max);
        }
        let mut moves = Vec::new();
        for (c, (&var, state)) in self.nb_vars.iter().zip(&self.nb_var_states).enumerate() {
            let (min, max) = self.working_bounds(var);
            let target = if state.at_min {
                min
            } else if state.at_max {
                max
            } else {
                continue;
            };
            if target != self.nb_var_vals[c] {
                moves.push((c, target));
            }
        }
        self.apply_flips(&moves);
    }

    /// Shift the working bound of every basic var that lies outside it
    /// beyond its tolerance to `(1 + r)` times the tolerance beyond where the
    /// var sits (`r` a pseudo-random fraction of the var), so that the step
    /// it came from stands. Run after every step and every recomputation of
    /// the values, it keeps the invariant the ratio test rests on: no basic
    /// var lies beyond its working bounds by more than its tolerance. A value
    /// that is not a number stays, and an infinite one moves its bound to
    /// infinity: the ratio test reports either once its row is in the
    /// entering column, and the verification at the phase exit otherwise.
    pub(super) fn correct_primal(&mut self) {
        for r in 0..self.num_constraints() {
            let var = self.basic_vars[r];
            let val = self.basic_var_vals[r];
            let tol = self.var_tols[var];
            let (min, max) = (self.basic_var_mins[r], self.basic_var_maxs[r]);
            let (min, max) = if val < min - tol.lo {
                (
                    val - (1.0 + random_fraction(var, Stream::Shift)) * tol.lo,
                    max,
                )
            } else if val > max + tol.hi {
                (
                    min,
                    val + (1.0 + random_fraction(var, Stream::Shift)) * tol.hi,
                )
            } else {
                continue;
            };
            trace!(
                "primal phase: basic var {var} of row {r} at {val:e} is outside its bounds; \
                 working bounds now [{min:e}, {max:e}]"
            );
            self.set_working_bounds(var, min, max);
        }
    }

    /// Make the working bounds of the basic `var`, about to leave the basis
    /// at `val`, hold that value as the bound the non-basic var sits at.
    ///
    /// A var that leaves where it sits, past the bound it leaves at (a zero
    /// step, see `choose_pivot`), moves that bound on to `val` only if the
    /// phase moved it, for the cleanup to move it back: past its true bound it
    /// is within its tolerance, as the Harris pass allows, and stays there,
    /// at that bound ([`super::NonBasicVarState::at`]). A var fixed in the
    /// model is fixed at its value, whatever the phase did to its working
    /// bounds, when it leaves within its tolerance of that value, and at `val`
    /// otherwise.
    pub(super) fn hold_leaving_value(&mut self, var: usize, val: f64) {
        let (min, max) = (self.orig_var_mins[var], self.orig_var_maxs[var]);
        let (working_min, working_max) = self.working_bounds(var);
        if min == max {
            let tol = self.var_tols[var];
            let fixed_at = if (val - min).abs() <= tol.lo.min(tol.hi) {
                min
            } else {
                val
            };
            if (working_min, working_max) != (fixed_at, fixed_at) {
                self.set_working_bounds(var, fixed_at, fixed_at);
            }
        } else if val < working_min && working_min != min {
            self.set_working_bounds(var, val, working_max);
        } else if val > working_max && working_max != max {
            self.set_working_bounds(var, working_min, val);
        }
    }

    /// End a primal phase, however it ends: the working bounds return to the
    /// true ones, and the non-basic states, the objective and primal
    /// feasibility are taken against them, so that `run_phases`, verifying
    /// the point, sends a violation the cleanup uncovers to the dual simplex.
    ///
    /// A non-basic var at a bound the phase moved (perturbed, shifted, or
    /// moved to where the var left the basis) moves back to the true bound,
    /// and when one did, the basic values are recomputed from scratch, which
    /// also clears the round-off the steps carried into them. A non-basic var
    /// past a true bound it is at (one that left the basis where it sat)
    /// stays there if it is within its tolerance at the current values (a
    /// slack's stored tolerance may be stale and far too loose), and moves
    /// onto the bound otherwise: moving it back even then would push its
    /// offset, divided by the tableau entries, into the basic vars, which a
    /// small entry turns into a violation beyond their tolerances, and the
    /// dual phase would take the var off its bound again, round after round.
    pub(super) fn end_primal_phase(&mut self) -> Result<(), Error> {
        self.cycle_guard.stop();
        let mut moved = 0usize;
        for c in 0..self.nb_vars.len() {
            let var = self.nb_vars[c];
            let val = self.nb_var_vals[c];
            let (min, max) = (self.orig_var_mins[var], self.orig_var_maxs[var]);
            let (working_min, working_max) = self.working_bounds(var);
            let (at_min, at_max) = (self.nb_var_states[c].at_min, self.nb_var_states[c].at_max);
            let target = if val == working_min && working_min != min {
                min
            } else if val == working_max && working_max != max {
                max
            } else if at_min && val != min && !self.within_current_tol(var, val, min, false) {
                min
            } else if at_max && val != max && !self.within_current_tol(var, val, max, true) {
                max
            } else {
                continue;
            };
            self.nb_var_vals[c] = target;
            moved += 1;
        }
        if !self.bounds_modified && moved == 0 {
            return Ok(());
        }
        self.bounds_modified = false;
        for r in 0..self.num_constraints() {
            let var = self.basic_vars[r];
            self.basic_var_mins[r] = self.orig_var_mins[var];
            self.basic_var_maxs[r] = self.orig_var_maxs[var];
        }
        for c in 0..self.nb_vars.len() {
            self.nb_var_states[c] = self.nonbasic_state(self.nb_vars[c], self.nb_var_vals[c]);
        }
        if moved > 0 {
            trace!("primal phase cleanup: {moved} non-basic vars back to their bounds");
            self.recalc_basic_var_vals()?;
        }
        self.cur_obj_val = self.exact_obj_val();
        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        Ok(())
    }

    /// Whether the non-basic `var` at `val` is within its tolerance of its
    /// `bound` (the upper one if `up`) at the current values: a structural
    /// var's own tolerance, or a slack's derived afresh from its row
    /// ([`Self::current_slack_tol`]) and stored.
    fn within_current_tol(&mut self, var: usize, val: f64, bound: f64, up: bool) -> bool {
        let tol = if var < self.num_vars {
            self.var_tols[var].toward(up)
        } else {
            let tol = self.current_slack_tol(var - self.num_vars);
            self.var_tols[var] = BoundTols::uniform(tol);
            tol
        };
        (val - bound).abs() <= tol
    }
}

/// The bound perturbation base, in user units: a bound moves by up to this
/// much, or this fraction of the bound when its magnitude exceeds one.
const BOUND_PERTURBATION: f64 = 5e-7;
