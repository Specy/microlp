//! The bounded-variable simplex engine.
//!
//! [`Solver`] is the one long-lived LP instance a solve (LP or MILP) works on:
//! it holds the scaled problem, the current basis, and the pivot loops that
//! move between vertices. Everything a bound change or an added row does goes
//! through it; nothing above it addresses rows by index.
//!
//! The pieces it is built from live beside it:
//!
//! * [`tolerances`] — the tolerance model (magnitudes, floors, pivot limits).
//! * [`scaling`] — user problem to scaled representation, and vertex seeding.
//! * [`basis`] — the basis inverse ([`BasisSolver`]) and pivot descriptions.

mod basis;
mod scaling;
mod tolerances;

pub(crate) use basis::*;
pub(crate) use scaling::*;
pub(crate) use tolerances::*;

use core::time::Duration;

use crate::{
    helpers::{resized_view, to_dense},
    lu::{lu_factorize, ScratchSpace},
    sparse::{ScatteredVec, SparseVec},
    ComparisonOp, CsVec, Error, StopReason, VarDomain,
};
use sprs::CompressedStorage;

use web_time::Instant;

pub(crate) type Deadline = Option<Instant>;

type CsMat = sprs::CsMatI<f64, usize>;

#[inline]
pub(crate) fn check_deadline(deadline: &Deadline) -> StopReason {
    if let Some(dl) = deadline {
        if Instant::now() >= *dl {
            return StopReason::Limit;
        }
    }
    StopReason::Finished
}

#[derive(Clone)]
pub(crate) struct Solver {
    pub(crate) num_vars: usize,
    pub(crate) deadline: Deadline,
    /// Duration granted to each subsequent public pure-LP operation.
    pub(crate) operation_time_limit: Option<Duration>,
    /// Total number of simplex pivots performed across all solves/reoptimizes on this instance.
    pub(crate) lp_iterations: u64,
    /// Wall-clock time accumulated by public pure-LP operations.
    pub(crate) elapsed: Duration,

    /// Objective coefficients in the engine's scaled units: `s_j c_j` for a
    /// structural var, zero for a slack. The objective VALUE is invariant
    /// under scaling (`s_j c_j · x_j / s_j = c_j x_j`), so `cur_obj_val` is
    /// always in user units.
    orig_obj_coeffs: Vec<f64>,
    /// Bounds in scaled units: `lo_j / s_j` and `hi_j / s_j` for a structural
    /// var, the slack's own bounds for a slack.
    orig_var_mins: Vec<f64>,
    orig_var_maxs: Vec<f64>,
    pub(crate) orig_var_domains: Vec<VarDomain>,
    orig_constraints: CsMat, // excluding rhs
    orig_constraints_csc: CsMat,
    orig_rhs: Vec<f64>,
    /// Positive per-row equilibration factors. Every row is multiplied by
    /// these internally; validation multiplies its absolute user tolerance by
    /// the same factor so the public feasibility contract stays unscaled.
    row_scales: Vec<f64>,
    /// Positive power-of-two column factors `s_j` for the structural vars
    /// (see [`column_scales`]). Everything the engine stores about a
    /// structural var is in the scaled units `x'_j = x_j / s_j`; every value
    /// or bound that crosses the `Solver` boundary is converted, so callers
    /// only ever see user units.
    col_scales: Vec<f64>,
    /// The user's absolute feasibility tolerance
    /// ([`crate::Tolerances::feasibility`]), in user units.
    feasibility: f64,
    /// Per total var, the absolute tolerance (in the engine's scaled units)
    /// within which the var counts as at a bound or within its bounds:
    /// [`structural_tol`] for a structural var (refreshed when its bounds
    /// change) and [`slack_tol`] for a slack (refreshed whenever the rows are
    /// checked, since its round-off floor depends on the values).
    var_tols: Vec<f64>,
    /// Per total var, the tolerance within which its reduced cost counts as
    /// zero ([`dual_tol`]); refreshed whenever the reduced costs are
    /// recomputed from the multipliers.
    dual_tols: Vec<f64>,
    /// Per structural var, the smallest [`snap_budget`] over its rows (see
    /// [`structural_tol`]); infinite for a var in no row. Static apart from
    /// rows added later.
    row_budgets: Vec<f64>,

    enable_primal_steepest_edge: bool,
    enable_dual_steepest_edge: bool,

    /// While the initial dual phase runs on the phase-1 artificial objective
    /// (the start being neither primal nor dual feasible), its cost vector
    /// (one entry per total var), so that any recomputation of the reduced
    /// costs during that phase uses the same objective the pivots priced.
    /// `None` once the real objective is in force.
    artificial_obj: Option<Vec<f64>>,

    is_primal_feasible: bool,
    is_dual_feasible: bool,

    /// Whether the basic values have been changed incrementally (pivots,
    /// bound shifts) since [`Self::verify_primal`] last checked them against
    /// the rows; when not, a phase exit needs no residual check and no
    /// refinement. A recompute through the factorization does NOT clear it:
    /// recomputed is not verified (see [`Self::recalc_basic_var_vals`]).
    values_dirty: bool,
    /// Whether the reduced costs have been updated incrementally since they
    /// were last recomputed exactly; when not, a phase exit needs no
    /// recomputation.
    reduced_costs_dirty: bool,
    /// Pivots since the basic values were last checked against the rows at
    /// a refactorization (see [`Self::refactorize`]).
    pivots_since_drift_check: usize,

    // Updated on each pivot
    /// For each var: whether it is basic/non-basic and the corresponding index.
    var_states: Vec<VarState>,
    basis_solver: BasisSolver,

    /// For each constraint the corresponding basic var.
    basic_vars: Vec<usize>,
    basic_var_vals: Vec<f64>,
    basic_var_mins: Vec<f64>,
    basic_var_maxs: Vec<f64>,
    dual_edge_sq_norms: Vec<f64>,

    /// Remaining variables. (idx -> var), 'nb' means 'non-basic'
    nb_vars: Vec<usize>,
    nb_var_obj_coeffs: Vec<f64>,
    nb_var_vals: Vec<f64>,
    nb_var_states: Vec<NonBasicVarState>,
    primal_edge_sq_norms: Vec<f64>,

    /// For each structural var fixed by [`Self::fix_var`], the bounds it had
    /// before the fix, so [`Self::unfix_var`] can restore them. A fix is a
    /// plain bound change to `[val, val]`; this is the only extra state.
    fixed_var_bounds: Vec<Option<(f64, f64)>>,

    pub(crate) cur_obj_val: f64,

    // Recomputed on each pivot
    col_coeffs: SparseVec,
    sq_norms_update_helper: Vec<f64>,
    inv_basis_row_coeffs: SparseVec,
    row_coeffs: ScatteredVec,
}

#[derive(Clone, Debug)]
enum VarState {
    Basic(usize),
    NonBasic(usize),
}

#[derive(Clone, Debug)]
struct NonBasicVarState {
    at_min: bool,
    at_max: bool,
}

/// Status of one variable in a simplex basis snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum VarStatus {
    Basic,
    AtLower,
    AtUpper,
    /// Non-basic free variable (both bounds infinite), pinned at 0.
    Free,
}

/// A compact simplex basis: one status per total var (structural + slack).
/// Together with the current variable bounds it fully determines a vertex.
#[derive(Clone, Debug)]
pub(crate) struct Basis(pub(crate) Vec<VarStatus>);

impl std::fmt::Debug for Solver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Solver")?;
        writeln!(
            f,
            "num_vars: {}, num_constraints: {}, is_primal_feasible: {}, is_dual_feasible: {}",
            self.num_vars,
            self.num_constraints(),
            self.is_primal_feasible,
            self.is_dual_feasible,
        )?;
        writeln!(f, "orig_obj_coeffs:\n{:?}", self.orig_obj_coeffs)?;
        writeln!(f, "orig_var_mins:\n{:?}", self.orig_var_mins)?;
        writeln!(f, "orig_var_maxs:\n{:?}", self.orig_var_maxs)?;
        writeln!(f, "orig_constraints:")?;
        for row in self.orig_constraints.outer_iterator() {
            writeln!(f, "{:?}", to_dense(&row))?;
        }
        writeln!(f, "orig_rhs:\n{:?}", self.orig_rhs)?;
        writeln!(f, "basic_vars:\n{:?}", self.basic_vars)?;
        writeln!(f, "basic_var_vals:\n{:?}", self.basic_var_vals)?;
        writeln!(f, "dual_edge_sq_norms:\n{:?}", self.dual_edge_sq_norms)?;
        writeln!(f, "nb_vars:\n{:?}", self.nb_vars)?;
        writeln!(f, "nb_var_vals:\n{:?}", self.nb_var_vals)?;
        writeln!(f, "nb_var_obj_coeffs:\n{:?}", self.nb_var_obj_coeffs)?;
        writeln!(f, "primal_edge_sq_norms:\n{:?}", self.primal_edge_sq_norms)?;
        writeln!(f, "cur_obj_val: {:?}", self.cur_obj_val)?;
        Ok(())
    }
}

impl Solver {
    pub(crate) fn try_new(
        obj_coeffs: &[f64],
        var_mins: &[f64],
        var_maxs: &[f64],
        constraints: &[(CsVec, ComparisonOp, f64)],
        var_domains: &[VarDomain],
        deadline: Deadline,
        feasibility: f64,
    ) -> Result<Self, Error> {
        let enable_steepest_edge = true; // TODO: make user-settable.

        let num_vars = obj_coeffs.len();

        assert_eq!(num_vars, var_mins.len());
        assert_eq!(num_vars, var_maxs.len());
        for v in 0..num_vars {
            if var_mins[v].is_nan() || var_maxs[v].is_nan() || var_mins[v] > var_maxs[v] {
                return Err(Error::Infeasible);
            }
        }

        // Everything below works in scaled units: x'_j = x_j / s_j.
        let col_scales = column_scales(obj_coeffs, var_mins, var_maxs, constraints);
        let mut orig_obj_coeffs: Vec<f64> = obj_coeffs
            .iter()
            .zip(&col_scales)
            .map(|(&c, &s)| c * s)
            .collect();
        let mut orig_var_mins: Vec<f64> = var_mins
            .iter()
            .zip(&col_scales)
            .map(|(&b, &s)| b / s)
            .collect();
        let mut orig_var_maxs: Vec<f64> = var_maxs
            .iter()
            .zip(&col_scales)
            .map(|(&b, &s)| b / s)
            .collect();
        // Rows in scaled units, prepared up front so that the integer vars'
        // rounding budgets are known before the tolerances are derived.
        let is_fixed = |var: usize| var_mins[var] == var_maxs[var];
        let mut prepared_rows = Vec::with_capacity(constraints.len());
        for (coeffs, cmp_op, rhs) in constraints {
            let coeffs = scale_columns(coeffs.clone(), &col_scales);
            if let Some(row) = prepare_row(coeffs, *cmp_op, *rhs, is_fixed)? {
                prepared_rows.push(row);
            }
        }
        let mut row_budgets = vec![f64::INFINITY; num_vars];
        for row in &prepared_rows {
            for (var, &coeff) in row.coeffs.iter() {
                row_budgets[var] =
                    row_budgets[var].min(snap_budget(feasibility, row.row_scale, coeff));
            }
        }
        let mut var_tols: Vec<f64> = (0..num_vars)
            .map(|v| {
                structural_tol(
                    col_scales[v],
                    orig_var_mins[v],
                    orig_var_maxs[v],
                    feasibility,
                    row_budgets[v],
                )
            })
            .collect();

        let mut var_states = vec![];

        let mut nb_vars = vec![];
        let mut nb_var_vals = vec![];
        let mut nb_var_states = vec![];

        let mut obj_val = 0.0;

        let mut is_dual_feasible = true;

        for v in 0..num_vars {
            // choose initial variable values

            let min = orig_var_mins[v];
            let max = orig_var_maxs[v];
            let tol = var_tols[v];

            // initially all user-created variables are non-basic
            var_states.push(VarState::NonBasic(nb_vars.len()));
            nb_vars.push(v);

            // Choose an initial value, preferring a bound that keeps this
            // variable's reduced cost dual-feasible.
            let (init_val, var_dual_feasible) = if at_bound(min, max, tol) {
                // Fixed variable: the obj. coeff doesn't matter.
                (min, true)
            } else {
                initial_nonbasic_value(orig_obj_coeffs[v], min, max)
            };
            if !var_dual_feasible {
                is_dual_feasible = false;
            }

            nb_var_vals.push(init_val);
            obj_val += init_val * orig_obj_coeffs[v];

            nb_var_states.push(NonBasicVarState {
                at_min: at_bound(init_val, min, tol),
                at_max: at_bound(init_val, max, tol),
            });
        }

        let mut constraint_coeffs = vec![];
        let mut orig_rhs = vec![];
        let mut row_scales = vec![];

        // Initially, all slack vars are basic.
        let mut basic_vars = vec![];
        let mut basic_var_vals = vec![];
        let mut basic_var_mins = vec![];
        let mut basic_var_maxs = vec![];

        for PreparedRow {
            coeffs,
            rhs,
            row_scale,
            slack_var_min,
            slack_var_max,
        } in prepared_rows
        {
            constraint_coeffs.push(coeffs.clone());
            orig_rhs.push(rhs);
            row_scales.push(row_scale);

            orig_var_mins.push(slack_var_min);
            orig_var_maxs.push(slack_var_max);

            basic_var_mins.push(slack_var_min);
            basic_var_maxs.push(slack_var_max);

            let cur_slack_var = var_states.len();
            var_states.push(VarState::Basic(basic_vars.len()));
            basic_vars.push(cur_slack_var);

            let mut lhs_val = 0.0;
            let mut magnitude = rhs.abs();
            for (var, &coeff) in coeffs.iter() {
                lhs_val += coeff * nb_var_vals[var];
                magnitude += (coeff * nb_var_vals[var]).abs();
            }
            basic_var_vals.push(rhs - lhs_val);
            var_tols.push(slack_tol(feasibility, row_scale, magnitude));
        }

        let num_constraints = constraint_coeffs.len();
        let num_total_vars = num_vars + num_constraints;

        orig_obj_coeffs.resize(num_total_vars, 0.0);

        let mut orig_constraints = CsMat::empty(CompressedStorage::CSR, num_total_vars);
        for (cur_slack_var, coeffs) in constraint_coeffs.into_iter().enumerate() {
            let mut coeffs = into_resized(coeffs, num_total_vars);
            coeffs.append(num_vars + cur_slack_var, 1.0);
            orig_constraints = orig_constraints.append_outer_csvec(coeffs.view());
        }
        let orig_constraints_csc = orig_constraints.to_csc();

        let is_primal_feasible = basic_var_vals
            .iter()
            .zip(&basic_var_mins)
            .zip(&basic_var_maxs)
            .all(|((&val, &min), &max)| val >= min && val <= max);

        let need_artificial_obj = !is_primal_feasible && !is_dual_feasible;

        let enable_dual_steepest_edge = enable_steepest_edge;
        let dual_edge_sq_norms = if enable_dual_steepest_edge {
            vec![1.0; basic_vars.len()]
        } else {
            vec![]
        };

        // If is dual feasible at start, we don't need lengthy primal phase2.
        // Thus we can skip expensive calculations for primal sq. norms.
        let enable_primal_steepest_edge = enable_steepest_edge && !is_dual_feasible;
        let sq_norms_update_helper = if enable_primal_steepest_edge {
            vec![0.0; num_total_vars - num_constraints]
        } else {
            vec![]
        };

        // Phase-1 artificial objective: push every non-basic var towards the
        // bound it starts at (cost +1 at a lower bound, -1 at an upper bound,
        // 0 if free or fixed), so the slack basis is dual feasible for it.
        let artificial_obj = need_artificial_obj.then(|| {
            let mut costs = vec![0.0; num_total_vars];
            for (&var, state) in nb_vars.iter().zip(&nb_var_states) {
                costs[var] = if state.at_min && !state.at_max {
                    1.0
                } else if state.at_max && !state.at_min {
                    -1.0
                } else {
                    0.0
                };
            }
            costs
        });
        let costs = artificial_obj.as_deref().unwrap_or(&orig_obj_coeffs);

        let mut nb_var_obj_coeffs = vec![];
        let mut primal_edge_sq_norms = vec![];
        for &var in &nb_vars {
            //guaranteed to be a valid index
            let col = orig_constraints_csc.outer_view(var).unwrap();

            nb_var_obj_coeffs.push(costs[var]);

            if enable_primal_steepest_edge {
                primal_edge_sq_norms.push(col.squared_l2_norm() + 1.0);
            }
        }

        let cur_obj_val = if need_artificial_obj { 0.0 } else { obj_val };

        // The slack basis has zero multipliers, so a reduced cost is just the
        // objective coefficient and its round-off floor follows from that.
        let dual_tols = orig_obj_coeffs.iter().map(|c| dual_tol(c.abs())).collect();

        let mut scratch = ScratchSpace::with_capacity(num_constraints);
        let lu_factors = lu_factorize(
            basic_vars.len(),
            |c| {
                orig_constraints_csc
                    .outer_view(basic_vars[c])
                    //guaranteed to be a valid index
                    .unwrap()
                    .into_raw_storage()
            },
            LU_STABILITY_THRESHOLD,
            &mut scratch,
        )?;
        let lu_factors_transp = lu_factors.transpose();

        let res = Self {
            num_vars,
            orig_obj_coeffs,
            orig_var_mins,
            orig_var_maxs,
            orig_constraints,
            orig_constraints_csc,
            orig_rhs,
            row_scales,
            col_scales,
            feasibility,
            var_tols,
            dual_tols,
            row_budgets,
            deadline,
            operation_time_limit: None,
            lp_iterations: 0,
            elapsed: Duration::ZERO,
            orig_var_domains: var_domains.to_vec(),
            enable_primal_steepest_edge,
            enable_dual_steepest_edge,
            artificial_obj,
            is_primal_feasible,
            is_dual_feasible,
            values_dirty: false,
            reduced_costs_dirty: false,
            pivots_since_drift_check: 0,
            var_states,
            basis_solver: BasisSolver {
                lu_factors,
                lu_factors_transp,
                scratch,
                eta_matrices: EtaMatrices::new(num_constraints),
                rhs: ScatteredVec::empty(num_constraints),
            },
            basic_vars,
            basic_var_vals,
            basic_var_mins,
            basic_var_maxs,
            dual_edge_sq_norms,
            nb_vars,
            nb_var_obj_coeffs,
            nb_var_vals,
            nb_var_states,
            primal_edge_sq_norms,
            fixed_var_bounds: vec![None; num_vars],
            cur_obj_val,
            col_coeffs: SparseVec::new(),
            sq_norms_update_helper,
            inv_basis_row_coeffs: SparseVec::new(),
            row_coeffs: ScatteredVec::empty(num_total_vars - num_constraints),
        };

        debug!(
            "initialized solver: vars: {}, constraints: {}, primal feasible: {}, dual feasible: {}, nnz: {}",
            res.num_vars,
            res.orig_constraints.rows(),
            res.is_primal_feasible,
            res.is_dual_feasible,
            res.orig_constraints.nnz(),
        );

        Ok(res)
    }

    /// A var's current value in user units: the value that is reported.
    pub(crate) fn get_value(&self, var: usize) -> f64 {
        let internal = if var < self.num_vars {
            self.reported_internal_value(var)
        } else {
            self.internal_value(var)
        };
        internal * self.col_scale(var)
    }

    /// The structural values to report, in user units: the point the last
    /// verification checked, since [`Self::verify_primal`] evaluates the rows
    /// at exactly these values.
    pub(crate) fn reported_values(&self) -> Vec<f64> {
        (0..self.num_vars).map(|v| self.get_value(v)).collect()
    }

    /// The column factor of `var`: `s_j` for a structural var, one for a slack.
    #[inline]
    fn col_scale(&self, var: usize) -> f64 {
        if var < self.num_vars {
            self.col_scales[var]
        } else {
            1.0
        }
    }

    /// A var's current value in the engine's scaled units.
    #[inline]
    fn internal_value(&self, var: usize) -> f64 {
        match self.var_states[var] {
            VarState::Basic(idx) => self.basic_var_vals[idx],
            VarState::NonBasic(idx) => self.nb_var_vals[idx],
        }
    }

    /// A structural var's value as it is reported, in scaled units: a FIXED
    /// var (`lo == hi`) is reported at its bound, not at the value the basis
    /// gives it. The simplex prices such a var out, but one that is basic
    /// (from a warm start, or fixed by a bound change while basic) still gets
    /// its value from `B^-1 (b - N x_N)` and may drift within its tolerance.
    /// The rest of the crate rests on a fixed var being EXACT: presolve drops
    /// rows and substitutes fixed vars out of the rows it keeps on that
    /// basis. The bound is also the more accurate answer; nothing about it
    /// was ever in question. The snap is the same move as rounding an
    /// integer var, and [`structural_tol`] caps a fixed var's drift by what
    /// its rows can absorb, so the reported point still meets the contract
    /// the engine verified at the basis values ([`Self::check_rows`]). The
    /// engine's own bookkeeping never snaps: the basis equations hold at the
    /// basis values, and a var fixed by a bound change while basic is moved
    /// onto its value by the dual simplex like any other bound violation.
    #[inline]
    fn reported_internal_value(&self, var: usize) -> f64 {
        if self.orig_var_mins[var] == self.orig_var_maxs[var] {
            self.orig_var_mins[var]
        } else {
            self.internal_value(var)
        }
    }

    /// The activity `Σ a_rj x_j` of row `r` over the structural vars at the
    /// scaled values `x`, and the magnitude `|b_r| + Σ|a_rj x_j|` it was
    /// computed from. This is THE evaluation of a row: the engine derives its
    /// basic slacks from it ([`Self::check_rows`]) and the checks at the
    /// boundary evaluate the reported point through it, term for term in the
    /// same order, so all of them see the same bits.
    fn row_activity(&self, r: usize, x: impl Fn(usize) -> f64) -> (f64, f64) {
        //guaranteed to be a valid index
        let row = self.orig_constraints.outer_view(r).unwrap();
        let mut lhs = 0.0;
        let mut magnitude = self.orig_rhs[r].abs();
        for (v, &coeff) in row.iter() {
            if v < self.num_vars {
                let term = coeff * x(v);
                lhs += term;
                magnitude += term.abs();
            }
        }
        (lhs, magnitude)
    }

    /// The slack tolerance of row `r` at the current values (see
    /// [`slack_tol`]).
    fn current_slack_tol(&self, r: usize) -> f64 {
        let (_, magnitude) = self.row_activity(r, |v| self.internal_value(v));
        slack_tol(self.feasibility, self.row_scales[r], magnitude)
    }

    /// Re-derive every reduced-cost tolerance from the magnitudes of the
    /// terms `c_j - sum_i a_ij y_i` was computed from (see [`dual_tol`]).
    fn refresh_dual_tols(&mut self, multipliers: &[f64]) {
        for var in 0..self.num_total_vars() {
            //guaranteed to be a valid index
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let magnitude = self.orig_obj_coeffs[var].abs()
                + col
                    .iter()
                    .map(|(r, coeff)| (coeff * multipliers[r]).abs())
                    .sum::<f64>();
            self.dual_tols[var] = dual_tol(magnitude);
        }
    }

    /// The first ORIGINAL row that `values` (one entry per structural var,
    /// user units) violates beyond the contract, with the violation in user
    /// units. A row's sense is encoded by its slack var's bounds
    /// (`lhs + s = rhs` with `s` in `[smin, smax]`); slack bounds are never
    /// touched by branching, so this always reflects the user's rows.
    ///
    /// This is the evaluation the engine verified before it reported the
    /// point ([`Self::check_rows`]): the row's activity through
    /// [`Self::row_activity`], the slack it implies, and [`outside`] against
    /// the slack's bounds within [`row_tolerance`] of the user's absolute
    /// tolerance in the row's scaled units. Multiplying the tolerance by the
    /// row's power-of-two factor is exact, so this is the check on the
    /// unscaled user row. The tolerance is deliberately NOT relative to the
    /// row's coefficients: the big-M trap is a violation that is tiny
    /// relative to huge coefficients (`5.0` on a `1e9`-scale row) and
    /// decisive in absolute terms. The round-off floor is relative to the
    /// row's ACTIVITY instead: what evaluating the row in `f64` can resolve
    /// at all.
    pub(crate) fn first_violated_row(&self, values: &[f64]) -> Option<(usize, f64)> {
        for r in 0..self.num_constraints() {
            // `values` are user units; the stored column is scaled by `s_v`,
            // and dividing by the power of two is exact, so the product
            // `coeff * (values[v] / s_v)` is the user term times the row's
            // factor.
            let (lhs, magnitude) = self.row_activity(r, |v| values[v] / self.col_scales[v]);
            let slack = self.num_vars + r;
            let (smin, smax) = (self.orig_var_mins[slack], self.orig_var_maxs[slack]);
            let implied = self.orig_rhs[r] - lhs;
            let tol = row_tolerance(self.feasibility * self.row_scales[r], magnitude);
            if outside(implied, smin, smax, tol) {
                return Some((r, excursion(implied, smin, smax) / self.row_scales[r]));
            }
        }
        None
    }

    /// The first structural var whose value (user units) violates its bounds
    /// beyond the contract, the user tolerance floored at the round-off of
    /// the bound ([`bound_tolerance`]), with the violation in user units.
    pub(crate) fn first_violated_bound(&self, values: &[f64]) -> Option<(usize, f64)> {
        values.iter().enumerate().find_map(|(v, &x)| {
            let (lo, hi) = self.get_var_bounds(v);
            let tol = bound_tolerance(self.feasibility, lo, hi);
            outside(x, lo, hi, tol).then(|| (v, excursion(x, lo, hi)))
        })
    }

    /// Objective value (internal minimize space) of an explicit structural-var
    /// value vector.
    pub(crate) fn objective_of(&self, values: &[f64]) -> f64 {
        values
            .iter()
            .enumerate()
            .map(|(v, &x)| (self.orig_obj_coeffs[v] / self.col_scales[v]) * x)
            .sum()
    }

    /// A var's bounds in user units.
    pub(crate) fn get_var_bounds(&self, var: usize) -> (f64, f64) {
        let s = self.col_scale(var);
        (self.orig_var_mins[var] * s, self.orig_var_maxs[var] * s)
    }

    /// Change a variable's bounds (user units) in place. Records the new bounds
    /// and repairs the invariants that depend on them; does NOT run simplex —
    /// call [`Self::reoptimize`] afterwards. Returns `Err(Infeasible)` with
    /// state untouched if either bound is NaN or `min > max`.
    pub(crate) fn set_var_bounds(&mut self, var: usize, min: f64, max: f64) -> Result<(), Error> {
        if min.is_nan() || max.is_nan() || min > max {
            return Err(Error::Infeasible);
        }
        // Dividing by a power of two is exact, so the user's bound maps to
        // exactly one internal bound.
        let s = self.col_scale(var);
        let (min, max) = (min / s, max / s);
        self.orig_var_mins[var] = min;
        self.orig_var_maxs[var] = max;
        if var < self.num_vars {
            self.var_tols[var] =
                structural_tol(s, min, max, self.feasibility, self.row_budgets[var]);
        }
        let tol = self.var_tols[var];
        match self.var_states[var] {
            VarState::Basic(row) => {
                self.basic_var_mins[row] = min;
                self.basic_var_maxs[row] = max;
                let val = self.basic_var_vals[row];
                if val < min - tol || val > max + tol {
                    self.is_primal_feasible = false;
                }
            }
            VarState::NonBasic(col) => {
                let cur = self.nb_var_vals[col];
                let new_val = cur.clamp(min, max);
                if new_val != cur {
                    // Shift the non-basic var to the nearest bound and propagate the
                    // delta into basic values (same mechanism as fix_var's non-basic arm).
                    self.calc_col_coeffs(col);
                    let diff = new_val - cur;
                    for (r, coeff) in self.col_coeffs.iter() {
                        self.basic_var_vals[r] -= diff * coeff;
                    }
                    self.cur_obj_val += diff * self.nb_var_obj_coeffs[col];
                    self.nb_var_vals[col] = new_val;
                    self.is_primal_feasible = false;
                    self.values_dirty = true;
                }
                self.nb_var_states[col] = NonBasicVarState {
                    at_min: at_bound(new_val, min, tol),
                    at_max: at_bound(new_val, max, tol),
                };
                // A var at a loosened bound may no longer justify its reduced cost.
                let dual_tol = self.dual_tols[var];
                self.is_dual_feasible = self.is_dual_feasible
                    && (self.nb_var_states[col].at_min && self.nb_var_obj_coeffs[col] > -dual_tol
                        || self.nb_var_states[col].at_max
                            && self.nb_var_obj_coeffs[col] < dual_tol
                        || self.nb_var_obj_coeffs[col].abs() < dual_tol);
            }
        }
        Ok(())
    }

    /// Re-solve after bound changes or a basis load: dual simplex to restore primal
    /// feasibility, then primal simplex if reduced costs became dual-infeasible
    /// (only happens after loosening bounds or a numerically imperfect basis load).
    pub(crate) fn reoptimize(&mut self) -> Result<StopReason, Error> {
        self.run_phases()
    }

    /// Alternate the dual (primal-feasibility) and primal (optimality) phases
    /// until a state passes verification: the basic values recomputed from
    /// the factorization satisfy the rows and their bounds, and the reduced
    /// costs recomputed from the multipliers are dual feasible. Each phase's
    /// own exit test runs on incrementally updated numbers, so a phase can end
    /// on drifted values; only a verified state is reported as `Finished`.
    fn run_phases(&mut self) -> Result<StopReason, Error> {
        for round in 0..MAX_PHASE_ROUNDS {
            if !self.is_primal_feasible && self.restore_feasibility()? == StopReason::Limit {
                return Ok(StopReason::Limit);
            }
            if !self.is_dual_feasible {
                // Leaving the dual phase for the primal one: the real
                // objective is in force from here on.
                self.artificial_obj = None;
                self.recalc_obj_coeffs()?;
                if self.optimize()?.0 == StopReason::Limit {
                    return Ok(StopReason::Limit);
                }
            }
            if !self.verify_primal()? {
                debug!("phase round {round}: recomputed values violate a bound; continuing");
                continue;
            }
            // Dual verification: exact reduced costs. If they still show an
            // infeasibility, let the primal simplex act on them; a column it
            // cannot exploit (see `PivotChoice::Unexploitable`) is not an
            // improving direction, so no pivot at all means the vertex is
            // optimal within tolerance.
            if self.artificial_obj.is_some() || self.reduced_costs_dirty {
                self.artificial_obj = None;
                self.recalc_obj_coeffs()?;
            }
            if self.calc_dual_infeasibility().0 == 0 {
                self.is_dual_feasible = true;
                return Ok(StopReason::Finished);
            }
            debug!("phase round {round}: recomputed reduced costs are dual infeasible; continuing");
            let (stop, pivoted) = self.optimize()?;
            if stop == StopReason::Limit {
                return Ok(StopReason::Limit);
            }
            if !pivoted && self.verify_primal()? {
                return Ok(StopReason::Finished);
            }
        }
        Err(Error::InternalError(format!(
            "simplex did not reach a verified optimum in {MAX_PHASE_ROUNDS} phase rounds"
        )))
    }

    /// Whether the current factorization was built for the current basis with
    /// no pivots applied on top of it.
    fn factorization_is_fresh(&self) -> bool {
        self.basis_solver.eta_matrices.len() == 0
    }

    /// Refactorize the basis and recompute everything derived from it: the
    /// basic values, the reduced costs and the objective. Used where the
    /// current numbers themselves are in doubt (a pivot disagreement, a stall
    /// before declaring infeasibility), so nothing is gated.
    fn rebuild(&mut self) -> Result<(), Error> {
        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars)?;
        self.recalc_basic_var_vals()?;
        self.recalc_obj_coeffs()
    }

    /// The periodic refactorization: a fresh factorization, and the basic
    /// values recomputed from it only when their residuals show they have
    /// drifted out of the rows' tolerances. A point that satisfies every row
    /// within tolerance is a valid state of the engine, and replacing it by
    /// the exact basic solution of the rounded data would not make it
    /// better: at the boundary of a bound it can make it worse, moving a var
    /// by a round-off amount the rows allow but its own tolerance does not.
    /// The reduced costs are left to their incremental updates; the phase
    /// exit recomputes them exactly before anything is reported.
    fn refactorize(&mut self) -> Result<(), Error> {
        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars)?;
        // The residual check costs a pass over the matrix; on a small basis
        // the factorization is renewed almost every pivot, so the check is
        // made once per full turnover of the basis (one pivot per row). The
        // phase exit always checks.
        if self.pivots_since_drift_check >= self.num_constraints().max(1) {
            self.pivots_since_drift_check = 0;
            let mut residuals = Vec::with_capacity(self.num_constraints());
            if !self.check_rows(&mut residuals) {
                self.recalc_basic_var_vals()?;
            }
        }
        Ok(())
    }

    /// Bring the rows and the values into agreement and measure what is left.
    /// Every BASIC slack is re-derived from its row, `s = b - Σ a_j x_j` at
    /// the current structural values, through the very evaluation the checks
    /// at the boundary repeat ([`Self::row_activity`]): a row's violation
    /// then lives in its slack's value, where the dual simplex sees it and
    /// where the guard will find the same bits (up to the snap of a fixed or
    /// integer var, which stays within the row's reserve). The slack
    /// tolerances are refreshed from the rows' magnitudes at these values.
    /// `residuals` receives `b - Σ a_j x_j - s` for every row: zero by
    /// construction where the slack is basic, so it measures how far the
    /// basic STRUCTURAL values are from solving the rows whose slack sits at
    /// a bound. Returns whether every residual is within its row's tolerance.
    fn check_rows(&mut self, residuals: &mut Vec<f64>) -> bool {
        residuals.clear();
        let mut hold = true;
        for r in 0..self.num_constraints() {
            let (lhs, magnitude) = self.row_activity(r, |v| self.internal_value(v));
            let implied = self.orig_rhs[r] - lhs;
            let slack = self.num_vars + r;
            let residual = match self.var_states[slack] {
                VarState::Basic(row) => {
                    self.basic_var_vals[row] = implied;
                    0.0
                }
                VarState::NonBasic(col) => implied - self.nb_var_vals[col],
            };
            let tol = slack_tol(self.feasibility, self.row_scales[r], magnitude);
            self.var_tols[slack] = tol;
            // A NaN residual holds nothing.
            if residual.is_nan() || residual.abs() > tol {
                hold = false;
            }
            residuals.push(residual);
        }
        hold
    }

    /// Make the values satisfy the rows to within their tolerances and
    /// measure primal feasibility on them. The rows are evaluated through
    /// [`Self::check_rows`], the same evaluation the boundary repeats on the
    /// reported point, so a `Finished` state and the point handed out agree
    /// bit for bit up to the snap of fixed and integer vars. The incrementally updated
    /// values are checked first (an O(nnz) pass); only if the basic
    /// structural values have drifted out of the rows' tolerances are they
    /// refined through the factorization, then through a fresh one. A
    /// residual that survives both means the basis is too ill-conditioned to
    /// represent its own vertex, which is reported loudly rather than as an
    /// optimum. Values within tolerance are not refined: the contract asks no
    /// more, and moving last bits steers the branch & bound. The objective is
    /// recomputed from the values so it always matches them. Returns whether
    /// the values are within their bounds (and sets `is_primal_feasible`
    /// accordingly).
    fn verify_primal(&mut self) -> Result<bool, Error> {
        if !self.values_dirty {
            // Verified values that nothing has touched since; only the
            // bounds may have changed.
            self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
            return Ok(self.is_primal_feasible);
        }
        let mut residuals = Vec::with_capacity(self.num_constraints());
        let mut hold = self.check_rows(&mut residuals);
        if !hold {
            hold = self.refine(&mut residuals);
        }
        if !hold {
            debug!("verify: residual persists through the factorization; refactorizing");
            self.basis_solver
                .reset(&self.orig_constraints_csc, &self.basic_vars)?;
            self.recalc_basic_var_vals()?;
            hold = self.check_rows(&mut residuals) || self.refine(&mut residuals);
            if !hold {
                return Err(Error::InternalError(
                    "basis too ill-conditioned: recomputed values do not satisfy the rows"
                        .to_string(),
                ));
            }
        }
        self.values_dirty = false;
        self.cur_obj_val = self.exact_obj_val();
        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        Ok(self.is_primal_feasible)
    }

    /// Iterative refinement of the basic values given the current
    /// `residuals` (see [`Self::verify_primal`]): solve `B delta = residual`
    /// and correct, up to [`REFINEMENT_STEPS`] times or until the rows hold.
    /// Unlike recomputing the values outright, the correction is accurate
    /// even through a poor eta file, because a solve's round-off scales with
    /// its right-hand side and the residual is tiny. Returns whether the
    /// rows hold afterwards.
    fn refine(&mut self, residuals: &mut Vec<f64>) -> bool {
        for step in 0..REFINEMENT_STEPS {
            self.basis_solver.solve_dense_with_etas(residuals);
            for (val, delta) in self.basic_var_vals.iter_mut().zip(residuals.iter()) {
                *val += delta;
            }
            let hold = self.check_rows(residuals);
            debug!(
                "verify: refinement step {} ({})",
                step + 1,
                if hold {
                    "rows hold"
                } else {
                    "rows still violated"
                }
            );
            if hold {
                return true;
            }
        }
        false
    }

    /// The objective of the current values under the cost vector in force
    /// (the artificial one during phase 1).
    fn exact_obj_val(&self) -> f64 {
        let costs = self
            .artificial_obj
            .as_deref()
            .unwrap_or(&self.orig_obj_coeffs);
        let mut obj = 0.0;
        for (r, &var) in self.basic_vars.iter().enumerate() {
            obj += costs[var] * self.basic_var_vals[r];
        }
        for (c, &var) in self.nb_vars.iter().enumerate() {
            obj += costs[var] * self.nb_var_vals[c];
        }
        obj
    }

    /// Whether the pivot element computed through the row and through the
    /// column agree: to [`PIVOT_AGREEMENT_TOL`] relative to the element, or
    /// to the round-off of the computations that produced it, `scale` being
    /// the largest entry of the tableau row and column involved (a genuine
    /// entry far below its neighbours is computed with an absolute error of
    /// that order and can never agree relatively).
    fn pivots_agree(a_row: f64, a_col: f64, scale: f64) -> bool {
        let diff = (a_row - a_col).abs();
        diff <= PIVOT_AGREEMENT_TOL * a_row.abs().max(a_col.abs()) || diff <= ENTRY_ROUNDOFF * scale
    }

    /// The entry of the current pivot column at `row`.
    fn col_coeff_at(&self, row: usize) -> f64 {
        self.col_coeffs
            .iter()
            .find(|&(r, _)| r == row)
            .map_or(0.0, |(_, &coeff)| coeff)
    }

    pub(crate) fn snapshot_basis(&self) -> Basis {
        let mut statuses = Vec::with_capacity(self.num_total_vars());
        for var in 0..self.num_total_vars() {
            statuses.push(match self.var_states[var] {
                VarState::Basic(_) => VarStatus::Basic,
                VarState::NonBasic(col) => {
                    let s = &self.nb_var_states[col];
                    if s.at_min {
                        VarStatus::AtLower
                    } else if s.at_max {
                        VarStatus::AtUpper
                    } else {
                        VarStatus::Free
                    }
                }
            });
        }
        Basis(statuses)
    }

    /// The all-slack basis (identity basis matrix). Loading it cannot fail with a
    /// singular factorization, so it is the universal fallback.
    pub(crate) fn slack_basis(&self) -> Basis {
        let mut statuses = Vec::with_capacity(self.num_total_vars());
        for var in 0..self.num_vars {
            let min = self.orig_var_mins[var];
            let max = self.orig_var_maxs[var];
            statuses.push(if min.is_finite() {
                VarStatus::AtLower
            } else if max.is_finite() {
                VarStatus::AtUpper
            } else {
                VarStatus::Free
            });
        }
        for _ in 0..self.num_constraints() {
            statuses.push(VarStatus::Basic);
        }
        Basis(statuses)
    }

    /// Rebuild the solver state from a basis snapshot and the CURRENT variable bounds:
    /// non-basic values come from statuses + bounds, basic values and reduced costs are
    /// recomputed from scratch, and the LU factorization is rebuilt. Feasibility flags
    /// are recomputed honestly, so any partially rebuilt pre-load state is discarded.
    ///
    /// Statuses are interpreted against the CURRENT bounds: a status referring to a
    /// bound that has since moved or become infinite is remapped to the nearest finite
    /// bound (else 0) rather than rejected — the branch & bound driver relies on this
    /// when loading a parent basis after changing variable bounds.
    ///
    /// # Errors
    ///
    /// If this returns `Err`, the solver's internal state is unspecified and must
    /// not be used for solving until a subsequent successful `load_basis` restores
    /// it (the all-slack basis from [`Self::slack_basis`] always loads
    /// successfully and is the designated recovery path).
    pub(crate) fn load_basis(&mut self, basis: &Basis) -> Result<(), Error> {
        let n = self.num_total_vars();
        let m = self.num_constraints();
        if basis.0.len() != n || basis.0.iter().filter(|s| **s == VarStatus::Basic).count() != m {
            return Err(Error::InternalError("basis shape mismatch".to_string()));
        }

        self.basic_vars.clear();
        self.basic_var_mins.clear();
        self.basic_var_maxs.clear();
        self.nb_vars.clear();
        self.nb_var_vals.clear();
        self.nb_var_states.clear();

        for var in 0..n {
            match basis.0[var] {
                VarStatus::Basic => {
                    self.var_states[var] = VarState::Basic(self.basic_vars.len());
                    self.basic_vars.push(var);
                    self.basic_var_mins.push(self.orig_var_mins[var]);
                    self.basic_var_maxs.push(self.orig_var_maxs[var]);
                }
                ref status => {
                    let min = self.orig_var_mins[var];
                    let max = self.orig_var_maxs[var];
                    let val = match status {
                        VarStatus::AtLower => {
                            if min.is_finite() {
                                min
                            } else if max.is_finite() {
                                max
                            } else {
                                0.0
                            }
                        }
                        VarStatus::AtUpper => {
                            if max.is_finite() {
                                max
                            } else if min.is_finite() {
                                min
                            } else {
                                0.0
                            }
                        }
                        VarStatus::Free => {
                            if min.is_finite() {
                                min
                            } else if max.is_finite() {
                                max
                            } else {
                                0.0
                            }
                        }
                        VarStatus::Basic => unreachable!(),
                    };
                    let tol = self.var_tols[var];
                    self.var_states[var] = VarState::NonBasic(self.nb_vars.len());
                    self.nb_vars.push(var);
                    self.nb_var_vals.push(val);
                    self.nb_var_states.push(NonBasicVarState {
                        at_min: at_bound(val, min, tol),
                        at_max: at_bound(val, max, tol),
                    });
                }
            }
        }

        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars)?;

        // Steepest-edge reference reset (standard practice after a warm-start load;
        // only affects pivot ordering quality, not correctness).
        if self.enable_dual_steepest_edge {
            self.dual_edge_sq_norms = vec![1.0; self.basic_vars.len()];
        }

        self.recalc_basic_var_vals()?;
        self.recalc_obj_coeffs()?;

        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        self.is_dual_feasible = self.calc_dual_infeasibility().0 == 0;
        Ok(())
    }

    /// Fix a structural var to `val` and re-solve. A fix is exactly a bound
    /// change to `[val, val]` followed by [`Self::reoptimize`]: the branch &
    /// bound primitive, applied through the public API. Whether the var is
    /// basic or non-basic does not matter — a basic var that already holds
    /// `val` simply stays basic with equal bounds, and one that does not is
    /// repaired by the dual simplex like any other bound violation. (The
    /// previous implementation evicted a basic var through the dual ratio
    /// test unconditionally, and when the only candidate entering column was
    /// a fixed equality slack it reported `Infeasible` for a value the solve
    /// itself had just returned: issue #47.)
    ///
    /// `val` must lie within the var's bounds as they were before any fix;
    /// fixing again replaces the previous fixed value.
    pub(crate) fn fix_var(&mut self, var: usize, val: f64) -> Result<StopReason, Error> {
        let (min, max) = self.fixed_var_bounds[var].unwrap_or(self.get_var_bounds(var));
        if val < min || val > max {
            return Err(Error::Infeasible);
        }
        self.fixed_var_bounds[var] = Some((min, max));
        self.set_var_bounds(var, val, val)?;
        self.reoptimize()
    }

    /// Restore the bounds a var had before [`Self::fix_var`] and re-solve.
    /// Return whether the var was really fixed and whether reoptimization
    /// finished within the active deadline.
    pub(crate) fn unfix_var(&mut self, var: usize) -> Result<(bool, StopReason), Error> {
        let Some((min, max)) = self.fixed_var_bounds[var].take() else {
            return Ok((false, StopReason::Finished));
        };
        self.set_var_bounds(var, min, max)?;
        let stop = self.reoptimize()?;
        Ok((true, stop))
    }

    pub(crate) fn num_constraints(&self) -> usize {
        self.orig_constraints.rows()
    }

    fn num_total_vars(&self) -> usize {
        self.num_vars + self.num_constraints()
    }

    pub(crate) fn initial_solve(&mut self) -> Result<StopReason, Error> {
        if check_deadline(&self.deadline) == StopReason::Limit {
            return Ok(StopReason::Limit);
        }

        let stop = self.run_phases()?;

        if stop == StopReason::Finished {
            // Disable updates of primal sq. norms, because lengthy primal simplex runs
            // are unlikely after the initial solve.
            self.enable_primal_steepest_edge = false;
        }

        Ok(stop)
    }

    /// Primal simplex on the current reduced costs. Returns whether any pivot
    /// was performed besides the stop reason; ends with `is_dual_feasible`
    /// provisionally set (the caller verifies it on recomputed reduced costs).
    fn optimize(&mut self) -> Result<(StopReason, bool), Error> {
        // Rows whose entry in the current entering column proved to be
        // round-off (see `PivotChoice::Inconsistent`) and columns found
        // unexploitable at the current vertex; both reset after a pivot.
        let mut excluded_rows: Vec<usize> = Vec::new();
        let mut excluded_cols: Vec<usize> = Vec::new();
        let mut pivoted = false;
        for iter in 0.. {
            self.lp_iterations += 1;
            if iter % DEADLINE_CHECK_INTERVAL == 0 {
                if check_deadline(&self.deadline) == StopReason::Limit {
                    return Ok((StopReason::Limit, pivoted));
                }

                let (num_vars, infeasibility) = self.calc_dual_infeasibility();
                debug!(
                    "optimize iter {}: obj.: {}, non-optimal coeffs: {} ({})",
                    iter, self.cur_obj_val, num_vars, infeasibility,
                );
            }

            match self.choose_pivot(&excluded_rows, &excluded_cols)? {
                PivotChoice::Unexploitable(col) => {
                    debug!(
                        "optimize iter {iter}: column of var {} blocked at tolerance level; \
                         not an improving direction",
                        self.nb_vars[col]
                    );
                    excluded_cols.push(col);
                }
                PivotChoice::Pivot(pivot_info) => {
                    trace!(
                        "primal pivot: var {} enters (col {}) with step {:e}, {}",
                        self.nb_vars[pivot_info.col],
                        pivot_info.col,
                        pivot_info.entering_diff,
                        match &pivot_info.elem {
                            Some(elem) => format!(
                                "var {} leaves row {} to {:e}, element {:e}",
                                self.basic_vars[elem.row],
                                elem.row,
                                elem.leaving_new_val,
                                elem.coeff
                            ),
                            None => "bound flip".to_string(),
                        }
                    );
                    self.pivot(&pivot_info)?;
                    pivoted = true;
                    excluded_rows.clear();
                    excluded_cols.clear();
                }
                PivotChoice::Inconsistent { row, tiny } => {
                    if !self.factorization_is_fresh() {
                        debug!("optimize iter {iter}: pivot row/column disagreement; rebuilding");
                        self.rebuild()?;
                    } else if tiny {
                        debug!("optimize iter {iter}: entry at row {row} is round-off; excluded");
                        excluded_rows.push(row);
                    } else {
                        return Err(Error::InternalError(
                            "pivot element disagrees between row and column on a fresh factorization"
                                .to_string(),
                        ));
                    }
                }
                PivotChoice::Optimal => {
                    debug!(
                        "found optimum in {} iterations, obj.: {}",
                        iter + 1,
                        self.cur_obj_val,
                    );
                    break;
                }
            }
        }

        // Provisional: `run_phases` confirms it on recomputed reduced costs.
        self.is_dual_feasible = true;
        Ok((StopReason::Finished, pivoted))
    }

    fn restore_feasibility(&mut self) -> Result<StopReason, Error> {
        let obj_str = if self.is_dual_feasible {
            "obj."
        } else {
            "artificial obj."
        };

        // Numerics valve, armed once per stall: before an infeasibility
        // declaration is allowed to stand, the basis gets refactorized and
        // the basic values recomputed from the original data. See below.
        let mut refreshed_since_pivot = false;
        // Non-basic columns whose entry in the current leaving row proved to
        // be round-off (see the agreement check below), for that row only.
        let mut excluded_cols: Vec<usize> = Vec::new();
        let mut excluded_row: Option<usize> = None;

        for iter in 0.. {
            self.lp_iterations += 1;
            if iter % DEADLINE_CHECK_INTERVAL == 0 {
                if check_deadline(&self.deadline) == StopReason::Limit {
                    return Ok(StopReason::Limit);
                }

                let (num_vars, infeasibility) = self.calc_primal_infeasibility();
                debug!(
                    "restore feasibility iter {}: {}: {}, infeas. vars: {} ({})",
                    iter, obj_str, self.cur_obj_val, num_vars, infeasibility,
                );
            }

            if let Some((row, leaving_new_val)) = self.choose_pivot_row_dual() {
                self.calc_row_coeffs(row);
                if excluded_row != Some(row) {
                    excluded_cols.clear();
                    excluded_row = Some(row);
                }
                let pivot_info =
                    match self.choose_entering_col_dual(row, leaving_new_val, &excluded_cols) {
                        Ok(pivot_info) => pivot_info,
                        Err(Error::Infeasible) if !refreshed_since_pivot => {
                            // "No eligible entering column" is a proof of primal
                            // infeasibility only in exact arithmetic. This deep
                            // in an eta-file chain, the leaving row can be a
                            // *phantom* violation — basic values drifted by
                            // accumulated round-off — whose (equally drifted)
                            // pivot row then blocks every candidate; declaring
                            // infeasibility here is a wrong answer (netlib/brandy
                            // did exactly this). Rebuild the factorization and
                            // everything derived from it and re-examine: a
                            // phantom dissolves, a real infeasibility survives
                            // the rebuild and the next declaration stands.
                            debug!(
                                "restore feasibility iter {}: no entering column for row {}; \
                             rebuilding before declaring infeasibility",
                                iter, row,
                            );
                            self.rebuild()?;
                            refreshed_since_pivot = true;
                            continue;
                        }
                        Err(e) => return Err(e),
                    };
                self.calc_col_coeffs(pivot_info.col);
                // The pivot element is known through the row (from the
                // transposed solve) and through the column (from the forward
                // solve). A disagreement means one of them is round-off: with
                // pivots stacked on the factorization, rebuild and redo the
                // iteration; on a fresh factorization the basis itself is
                // numerically unusable.
                let a_row = pivot_info.elem.as_ref().map_or(0.0, |e| e.coeff);
                let a_col = self.col_coeff_at(row);
                let scale = self
                    .row_coeffs
                    .iter()
                    .chain(self.col_coeffs.iter())
                    .map(|(_, coeff)| coeff.abs())
                    .fold(0.0, f64::max);
                if !Self::pivots_agree(a_row, a_col, scale) {
                    if !self.factorization_is_fresh() {
                        debug!(
                            "restore feasibility iter {iter}: pivot row/column disagreement \
                             ({a_row:e} vs {a_col:e}); rebuilding"
                        );
                        self.rebuild()?;
                    } else if a_row.abs() < PIVOT_REL_TOL * scale {
                        debug!(
                            "restore feasibility iter {iter}: entry {a_row:e} of column {} in row \
                             {row} is round-off; excluded",
                            pivot_info.col
                        );
                        excluded_cols.push(pivot_info.col);
                    } else {
                        return Err(Error::InternalError(
                            "pivot element disagrees between row and column on a fresh factorization"
                                .to_string(),
                        ));
                    }
                    continue;
                }
                trace!(
                    "dual pivot: var {} enters (col {}), var {} leaves row {row} to {leaving_new_val:e}, \
                     element {a_row:e}, entering step {:e}",
                    self.nb_vars[pivot_info.col],
                    pivot_info.col,
                    self.basic_vars[row],
                    pivot_info.entering_diff,
                );
                self.pivot(&pivot_info)?;
                // Any successful pivot is progress: re-arm the valve.
                refreshed_since_pivot = false;
                excluded_row = None;
            } else {
                debug!(
                    "restored feasibility in {} iterations, {}: {}",
                    iter + 1,
                    obj_str,
                    self.cur_obj_val,
                );
                break;
            }
        }

        // Provisional: `run_phases` confirms it on recomputed values.
        self.is_primal_feasible = true;
        Ok(StopReason::Finished)
    }

    pub(crate) fn add_constraint(
        &mut self,
        coeffs: CsVec,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<StopReason, Error> {
        assert!(self.is_primal_feasible);
        assert!(self.is_dual_feasible);

        let coeffs = scale_columns(coeffs, &self.col_scales);
        let Some(PreparedRow {
            mut coeffs,
            rhs,
            row_scale,
            slack_var_min,
            slack_var_max,
        }) = prepare_row(coeffs, cmp_op, rhs, |var| {
            self.orig_var_mins[var] == self.orig_var_maxs[var]
        })?
        else {
            return Ok(StopReason::Finished);
        };

        let slack_var = self.num_total_vars();

        self.orig_obj_coeffs.push(0.0);
        self.orig_var_mins.push(slack_var_min);
        self.orig_var_maxs.push(slack_var_max);
        self.var_states.push(VarState::Basic(self.basic_vars.len()));
        self.basic_vars.push(slack_var);
        self.basic_var_mins.push(slack_var_min);
        self.basic_var_maxs.push(slack_var_max);

        let mut lhs_val = 0.0;
        let mut magnitude = rhs.abs();
        for (var, &coeff) in coeffs.iter() {
            let term = coeff * self.internal_value(var);
            lhs_val += term;
            magnitude += term.abs();
        }
        self.basic_var_vals.push(rhs - lhs_val);
        self.var_tols
            .push(slack_tol(self.feasibility, row_scale, magnitude));
        for (var, &coeff) in coeffs.iter() {
            let budget = snap_budget(self.feasibility, row_scale, coeff);
            if budget < self.row_budgets[var] {
                self.row_budgets[var] = budget;
                self.var_tols[var] = structural_tol(
                    self.col_scales[var],
                    self.orig_var_mins[var],
                    self.orig_var_maxs[var],
                    self.feasibility,
                    budget,
                );
            }
        }
        // The new slack's reduced cost is exactly zero until the multipliers
        // are next recomputed.
        self.dual_tols.push(dual_tol(0.0));

        let new_num_total_vars = self.num_total_vars() + 1;
        let mut new_orig_constraints = CsMat::empty(CompressedStorage::CSR, new_num_total_vars);
        for row in self.orig_constraints.outer_iterator() {
            new_orig_constraints =
                new_orig_constraints.append_outer_csvec(resized_view(&row, new_num_total_vars));
        }
        coeffs = into_resized(coeffs, new_num_total_vars);
        coeffs.append(slack_var, 1.0);
        new_orig_constraints = new_orig_constraints.append_outer_csvec(coeffs.view());

        self.orig_rhs.push(rhs);
        self.row_scales.push(row_scale);

        self.orig_constraints = new_orig_constraints;
        self.orig_constraints_csc = self.orig_constraints.to_csc();

        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars)?;

        if self.enable_primal_steepest_edge || self.enable_dual_steepest_edge {
            // existing tableau rows didn't change, so we calc the last row
            // and add its contribution to the sq. norms.
            self.calc_row_coeffs(self.num_constraints() - 1);

            if self.enable_primal_steepest_edge {
                for (c, &coeff) in self.row_coeffs.iter() {
                    self.primal_edge_sq_norms[c] += coeff * coeff;
                }
            }

            if self.enable_dual_steepest_edge {
                self.dual_edge_sq_norms
                    .push(self.inv_basis_row_coeffs.sq_norm());
            }
        }

        self.is_primal_feasible = false;
        self.values_dirty = true;
        self.reduced_costs_dirty = true;
        self.run_phases()
    }

    /// Number of infeasible basic vars and sum of their infeasibilities.
    fn calc_primal_infeasibility(&self) -> (usize, f64) {
        let mut num_vars = 0;
        let mut infeasibility = 0.0;
        for (r, ((&val, &min), &max)) in self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
            .enumerate()
        {
            let tol = self.var_tols[self.basic_vars[r]];
            if val < min - tol {
                num_vars += 1;
                infeasibility += min - val;
            } else if val > max + tol {
                num_vars += 1;
                infeasibility += val - max;
            }
        }
        (num_vars, infeasibility)
    }

    /// Number of infeasible obj. coeffs and sum of their infeasibilities.
    fn calc_dual_infeasibility(&self) -> (usize, f64) {
        let mut num_vars = 0;
        let mut infeasibility = 0.0;
        for (c, (&obj_coeff, var_state)) in self
            .nb_var_obj_coeffs
            .iter()
            .zip(&self.nb_var_states)
            .enumerate()
        {
            let tol = self.dual_tols[self.nb_vars[c]];
            if !(var_state.at_min && obj_coeff > -tol || var_state.at_max && obj_coeff < tol) {
                num_vars += 1;
                infeasibility += obj_coeff.abs();
            }
        }
        (num_vars, infeasibility)
    }

    /// Calculate current coeffs column for a single non-basic variable.
    fn calc_col_coeffs(&mut self, c_var: usize) {
        let var = self.nb_vars[c_var];
        //guaranteed to be a valid index
        let orig_col = self.orig_constraints_csc.outer_view(var).unwrap();
        self.basis_solver
            .solve(orig_col.iter())
            .to_sparse_vec(&mut self.col_coeffs);
    }

    /// Calculate current coeffs row for a single constraint (permuted according to nb_vars).
    fn calc_row_coeffs(&mut self, r_constr: usize) {
        self.basis_solver
            .solve_transp(std::iter::once((r_constr, &1.0)))
            .to_sparse_vec(&mut self.inv_basis_row_coeffs);

        self.row_coeffs.clear_and_resize(self.nb_vars.len());
        for (r, &coeff) in self.inv_basis_row_coeffs.iter() {
            //guaranteed to be a valid index
            for (v, &val) in self.orig_constraints.outer_view(r).unwrap().iter() {
                if let VarState::NonBasic(idx) = self.var_states[v] {
                    *self.row_coeffs.get_mut(idx) += val * coeff;
                }
            }
        }
    }

    /// `excluded_rows` lists rows judged numerically zero in the entering
    /// column (their entry disagreed with the row computation on a fresh
    /// factorization) and therefore not candidate pivots.
    /// `excluded_cols` lists columns found unexploitable at this vertex (see
    /// [`PivotChoice::Unexploitable`]).
    fn choose_pivot(
        &mut self,
        excluded_rows: &[usize],
        excluded_cols: &[usize],
    ) -> Result<PivotChoice, Error> {
        let entering_c = {
            let filtered_obj_coeffs = self
                .nb_var_obj_coeffs
                .iter()
                .zip(&self.nb_var_states)
                .enumerate()
                .filter_map(|(col, (&obj_coeff, var_state))| {
                    // Choose only among non-basic vars that can be changed
                    // with objective decreasing.
                    let tol = self.dual_tols[self.nb_vars[col]];
                    if (var_state.at_min && obj_coeff > -tol)
                        || (var_state.at_max && obj_coeff < tol)
                        || excluded_cols.contains(&col)
                    {
                        None
                    } else {
                        Some((col, obj_coeff))
                    }
                });

            let mut best_col = None;
            let mut best_score = f64::NEG_INFINITY;
            if self.enable_primal_steepest_edge {
                for (col, obj_coeff) in filtered_obj_coeffs {
                    let score = obj_coeff * obj_coeff / self.primal_edge_sq_norms[col];
                    if score > best_score {
                        best_col = Some(col);
                        best_score = score;
                    }
                }
            } else {
                for (col, obj_coeff) in filtered_obj_coeffs {
                    let score = obj_coeff.abs();
                    if score > best_score {
                        best_col = Some(col);
                        best_score = score;
                    }
                }
            }

            if let Some(col) = best_col {
                col
            } else {
                return Ok(PivotChoice::Optimal);
            }
        };

        let entering_cur_val = self.nb_var_vals[entering_c];
        // If true, entering variable will increase (because the objective function must decrease).
        let entering_diff_sign = self.nb_var_obj_coeffs[entering_c] < 0.0;
        let entering_other_val = if entering_diff_sign {
            self.orig_var_maxs[self.nb_vars[entering_c]]
        } else {
            self.orig_var_mins[self.nb_vars[entering_c]]
        };

        self.calc_col_coeffs(entering_c);

        let get_leaving_var_step = |r: usize, coeff: f64| -> f64 {
            let val = self.basic_var_vals[r];
            // leaving_diff = -entering_diff * coeff. From this we can determine
            // in which direction this basic var will change and select appropriate bound.
            if (entering_diff_sign && coeff < 0.0) || (!entering_diff_sign && coeff > 0.0) {
                let max = self.basic_var_maxs[r];
                if val < max {
                    max - val
                } else {
                    0.0
                }
            } else {
                let min = self.basic_var_mins[r];
                if val > min {
                    val - min
                } else {
                    0.0
                }
            }
        };

        // Harris rule. See e.g.
        // Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989).
        // A practical anti-cycling procedure for linearly constrained optimization.
        // Mathematical Programming, 45(1-3), 437-474.
        //
        // https://link.springer.com/content/pdf/10.1007/BF01589114.pdf

        // Entries below the pivot tolerance are not candidate pivots: an
        // entry within the round-off of computing the column (relative to
        // its largest entry, see ENTRY_ROUNDOFF) is noise, and an entry
        // seven orders below the largest entry among the rows whose basic
        // var can block the step at all (see PIVOT_REL_TOL) is too unstable
        // while a better candidate exists; a row with no finite bound in the
        // way says nothing about the scale of the candidates.
        let col_max_all = self
            .col_coeffs
            .iter()
            .map(|(_, coeff)| coeff.abs())
            .fold(0.0, f64::max);
        let col_max = self
            .col_coeffs
            .iter()
            .filter(|&(r, &coeff)| {
                !excluded_rows.contains(&r) && get_leaving_var_step(r, coeff).is_finite()
            })
            .map(|(_, coeff)| coeff.abs())
            .fold(0.0, f64::max);
        let pivot_floor = (ENTRY_ROUNDOFF * col_max_all).max(PIVOT_REL_TOL * col_max);

        // First, we determine the max change in entering variable so that basic variables
        // remain feasible using relaxed bounds (relaxed by each basic var's own tolerance).
        let mut max_step = (entering_other_val - entering_cur_val).abs();
        for (r, &coeff) in self.col_coeffs.iter() {
            let coeff_abs = coeff.abs();
            if coeff_abs < pivot_floor || excluded_rows.contains(&r) {
                continue;
            }

            // By which amount can we change the entering variable so that the limit on this
            // basic var is not violated. The var with the minimum such amount becomes leaving.
            let tol = self.var_tols[self.basic_vars[r]];
            let cur_step = (get_leaving_var_step(r, coeff) + tol) / coeff_abs;
            if cur_step < max_step {
                max_step = cur_step;
            }
        }

        // Second, we choose among variables with steps less than max_step a variable with the biggest
        // abs. coefficient as the leaving variable. This means that we get numerically more stable
        // basis at the price of slight infeasibility of some basic variables.
        let mut leaving_r = None;
        let mut leaving_new_val = 0.0;
        let mut pivot_coeff_abs = f64::NEG_INFINITY;
        let mut pivot_coeff = 0.0;
        for (r, &coeff) in self.col_coeffs.iter() {
            let coeff_abs = coeff.abs();
            if coeff_abs < pivot_floor || excluded_rows.contains(&r) {
                continue;
            }

            let cur_step = get_leaving_var_step(r, coeff) / coeff_abs;
            if cur_step <= max_step && coeff_abs > pivot_coeff_abs {
                leaving_r = Some(r);
                leaving_new_val = if (entering_diff_sign && coeff < 0.0)
                    || (!entering_diff_sign && coeff > 0.0)
                {
                    self.basic_var_maxs[r]
                } else {
                    self.basic_var_mins[r]
                };
                pivot_coeff = coeff;
                pivot_coeff_abs = coeff_abs;
            }
        }

        if let Some(row) = leaving_r {
            self.calc_row_coeffs(row);

            // Same pivot-element cross-check as the dual phase (see
            // `restore_feasibility`): the column gave `pivot_coeff`, the row
            // must agree.
            let row_max_all = self
                .row_coeffs
                .iter()
                .map(|(_, coeff)| coeff.abs())
                .fold(0.0, f64::max);
            let a_row = *self.row_coeffs.get(entering_c);
            if !Self::pivots_agree(a_row, pivot_coeff, col_max_all.max(row_max_all)) {
                return Ok(PivotChoice::Inconsistent {
                    row,
                    tiny: pivot_coeff.abs() < PIVOT_REL_TOL * col_max_all,
                });
            }

            // The step never runs against the entering var's improving
            // direction. A leaving var that already sits at or (within its
            // tolerance) past the bound it leaves at gives a degenerate
            // pivot: the entering var keeps its value and the leaving var is
            // set onto its bound, which turns its tolerance-level overshoot
            // into a tolerance-level row residual instead of walking the
            // entering var backwards out of its own bounds.
            let raw_diff = (self.basic_var_vals[row] - leaving_new_val) / pivot_coeff;
            let entering_diff = if entering_diff_sign {
                raw_diff.max(0.0)
            } else {
                raw_diff.min(0.0)
            };
            // The next recomputation from the factorization will move the
            // entering var by the clamped-away `raw_diff`; if that exceeds
            // its tolerance the pivot would land on an infeasible basis.
            if entering_diff == 0.0 && raw_diff.abs() > self.var_tols[self.nb_vars[entering_c]] {
                return Ok(PivotChoice::Unexploitable(entering_c));
            }
            let entering_new_val = entering_cur_val + entering_diff;

            Ok(PivotChoice::Pivot(PivotInfo {
                col: entering_c,
                entering_new_val,
                entering_diff,
                elem: Some(PivotElem {
                    row,
                    coeff: pivot_coeff,
                    leaving_new_val,
                }),
            }))
        } else {
            if entering_other_val.is_infinite() {
                return Err(Error::Unbounded);
            }

            Ok(PivotChoice::Pivot(PivotInfo {
                col: entering_c,
                entering_new_val: entering_other_val,
                entering_diff: entering_other_val - entering_cur_val,
                elem: None,
            }))
        }
    }

    fn choose_pivot_row_dual(&self) -> Option<(usize, f64)> {
        let infeasibilities = self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
            .enumerate()
            .filter_map(|(r, ((&val, &min), &max))| {
                let tol = self.var_tols[self.basic_vars[r]];
                if val < min - tol {
                    Some((r, min - val))
                } else if val > max + tol {
                    Some((r, val - max))
                } else {
                    None
                }
            });

        let mut leaving_r = None;
        let mut max_score = f64::NEG_INFINITY;
        if self.enable_dual_steepest_edge {
            for (r, infeasibility) in infeasibilities {
                let sq_norm = self.dual_edge_sq_norms[r];
                let score = infeasibility * infeasibility / sq_norm;
                if score > max_score {
                    leaving_r = Some(r);
                    max_score = score;
                }
            }
        } else {
            for (r, infeasibility) in infeasibilities {
                if infeasibility > max_score {
                    leaving_r = Some(r);
                    max_score = infeasibility;
                }
            }
        }

        leaving_r.map(|r| {
            let val = self.basic_var_vals[r];
            let min = self.basic_var_mins[r];
            let max = self.basic_var_maxs[r];

            // If we choose this var as leaving, its new val will be at the boundary
            // which is violated.
            // Why is that? We must maintain primal optimality (a.k.a. dual feasibility) for
            // the leaving variable, thus new_obj_coeff must be >= 0 if new_val is min, and <= 0
            // if new_val is max. Sign of the leaving var obj coeff:
            // sign(new_obj_coeff) = -sign(old_obj_coeff) * sign(pivot_coeff).
            // Another constraint is that we must not decrease primal objective.
            // As sign(obj_val_diff) = -sign(old_obj_coeff) * sign(leaving_diff) * sign(pivot_coeff)
            // must be >= 0, we conclude that sign(new_obj_coeff) = sign(leaving_diff).
            // From this we see that if old val was < min, dual feasibility is maintained if the
            // new var is min (analogously for max).
            let new_val = if val < min {
                min
            } else if val > max {
                max
            } else {
                unreachable!();
            };
            (r, new_val)
        })
    }

    /// `excluded` lists non-basic columns judged numerically zero in this row
    /// (their entry disagreed with the column computation on a fresh
    /// factorization) and therefore not candidates.
    fn choose_entering_col_dual(
        &self,
        row: usize,
        leaving_new_val: f64,
        excluded: &[usize],
    ) -> Result<PivotInfo, Error> {
        // True if the new obj. coeff. must be nonnegative in a dual-feasible configuration.
        let leaving_diff_sign = leaving_new_val > self.basic_var_vals[row];

        fn clamp_obj_coeff(mut obj_coeff: f64, var_state: &NonBasicVarState) -> f64 {
            if var_state.at_min && obj_coeff < 0.0 {
                obj_coeff = 0.0;
            }
            if var_state.at_max && obj_coeff > 0.0 {
                obj_coeff = 0.0;
            }
            obj_coeff
        }

        // How far the leaving var's move would carry an entering var with
        // this tableau entry.
        let leaving_move = self.basic_var_vals[row] - leaving_new_val;

        // Whether the non-basic var `c` with this tableau entry can enter in
        // the direction the pivot implies: away from the bound it sits at.
        let direction_ok = |coeff: f64, var_state: &NonBasicVarState| -> bool {
            if coeff == 0.0 {
                return false;
            }
            let entering_diff_sign = if coeff > 0.0 {
                !leaving_diff_sign
            } else {
                leaving_diff_sign
            };
            if entering_diff_sign {
                !var_state.at_max
            } else {
                !var_state.at_min
            }
        };

        // Whether the non-basic var `c` could instead absorb the leaving
        // var's move by crossing the bound it sits at (or, if fixed, its
        // value) by no more than its own tolerance. "At a bound" means
        // within tolerance of it throughout the engine, so such a var ends
        // the pivot still at its bound in that sense. This is how a
        // round-off-level violation that a var cannot carry within its own
        // tolerance (an integer var, or a var with a tight bound) is handed
        // to a var or row that can: without it, a model whose data are
        // rounded at a level the row tolerances allow (an equality on a 1e9
        // activity pinning a var 3e-7 past its bound) could only be
        // reported infeasible, since the vertex that satisfies every
        // tolerance has the equality's slack basic and nothing else can
        // bring it into the basis. An integer var never absorbs: branching
        // and rounding rely on it sitting exactly on its bounds.
        let absorbs = |c: usize, coeff: f64| -> bool {
            if coeff == 0.0 {
                return false;
            }
            let var = self.nb_vars[c];
            let integer = var < self.num_vars
                && matches!(
                    self.orig_var_domains[var],
                    VarDomain::Integer | VarDomain::Boolean
                );
            // A slack's stored tolerance carries a round-off floor derived
            // at the last recomputation; what it may absorb is judged on
            // its row's magnitude at the current point.
            let tol = if var < self.num_vars {
                self.var_tols[var]
            } else {
                self.current_slack_tol(var - self.num_vars)
            };
            // The comparison itself is between two rounded quantities.
            !integer && (leaving_move / coeff).abs() <= tol * (1.0 + ROUNDOFF_FLOOR)
        };

        // The Harris two-pass ratio test over the vars `can_enter` admits;
        // returns the entering column and its pivot element.
        //
        // Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989).
        // A practical anti-cycling procedure for linearly constrained optimization.
        // Mathematical Programming, 45(1-3), 437-474.
        //
        // https://link.springer.com/content/pdf/10.1007/BF01589114.pdf
        let select = |can_enter: &dyn Fn(usize, f64, &NonBasicVarState) -> bool| {
            // Entries below the pivot tolerance are not candidate pivots: an
            // entry within the round-off of computing the row (relative to
            // its largest entry, see ENTRY_ROUNDOFF: the row vector comes
            // from a solve whose round-off is spread over its components in
            // proportion to the largest) is noise, and an entry seven orders
            // below the largest genuine entry among the vars that could
            // enter (see PIVOT_REL_TOL) is too unstable while a better
            // candidate exists. The entry of a var that can never enter,
            // such as a fixed one, counts for the round-off scale but not
            // for the candidates'.
            let (mut row_max_all, mut row_max) = (0.0f64, 0.0f64);
            for (c, &coeff) in self.row_coeffs.iter() {
                let abs = coeff.abs();
                if abs > row_max_all {
                    row_max_all = abs;
                }
                if abs > row_max && can_enter(c, coeff, &self.nb_var_states[c]) {
                    row_max = abs;
                }
            }
            let noise_floor = ENTRY_ROUNDOFF * row_max_all;
            let pivot_floor = noise_floor.max(PIVOT_REL_TOL * row_max);

            let is_eligible_var = |c: usize, coeff: f64, var_state: &NonBasicVarState| -> bool {
                coeff.abs() >= pivot_floor
                    && !excluded.contains(&c)
                    && can_enter(c, coeff, var_state)
            };

            // First, we determine the max step (change in the leaving variable obj. coeff that still
            // leaves us with a dual-feasible state) using relaxed bounds.
            let mut max_step = f64::INFINITY;
            for (c, &coeff) in self.row_coeffs.iter() {
                let var_state = &self.nb_var_states[c];
                if !is_eligible_var(c, coeff, var_state) {
                    continue;
                }

                let obj_coeff = clamp_obj_coeff(self.nb_var_obj_coeffs[c], var_state);
                let tol = self.dual_tols[self.nb_vars[c]];
                let cur_step = (obj_coeff.abs() + tol) / coeff.abs();
                if cur_step < max_step {
                    max_step = cur_step;
                }
            }

            // Second, we choose among the variables satisfying the relaxed step bound
            // the one with the biggest pivot coefficient. This allows for a much more
            // numerically stable basis at the price of slight infeasibility in dual variables.
            let mut entering_c = None;
            let mut pivot_coeff_abs = f64::NEG_INFINITY;
            let mut pivot_coeff = 0.0;
            for (c, &coeff) in self.row_coeffs.iter() {
                let var_state = &self.nb_var_states[c];
                if !is_eligible_var(c, coeff, var_state) {
                    continue;
                }

                let obj_coeff = clamp_obj_coeff(self.nb_var_obj_coeffs[c], var_state);

                // If we change obj. coeff of the leaving variable by this amount,
                // obj. coeff if the current variable will reach the bound of dual infeasibility.
                // Variable with the tightest such bound is the entering variable.
                let cur_step = obj_coeff.abs() / coeff.abs();
                if cur_step <= max_step {
                    let coeff_abs = coeff.abs();
                    if coeff_abs > pivot_coeff_abs {
                        entering_c = Some(c);
                        pivot_coeff_abs = coeff_abs;
                        pivot_coeff = coeff;
                    }
                }
            }
            (entering_c, pivot_coeff, pivot_floor)
        };

        // Regular candidates first; vars that could only absorb the move
        // within their tolerance are a fallback, since their pivot does not
        // keep dual feasibility by construction (the primal phase repairs
        // that) while a regular one does.
        let strict = |_c: usize, coeff: f64, state: &NonBasicVarState| direction_ok(coeff, state);
        let (mut entering_c, mut pivot_coeff, mut pivot_floor) = select(&strict);
        if entering_c.is_none() {
            let relaxed = |c: usize, coeff: f64, state: &NonBasicVarState| {
                direction_ok(coeff, state) || absorbs(c, coeff)
            };
            (entering_c, pivot_coeff, pivot_floor) = select(&relaxed);
        }

        if let Some(col) = entering_c {
            let entering_diff = (self.basic_var_vals[row] - leaving_new_val) / pivot_coeff;
            let entering_new_val = self.nb_var_vals[col] + entering_diff;

            Ok(PivotInfo {
                col,
                entering_new_val,
                entering_diff,
                elem: Some(PivotElem {
                    row,
                    leaving_new_val,
                    coeff: pivot_coeff,
                }),
            })
        } else {
            if log::log_enabled!(log::Level::Debug) {
                debug!(
                    "no entering column for row {row} (basic var {} = {:e}, bounds [{:e}, {:e}], \
                     leaving to {leaving_new_val:e}, pivot floor {pivot_floor:e}); row entries:",
                    self.basic_vars[row],
                    self.basic_var_vals[row],
                    self.basic_var_mins[row],
                    self.basic_var_maxs[row],
                );
                for (c, &coeff) in self.row_coeffs.iter() {
                    let state = &self.nb_var_states[c];
                    debug!(
                        "  var {} coeff {coeff:e} value {:e} at_min {} at_max {} tol {:e} \
                         direction_ok {} absorbs {}",
                        self.nb_vars[c],
                        self.nb_var_vals[c],
                        state.at_min,
                        state.at_max,
                        self.var_tols[self.nb_vars[c]],
                        direction_ok(coeff, state),
                        absorbs(c, coeff),
                    );
                }
            }
            Err(Error::Infeasible)
        }
    }

    fn pivot(&mut self, pivot_info: &PivotInfo) -> Result<(), Error> {
        self.values_dirty = true;
        self.reduced_costs_dirty = true;
        self.pivots_since_drift_check += 1;
        self.cur_obj_val += self.nb_var_obj_coeffs[pivot_info.col] * pivot_info.entering_diff;

        let entering_var = self.nb_vars[pivot_info.col];

        if pivot_info.elem.is_none() {
            // "entering" var is still non-basic, it just changes value from one limit
            // to the other.
            self.nb_var_vals[pivot_info.col] = pivot_info.entering_new_val;
            for (r, coeff) in self.col_coeffs.iter() {
                self.basic_var_vals[r] -= pivot_info.entering_diff * coeff;
            }
            let tol = self.var_tols[entering_var];
            let var_state = &mut self.nb_var_states[pivot_info.col];
            var_state.at_min = at_bound(
                pivot_info.entering_new_val,
                self.orig_var_mins[entering_var],
                tol,
            );
            var_state.at_max = at_bound(
                pivot_info.entering_new_val,
                self.orig_var_maxs[entering_var],
                tol,
            );
            return Ok(());
        }
        //guaranteed, none variant already handled
        let pivot_elem = pivot_info.elem.as_ref().unwrap();
        let pivot_coeff = pivot_elem.coeff;

        // Update basic vars stuff

        for (r, coeff) in self.col_coeffs.iter() {
            if r == pivot_elem.row {
                self.basic_var_vals[r] = pivot_info.entering_new_val;
            } else {
                self.basic_var_vals[r] -= pivot_info.entering_diff * coeff;
            }
        }

        self.basic_var_mins[pivot_elem.row] = self.orig_var_mins[entering_var];
        self.basic_var_maxs[pivot_elem.row] = self.orig_var_maxs[entering_var];

        if self.enable_dual_steepest_edge {
            self.update_dual_sq_norms(pivot_elem.row, pivot_coeff);
        }

        // Update non-basic vars stuff

        let leaving_var = self.basic_vars[pivot_elem.row];

        self.nb_var_vals[pivot_info.col] = pivot_elem.leaving_new_val;
        let leaving_tol = self.var_tols[leaving_var];
        let leaving_var_state = &mut self.nb_var_states[pivot_info.col];
        leaving_var_state.at_min = at_bound(
            pivot_elem.leaving_new_val,
            self.orig_var_mins[leaving_var],
            leaving_tol,
        );
        leaving_var_state.at_max = at_bound(
            pivot_elem.leaving_new_val,
            self.orig_var_maxs[leaving_var],
            leaving_tol,
        );

        let pivot_obj = self.nb_var_obj_coeffs[pivot_info.col] / pivot_coeff;
        for (c, &coeff) in self.row_coeffs.iter() {
            if c == pivot_info.col {
                self.nb_var_obj_coeffs[c] = -pivot_obj;
            } else {
                self.nb_var_obj_coeffs[c] -= pivot_obj * coeff;
            }
        }

        if self.enable_primal_steepest_edge {
            self.update_primal_sq_norms(pivot_info.col, pivot_coeff);
        }

        // Update basis itself

        self.basic_vars[pivot_elem.row] = entering_var;
        self.var_states[entering_var] = VarState::Basic(pivot_elem.row);
        self.nb_vars[pivot_info.col] = leaving_var;
        self.var_states[leaving_var] = VarState::NonBasic(pivot_info.col);

        // A simple heuristic to choose when to recompute LU factorization.
        // Note: a possible failure mode is that the LU factorization accidentally
        // generates a lot of fill-in and doesn't get recomputed for a long time.
        // A refactorization also recomputes the values and reduced costs from
        // it, so incremental round-off never outlives one eta file.
        let eta_matrices_nnz = self.basis_solver.eta_matrices.coeff_cols.nnz();
        if eta_matrices_nnz < self.basis_solver.lu_factors.nnz() {
            self.basis_solver
                .push_eta_matrix(&self.col_coeffs, pivot_elem.row, pivot_coeff);
        } else {
            self.refactorize()?;
        }
        Ok(())
    }

    fn update_primal_sq_norms(&mut self, entering_col: usize, pivot_coeff: f64) {
        // Computations for the steepest edge pivoting rule. See
        // Forrest, J. J., & Goldfarb, D. (1992).
        // Steepest-edge simplex algorithms for linear programming.
        // Mathematical programming, 57(1-3), 341-374.
        //
        // https://link.springer.com/content/pdf/10.1007/BF01581089.pdf

        let tmp = self.basis_solver.solve_transp(self.col_coeffs.iter());
        // now tmp contains the v vector from the article.

        for &r in tmp.indices() {
            //guaranteed to be a valid index
            for &v in self.orig_constraints.outer_view(r).unwrap().indices() {
                if let VarState::NonBasic(idx) = self.var_states[v] {
                    self.sq_norms_update_helper[idx] = 0.0;
                }
            }
        }
        // now significant positions in sq_norms_update_helper are cleared.

        for (r, &coeff) in tmp.iter() {
            //guaranteed to be a valid index
            for (v, &val) in self.orig_constraints.outer_view(r).unwrap().iter() {
                if let VarState::NonBasic(idx) = self.var_states[v] {
                    self.sq_norms_update_helper[idx] += val * coeff;
                }
            }
        }
        // now sq_norms_update_helper contains transp(N) * v vector.

        // Calculate pivot_sq_norm directly to avoid loss of precision.
        let pivot_sq_norm = self.col_coeffs.sq_norm() + 1.0;
        // assert!((self.primal_edge_sq_norms[entering_col] - pivot_sq_norm).abs() < 0.1);

        let pivot_coeff_sq = pivot_coeff * pivot_coeff;
        for (c, &r_coeff) in self.row_coeffs.iter() {
            if c == entering_col {
                self.primal_edge_sq_norms[c] = pivot_sq_norm / pivot_coeff_sq;
            } else {
                self.primal_edge_sq_norms[c] += -2.0 * r_coeff * self.sq_norms_update_helper[c]
                    / pivot_coeff
                    + pivot_sq_norm * r_coeff * r_coeff / pivot_coeff_sq;
            }

            assert!(self.primal_edge_sq_norms[c].is_finite());
        }
    }

    fn update_dual_sq_norms(&mut self, leaving_row: usize, pivot_coeff: f64) {
        // Computations for the dual steepest edge pivoting rule.
        // See the same reference (Forrest, Goldfarb).

        let tau = self.basis_solver.solve(self.inv_basis_row_coeffs.iter());

        // Calculate pivot_sq_norm directly to avoid loss of precision.
        let pivot_sq_norm = self.inv_basis_row_coeffs.sq_norm();
        // assert!((self.dual_edge_sq_norms[leaving_row] - pivot_sq_norm).abs() < 0.1);

        let pivot_coeff_sq = pivot_coeff * pivot_coeff;
        for (r, &col_coeff) in self.col_coeffs.iter() {
            if r == leaving_row {
                self.dual_edge_sq_norms[r] = pivot_sq_norm / pivot_coeff_sq;
            } else {
                self.dual_edge_sq_norms[r] += -2.0 * col_coeff * tau.get(r) / pivot_coeff
                    + pivot_sq_norm * col_coeff * col_coeff / pivot_coeff_sq;
            }

            assert!(self.dual_edge_sq_norms[r].is_finite());
        }
    }

    fn recalc_basic_var_vals(&mut self) -> Result<(), Error> {
        let mut cur_vals = self.orig_rhs.clone();
        for (i, var) in self.nb_vars.iter().enumerate() {
            let val = self.nb_var_vals[i];
            if val != 0.0 {
                //guaranteed to be a valid index
                for (r, &coeff) in self.orig_constraints_csc.outer_view(*var).unwrap().iter() {
                    cur_vals[r] -= val * coeff;
                }
            }
        }

        // Etas are applied to the dense solve directly; a pending eta file
        // does not require a full basis refactorization.
        self.basis_solver.solve_dense_with_etas(&mut cur_vals);
        self.basic_var_vals = cur_vals;
        // Basic slacks re-derived from their rows and the slack tolerances
        // refreshed; the residuals themselves are the caller's business.
        let mut residuals = Vec::with_capacity(self.num_constraints());
        self.check_rows(&mut residuals);
        // Recomputed is not verified: on an ill-conditioned basis a solve
        // through the factorization leaves a residual above the rows'
        // tolerances, which only `verify_primal`'s refinement removes. So
        // `values_dirty` stays as it is; clearing it here let a phase exit
        // skip verification right after a refactorization had recomputed the
        // values, and report a point that missed a row by four times its
        // tolerance (the magnitude fixture at 1e8).
        self.pivots_since_drift_check = 0;
        Ok(())
    }

    /// Recompute the reduced costs and the objective from the cost vector in
    /// force (the phase-1 artificial objective while it is active, else the
    /// real one).
    fn recalc_obj_coeffs(&mut self) -> Result<(), Error> {
        // Same as recalc_basic_var_vals: pending etas participate in the
        // (transposed) dense solve instead of forcing a refactorization.
        let multipliers = {
            let costs = self
                .artificial_obj
                .as_deref()
                .unwrap_or(&self.orig_obj_coeffs);
            let mut rhs = vec![0.0; self.num_constraints()];
            for (c, &var) in self.basic_vars.iter().enumerate() {
                rhs[c] = costs[var];
            }
            self.basis_solver.solve_transp_dense_with_etas(&mut rhs);
            rhs
        };
        self.refresh_dual_tols(&multipliers);

        let costs = self
            .artificial_obj
            .as_deref()
            .unwrap_or(&self.orig_obj_coeffs);
        self.nb_var_obj_coeffs.clear();
        for &var in &self.nb_vars {
            //guaranteed to be a valid index
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let dot_prod: f64 = col.iter().map(|(r, val)| val * multipliers[r]).sum();
            self.nb_var_obj_coeffs.push(costs[var] - dot_prod);
        }

        self.cur_obj_val = self.exact_obj_val();
        self.reduced_costs_dirty = false;
        Ok(())
    }

    #[allow(dead_code)]
    fn recalc_primal_sq_norms(&mut self) {
        self.primal_edge_sq_norms.clear();
        for &var in &self.nb_vars {
            //guaranteed to be a valid index
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let sq_norm = self.basis_solver.solve(col.iter()).sq_norm() + 1.0;
            self.primal_edge_sq_norms.push(sq_norm);
        }
    }
}

#[cfg(test)]
mod tests;
