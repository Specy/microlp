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
//! * [`costs`] — the dual simplex's working costs: perturbation, shifting,
//!   the correction of dual infeasibilities, and the cleanup.
//! * [`bounds`] — the primal simplex's working bounds: perturbation,
//!   shifting, and the cleanup.
//! * [`cycling`] — the cycling and stall detection of both.
//! * [`proof`] — the proof of infeasibility the dual simplex needs before it
//!   may return `Infeasible`.

mod basis;
mod bounds;
mod costs;
mod cycling;
mod proof;
mod scaling;
mod tolerances;

pub(crate) use basis::*;
pub(crate) use scaling::*;
pub(crate) use tolerances::*;

use cycling::{CycleGuard, VertexChange, STALL_ITERATIONS};

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

/// A 64-bit hash of `x` (splitmix64's finalizer): the vertex hash of the
/// cycling detection and the pseudo-random fractions of the perturbations.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Which pseudo-random stream a fraction is drawn from.
#[derive(Clone, Copy)]
enum Stream {
    /// The perturbation of a var's cost (dual simplex) or bounds (primal
    /// simplex): HiGHS's `numTotRandomValue_`, which both its
    /// `initialiseCost` and its `initialiseBound` draw from.
    Perturbation,
    /// The margin a shift leaves a var's reduced cost (dual simplex) or
    /// value (primal simplex) on its feasible side.
    Shift,
}

/// A pseudo-random fraction in `[0, 1)` fixed by the var and the stream.
/// HiGHS draws its perturbations from random values fixed per var for the
/// LP (`numTotRandomValue_`, `HEkk::initialiseSimplexLpRandomVectors`);
/// fixing them by the var's index keeps every solve of a model, and every
/// B&B node, deterministic without a generator state to carry around.
fn random_fraction(var: usize, stream: Stream) -> f64 {
    let salt = match stream {
        Stream::Perturbation => 0x243F_6A88_85A3_08D3,
        Stream::Shift => 0x1319_8A2E_0370_7344,
    };
    (mix(var as u64 ^ salt) >> 11) as f64 / (1u64 << 53) as f64
}

/// A phase that starts with no more than this many infeasibilities of the
/// kind it removes (violated rows for the dual simplex, dual infeasible
/// reduced costs for the primal) can be near optimal, and is then not
/// perturbed (HiGHS's `HEkkDual::solve` and `HEkkPrimal::solve`: a value
/// that is unimportant, since a start that is far from optimal has far more).
const NEAR_OPTIMAL_MAX_INFEASIBILITIES: usize = 1000;

/// A phase whose infeasibilities of the kind it removes are all below this,
/// in the engine's scaled units, can be near optimal (HiGHS's
/// `HEkkDual::solve` and `HEkkPrimal::solve`, where the infeasibilities of a
/// near-optimal start are orders of magnitude smaller).
const NEAR_OPTIMAL_MAX_INFEASIBILITY: f64 = 1e-3;

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
    /// Per structural var, whether both scalings left its column out: a var
    /// fixed when the model entered the engine (see [`column_scales`] and
    /// the row equilibration), whose column keeps the user's units. Its
    /// entries are then no measure of the scaled matrix's magnitudes (a
    /// fixed column of `1e17` stays `1e17`), so the dual ratio test leaves
    /// them out of its round-off scale.
    unscaled_cols: Vec<bool>,
    /// The user's absolute feasibility tolerance
    /// ([`crate::Tolerances::feasibility`]), in user units.
    feasibility: f64,
    /// Per total var, the absolute tolerances (in the engine's scaled units)
    /// within which the var counts as at or within each of its bounds:
    /// [`structural_tol`] for a structural var (refreshed when its bounds
    /// change) and [`slack_tol`] at both bounds for a slack (refreshed
    /// whenever the rows are checked, since its round-off floor depends on
    /// the values).
    var_tols: Vec<BoundTols>,
    /// Per total var, the tolerance within which its reduced cost counts as
    /// zero ([`dual_tol`]); refreshed whenever the reduced costs are
    /// recomputed from the multipliers.
    dual_tols: Vec<f64>,
    /// Per structural var, the smallest [`snap_budget`] over its rows (see
    /// [`structural_tol`]); infinite for a var in no row. Static apart from
    /// rows added later.
    row_budgets: Vec<f64>,
    /// The user's rows, as written and in user units, that the engine does
    /// not hold because presolve dropped them (implied by the bounds it
    /// emitted) or rewrote them (valid for integer points only): the answer
    /// is checked against them at the boundary, like against the engine's
    /// own rows ([`Self::first_violated_row`]), but no pivot ever sees them.
    /// A point that holds the engine's rows and violates one of these is a
    /// presolve bug, which the boundary then reports instead of returning
    /// the point.
    validation_rows: Vec<(CsVec, ComparisonOp, f64)>,

    enable_primal_steepest_edge: bool,
    enable_dual_steepest_edge: bool,

    /// While the initial dual phase runs on the phase-1 artificial objective
    /// (the start being neither primal nor dual feasible), its cost vector
    /// (one entry per total var), so that any recomputation of the reduced
    /// costs during that phase uses the same objective the pivots priced.
    /// `None` once the real objective is in force.
    artificial_obj: Option<Vec<f64>>,

    /// Per total var, what a dual phase adds to the var's cost in force:
    /// the perturbation drawn when the phase starts and the shifts that
    /// corrected its reduced cost (HiGHS's `workCost_` less the LP's cost).
    /// Zero outside a dual phase. See [`costs`].
    cost_perturbation: Vec<f64>,
    /// Per total var, the shift a zero dual step gave the var's cost when it
    /// entered the basis (HiGHS's `workShift_`), given back when it leaves.
    /// Zero outside a dual phase. See [`costs`].
    cost_shifts: Vec<f64>,
    /// Whether any entry of the two above is nonzero, i.e. the simplex works
    /// with costs other than the costs in force.
    costs_modified: bool,
    /// Per total var, its bounds as a primal phase has moved them: perturbed
    /// when the phase starts, shifted where its steps left a violation or a
    /// var left the basis past its bound (HiGHS's `workLower_`/`workUpper_`).
    /// Taken from the true bounds at the first move of a phase, and read only
    /// while `bounds_modified`, through [`Self::working_bounds`]. See
    /// [`bounds`].
    working_mins: Vec<f64>,
    working_maxs: Vec<f64>,
    /// Whether a primal phase has moved any bound, i.e. the simplex works
    /// with bounds other than the true ones.
    bounds_modified: bool,
    /// The cycling detection of both simplex phases (see [`cycling`]).
    cycle_guard: CycleGuard,

    is_primal_feasible: bool,
    is_dual_feasible: bool,

    /// Whether the basic values have changed (pivots, bound shifts, a
    /// recompute through the factorization) since [`Self::verify_primal`]
    /// last checked them against the rows; when not, a phase exit needs no
    /// residual check and no refinement. A recompute sets it: recomputed is
    /// not verified (see [`Self::recalc_basic_var_vals`]).
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
    /// The combined column `Σ Δx_c a_c` of the bound flips of one dual step
    /// (see [`Self::apply_flips`]).
    flip_rhs: ScatteredVec,
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

impl NonBasicVarState {
    /// Whether a non-basic var in this state with reduced cost `d` offers no
    /// improving move beyond the tolerance `tol`: its cost pushes it against
    /// a bound it sits at, or moving it changes the objective by less than
    /// `tol` per unit. A var at neither bound (a free var, or one a bound
    /// change left between its bounds) can move both ways, so only the
    /// second holds for it; entering such a var with a zero reduced cost is
    /// a move that improves nothing, and with no bound in the way an
    /// infinite one.
    fn is_dual_feasible(&self, d: f64, tol: f64) -> bool {
        self.dual_infeasibility(d) < tol
    }
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

        // Everything below works in scaled units: x'_j = x_j / s_j. A var
        // fixed here is left out of both scalings.
        let col_scales = column_scales(obj_coeffs, var_mins, var_maxs, constraints);
        let unscaled_cols: Vec<bool> = (0..num_vars).map(|v| var_mins[v] == var_maxs[v]).collect();
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
        let is_fixed = |var: usize| unscaled_cols[var];
        let mut prepared_rows = Vec::with_capacity(constraints.len());
        for (coeffs, cmp_op, rhs) in constraints {
            let coeffs = scale_columns(coeffs.clone(), &col_scales);
            if let Some(row) = prepare_row(coeffs, *cmp_op, *rhs, feasibility, is_fixed)? {
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
        let mut var_tols: Vec<BoundTols> = (0..num_vars)
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
            let (init_val, var_dual_feasible) = if at_bound(min, max, tol.lo.min(tol.hi)) {
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
                at_min: at_bound(init_val, min, tol.lo),
                at_max: at_bound(init_val, max, tol.hi),
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
            var_tols.push(BoundTols::uniform(slack_tol(
                feasibility,
                row_scale,
                magnitude,
            )));
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
            unscaled_cols,
            feasibility,
            var_tols,
            dual_tols,
            row_budgets,
            validation_rows: Vec::new(),
            deadline,
            operation_time_limit: None,
            lp_iterations: 0,
            elapsed: Duration::ZERO,
            orig_var_domains: var_domains.to_vec(),
            enable_primal_steepest_edge,
            enable_dual_steepest_edge,
            artificial_obj,
            cost_perturbation: vec![0.0; num_total_vars],
            cost_shifts: vec![0.0; num_total_vars],
            costs_modified: false,
            working_mins: Vec::new(),
            working_maxs: Vec::new(),
            bounds_modified: false,
            cycle_guard: CycleGuard::default(),
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
            flip_rhs: ScatteredVec::empty(num_constraints),
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
    ///
    /// The rows presolve dropped or rewrote are checked after the engine's
    /// own, as the user wrote them ([`Self::validation_rows`]), so every
    /// caller holds the answer to the whole model: the MIP guard, the LP
    /// boundary, and the LP edits and resumes that go through it.
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
        // The rows presolve took over: the same contract on the rows as the
        // user wrote them (numbered after the engine's own).
        self.validation_rows
            .iter()
            .enumerate()
            .find_map(|(k, (coeffs, op, rhs))| {
                row_violation(coeffs, *op, *rhs, |v| values[v], self.feasibility)
                    .map(|violation| (self.num_constraints() + k, violation))
            })
    }

    /// Install the rows the answer must satisfy beyond the engine's own (see
    /// [`Self::validation_rows`]): those presolve dropped or rewrote.
    pub(crate) fn set_validation_rows(&mut self, rows: Vec<(CsVec, ComparisonOp, f64)>) {
        self.validation_rows = rows;
    }

    /// The first structural var whose value (user units) violates its bounds
    /// beyond the contract, the user tolerance floored at the round-off of
    /// each bound ([`outside_bounds`]), with the violation in user units.
    pub(crate) fn first_violated_bound(&self, values: &[f64]) -> Option<(usize, f64)> {
        values.iter().enumerate().find_map(|(v, &x)| {
            let (lo, hi) = self.get_var_bounds(v);
            outside_bounds(x, lo, hi, self.feasibility).then(|| (v, excursion(x, lo, hi)))
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
        debug_assert!(
            !self.bounds_modified,
            "a bound changed during a primal phase"
        );
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
                if val < min - tol.lo || val > max + tol.hi {
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
                    at_min: at_bound(new_val, min, tol.lo),
                    at_max: at_bound(new_val, max, tol.hi),
                };
                // A var at a loosened bound may no longer justify its reduced cost.
                self.downgrade_dual_feasibility(col);
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
    ///
    /// A dual phase works with perturbed and shifted costs and ends by
    /// restoring the costs in force (see [`costs`]); a primal phase works with
    /// perturbed and shifted bounds and ends by restoring the true ones (see
    /// [`bounds`]). The verification below then judges the point on the true
    /// bounds and recomputes the reduced costs from the costs in force. The
    /// dual infeasibility a dual phase's cleanup uncovers is removed by the
    /// primal phase that follows, as HiGHS follows its dual cleanup with one
    /// primal simplex pass (`HEkkDual::solve`), and the primal infeasibility a
    /// primal phase's cleanup uncovers by the dual phase of the next round, as
    /// HiGHS follows its primal cleanup with one dual simplex pass
    /// (`HEkkPrimal::solve`). Only the first phase of a re-solve starts from an
    /// arbitrary state and may perturb; every later phase cleans up after the
    /// one before it and runs as HiGHS runs its cleanup passes: unperturbed,
    /// and a dual phase with phase 2 forced, its dual infeasibilities shifted
    /// rather than flipped (`pass_force_phase2`; the phase-1 dual on the
    /// artificial objective does not count: the primal phase after it is the
    /// real optimization, not its cleanup). Where HiGHS then accepts whatever
    /// is left (`max_dual_simplex_cleanup_level = 1`), this loop verifies and
    /// alternates the phases at most [`MAX_PHASE_ROUNDS`] times, and fails
    /// loudly after that. As SoPlex does when its algorithms loop, each round
    /// after the first cuts the tolerance of the primal ratio test a
    /// hundredfold ([`LOOP_HARRIS_REDUCTION`]), so that a step no longer ends
    /// at a tolerance the verification's recomputation finds exceeded.
    fn run_phases(&mut self) -> Result<StopReason, Error> {
        let mut cleanup = false;
        for round in 0..MAX_PHASE_ROUNDS {
            // Each round after the first cuts the primal ratio test's
            // tolerance a hundredfold (see `LOOP_HARRIS_REDUCTION`).
            let harris_scale = LOOP_HARRIS_REDUCTION.powi(round as i32);
            if !self.is_primal_feasible {
                if self.restore_feasibility(cleanup)? == StopReason::Limit {
                    return Ok(StopReason::Limit);
                }
                cleanup = cleanup || self.artificial_obj.is_none();
            }
            if !self.is_dual_feasible {
                // Leaving the dual phase for the primal one: the real
                // objective is in force from here on.
                self.artificial_obj = None;
                self.recalc_obj_coeffs()?;
                let stop = self.optimize(cleanup, harris_scale)?.0;
                cleanup = true;
                if stop == StopReason::Limit {
                    return Ok(StopReason::Limit);
                }
            }
            if !self.verify_primal()? {
                debug!("phase round {round}: recomputed values violate a bound; continuing");
                continue;
            }
            // Dual verification: exact reduced costs. If they still show an
            // infeasibility, let the primal simplex act on them; no pivot at
            // all means the vertex is optimal within tolerance.
            if self.artificial_obj.is_some() || self.reduced_costs_dirty {
                self.artificial_obj = None;
                self.recalc_obj_coeffs()?;
            }
            if self.calc_dual_infeasibility().0 == 0 {
                self.is_dual_feasible = true;
                return Ok(StopReason::Finished);
            }
            debug!("phase round {round}: recomputed reduced costs are dual infeasible; continuing");
            let (stop, pivoted) = self.optimize(cleanup, harris_scale)?;
            cleanup = true;
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
            self.var_tols[slack] = BoundTols::uniform(tol);
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

    /// The objective of the current values under the costs the simplex
    /// works with: the cost vector in force (the artificial one during phase
    /// 1), plus the perturbation and shifts of a running dual phase (see
    /// [`costs`]).
    fn exact_obj_val(&self) -> f64 {
        let costs = self
            .artificial_obj
            .as_deref()
            .unwrap_or(&self.orig_obj_coeffs);
        let cost = |var: usize| {
            if self.costs_modified {
                costs[var] + self.cost_perturbation[var] + self.cost_shifts[var]
            } else {
                costs[var]
            }
        };
        let mut obj = 0.0;
        for (r, &var) in self.basic_vars.iter().enumerate() {
            obj += cost(var) * self.basic_var_vals[r];
        }
        for (c, &var) in self.nb_vars.iter().enumerate() {
            obj += cost(var) * self.nb_var_vals[c];
        }
        obj
    }

    /// Whether the pivot element computed through the row and through the
    /// column agree: to [`PIVOT_AGREEMENT_TOL`] relative to the element, or
    /// to the round-off of the computations that produced it, `scale` being
    /// the largest entry of the tableau row and column involved (a genuine
    /// entry far below its neighbours is computed with an absolute error of
    /// that order and can never agree relatively). Agreement within that
    /// round-off vouches only for an element above it: one within the
    /// round-off floor of the scale ([`LU_SINGULAR_REL`], where the
    /// factorization would call its basis singular) agrees with anything of
    /// its own size, round-off included, so there only relative agreement
    /// tells a genuine element from noise. (The genuine tiny pivots of wide
    /// coefficient chains, down to 4e-15 of their column, come out of both
    /// computations identical; a round-off entry of 1.4e-15, the only one
    /// "blocking" an unbounded ray, came out 2% apart and was taken as a
    /// pivot with a step of 1.6e15, into a basis the factorization refused.)
    fn pivots_agree(a_row: f64, a_col: f64, scale: f64) -> bool {
        let diff = (a_row - a_col).abs();
        let element = a_row.abs().min(a_col.abs());
        diff <= PIVOT_AGREEMENT_TOL * a_row.abs().max(a_col.abs())
            || (diff <= ENTRY_ROUNDOFF * scale && element > LU_SINGULAR_REL * scale)
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
        debug_assert!(!self.costs_modified, "a basis loaded during a dual phase");
        debug_assert!(
            !self.bounds_modified,
            "a basis loaded during a primal phase"
        );

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
                        at_min: at_bound(val, min, tol.lo),
                        at_max: at_bound(val, max, tol.hi),
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
    /// `val` must lie within the var's bounds as they were before any fix,
    /// within the contract (a value a solve reports may sit outside a bound
    /// by round-off, issue #47): a value outside a bound within its
    /// tolerance fixes the var at that bound. Fixing again replaces the
    /// previous fixed value.
    pub(crate) fn fix_var(&mut self, var: usize, val: f64) -> Result<StopReason, Error> {
        let (min, max) = self.fixed_var_bounds[var].unwrap_or(self.get_var_bounds(var));
        if outside_bounds(val, min, max, self.feasibility) {
            return Err(Error::Infeasible);
        }
        let val = val.clamp(min, max);
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

    /// The primal simplex phase: reach dual feasibility while keeping the
    /// basic values primal feasible. It runs on working bounds (see
    /// [`bounds`]): [`Self::begin_primal_phase`] perturbs them unless the
    /// start is near optimal or the phase is a `cleanup` after another (see
    /// [`Self::run_phases`]), and whatever way the phase ends, they return to
    /// the true bounds ([`Self::end_primal_phase`]).
    /// Returns whether any pivot was performed besides the stop reason; ends
    /// with `is_dual_feasible` provisionally set (the caller verifies it on
    /// recomputed reduced costs) and `is_primal_feasible` measured on the
    /// true bounds.
    fn optimize(&mut self, cleanup: bool, harris_scale: f64) -> Result<(StopReason, bool), Error> {
        self.begin_primal_phase(cleanup);
        let result = self.primal_phase(harris_scale);
        let cleanup = self.end_primal_phase();
        let result = result?;
        cleanup?;
        Ok(result)
    }

    /// The pivots of the primal simplex phase, on the working bounds.
    ///
    /// Each iteration prices the non-basic vars ([`Self::price_primal`]),
    /// skipping a var whose entry into the basis was found to keep returning
    /// to visited vertices ([`CycleGuard`], HiGHS's
    /// `HEkk::applyTabooVariableIn`), and runs the ratio test for the
    /// entering column ([`Self::choose_pivot`]); it skips the basis change if
    /// it is bad, and otherwise applies it, then shifts the working bound of
    /// every basic var the step left outside its bounds beyond its tolerance
    /// ([`Self::correct_primal`], after a recomputation of the values too).
    ///
    /// The phase fails loudly rather than run forever: when, after a rebuild,
    /// the only vars left to enter are taboo (every basis change left returns
    /// to visited vertices, HiGHS's `kSolvePhaseTabooBasis`), and when the
    /// objective has not fallen beyond round-off in [`STALL_ITERATIONS`]
    /// iterations (SoPlex's stall check).
    fn primal_phase(&mut self, harris_scale: f64) -> Result<(StopReason, bool), Error> {
        // Rows whose entry in the entering column `excluded_col` proved to
        // be round-off (see `PivotChoice::Inconsistent`), for that column
        // only, as the dual phase keeps its exclusions for its leaving row:
        // the verdict is on one entry of one column in the current basis.
        // They are dropped when another column is priced (as after a taboo
        // made the column's var skip its basis change: excluded for the next
        // column too, a genuine blocking entry of that column went unseen,
        // and the step passed it, or the phase took a bounded ray for an
        // unbounded one) and after any pivot, and kept through a rebuild,
        // which factorizes the same basis again.
        let mut excluded_rows: Vec<usize> = Vec::new();
        let mut excluded_col: Option<usize> = None;
        let mut pivoted = false;
        // Whether the data are fresh from a rebuild with no step since: only
        // then do taboo vars alone left to enter mean the phase is stuck.
        let mut refreshed_since_step = false;
        // The objective at the last stall check.
        let mut stall_reference = self.cur_obj_val;
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
            if iter > 0 && iter % STALL_ITERATIONS == 0 {
                // Every step of the primal simplex is forward or zero (a step
                // that would run backwards is taken as zero, see
                // `choose_pivot`), and with the bounds perturbed zero steps
                // are the exception, so a phase that progresses lowers the
                // objective (a rebuild moves it by the round-off of
                // recomputing the values, either way). One that has not
                // fallen beyond round-off since the last check (or is not a
                // number) is taken to be stuck in a cycle the vertex hash has
                // not caught (see `cycling`).
                let obj = self.cur_obj_val;
                let resolution = ROUNDOFF_FLOOR * obj.abs().max(stall_reference.abs());
                let fallen = stall_reference - obj > resolution;
                if !fallen {
                    return Err(Error::InternalError(format!(
                        "primal simplex stalled: the objective did not improve in \
                         {STALL_ITERATIONS} iterations"
                    )));
                }
                stall_reference = obj;
            }

            let Some(col) = self.price_primal()? else {
                if self.taboo_var_dual_infeasible() {
                    // Only vars whose entry returns to visited vertices are
                    // left. Try them again on fresh data, where the reduced
                    // costs (and so the pricing) may differ; if nothing
                    // moved since the last rebuild, the phase is stuck.
                    if refreshed_since_step {
                        return Err(Error::InternalError(
                            "primal simplex cycles: every basis change left returns to visited \
                             vertices"
                                .to_string(),
                        ));
                    }
                    debug!("optimize iter {iter}: only taboo vars can enter; rebuilding");
                    self.rebuild_primal()?;
                    refreshed_since_step = true;
                    continue;
                }
                debug!(
                    "found optimum in {} iterations, obj.: {}",
                    iter + 1,
                    self.cur_obj_val,
                );
                break;
            };
            if excluded_col != Some(col) {
                excluded_rows.clear();
                excluded_col = Some(col);
            }
            match self.choose_pivot(col, &excluded_rows, harris_scale)? {
                PivotChoice::Pivot(pivot_info) => {
                    if let Some(elem) = &pivot_info.elem {
                        let next = self.cycle_guard.hash_after(
                            self.pivot_vertex_changes(
                                pivot_info.col,
                                elem.row,
                                elem.leaving_new_val,
                            )
                            .into_iter(),
                        );
                        if self.cycle_guard.is_bad_change(
                            elem.row,
                            self.basic_vars[elem.row],
                            self.nb_vars[pivot_info.col],
                            next,
                        ) {
                            // The entering var is taboo now; the next
                            // iteration prices another.
                            continue;
                        }
                    }
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
                    if let Some(elem) = &pivot_info.elem {
                        self.hold_leaving_value(self.basic_vars[elem.row], elem.leaving_new_val);
                    }
                    self.pivot(&pivot_info)?;
                    self.forget_moved_rows(pivot_info.entering_diff);
                    self.correct_primal();
                    pivoted = true;
                    refreshed_since_step = false;
                    excluded_col = None;
                }
                PivotChoice::Inconsistent { row, tiny } => {
                    if !self.factorization_is_fresh() {
                        debug!("optimize iter {iter}: pivot row/column disagreement; rebuilding");
                        self.rebuild_primal()?;
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
            }
        }

        // Provisional: `run_phases` confirms it on recomputed reduced costs.
        self.is_dual_feasible = true;
        Ok((StopReason::Finished, pivoted))
    }

    /// A rebuild within the primal phase, as HiGHS's `HEkkPrimal::rebuild`:
    /// a fresh factorization and everything derived from it
    /// ([`Self::rebuild`]), the working bounds of the basic vars the
    /// recomputed values put outside them shifted (HiGHS's `correctPrimal`),
    /// and every taboo lifted, so that on fresh data every var may enter
    /// again (the records of bad basis changes stay: see [`CycleGuard`]).
    fn rebuild_primal(&mut self) -> Result<(), Error> {
        self.rebuild()?;
        self.correct_primal();
        self.cycle_guard.clear_taboo();
        Ok(())
    }

    /// Whether a non-basic var whose entry into the basis is taboo is dual
    /// infeasible, i.e. would be priced if it were not taboo.
    fn taboo_var_dual_infeasible(&self) -> bool {
        self.cycle_guard.any_taboo()
            && self.nb_vars.iter().enumerate().any(|(c, &var)| {
                self.cycle_guard.is_taboo_entering(var)
                    && !self.nb_var_states[c]
                        .is_dual_feasible(self.nb_var_obj_coeffs[c], self.dual_tols[var])
            })
    }

    /// Drop the records of bad basis changes whose row the step just made
    /// (the entering var's move `entering_diff` along the column
    /// `col_coeffs`, a pivot or a bound flip) has moved by at least the row's
    /// tolerance: the state the change was bad in is gone (see
    /// [`CycleGuard::forget_moved_rows`]).
    fn forget_moved_rows(&mut self, entering_diff: f64) {
        let (col_coeffs, basic_vars, var_tols) =
            (&self.col_coeffs, &self.basic_vars, &self.var_tols);
        self.cycle_guard.forget_moved_rows(|r| {
            let tol = var_tols[basic_vars[r]];
            let coeff = col_coeffs
                .iter()
                .find(|&(i, _)| i == r)
                .map_or(0.0, |(_, &coeff)| coeff);
            (coeff * entering_diff).abs() >= tol.lo.min(tol.hi)
        });
    }

    /// The dual simplex phase: restore primal feasibility while keeping the
    /// reduced costs dual feasible. It runs on working costs (see [`costs`]):
    /// [`Self::begin_dual_phase`] perturbs them unless the start is near
    /// optimal or the phase is a `cleanup` after another (see
    /// [`Self::run_phases`]) and corrects the dual infeasibilities the phase
    /// starts with, and whatever way the phase ends, they return to the costs
    /// in force ([`Self::end_dual_phase`]).
    fn restore_feasibility(&mut self, cleanup: bool) -> Result<StopReason, Error> {
        let result = self
            .begin_dual_phase(cleanup)
            .and_then(|()| self.dual_phase());
        self.end_dual_phase();
        result
    }

    /// The pivots of the dual simplex phase, on the working costs.
    ///
    /// Each iteration takes the violated row with the best steepest-edge
    /// score, skipping a row whose basis change was found to keep returning
    /// to visited vertices ([`CycleGuard`]); computes the row and runs the
    /// ratio test; skips the change if it is bad, and otherwise applies it:
    /// the flips the step passed, then the pivot, with a zero dual step and
    /// the entering var's cost shifted where its reduced cost lies on the
    /// wrong side of zero (see [`DualStep::zero_step`]). Every rebuild in the phase
    /// ([`Self::rebuild_dual`]) also corrects the dual infeasibilities the
    /// recomputed reduced costs show (HiGHS's `HEkkDual::rebuild`), which
    /// is also what a candidate found dual infeasible beyond its tolerance
    /// asks for ([`DualRatioTest::DualInfeasible`]).
    ///
    /// A row that no var can repair ([`DualRatioTest::NoCandidate`]) is
    /// looked at again on fresh data (a rebuild), and then proves the model
    /// infeasible only if the certificate its row of the basis inverse
    /// suggests holds on the model's data ([`Self::prove_infeasible`]):
    /// `Infeasible` is returned on that proof and on nothing else, as HiGHS
    /// declares infeasibility only when `HEkk::proofOfPrimalInfeasibility`
    /// holds (`HEkkDual::assessPossiblyDualUnbounded`). A row whose proof
    /// fails is made taboo and the phase goes on with the other rows (HiGHS
    /// again). When only taboo rows are left on fresh data and one of them
    /// failed its proof, that row gets SoPlex's last resort, a pivot on an
    /// entry the regular test refuses as unstable (its "instable leave",
    /// `SPxSolverBase::solve`, src/soplex/spxsolve.hpp, which SoPlex takes
    /// when a row whose only entering var is unstable is all that is left):
    /// the ratio test runs again with every entry of at least
    /// [`INSTABLE_PIVOT_TOL`] a candidate. If that makes no pivot and the row
    /// is left again, the vars whose entries are below round-off move as the
    /// row's certificate says they can repair it ([`Self::round_off_repair`],
    /// once per row in a phase), which no reference does: HiGHS fails on the
    /// models that need it (coefficient chains spanning twelve to fifteen
    /// orders of magnitude) without presolve.
    ///
    /// The phase fails loudly rather than run forever or guess: when the
    /// rows still violated after a rebuild are all taboo (every basis change
    /// left returns to visited vertices, HiGHS's `kSolvePhaseTabooBasis`),
    /// when a row that does not prove infeasibility cannot be repaired even
    /// by the last resorts, and when the dual objective has not risen beyond
    /// round-off in [`STALL_ITERATIONS`] iterations (SoPlex's stall check).
    /// The room such a row's certificate leaves then lies where no step
    /// reaches: in an entry below [`INSTABLE_PIVOT_TOL`] of a var with no
    /// bound in its direction, in a coefficient below the round-off of its
    /// terms, which the repair does not move by, or within the round-off of
    /// the data (see the known limitations in ARCHITECTURE §7).
    fn dual_phase(&mut self) -> Result<StopReason, Error> {
        let obj_str = if self.artificial_obj.is_some() {
            "artificial obj."
        } else {
            "obj."
        };

        // Whether the data are fresh from a rebuild, with no pivot since:
        // a row's verdict, or the loud error when only taboo rows are left,
        // is drawn on fresh data only (see below).
        let mut refreshed_since_pivot = false;
        // Non-basic columns whose entry in the current leaving row proved to
        // be round-off (see the agreement check below), for that row only.
        let mut excluded_cols: Vec<usize> = Vec::new();
        let mut excluded_row: Option<usize> = None;
        // The row SoPlex's instable leave takes next (see above), and the row
        // it was tried on since the data were last refreshed, which gets the
        // round-off repair instead if it is left again. One row at a time:
        // the unproven row with the best score gets the last resorts, and if
        // they fail on it the phase fails without trying another unproven
        // row, whose last resorts might have got further: conservative, a
        // loud error where there might have been an answer, never a wrong
        // answer.
        let mut instable_row: Option<usize> = None;
        let mut instable_tried: Option<usize> = None;
        // The rows the round-off repair has repaired in this phase: a row
        // that needs it again fails the phase (see above).
        let mut repaired_rows: Vec<usize> = Vec::new();
        // The dual objective at the last stall check.
        let mut stall_reference = self.cur_obj_val;

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
            if iter > 0 && iter % STALL_ITERATIONS == 0 {
                // Every dual step is forward or zero and the perturbation
                // makes zero steps rare, so a phase that progresses raises
                // the dual objective of the working costs over many
                // iterations, though not at every one: a shift moves it by
                // the shift times its var's value, either way, and is given
                // back when the var leaves. One that has not risen beyond
                // round-off since the last check (or is not a number) is
                // taken to be stuck in a cycle the vertex hash has not
                // caught (see `cycling`, and the limits of this measure at
                // `STALL_ITERATIONS`).
                let obj = self.cur_obj_val;
                let resolution = ROUNDOFF_FLOOR * obj.abs().max(stall_reference.abs());
                let risen = obj - stall_reference > resolution;
                if !risen {
                    return Err(Error::InternalError(format!(
                        "dual simplex stalled: the objective did not improve in \
                         {STALL_ITERATIONS} iterations"
                    )));
                }
                stall_reference = obj;
            }

            let (row, leaving_new_val, instable) = if let Some(row) = instable_row.take() {
                (row, self.leaving_target(row), true)
            } else if let Some((row, leaving_new_val)) = self.choose_pivot_row_dual() {
                (row, leaving_new_val, false)
            } else {
                if self.taboo_row_violated() {
                    // Only taboo rows are left: rows whose basis change
                    // returns to visited vertices, or that no var can repair
                    // and that do not prove infeasibility. Try them again on
                    // fresh data, where the reduced costs (and so the
                    // entering var) may differ. If nothing moved since the
                    // last rebuild, an unproven row gets the last resort;
                    // otherwise the phase is stuck.
                    if refreshed_since_pivot {
                        if let Some(row) = self.unproven_row() {
                            if instable_tried != Some(row) {
                                debug!(
                                    "restore feasibility iter {iter}: only taboo rows are left; \
                                     instable leave on unproven row {row}"
                                );
                                instable_tried = Some(row);
                                instable_row = Some(row);
                                continue;
                            }
                            // The instable leave made no pivot: the round-off
                            // repair, once per row in a phase, or nothing can
                            // repair the row.
                            if !repaired_rows.contains(&row) && self.round_off_repair(row) {
                                repaired_rows.push(row);
                                refreshed_since_pivot = false;
                                instable_tried = None;
                                continue;
                            }
                            return Err(Error::InternalError(format!(
                                "dual simplex: no var can repair row {row} (basic var {}) by an \
                                 accepted pivot or through a coefficient above round-off, and \
                                 the row does not prove the model infeasible",
                                self.basic_vars[row]
                            )));
                        }
                        return Err(Error::InternalError(
                            "dual simplex cycles: every basis change left returns to visited \
                             vertices"
                                .to_string(),
                        ));
                    }
                    debug!(
                        "restore feasibility iter {iter}: only taboo rows are violated; rebuilding"
                    );
                    self.rebuild_dual()?;
                    refreshed_since_pivot = true;
                    instable_tried = None;
                    continue;
                }
                debug!(
                    "restored feasibility in {} iterations, {}: {}",
                    iter + 1,
                    obj_str,
                    self.cur_obj_val,
                );
                break;
            };
            self.calc_row_coeffs(row);
            if excluded_row != Some(row) {
                excluded_cols.clear();
                excluded_row = Some(row);
            }
            let test =
                self.choose_entering_col_dual(row, leaving_new_val, &excluded_cols, instable);
            let step = match test {
                Ok(DualRatioTest::Step(step)) => step,
                Ok(DualRatioTest::DualInfeasible) => {
                    // A candidate's reduced cost lies beyond its tolerance
                    // on its infeasible side (drift of the incremental
                    // updates, or an earlier step its entry, below the pivot
                    // floor, did not bound: see `rebuild_dual`): recompute
                    // and correct, as HiGHS rebuilds when `workTheta <= 0`.
                    // The correction leaves every reduced cost feasible, so
                    // this cannot follow a rebuild.
                    if refreshed_since_pivot {
                        return Err(Error::InternalError(
                            "dual simplex: a reduced cost is dual infeasible right after its \
                             correction"
                                .to_string(),
                        ));
                    }
                    debug!(
                        "restore feasibility iter {iter}: a candidate of row {row} is dual \
                         infeasible; rebuilding and correcting"
                    );
                    self.rebuild_dual()?;
                    refreshed_since_pivot = true;
                    instable_tried = None;
                    continue;
                }
                Ok(DualRatioTest::NoCandidate) if instable => {
                    // Not even an unstable pivot: the row is left once more,
                    // for the round-off repair (see above).
                    continue;
                }
                Ok(DualRatioTest::NoCandidate) if !refreshed_since_pivot => {
                    // "No eligible entering column" can be a phantom: deep
                    // in an eta-file chain the basic values drift, and the
                    // equally drifted pivot row then blocks every candidate
                    // (netlib/brandy's leaving row did exactly this). The
                    // row is looked at again on fresh data, where a phantom
                    // dissolves, as HiGHS rebuilds on
                    // `kRebuildReasonPossiblyDualUnbounded` and SoPlex
                    // refactorizes in `leave()` before either draws a
                    // conclusion.
                    debug!(
                        "restore feasibility iter {iter}: no entering column for row {row}; \
                         rebuilding before a verdict"
                    );
                    self.rebuild_dual()?;
                    refreshed_since_pivot = true;
                    instable_tried = None;
                    continue;
                }
                Ok(DualRatioTest::NoCandidate) => {
                    // On fresh data: the row proves the model infeasible, or
                    // it is made taboo and the phase goes on with the other
                    // rows (see above).
                    if self
                        .prove_infeasible(row, leaving_new_val > self.basic_var_vals[row])
                        .holds
                    {
                        return Err(Error::Infeasible);
                    }
                    self.cycle_guard.add_failed_proof(row, self.basic_vars[row]);
                    continue;
                }
                Err(e) => return Err(e),
            };
            let (col, a_row) = step.entering;
            let next = self.cycle_guard.hash_after(
                self.dual_step_vertex_changes(&step, col, row, leaving_new_val)
                    .into_iter(),
            );
            if self
                .cycle_guard
                .is_bad_change(row, self.basic_vars[row], self.nb_vars[col], next)
            {
                // The row is taboo now; the next iteration takes another.
                continue;
            }
            self.calc_col_coeffs(col);
            // The pivot element is known through the row (from the
            // transposed solve) and through the column (from the forward
            // solve). A disagreement means one of them is round-off: with
            // pivots stacked on the factorization, rebuild and redo the
            // iteration; on a fresh factorization the basis itself is
            // numerically unusable.
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
                    self.rebuild_dual()?;
                    instable_tried = None;
                } else if a_row.abs() < PIVOT_REL_TOL * scale {
                    debug!(
                        "restore feasibility iter {iter}: entry {a_row:e} of column {col} in \
                         row {row} is round-off; excluded"
                    );
                    excluded_cols.push(col);
                } else {
                    return Err(Error::InternalError(
                        "pivot element disagrees between row and column on a fresh factorization"
                            .to_string(),
                    ));
                }
                continue;
            }
            // The flips the step passed come first. They move basic
            // values, not the basis, so the pivot column stays as
            // computed; the entering var's step is then measured from
            // where they leave the leaving var. The pivot is made even if
            // the flips, computed through their columns, carried the
            // leaving var up to or past its bound (their row predicted
            // less): the entering var's step is then tiny or reversed.
            // The flips never come with a zero dual step (see
            // `choose_entering_col_dual`), so the dual objective grows
            // with every step that flips and flips cannot alternate
            // without progress. (A reversed step can leave the entering
            // var outside its bounds; that is a primal infeasibility like
            // any other, for the next iterations.)
            //
            // Now that the two computations agree, each is used where it
            // belongs (HiGHS `HEkkDual::updatePrimal` and
            // `updateDual`): the column's element for the primal step,
            // the eta file and the steepest-edge updates, which apply it
            // together with the rest of the column, and the row's for the
            // reduced costs, which are updated along the row.
            self.apply_flips(&step.flips);
            if step.zero_step {
                // The dual step `d_q / a_row` would run backwards: the
                // entering var's cost is shifted so its reduced cost is
                // zero, and the step, zero, changes no other.
                self.shift_entering_cost(col);
            }
            let val = self.basic_var_vals[row];
            let entering_diff = (val - leaving_new_val) / a_col;
            let pivot_info = PivotInfo {
                col,
                entering_new_val: self.nb_var_vals[col] + entering_diff,
                entering_diff,
                elem: Some(PivotElem {
                    row,
                    leaving_new_val,
                    coeff: a_col,
                    dual_step_coeff: a_row,
                }),
            };
            trace!(
                "dual pivot: var {} enters (col {col}), var {} leaves row {row} to \
                 {leaving_new_val:e}, element {a_col:e} (row {a_row:e}), entering step \
                 {entering_diff:e}, {} bound flips{}",
                self.nb_vars[col],
                self.basic_vars[row],
                step.flips.len(),
                if step.zero_step {
                    ", zero dual step"
                } else {
                    ""
                },
            );
            self.pivot(&pivot_info)?;
            self.forget_moved_rows(entering_diff);
            // Any successful pivot is progress: re-arm the valve.
            refreshed_since_pivot = false;
            instable_tried = None;
            excluded_row = None;
        }

        // Provisional: `run_phases` confirms it on recomputed values.
        self.is_primal_feasible = true;
        Ok(StopReason::Finished)
    }

    /// A rebuild within the dual phase, as HiGHS's `HEkkDual::rebuild`: a
    /// fresh factorization and everything derived from it
    /// ([`Self::rebuild`]), then the dual infeasibilities the recomputed
    /// reduced costs show corrected, and every taboo lifted, so that on
    /// fresh data every row may be tried again (the records of bad basis
    /// changes stay, as HiGHS keeps them through the rebuilds that stand
    /// for these: see [`CycleGuard`]).
    ///
    /// The correction here shifts every infeasible cost, boxed vars' too, as
    /// SoPlex shifts every violated dual bound, where HiGHS moves a boxed var
    /// to its other bound. HiGHS's ratio test bounds the step by every entry
    /// above a tiny absolute tolerance, so what a rebuild finds is drift;
    /// this one leaves an entry seven orders below the row's largest out of
    /// the bound ([`PIVOT_REL_TOL`]), and a step can leave such a var dual
    /// infeasible by far more. Moving it back to the other bound undid the
    /// step that had moved it and returned the phase to a vertex it had left:
    /// a model of the differential fuzz went round that loop until the
    /// cycling detection gave up (`wide_lp_1897_solves_within_the_contract`
    /// in `src/tests/magnitudes.rs`). A shift keeps the vertex, and the
    /// cleanup gives it back.
    fn rebuild_dual(&mut self) -> Result<(), Error> {
        self.rebuild()?;
        self.correct_dual_infeasibilities(true);
        self.cycle_guard.clear_taboo();
        Ok(())
    }

    /// Start the cycling detection of a phase from the current vertex: every
    /// basic var, and every non-basic var at its upper bound (see
    /// [`CycleGuard`]).
    fn start_cycle_detection(&mut self) {
        let vertex = self
            .basic_vars
            .iter()
            .map(|&var| VertexChange::Basic(var))
            .chain(
                self.nb_vars
                    .iter()
                    .zip(&self.nb_var_states)
                    .filter(|(_, state)| state.at_upper())
                    .map(|(&var, _)| VertexChange::Upper(var)),
            );
        self.cycle_guard.start(vertex);
    }

    /// Whether a row whose basis change is taboo is violated.
    fn taboo_row_violated(&self) -> bool {
        self.cycle_guard.any_taboo()
            && (0..self.num_constraints())
                .any(|r| self.cycle_guard.is_taboo(r) && !self.basic_var_within_bounds(r))
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
        }) = prepare_row(coeffs, cmp_op, rhs, self.feasibility, |var| {
            self.orig_var_mins[var] == self.orig_var_maxs[var]
        })?
        else {
            return Ok(StopReason::Finished);
        };

        let slack_var = self.num_total_vars();

        debug_assert!(!self.costs_modified, "a row added during a dual phase");
        debug_assert!(!self.bounds_modified, "a row added during a primal phase");
        self.orig_obj_coeffs.push(0.0);
        self.cost_perturbation.push(0.0);
        self.cost_shifts.push(0.0);
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
        self.var_tols.push(BoundTols::uniform(slack_tol(
            self.feasibility,
            row_scale,
            magnitude,
        )));
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
            if val < min - tol.lo {
                num_vars += 1;
                infeasibility += min - val;
            } else if val > max + tol.hi {
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
            if !var_state.is_dual_feasible(obj_coeff, self.dual_tols[self.nb_vars[c]]) {
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

    /// The primal simplex's pricing: the entering var, the dual infeasible
    /// one with the best steepest-edge score (the largest reduced cost when
    /// the weights are off) that is not taboo ([`CycleGuard`]), as its
    /// non-basic column; `None` when no var that may enter can improve the
    /// objective.
    fn price_primal(&self) -> Result<Option<usize>, Error> {
        let any_taboo = self.cycle_guard.any_taboo();
        let filtered_obj_coeffs = self
            .nb_var_obj_coeffs
            .iter()
            .zip(&self.nb_var_states)
            .enumerate()
            .filter_map(|(col, (&obj_coeff, var_state))| {
                // Choose only among non-basic vars that can be changed with
                // objective decreasing, and whose entry into the basis is
                // not taboo (HiGHS's `applyTabooVariableIn`).
                let var = self.nb_vars[col];
                if var_state.is_dual_feasible(obj_coeff, self.dual_tols[var])
                    || (any_taboo && self.cycle_guard.is_taboo_entering(var))
                {
                    None
                } else {
                    Some((col, obj_coeff))
                }
            });

        // A reduced cost that is not a number is dual infeasible (every
        // comparison with it is false), but its score, not a number either,
        // never wins the choice below, and a phase whose only candidates
        // were such costs ended as optimal. It is the loud error it is
        // instead.
        let mut best_col = None;
        let mut best_score = f64::NEG_INFINITY;
        for (col, obj_coeff) in filtered_obj_coeffs {
            if !obj_coeff.is_finite() {
                return Err(Error::InternalError(format!(
                    "reduced cost of var {} is not finite: {obj_coeff}",
                    self.nb_vars[col]
                )));
            }
            let score = if self.enable_primal_steepest_edge {
                obj_coeff * obj_coeff / self.primal_edge_sq_norms[col]
            } else {
                obj_coeff.abs()
            };
            if score > best_score {
                best_col = Some(col);
                best_score = score;
            }
        }
        Ok(best_col)
    }

    /// The primal simplex's ratio test for the entering column `entering_c`
    /// (see [`Self::price_primal`]), on the working bounds of the phase (see
    /// [`bounds`]): the basic var that leaves for it, or a flip of the
    /// entering var to its other bound.
    ///
    /// `excluded_rows` lists rows judged numerically zero in this entering
    /// column (their entry disagreed with the row computation on a fresh
    /// factorization of the current basis) and therefore not candidate
    /// pivots. `harris_scale` scales the tolerance the ratio test lets a
    /// basic var pass its bound by (see `run_phases`).
    fn choose_pivot(
        &mut self,
        entering_c: usize,
        excluded_rows: &[usize],
        harris_scale: f64,
    ) -> Result<PivotChoice, Error> {
        let entering_cur_val = self.nb_var_vals[entering_c];
        // If true, entering variable will increase (because the objective function must decrease).
        let entering_diff_sign = self.nb_var_obj_coeffs[entering_c] < 0.0;
        let (entering_min, entering_max) = self.working_bounds(self.nb_vars[entering_c]);
        let entering_other_val = if entering_diff_sign {
            entering_max
        } else {
            entering_min
        };

        self.calc_col_coeffs(entering_c);

        // An entry of the column, or the value of a basic var in a row the
        // column moves, that is not a finite number is a breakdown, not
        // data: it compares false with every bound below, so its row would
        // drop out of the test unseen and the step could run past it, or be
        // taken for an unbounded ray. It is the loud error it is instead (as
        // in the dual ratio test).
        for (r, &coeff) in self.col_coeffs.iter() {
            if !coeff.is_finite() || !self.basic_var_vals[r].is_finite() {
                return Err(Error::InternalError(format!(
                    "primal simplex: entry {coeff} of var {} in row {r}, whose basic var is at \
                     {}, is not finite",
                    self.nb_vars[entering_c], self.basic_var_vals[r]
                )));
            }
        }

        // leaving_diff = -entering_diff * coeff: whether the basic var of a
        // row with this entry moves towards its max as the entering var moves.
        let moves_up = |coeff: f64| {
            (entering_diff_sign && coeff < 0.0) || (!entering_diff_sign && coeff > 0.0)
        };
        // How far the basic var of row `r` is from the (working) bound it
        // moves towards: negative when it already sits past that bound
        // (within its tolerance), infinite when that bound is.
        let signed_room = |r: usize, coeff: f64| -> f64 {
            let val = self.basic_var_vals[r];
            if moves_up(coeff) {
                self.basic_var_maxs[r] - val
            } else {
                val - self.basic_var_mins[r]
            }
        };

        // Harris rule. See e.g.
        // Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989).
        // A practical anti-cycling procedure for linearly constrained optimization.
        // Mathematical Programming, 45(1-3), 437-474.
        //
        // https://link.springer.com/content/pdf/10.1007/BF01589114.pdf

        // An entry within the round-off of computing the column (relative to
        // its largest entry, see ENTRY_ROUNDOFF) is noise: it neither blocks
        // the step nor is a pivot. The step still moves its basic var, and
        // where that takes the var outside its bounds beyond its tolerance,
        // the var's bound is shifted after the step (`correct_primal`), as
        // HiGHS shifts the bounds of the rows its ratio test ignores. Every
        // other entry blocks the step. An entry seven orders below the
        // largest entry among the rows whose basic var can block the step at
        // all (see PIVOT_REL_TOL) is too unstable to pivot on while a larger
        // entry can take the step, and is the pivot only when none can; a row
        // with no finite bound in the way says nothing about the scale of the
        // candidates.
        let col_max_all = self
            .col_coeffs
            .iter()
            .map(|(_, coeff)| coeff.abs())
            .fold(0.0, f64::max);
        let col_max = self
            .col_coeffs
            .iter()
            .filter(|&(r, &coeff)| !excluded_rows.contains(&r) && signed_room(r, coeff).is_finite())
            .map(|(_, coeff)| coeff.abs())
            .fold(0.0, f64::max);
        let noise_floor = ENTRY_ROUNDOFF * col_max_all;
        let pivot_floor = noise_floor.max(PIVOT_REL_TOL * col_max);

        // First, the largest step of the entering var after which no basic
        // var has left its working bound by more than its tolerance (Harris's
        // relaxed bounds, HiGHS's `relaxSpace` in `HEkkPrimal::chooseRow`),
        // counted from where the var sits: one already past its bound has
        // only the rest of its tolerance left, and one past it by all of it
        // admits no step toward it at all. The tolerance is the whole of the
        // one the phase exit verifies with, in the first round of
        // `run_phases`. A step that leaves a var at it, and a recomputation
        // that then finds the var an ulp past, give a violation like any
        // other, which the dual simplex removes after the cleanup, and each
        // further round allows a hundredth of the tolerance more (see
        // `LOOP_HARRIS_REDUCTION`); cutting the allowance by the round-off
        // of recomputing the var instead (as once done) left no allowance at
        // all on rows above about 5e6 in magnitude, where scaling is worst
        // (C3-10 of the solver review).
        //
        // The limit is a step, never below zero. In the first round every basic
        // var lies within its working bounds up to the tolerance
        // (`correct_primal`), so no limit falls below zero beyond round-off, as
        // none does in HiGHS, whose shifts keep every var within the tolerance
        // its ratio test allows. In a later round a var can sit past its bound
        // by more than the allowance: it admits no step toward it, a limit of
        // zero, not the negative step that would take it back within the
        // allowance, which is no step the simplex takes (see below). A negative
        // limit let only the rows of vars past their bounds qualify, the most
        // negative from the smallest entries, and passed over a row at its
        // bound with an entry of one for an entry of `1.3e-8` below the pivot
        // floor (review of R1b,
        // `a_var_past_its_bound_beyond_the_allowance_limits_the_step_to_zero`).
        // SoPlex's limit is never negative either: it counts a violating var's
        // allowance from where the var sits (`SPxFastRT::maxDelta`).
        let mut max_step = (entering_other_val - entering_cur_val).abs();
        for (r, &coeff) in self.col_coeffs.iter() {
            let coeff_abs = coeff.abs();
            if coeff_abs < noise_floor || excluded_rows.contains(&r) {
                continue;
            }

            let tol = harris_scale * self.var_tols[self.basic_vars[r]].toward(moves_up(coeff));
            let cur_step = (signed_room(r, coeff) + tol) / coeff_abs;
            if cur_step < max_step {
                max_step = cur_step;
            }
        }
        let max_step = max_step.max(0.0);

        // Second, among the rows whose own step (to the bound itself; below
        // zero for a var past its bound) is within that limit, the one with
        // the largest entry leaves: a numerically more stable basis at the
        // price of a tolerance-level infeasibility of the others. Entries
        // above the pivot floor are preferred; the row that set the limit
        // always qualifies, so a blocked step always finds a pivot.
        let mut leaving_r = None;
        let mut leaving_new_val = 0.0;
        let mut pivot_coeff_abs = f64::NEG_INFINITY;
        let mut pivot_coeff = 0.0;
        for floor in [pivot_floor, noise_floor] {
            for (r, &coeff) in self.col_coeffs.iter() {
                let coeff_abs = coeff.abs();
                if coeff_abs < floor || excluded_rows.contains(&r) {
                    continue;
                }

                let cur_step = signed_room(r, coeff) / coeff_abs;
                if cur_step.is_finite() && cur_step <= max_step && coeff_abs > pivot_coeff_abs {
                    leaving_r = Some(r);
                    leaving_new_val = if moves_up(coeff) {
                        self.basic_var_maxs[r]
                    } else {
                        self.basic_var_mins[r]
                    };
                    pivot_coeff = coeff;
                    pivot_coeff_abs = coeff_abs;
                }
            }
            if leaving_r.is_some() {
                break;
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

            // The step takes the leaving var exactly onto its working bound
            // (HiGHS's `theta_primal`, `HEkkPrimal::considerBoundSwap`), and
            // a var fixed in the model onto its value, whatever the phase did
            // to its working bounds (HiGHS's `adjustPerturbedEquationOut`: an
            // equality leaves the basis holding at its right-hand side). But
            // where the leaving var sits past that target (within its
            // tolerance, as the Harris pass allows), the step would run
            // backwards: the entering var would move against its improving
            // direction by the overshoot over the element, and every basic
            // var with it. The step is zero instead, and the leaving var
            // leaves where it sits, its working bound moved there
            // (`hold_leaving_value`): SoPlex's `SPxFastRT::maxReLeave`, which
            // takes a backward step only when it moves no basic var by more
            // than the tolerance, a bound no one tolerance gives here. HiGHS
            // takes the step: under its uniform tolerance of `1e-7` the
            // overshoot, divided by the element, stays harmless; under the
            // engine's, a var's is capped by what its rows can absorb, and
            // the round-off a leaving var carried (an ulp of 128), divided by
            // an element of 5e-5, took the entering var `1.5e-10` past a
            // bound whose tolerance was `1e-10`. The zero step replaces a
            // step clamped to zero with the leaving var left where it sat off
            // its bound, unrecorded (C3-21 of the solver review), and before
            // it the refusal of such a column, which declared optimal a vertex
            // with an improving direction left, 3e-3 short of the optimum.
            let leaving_var = self.basic_vars[row];
            if self.orig_var_mins[leaving_var] == self.orig_var_maxs[leaving_var] {
                leaving_new_val = self.orig_var_mins[leaving_var];
            }
            let raw_diff = (self.basic_var_vals[row] - leaving_new_val) / pivot_coeff;
            let forward = if entering_diff_sign {
                raw_diff >= 0.0
            } else {
                raw_diff <= 0.0
            };
            let entering_diff = if forward {
                raw_diff
            } else {
                leaving_new_val = self.basic_var_vals[row];
                0.0
            };
            let entering_new_val = entering_cur_val + entering_diff;

            Ok(PivotChoice::Pivot(PivotInfo {
                col: entering_c,
                entering_new_val,
                entering_diff,
                elem: Some(PivotElem {
                    row,
                    coeff: pivot_coeff,
                    dual_step_coeff: pivot_coeff,
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

    /// The row to leave the basis: the violated row with the best dual
    /// steepest-edge score, not counting a row whose basis change is taboo
    /// (HiGHS's `HEkk::applyTabooRowOut` in `HEkkDual::chooseRow`), and the
    /// bound its basic var leaves at.
    fn choose_pivot_row_dual(&self) -> Option<(usize, f64)> {
        let any_taboo = self.cycle_guard.any_taboo();
        self.best_violated_row(|r| !(any_taboo && self.cycle_guard.is_taboo(r)))
            .map(|r| (r, self.leaving_target(r)))
    }

    /// The violated row with the best dual steepest-edge score among those
    /// that no var can repair and that do not prove infeasibility (see
    /// [`CycleGuard::add_failed_proof`]): the row SoPlex's instable leave
    /// takes (see [`Self::dual_phase`]).
    fn unproven_row(&self) -> Option<usize> {
        self.best_violated_row(|r| self.cycle_guard.is_unproven(r))
    }

    /// The violated row with the best dual steepest-edge score (the largest
    /// violation when the weights are off) among the rows `eligible` admits.
    fn best_violated_row(&self, eligible: impl Fn(usize) -> bool) -> Option<usize> {
        let infeasibilities = self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
            .enumerate()
            .filter_map(|(r, ((&val, &min), &max))| {
                let tol = self.var_tols[self.basic_vars[r]];
                if !eligible(r) {
                    None
                } else if val < min - tol.lo {
                    Some((r, min - val))
                } else if val > max + tol.hi {
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
        leaving_r
    }

    /// The bound the basic var of the violated `row` leaves at: the one it
    /// violates.
    fn leaving_target(&self, r: usize) -> f64 {
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
        if val < min {
            min
        } else if val > max {
            max
        } else {
            unreachable!();
        }
    }

    /// The dual ratio test for the leaving `row`, with bound flipping.
    ///
    /// The leaving var must move to `leaving_new_val`. A non-basic var whose
    /// entry in the row can carry that move does so by entering the basis
    /// (a pivot) or, when both its bounds are finite, by moving to its other
    /// bound (a flip, which leaves the basis as it is). The vars are taken
    /// in the order the dual step brings their reduced costs to zero (their
    /// breakpoints), in Harris groups as HiGHS's `HEkkDualRow::chooseFinal`
    /// takes them: while the groups so far still leave the leaving var
    /// beyond its bound, the step passes the next group; the pivot is the
    /// largest entry of the last group (or of an earlier one, when the last
    /// group has no stable entry), and every group before the pivot's is
    /// flipped. When the first group can carry the move, that is exactly the
    /// Harris pass with nothing flipped. The step passes the breakpoint of
    /// every var it flips, so each flipped var's reduced cost crosses zero
    /// and the var is dual feasible at the bound it moved to; and it flips
    /// only on a positive dual step. This is the bound-flipping
    /// ("long-step") ratio test of the dual simplex (Maros, EJOR 149, 2003;
    /// Koberstein, PhD thesis, Paderborn 2005): where vars are boxed
    /// (binaries) one pivot does the work of several.
    ///
    /// A var's breakpoint is its reduced cost on its feasible side (its
    /// `slack`: the reduced cost itself for a var moving up, its negative
    /// for one moving down) over its entry, as HiGHS's `workMove * workDual`
    /// over `alpha`. The Harris pass lets a reduced cost cross zero by its
    /// tolerance, so a slack can be slightly negative; nothing clamps it. If
    /// the entering var's is not positive, the step is zero and the var's
    /// cost is shifted ([`DualStep::zero_step`]), instead of a step
    /// backwards (HiGHS takes `workTheta = 0` and `shiftCost`s). If a
    /// candidate's slack is below minus its tolerance, no step is dual
    /// feasible, and the reduced costs are to be recomputed and corrected
    /// first ([`DualRatioTest::DualInfeasible`]). An entry of the row, or the
    /// reduced cost of a var the test weighs, that is not a finite number
    /// fails the test with an `InternalError` ([`Self::finite_reduced_cost`]):
    /// it would compare false with every bound and drop out unseen, leaving
    /// the row to the vars left.
    ///
    /// The row's round-off scale is its largest entry over the columns that
    /// are not excluded and that the scalings made comparable: the entries
    /// of one row share the round-off of one transposed solve, so a var that
    /// cannot help still measures it, but the column of a var fixed when the
    /// model entered the engine keeps the user's units (a fixed `1e17` beside
    /// a genuine `1`) and measures only its own data. An entry below the
    /// round-off of computing the row (`ENTRY_ROUNDOFF` of the scale) is no
    /// candidate.
    ///
    /// When no var can enter, a var that can absorb the leaving var's move
    /// by crossing its own bound within its tolerance enters instead, on a
    /// zero dual step (see `absorbs` below). When none can either, the test
    /// says so ([`DualRatioTest::NoCandidate`]) and draws no conclusion:
    /// whether the row proves the model infeasible is decided on the model's
    /// data ([`Self::prove_infeasible`], called by [`Self::dual_phase`]), not
    /// on this row, whose entries were computed through the factorization
    /// and summed in `f64`, and of which those below the round-off floor
    /// count here as the zero they may be.
    ///
    /// `instable` makes the test SoPlex's last resort for a row that no var
    /// can repair and that does not prove infeasibility, once no other row
    /// is left (the "instable leave" of `SPxSolverBase::solve`,
    /// src/soplex/spxsolve.hpp, and `SPxBoundFlippingRT::selectEnter`,
    /// src/soplex/spxboundflippingrt.hpp): every entry of at least
    /// [`INSTABLE_PIVOT_TOL`] is a candidate, SoPlex's `lowstab`, instead of
    /// the relative floors.
    ///
    /// `excluded` lists non-basic columns judged numerically zero in this row
    /// (their entry disagreed with the column computation on a fresh
    /// factorization) and therefore not candidates.
    fn choose_entering_col_dual(
        &self,
        row: usize,
        leaving_new_val: f64,
        excluded: &[usize],
        instable: bool,
    ) -> Result<DualRatioTest, Error> {
        // True if the new obj. coeff. must be nonnegative in a dual-feasible configuration.
        let leaving_diff_sign = leaving_new_val > self.basic_var_vals[row];

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
        // and rounding rely on it sitting exactly on its bounds. Such a
        // pivot is no step of the dual simplex (the var moves the way its
        // bound forbids), so it takes a zero dual step, its cost shifted as
        // any entering var's with a reduced cost on the wrong side: it
        // changes no reduced cost, where a step `d_q / a_q` would have run
        // backwards by any amount (C2-12 of the solver review). Cost
        // shifting does not make the fallback unnecessary: the violations it
        // repairs are primal, from the engine's tolerances (half the
        // contract for a row, a bound tolerance capped by the rows' reserve),
        // and no dual step removes them.
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
            // its row's magnitude at the current point. A structural var
            // crosses the bound in the direction the pivot moves it.
            let tol = if var < self.num_vars {
                self.var_tols[var].toward(leaving_move / coeff > 0.0)
            } else {
                self.current_slack_tol(var - self.num_vars)
            };
            // The comparison itself is between two rounded quantities.
            !integer && (leaving_move / coeff).abs() <= tol * (1.0 + ROUNDOFF_FLOOR)
        };

        // The round-off scale of the row: its largest entry over the columns
        // that are not excluded and that the scalings made comparable (every
        // column but those of vars fixed when the model entered the engine,
        // which keep the user's units). The entries of one row all come from
        // the same transposed solve `rho_r`, whose round-off is spread over
        // its components in proportion to the largest, so the entry of a var
        // that cannot help this row still measures it (a helping entry of
        // 1.1e-16 beside a slack's 1.05 was the residue of two parallel rows
        // cancelling, and pivoting on it left a singular basis); but an
        // unscaled fixed column's `1e17` measures its own data, and taken
        // into the scale it made a genuine entry of one "noise" and the row a
        // false proof of infeasibility. An entry below `ENTRY_ROUNDOFF` of
        // this scale is noise, never a pivot (see the verdict below).
        //
        // An entry that is not a finite number is a breakdown, not data: it
        // compares false with every floor below, so its var would drop out of
        // the test silently, and the row could prove infeasibility without
        // it. It is the loud error it is instead (as a reduced cost that is
        // not finite, see `finite_reduced_cost`).
        let mut scale = 0.0f64;
        for (c, &coeff) in self.row_coeffs.iter() {
            let var = self.nb_vars[c];
            if !coeff.is_finite() {
                return Err(Error::InternalError(format!(
                    "dual simplex: entry of var {var} in row {row} is not finite: {coeff}"
                )));
            }
            if !excluded.contains(&c) && !(var < self.num_vars && self.unscaled_cols[var]) {
                scale = scale.max(coeff.abs());
            }
        }
        let noise_floor = ENTRY_ROUNDOFF * scale;

        // The leaving var's distance from its bound, and the tolerance within
        // which it counts as there.
        let infeasibility = leaving_move.abs();
        let leaving_tols = self.var_tols[self.basic_vars[row]];
        let target_tol = if leaving_diff_sign {
            leaving_tols.lo
        } else {
            leaving_tols.hi
        };

        // Every var that can help, as a breakpoint of the dual step: where
        // its reduced cost reaches zero (`ratio`, and the Harris-relaxed
        // ratio), the bound it moves towards, and how far a flip onto that
        // bound carries the leaving var (infinite for an infinite bound).
        struct Breakpoint {
            col: usize,
            coeff: f64,
            ratio: f64,
            relaxed_ratio: f64,
            bound: f64,
            capacity: f64,
        }
        let breakpoint = |c: usize, coeff: f64| -> Breakpoint {
            let var = self.nb_vars[c];
            let up = (coeff > 0.0) != leaving_diff_sign;
            let bound = if up {
                self.orig_var_maxs[var]
            } else {
                self.orig_var_mins[var]
            };
            // The reduced cost on the var's feasible side: HiGHS's
            // `workMove * workDual`, negative for a reduced cost the Harris
            // pass let cross zero.
            let d = self.nb_var_obj_coeffs[c];
            let slack = if up { d } else { -d };
            Breakpoint {
                col: c,
                coeff,
                ratio: slack / coeff.abs(),
                relaxed_ratio: (slack + self.dual_tols[var]) / coeff.abs(),
                bound,
                capacity: coeff.abs() * (bound - self.nb_var_vals[c]).abs(),
            }
        };
        // The pivot floor: an entry seven orders below the largest entry
        // among the vars that can enter and are not excluded (see
        // PIVOT_REL_TOL) is not a candidate while a larger one can take the
        // step; an excluded column, whose entry proved to be round-off, no
        // longer sets it. Every candidate is above the noise floor. SoPlex's
        // instable leave takes every entry above its absolute `lowstab`.
        let mut row_max = 0.0f64;
        for (c, &coeff) in self.row_coeffs.iter() {
            if direction_ok(coeff, &self.nb_var_states[c]) && !excluded.contains(&c) {
                row_max = row_max.max(coeff.abs());
            }
        }
        let pivot_floor = if instable {
            INSTABLE_PIVOT_TOL
        } else {
            noise_floor.max(PIVOT_REL_TOL * row_max)
        };
        // The vars that could enter regularly (the two-pass's candidates).
        let candidate = |c: usize, coeff: f64| {
            coeff.abs() >= pivot_floor
                && direction_ok(coeff, &self.nb_var_states[c])
                && !excluded.contains(&c)
        };

        // The Harris bound on the step: the smallest relaxed ratio. Below
        // zero, a candidate's reduced cost is beyond its tolerance on its
        // infeasible side and no step is dual feasible (HiGHS's `workTheta
        // <= 0`): the reduced costs are to be corrected first. Every
        // candidate passes here first, so from here on every candidate's
        // reduced cost is finite and every ratio a number.
        let mut max_step = f64::INFINITY;
        for (c, &coeff) in self.row_coeffs.iter() {
            if candidate(c, coeff) {
                self.finite_reduced_cost(c)?;
                max_step = max_step.min(breakpoint(c, coeff).relaxed_ratio);
            }
        }
        if max_step < 0.0 {
            return Ok(DualRatioTest::DualInfeasible);
        }
        // The Harris pass: among the candidates whose ratio is within the
        // bound, the one with the largest entry, for a stable basis at the
        // price of a slight dual infeasibility (Gill, Murray, Saunders &
        // Wright, "A practical anti-cycling procedure for linearly
        // constrained optimization", Mathematical Programming 45, 1989).
        // These candidates are also the first group of the bound-flipping
        // walk below, so the same pass measures how far flipping all of them
        // would carry the leaving var.
        let mut entering: Option<(usize, f64, f64)> = None;
        let mut first_group_capacity = 0.0;
        let mut beyond_first_group = false;
        for (c, &coeff) in self.row_coeffs.iter() {
            if candidate(c, coeff) {
                let b = breakpoint(c, coeff);
                if b.ratio <= max_step {
                    first_group_capacity += b.capacity;
                    if entering.is_none_or(|(_, best, _)| coeff.abs() > best.abs()) {
                        entering = Some((c, coeff, b.ratio));
                    }
                } else {
                    beyond_first_group = true;
                }
            }
        }

        // Bound flipping, as HiGHS's `HEkkDualRow::chooseFinal`
        // (`chooseFinalWorkGroupQuad`, `chooseFinalLargeAlpha`,
        // highs/simplex/HEkkDualRow.cpp). The breakpoints are taken in Harris
        // groups: the first group is the candidates the Harris pass above
        // chose from, each next one the remaining candidates whose ratio is
        // within the smallest relaxed ratio among the remaining, until the
        // groups so far can carry the leaving var within its tolerance of
        // its bound (or every candidate is grouped). The pivot is the largest
        // entry of the last group, unless that group has no entry above a
        // tenth of the largest grouped entry (or above one): the scan then
        // steps back to the last group that has one. Every group before the
        // pivot's is flipped; the pivot's own group and the rest are not.
        // Ties go to the first entry of the row, as in the Harris pass.
        //
        // So when the first group can carry the move, nothing is flipped and
        // the pivot is exactly the Harris pass's (the common case, found
        // without collecting anything); and among breakpoints the step
        // reaches together, the pivot is the largest entry, not whichever a
        // one-at-a-time walk happened to stop at (which flipped an entry of
        // 10 and pivoted on one of 1e-6 at the same ratio). A flip also never
        // comes with a zero dual step, as in both references (HiGHS flips
        // nothing when `workTheta == 0`, SoPlex's `SPxBoundFlippingRT` only
        // on a non-degenerate step): a group after the first holds only
        // ratios above the first group's bound, which is not negative, so
        // the pivot's own ratio, the dual step, is positive whenever there
        // is a group to flip. If the leaving var is still short after every
        // group, the pivot's step carries the entering var past its other
        // bound, which a later iteration repairs, as in HiGHS.
        if entering.is_some()
            && beyond_first_group
            && infeasibility - first_group_capacity > target_tol
        {
            let mut pending: Vec<Breakpoint> = self
                .row_coeffs
                .iter()
                .filter(|&(c, &coeff)| candidate(c, coeff))
                .map(|(c, &coeff)| breakpoint(c, coeff))
                .collect();
            let mut groups: Vec<Vec<Breakpoint>> = Vec::new();
            let mut carried = 0.0;
            let mut bound = max_step;
            loop {
                let (group, rest): (Vec<Breakpoint>, Vec<Breakpoint>) =
                    pending.into_iter().partition(|b| b.ratio <= bound);
                pending = rest;
                // The smallest relaxed ratio's own breakpoint always joins
                // (its ratio is below its relaxed ratio, and every ratio is
                // a number: the Harris bound above failed on a reduced cost
                // that is not), so no group is empty; the check only keeps
                // the walk from running on if one were.
                if group.is_empty() {
                    break;
                }
                carried += group.iter().map(|b| b.capacity).sum::<f64>();
                groups.push(group);
                if infeasibility - carried <= target_tol || pending.is_empty() {
                    break;
                }
                bound = pending
                    .iter()
                    .map(|b| b.relaxed_ratio)
                    .fold(f64::INFINITY, f64::min);
            }
            let largest = |group: &[Breakpoint]| -> (usize, f64, f64) {
                let mut best = (group[0].col, group[0].coeff, group[0].ratio);
                for b in &group[1..] {
                    if b.coeff.abs() > best.1.abs() {
                        best = (b.col, b.coeff, b.ratio);
                    }
                }
                best
            };
            let largest_grouped = groups
                .iter()
                .flatten()
                .map(|b| b.coeff.abs())
                .fold(0.0, f64::max);
            let final_compare = (0.1 * largest_grouped).min(1.0);
            // The group holding the largest entry always qualifies, so the
            // backward scan stops there at the latest.
            let (pivot_group, (col, coeff, ratio)) = (0..groups.len())
                .rev()
                .map(|g| (g, largest(&groups[g])))
                .find(|&(_, (_, coeff, _))| coeff.abs() > final_compare)
                .expect("the group of the largest grouped entry qualifies as the pivot's");
            let flips: Vec<(usize, f64)> = groups[..pivot_group]
                .iter()
                .flatten()
                .map(|b| (b.col, b.bound))
                .collect();
            debug_assert!(
                flips.is_empty() || ratio > 0.0,
                "bound flips with a zero dual step"
            );
            return Ok(DualRatioTest::Step(DualStep {
                flips,
                entering: (col, coeff),
                zero_step: ratio <= 0.0,
            }));
        }
        if let Some((col, coeff, ratio)) = entering {
            return Ok(DualRatioTest::Step(DualStep {
                flips: Vec::new(),
                entering: (col, coeff),
                zero_step: ratio <= 0.0,
            }));
        }

        // No var can enter regularly. A var that can absorb the move within
        // its tolerance enters instead, on a zero dual step with its cost
        // shifted by its whole reduced cost (see `absorbs` above), and the
        // choice is the absorber whose shift can do the least harm when the
        // var leaves again and the shift is given back (HiGHS's `shiftBack`):
        // a fixed var's none, since any reduced cost is feasible for it, any
        // other's as much as its reduced cost. So the ratio is that reduced
        // cost (zero for a fixed var) over the entry, and among the absorbers
        // within the Harris relaxation of the smallest ratio the one with the
        // largest entry enters, for a stable basis; as among the regular
        // candidates, an entry `PIVOT_REL_TOL` below the largest absorbing
        // entry is no candidate. (Taking the absorber with the largest entry
        // instead left a model of the differential fuzz above its optimum:
        // see `wide_lp_558_reaches_the_optimum` in `src/tests/magnitudes.rs`.)
        let absorber_ratio = |c: usize, coeff: f64, tol: f64| -> f64 {
            let state = &self.nb_var_states[c];
            let d = if state.at_min && state.at_max {
                0.0
            } else {
                self.nb_var_obj_coeffs[c].abs()
            };
            (d + tol) / coeff.abs()
        };
        let mut absorbing_max = 0.0f64;
        for (c, &coeff) in self.row_coeffs.iter() {
            if !excluded.contains(&c) && absorbs(c, coeff) {
                absorbing_max = absorbing_max.max(coeff.abs());
            }
        }
        let absorbing_floor = noise_floor.max(PIVOT_REL_TOL * absorbing_max);
        let absorber = |c: usize, coeff: f64| {
            coeff.abs() >= absorbing_floor && !excluded.contains(&c) && absorbs(c, coeff)
        };
        let mut max_step = f64::INFINITY;
        for (c, &coeff) in self.row_coeffs.iter() {
            if absorber(c, coeff) {
                // Its whole reduced cost becomes its shift, so it must be a
                // number even for a fixed var.
                self.finite_reduced_cost(c)?;
                let tol = self.dual_tols[self.nb_vars[c]];
                max_step = max_step.min(absorber_ratio(c, coeff, tol));
            }
        }
        let mut absorbing: Option<(usize, f64)> = None;
        for (c, &coeff) in self.row_coeffs.iter() {
            if absorber(c, coeff)
                && absorber_ratio(c, coeff, 0.0) <= max_step
                && absorbing.is_none_or(|(_, best)| coeff.abs() > best.abs())
            {
                absorbing = Some((c, coeff));
            }
        }
        if let Some(entering) = absorbing {
            return Ok(DualRatioTest::Step(DualStep {
                flips: Vec::new(),
                entering,
                zero_step: true,
            }));
        }

        // No var can enter or absorb the move. Whether the row proves the
        // model infeasible is the proof's to decide (see the doc above).
        if log::log_enabled!(log::Level::Debug) {
            debug!(
                "no entering column for row {row} (basic var {} = {:e}, bounds [{:e}, {:e}], \
                 leaving to {leaving_new_val:e}, pivot floor {pivot_floor:e}{}); row entries:",
                self.basic_vars[row],
                self.basic_var_vals[row],
                self.basic_var_mins[row],
                self.basic_var_maxs[row],
                if instable { ", instable" } else { "" },
            );
            for (c, &coeff) in self.row_coeffs.iter() {
                let state = &self.nb_var_states[c];
                debug!(
                    "  var {} coeff {coeff:e} value {:e} at_min {} at_max {} tol {:?} \
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
        Ok(DualRatioTest::NoCandidate)
    }

    /// The reduced cost of the non-basic var at `col`, which the dual ratio
    /// test is about to weigh. One that is not a finite number compares false
    /// with every bound of the test, so its var would drop out of it unseen:
    /// working costs perturbed to infinity once left a row of a feasible
    /// model with no candidate but such vars, and its verdict was a false
    /// proof of infeasibility. It is the loud error it is instead, as in the
    /// primal pricing ([`Self::price_primal`]).
    fn finite_reduced_cost(&self, col: usize) -> Result<f64, Error> {
        let d = self.nb_var_obj_coeffs[col];
        if d.is_finite() {
            Ok(d)
        } else {
            Err(Error::InternalError(format!(
                "reduced cost of var {} is not finite: {d}",
                self.nb_vars[col]
            )))
        }
    }

    /// Move each non-basic column in `flips` to the value given with it (its
    /// other bound, its perturbed bound when a primal phase starts (see
    /// [`bounds`]), or for the last var the round-off repair moves a value
    /// between its bounds (see [`Self::round_off_repair`]); either way the
    /// basis stays as it is) and carry the moves
    /// into the basic values, `x_B -= B^-1 Σ Δx_c a_c`, through one solve
    /// with the combined column. The objective moves by the reduced costs,
    /// as in a pivot.
    fn apply_flips(&mut self, flips: &[(usize, f64)]) {
        if flips.is_empty() {
            return;
        }
        self.values_dirty = true;
        self.flip_rhs.clear_and_resize(self.num_constraints());
        for &(col, new_val) in flips {
            let var = self.nb_vars[col];
            let diff = new_val - self.nb_var_vals[col];
            self.cur_obj_val += self.nb_var_obj_coeffs[col] * diff;
            self.nb_var_vals[col] = new_val;
            let state = self.nonbasic_state(var, new_val);
            if state.at_upper() != self.nb_var_states[col].at_upper() {
                self.cycle_guard.apply(VertexChange::Upper(var));
            }
            self.nb_var_states[col] = state;
            //guaranteed to be a valid index
            for (r, &coeff) in self.orig_constraints_csc.outer_view(var).unwrap().iter() {
                *self.flip_rhs.get_mut(r) += diff * coeff;
            }
        }
        let delta = self.basis_solver.solve(self.flip_rhs.iter());
        for (r, &d) in delta.iter() {
            self.basic_var_vals[r] -= d;
        }
    }

    /// The state of the non-basic `var` at `val`: at a bound when within its
    /// tolerance of it. The bounds are the working bounds of a running
    /// primal phase (see [`bounds`]), where a non-basic var sits at a moved
    /// bound.
    fn nonbasic_state(&self, var: usize, val: f64) -> NonBasicVarState {
        let tol = self.var_tols[var];
        let (min, max) = self.working_bounds(var);
        NonBasicVarState {
            at_min: at_bound(val, min, tol.lo),
            at_max: at_bound(val, max, tol.hi),
        }
    }

    /// The state the basic `var` takes when it leaves the basis at `val`
    /// ([`Self::nonbasic_state`]), except that a var fixed in the model is
    /// fixed where it leaves, at both of its working bounds (see
    /// [`Self::hold_leaving_value`]).
    fn leaving_state(&self, var: usize, val: f64) -> NonBasicVarState {
        if self.orig_var_mins[var] == self.orig_var_maxs[var] {
            NonBasicVarState {
                at_min: true,
                at_max: true,
            }
        } else {
            self.nonbasic_state(var, val)
        }
    }

    /// The vertex changes (see [`CycleGuard`]) of the pivot that brings the
    /// non-basic var at `col` into the basis in place of the basic var of
    /// `row`, which leaves at `leaving_new_val`.
    fn pivot_vertex_changes(
        &self,
        col: usize,
        row: usize,
        leaving_new_val: f64,
    ) -> Vec<VertexChange> {
        let (entering, leaving) = (self.nb_vars[col], self.basic_vars[row]);
        let mut changes = Vec::with_capacity(4);
        if self.nb_var_states[col].at_upper() {
            changes.push(VertexChange::Upper(entering));
        }
        changes.push(VertexChange::Basic(entering));
        changes.push(VertexChange::Basic(leaving));
        if self.leaving_state(leaving, leaving_new_val).at_upper() {
            changes.push(VertexChange::Upper(leaving));
        }
        changes
    }

    /// The vertex changes (see [`CycleGuard`]) of a dual step: the flips of
    /// `step`, then the pivot ([`Self::pivot_vertex_changes`]).
    fn dual_step_vertex_changes(
        &self,
        step: &DualStep,
        col: usize,
        row: usize,
        leaving_new_val: f64,
    ) -> Vec<VertexChange> {
        let mut changes: Vec<VertexChange> = step
            .flips
            .iter()
            .filter(|&&(c, new_val)| {
                self.nonbasic_state(self.nb_vars[c], new_val).at_upper()
                    != self.nb_var_states[c].at_upper()
            })
            .map(|&(c, _)| VertexChange::Upper(self.nb_vars[c]))
            .collect();
        changes.extend(self.pivot_vertex_changes(col, row, leaving_new_val));
        changes
    }

    /// Clear `is_dual_feasible` if the non-basic var at `col` no longer
    /// justifies its reduced cost at the bound it now sits at: after a move
    /// that no dual step went with (a bound change).
    fn downgrade_dual_feasibility(&mut self, col: usize) {
        let var = self.nb_vars[col];
        self.is_dual_feasible = self.is_dual_feasible
            && self.nb_var_states[col]
                .is_dual_feasible(self.nb_var_obj_coeffs[col], self.dual_tols[var]);
    }

    /// Whether the basic var of `row` is within its bounds up to its
    /// tolerances: the test [`Self::choose_pivot_row_dual`] makes, with a
    /// value that is not a number counted as outside.
    fn basic_var_within_bounds(&self, row: usize) -> bool {
        let val = self.basic_var_vals[row];
        let tol = self.var_tols[self.basic_vars[row]];
        val >= self.basic_var_mins[row] - tol.lo && val <= self.basic_var_maxs[row] + tol.hi
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
            let state = self.nonbasic_state(entering_var, pivot_info.entering_new_val);
            if state.at_upper() != self.nb_var_states[pivot_info.col].at_upper() {
                self.cycle_guard.apply(VertexChange::Upper(entering_var));
            }
            self.nb_var_states[pivot_info.col] = state;
            return Ok(());
        }
        //guaranteed, none variant already handled
        let pivot_elem = pivot_info.elem.as_ref().unwrap();
        // The element the eta file and the steepest-edge updates apply with
        // the column must be the column's own (see `PivotElem::coeff`).
        debug_assert_eq!(pivot_elem.coeff, self.col_coeff_at(pivot_elem.row));
        let pivot_coeff = pivot_elem.coeff;

        // Update basic vars stuff

        for (r, coeff) in self.col_coeffs.iter() {
            if r == pivot_elem.row {
                self.basic_var_vals[r] = pivot_info.entering_new_val;
            } else {
                self.basic_var_vals[r] -= pivot_info.entering_diff * coeff;
            }
        }

        let (entering_min, entering_max) = self.working_bounds(entering_var);
        self.basic_var_mins[pivot_elem.row] = entering_min;
        self.basic_var_maxs[pivot_elem.row] = entering_max;

        if self.enable_dual_steepest_edge {
            self.update_dual_sq_norms(pivot_elem.row, pivot_coeff)?;
        }

        // Update non-basic vars stuff

        let leaving_var = self.basic_vars[pivot_elem.row];

        self.nb_var_vals[pivot_info.col] = pivot_elem.leaving_new_val;
        let leaving_state = self.nonbasic_state(leaving_var, pivot_elem.leaving_new_val);
        if self.nb_var_states[pivot_info.col].at_upper() {
            self.cycle_guard.apply(VertexChange::Upper(entering_var));
        }
        if leaving_state.at_upper() {
            self.cycle_guard.apply(VertexChange::Upper(leaving_var));
        }
        self.nb_var_states[pivot_info.col] = leaving_state;

        let pivot_obj = self.nb_var_obj_coeffs[pivot_info.col] / pivot_elem.dual_step_coeff;
        for (c, &coeff) in self.row_coeffs.iter() {
            if c == pivot_info.col {
                self.nb_var_obj_coeffs[c] = -pivot_obj;
            } else {
                self.nb_var_obj_coeffs[c] -= pivot_obj * coeff;
            }
        }
        // A leaving var gives back the shift a zero dual step gave its cost
        // when it entered (HiGHS's `HEkkDual::shiftBack`): it is non-basic
        // again, and its reduced cost is that of its unshifted cost.
        if self.costs_modified {
            let shift = std::mem::take(&mut self.cost_shifts[leaving_var]);
            if shift != 0.0 {
                self.nb_var_obj_coeffs[pivot_info.col] -= shift;
                self.cur_obj_val -= shift * pivot_elem.leaving_new_val;
            }
        }

        if self.enable_primal_steepest_edge {
            self.update_primal_sq_norms(pivot_info.col, pivot_coeff)?;
        }

        // Update basis itself

        self.basic_vars[pivot_elem.row] = entering_var;
        self.var_states[entering_var] = VarState::Basic(pivot_elem.row);
        self.nb_vars[pivot_info.col] = leaving_var;
        self.var_states[leaving_var] = VarState::NonBasic(pivot_info.col);
        self.cycle_guard.apply(VertexChange::Basic(entering_var));
        self.cycle_guard.apply(VertexChange::Basic(leaving_var));
        self.cycle_guard.record_pivot();

        // A simple heuristic to choose when to recompute LU factorization.
        // Note: a possible failure mode is that the LU factorization accidentally
        // generates a lot of fill-in and doesn't get recomputed for a long time.
        // A refactorization renews only the factorization: the values are
        // recomputed from it when the check once per basis turnover finds
        // them off the rows (see `refactorize`), and the reduced costs only
        // by a rebuild, a basis load or the phase exit, so incremental
        // round-off can outlive an eta file until the phase exit verifies.
        let eta_matrices_nnz = self.basis_solver.eta_matrices.coeff_cols.nnz();
        if eta_matrices_nnz < self.basis_solver.lu_factors.nnz() {
            self.basis_solver
                .push_eta_matrix(&self.col_coeffs, pivot_elem.row, pivot_coeff);
        } else {
            self.refactorize()?;
        }
        Ok(())
    }

    /// Update the primal steepest-edge weights `1 + ||B^-1 a_j||^2` for the
    /// pivot on `pivot_coeff` (the column's element, see
    /// [`PivotElem::coeff`]). The update is exact in exact arithmetic, but
    /// cancellation can take a weight below the true one, even to zero or
    /// below, where it would make its column always or never the choice:
    /// every updated weight is floored at `1 + lambda^2` (`lambda` the
    /// column's entry in the pivot row over the pivot), a lower bound of its
    /// true value, as HiGHS's `HEkkPrimal::updatePrimalSteepestEdgeWeights`
    /// does. A weight the update leaves not finite (overflowed by a pivot
    /// below about 1e-154, or reached by a NaN) is the engine's error, not a
    /// panic, and is caught before the floor could make it finite.
    fn update_primal_sq_norms(
        &mut self,
        entering_col: usize,
        pivot_coeff: f64,
    ) -> Result<(), Error> {
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
            let weight = &mut self.primal_edge_sq_norms[c];
            if c == entering_col {
                *weight = pivot_sq_norm / pivot_coeff_sq;
            } else {
                *weight += -2.0 * r_coeff * self.sq_norms_update_helper[c] / pivot_coeff
                    + pivot_sq_norm * r_coeff * r_coeff / pivot_coeff_sq;
            }
            // Before the floor: `f64::max` would turn a NaN or `-inf` into
            // the floor and hide the breakdown.
            if !weight.is_finite() {
                return Err(Error::InternalError(format!(
                    "primal steepest-edge weight is not finite after a pivot on {pivot_coeff:e}"
                )));
            }
            if c != entering_col {
                let lambda = r_coeff / pivot_coeff;
                *weight = weight.max(1.0 + lambda * lambda);
            }
        }
        Ok(())
    }

    /// Update the dual steepest-edge weights `||e_r^T B^-1||^2` for the pivot
    /// on `pivot_coeff` (the column's element, see [`PivotElem::coeff`]):
    /// the pivot row's new weight exactly, from the row just computed, the
    /// others by the Forrest-Goldfarb update. A weight that starts as an
    /// underestimate (every weight is reset to one when a basis is loaded)
    /// can be driven to zero or below by the update, where its row would
    /// dominate the choice or never be chosen: every updated weight is
    /// floored at [`MIN_DUAL_EDGE_WEIGHT`], as HiGHS's
    /// `HEkk::updateDualSteepestEdgeWeights` does. A weight the update leaves
    /// not finite (overflowed by a pivot below about 1e-154, or reached by a
    /// NaN) is the engine's error, not a panic, and is caught before the
    /// floor could make it finite.
    fn update_dual_sq_norms(&mut self, leaving_row: usize, pivot_coeff: f64) -> Result<(), Error> {
        // Computations for the dual steepest edge pivoting rule.
        // See the same reference (Forrest, Goldfarb).

        let tau = self.basis_solver.solve(self.inv_basis_row_coeffs.iter());

        // Calculate pivot_sq_norm directly to avoid loss of precision.
        let pivot_sq_norm = self.inv_basis_row_coeffs.sq_norm();
        // assert!((self.dual_edge_sq_norms[leaving_row] - pivot_sq_norm).abs() < 0.1);

        let pivot_coeff_sq = pivot_coeff * pivot_coeff;
        for (r, &col_coeff) in self.col_coeffs.iter() {
            let weight = &mut self.dual_edge_sq_norms[r];
            if r == leaving_row {
                *weight = pivot_sq_norm / pivot_coeff_sq;
            } else {
                *weight += -2.0 * col_coeff * tau.get(r) / pivot_coeff
                    + pivot_sq_norm * col_coeff * col_coeff / pivot_coeff_sq;
            }
            // Before the floor: `f64::max` would turn a NaN or `-inf` into
            // the floor and hide the breakdown (HiGHS's `std::max` floors a
            // NaN; the engine's rule is to fail loudly).
            if !weight.is_finite() {
                return Err(Error::InternalError(format!(
                    "dual steepest-edge weight is not finite after a pivot on {pivot_coeff:e}"
                )));
            }
            if r != leaving_row {
                *weight = weight.max(MIN_DUAL_EDGE_WEIGHT);
            }
        }
        Ok(())
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
        // the values are dirty until the next phase exit checks them.
        // Clearing the flag here let a phase exit skip verification right
        // after a refactorization had recomputed the values, and report a
        // point that missed a row by four times its tolerance (the magnitude
        // fixture at 1e8); leaving it as it was let a node reached by a basis
        // load finish on values nothing had checked.
        self.values_dirty = true;
        self.pivots_since_drift_check = 0;
        Ok(())
    }

    /// Recompute the reduced costs and the objective from the costs the
    /// simplex works with: the cost vector in force (the phase-1 artificial
    /// objective while it is active, else the real one), plus the
    /// perturbation and shifts of a running dual phase (see [`costs`]).
    fn recalc_obj_coeffs(&mut self) -> Result<(), Error> {
        let costs = self
            .artificial_obj
            .as_deref()
            .unwrap_or(&self.orig_obj_coeffs);
        let (perturbation, shifts) = (&self.cost_perturbation, &self.cost_shifts);
        let modified = self.costs_modified;
        let cost = |var: usize| {
            if modified {
                costs[var] + perturbation[var] + shifts[var]
            } else {
                costs[var]
            }
        };
        // Same as recalc_basic_var_vals: pending etas participate in the
        // (transposed) dense solve instead of forcing a refactorization.
        let mut multipliers = vec![0.0; self.basic_vars.len()];
        for (c, &var) in self.basic_vars.iter().enumerate() {
            multipliers[c] = cost(var);
        }
        self.basis_solver
            .solve_transp_dense_with_etas(&mut multipliers);

        self.nb_var_obj_coeffs.clear();
        for &var in &self.nb_vars {
            //guaranteed to be a valid index
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let dot_prod: f64 = col.iter().map(|(r, val)| val * multipliers[r]).sum();
            self.nb_var_obj_coeffs.push(cost(var) - dot_prod);
        }
        self.refresh_dual_tols(&multipliers);

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
