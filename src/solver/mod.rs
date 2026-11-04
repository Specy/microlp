//! The bounded-variable simplex engine.
//!
//! [`Solver`] is the one long-lived LP instance a solve (LP or MILP) works on:
//! it holds the scaled problem, the current basis, and the pivot loops. Every
//! bound change or added row goes through it; nothing above it addresses rows
//! by index.
//!
//! * [`tolerances`] — the tolerance model (magnitudes, floors, pivot limits).
//! * [`scaling`] — user problem to scaled representation, and vertex seeding.
//! * [`basis`] — the basis inverse ([`BasisSolver`]) and pivot descriptions.
//! * [`costs`] — the dual simplex's working costs: perturbation, shifting,
//!   correction of dual infeasibilities, and cleanup.
//! * [`bounds`] — the primal simplex's working bounds: perturbation, shifting,
//!   and cleanup.
//! * [`cycling`] — cycling and stall detection for both.
//! * [`proof`] — the infeasibility proof the dual simplex needs before it may
//!   return `Infeasible`.

mod basis;
mod bounds;
mod costs;
mod cycling;
mod proof;
mod scaling;
mod tolerances;

pub(crate) use basis::*;
pub(crate) use proof::CDouble;
use proof::{compensated_error, refinement_error};
pub(crate) use scaling::*;
pub(crate) use tolerances::*;

use cycling::{CycleGuard, VertexChange, STALL_ITERATIONS};

use core::time::Duration;

use crate::{
    helpers::{resized_view, to_dense},
    lu::{lu_factorize, ScratchSpace},
    range::CoefficientRange,
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

/// The deadline `limit` after `start`: none without a limit, nor for a limit
/// too long to add to `start`, which no solve outlasts.
pub(crate) fn deadline_after(start: Instant, limit: Option<Duration>) -> Deadline {
    limit.and_then(|limit| start.checked_add(limit))
}

/// The rank by which a basis repair's second factorization takes the column
/// of a var with these bounds ([`Solver::factorize_at`]): one with no finite
/// bound first, then one finite bound, then two, a fixed var last (Bixby
/// 1992's order of preference for the basis).
fn bound_rank(min: f64, max: f64) -> u8 {
    match (min.is_finite(), max.is_finite()) {
        _ if min == max => 3,
        (true, true) => 2,
        (false, false) => 0,
        _ => 1,
    }
}

/// Whether the basic var of a row where the entering var's column has
/// `coeff` moves up as the entering var moves up (`entering_up`) or down:
/// it changes by `-coeff` times the entering var's step.
fn moves_up(entering_up: bool, coeff: f64) -> bool {
    (entering_up && coeff < 0.0) || (!entering_up && coeff > 0.0)
}

/// A refined pivot column and the arithmetic its refinement discarded.
struct RefinedColumn {
    errors: Option<Vec<f64>>,
    values: Vec<CDouble>,
    residual_roundoff: Vec<f64>,
    value_roundoff: Vec<f64>,
}

/// A refined pivot column as the primal ratio test judges it
/// ([`Solver::judge_pivot_col`]).
struct ColumnJudgment {
    column: RefinedColumn,
    /// Whether the refinement converged: only then did it measure the error a
    /// ray past an unresolved entry needs.
    converged: bool,
    /// Whether it was made for a pivot element the plain computations
    /// disagreed on ([`element_verdict`]).
    disputed: bool,
    /// Per row, the error within which its entry is unresolved.
    errors: Vec<f64>,
    /// The rows whose error includes their round-off floor.
    floored: Vec<usize>,
    /// The entries (row, value) within their error.
    unresolved: Vec<(usize, f64)>,
}

impl ColumnJudgment {
    /// Take the entry `entry` of `row` as unresolved: its error grows to its
    /// size at least, and to `row_bound`, the most the refined row leaves it.
    fn unresolve(&mut self, row: usize, entry: f64, row_bound: f64) {
        self.errors[row] = self.errors[row].max(row_bound).max(entry.abs());
        self.unresolved.push((row, entry));
    }
}

/// What the primal ratio test makes of a refined pivot element
/// ([`element_verdict`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ElementVerdict {
    /// Pivot on it, with the column's value.
    Take,
    /// It neither bounds the step nor pivots.
    Unresolve,
    /// The refined row has it with the other sign: a loud error.
    Refuse,
}

/// The verdict on a pivot element the refined column computes as `col`,
/// within `col_error`, and the refined row as `row`, within `row_error`.
///
/// Taken where the two agree, where the row resolves it with the same sign,
/// or, beside a column whose refinement `converged`, where their errors
/// overlap: a row refined twice can stay unconverged where the column
/// converges. Refused where the row resolves it with the other sign, and
/// unresolved otherwise. A judgment made for a `disputed` element takes
/// nothing unless the column's refinement converged: a row and a column
/// refined alike through the same factors are the same number in exact
/// arithmetic, so their agreement proves nothing.
fn element_verdict(
    row: f64,
    row_error: f64,
    col: f64,
    col_error: f64,
    converged: bool,
    disputed: bool,
) -> ElementVerdict {
    if disputed && !converged {
        return ElementVerdict::Unresolve;
    }
    let row_resolves = row.abs() > row_error;
    if row_resolves && (row > 0.0) != (col > 0.0) {
        return ElementVerdict::Refuse;
    }
    let overlap = (row - col).abs() <= proof::add_up(col_error, row_error);
    if Solver::pivots_agree(row, col) || row_resolves || (converged && overlap) {
        ElementVerdict::Take
    } else {
        ElementVerdict::Unresolve
    }
}

/// The `InternalError` of a refinement of the pivot column of `var`, in the
/// `phase` simplex, that computes a number that is not finite.
fn refinement_breakdown(phase: &str, var: usize) -> Error {
    Error::InternalError(format!(
        "{phase} simplex: the refinement of the column of var {var} is not finite"
    ))
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
    /// simplex).
    Perturbation,
    /// The margin a shift leaves a var's reduced cost (dual simplex) or
    /// value (primal simplex) on its feasible side.
    Shift,
}

/// A pseudo-random fraction in `[0, 1)` fixed by the var and the stream, so
/// that every solve of a model, and every B&B node, is deterministic without
/// a generator state to carry around.
fn random_fraction(var: usize, stream: Stream) -> f64 {
    let salt = match stream {
        Stream::Perturbation => 0x243F_6A88_85A3_08D3,
        Stream::Shift => 0x1319_8A2E_0370_7344,
    };
    (mix(var as u64 ^ salt) >> 11) as f64 / (1u64 << 53) as f64
}

/// A phase that starts with fewer than this many infeasibilities of the kind
/// it removes (violated rows for the dual simplex, dual infeasible reduced
/// costs for the primal) can be near optimal, and is then not perturbed. The
/// exact value matters little: a start far from optimal has far more.
const NEAR_OPTIMAL_MAX_INFEASIBILITIES: usize = 1000;

/// A phase whose infeasibilities of the kind it removes are all below this,
/// in the engine's scaled units, can be near optimal (a near-optimal start's
/// are orders of magnitude smaller).
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

    /// Objective coefficients in scaled units: `s_j c_j` for a structural var,
    /// zero for a slack. The objective value is invariant under scaling
    /// (`s_j c_j · x_j / s_j = c_j x_j`), so `cur_obj_val` is in user units.
    orig_obj_coeffs: Vec<f64>,
    /// Bounds in scaled units: `lo_j / s_j` and `hi_j / s_j` for a structural
    /// var, the slack's own bounds for a slack. These are the bounds in force:
    /// the model's, but where the dual simplex widened one for the rest of the
    /// solve (`widened`).
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
    /// Per row, the factor between its slack's reduced cost in the engine's
    /// units and in the user's ([`slack_dual_unit`]).
    slack_dual_units: Vec<f64>,
    /// Positive power-of-two column factors `s_j` of the structural vars (see
    /// [`column_scales`]). The engine stores a structural var in the scaled
    /// units `x'_j = x_j / s_j`; every value or bound that crosses the
    /// `Solver` boundary is converted, so callers only see user units.
    col_scales: Vec<f64>,
    /// The threshold of the factorizations ([`LU_STABILITY_THRESHOLD`]),
    /// raised for the rest of the solver's life once the two computations of
    /// a pivot element disagree ([`Self::raise_lu_threshold`]).
    lu_threshold: f64,
    /// The user's absolute feasibility tolerance
    /// ([`crate::Tolerances::feasibility`]), in user units.
    feasibility: f64,
    /// The spread of the user's rows as written, which a pure-LP solve's loud
    /// errors and those of its edits name ([`CoefficientRange`]): set by the
    /// caller, which sees the model before presolve, and extended by each row
    /// an edit adds.
    pub(crate) range: CoefficientRange,
    /// Per total var, the absolute tolerances (scaled units) within which the
    /// var counts as within each of its bounds: [`structural_tol`] for a
    /// structural var, refreshed when its bounds change, and [`slack_tol`]
    /// for a slack, refreshed whenever the rows are checked since its
    /// round-off depends on the values. A side whose bound the dual simplex
    /// widened keeps none (a slack, its evaluation margin): its bound holds
    /// the rest ([`Self::widen`]).
    var_tols: Vec<BoundTols>,
    /// Per total var, the tolerance within which its reduced cost counts as
    /// zero ([`dual_tol`]); refreshed whenever the reduced costs are
    /// recomputed from the multipliers.
    dual_tols: Vec<f64>,
    /// The user's rows, in user units, that presolve dropped (implied by the
    /// bounds it emitted) or rewrote (valid for integer points only). No pivot
    /// sees them; the boundary checks the answer against them
    /// ([`Self::first_violated_row`]): an LP takes over a row its answer
    /// misses ([`Self::hold_validation_rows`]), and the MIP guard rejects such
    /// a candidate.
    validation_rows: Vec<(CsVec, ComparisonOp, f64)>,
    /// Per row of the user's model (the problem's rows in the order added,
    /// then those [`Self::add_constraint`] added), the engine row that holds
    /// it, or `None` for one the engine does not hold: a row every point
    /// satisfies ([`prepare_row`]), or one presolve dropped that the engine
    /// has not taken over. Read by [`Self::duals`] only.
    user_rows: Vec<Option<usize>>,
    /// Per validation row, its index among the user's rows (empty where the
    /// caller did not say, as in the MIP, whose duals are not read).
    validation_user_rows: Vec<usize>,

    enable_primal_steepest_edge: bool,
    enable_dual_steepest_edge: bool,

    /// The phase-1 artificial cost vector (per total var) while the initial
    /// dual phase runs on it, so that recomputed reduced costs price what the
    /// pivots priced. `None` once the real objective is in force.
    artificial_obj: Option<Vec<f64>>,

    /// Per total var, what a dual phase adds to the var's cost in force: the
    /// perturbation drawn when the phase starts and the shifts that corrected
    /// its reduced cost. Zero outside a dual phase. See [`costs`].
    cost_perturbation: Vec<f64>,
    /// Per total var, the shift a zero dual step gave the var's cost when it
    /// entered the basis, given back when it leaves. Zero outside a dual
    /// phase. See [`costs`].
    cost_shifts: Vec<f64>,
    /// Whether any entry of the two above is nonzero, i.e. the simplex works
    /// with costs other than the costs in force.
    costs_modified: bool,
    /// Per total var, its bounds as a primal phase has moved them: perturbed
    /// when the phase starts, shifted where its steps left a violation or a
    /// var left the basis past its bound. Taken from the true bounds at the
    /// first move of a phase, and read only while `bounds_modified`, through
    /// [`Self::working_bounds`]. See [`bounds`].
    working_mins: Vec<f64>,
    working_maxs: Vec<f64>,
    /// Whether a primal phase has moved any bound, i.e. the simplex works
    /// with bounds other than the true ones.
    bounds_modified: bool,
    /// The value of every var (scaled units, by var index) the running
    /// primal phase started from, where the verification accepted them: the
    /// point its rays start from ([`Self::unresolved_entries_leave_a_ray`]).
    /// `None` outside a primal phase, and for one that started on values the
    /// verification did not accept.
    primal_start: Option<Vec<f64>>,
    /// The cycling detection of both simplex phases (see [`cycling`]).
    cycle_guard: CycleGuard,

    is_primal_feasible: bool,

    /// Whether the basic values have changed (pivots, bound shifts, a
    /// recompute, which is not a verification) since [`Self::verify_primal`]
    /// last checked them against the rows; if not, a phase exit needs no
    /// residual check or refinement.
    values_dirty: bool,
    /// Whether the reduced costs have been updated incrementally since they
    /// were last recomputed exactly; if not, a phase exit needs no
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
    /// The bounds a var crossed by absorbing a move in this solve, as (var,
    /// upper): a var crosses a bound by absorbing once per solve (see
    /// `choose_entering_col_dual`).
    absorbed: Vec<(usize, bool)>,
    /// The bounds the phases went past on refined values in this attempt of
    /// [`Self::run_phases`], as (var, upper): refined values go on only where
    /// they miss a bound not held yet ([`Self::refined_first`]), so at most
    /// twice per var and attempt, each time at the cost of a phase round and
    /// of the cut of the primal ratio test's tolerance that comes with it
    /// ([`LOOP_HARRIS_REDUCTION`]). A held bound is passed again where refined
    /// values miss it beside a new one.
    held: Vec<(usize, bool)>,
    /// The refined values the last verification undid ([`Self::rows_hold`]),
    /// which [`Self::refined_first`] weighs once the dual check accepts the
    /// vertex; `None` where the verification undid none.
    undone: Option<UndoneRefinement>,
    /// The answer the first excursion on refined values in this attempt
    /// started from ([`Self::refined_first`]), which [`Self::run_phases`]
    /// returns to where the excursion ends worse or fails
    /// ([`Self::keep_the_better_answer`]).
    fallback: Option<Box<VerifiedAnswer>>,
    /// The row at which the last dual phase failed loudly, neither repaired by
    /// any var nor proving the model infeasible: [`Self::run_phases`] starts
    /// over once from the slack basis.
    unrepaired_row: Option<usize>,
    /// The bounds the certificate repair widened by their var's tolerance in
    /// this solve, with the model's values ([`Self::widen`]): `orig_var_mins`
    /// and `orig_var_maxs` hold the bounds in force. The boundary's checks
    /// and the edits read the model's ([`Self::model_bounds`]), and every
    /// solve starts by restoring them ([`Self::restore_model_bounds`]).
    widened: Vec<WidenedBound>,

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

/// Refined values the verification undid ([`Solver::rows_hold`]), one per
/// basis position, and the bounds they miss, as (var, upper).
#[derive(Clone, Debug)]
struct UndoneRefinement {
    values: Vec<f64>,
    misses: Vec<(usize, bool)>,
}

/// An answer the verification and the dual check accepted, to return to
/// ([`Solver::return_to`]): the basis, the values, the bounds in force and
/// what the checks derived from them. The factorization is left out: a return
/// builds it afresh.
#[derive(Clone, Debug)]
struct VerifiedAnswer {
    basic_vars: Vec<usize>,
    basic_var_vals: Vec<f64>,
    nb_vars: Vec<usize>,
    nb_var_vals: Vec<f64>,
    nb_var_states: Vec<NonBasicVarState>,
    nb_var_obj_coeffs: Vec<f64>,
    dual_tols: Vec<f64>,
    orig_var_mins: Vec<f64>,
    orig_var_maxs: Vec<f64>,
    var_tols: Vec<BoundTols>,
    widened: Vec<WidenedBound>,
    dual_edge_sq_norms: Vec<f64>,
    primal_edge_sq_norms: Vec<f64>,
    /// Its objective, and how far above it another answer still counts as no
    /// worse ([`Solver::refinement_gain`]).
    objective: f64,
    allowance: f64,
}

/// A bound of `var`, the upper one if `up`, that the certificate repair
/// widened by the var's tolerance ([`Solver::widen`]), and its value in the
/// model.
#[derive(Clone, Copy, Debug)]
struct WidenedBound {
    var: usize,
    up: bool,
    model: f64,
}

#[derive(Clone, Debug)]
enum VarState {
    Basic(usize),
    NonBasic(usize),
}

/// Where a non-basic var sits: at its lower bound, its upper bound, both (a
/// fixed var), or neither (a free var at zero, or one a bound change left
/// between its bounds).
#[derive(Clone, Debug)]
struct NonBasicVarState {
    at_min: bool,
    at_max: bool,
}

impl NonBasicVarState {
    /// The state of a non-basic var at `val` with bounds `min..max`: at a
    /// bound when its value is that bound, and at both when they are equal.
    /// A move puts a non-basic var exactly on a bound, but a var can sit past
    /// one, within its tolerance: one that left the basis where it sat
    /// (`hold_leaving_value`), which the primal cleanup leaves there. It is at
    /// that bound, and one strictly between its bounds (where a bound change
    /// left it) is at neither. No tolerance is involved, so a var whose range
    /// is narrower than its tolerance is at one bound and can move to the
    /// other.
    fn at(val: f64, min: f64, max: f64) -> Self {
        let fixed = min == max;
        Self {
            at_min: fixed || val <= min,
            at_max: fixed || val >= max,
        }
    }

    /// Whether a non-basic var in this state with reduced cost `d` offers no
    /// improving move beyond the tolerance `tol`: its cost pushes it against
    /// a bound it sits at, or moving it changes the objective by less than
    /// `tol` per unit. A var at neither bound (a free var, or one a bound
    /// change left between its bounds) can move both ways, so only the second
    /// applies to it.
    fn is_dual_feasible(&self, d: f64, tol: f64) -> bool {
        self.dual_infeasibility(d) < tol
    }
}

/// What [`Solver::verify_primal`] finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// The rows hold at the values: whether every basic var is within its
    /// bounds. `false` too when a fresh factorization repaired the basis,
    /// which leaves another vertex, unverified.
    Checked { feasible: bool },
    /// The values miss the rows through every factorization tried: the basis
    /// cannot represent its vertex.
    Unrepresentable,
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
            "num_vars: {}, num_constraints: {}, is_primal_feasible: {}",
            self.num_vars,
            self.num_constraints(),
            self.is_primal_feasible,
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
            // Numbers, each infinite only on its open side: the bounds the
            // model's validation takes (`Problem::solve_with`).
            debug_assert!(
                var_mins[v] < f64::INFINITY && var_maxs[v] > f64::NEG_INFINITY,
                "bounds the model's validation refuses reached the engine"
            );
            if var_mins[v] > var_maxs[v] {
                return Err(Error::Infeasible);
            }
        }

        // Everything below works in scaled units: x'_j = x_j / s_j. A var
        // fixed here is left out of both scalings.
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
        // Rows in scaled units.
        let is_fixed = |var: usize| var_mins[var] == var_maxs[var];
        let mut prepared_rows = Vec::with_capacity(constraints.len());
        let mut slack_dual_units = Vec::with_capacity(constraints.len());
        let mut user_rows = Vec::with_capacity(constraints.len());
        for (coeffs, cmp_op, rhs) in constraints {
            let scaled = scale_columns(coeffs.clone(), &col_scales);
            if let Some(row) = prepare_row(scaled, *cmp_op, *rhs, feasibility, is_fixed)? {
                user_rows.push(Some(prepared_rows.len()));
                slack_dual_units.push(slack_dual_unit(coeffs, row.row_scale, is_fixed));
                prepared_rows.push(row);
            } else {
                user_rows.push(None);
            }
        }
        let var_tols: Vec<BoundTols> = (0..num_vars)
            .map(|v| {
                structural_tol(
                    col_scales[v],
                    orig_var_mins[v],
                    orig_var_maxs[v],
                    feasibility,
                    matches!(var_domains[v], VarDomain::Integer | VarDomain::Boolean),
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

            // initially all user-created variables are non-basic
            var_states.push(VarState::NonBasic(nb_vars.len()));
            nb_vars.push(v);

            // Choose an initial value, preferring a bound that keeps this
            // variable's reduced cost dual-feasible.
            let (init_val, var_dual_feasible) = if min == max {
                // Fixed variable: the obj. coeff doesn't matter.
                (min, true)
            } else {
                // At the slack basis a reduced cost is the objective coefficient.
                let tol = dual_tol(orig_obj_coeffs[v].abs(), col_scales[v]);
                initial_nonbasic_value(orig_obj_coeffs[v], min, max, tol)
            };
            if !var_dual_feasible {
                is_dual_feasible = false;
            }

            nb_var_vals.push(init_val);
            obj_val += init_val * orig_obj_coeffs[v];

            nb_var_states.push(NonBasicVarState::at(init_val, min, max));
        }

        let mut constraint_coeffs = vec![];
        let mut orig_rhs = vec![];
        let mut row_scales = vec![];

        // Initially, all slack vars are basic, their values and tolerances
        // set from their rows once the solver holds them.
        let mut basic_vars = vec![];
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
            constraint_coeffs.push(coeffs);
            orig_rhs.push(rhs);
            row_scales.push(row_scale);

            orig_var_mins.push(slack_var_min);
            orig_var_maxs.push(slack_var_max);

            basic_var_mins.push(slack_var_min);
            basic_var_maxs.push(slack_var_max);

            let cur_slack_var = var_states.len();
            var_states.push(VarState::Basic(basic_vars.len()));
            basic_vars.push(cur_slack_var);
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

        let mut nb_var_obj_coeffs = vec![];
        let mut primal_edge_sq_norms = vec![];
        for &var in &nb_vars {
            //guaranteed to be a valid index
            let col = orig_constraints_csc.outer_view(var).unwrap();

            nb_var_obj_coeffs.push(orig_obj_coeffs[var]);

            if enable_primal_steepest_edge {
                primal_edge_sq_norms.push(col.squared_l2_norm() + 1.0);
            }
        }

        let mut scratch = ScratchSpace::with_capacity(num_constraints);
        let (lu_factors, replaced) = lu_factorize(
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
        );
        debug_assert!(replaced.is_empty(), "the slack basis is the identity");
        let lu_factors_transp = lu_factors.transpose();

        let mut res = Self {
            num_vars,
            orig_obj_coeffs,
            orig_var_mins,
            orig_var_maxs,
            orig_constraints,
            orig_constraints_csc,
            orig_rhs,
            row_scales,
            slack_dual_units,
            col_scales,
            lu_threshold: LU_STABILITY_THRESHOLD,
            feasibility,
            range: CoefficientRange::default(),
            var_tols,
            dual_tols: vec![0.0; num_total_vars],
            validation_rows: Vec::new(),
            user_rows,
            validation_user_rows: Vec::new(),
            deadline,
            operation_time_limit: None,
            lp_iterations: 0,
            elapsed: Duration::ZERO,
            orig_var_domains: var_domains.to_vec(),
            enable_primal_steepest_edge,
            enable_dual_steepest_edge,
            artificial_obj: None,
            cost_perturbation: vec![0.0; num_total_vars],
            cost_shifts: vec![0.0; num_total_vars],
            costs_modified: false,
            working_mins: Vec::new(),
            working_maxs: Vec::new(),
            bounds_modified: false,
            primal_start: None,
            cycle_guard: CycleGuard::default(),
            // Set below, with the slacks' values.
            is_primal_feasible: false,
            // The starting values are verified as they stand: each slack is
            // its row as `check_rows` evaluates it (`row_slack`, below), and
            // with every slack basic the refinement has no residual to remove.
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
            basic_var_vals: Vec::with_capacity(num_constraints),
            basic_var_mins,
            basic_var_maxs,
            dual_edge_sq_norms,
            nb_vars,
            nb_var_obj_coeffs,
            nb_var_vals,
            nb_var_states,
            primal_edge_sq_norms,
            fixed_var_bounds: vec![None; num_vars],
            absorbed: Vec::new(),
            held: Vec::new(),
            undone: None,
            fallback: None,
            unrepaired_row: None,
            widened: Vec::new(),
            cur_obj_val: obj_val,
            col_coeffs: SparseVec::new(),
            sq_norms_update_helper,
            inv_basis_row_coeffs: SparseVec::new(),
            row_coeffs: ScatteredVec::empty(num_total_vars - num_constraints),
            flip_rhs: ScatteredVec::empty(num_constraints),
        };
        for r in 0..num_constraints {
            let (value, tols) = res.row_slack(r);
            res.basic_var_vals.push(value);
            res.var_tols.push(tols);
        }
        res.is_primal_feasible = res
            .basic_var_vals
            .iter()
            .zip(&res.basic_var_mins)
            .zip(&res.basic_var_maxs)
            .all(|((&val, &min), &max)| val >= min && val <= max);
        if !res.is_primal_feasible && !is_dual_feasible {
            // Phase-1 artificial objective: push every non-basic var towards
            // the bound it starts at (cost +1 at a lower bound, -1 at an upper
            // bound, 0 if free or fixed), so the slack basis is dual feasible
            // for it.
            let mut costs = vec![0.0; num_total_vars];
            for (c, (&var, state)) in res.nb_vars.iter().zip(&res.nb_var_states).enumerate() {
                costs[var] = if state.at_min && !state.at_max {
                    1.0
                } else if state.at_max && !state.at_min {
                    -1.0
                } else {
                    0.0
                };
                res.nb_var_obj_coeffs[c] = costs[var];
            }
            res.artificial_obj = Some(costs);
            res.cur_obj_val = 0.0;
        }
        // The slack basis has zero multipliers: a reduced cost is the
        // objective coefficient.
        res.refresh_dual_tols(&vec![0.0; num_constraints]);

        debug!(
            "initialized solver: vars: {}, constraints: {}, primal feasible: {}, dual feasible: {}, nnz: {}",
            res.num_vars,
            res.orig_constraints.rows(),
            res.is_primal_feasible,
            is_dual_feasible,
            res.orig_constraints.nnz(),
        );

        Ok(res)
    }

    /// A var's current value in user units: the value that is reported, as
    /// the engine computed it. A fixed var's value is within its bound
    /// tolerance of its value (a fixed integer var's on it, see
    /// [`structural_tol`]), an integer var's within the search's integrality
    /// tolerance of an integer; nothing is snapped, since a snapped value
    /// would no longer be the point the rows were verified at.
    pub(crate) fn get_value(&self, var: usize) -> f64 {
        self.internal_value(var) * self.col_scale(var)
    }

    /// The structural values to report, in user units: the point the last
    /// verification checked, since [`Self::verify_primal`] evaluates the rows
    /// at exactly these values.
    pub(crate) fn reported_values(&self) -> Vec<f64> {
        (0..self.num_vars).map(|v| self.get_value(v)).collect()
    }

    /// The duals of a finished solve in user units, internal (minimize)
    /// space: per row of the user's model ([`Self::user_rows`]) the rate of
    /// change of the objective with its right-hand side, and per structural
    /// var its reduced cost, `c_j - Σ_i a_ij y_i`, zero for a basic var.
    ///
    /// They come from the multipliers `B^T y = c_B` of the final basis,
    /// refined once ([`Self::refine_multipliers`]): those the optimality
    /// check accepted. The engine's row `r` is the user's times the factor
    /// `ρ_r`, and its column `j` the user's times `s_j`, so the user's dual
    /// is `ρ_r y_r` and the user's reduced cost the engine's over `s_j`. A
    /// row the engine does not hold has dual zero: every point satisfies it,
    /// or presolve dropped it as implied by the bounds, which then price it.
    pub(crate) fn duals(&mut self) -> (Vec<f64>, Vec<f64>) {
        debug_assert!(
            self.artificial_obj.is_none() && !self.costs_modified,
            "duals read off a solve that is not finished"
        );
        let mut multipliers = self.multipliers();
        self.refine_multipliers(&mut multipliers);
        let row_duals = self
            .user_rows
            .iter()
            .map(|row| row.map_or(0.0, |r| multipliers[r] * self.row_scales[r]))
            .collect();
        let reduced_costs = (0..self.num_vars)
            .map(|var| match self.var_states[var] {
                VarState::Basic(_) => 0.0,
                VarState::NonBasic(_) => {
                    let mut sum = CDouble::from(self.orig_obj_coeffs[var]);
                    //guaranteed to be a valid index
                    for (r, &coeff) in self.orig_constraints_csc.outer_view(var).unwrap().iter() {
                        sum.add_product(-coeff, multipliers[r]);
                    }
                    sum.value() / self.col_scales[var]
                }
            })
            .collect();
        (row_duals, reduced_costs)
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

    /// The factor between `var`'s reduced cost in the engine's units and in
    /// the user's (see [`dual_tol`]): `s_j` for a structural var, and for a
    /// slack its row's [`slack_dual_unit`].
    #[inline]
    fn dual_unit(&self, var: usize) -> f64 {
        if var < self.num_vars {
            self.col_scales[var]
        } else {
            self.slack_dual_units[var - self.num_vars]
        }
    }

    /// Whether `var` is an integer structural var (slacks are continuous).
    #[inline]
    fn is_integer(&self, var: usize) -> bool {
        var < self.num_vars
            && matches!(
                self.orig_var_domains[var],
                VarDomain::Integer | VarDomain::Boolean
            )
    }

    /// Whether the engine keeps `var` on its bounds while it is non-basic: an
    /// integer var, and a structural var its bounds fix.
    #[inline]
    fn holds_on_bounds(&self, var: usize) -> bool {
        self.is_integer(var)
            || (var < self.num_vars && self.orig_var_mins[var] == self.orig_var_maxs[var])
    }

    /// Whether moving the non-basic var at `col` by `step` takes it past its
    /// bound in that direction.
    fn crosses_bound(&self, col: usize, step: f64) -> bool {
        let (var, val) = (self.nb_vars[col], self.nb_var_vals[col]);
        if step > 0.0 {
            val + step > self.orig_var_maxs[var]
        } else {
            val + step < self.orig_var_mins[var]
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

    /// The activity `Σ a_rj x_j` of row `r` over the structural vars at the
    /// scaled values `x`, and the magnitude `|b_r| + Σ|a_rj x_j|` it was
    /// computed from, summed in plain `f64`: the engine's evaluation of a row
    /// where its round-off fits the margin the row's tolerance leaves out for
    /// it ([`Self::row_slack`], [`slack_tol`]).
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

    /// The slack `b_r - Σ a_rj x_j` row `r` implies at the current values,
    /// and its tolerances there ([`Self::slack_tols`]): the plain sum of
    /// [`Self::row_activity`] where its round-off fits the row's evaluation
    /// margin ([`plain_sum_resolves`]), else compensated ([`implied_slack`]).
    /// A plain sum that puts the slack beyond its edge by at most twice that
    /// margin is summed again, compensated: there the exact slack can lie
    /// within the room, where the infeasibility proof, summed exactly, cannot
    /// prove the row violated ([`Self::leaving_slack_cut`]). A plain
    /// acceptance stands, the margin covering it. The one evaluation every
    /// slack set from its row takes (the row check, the starting vertex, an
    /// added row).
    fn row_slack(&self, r: usize) -> (f64, BoundTols) {
        let (lhs, magnitude) = self.row_activity(r, |v| self.internal_value(v));
        let (tol, terms) = (self.feasibility * self.row_scales[r], self.row_terms(r));
        let tols = self.slack_tols(r, magnitude);
        let compensated = || {
            //guaranteed to be a valid index
            let row = self.orig_constraints.outer_view(r).unwrap();
            let terms = row
                .iter()
                .filter(|&(v, _)| v < self.num_vars)
                .map(|(v, &coeff)| (coeff, self.internal_value(v)));
            implied_slack(self.orig_rhs[r], terms).0
        };
        if !plain_sum_resolves(tol, magnitude, terms) {
            return (compensated(), tols);
        }
        let plain = self.orig_rhs[r] - lhs;
        let beyond = self.slack_excess(r, plain, tols);
        if beyond > 0.0 && beyond <= 2.0 * slack_margin(tol, magnitude, terms) {
            return (compensated(), tols);
        }
        (plain, tols)
    }

    /// How far the slack of row `r`, at `value`, lies beyond the edge the
    /// engine holds it to with the tolerances `tols` (not positive within it):
    /// a basic slack's bounds in the basis, a non-basic slack's residual
    /// `value` less its own value, which [`Self::check_rows`] holds within
    /// `tols` of zero.
    fn slack_excess(&self, r: usize, value: f64, tols: BoundTols) -> f64 {
        match self.var_states[self.num_vars + r] {
            VarState::Basic(pos) => (self.basic_var_mins[pos] - tols.lo - value)
                .max(value - (self.basic_var_maxs[pos] + tols.hi)),
            VarState::NonBasic(col) => {
                let residual = value - self.nb_var_vals[col];
                (-tols.lo - residual).max(residual - tols.hi)
            }
        }
    }

    /// The number of structural nonzeros of row `r`, which the round-off of
    /// evaluating it grows with (see [`slack_tol`]).
    fn row_terms(&self, r: usize) -> usize {
        //guaranteed to be a valid index; the row holds its slack's entry too
        self.orig_constraints.outer_view(r).unwrap().nnz() - 1
    }

    /// The tolerances of the slack of row `r` at the current values
    /// ([`Self::slack_tols`]).
    fn current_slack_tols(&self, r: usize) -> BoundTols {
        let (_, magnitude) = self.row_activity(r, |v| self.internal_value(v));
        self.slack_tols(r, magnitude)
    }

    /// The tolerances of the slack of row `r` at the row's `magnitude`:
    /// [`slack_tol`], or only the evaluation margin ([`slack_margin`]) on a
    /// side the certificate repair widened, whose bound holds the rest
    /// ([`Self::widen`]).
    fn slack_tols(&self, r: usize, magnitude: f64) -> BoundTols {
        let (tol, terms) = (self.feasibility * self.row_scales[r], self.row_terms(r));
        let mut tols = BoundTols::uniform(slack_tol(tol, magnitude, terms));
        for w in self.widened.iter().filter(|w| w.var == self.num_vars + r) {
            let margin = slack_margin(tol, magnitude, terms);
            if w.up {
                tols.hi = margin;
            } else {
                tols.lo = margin;
            }
        }
        tols
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
            self.dual_tols[var] = dual_tol(magnitude, self.dual_unit(var));
        }
    }

    /// The first row that `values` (one entry per structural var, user units)
    /// violates beyond the contract, with the violation in user units, in the
    /// engine's numbering: its own rows, then the validation rows. A row's
    /// sense is encoded by its slack's bounds (`lhs + s = rhs` with `s` in
    /// `[smin, smax]`), which branching never touches.
    ///
    /// The slack the row implies, summed in compensated arithmetic
    /// ([`implied_slack`]), is held by [`outside`] to the slack's bounds within
    /// [`row_tolerance`] of the user's absolute tolerance in the row's scaled
    /// units (exactly the check on the user's row, the row factor being a
    /// power of two). The engine holds the row to that contract less the
    /// round-off of its own evaluation ([`slack_tol`]), so a point it
    /// verified passes, whatever the order of either sum. The tolerance is not
    /// relative to the coefficients, so that a violation of `5.0` on a
    /// `1e9`-scale big-M row counts; only its round-off floor is relative, to
    /// the row's activity. The rows presolve dropped or rewrote
    /// ([`Self::validation_rows`]) are then checked as the user wrote them.
    pub(crate) fn first_violated_row(&self, values: &[f64]) -> Option<(usize, f64)> {
        for r in 0..self.num_constraints() {
            //guaranteed to be a valid index
            let row = self.orig_constraints.outer_view(r).unwrap();
            // `values` are in user units; dividing by the power-of-two `s_v`
            // is exact, so each term is the user's term times the row factor.
            let terms = row
                .iter()
                .filter(|&(v, _)| v < self.num_vars)
                .map(|(v, &coeff)| (coeff, values[v] / self.col_scales[v]));
            let (implied, magnitude) = implied_slack(self.orig_rhs[r], terms);
            let (smin, smax) = self.model_bounds(self.num_vars + r);
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

    /// Number the engine's rows by the user's rows once presolve has reduced
    /// them: the engine was built on presolve's rows, the input rows
    /// `kept_rows` (in that order) of `num_user_rows`, and the validation
    /// rows are the input rows `validation_origs`.
    pub(crate) fn map_presolved_rows(
        &mut self,
        kept_rows: &[usize],
        validation_origs: Vec<usize>,
        num_user_rows: usize,
    ) {
        debug_assert_eq!(kept_rows.len(), self.user_rows.len());
        debug_assert_eq!(validation_origs.len(), self.validation_rows.len());
        let mut user_rows = vec![None; num_user_rows];
        for (&orig, &row) in kept_rows.iter().zip(&self.user_rows) {
            user_rows[orig] = row;
        }
        self.user_rows = user_rows;
        self.validation_user_rows = validation_origs;
    }

    /// Hold a finished LP's answer to the rows presolve dropped or rewrote
    /// ([`Self::validation_rows`]): a row the reported point misses beyond the
    /// contract left presolve's reduction short of the model as written, so
    /// the engine takes the row over ([`Self::add_constraint`]) and the solve
    /// resumes from the current basis. Each row is taken over once, so this
    /// ends, and an `Infeasible` re-solve holds for the model as written,
    /// which has the row too.
    pub(crate) fn hold_validation_rows(&mut self) -> Result<StopReason, Error> {
        loop {
            let values = self.reported_values();
            let missed = self.validation_rows.iter().position(|(coeffs, op, rhs)| {
                row_violation(coeffs, *op, *rhs, |v| values[v], self.feasibility).is_some()
            });
            let Some(k) = missed else {
                return Ok(StopReason::Finished);
            };
            debug!("the answer misses validation row {k}; the engine takes it over");
            let (coeffs, op, rhs) = self.validation_rows.remove(k);
            let user_row =
                (k < self.validation_user_rows.len()).then(|| self.validation_user_rows.remove(k));
            if self.add_row(coeffs, op, rhs, user_row)? == StopReason::Limit {
                return Ok(StopReason::Limit);
            }
        }
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

    /// How far a point of the bounds in force that meets the rows exactly can
    /// lower the objective below its value at the current vertex: each
    /// non-basic var's reduced cost on its infeasible side, which the dual
    /// check accepts within its tolerance, times the var's range, infinite
    /// where the range is, and not a number where a reduced cost is not. The
    /// round-off of the reduced costs themselves is not counted.
    pub(crate) fn dual_slack(&self) -> f64 {
        let mut slack = 0.0;
        for (c, &var) in self.nb_vars.iter().enumerate() {
            let wrong = self.nb_var_states[c].dual_infeasibility(self.nb_var_obj_coeffs[c]);
            if wrong > 0.0 || wrong.is_nan() {
                slack += wrong * (self.orig_var_maxs[var] - self.orig_var_mins[var]);
            }
        }
        slack
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

    /// A var's bounds in the model, in user units.
    pub(crate) fn get_var_bounds(&self, var: usize) -> (f64, f64) {
        let s = self.col_scale(var);
        let (min, max) = self.model_bounds(var);
        (min * s, max * s)
    }

    /// A var's bounds in the model, in scaled units: the bounds in force but
    /// where the certificate repair widened one ([`Self::widen`]).
    fn model_bounds(&self, var: usize) -> (f64, f64) {
        let (mut min, mut max) = (self.orig_var_mins[var], self.orig_var_maxs[var]);
        for w in self.widened.iter().filter(|w| w.var == var) {
            if w.up {
                max = w.model;
            } else {
                min = w.model;
            }
        }
        (min, max)
    }

    /// Change a variable's bounds (user units) in place. Records the new bounds
    /// and repairs the invariants that depend on them; does NOT run simplex —
    /// call [`Self::reoptimize`] afterwards. Returns `Err(Infeasible)` with
    /// state untouched if `min > max`, and `Err(InternalError)` with state
    /// untouched if a bound in the engine's units is NaN, a lower bound of
    /// `+∞` or an upper bound of `-∞`: a NaN, or a value the column factor
    /// takes beyond `f64`.
    pub(crate) fn set_var_bounds(&mut self, var: usize, min: f64, max: f64) -> Result<(), Error> {
        if min > max {
            return Err(Error::Infeasible);
        }
        debug_assert!(
            !self.bounds_modified,
            "a bound changed during a primal phase"
        );
        // Dividing by a power of two is exact, so the user's bound maps to
        // exactly one internal bound, unless it leaves the range of `f64`.
        let s = self.col_scale(var);
        let (scaled_min, scaled_max) = (min / s, max / s);
        if scaled_min.is_nan()
            || scaled_max.is_nan()
            || scaled_min == f64::INFINITY
            || scaled_max == f64::NEG_INFINITY
        {
            return Err(Error::InternalError(format!(
                "variable {var} cannot take the bounds [{min:e}, {max:e}] in the engine's \
                 scaled units"
            )));
        }
        let (min, max) = (scaled_min, scaled_max);
        // The new bounds are the model's, in force from here on.
        self.widened.retain(|w| w.var != var);
        self.orig_var_mins[var] = min;
        self.orig_var_maxs[var] = max;
        if var < self.num_vars {
            self.var_tols[var] =
                structural_tol(s, min, max, self.feasibility, self.is_integer(var));
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
                // The var stays at the bound its state names, now the new
                // one; one that was fixed goes to the bound at which its
                // reduced cost is dual feasible, and one between its bounds
                // stays where it is if they still hold it.
                let cur = self.nb_var_vals[col];
                let state = &self.nb_var_states[col];
                let lower = [min, max].into_iter().find(|b| b.is_finite());
                let upper = [max, min].into_iter().find(|b| b.is_finite());
                let new_val = match (state.at_min, state.at_max) {
                    (true, true) if self.nb_var_obj_coeffs[col] < 0.0 => upper,
                    (true, _) => lower,
                    (false, true) => upper,
                    (false, false) => Some(cur.clamp(min, max)),
                }
                .unwrap_or(0.0);
                if new_val != cur {
                    // Shift the non-basic var to the nearest bound and propagate the
                    // delta into basic values through its tableau column.
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
                self.nb_var_states[col] = NonBasicVarState::at(new_val, min, max);
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
    /// costs recomputed from the multipliers are dual feasible. A phase's own
    /// exit test runs on incrementally updated numbers and can end on drifted
    /// values, so only a verified state is reported as `Finished`, and a
    /// primal phase starts only on verified values: its `Unbounded` then rests
    /// on a point the model admits. The solve starts from the model's bounds,
    /// restoring the ones a previous solve widened
    /// ([`Self::restore_model_bounds`]).
    ///
    /// A phase ends by restoring the costs in force ([`costs`]) or the true
    /// bounds ([`bounds`]), and the infeasibility that uncovers is left to the
    /// next phase of the other kind. Only the first phase of a re-solve may
    /// perturb; every later one is a cleanup: unperturbed, and a dual one with
    /// phase 2 forced, so that its dual infeasibilities are shifted rather
    /// than flipped. Phase 1, a dual phase on the artificial objective, is
    /// not done until its values pass the verification, so it runs again
    /// where they fail, and the primal phase after it is the real
    /// optimization; neither is a cleanup. The loop fails
    /// loudly after [`MAX_PHASE_ROUNDS`] rounds; each round after the first
    /// cuts the primal ratio test's tolerance a hundredfold
    /// ([`LOOP_HARRIS_REDUCTION`]), so that a step no longer ends at a
    /// tolerance the verification's recomputation finds exceeded.
    ///
    /// When no factorization represents the vertex
    /// ([`Verdict::Unrepresentable`]), or the dual phase meets a violated row
    /// that no var can repair and that does not prove the model infeasible,
    /// the solve starts over once from the slack basis, the last rung of the
    /// ladder; a second time is an `InternalError`. An attempt that went on
    /// from refined values ends no worse than the answer its first excursion
    /// started from ([`Self::keep_the_better_answer`]).
    fn run_phases(&mut self) -> Result<StopReason, Error> {
        self.start_attempt();
        self.unrepaired_row = None;
        let first = self.phase_rounds();
        let first = self.keep_the_better_answer(first);
        match (first, self.unrepaired_row.take()) {
            (Ok(Some(stop)), _) => return Ok(stop),
            (Ok(None), _) => debug!(
                "verify: no factorization represents the vertex; solving again from the slack basis"
            ),
            (Err(_), Some(row)) => debug!(
                "dual simplex: no var can repair row {row}, which proves nothing; solving again \
                 from the slack basis"
            ),
            (Err(error), None) => return Err(error),
        }
        let second = self.phase_rounds_from_slack_basis();
        match self.keep_the_better_answer(second) {
            Ok(Some(stop)) => Ok(stop),
            Ok(None) => Err(Error::InternalError(
                "basis too ill-conditioned: recomputed values do not satisfy the rows".to_string(),
            )),
            Err(Error::InternalError(message)) if self.unrepaired_row.is_some() => Err(
                Error::InternalError(format!("{message} (from the slack basis too)")),
            ),
            Err(error) => Err(error),
        }
    }

    /// What each attempt of [`Self::run_phases`] starts from: the model's
    /// bounds, no absorption spent ([`Self::absorbed`]), no bound held
    /// ([`Self::held`]) and no answer to return to ([`Self::fallback`]).
    fn start_attempt(&mut self) {
        self.restore_model_bounds();
        self.absorbed.clear();
        self.held.clear();
        self.undone = None;
        self.fallback = None;
    }

    /// The outcome of an attempt of [`Self::run_phases`] that went on from
    /// refined values ([`Self::fallback`]): its own where it ends with an
    /// answer worse by at most the allowance than the one its first excursion
    /// started from; else, where it ends worse, without an answer or in an
    /// error, that answer, which the verification and the dual check accepted
    /// ([`Self::return_to`]). At the deadline the solve stops there, still a
    /// `Limit`, and a later call ends at once on that answer: its values are
    /// verified and no refined values wait to be weighed.
    fn keep_the_better_answer(
        &mut self,
        outcome: Result<Option<StopReason>, Error>,
    ) -> Result<Option<StopReason>, Error> {
        let Some(answer) = self.fallback.take() else {
            return outcome;
        };
        if matches!(outcome, Ok(Some(StopReason::Finished)))
            && self.cur_obj_val <= answer.objective + answer.allowance
        {
            return outcome;
        }
        let (ended, objective) = (self.cur_obj_val, answer.objective);
        if !self.return_to(*answer) {
            return outcome;
        }
        debug!(
            "verify: the excursion on refined values ended at {outcome:?}, objective {ended}; back \
             to the answer it left, objective {objective}"
        );
        match outcome {
            Ok(Some(StopReason::Limit)) => Ok(Some(StopReason::Limit)),
            _ => Ok(Some(StopReason::Finished)),
        }
    }

    /// The current state as an answer to return to ([`VerifiedAnswer`]),
    /// another counting as no worse within `allowance` of its objective.
    fn verified_answer(&self, allowance: f64) -> VerifiedAnswer {
        VerifiedAnswer {
            basic_vars: self.basic_vars.clone(),
            basic_var_vals: self.basic_var_vals.clone(),
            nb_vars: self.nb_vars.clone(),
            nb_var_vals: self.nb_var_vals.clone(),
            nb_var_states: self.nb_var_states.clone(),
            nb_var_obj_coeffs: self.nb_var_obj_coeffs.clone(),
            dual_tols: self.dual_tols.clone(),
            orig_var_mins: self.orig_var_mins.clone(),
            orig_var_maxs: self.orig_var_maxs.clone(),
            var_tols: self.var_tols.clone(),
            widened: self.widened.clone(),
            dual_edge_sq_norms: self.dual_edge_sq_norms.clone(),
            primal_edge_sq_norms: self.primal_edge_sq_norms.clone(),
            objective: self.cur_obj_val,
            allowance,
        }
    }

    /// Return to `answer`: its basis, factorized afresh, and its values,
    /// bounds and reduced costs, as the checks accepted them. Where the
    /// factorization finds the basis singular, nothing changes and the return
    /// fails.
    fn return_to(&mut self, answer: VerifiedAnswer) -> bool {
        let m = answer.basic_vars.len();
        let mut scratch = ScratchSpace::with_capacity(m);
        let (lu_factors, replaced) = lu_factorize(
            m,
            |c| {
                self.orig_constraints_csc
                    .outer_view(answer.basic_vars[c])
                    //guaranteed to be a valid index
                    .unwrap()
                    .into_raw_storage()
            },
            self.lu_threshold,
            &mut scratch,
        );
        if !replaced.is_empty() {
            return false;
        }
        self.basis_solver = BasisSolver {
            lu_factors_transp: lu_factors.transpose(),
            lu_factors,
            scratch,
            eta_matrices: EtaMatrices::new(m),
            rhs: ScatteredVec::empty(m),
        };
        for (pos, &var) in answer.basic_vars.iter().enumerate() {
            self.var_states[var] = VarState::Basic(pos);
        }
        for (col, &var) in answer.nb_vars.iter().enumerate() {
            self.var_states[var] = VarState::NonBasic(col);
        }
        self.basic_var_mins = answer
            .basic_vars
            .iter()
            .map(|&var| answer.orig_var_mins[var])
            .collect();
        self.basic_var_maxs = answer
            .basic_vars
            .iter()
            .map(|&var| answer.orig_var_maxs[var])
            .collect();
        self.basic_vars = answer.basic_vars;
        self.basic_var_vals = answer.basic_var_vals;
        self.nb_vars = answer.nb_vars;
        self.nb_var_vals = answer.nb_var_vals;
        self.nb_var_states = answer.nb_var_states;
        self.nb_var_obj_coeffs = answer.nb_var_obj_coeffs;
        self.dual_tols = answer.dual_tols;
        self.orig_var_mins = answer.orig_var_mins;
        self.orig_var_maxs = answer.orig_var_maxs;
        self.var_tols = answer.var_tols;
        self.widened = answer.widened;
        self.dual_edge_sq_norms = answer.dual_edge_sq_norms;
        self.primal_edge_sq_norms = answer.primal_edge_sq_norms;
        self.cur_obj_val = answer.objective;
        self.is_primal_feasible = true;
        self.values_dirty = false;
        self.reduced_costs_dirty = false;
        self.pivots_since_drift_check = 0;
        self.unrepaired_row = None;
        self.undone = None;
        true
    }

    /// The second attempt of [`Self::run_phases`]: its rounds from the slack
    /// basis, on the real objective, started as the first.
    fn phase_rounds_from_slack_basis(&mut self) -> Result<Option<StopReason>, Error> {
        self.start_attempt();
        self.artificial_obj = None;
        let basis = self.slack_basis();
        self.load_basis(&basis)?;
        self.phase_rounds()
    }

    /// The rounds of [`Self::run_phases`], or `None` when the verification
    /// finds that no factorization represents the vertex.
    fn phase_rounds(&mut self) -> Result<Option<StopReason>, Error> {
        let mut cleanup = false;
        for round in 0..MAX_PHASE_ROUNDS {
            let harris_scale = LOOP_HARRIS_REDUCTION.powi(round as i32);
            if !self.is_primal_feasible {
                if self.restore_feasibility(cleanup)? == StopReason::Limit {
                    return Ok(Some(StopReason::Limit));
                }
                cleanup = cleanup || self.artificial_obj.is_none();
            }
            match self.verify_primal()? {
                Verdict::Checked { feasible: true } => {}
                Verdict::Checked { feasible: false } => {
                    debug!("phase round {round}: recomputed values violate a bound; continuing");
                    continue;
                }
                Verdict::Unrepresentable => return Ok(None),
            }
            // Dual verification: the reduced costs recomputed on multipliers
            // refined once. If they still show an infeasibility, let the
            // primal simplex act on them; no pivot at all means the vertex is
            // optimal within tolerance.
            self.artificial_obj = None;
            self.recalc_obj_coeffs_refined();
            if self.calc_dual_infeasibility().0 == 0 {
                if self.refined_first() {
                    continue;
                }
                return Ok(Some(StopReason::Finished));
            }
            debug!("phase round {round}: recomputed reduced costs are dual infeasible; continuing");
            let (stop, pivoted) = self.optimize(cleanup, harris_scale)?;
            cleanup = true;
            if stop == StopReason::Limit {
                return Ok(Some(StopReason::Limit));
            }
            if !pivoted {
                match self.verify_primal()? {
                    Verdict::Checked { feasible: true } => {
                        if self.refined_first() {
                            continue;
                        }
                        return Ok(Some(StopReason::Finished));
                    }
                    Verdict::Checked { feasible: false } => {}
                    Verdict::Unrepresentable => return Ok(None),
                }
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
        self.factorize();
        self.recalc_basic_var_vals()?;
        self.recalc_obj_coeffs()
    }

    /// Factorize the basis afresh, and repair it if the factorization finds
    /// it singular: each dependent var leaves the basis for the slack of a row
    /// that no pivot took ([`Self::replace_with_slack`]). Returns whether it
    /// repaired the basis, which moves the vertex: the caller then recomputes
    /// the values and the reduced costs.
    ///
    /// # Panics
    ///
    /// If a singular basis has inconsistent bookkeeping (a var basic in two
    /// positions makes the basis singular too): a repair would hide the bug
    /// and leave the var both basic and non-basic.
    fn factorize(&mut self) -> bool {
        self.factorize_at(self.lu_threshold)
    }

    /// [`Self::factorize`] at the pivot `threshold` of the factorization.
    ///
    /// A dependency loses the column the factorization takes last, which the
    /// sparsity order decides. Where that is a var with an infinite bound,
    /// the dependency leaves its coefficient in every later certificate,
    /// genuine but below what the arithmetic resolves, on that bound unless
    /// its sign points the other way, where it voids the proof
    /// ([`Self::proof_of`]). So the basis is factorized again with the
    /// columns ordered by their vars' bounds ([`bound_rank`]), and that repair
    /// is taken where it replaces as many columns, each a var with both
    /// bounds finite, whose coefficient counts at a bound whatever its sign.
    /// Otherwise the first repair stands: one that drops a var with an
    /// infinite bound instead guarantees the proof nothing for the change of
    /// path, and a second factorization that finds another number of
    /// dependencies would keep a basis the first found singular, or drop
    /// more.
    fn factorize_at(&mut self, threshold: f64) -> bool {
        let mut replaced =
            self.basis_solver
                .reset(&self.orig_constraints_csc, &self.basic_vars, threshold);
        if replaced.is_empty() {
            return false;
        }
        let (mins, maxs) = (&self.orig_var_mins, &self.orig_var_maxs);
        let boxed = |var: usize| mins[var].is_finite() && maxs[var].is_finite();
        if !replaced.iter().all(|&(pos, _)| boxed(self.basic_vars[pos])) {
            let (lu_factors, ranked) = self.basis_solver.factorize_ranked(
                &self.orig_constraints_csc,
                &self.basic_vars,
                threshold,
                |var| bound_rank(mins[var], maxs[var]),
            );
            if ranked.len() == replaced.len()
                && ranked.iter().all(|&(pos, _)| boxed(self.basic_vars[pos]))
            {
                self.basis_solver.install(lu_factors);
                replaced = ranked;
            }
        }
        for (pos, &var) in self.basic_vars.iter().enumerate() {
            assert!(
                matches!(self.var_states[var], VarState::Basic(p) if p == pos),
                "basis bookkeeping is corrupt: var {var} at basis position {pos} has state {:?}",
                self.var_states[var]
            );
        }
        self.cycle_guard.note_repair();
        for &(pos, row) in &replaced {
            self.replace_with_slack(pos, row);
        }
        true
    }

    /// Make the slack of `row` basic at position `pos` in place of the var
    /// there, as the factors already have it. The var leaves at the working
    /// bound nearest its value (0 if free), with its cost shift given back and
    /// its edge weights reset, and within a phase the change back is taboo
    /// ([`CycleGuard::add_singular_change`]). The values and reduced costs
    /// are left for the caller to recompute.
    fn replace_with_slack(&mut self, pos: usize, row: usize) {
        let var = self.basic_vars[pos];
        let slack = self.num_vars + row;
        // A basic slack's unit column takes its own row as the pivot unless an
        // earlier column did, so a row left without a pivot has its slack
        // non-basic.
        let VarState::NonBasic(col) = self.var_states[slack] else {
            unreachable!("the factorization left the row of the basic slack {slack} unpivoted");
        };
        assert_eq!(
            self.nb_vars[col],
            slack,
            "basis bookkeeping is corrupt: slack {slack}'s non-basic position {col} holds another var"
        );
        let val = self.basic_var_vals[pos];
        let (min, max) = self.working_bounds(var);
        let new_val = match (min.is_finite(), max.is_finite()) {
            (true, true) if val - min <= max - val => min,
            (true, true) => max,
            (true, false) => min,
            (false, true) => max,
            (false, false) => 0.0,
        };
        debug!(
            "basis repair: var {var} at {val:e} in row {pos} depends on the others; the slack of \
             row {row} replaces it, and it leaves at {new_val:e}"
        );

        if self.nb_var_states[col].at_upper() {
            self.cycle_guard.apply(VertexChange::Upper(slack));
        }
        let (slack_min, slack_max) = self.working_bounds(slack);
        self.basic_vars[pos] = slack;
        self.var_states[slack] = VarState::Basic(pos);
        self.basic_var_mins[pos] = slack_min;
        self.basic_var_maxs[pos] = slack_max;

        let state = self.nonbasic_state(var, new_val);
        if state.at_upper() {
            self.cycle_guard.apply(VertexChange::Upper(var));
        }
        self.nb_vars[col] = var;
        self.var_states[var] = VarState::NonBasic(col);
        self.nb_var_vals[col] = new_val;
        self.nb_var_states[col] = state;
        self.cycle_guard.apply(VertexChange::Basic(var));
        self.cycle_guard.apply(VertexChange::Basic(slack));
        self.cycle_guard.add_singular_change(pos, slack, var);

        // A zero dual step's shift lives only while its var is basic.
        self.cost_shifts[var] = 0.0;
        if self.enable_dual_steepest_edge {
            self.dual_edge_sq_norms[pos] = 1.0;
        }
        if self.enable_primal_steepest_edge {
            self.primal_edge_sq_norms[col] = 1.0;
        }
        self.values_dirty = true;
        self.reduced_costs_dirty = true;
    }

    /// The periodic refactorization: a fresh factorization, and the basic
    /// values recomputed from it only when their residuals show they have
    /// drifted out of the rows' tolerances. Recomputing values that satisfy
    /// every row would not improve them and, next to a bound, can move a var
    /// by a round-off amount the rows allow but its own tolerance does not.
    /// Values found drifted also restart the cycling detection, as a
    /// widening does, forgetting every vertex and bad change of the phase, a
    /// basis repair's too: the phase reached its vertex on those values, so a
    /// return to one it visited need not be a cycle. The reduced costs are
    /// left to their incremental updates; the phase exit recomputes them
    /// exactly. A repaired basis is a new vertex, whose values and reduced
    /// costs are recomputed, as in a rebuild.
    fn refactorize(&mut self) -> Result<(), Error> {
        if self.factorize() {
            self.recalc_basic_var_vals()?;
            return self.recalc_obj_coeffs();
        }
        // The residual check costs a pass over the matrix and a small basis is
        // refactorized almost every pivot, so check once per turnover of the
        // basis (one pivot per row). The phase exit always checks.
        if self.pivots_since_drift_check >= self.num_constraints().max(1) {
            self.pivots_since_drift_check = 0;
            if !self.check_rows() {
                self.recalc_basic_var_vals()?;
                self.start_cycle_detection();
            }
        }
        Ok(())
    }

    /// Bring the rows and the values into agreement and measure what is left.
    /// Every basic slack is re-derived from its row, `s = b - Σ a_j x_j`
    /// ([`Self::row_slack`]), so a row's violation lives in its slack, where
    /// the dual simplex sees it; the slack tolerances are refreshed at these
    /// values. Returns whether the residual `b - Σ a_j x_j - s` of every row
    /// whose slack is non-basic (how far the basic structural values are from
    /// solving it) lies within the slack's tolerance on the side it moves the
    /// slack (on a widened side, the evaluation margin alone).
    fn check_rows(&mut self) -> bool {
        let mut hold = true;
        for r in 0..self.num_constraints() {
            let (implied, tols) = self.row_slack(r);
            let slack = self.num_vars + r;
            let residual = match self.var_states[slack] {
                VarState::Basic(row) => {
                    self.basic_var_vals[row] = implied;
                    0.0
                }
                VarState::NonBasic(col) => implied - self.nb_var_vals[col],
            };
            self.var_tols[slack] = tols;
            // A NaN residual holds nothing, and one is held to the tolerance
            // on the side it moves the slack.
            let tol = if residual < 0.0 { tols.lo } else { tols.hi };
            if residual.is_nan() || residual.abs() > tol {
                hold = false;
            }
        }
        hold
    }

    /// Make the values satisfy the rows to within their tolerances and
    /// measure primal feasibility on them. A widened slack bound first
    /// follows its row's magnitude ([`Self::refit_widened_slacks`]). Then a
    /// ladder until the rows hold: the values as they are, then recomputed
    /// through a fresh factorization, then through one at
    /// [`LU_STABLE_THRESHOLD`] where the threshold in force is lower, each
    /// checked and refined once
    /// ([`Self::rows_hold`]): a residual the rows accept, left in any basic
    /// var, can be amplified into the objective. Recomputes the objective
    /// from the values and sets `is_primal_feasible`. A fresh factorization
    /// that repairs the basis leaves another vertex, with its values, reduced
    /// costs and feasibility flags recomputed: no verification. Values that
    /// miss the rows on every rung are [`Verdict::Unrepresentable`]: the
    /// basis cannot represent its vertex, and [`Self::run_phases`] solves
    /// again from the slack basis.
    fn verify_primal(&mut self) -> Result<Verdict, Error> {
        if !self.values_dirty {
            // Verified values that nothing has touched since; only the
            // bounds may have changed.
            self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
            return Ok(Verdict::Checked {
                feasible: self.is_primal_feasible,
            });
        }
        self.undone = None;
        self.refit_widened_slacks();
        let mut hold = self.rows_hold();
        let ladder = [self.lu_threshold, LU_STABLE_THRESHOLD];
        let rungs = if self.lu_threshold < LU_STABLE_THRESHOLD {
            &ladder[..]
        } else {
            &ladder[..1]
        };
        for &threshold in rungs {
            if hold {
                break;
            }
            debug!("verify: the rows miss; recomputing through a factorization at {threshold}");
            if self.factorize_at(threshold) {
                self.recalc_basic_var_vals()?;
                self.recalc_obj_coeffs()?;
                self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
                return Ok(Verdict::Checked { feasible: false });
            }
            self.recalc_basic_var_vals()?;
            hold = self.rows_hold();
        }
        if !hold {
            return Ok(Verdict::Unrepresentable);
        }
        self.values_dirty = false;
        self.cur_obj_val = self.exact_obj_val();
        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        Ok(Verdict::Checked {
            feasible: self.is_primal_feasible,
        })
    }

    /// Whether the rows hold at the values ([`Self::check_rows`]) after one
    /// refinement ([`Self::refine`]). Values the rows hold are refined too: a
    /// residual they accept, which a plain sum may not even see, can sit in
    /// any basic var, inside its bounds or past one, and the basis or another
    /// row can amplify it into the objective. Such a refinement is undone
    /// where it turns values the rows and bounds accept into ones they do
    /// not, as the exact values of an ill-conditioned vertex can be; where
    /// only the bounds reject them, the refined values are kept aside
    /// ([`Self::undone`]), for [`Self::refined_first`] to weigh.
    fn rows_hold(&mut self) -> bool {
        if self.check_rows() {
            let computed = self.basic_var_vals.clone();
            let accepted = self.calc_primal_infeasibility().0 == 0;
            self.refine();
            let rows = self.check_rows();
            let misses = self.bound_misses();
            let kept = rows && (!accepted || misses.is_empty());
            debug!("verify: refined values the rows hold (kept: {kept})");
            if !kept {
                let refined = std::mem::replace(&mut self.basic_var_vals, computed);
                if rows {
                    self.undone = Some(UndoneRefinement {
                        values: refined,
                        misses,
                    });
                }
                self.check_rows();
            }
            return true;
        }
        self.refine();
        let hold = self.check_rows();
        debug!(
            "verify: refined ({})",
            if hold { "rows hold" } else { "rows still miss" }
        );
        hold
    }

    /// Whether the phases go on from the refined values the verification
    /// undid ([`Self::undone`]) at a vertex the dual check accepts, rather
    /// than end there: where some bound they miss is not held yet, and they
    /// gain the objective more than its allowance ([`Self::refinement_gain`]),
    /// evidence that the computed values fall short of the vertex. The bounds
    /// they miss are then held ([`Self::held`]), the answer is kept to return
    /// to where none is yet ([`Self::fallback`]), and the refined values,
    /// which the rows accept and a bound does not, are the values the next
    /// round starts from. Elsewhere the computed values stay, within the
    /// contract, and the answer is theirs.
    fn refined_first(&mut self) -> bool {
        let Some(undone) = self.undone.take() else {
            return false;
        };
        if undone.misses.iter().all(|miss| self.held.contains(miss)) {
            return false;
        }
        let (gain, allowance) = self.refinement_gain(&undone.values);
        let worse = gain > allowance;
        debug!(
            "verify: the refined values gain {gain:e} on the computed ones (allowance \
             {allowance:e}): {}",
            if worse {
                format!("they go on, past {:?}", undone.misses)
            } else {
                "the computed values stay".to_string()
            }
        );
        if !worse {
            return false;
        }
        if self.fallback.is_none() {
            self.fallback = Some(Box::new(self.verified_answer(allowance)));
        }
        for miss in undone.misses {
            if !self.held.contains(&miss) {
                self.held.push(miss);
            }
        }
        self.basic_var_vals = undone.values;
        self.check_rows();
        self.cur_obj_val = self.exact_obj_val();
        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        true
    }

    /// How much the refined values `x̄` (`refined`, one per basis position)
    /// lower the objective from the current, computed ones `x`, `c_B (x - x̄)`
    /// summed in compensated arithmetic, and the allowance for what is no
    /// evidence that the computed values fall short of the vertex. On reduced costs the dual
    /// check accepts, the refined values' objective is the vertex's dual bound
    /// (Neumaier and Shcherbina 2004) to within the dual tolerance, so the gain
    /// is how far the computed values lie above it. Where each basic var lies
    /// within its bound tolerance `tol_j` of its refined value, which the
    /// contract cannot tell apart, the gain is at most `Σ_B |c_j| tol_j`; the
    /// allowance adds the round-off of plain sums of both objectives,
    /// `(n + 2) 2^-53` of their magnitude for `n` terms, which also bounds
    /// what rounding each value to `f64` moves them by.
    fn refinement_gain(&self, refined: &[f64]) -> (f64, f64) {
        let mut gain = CDouble::default();
        let (mut magnitude, mut terms, mut room) = (0.0, 0u32, 0.0);
        for (pos, &var) in self.basic_vars.iter().enumerate() {
            let cost = self.cost_in_force(var);
            if cost == 0.0 {
                continue;
            }
            let (x, x_refined) = (self.basic_var_vals[pos], refined[pos]);
            gain.add_product(cost, x);
            gain.add_product(-cost, x_refined);
            magnitude += (cost * x).abs() + (cost * x_refined).abs();
            terms += 2;
            let tols = self.var_tols[var];
            room += cost.abs() * tols.lo.max(tols.hi);
        }
        let round_off = (f64::from(terms) + 2.0) * UNIT_ROUNDOFF * magnitude;
        (gain.value(), room + round_off)
    }

    /// The bounds the basic vars lie past beyond their tolerances, as (var,
    /// upper).
    fn bound_misses(&self) -> Vec<(usize, bool)> {
        let mut misses = Vec::new();
        for (r, &val) in self.basic_var_vals.iter().enumerate() {
            let var = self.basic_vars[r];
            let tol = self.var_tols[var];
            if val < self.basic_var_mins[r] - tol.lo {
                misses.push((var, false));
            } else if val > self.basic_var_maxs[r] + tol.hi {
                misses.push((var, true));
            }
        }
        misses
    }

    /// One step of iterative refinement of the basic values: the residual
    /// `b - Σ a_j x_j - s` of every row whose slack is non-basic, summed in
    /// compensated arithmetic ([`implied_slack`]) so that it is the values'
    /// own to about an ulp however large the row's terms, then
    /// `B delta = residual` and `x_B += delta`. The correction stays accurate
    /// even through a poor eta file, since a solve's round-off scales with its
    /// right-hand side. A row whose slack is basic has none here: it moves
    /// only its slack, which [`Self::check_rows`] re-derives from the row.
    fn refine(&mut self) {
        let mut residuals: Vec<f64> = (0..self.num_constraints())
            .map(|r| {
                if let VarState::Basic(_) = self.var_states[self.num_vars + r] {
                    return 0.0;
                }
                //guaranteed to be a valid index
                let row = self.orig_constraints.outer_view(r).unwrap();
                // The slack's entry, a one, is among the row's terms.
                let terms = row
                    .iter()
                    .map(|(v, &coeff)| (coeff, self.internal_value(v)));
                implied_slack(self.orig_rhs[r], terms).0
            })
            .collect();
        self.basis_solver.solve_dense_with_etas(&mut residuals);
        for (val, delta) in self.basic_var_vals.iter_mut().zip(&residuals) {
            *val += delta;
        }
    }

    /// The objective of the current values under the costs the simplex works
    /// with: the cost vector in force (the artificial one during phase 1) plus
    /// a running dual phase's perturbation and shifts (see [`costs`]).
    fn exact_obj_val(&self) -> f64 {
        let mut obj = 0.0;
        for (r, &var) in self.basic_vars.iter().enumerate() {
            obj += self.cost_in_force(var) * self.basic_var_vals[r];
        }
        for (c, &var) in self.nb_vars.iter().enumerate() {
            obj += self.cost_in_force(var) * self.nb_var_vals[c];
        }
        obj
    }

    /// The cost `var` has in the simplex: its cost in force (the artificial
    /// one during phase 1), plus a running dual phase's perturbation and shift.
    fn cost_in_force(&self, var: usize) -> f64 {
        let costs = self
            .artificial_obj
            .as_deref()
            .unwrap_or(&self.orig_obj_coeffs);
        if self.costs_modified {
            costs[var] + self.cost_perturbation[var] + self.cost_shifts[var]
        } else {
            costs[var]
        }
    }

    /// Whether the pivot element computed through the row and through the
    /// column agree, to [`PIVOT_AGREEMENT_TOL`] relative to the larger, signs
    /// included. Every pivot exceeds the absolute pivot tolerance of its ratio
    /// test ([`dual_pivot_tolerance`], [`primal_pivot_tolerance`]), far above
    /// the round-off of computing it, so no element needs an allowance for
    /// that round-off.
    fn pivots_agree(a_row: f64, a_col: f64) -> bool {
        (a_row - a_col).abs() <= PIVOT_AGREEMENT_TOL * a_row.abs().max(a_col.abs())
    }

    /// Whether a fresh factorization can still settle a pivot element whose
    /// two computations disagree: the current one has updates on it, or its
    /// threshold can rise ([`Self::raise_lu_threshold`]). At the highest
    /// threshold, fresh, the element is what the arithmetic gives.
    fn can_renew_factorization(&self) -> bool {
        !self.factorization_is_fresh() || self.lu_threshold < LU_STABLE_THRESHOLD
    }

    /// Raise the threshold of the factorizations fivefold, up to
    /// [`LU_STABLE_THRESHOLD`], after the two computations of a pivot element
    /// disagreed, unless ten or more updates sit on the factorization, which
    /// are then the likelier cause. The caller renews the factorization.
    fn raise_lu_threshold(&mut self) {
        if self.basis_solver.eta_matrices.len() < 10 && self.lu_threshold < LU_STABLE_THRESHOLD {
            self.lu_threshold = (self.lu_threshold * 5.0).min(LU_STABLE_THRESHOLD);
            debug!(
                "the factorization's threshold rises to {}",
                self.lu_threshold
            );
        }
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

    /// The all-slack basis (identity basis matrix): the start a node LP's
    /// retry solves from. A non-basic var sits at its lower bound if finite,
    /// else at its upper bound, else at 0.
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
    /// A basis the factorization finds singular is repaired
    /// ([`Self::factorize`]): a dependent var leaves at the value
    /// [`Self::slack_basis`] would give it.
    ///
    /// # Errors
    ///
    /// `InternalError` for a basis of the wrong shape, with the state untouched.
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
        self.basic_var_vals.clear();
        self.basic_var_mins.clear();
        self.basic_var_maxs.clear();
        self.nb_vars.clear();
        self.nb_var_vals.clear();
        self.nb_var_states.clear();

        for var in 0..n {
            match basis.0[var] {
                VarStatus::Basic => {
                    let (min, max) = (self.orig_var_mins[var], self.orig_var_maxs[var]);
                    self.var_states[var] = VarState::Basic(self.basic_vars.len());
                    self.basic_vars.push(var);
                    // Until the recompute below, the value the slack basis
                    // gives the var, where it goes if the repair takes it out.
                    self.basic_var_vals.push(if min.is_finite() {
                        min
                    } else if max.is_finite() {
                        max
                    } else {
                        0.0
                    });
                    self.basic_var_mins.push(min);
                    self.basic_var_maxs.push(max);
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
                    self.var_states[var] = VarState::NonBasic(self.nb_vars.len());
                    self.nb_vars.push(var);
                    self.nb_var_vals.push(val);
                    self.nb_var_states.push(NonBasicVarState::at(val, min, max));
                }
            }
        }

        self.factorize();

        // The weights belong to the positions the load renumbers (they only
        // affect the pivot order, not correctness). The dual ones restart from
        // a reference reset, standard after a warm-start load; the primal ones
        // are recomputed exactly, cheap for the slack basis, the one basis
        // loaded while they are in use (the restart of `run_phases`).
        if self.enable_dual_steepest_edge {
            self.dual_edge_sq_norms = vec![1.0; self.basic_vars.len()];
        }
        if self.enable_primal_steepest_edge {
            self.recalc_primal_sq_norms();
        }

        self.recalc_basic_var_vals()?;
        self.recalc_obj_coeffs()?;

        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        Ok(())
    }

    /// Fix a structural var to `val` and re-solve: a bound change to
    /// `[val, val]` followed by [`Self::reoptimize`], whether the var is basic
    /// or not. A basic var that does not hold `val` is repaired by the dual
    /// simplex like any other bound violation.
    ///
    /// `val` must lie within the var's bounds as they were before any fix, to
    /// within the contract, and one beyond it returns `Err(Infeasible)`. The
    /// var is fixed at `val` itself, also past a bound: a value a solve
    /// reported there met its rows there, and at the bound a row could miss
    /// by the offset times its coefficient, which the proof, granting a fixed
    /// var no room, would call infeasible. Fixing again replaces the previous
    /// fixed value.
    pub(crate) fn fix_var(&mut self, var: usize, val: f64) -> Result<StopReason, Error> {
        let (min, max) = self.fixed_var_bounds[var].unwrap_or(self.get_var_bounds(var));
        if outside_bounds(val, min, max, self.feasibility) {
            return Err(Error::Infeasible);
        }
        self.set_var_bounds(var, val, val)?;
        self.fixed_var_bounds[var] = Some((min, max));
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
    /// [`bounds`]) that [`Self::begin_primal_phase`] perturbs unless the start
    /// is near optimal or the phase is a `cleanup` (see [`Self::run_phases`]),
    /// and that return to the true bounds however the phase ends. Returns the
    /// stop reason and whether any pivot was made; ends with
    /// `is_primal_feasible` measured on the true bounds.
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
    /// Each iteration prices a var that is not taboo ([`Self::price_primal`]),
    /// runs the ratio test ([`Self::choose_pivot`]) and, unless the basis
    /// change is bad ([`CycleGuard`]), applies it, then shifts the working
    /// bound of every basic var the step left outside its bounds beyond its
    /// tolerance ([`Self::correct_primal`]). Where the two computations of the
    /// pivot element disagree, the factorization is renewed, at a raised
    /// threshold ([`Self::raise_lu_threshold`]), and the iteration redone; at
    /// the highest threshold the ratio test settles the element on the refined
    /// column instead.
    /// Where the reduced costs were updated since their last recompute, the
    /// entering var's is first recomputed from its column: a var it shows no
    /// longer improving does not enter, and a value on the other side of
    /// zero beyond its tolerance has the data rebuilt.
    ///
    /// A ray is `Unbounded` only on fresh data: a ray found since the last
    /// rebuild is priced again after one, so that its reduced cost comes from
    /// refined multipliers and its column through a fresh factorization.
    ///
    /// The phase fails loudly rather than run forever: when, after a rebuild,
    /// the only vars left to enter are taboo, and when the objective has not
    /// fallen beyond round-off in [`STALL_ITERATIONS`] iterations.
    fn primal_phase(&mut self, harris_scale: f64) -> Result<(StopReason, bool), Error> {
        let mut pivoted = false;
        // Whether the data are fresh from a rebuild with no step since: only
        // then do taboo vars alone left to enter mean the phase is stuck, and
        // a ray mean `Unbounded`.
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
                // Every primal step is forward or zero (a backward one is
                // taken as zero, see `choose_pivot`) and, with the bounds
                // perturbed, zero steps are the exception, so a phase that
                // progresses lowers the objective (a rebuild moves it by
                // round-off either way, and a basis repair by any amount, so
                // a repair shortly before a check can end the phase here, a
                // rare and loud end). One that has not fallen beyond round-off
                // since the last check, or is not a number, is taken to be
                // stuck in a cycle the vertex hash has not caught.
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
                    // left. Retry them on fresh data, where the pricing may
                    // differ; if nothing moved since the last rebuild, the
                    // phase is stuck.
                    if refreshed_since_step {
                        return Err(self.stuck_error("primal"));
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
            match self.choose_pivot(col, harris_scale)? {
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
                }
                PivotChoice::Inconsistent { row } => {
                    debug!(
                        "optimize iter {iter}: pivot row/column disagreement at row {row}; \
                         rebuilding"
                    );
                    self.raise_lu_threshold();
                    self.rebuild_primal()?;
                    refreshed_since_step = true;
                }
                PivotChoice::Unresolved { row } => {
                    debug!(
                        "optimize iter {iter}: only entries of unresolved sign block the step, \
                         the last in row {row}; rebuilding"
                    );
                    self.raise_lu_threshold();
                    self.rebuild_primal()?;
                    refreshed_since_step = true;
                }
                PivotChoice::Unbounded => {
                    if refreshed_since_step {
                        return Err(Error::Unbounded);
                    }
                    debug!("optimize iter {iter}: a ray on updated data; rebuilding");
                    self.rebuild_primal()?;
                    refreshed_since_step = true;
                }
                PivotChoice::Drifted { sign_error } => {
                    // The var's reduced cost is its column's now; price
                    // again, on fresh data where the drift crossed zero.
                    if sign_error {
                        debug!(
                            "optimize iter {iter}: an entering reduced cost has the other sign \
                             on its column; rebuilding"
                        );
                        self.rebuild_primal()?;
                        refreshed_since_step = true;
                    }
                }
            }
        }

        Ok((StopReason::Finished, pivoted))
    }

    /// A rebuild within the primal phase: as [`Self::rebuild`], with the
    /// reduced costs on multipliers refined once, as at the verification,
    /// since a ray is declared, and a phase that made no pivot ends, only on
    /// the reduced costs of a rebuild or of the verification; then the
    /// working bounds shifted where the recomputed values violate them, and
    /// every taboo lifted so that any var may enter on fresh data (the
    /// records of bad basis changes stay: see [`CycleGuard`]).
    fn rebuild_primal(&mut self) -> Result<(), Error> {
        self.factorize();
        self.recalc_basic_var_vals()?;
        self.recalc_obj_coeffs_refined();
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

    /// Drop the records of bad basis changes whose row the step just made (the
    /// entering var's move `entering_diff` along `col_coeffs`, a pivot or a
    /// bound flip) moved by more than the row's tolerance, at all for a row
    /// whose basic var is held exactly: the state the change was bad in is
    /// gone (see [`CycleGuard::forget_moved_rows`]). It runs after the pivot,
    /// so the change back that a repair at the pivot's refactorization records
    /// goes too if the step moved its row: the phase may take that change
    /// again, on the eta file, rather than run out of changes.
    fn forget_moved_rows(&mut self, entering_diff: f64) {
        let (col_coeffs, basic_vars, var_tols) =
            (&self.col_coeffs, &self.basic_vars, &self.var_tols);
        self.cycle_guard.forget_moved_rows(|r| {
            let tol = var_tols[basic_vars[r]];
            let coeff = col_coeffs
                .iter()
                .find(|&(i, _)| i == r)
                .map_or(0.0, |(_, &coeff)| coeff);
            (coeff * entering_diff).abs() > tol.lo.min(tol.hi)
        });
    }

    /// The dual simplex phase: restore primal feasibility while keeping the
    /// reduced costs dual feasible. It runs on working costs (see [`costs`])
    /// that [`Self::begin_dual_phase`] perturbs unless the start is near
    /// optimal or the phase is a `cleanup`, and whose initial dual
    /// infeasibilities it corrects; they return to the costs in force however
    /// the phase ends.
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
    /// score that is not taboo ([`CycleGuard`]), runs the ratio test, and
    /// applies the step (the flips it passed, then the pivot) unless the basis
    /// change is bad. Every rebuild ([`Self::rebuild_dual`]) corrects the dual
    /// infeasibilities the recomputed reduced costs show.
    ///
    /// `Infeasible` is returned only when a row that no var can repair on
    /// fresh data ([`DualRatioTest::NoCandidate`]) has a certificate that
    /// holds on the model's data ([`Self::prove_infeasible`]); a row whose
    /// proof fails is made taboo. When only taboo rows are left on fresh data,
    /// one whose proof failed gets the last resorts: a pivot on a genuine
    /// entry below the pivot tolerance ([`Self::genuine_entry_step`]), then
    /// the certificate repair ([`Self::certificate_repair`]): the bounds the
    /// non-basic vars the row's certificate says can help sit at, widened by
    /// their room for the rest of the solve, after which the cycling detection
    /// starts afresh, the LP having changed.
    ///
    /// Where the two computations of the pivot element disagree, the
    /// factorization is renewed, at a raised threshold
    /// ([`Self::raise_lu_threshold`]), and the iteration redone; on a fresh
    /// factorization at the highest threshold the column's element is taken
    /// if it is a pivot itself, of the row's sign and above the pivot
    /// tolerance, and the solve fails loudly otherwise.
    ///
    /// The phase fails loudly rather than run forever or guess: when only
    /// taboo rows are left after a rebuild and no last resort remains (a row
    /// the last resorts leave unrepaired has [`Self::run_phases`] start over
    /// once from the slack basis), and when the dual objective has not risen
    /// beyond round-off in [`STALL_ITERATIONS`] iterations. The room an
    /// unrepaired row's certificate leaves then lies where no step reaches:
    /// in an entry the refinement cannot settle of a var with no bound in its
    /// direction, in a coefficient below the round-off of its terms, within
    /// the round-off of the room every term of a certificate of several rows
    /// can give ([`Self::leaving_slack_cut`]), or beyond the values computed
    /// through the basis (see the known limitations in ARCHITECTURE §7.9).
    fn dual_phase(&mut self) -> Result<StopReason, Error> {
        let obj_str = if self.artificial_obj.is_some() {
            "artificial obj."
        } else {
            "obj."
        };

        // Whether the data are fresh from a rebuild, with no pivot since: a
        // row's verdict, or the loud error when only taboo rows are left, is
        // drawn on fresh data only. Only the unproven row with the best score
        // gets the last resorts: if they fail on it, the phase fails without
        // trying another (a loud error where there might have been an answer,
        // never a wrong answer).
        let mut refreshed_since_pivot = false;
        // The dual objective at the last stall check.
        let mut stall_reference = self.cur_obj_val;
        // The unproven row the next iteration gives the genuine entries of its
        // refined row, and the last row given them since the last rebuild.
        let mut last_resort_row: Option<usize> = None;
        let mut last_resort_tried: Option<usize> = None;

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
                // makes zero steps rare, so a phase that progresses raises the
                // dual objective of the working costs over many iterations (a
                // shift moves it either way, and is given back when its var
                // leaves; a basis repair moves it by any amount). One that has
                // not risen beyond round-off since the last check, or is not a
                // number, is taken to be stuck in a cycle the vertex hash
                // missed (see `STALL_ITERATIONS`).
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

            let (row, leaving_new_val, last_resort) = if let Some(row) = last_resort_row.take() {
                (row, self.leaving_target(row), true)
            } else if let Some((row, leaving_new_val)) = self.choose_pivot_row_dual() {
                (row, leaving_new_val, false)
            } else {
                if self.taboo_row_violated() {
                    // Only taboo rows are left: rows whose basis change
                    // returns to visited vertices, or that no var can repair
                    // and that do not prove infeasibility. Retry them on
                    // fresh data, where the entering var may differ; if
                    // nothing moved since the last rebuild, an unproven row
                    // gets the last resorts, and otherwise the phase is stuck.
                    if refreshed_since_pivot {
                        if let Some(row) = self.unproven_row() {
                            // First a genuine entry below the pivot tolerance,
                            // once per rebuild ([`Self::genuine_entry_step`]).
                            if last_resort_tried != Some(row) {
                                debug!(
                                    "restore feasibility iter {iter}: only taboo rows are left; \
                                     the genuine entries of unproven row {row}"
                                );
                                last_resort_tried = Some(row);
                                last_resort_row = Some(row);
                                continue;
                            }
                            // Then the certificate repair's widening, or
                            // nothing can repair the row. A widening changes
                            // the LP, so the cycling detection starts afresh.
                            if !self.certificate_repair(row) {
                                self.unrepaired_row = Some(row);
                                return Err(Error::InternalError(format!(
                                    "dual simplex: no var can repair row {row} (basic var {}) by \
                                     a pivot or through a coefficient above round-off, and the \
                                     row does not prove the model infeasible",
                                    self.basic_vars[row]
                                )));
                            }
                            self.start_cycle_detection();
                            refreshed_since_pivot = false;
                            last_resort_tried = None;
                            continue;
                        }
                        return Err(self.stuck_error("dual"));
                    }
                    debug!(
                        "restore feasibility iter {iter}: only taboo rows are violated; rebuilding"
                    );
                    self.rebuild_dual()?;
                    refreshed_since_pivot = true;
                    last_resort_tried = None;
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
            let test = if last_resort {
                self.genuine_entry_step(row, leaving_new_val)
            } else {
                self.calc_row_coeffs(row);
                self.choose_entering_col_dual(row, leaving_new_val)
            };
            let step = match test {
                Ok(DualRatioTest::Step(step)) => step,
                // The certificate repair is next for a row no genuine entry
                // repairs either.
                Ok(DualRatioTest::NoCandidate) if last_resort => continue,
                Ok(DualRatioTest::DualInfeasible) => {
                    // A candidate's reduced cost lies beyond its tolerance
                    // on its infeasible side (drift of the incremental
                    // updates, or an earlier step its entry, at or below the
                    // pivot tolerance, did not bound: see `rebuild_dual`):
                    // recompute and correct. The correction leaves every
                    // reduced cost feasible, so this cannot follow a rebuild.
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
                    continue;
                }
                Ok(DualRatioTest::NoCandidate) if !refreshed_since_pivot => {
                    // "No eligible entering column" can be a phantom: deep
                    // in an eta-file chain the basic values drift, and the
                    // equally drifted pivot row then blocks every candidate.
                    // Look at the row again on fresh data, where a phantom
                    // dissolves, before drawing a conclusion.
                    debug!(
                        "restore feasibility iter {iter}: no entering column for row {row}; \
                         rebuilding before a verdict"
                    );
                    self.rebuild_dual()?;
                    refreshed_since_pivot = true;
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
            // The pivot element is known through the row (transposed solve)
            // and the column (forward solve), a refined row's against the
            // refined column: like with like. Where they disagree, the
            // factorization is renewed, at a raised threshold, and the
            // iteration redone. On a fresh factorization at the highest
            // threshold the column's element is what the arithmetic gives: it
            // divides the step, so it must have the row's sign and, unless
            // both were refined, exceed the pivot tolerance, or the solve
            // fails loudly. A basis the pivot leaves singular is repaired at
            // the next factorization.
            self.calc_col_coeffs(col);
            if step.refined {
                self.refine_dual_pivot_col(col)?;
            }
            let a_col = self.col_coeff_at(row);
            if !Self::pivots_agree(a_row, a_col) {
                if self.can_renew_factorization() {
                    debug!(
                        "restore feasibility iter {iter}: pivot row/column disagreement \
                         ({a_row:e} vs {a_col:e}); rebuilding"
                    );
                    self.raise_lu_threshold();
                    self.rebuild_dual()?;
                    continue;
                }
                let pivot_tol = dual_pivot_tolerance(self.basis_solver.eta_matrices.len());
                if a_col == 0.0
                    || (a_row > 0.0) != (a_col > 0.0)
                    || (!step.refined && a_col.abs() <= pivot_tol)
                {
                    return Err(Error::InternalError(format!(
                        "dual simplex: the pivot element of var {} in row {row} is {a_row:e} \
                         through the row and {a_col:e} through the column of a fresh \
                         factorization",
                        self.nb_vars[col]
                    )));
                }
                debug!(
                    "restore feasibility iter {iter}: pivot row/column disagreement \
                     ({a_row:e} vs {a_col:e}) on a fresh factorization; pivoting"
                );
            }
            // The flips the step passed come first. They move basic values,
            // not the basis, so the pivot column stays valid and the entering
            // var's step is measured from where they leave the leaving var.
            // The pivot is made even if they carried it to or past its bound
            // (their row predicted less): the step is then tiny or reversed,
            // and a reversed step's infeasibility is left to the next
            // iterations. Flips never come with a zero dual step, so they
            // cannot alternate without progress.
            //
            // The column's element drives the primal step, the eta file and
            // the steepest-edge updates, which apply it with the rest of the
            // column; the row's drives the reduced costs, updated along it.
            self.apply_flips(&step.flips);
            if step.zero_step {
                // The dual step `d_q / a_row` would run backwards: the
                // entering var's cost is shifted so its reduced cost is
                // zero, and the step, zero, changes no other.
                self.shift_entering_cost(col);
            }
            if let Some(up) = step.crosses {
                self.absorbed.push((self.nb_vars[col], up));
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
        }

        // Provisional: `run_phases` confirms it on recomputed values.
        self.is_primal_feasible = true;
        Ok(StopReason::Finished)
    }

    /// A rebuild within the dual phase: a fresh factorization and everything
    /// derived from it ([`Self::rebuild`]), the dual infeasibilities the
    /// recomputed reduced costs show corrected, and every taboo lifted so that
    /// any row may be tried again (the records of bad basis changes stay: see
    /// [`CycleGuard`]).
    ///
    /// The correction shifts every infeasible cost, a boxed var's too: the
    /// ratio test leaves an entry at or below the pivot tolerance out of the
    /// step's bound, so a step can leave its var far dual infeasible, and
    /// moving it to its other bound would return the phase to a vertex it left
    /// (as on the model of `wide_lp_1897_solves_within_the_contract` in
    /// `src/tests/magnitudes.rs`). A shift keeps the vertex, and the cleanup
    /// gives it back.
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

    /// The loud end of the `phase` ("dual" or "primal") when fresh data leave
    /// only taboo basis changes, naming a basis the factorization found
    /// singular if a change left leads to one.
    fn stuck_error(&self, phase: &str) -> Error {
        Error::InternalError(if self.cycle_guard.any_taboo_singular() {
            format!(
                "{phase} simplex: every basis change left leads to a singular basis or returns \
                 to visited vertices"
            )
        } else {
            format!("{phase} simplex cycles: every basis change left returns to visited vertices")
        })
    }

    /// Add a row of the user's model, numbered after the user's rows so far,
    /// and re-solve.
    pub(crate) fn add_constraint(
        &mut self,
        coeffs: CsVec,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<StopReason, Error> {
        let user_row = self.user_rows.len();
        self.user_rows.push(None);
        self.add_row(coeffs, cmp_op, rhs, Some(user_row))
    }

    /// Add the row `coeffs cmp_op rhs` to the engine, as the user's row
    /// `user_row` where one is given ([`Self::user_rows`]), and re-solve.
    fn add_row(
        &mut self,
        coeffs: CsVec,
        cmp_op: ComparisonOp,
        rhs: f64,
        user_row: Option<usize>,
    ) -> Result<StopReason, Error> {
        assert!(self.is_primal_feasible);

        let is_fixed = |var: usize| self.orig_var_mins[var] == self.orig_var_maxs[var];
        let user_coeffs = coeffs.clone();
        let coeffs = scale_columns(coeffs, &self.col_scales);
        let Some(PreparedRow {
            mut coeffs,
            rhs,
            row_scale,
            slack_var_min,
            slack_var_max,
        }) = prepare_row(coeffs, cmp_op, rhs, self.feasibility, is_fixed)?
        else {
            return Ok(StopReason::Finished);
        };
        let slack_unit = slack_dual_unit(&user_coeffs, row_scale, is_fixed);

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
        // The new slack's reduced cost is exactly zero until the multipliers
        // are next recomputed.
        self.dual_tols.push(dual_tol(0.0, slack_unit));

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
        self.slack_dual_units.push(slack_unit);

        self.orig_constraints = new_orig_constraints;
        self.orig_constraints_csc = self.orig_constraints.to_csc();

        // The new slack, basic: its row evaluated at the current values.
        let new_row = self.num_constraints() - 1;
        if let Some(user_row) = user_row {
            self.user_rows[user_row] = Some(new_row);
        }
        let (value, tols) = self.row_slack(new_row);
        self.basic_var_vals.push(value);
        self.var_tols.push(tols);

        if self.enable_dual_steepest_edge {
            // A placeholder, so that every row has its weight when the
            // factorization's repair resets some; the new row's is set below.
            self.dual_edge_sq_norms.push(1.0);
        }
        let repaired = self.factorize();

        if self.enable_primal_steepest_edge || self.enable_dual_steepest_edge {
            // The new row's contribution to the weights. The existing tableau
            // rows are unchanged unless the factorization repaired the basis;
            // then the weights are approximate, and they only steer pricing.
            self.calc_row_coeffs(new_row);

            if self.enable_primal_steepest_edge {
                for (c, &coeff) in self.row_coeffs.iter() {
                    self.primal_edge_sq_norms[c] += coeff * coeff;
                }
            }

            if self.enable_dual_steepest_edge {
                self.dual_edge_sq_norms[new_row] = self.inv_basis_row_coeffs.sq_norm();
            }
        }
        if repaired {
            self.recalc_basic_var_vals()?;
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

    /// The pivot row of `row` refined, for a pivot the growth test finds
    /// small, a row no pivot above the tolerance repairs (see
    /// [`Self::choose_entering_col_dual`]), or a pivot read off a refined
    /// column (see [`Self::choose_pivot`]): the row of the basis inverse the
    /// infeasibility proof reads, refined twice in doubled precision
    /// ([`Self::refined_basis_inverse_row`]), and the entries of the
    /// non-basic columns summed from it in compensated arithmetic, their
    /// products exact. The refined row replaces `row_coeffs` and, rounded,
    /// `inv_basis_row_coeffs`, so that the ratio test, the pivot's
    /// reduced-cost update and the steepest-edge updates all use it.
    ///
    /// Returns, per non-basic column, the error of its entry: how far the
    /// second step moved it, plus the round-off of the compensated sum. An
    /// entry beyond its error is genuine: the refinement has settled it. The
    /// second correction carried through the column's entries is that move,
    /// with the cancellation the multipliers themselves have in that column:
    /// through an ill-conditioned basis the multipliers' corrections are
    /// large, but along the column's own direction they cancel as the
    /// multipliers do. An entry that is zero but for the error of computing it
    /// moves by its own size and is not genuine, however it came out; one the
    /// data leave, however small and however inexactly the plain row computed
    /// it, is. A refinement that is not finite settles nothing: every error is
    /// infinite.
    fn refine_pivot_row(&mut self, row: usize) -> Vec<f64> {
        let refined = self.refined_basis_inverse_row(row);
        let num_cols = self.nb_vars.len();
        let mut entries = vec![CDouble::default(); num_cols];
        // Per column, how far the second step moved its entry, and the
        // magnitude and count of the terms the entry is summed from.
        let mut moves = vec![CDouble::default(); num_cols];
        let mut magnitudes = vec![0.0; num_cols];
        let mut terms = vec![0u32; num_cols];
        for (r, &y) in refined.rho.iter().enumerate() {
            let y = y.normalized();
            let y_value = y.value();
            let second = refined.second.get(r).copied().unwrap_or(0.0);
            if y_value == 0.0 && second == 0.0 {
                continue;
            }
            //guaranteed to be a valid index
            for (v, &val) in self.orig_constraints.outer_view(r).unwrap().iter() {
                if let VarState::NonBasic(idx) = self.var_states[v] {
                    entries[idx].add_scaled(y, val);
                    moves[idx].add_product(second, val);
                    magnitudes[idx] += (y_value * val).abs();
                    terms[idx] += 1;
                }
            }
        }
        let settled = !refined.second.is_empty();
        let errors = (0..num_cols)
            .map(|idx| {
                let count = terms[idx] as usize;
                let sum_error = compensated_error(count, count, magnitudes[idx]);
                if settled {
                    moves[idx].value().abs() + sum_error
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        self.row_coeffs.clear_and_resize(self.nb_vars.len());
        for (idx, entry) in entries.iter().enumerate() {
            let value = entry.value();
            if value != 0.0 {
                *self.row_coeffs.get_mut(idx) = value;
            }
        }
        let mut rho = ScatteredVec::empty(self.num_constraints());
        for (r, y) in refined.rho.iter().enumerate() {
            let value = y.value();
            if value != 0.0 {
                *rho.get_mut(r) = value;
            }
        }
        rho.to_sparse_vec(&mut self.inv_basis_row_coeffs);
        errors
    }

    /// The primal simplex's pricing: the non-basic column of the dual
    /// infeasible var, not taboo ([`CycleGuard`]), with the best steepest-edge
    /// score (the largest reduced cost when the weights are off); `None` when
    /// no var that may enter can improve the objective.
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
                // not taboo.
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
        // comparison with it is false), but its score would never win the
        // choice below, and a phase whose only candidates were such costs
        // would end as optimal: it is a loud error instead.
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

    /// The primal simplex's ratio test for the entering column `entering_c`,
    /// on the phase's working bounds (see [`bounds`]): the basic var that
    /// leaves for it, a flip of the entering var to its other bound, or an
    /// unbounded ray; none where the reduced cost that priced the var, updated
    /// since its last recompute, is not confirmed by the column
    /// ([`Self::check_entering_reduced_cost`]).
    /// `harris_scale` scales the tolerance by which the test lets a basic var
    /// pass its bound (see `run_phases`).
    fn choose_pivot(&mut self, entering_c: usize, harris_scale: f64) -> Result<PivotChoice, Error> {
        self.calc_col_coeffs(entering_c);

        // An entry of the column, or the value of a basic var in a row the
        // column moves, that is not a finite number is a breakdown, not
        // data: it compares false with every bound below, so its row would
        // drop out of the test unseen and the step could run past it, or be
        // taken for an unbounded ray. It is a loud error instead.
        for (r, &coeff) in self.col_coeffs.iter() {
            if !coeff.is_finite() || !self.basic_var_vals[r].is_finite() {
                return Err(Error::InternalError(format!(
                    "primal simplex: entry {coeff} of var {} in row {r}, whose basic var is at \
                     {}, is not finite",
                    self.nb_vars[entering_c], self.basic_var_vals[r]
                )));
            }
        }

        // Reduced costs updated pivot by pivot drift: the one that priced the
        // var is checked against its column. Fresh ones, from refined
        // multipliers, are not: a column computed through the factorization
        // is no better.
        if self.reduced_costs_dirty {
            if let Some(drifted) = self.check_entering_reduced_cost(entering_c)? {
                return Ok(drifted);
            }
        }
        let entering_cur_val = self.nb_var_vals[entering_c];
        // If true, entering variable will increase (because the objective function must decrease).
        let entering_diff_sign = self.nb_var_obj_coeffs[entering_c] < 0.0;
        let (entering_min, entering_max) = self.working_bounds(self.nb_vars[entering_c]);
        let entering_other_val = if entering_diff_sign {
            entering_max
        } else {
            entering_min
        };

        // An entry at or below the pivot tolerance (absolute, larger the
        // longer the eta file the column was computed through, see
        // `primal_pivot_tolerance`) neither bounds the step nor is a pivot,
        // and its basic var's bound is shifted after the step where the var
        // ends outside it beyond its tolerance (`correct_primal`). Every other
        // entry bounds the step and can be the pivot. A step that nothing
        // above the tolerance bounds, toward an infinite bound, would leave
        // such a var past its bound by an unbounded amount, which no shift
        // makes good: the column is refined and judged
        // ([`Self::judge_pivot_col`]), and every entry beyond its error bounds
        // the step, however small. No entry within its error can justify a
        // pivot, which could leave the basis singular: it may be a genuine
        // blocker whose sign the arithmetic cannot tell. A column toward an
        // infinite bound blocked only by such entries is retried on a stronger
        // fresh factorization; there it is a ray where no entry of unresolved
        // sign, blocking or not, can take its basic var from the point the
        // phase started on past a bound by more than its tolerance before the
        // entering var reaches the magnitude taken for infinite
        // ([`Self::unresolved_entries_leave_a_ray`]), and the solve fails
        // loudly otherwise.
        let var = self.nb_vars[entering_c];
        let pivot_tol = primal_pivot_tolerance(self.basis_solver.eta_matrices.len());
        let max_step = (entering_other_val - entering_cur_val).abs();
        let toward_infinite = entering_other_val.is_infinite();
        let mut choice =
            self.primal_harris(entering_diff_sign, max_step, harris_scale, |_, coeff| {
                coeff.abs() > pivot_tol
            });
        let mut judgment = None;
        if choice.is_none() && toward_infinite {
            let judged = self.judge_pivot_col(entering_c, entering_diff_sign, None)?;
            choice = self.judged_harris(
                &judged,
                entering_diff_sign,
                max_step,
                harris_scale,
                toward_infinite,
            );
            judgment = Some(judged);
        }

        // The pivot element is checked through the row as in the dual phase
        // (see `dual_phase`), a refined one through the refined row, like
        // with like ([`element_verdict`]); a disagreement renews the
        // factorization (`PivotChoice::Inconsistent`). On a fresh one at the
        // highest threshold the plain values cannot settle the element, whose
        // pivot leaves the basis singular if it is zero: the column is judged
        // refined instead, the element unresolved unless the refinement
        // resolves it. The column's value drives the step, the row's the
        // reduced costs.
        while let Some((row, _, pivot_coeff)) = choice {
            self.calc_row_coeffs(row);
            let row_errors = judgment.is_some().then(|| self.refine_pivot_row(row));
            let a_row = *self.row_coeffs.get(entering_c);
            let agree = Self::pivots_agree(a_row, pivot_coeff);
            if !agree && self.can_renew_factorization() {
                return Ok(PivotChoice::Inconsistent { row });
            }
            let (Some(judged), Some(row_errors)) = (judgment.as_mut(), &row_errors) else {
                if agree {
                    break;
                }
                debug!(
                    "primal ratio test: pivot row/column disagreement ({a_row:e} vs \
                     {pivot_coeff:e}) in row {row} on a fresh factorization; judging the \
                     refined column"
                );
                let judged = self.judge_pivot_col(entering_c, entering_diff_sign, Some(row))?;
                choice = self.judged_harris(
                    &judged,
                    entering_diff_sign,
                    max_step,
                    harris_scale,
                    toward_infinite,
                );
                judgment = Some(judged);
                continue;
            };
            let row_error = row_errors[entering_c];
            let verdict = element_verdict(
                a_row,
                row_error,
                pivot_coeff,
                judged.errors[row],
                judged.converged,
                judged.disputed,
            );
            match verdict {
                ElementVerdict::Take => {
                    if !agree {
                        debug!(
                            "primal ratio test: pivot row/column disagreement ({a_row:e} vs \
                             {pivot_coeff:e}) on a fresh factorization; pivoting"
                        );
                    }
                    break;
                }
                ElementVerdict::Refuse => {
                    return Err(Error::InternalError(format!(
                        "primal simplex: the pivot element of var {var} in row {row} is \
                         {a_row:e} through the refined row and {pivot_coeff:e} through the \
                         refined column of a fresh factorization"
                    )));
                }
                ElementVerdict::Unresolve => {
                    debug!(
                        "primal ratio test: the pivot element {pivot_coeff:e} of var {var} in \
                         row {row}, {a_row:e} within {row_error:e} through the refined row, is \
                         unresolved"
                    );
                    judged.unresolve(row, pivot_coeff, proof::add_up(a_row.abs(), row_error));
                    choice = self.judged_harris(
                        judged,
                        entering_diff_sign,
                        max_step,
                        harris_scale,
                        toward_infinite,
                    );
                }
            }
        }

        if let Some((row, mut leaving_new_val, pivot_coeff)) = choice {
            // The step takes the leaving var exactly onto its working bound,
            // and a var fixed in the model onto its value, whatever the phase
            // did to its working bounds: an equality's slack then holds its
            // row as computed, and a fixed integer var the value a row
            // presolve rewrote relies on. Where the leaving var sits past that
            // target (within its tolerance, as the Harris pass allows), the
            // step would run backwards, moving the entering var and every
            // basic var against the improving direction by the overshoot over
            // the element, which a small element makes more than their
            // tolerances. The step is zero instead, and the leaving var leaves
            // where it sits (`hold_leaving_value`).
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

            return Ok(PivotChoice::Pivot(PivotInfo {
                col: entering_c,
                entering_new_val,
                entering_diff,
                elem: Some(PivotElem {
                    row,
                    coeff: pivot_coeff,
                    dual_step_coeff: pivot_coeff,
                    leaving_new_val,
                }),
            }));
        }
        // Nothing resolved bounds a step toward a finite bound: the entering
        // var moves to it, and a basic var an unresolved entry moves past its
        // bound has that bound shifted, as for an entry at or below the
        // tolerance.
        if !toward_infinite {
            return Ok(PivotChoice::Pivot(PivotInfo {
                col: entering_c,
                entering_new_val: entering_other_val,
                entering_diff: entering_other_val - entering_cur_val,
                elem: None,
            }));
        }

        // An entry within its error is unresolved, not zero: it cannot justify
        // a pivot on that entry, nor a ray unless it is negligible over the
        // whole step. Retry on a stronger fresh factorization first; there,
        // only a refinement that converged measured the error such a ray
        // needs, and the ray covers every other entry whose sign is open too.
        if let Some(mut judged) = judgment {
            if let Some(&(row, _)) = judged.unresolved.last() {
                if self.can_renew_factorization() {
                    return Ok(PivotChoice::Unresolved { row });
                }
                if judged.converged {
                    let mut open = judged.unresolved.clone();
                    open.extend(self.other_entries_of_open_sign(
                        var,
                        &judged.column,
                        &judged.floored,
                        &mut judged.errors,
                    )?);
                    if self.unresolved_entries_leave_a_ray(entering_c, &open, &judged.errors) {
                        debug!(
                            "primal ratio test: no entry of var {var}'s column of unresolved \
                             sign takes its basic var beyond its tolerance up to the infinite \
                             magnitude: a ray"
                        );
                        return Ok(PivotChoice::Unbounded);
                    }
                }
                return Err(Error::InternalError(format!(
                    "primal simplex: a nonzero blocking entry of var {var} in row {row} \
                     is unresolved within residual round-off; unboundedness cannot be determined"
                )));
            }
        }
        Ok(PivotChoice::Unbounded)
    }

    /// The pivot column of `entering_c` refined until a correction converges
    /// ([`Self::refine_pivot_col`]) and judged for an entering var that moves
    /// up (`entering_up`) or down: per row, the error within which its entry
    /// is unresolved, the error the refinement measured for it plus, for an
    /// entry beyond that which moves its basic var toward a finite bound, the
    /// round-off floor below which the refinement cannot tell it from zero
    /// ([`Self::refined_col_floors`]); and the entries within it. The entry of
    /// the `disputed` row, whose plain computations disagreed, gets its floor
    /// whatever its size and sign, zero included: it is unresolved unless the
    /// refinement resolves it.
    fn judge_pivot_col(
        &mut self,
        entering_c: usize,
        entering_up: bool,
        disputed: Option<usize>,
    ) -> Result<ColumnJudgment, Error> {
        let mut column = self.refine_pivot_col(entering_c)?;
        let converged = column.errors.is_some();
        let mut errors = column
            .errors
            .take()
            .unwrap_or_else(|| vec![0.0; self.num_constraints()]);
        // The floors of the entries beyond their measured error that move
        // their basic var toward a finite bound, the only ones that can bound
        // the step, and of the disputed entry.
        let (mut floored, mut entries): (Vec<usize>, Vec<f64>) = self
            .col_coeffs
            .iter()
            .filter(|&(r, &coeff)| {
                let bound = if moves_up(entering_up, coeff) {
                    self.basic_var_maxs[r]
                } else {
                    self.basic_var_mins[r]
                };
                Some(r) == disputed || (coeff.abs() > errors[r] && bound.is_finite())
            })
            .map(|(r, &coeff)| (r, coeff))
            .unzip();
        if let Some(row) = disputed {
            if !floored.contains(&row) {
                floored.push(row);
                entries.push(0.0);
            }
        }
        let var = self.nb_vars[entering_c];
        let floors = self.refined_col_floors(var, &floored, &column)?;
        let mut unresolved = Vec::new();
        for ((&r, entry), floor) in floored.iter().zip(entries).zip(floors) {
            errors[r] = proof::add_up(errors[r], floor);
            if entry.abs() <= errors[r] {
                unresolved.push((r, entry));
                debug!(
                    "primal ratio test: entry {entry:e} of var {var} in row {r} is within the \
                     round-off floor {floor:e} of the refined column; its sign is unresolved"
                );
            }
        }
        Ok(ColumnJudgment {
            column,
            converged,
            disputed: disputed.is_some(),
            errors,
            floored,
            unresolved,
        })
    }

    /// The primal ratio test ([`Self::primal_harris`]) over the entries of the
    /// refined column `judged` beyond their error, for an entering var that
    /// moves up (`entering_up`) or down by at most `max_step`; unless the step
    /// is `toward_infinite`, only those above the pivot tolerance, as on the
    /// plain column.
    fn judged_harris(
        &self,
        judged: &ColumnJudgment,
        entering_up: bool,
        max_step: f64,
        harris_scale: f64,
        toward_infinite: bool,
    ) -> Option<(usize, f64, f64)> {
        let pivot_tol = primal_pivot_tolerance(self.basis_solver.eta_matrices.len());
        self.primal_harris(entering_up, max_step, harris_scale, |r, coeff| {
            coeff.abs() > judged.errors[r] && (toward_infinite || coeff.abs() > pivot_tol)
        })
    }

    /// Check the reduced cost that priced the entering var at `col`, updated
    /// pivot by pivot, against the one its column `col_coeffs` gives,
    /// `c_q - sum_r c_B(r) alpha_r`, summed in compensated arithmetic. The
    /// column's value replaces it; where that value no longer improves the
    /// objective, or has the other sign, the var does not enter
    /// ([`PivotChoice::Drifted`]).
    fn check_entering_reduced_cost(&mut self, col: usize) -> Result<Option<PivotChoice>, Error> {
        let var = self.nb_vars[col];
        let mut sum = CDouble::from(self.cost_in_force(var));
        for (r, &alpha) in self.col_coeffs.iter() {
            sum.add_product(-self.cost_in_force(self.basic_vars[r]), alpha);
        }
        let computed = sum.value();
        if !computed.is_finite() {
            return Err(Error::InternalError(format!(
                "reduced cost of var {var} computed from its column is not finite: {computed}"
            )));
        }
        let updated = std::mem::replace(&mut self.nb_var_obj_coeffs[col], computed);
        let tol = self.dual_tols[var];
        let same_sign = (updated > 0.0 && computed > 0.0) || (updated < 0.0 && computed < 0.0);
        if same_sign && !self.nb_var_states[col].is_dual_feasible(computed, tol) {
            return Ok(None);
        }
        Ok(Some(PivotChoice::Drifted {
            sign_error: !same_sign && computed.abs() >= tol,
        }))
    }

    /// The two passes of the primal ratio test (Harris's rule) over the
    /// entries of `col_coeffs` that `takes_part` admits, for an entering var
    /// that moves up (`entering_up`) or down by at most `max_step` before it
    /// reaches its other bound: the row that leaves, the bound its basic var
    /// leaves at, and the column's element there; `None` when no admitted
    /// entry bounds the step. `harris_scale` scales the tolerance the test
    /// lets a basic var pass its bound by (see `run_phases`).
    ///
    /// Harris rule. See e.g.
    /// Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989).
    /// A practical anti-cycling procedure for linearly constrained
    /// optimization. Mathematical Programming, 45(1-3), 437-474.
    /// <https://link.springer.com/content/pdf/10.1007/BF01589114.pdf>
    fn primal_harris(
        &self,
        entering_up: bool,
        max_step: f64,
        harris_scale: f64,
        takes_part: impl Fn(usize, f64) -> bool,
    ) -> Option<(usize, f64, f64)> {
        let moves_up = |coeff: f64| moves_up(entering_up, coeff);
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

        // First, the largest step after which no basic var has left its
        // working bound by more than its tolerance (Harris's relaxed bounds),
        // counted from where the var sits, so one already past its bound has
        // only the rest of its tolerance left. The first round of `run_phases`
        // allows the whole tolerance the phase exit verifies with (a var the
        // recomputation then finds an ulp past is a violation for the dual
        // simplex), each later round a hundredth as much. The allowance is
        // not cut by the round-off of recomputing the var: on rows above about
        // 5e6 in magnitude, that would leave none.
        //
        // The limit is never below zero: a var past its bound by more than the
        // allowance (possible after the first round) admits no step toward it.
        // A negative limit would let only the rows of vars past their bounds
        // qualify, passing over a row at its bound with an entry of one for a
        // tiny entry
        // (`a_var_past_its_bound_beyond_the_allowance_limits_the_step_to_zero`).
        let mut max_step = max_step;
        for (r, &coeff) in self.col_coeffs.iter() {
            if !takes_part(r, coeff) {
                continue;
            }
            let tol = harris_scale * self.var_tols[self.basic_vars[r]].toward(moves_up(coeff));
            let cur_step = (signed_room(r, coeff) + tol) / coeff.abs();
            if cur_step < max_step {
                max_step = cur_step;
            }
        }
        let max_step = max_step.max(0.0);

        // Second, among the rows whose own step (to the bound itself; below
        // zero for a var past its bound) is within that limit, the one with
        // the largest entry leaves: a numerically more stable basis at the
        // price of a tolerance-level infeasibility of the others. The row that
        // set the limit always qualifies, so a blocked step always finds a
        // pivot.
        let mut choice: Option<(usize, f64, f64)> = None;
        for (r, &coeff) in self.col_coeffs.iter() {
            if !takes_part(r, coeff) {
                continue;
            }
            let cur_step = signed_room(r, coeff) / coeff.abs();
            if cur_step.is_finite()
                && cur_step <= max_step
                && choice.is_none_or(|(_, _, best)| coeff.abs() > best.abs())
            {
                let bound = if moves_up(coeff) {
                    self.basic_var_maxs[r]
                } else {
                    self.basic_var_mins[r]
                };
                choice = Some((r, bound, coeff));
            }
        }
        choice
    }

    /// The pivot column of `entering_c` refined for the primal ratio test
    /// ([`Self::judge_pivot_col`]), the way the proof refines its row of the
    /// basis inverse ([`Self::refine_basis_inverse_row`]): each step solves
    /// for the residual `a_q - B alpha`, summed in compensated arithmetic over
    /// the basic columns, and adds the solution to the column, kept in doubled
    /// precision. The refined column replaces `col_coeffs`. Two steps, and
    /// more until a correction converges, for up to `REFINEMENT_MAX_STEPS`
    /// steps while each is at most `REFINEMENT_RATIO` of the one before, as
    /// the proof refines a row two steps leave unconverged
    /// ([`Self::converged_basis_inverse_row`]).
    ///
    /// Returns the refined values, each entry's measured error, its last
    /// correction, where the refinement converged ([`refinement_error`]), and
    /// the arithmetic discarded in every step. The ratio test carries that
    /// round-off into the residual floor ([`Self::refined_col_floors`]), below
    /// which no correction measures anything. A number of the refinement
    /// that is not finite is an `InternalError`: a breakdown, which the ratio
    /// test would otherwise take for an unbounded ray.
    fn refine_pivot_col(&mut self, entering_c: usize) -> Result<RefinedColumn, Error> {
        let num_rows = self.num_constraints();
        let var = self.nb_vars[entering_c];
        let mut alpha = self.compensated_col_coeffs();
        let mut second = Vec::new();
        let mut residual_roundoff = vec![0.0; num_rows];
        let mut value_roundoff = vec![0.0; num_rows];
        let mut last = f64::INFINITY;
        for step in 0..proof::REFINEMENT_MAX_STEPS {
            let (correction, discarded) = self
                .pivot_col_correction(var, &alpha, true)
                .ok_or_else(|| refinement_breakdown("primal", var))?;
            for (r, (a, &c)) in alpha.iter_mut().zip(&correction).enumerate() {
                residual_roundoff[r] = proof::add_up(residual_roundoff[r], discarded[r]);
                value_roundoff[r] = proof::add_up(value_roundoff[r], a.add_with_error(c));
            }
            let size = correction.iter().fold(0.0f64, |m, c| m.max(c.abs()));
            second = correction;
            if step >= 1
                && (refinement_error(&alpha, &second).is_some()
                    || size > proof::REFINEMENT_RATIO * last)
            {
                break;
            }
            last = size;
        }
        self.set_refined_col(var, &alpha, "primal")?;
        let (_, final_roundoff) = self.pivot_col_residual(var, &alpha, true);
        for r in 0..num_rows {
            residual_roundoff[r] = proof::add_up(residual_roundoff[r], final_roundoff[r]);
            if !residual_roundoff[r].is_finite() || !value_roundoff[r].is_finite() {
                return Err(Error::InternalError(format!(
                    "primal simplex: the residual round-off of the column of var {var} is not finite"
                )));
            }
        }
        Ok(RefinedColumn {
            errors: refinement_error(&alpha, &second),
            values: alpha,
            residual_roundoff,
            value_roundoff,
        })
    }

    /// The pivot column of `entering_c` refined twice, by the steps of
    /// [`Self::refine_pivot_col`], for a pivot read off a refined row (see
    /// [`Self::dual_phase`]), which reads only the refined values:
    /// what the arithmetic discards is not measured. A number of the
    /// refinement that is not finite is an `InternalError`.
    fn refine_dual_pivot_col(&mut self, entering_c: usize) -> Result<(), Error> {
        let var = self.nb_vars[entering_c];
        let mut alpha = self.compensated_col_coeffs();
        for _ in 0..2 {
            let (correction, _) = self
                .pivot_col_correction(var, &alpha, false)
                .ok_or_else(|| refinement_breakdown("dual", var))?;
            for (a, &c) in alpha.iter_mut().zip(&correction) {
                a.add(c);
            }
        }
        self.set_refined_col(var, &alpha, "dual")
    }

    /// `col_coeffs`, dense and compensated, for a refinement to go into.
    fn compensated_col_coeffs(&self) -> Vec<CDouble> {
        let mut alpha = vec![CDouble::default(); self.num_constraints()];
        for (r, &coeff) in self.col_coeffs.iter() {
            alpha[r] = CDouble::from(coeff);
        }
        alpha
    }

    /// Replace `col_coeffs` by `alpha`, the refined column of `var`, rounded
    /// to `f64`; an entry that is not finite is an `InternalError` of the
    /// `phase` simplex.
    fn set_refined_col(&mut self, var: usize, alpha: &[CDouble], phase: &str) -> Result<(), Error> {
        let mut refined = ScatteredVec::empty(alpha.len());
        for (r, a) in alpha.iter().enumerate() {
            let value = a.value();
            if !value.is_finite() {
                return Err(refinement_breakdown(phase, var));
            }
            if value != 0.0 {
                *refined.get_mut(r) = value;
            }
        }
        refined.to_sparse_vec(&mut self.col_coeffs);
        Ok(())
    }

    /// The correction of one step of a refinement of the column of `var`
    /// ([`Self::refine_pivot_col`]): `B^-1 (a_q - B alpha)`, the residual
    /// summed in compensated arithmetic over the basic columns and scaled for
    /// the solve, exactly, by the power of two that puts its largest entry in
    /// `[1, 2)`; zero where the residual is. With it, where `measure`, the
    /// arithmetic the residual discarded ([`Self::pivot_col_residual`]).
    /// `None` where the residual, or what it discarded, is not finite.
    fn pivot_col_correction(
        &mut self,
        var: usize,
        alpha: &[CDouble],
        measure: bool,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        let (residual, roundoff) = self.pivot_col_residual(var, alpha, measure);
        if roundoff.iter().any(|error| !error.is_finite()) {
            return None;
        }
        let mut correction: Vec<f64> = residual.iter().map(|r| r.value()).collect();
        let norm = correction.iter().fold(0.0f64, |m, c| m.max(c.abs()));
        if !norm.is_finite() {
            return None;
        }
        if norm > 0.0 {
            let scale = 2f64.powi(-(norm.log2().floor() as i32));
            correction.iter_mut().for_each(|c| *c *= scale);
            self.basis_solver.solve_dense_with_etas(&mut correction);
            correction.iter_mut().for_each(|c| *c /= scale);
        }
        Some((correction, roundoff))
    }

    /// The residual `a_q - B alpha` of the column of `var`, summed in
    /// compensated arithmetic over the basic columns, and where `measure`,
    /// per row an upper bound on the arithmetic the sum discarded, each
    /// low-part addition and product and its conversion to `f64` (empty
    /// otherwise). Measured or not, the residual is the same.
    fn pivot_col_residual(
        &self,
        var: usize,
        alpha: &[CDouble],
        measure: bool,
    ) -> (Vec<CDouble>, Vec<f64>) {
        let num_rows = self.num_constraints();
        let mut residual = vec![CDouble::default(); num_rows];
        let mut roundoff = vec![0.0; if measure { num_rows } else { 0 }];
        //guaranteed to be a valid index
        for (i, &a) in self.orig_constraints_csc.outer_view(var).unwrap().iter() {
            if measure {
                roundoff[i] = proof::add_up(roundoff[i], residual[i].add_with_error(a));
            } else {
                residual[i].add(a);
            }
        }
        for (k, &basic) in self.basic_vars.iter().enumerate() {
            if alpha[k].value() == 0.0 {
                continue;
            }
            //guaranteed to be a valid index
            for (i, &a) in self.orig_constraints_csc.outer_view(basic).unwrap().iter() {
                if measure {
                    let error = residual[i].add_scaled_with_error(alpha[k], -a);
                    roundoff[i] = proof::add_up(roundoff[i], error);
                } else {
                    residual[i].add_scaled(alpha[k], -a);
                }
            }
        }
        for (r, error) in roundoff.iter_mut().enumerate() {
            *error = proof::add_up(*error, residual[r].value_error());
        }
        (residual, roundoff)
    }

    /// The round-off floor of the requested entries of the refined column:
    /// the arithmetic discarded while evaluating its residuals, in every
    /// refinement step and after them, carried into each entry through an
    /// approximate row of the basis inverse, plus the round-off of adding the
    /// corrections and of rounding the entry to `f64`. Each low-part addition
    /// and product counts what it discards, so arithmetic that is exact adds
    /// nothing to the floor, however small a genuine entry is.
    ///
    /// The inverse row comes from the floating-point factorization, so this
    /// remains a numerical estimate. An entry inside this floor is unresolved:
    /// it cannot be used to pivot, and leaves a ray only where it is
    /// negligible over the step to the magnitude taken for infinite
    /// ([`Self::unresolved_entries_leave_a_ray`]).
    /// A floor that is not finite is a loud error.
    fn refined_col_floors(
        &mut self,
        var: usize,
        rows: &[usize],
        column: &RefinedColumn,
    ) -> Result<Vec<f64>, Error> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let roundoff = &column.residual_roundoff;
        rows.iter()
            .map(|&row| {
                let floor: f64 = self
                    .basis_solver
                    .solve_transp(std::iter::once((row, &1.0)))
                    .iter()
                    .map(|(i, &rho)| proof::residual_error_product(rho.abs(), roundoff[i]))
                    .fold(
                        proof::add_up(column.value_roundoff[row], column.values[row].value_error()),
                        proof::add_up,
                    );
                if floor.is_finite() {
                    Ok(floor)
                } else {
                    Err(Error::InternalError(format!(
                        "primal simplex: the round-off floor of the column of var {var} in row \
                         {row} is not finite"
                    )))
                }
            })
            .collect()
    }

    /// The nonzero entries (row, value) of the refined column of `var` beside
    /// the `floored` rows whose basic var has a finite bound and whose sign the
    /// refinement leaves open: within their measured error and round-off floor
    /// ([`Self::refined_col_floors`]), which `errors` gains. The primal ratio
    /// test passes such an entry as zero, or as moving its var toward an
    /// infinite bound, so a ray past unresolved entries covers it too.
    fn other_entries_of_open_sign(
        &mut self,
        var: usize,
        column: &RefinedColumn,
        floored: &[usize],
        errors: &mut [f64],
    ) -> Result<Vec<(usize, f64)>, Error> {
        let mut skip = vec![false; self.num_constraints()];
        floored.iter().for_each(|&r| skip[r] = true);
        let (rows, entries): (Vec<usize>, Vec<f64>) = self
            .col_coeffs
            .iter()
            .filter(|&(r, _)| {
                !skip[r]
                    && (self.basic_var_mins[r].is_finite() || self.basic_var_maxs[r].is_finite())
            })
            .map(|(r, &coeff)| (r, coeff))
            .unzip();
        let floors = self.refined_col_floors(var, &rows, column)?;
        let mut open = Vec::new();
        for ((&r, entry), floor) in rows.iter().zip(entries).zip(floors) {
            errors[r] = proof::add_up(errors[r], floor);
            if entry.abs() <= errors[r] {
                open.push((r, entry));
            }
        }
        Ok(open)
    }

    /// Whether the refined column of `entering_c` is a ray within the
    /// tolerances from the point the phase started on, which the verification
    /// accepted ([`Self::primal_start`]; none without it), where its only
    /// entries that could bound the step are the `open` ones (row, value),
    /// whose sign the refinement leaves open, each within its error in `errors`
    /// (of a refinement that converged). Whatever its sign, such an entry is at
    /// most `|value| + error`, and the entering var moves at most `2^52` of its
    /// user units plus its magnitude (at the start or now, the larger) before
    /// it reaches the magnitude taken for infinite ([`INFINITE_BOUND`]): over
    /// that step, the product rounded up, the entry must move its basic var by
    /// no more than the room the start leaves it to the edge of the tolerance
    /// of each finite bound in force ([`band_edge`]; a slack's value and
    /// tolerance its row's at the start), rounded down, so that the var stays
    /// within its tolerances all along.
    fn unresolved_entries_leave_a_ray(
        &self,
        entering_c: usize,
        open: &[(usize, f64)],
        errors: &[f64],
    ) -> bool {
        let Some(start) = &self.primal_start else {
            return false;
        };
        let var = self.nb_vars[entering_c];
        // One user unit of the entering var in the engine's units.
        let unit = if var < self.num_vars {
            1.0 / self.col_scales[var]
        } else {
            self.row_scales[var - self.num_vars]
        };
        let entering_magnitude = self.nb_var_vals[entering_c].abs().max(start[var].abs());
        let horizon = proof::add_up(INFINITE_BOUND * unit, entering_magnitude);
        open.iter().all(|&(r, entry)| {
            let basic = self.basic_vars[r];
            let (val, tols) = if basic < self.num_vars {
                (start[basic], self.var_tols[basic])
            } else {
                // A slack at the start is its row's, summed as the row check
                // sums it, whether it was basic there or not.
                let row = basic - self.num_vars;
                //guaranteed to be a valid index
                let terms = self.orig_constraints.outer_view(row).unwrap();
                let (val, magnitude) = implied_slack(
                    self.orig_rhs[row],
                    terms
                        .iter()
                        .filter(|&(v, _)| v < self.num_vars)
                        .map(|(v, &coeff)| (coeff, start[v])),
                );
                (val, self.slack_tols(row, magnitude))
            };
            let (min, max) = (self.orig_var_mins[basic], self.orig_var_maxs[basic]);
            let mut room = f64::INFINITY;
            if min.is_finite() {
                room = room.min(-proof::add_up(band_edge(min, tols.lo, false), -val));
            }
            if max.is_finite() {
                room = room.min(-proof::add_up(val, -band_edge(max, tols.hi, true)));
            }
            let drift = (proof::add_up(entry.abs(), errors[r]) * horizon).next_up();
            room > 0.0 && drift <= room
        })
    }

    /// The row to leave the basis: the violated row with the best dual
    /// steepest-edge score, not counting a row whose basis change is taboo,
    /// and the bound its basic var leaves at.
    fn choose_pivot_row_dual(&self) -> Option<(usize, f64)> {
        let any_taboo = self.cycle_guard.any_taboo();
        self.best_violated_row(|r| !(any_taboo && self.cycle_guard.is_taboo(r)))
            .map(|r| (r, self.leaving_target(r)))
    }

    /// The violated row with the best dual steepest-edge score among those
    /// that no var can repair and that do not prove infeasibility (see
    /// [`CycleGuard::add_failed_proof`]): the row the last resorts of
    /// [`Self::dual_phase`] take.
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

    /// The dual ratio test for the leaving `row` ([`Self::dual_ratio_pass`]
    /// over `row_coeffs`, as [`Self::calc_row_coeffs`] computed them), with
    /// the growth test of its chosen pivot: at most [`PIVOT_GROWTH_TOL`] of
    /// the row's largest entry, a pivot is small, however far above the pivot
    /// tolerance it lies, and may be made of the round-off that every entry of
    /// the row shares. The first time, the row is refined
    /// ([`Self::refine_pivot_row`]) and the pass runs again on it: a genuine
    /// pivot comes out as it was, a round-off one changes or drops out. After
    /// that, a small pivot is left out of the row, and the pass runs again
    /// without it, until a pivot passes or none is left
    /// ([`DualRatioTest::NoCandidate`]: the row is the proof's to decide, and
    /// a row the proof leaves open gets [`Self::genuine_entry_step`]).
    ///
    /// The row's largest entry is taken once, from the row as first computed,
    /// rounded up to a power of two, over the slacks and the structural vars
    /// whose bounds do not fix them: a var its bounds fix never enters, and
    /// its entry says nothing of the scale of the others (a var fixed on entry
    /// keeps the user's units, which the scalings leave alone, so a fixed
    /// `1e17` can sit beside a genuine `1`).
    fn choose_entering_col_dual(
        &mut self,
        row: usize,
        leaving_new_val: f64,
    ) -> Result<DualRatioTest, Error> {
        let row_max = self
            .row_coeffs
            .iter()
            .filter(|&(c, _)| {
                let var = self.nb_vars[c];
                var >= self.num_vars || self.orig_var_mins[var] != self.orig_var_maxs[var]
            })
            .map(|(_, coeff)| coeff.abs())
            .fold(0.0, f64::max);
        // The power of two that takes the largest entry into `(0.5, 1]`.
        let growth_scale = if row_max > 0.0 && row_max.is_finite() {
            2f64.powi(-(row_max.log2().ceil() as i32))
        } else {
            1.0
        };
        let pivot_tol = dual_pivot_tolerance(self.basis_solver.eta_matrices.len());
        let mut left_out = vec![false; self.nb_vars.len()];
        let mut refined = false;
        loop {
            let mut test = self.dual_ratio_pass(row, leaving_new_val, |c, coeff| {
                coeff.abs() > pivot_tol && !left_out[c]
            })?;
            let (col, a_row) = match &mut test {
                DualRatioTest::Step(step) => {
                    step.refined = refined;
                    step.entering
                }
                DualRatioTest::NoCandidate | DualRatioTest::DualInfeasible => return Ok(test),
            };
            if (a_row * growth_scale).abs() > PIVOT_GROWTH_TOL {
                return Ok(test);
            }
            if refined {
                debug!(
                    "dual ratio test for row {row}: pivot {a_row:e} of var {} is small beside \
                     the row's largest entry {row_max:e} on the refined row; left out",
                    self.nb_vars[col]
                );
                left_out[col] = true;
            } else {
                debug!(
                    "dual ratio test for row {row}: pivot {a_row:e} of var {} is small beside \
                     the row's largest entry {row_max:e}; refining the row",
                    self.nb_vars[col]
                );
                self.refine_pivot_row(row);
                refined = true;
            }
        }
    }

    /// The first last resort of the dual simplex, for an unproven `row` when
    /// only taboo rows are left on fresh data (see [`Self::dual_phase`]): the
    /// dual ratio test over the genuine entries of the refined row
    /// ([`Self::refine_pivot_row`]), whatever their size. A genuine entry
    /// below the pivot tolerance, or small beside its row, can be the only one
    /// that repairs the row, as in a row the model's ratios carry through a
    /// chain of others or two nearly parallel rows leave, and only how far the
    /// refinement moves it tells it from round-off, which no size can. The
    /// proof counts these entries too, so a row it leaves open has one,
    /// unless a non-basic entry within the refinement's error keeps it open,
    /// which neither concedes. Every regular pivot and every proof of the
    /// other rows comes first: the step through a tiny entry is huge and the
    /// basis it leads to ill-conditioned, as when a boxed var whose genuine
    /// but tiny entry cannot carry a row the proof closes enters far past its
    /// other bound, or a basis of nearly parallel rows keeps the next proof's
    /// refinement from converging.
    fn genuine_entry_step(
        &mut self,
        row: usize,
        leaving_new_val: f64,
    ) -> Result<DualRatioTest, Error> {
        let errors = self.refine_pivot_row(row);
        let genuine = |c: usize, coeff: f64| coeff.abs() > errors[c];
        debug!(
            "dual ratio test for row {row}: no pivot above the tolerance and no proof; the {} \
             genuine entries of the refined row take part",
            self.row_coeffs
                .iter()
                .filter(|&(c, &coeff)| genuine(c, coeff))
                .count()
        );
        let mut test = self.dual_ratio_pass(row, leaving_new_val, genuine)?;
        if let DualRatioTest::Step(step) = &mut test {
            step.refined = true;
        }
        Ok(test)
    }

    /// One pass of the dual ratio test for the leaving `row`, with bound
    /// flipping, over the entries of `row_coeffs` that `takes_part` admits
    /// (see [`Self::choose_entering_col_dual`]).
    ///
    /// The leaving var must move to `leaving_new_val`. A non-basic var whose
    /// entry in the row can carry that move does so by entering the basis (a
    /// pivot) or, when both its bounds are finite, by moving to its other
    /// bound (a flip, which leaves the basis as it is). The vars are taken in
    /// the order the dual step brings their reduced costs to zero (their
    /// breakpoints), in Harris groups; every group before the pivot's is
    /// flipped, and the step passes each flipped var's breakpoint, so the var
    /// is dual feasible at the bound it moved to.
    ///
    /// Where the step would run backwards, it is zero and the entering var's
    /// cost is shifted ([`DualStep::zero_step`]); where a candidate's reduced
    /// cost is beyond its tolerance on its infeasible side, no step is dual
    /// feasible ([`DualRatioTest::DualInfeasible`]). When no var can enter,
    /// one that can absorb the move within its tolerance enters on a zero
    /// dual step; when none can, the test returns
    /// [`DualRatioTest::NoCandidate`] and draws no conclusion: infeasibility
    /// is decided on the model's data ([`Self::prove_infeasible`]), not on
    /// this computed row. A var whose entry does not take part neither bounds
    /// the dual step nor pivots, and a reduced cost the step leaves dual
    /// infeasible is corrected at the next rebuild ([`Self::rebuild_dual`]).
    /// A non-finite entry or reduced cost among those weighed is an
    /// `InternalError` ([`Self::finite_reduced_cost`]).
    fn dual_ratio_pass(
        &self,
        row: usize,
        leaving_new_val: f64,
        takes_part: impl Fn(usize, f64) -> bool,
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

        // Whether the non-basic var `c` could instead absorb the leaving var's
        // move by crossing the bound it sits at (or, if fixed, its value) by
        // no more than its own tolerance. This hands a violation that a var
        // cannot carry (an integer var, a tight bound) to a term whose
        // tolerance the infeasibility proof counts, such as an equality's
        // slack, at once rather than through the widening. An integer var
        // never absorbs, nor does a fixed structural var (see
        // `holds_on_bounds`), and a var crosses a bound by absorbing once per
        // solve (`absorbed`), so that absorptions cannot hand a violation back
        // and forth for ever: a later need for that room goes to the proof and
        // the widening. The var moves the way its bound forbids, so the pivot
        // takes a zero dual step with the var's cost shifted, rather than a
        // step `d_q / a_q` that would run backwards by any amount.
        let absorbs = |c: usize, coeff: f64| -> bool {
            if coeff == 0.0 {
                return false;
            }
            let var = self.nb_vars[c];
            let step = leaving_move / coeff;
            let up = step > 0.0;
            // The var is to stay within the tolerance the phase holds it to
            // once basic, so that it is not violated in turn: its own, for a
            // structural var the bound it crosses in the direction the pivot
            // moves it; a slack's stored one, derived at the last
            // recomputation, and no more than its row's at the current point.
            let tol = if var < self.num_vars {
                self.var_tols[var].toward(up)
            } else {
                self.var_tols[var]
                    .toward(up)
                    .min(self.current_slack_tols(var - self.num_vars).toward(up))
            };
            !self.holds_on_bounds(var)
                && step.abs() <= tol
                && !(self.crosses_bound(c, step) && self.absorbed.contains(&(var, up)))
        };

        // An entry that is not a finite number is a breakdown, not data: it
        // compares false with the pivot tolerance below, so its var would drop
        // out of the test silently, and the row could prove infeasibility
        // without it. It is a loud error instead.
        for (c, &coeff) in self.row_coeffs.iter() {
            if !coeff.is_finite() {
                return Err(Error::InternalError(format!(
                    "dual simplex: entry of var {} in row {row} is not finite: {coeff}",
                    self.nb_vars[c]
                )));
            }
        }

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
            // The reduced cost on the var's feasible side, negative for a
            // reduced cost the Harris pass let cross zero.
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
        // The entries that take part (see the doc above).
        let in_row = |c: usize, coeff: f64| coeff != 0.0 && takes_part(c, coeff);
        // The vars that could enter regularly (the two-pass's candidates).
        let candidate =
            |c: usize, coeff: f64| in_row(c, coeff) && direction_ok(coeff, &self.nb_var_states[c]);

        // The Harris bound on the step: the smallest relaxed ratio. Below
        // zero, a candidate's reduced cost is beyond its tolerance on its
        // infeasible side and no step is dual feasible: the reduced costs are
        // to be corrected first. Every candidate passes here, so below this
        // every candidate's reduced cost is finite and every ratio a number.
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
        // price of a slight dual infeasibility. These candidates are also the
        // first group of the bound-flipping walk below, so the same pass
        // measures how far flipping all of them would carry the leaving var.
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

        // Bound flipping. The breakpoints are taken in Harris groups: the
        // first is the candidates the Harris pass chose from, each next one
        // the remaining candidates whose ratio is within the smallest relaxed
        // ratio among them, until the groups so far can carry the leaving var
        // within its tolerance of its bound (or every candidate is grouped).
        // The pivot is the largest entry of the last group, unless that group
        // has none above a tenth of the largest grouped entry (or above one):
        // the scan then steps back to the last group that has one. So among
        // breakpoints the step reaches together, the pivot is the largest
        // entry, not whichever a one-at-a-time walk would stop at. Every group
        // before the pivot's is flipped; ties go to the first entry of the row.
        //
        // A flip never comes with a zero dual step: a group after the first
        // holds only ratios above the first group's bound, which is not
        // negative, so the dual step is positive whenever there is a group to
        // flip. If the leaving var is still short after every group, the
        // pivot's step carries the entering var past its other bound, which a
        // later iteration repairs.
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
                // The breakpoint of the smallest relaxed ratio always joins
                // (its ratio is below its relaxed ratio, and every ratio is a
                // number), so no group is empty; the check is a safeguard.
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
                crosses: None,
                refined: false,
            }));
        }
        if let Some((col, coeff, ratio)) = entering {
            return Ok(DualRatioTest::Step(DualStep {
                flips: Vec::new(),
                entering: (col, coeff),
                zero_step: ratio <= 0.0,
                crosses: None,
                refined: false,
            }));
        }

        // No var can enter regularly. A var that can absorb the move within
        // its tolerance enters instead, on a zero dual step with its cost
        // shifted by its whole reduced cost (see `absorbs` above), given back
        // when it leaves. The ratio is that reduced cost (zero for a fixed
        // var, for which any reduced cost is feasible) over the entry, so that
        // the shift doing the least harm wins: among the absorbers within the
        // Harris relaxation of the smallest ratio, the one with the largest
        // entry enters, held to the growth test as any pivot is. (By the
        // largest entry alone, a solve can stop above the optimum:
        // `wide_lp_558_reaches_the_optimum` in `src/tests/magnitudes.rs`.)
        let absorber_ratio = |c: usize, coeff: f64, tol: f64| -> f64 {
            let state = &self.nb_var_states[c];
            let d = if state.at_min && state.at_max {
                0.0
            } else {
                self.nb_var_obj_coeffs[c].abs()
            };
            (d + tol) / coeff.abs()
        };
        let absorber = |c: usize, coeff: f64| in_row(c, coeff) && absorbs(c, coeff);
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
        if let Some((c, coeff)) = absorbing {
            let step = leaving_move / coeff;
            return Ok(DualRatioTest::Step(DualStep {
                flips: Vec::new(),
                entering: (c, coeff),
                zero_step: true,
                crosses: self.crosses_bound(c, step).then_some(step > 0.0),
                refined: false,
            }));
        }

        // No var can enter or absorb the move. Whether the row proves the
        // model infeasible is the proof's to decide (see the doc above).
        if log::log_enabled!(log::Level::Debug) {
            debug!(
                "no entering column for row {row} (basic var {} = {:e}, bounds [{:e}, {:e}], \
                 leaving to {leaving_new_val:e}); row entries:",
                self.basic_vars[row],
                self.basic_var_vals[row],
                self.basic_var_mins[row],
                self.basic_var_maxs[row],
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
    /// with every bound of the test, so its var would drop out unseen and the
    /// row could become a false proof of infeasibility: it is an
    /// `InternalError` instead, as in the primal pricing.
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
    /// other bound, its perturbed bound when a primal phase starts, or, for a
    /// var the certificate repair moves, the bound it widened), the basis
    /// staying as it is, and carry the moves into the basic values,
    /// `x_B -= B^-1 Σ Δx_c a_c`, through one solve with the combined column.
    /// The objective moves by the reduced costs, as in a pivot.
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

    /// The state of the non-basic `var` at `val` ([`NonBasicVarState::at`]),
    /// against its working bounds in a running primal phase (see [`bounds`]),
    /// where a non-basic var sits at a moved bound.
    fn nonbasic_state(&self, var: usize, val: f64) -> NonBasicVarState {
        let (min, max) = self.working_bounds(var);
        NonBasicVarState::at(val, min, max)
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
        // when it entered: it is non-basic again, and its reduced cost is
        // that of its unshifted cost.
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
    /// [`PivotElem::coeff`]). Cancellation can take an updated weight below
    /// its true value, even to zero or below, where its column would always
    /// or never be chosen, so every updated weight is floored at
    /// `1 + lambda^2` (`lambda` the column's pivot-row entry over the pivot),
    /// a lower bound of the true value. A weight the update leaves not finite
    /// (an overflow or a NaN) is an `InternalError`, not a panic, caught
    /// before the floor could make it finite.
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
    /// on `pivot_coeff` (the column's element, see [`PivotElem::coeff`]): the
    /// pivot row's new weight exactly, from the row just computed, the others
    /// by the Forrest-Goldfarb update. A weight that starts as an
    /// underestimate (every weight is reset to one when a basis is loaded)
    /// can be driven to zero or below, where its row would dominate the choice
    /// or never be chosen, so every updated weight is floored at
    /// [`MIN_DUAL_EDGE_WEIGHT`]. A weight the update leaves not finite (an
    /// overflow or a NaN) is an `InternalError`, not a panic, caught before
    /// the floor could make it finite.
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
            // the floor and hide the breakdown.
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

    /// Recompute the basic values from the non-basic ones: a refinement
    /// ([`Self::refine`]) from zero, so that `b - N x_N` is summed in
    /// compensated arithmetic. A plain sum keeps the round-off of a row's
    /// terms where they cancel (a fixed var's large term against the
    /// right-hand side), which the row's own tolerance covers, but the solve
    /// carries it into the other basic vars, and through them into rows whose
    /// tolerance it can exceed many times over. Etas are applied to the dense
    /// solve directly; a pending eta file does not require a refactorization.
    fn recalc_basic_var_vals(&mut self) -> Result<(), Error> {
        self.basic_var_vals.fill(0.0);
        self.refine();
        // Basic slacks re-derived from their rows and the slack tolerances
        // refreshed; whether the rows hold is the caller's business.
        self.check_rows();
        // Recomputed is not verified: on an ill-conditioned basis a solve
        // through the factorization leaves a residual above the rows'
        // tolerances, which only `verify_primal`'s refinement removes. So
        // the values are dirty until the next phase exit checks them, even
        // right after a refactorization or a basis load.
        self.values_dirty = true;
        self.pivots_since_drift_check = 0;
        Ok(())
    }

    /// Recompute the reduced costs and the objective from the costs the
    /// simplex works with ([`Self::cost_in_force`]).
    fn recalc_obj_coeffs(&mut self) -> Result<(), Error> {
        let multipliers = self.multipliers();
        self.set_reduced_costs(&multipliers);
        Ok(())
    }

    /// [`Self::recalc_obj_coeffs`] on multipliers refined once
    /// ([`Self::refine_multipliers`]): the reduced costs a solve's optimality
    /// is verified on.
    fn recalc_obj_coeffs_refined(&mut self) {
        let mut multipliers = self.multipliers();
        self.refine_multipliers(&mut multipliers);
        self.set_reduced_costs(&multipliers);
    }

    /// The multipliers `y` of the basis, `B^T y = c_B`. As in
    /// recalc_basic_var_vals, pending etas participate in the (transposed)
    /// dense solve instead of forcing a refactorization.
    fn multipliers(&mut self) -> Vec<f64> {
        let mut multipliers: Vec<f64> = self
            .basic_vars
            .iter()
            .map(|&var| self.cost_in_force(var))
            .collect();
        self.basis_solver
            .solve_transp_dense_with_etas(&mut multipliers);
        multipliers
    }

    /// One step of iterative refinement of the multipliers: the residual
    /// `c_B - B^T y`, each entry summed in compensated arithmetic over its
    /// basic column, solved through the factorization and added to `y`. A
    /// solve leaves `y` with the round-off of the basis inverse, which a
    /// reduced cost can carry beyond its tolerance, and the phases would
    /// then alternate over it. Skipped where the residual or the correction
    /// is not finite.
    fn refine_multipliers(&mut self, multipliers: &mut [f64]) {
        let mut residual: Vec<f64> = self
            .basic_vars
            .iter()
            .map(|&var| {
                let mut sum = CDouble::from(self.cost_in_force(var));
                //guaranteed to be a valid index
                for (r, &coeff) in self.orig_constraints_csc.outer_view(var).unwrap().iter() {
                    sum.add_product(-coeff, multipliers[r]);
                }
                sum.value()
            })
            .collect();
        if !residual.iter().all(|r| r.is_finite()) {
            return;
        }
        self.basis_solver
            .solve_transp_dense_with_etas(&mut residual);
        if residual.iter().all(|d| d.is_finite()) {
            for (y, delta) in multipliers.iter_mut().zip(&residual) {
                *y += delta;
            }
        }
    }

    /// The reduced costs `c_j - a_j^T y` of the non-basic vars, their
    /// tolerances and the objective, from the multipliers `y`.
    fn set_reduced_costs(&mut self, multipliers: &[f64]) {
        self.nb_var_obj_coeffs.clear();
        for c in 0..self.nb_vars.len() {
            let var = self.nb_vars[c];
            //guaranteed to be a valid index
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let dot_prod: f64 = col.iter().map(|(r, val)| val * multipliers[r]).sum();
            self.nb_var_obj_coeffs
                .push(self.cost_in_force(var) - dot_prod);
        }
        self.refresh_dual_tols(multipliers);

        self.cur_obj_val = self.exact_obj_val();
        self.reduced_costs_dirty = false;
    }

    /// The exact primal steepest-edge weights `||B^-1 a_j||^2 + 1` of the
    /// non-basic columns: one solve each.
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
