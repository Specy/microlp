use core::time::Duration;

use crate::{
    helpers::{resized_view, to_dense},
    lu::{lu_factorize, LUFactors, ScratchSpace},
    sparse::{ScatteredVec, SparseMat, SparseVec},
    ComparisonOp, CsVec, Error, StopReason, VarDomain,
};
use sprs::CompressedStorage;

use web_time::Instant;

pub(crate) type Deadline = Option<Instant>;

type CsMat = sprs::CsMatI<f64, usize>;

/// The simplex engine's working tolerance: pivot eligibility, ratio-test
/// steps, reduced-cost optimality checks, bound-violation candidacy, and
/// `float_eq`.
///
/// Deliberately tight because the big-M correctness models rely on node LPs
/// resolving basic integer values sharply onto their bounds. Loosening it
/// (globally or just for bound-violation candidacy) lets basic values sit
/// about `1e-8` away from their bounds, which 1e9-scale big-M rows amplify
/// past the MIP layer's
/// rounded-incumbent feasibility guard and the branch-and-bound tree
/// explodes. The flip side of running this tight — round-off noise being
/// promoted into phantom infeasibilities — is handled where it bites, by
/// the refresh valve in [`Solver::restore_feasibility`] and by the terminal
/// rebuild gated by [`REBUILD_RESIDUAL_FACTOR`]. Slack variables are held to
/// their own tolerance — tighter than this on heavily scaled rows, looser
/// where the row's round-off demands it — see [`contract_tol`] and
/// [`Solver::var_tol`]; and each non-basic column's reduced cost is priced
/// with its own round-off floor, never below this — see [`Solver::col_tol`].
/// Structural variables keep this flat value, which is
/// below one ulp for values past ~1e6: a known limitation (such models are
/// best posed with rescaled variables).
pub const EPS: f64 = 1e-10;

/// Default `Tolerances::feasibility`, shared with the MIP layer: the absolute
/// row and bound tolerance, in the user's units, that solutions are validated
/// against and that the engine itself works to (see [`contract_tol`]).
pub(crate) const DEFAULT_FEASIBILITY_TOL: f64 = 1e-7;

/// A slack variable's *contract* tolerance in the engine's row-equilibrated
/// units: the user's absolute row tolerance expressed in the row's own units,
/// never looser than `EPS`.
///
/// Rows are scaled by a power of two so their largest coefficient is O(1),
/// and basic values are compared to their bounds with an absolute tolerance —
/// so a flat `EPS` on a row's slack tolerates `EPS / row_scale` in the user's
/// units: `8e-7` on a row whose largest coefficient is `1e4`, `5e-5` at `1e6`,
/// `5e-2` at `1e9`. The public contract (`Tolerances::feasibility`, absolute
/// in user units) promises much less, and the MIP layer's guard enforces it,
/// so the engine would end a phase on a point the guard then rejects — the
/// second model of issue #44: a row `-2e-4·k·x − 1e3·k·y = 0` pinning `x` to
/// zero, where the engine left `x = -1e-4` (off the row by `2e-8·k`) for
/// every `k`, an `InternalError` for `k ≥ 10` and a silently wrong LP answer
/// otherwise. Each slack is therefore held to `feasibility × row_scale`,
/// capped at `EPS` so that well-scaled rows stay at the engine's resolution
/// instead of loosening to the user's tolerance. What arithmetic can deliver
/// is a separate,
/// point-dependent floor — see [`Solver::var_tol`] — never a constant: a
/// basic value carries round-off proportional to its row's activity, and a
/// constant floor (`1e-12` was tried) is below the noise on rows with
/// activities around `1e5`, where pricing then chases a phantom violation no
/// refresh can remove, or declares a feasible model infeasible.
fn contract_tol(feasibility: f64, row_scale: f64) -> f64 {
    (feasibility * row_scale).min(EPS)
}

/// The tolerance a constraint row is *validated* to: the absolute tolerance
/// `tol`, in whatever units the row is expressed in, floored at the row's
/// round-off — [`REBUILD_NOISE_FLOOR`] times the magnitude of its activity,
/// `1 + |b| + Σ|aᵢxᵢ|`. Below that floor a violation cannot be told from
/// round-off in double precision, so no point could pass. The engine holds
/// each row to the same floor ([`Solver::var_tol`]) and the MIP layer's guard
/// and warm-start pre-filter check with this, so what the engine accepts and
/// what validation accepts agree wherever double precision allows. The floor
/// is a few hundred ulps of the activity, far below any violation the
/// absolute guard exists to catch (a rounded big-M binary moves its row by
/// the whole M).
pub(crate) fn row_tolerance(tol: f64, rhs: f64, magnitude: f64) -> f64 {
    tol.max(REBUILD_NOISE_FLOOR * (1.0 + rhs.abs() + magnitude))
}

/// A simplex phase may end only when every row's residual `|a·x − b|` is
/// within this multiple of the row's own tolerance ([`Solver::var_tol`]).
///
/// Basic values are updated incrementally pivot after pivot, and a pivot on a
/// small element (the ratio tests accept anything above `EPS`) multiplies
/// round-off by its reciprocal: a `1e-8` pivot turns machine epsilon into a
/// `1e-8` error in a basic value. The phase loops only ever compare basic
/// values against their *bounds*, so such a point can pass as feasible while
/// no longer satisfying the rows it was derived from (issue #44: a perfectly
/// conditioned final basis whose exact solution is `0`, reported as `2^-26`).
/// Before a phase is allowed to end, every row's residual is measured against
/// the tolerance that row is held to; anything beyond this factor discards the
/// incremental values, refactorizes, recomputes them from the original data,
/// and re-examines. Every row's tolerance is at least its own round-off floor
/// ([`REBUILD_NOISE_FLOOR`]), so this fires only on drift a rebuild can
/// remove, and it is bounded per phase ([`MAX_TERMINAL_RESTARTS`]) so it
/// cannot loop. The multiple is one — the tolerance itself: a residual within
/// it is what the bound checks and the validation guard already allow, and
/// anything beyond it is drift. (A first cut used ten, which left a window in
/// which the engine could end on a residual the MIP guard then rejected.)
pub(crate) const REBUILD_RESIDUAL_FACTOR: f64 = 1.0;

/// Relative drift of the incrementally-updated objective above which a
/// mid-phase refresh replaces it with the exact value; see `refresh_values`.
const OBJECTIVE_DRIFT_TOL: f64 = 1e-9;

/// Number of incremental reduced-cost updates after which a refactorization
/// also recomputes the reduced costs from the original data (a phase end
/// always does). Their drift grows with the length of the incremental chain,
/// not with the eta file that triggers refactorizations — on small problems
/// that is every few pivots, where an extra transposed solve each time buys
/// nothing measurable — so the two are decoupled. Values are recomputed at
/// every refactorization regardless: the terminal residual gate relies on it.
const REDUCED_COST_UPDATE_LIMIT: u64 = 50;

/// Round-off floor of a row's value, per unit of the row's activity magnitude
/// `1 + |b| + Σ|aᵢxᵢ|`: a few hundred ulps, which is what an LU solve with
/// threshold pivoting can leave behind. It is the floor of every slack's
/// tolerance — a basic value cannot be held closer to its bound than its own
/// round-off, see [`Solver::var_tol`] — and, for the same reason, the level
/// below which a row residual is not drift and a rebuild would not reduce it.
/// Re-derived from the current values whenever they are rebuilt, so it
/// follows the point rather than a static bound.
pub(crate) const REBUILD_NOISE_FLOOR: f64 = 1e-13;

/// Upper bound on the times one phase may re-examine its terminal stall after
/// recomputing values or reduced costs. Exact recomputation can expose
/// noise-level infeasibilities that a degenerate pivot then "fixes", after
/// which the incremental update hides them again — a cycle with no objective
/// progress to break it. Beyond the bound the phase ends on the point it has,
/// and the flags are measured honestly.
const MAX_TERMINAL_RESTARTS: usize = 4;

/// Upper bound on dual/primal phase alternations in [`Solver::run_phases`].
/// Each phase ends by measuring the flag the other phase owns, so one round is
/// the norm and two the exception. The Harris ratio tests relax *in the
/// step* (by the row's tolerance on the primal side, `EPS` on the dual), so a
/// basic value whose column coefficient dwarfs the binding
/// one can end a phase violated by a multiple of `EPS`, and the other phase
/// then pays for the repair with fresh `EPS`-level infeasibility on its side:
/// on rows with a large coefficient spread the two can trade tolerance-level
/// violations indefinitely. After this many rounds the point is accepted as
/// it stands and a `warn!` records it — those violations are the engine's own
/// tolerance, not a wrong answer, and an error here would refuse problems the
/// previous, unmeasured alternation answered.
const MAX_PHASE_ROUNDS: usize = 8;

/// When [`Solver::run_phases`] exhausts [`MAX_PHASE_ROUNDS`], the point is
/// accepted only if its worst remaining violation, primal or dual, is within
/// this multiple of the variable's own tolerance. The Harris ratio tests bound
/// what a phase can leave on the other side by one tolerance per variable
/// (plus round-off), so anything beyond a small multiple is not the phases
/// trading tolerance-level violations but a failure to converge, and it is
/// reported as such rather than certified as optimal.
const PHASE_ACCEPT_FACTOR: f64 = 10.0;

/// How often (in simplex iterations) the primal/dual loops in `optimize` and
/// `restore_feasibility` check the deadline and emit a progress `debug!` log.
/// Checking every iteration would make the deadline check itself a
/// significant fraction of the per-iteration cost on easy problems; checking
/// too rarely would make a time limit overshoot by a visible amount on hard
/// ones. 1000 keeps the check overhead negligible while still bounding the
/// worst-case overshoot to about a thousand pivots.
pub(crate) const DEADLINE_CHECK_INTERVAL: u64 = 1000;

/// Threshold-pivoting stability coefficient passed to [`lu_factorize`] for
/// every LU (re)factorization the simplex performs: a candidate pivot is
/// accepted only if its magnitude is at least this fraction of the column's
/// largest eligible entry. 0.1 is the standard textbook default for
/// Gilbert-Peierls sparse LU (see `lu_factorize`'s doc reference) — it
/// balances numerical stability (higher would refuse more marginal pivots,
/// at the cost of extra fill-in) against sparsity (lower risks amplifying
/// rounding error through a poorly-conditioned pivot).
pub(crate) const LU_STABILITY_THRESHOLD: f64 = 0.1;

/// A variable bound whose magnitude is at least this large is treated as
/// infinite when choosing a non-basic variable's INITIAL value (see
/// [`initial_nonbasic_value`]). Seeding a non-basic variable *at* such a bound
/// floods the tableau with a value that swamps the actual problem data — the
/// rhs and structural coefficients lose all significance against it and the
/// solve converges to a wrong vertex or NaN. This is issue #3: `f64::MAX`,
/// `f32::MAX` and `i64::MAX` upper bounds produced non-optimal answers where
/// `f64::INFINITY` did not, because only the latter skipped the seed-at-bound
/// step. 2^52 is the largest f64 whose unit (`1.0`) is still exactly
/// representable; beyond it a finite bound is numerically a stand-in for
/// infinity, so we seed as if it were infinite. The true bound is left
/// untouched in `orig_var_mins`/`orig_var_maxs`, so the ratio test still
/// honours it exactly — only the starting vertex changes.
const SEED_AS_INFINITE: f64 = 4_503_599_627_370_496.0; // 2^52

/// Power-of-two row equilibration keeps the largest structural coefficient
/// near one without rounding its mantissa. If scaling would overflow the RHS,
/// leave the row unchanged and let the solver report any resulting numerical
/// failure explicitly.
fn equilibration_scale(coeffs: &CsVec, rhs: f64) -> f64 {
    let max_coeff = coeffs
        .data()
        .iter()
        .map(|coeff| coeff.abs())
        .fold(0.0, f64::max);
    if max_coeff == 0.0 || !max_coeff.is_finite() {
        return 1.0;
    }

    let exponent = (max_coeff.log2().floor() as i32).clamp(-1023, 1023);
    let scale = 2.0_f64.powi(-exponent);
    if scale.is_finite() && (rhs * scale).is_finite() {
        scale
    } else {
        1.0
    }
}

/// A non-empty constraint row in the exact representation consumed by the
/// simplex engine. Structural coefficients and the right-hand side share the
/// same power-of-two scale; the slack coefficient remains one.
struct PreparedRow {
    coeffs: CsVec,
    rhs: f64,
    row_scale: f64,
    slack_var_min: f64,
    slack_var_max: f64,
}

/// Validate empty-row semantics and prepare one retained row for storage.
/// `None` denotes a tautology that does not need a slack variable.
fn prepare_row(
    mut coeffs: CsVec,
    cmp_op: ComparisonOp,
    rhs: f64,
) -> Result<Option<PreparedRow>, Error> {
    if coeffs.indices().is_empty() {
        let tautological = match cmp_op {
            ComparisonOp::Eq => float_eq(rhs, 0.0),
            ComparisonOp::Le => 0.0 <= rhs,
            ComparisonOp::Ge => 0.0 >= rhs,
        };
        return if tautological {
            Ok(None)
        } else {
            Err(Error::Infeasible)
        };
    }

    let row_scale = equilibration_scale(&coeffs, rhs);
    if row_scale != 1.0 {
        coeffs.map_inplace(|coeff| coeff * row_scale);
    }
    let (slack_var_min, slack_var_max) = match cmp_op {
        ComparisonOp::Le => (0.0, f64::INFINITY),
        ComparisonOp::Ge => (f64::NEG_INFINITY, 0.0),
        ComparisonOp::Eq => (0.0, 0.0),
    };
    Ok(Some(PreparedRow {
        coeffs,
        rhs: rhs * row_scale,
        row_scale,
        slack_var_min,
        slack_var_max,
    }))
}

/// Whether a non-basic column's reduced cost is dual feasible for its current
/// position, within the tolerance `tol` the column is priced with
/// ([`Solver::col_tol`]): at its lower bound it must not pay to increase, at
/// its upper it must not pay to decrease, and a free column (at neither
/// bound) is feasible only when its reduced cost is zero within `tol`.
/// Without that last clause a free zero-cost column counts as infeasible
/// forever, and the primal phase it triggers picks it as entering with an
/// infinite step.
fn nb_dual_feasible(state: &NonBasicVarState, obj_coeff: f64, tol: f64) -> bool {
    state.at_min && obj_coeff > -tol || state.at_max && obj_coeff < tol || obj_coeff.abs() < tol
}

pub(crate) fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

/// Initial value for a non-basic structural variable, preferring the bound that
/// keeps its reduced cost dual-feasible. Returns `(value, dual_feasible)`, where
/// `dual_feasible` is false when no finite bound can satisfy dual feasibility (a
/// free variable, or a variable unbounded on the side its objective coefficient
/// pushes toward). Callers handle a fixed variable (`min == max`) separately.
///
/// A bound at or beyond [`SEED_AS_INFINITE`] is treated as infinite here so that
/// a huge finite bound is never used as the seed value (issue #3); the caller's
/// stored bounds are left untouched, so the ratio test still honours them.
fn initial_nonbasic_value(obj_coeff: f64, min: f64, max: f64) -> (f64, bool) {
    let min = if min <= -SEED_AS_INFINITE {
        f64::NEG_INFINITY
    } else {
        min
    };
    let max = if max >= SEED_AS_INFINITE {
        f64::INFINITY
    } else {
        max
    };

    if min.is_infinite() && max.is_infinite() {
        // Free variable: dual-feasible only if the objective coefficient is zero.
        (0.0, float_eq(obj_coeff, 0.0))
    } else if obj_coeff > 0.0 {
        // Prefer the lower bound; fall back to the upper if the lower is infinite.
        if min.is_finite() {
            (min, true)
        } else {
            (max, false)
        }
    } else if obj_coeff < 0.0 {
        // Prefer the upper bound; fall back to the lower if the upper is infinite.
        if max.is_finite() {
            (max, true)
        } else {
            (min, false)
        }
    } else if min.is_finite() {
        // Zero objective coefficient: any finite bound is dual-feasible.
        (min, true)
    } else {
        (max, true)
    }
}

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

    orig_obj_coeffs: Vec<f64>,
    orig_var_mins: Vec<f64>,
    orig_var_maxs: Vec<f64>,
    /// Per total variable, its *contract* tolerance: `EPS` for structural
    /// variables, [`contract_tol`] for each row's slack. The tolerance actually
    /// applied is [`Self::var_tol`].
    orig_var_tols: Vec<f64>,
    pub(crate) orig_var_domains: Vec<VarDomain>,
    orig_constraints: CsMat, // excluding rhs
    orig_constraints_csc: CsMat,
    orig_rhs: Vec<f64>,
    /// Positive per-row equilibration factors. Every row is multiplied by
    /// these internally; validation multiplies its absolute user tolerance by
    /// the same factor so the public feasibility contract stays unscaled.
    row_scales: Vec<f64>,
    /// Per row, in the engine's units: [`REBUILD_NOISE_FLOOR`] times the row's
    /// activity magnitude at the last rebuild — the round-off floor of its
    /// slack's tolerance. See [`Self::var_tol`] and [`Self::measure_rows`].
    row_noise: Vec<f64>,
    /// Per row, `|yᵢ|` of the multipliers `y = Bᵀ⁻¹ c_B` at the last exact
    /// reduced-cost recomputation: the magnitudes a column's reduced cost is
    /// computed from, hence its round-off floor. See [`Self::col_tol`].
    multiplier_abs: Vec<f64>,
    /// The absolute row tolerance, in user units, the slack tolerances are
    /// derived from; see [`Self::set_feasibility_tolerance`].
    feasibility_tol: f64,

    enable_primal_steepest_edge: bool,
    enable_dual_steepest_edge: bool,

    is_primal_feasible: bool,
    is_dual_feasible: bool,
    /// True while `nb_var_obj_coeffs`/`cur_obj_val` carry the phase-1
    /// artificial objective installed at construction (neither primal nor dual
    /// feasible to start). Cleared by `recalc_obj_coeffs`, which installs the
    /// real reduced costs. A rebuild must leave the reduced costs alone while
    /// this is set: the artificial objective has no original coefficients to
    /// recompute from.
    artificial_obj: bool,

    // Updated on each pivot
    /// For each var: whether it is basic/non-basic and the corresponding index.
    var_states: Vec<VarState>,
    basis_solver: BasisSolver,

    /// For each constraint the corresponding basic var.
    basic_vars: Vec<usize>,
    basic_var_vals: Vec<f64>,
    basic_var_mins: Vec<f64>,
    basic_var_maxs: Vec<f64>,
    /// [`Self::var_tol`] of the basic variable in each position.
    basic_var_tols: Vec<f64>,
    dual_edge_sq_norms: Vec<f64>,

    /// Remaining variables. (idx -> var), 'nb' means 'non-basic'
    nb_vars: Vec<usize>,
    nb_var_obj_coeffs: Vec<f64>,
    /// [`Self::col_tol`] of the non-basic variable in each position: the
    /// tolerance its reduced cost is priced with.
    nb_var_tols: Vec<f64>,
    /// Incremental reduced-cost updates since the last exact recomputation;
    /// see [`REDUCED_COST_UPDATE_LIMIT`].
    reduced_cost_updates: u64,
    nb_var_vals: Vec<f64>,
    nb_var_states: Vec<NonBasicVarState>,
    nb_var_is_fixed: Vec<bool>,
    primal_edge_sq_norms: Vec<f64>,

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
    ) -> Result<Self, Error> {
        let enable_steepest_edge = true; // TODO: make user-settable.

        let num_vars = obj_coeffs.len();

        assert_eq!(num_vars, var_mins.len());
        assert_eq!(num_vars, var_maxs.len());
        let mut orig_var_mins = var_mins.to_vec();
        let mut orig_var_maxs = var_maxs.to_vec();

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
            if min.is_nan() || max.is_nan() || min > max {
                return Err(Error::Infeasible);
            }

            // initially all user-created variables are non-basic
            var_states.push(VarState::NonBasic(nb_vars.len()));
            nb_vars.push(v);

            // Choose an initial value, preferring a bound that keeps this
            // variable's reduced cost dual-feasible.
            let (init_val, var_dual_feasible) = if float_eq(min, max) {
                // Fixed variable: the obj. coeff doesn't matter.
                (min, true)
            } else {
                initial_nonbasic_value(obj_coeffs[v], min, max)
            };
            if !var_dual_feasible {
                is_dual_feasible = false;
            }

            nb_var_vals.push(init_val);
            obj_val += init_val * obj_coeffs[v];

            nb_var_states.push(NonBasicVarState {
                at_min: float_eq(init_val, min),
                at_max: float_eq(init_val, max),
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
        let mut basic_var_tols = vec![];
        let mut orig_var_tols = vec![EPS; num_vars];

        for (coeffs, cmp_op, rhs) in constraints {
            let Some(PreparedRow {
                coeffs,
                rhs,
                row_scale,
                slack_var_min,
                slack_var_max,
            }) = prepare_row(coeffs.clone(), *cmp_op, *rhs)?
            else {
                continue;
            };

            constraint_coeffs.push(coeffs.clone());
            orig_rhs.push(rhs);
            row_scales.push(row_scale);

            orig_var_mins.push(slack_var_min);
            orig_var_maxs.push(slack_var_max);

            basic_var_mins.push(slack_var_min);
            basic_var_maxs.push(slack_var_max);

            let tol = contract_tol(DEFAULT_FEASIBILITY_TOL, row_scale);
            orig_var_tols.push(tol);
            basic_var_tols.push(tol);

            let cur_slack_var = var_states.len();
            var_states.push(VarState::Basic(basic_vars.len()));
            basic_vars.push(cur_slack_var);

            let mut lhs_val = 0.0;
            for (var, &coeff) in coeffs.iter() {
                lhs_val += coeff * nb_var_vals[var];
            }
            basic_var_vals.push(rhs - lhs_val);
        }

        let num_constraints = constraint_coeffs.len();
        let num_total_vars = num_vars + num_constraints;

        let mut orig_obj_coeffs = obj_coeffs.to_vec();
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

        let mut nb_var_obj_coeffs = vec![];
        let mut primal_edge_sq_norms = vec![];
        for (&var, state) in nb_vars.iter().zip(&nb_var_states) {
            //guaranteed to be a valid index
            let col = orig_constraints_csc.outer_view(var).unwrap();

            if need_artificial_obj {
                let coeff = if state.at_min && !state.at_max {
                    1.0
                } else if state.at_max && !state.at_min {
                    -1.0
                } else {
                    0.0
                };
                nb_var_obj_coeffs.push(coeff);
            } else {
                nb_var_obj_coeffs.push(orig_obj_coeffs[var]);
            }

            if enable_primal_steepest_edge {
                primal_edge_sq_norms.push(col.squared_l2_norm() + 1.0);
            }
        }

        let cur_obj_val = if need_artificial_obj { 0.0 } else { obj_val };

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

        let nb_var_is_fixed = vec![false; nb_vars.len()];
        // The slack basis has zero multipliers, so a reduced cost is its own
        // objective coefficient and carries that magnitude's round-off.
        let nb_var_tols: Vec<f64> = nb_vars
            .iter()
            .map(|&var| EPS.max(REBUILD_NOISE_FLOOR * orig_obj_coeffs[var].abs()))
            .collect();

        let mut res = Self {
            num_vars,
            orig_obj_coeffs,
            orig_var_mins,
            orig_var_maxs,
            orig_var_tols,
            orig_constraints,
            orig_constraints_csc,
            orig_rhs,
            row_noise: vec![0.0; row_scales.len()],
            multiplier_abs: vec![0.0; row_scales.len()],
            row_scales,
            feasibility_tol: DEFAULT_FEASIBILITY_TOL,
            deadline,
            operation_time_limit: None,
            lp_iterations: 0,
            elapsed: Duration::ZERO,
            orig_var_domains: var_domains.to_vec(),
            enable_primal_steepest_edge,
            enable_dual_steepest_edge,
            is_primal_feasible,
            is_dual_feasible,
            artificial_obj: need_artificial_obj,
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
            basic_var_tols,
            dual_edge_sq_norms,
            nb_vars,
            nb_var_obj_coeffs,
            nb_var_tols,
            reduced_cost_updates: 0,
            nb_var_vals,
            nb_var_states,
            nb_var_is_fixed,
            primal_edge_sq_norms,
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
        res.measure_rows();

        Ok(res)
    }

    /// Set the absolute row tolerance, in the user's units, that the engine
    /// holds every constraint row to (`Tolerances::feasibility`). Each slack's
    /// tolerance in the engine's own units follows from it and the row's
    /// equilibration factor — see [`contract_tol`] and [`Self::var_tol`];
    /// structural variables keep `EPS`.
    pub(crate) fn set_feasibility_tolerance(&mut self, feasibility: f64) {
        self.feasibility_tol = feasibility;
        for (r, &scale) in self.row_scales.iter().enumerate() {
            self.orig_var_tols[self.num_vars + r] = contract_tol(feasibility, scale);
        }
        self.measure_rows();
    }

    /// The tolerance a variable's value is held to its bounds with: `EPS` for
    /// a structural variable; for a row's slack, the contract tolerance
    /// ([`contract_tol`]) or the row's round-off floor, whichever is larger.
    /// The floor follows the current point (`row_noise`, re-derived at every
    /// rebuild): a basic value carries round-off proportional to its row's
    /// activity, and demanding less than that makes pricing chase a phantom
    /// violation that no refresh can remove.
    fn var_tol(&self, var: usize) -> f64 {
        if var < self.num_vars {
            EPS
        } else {
            self.orig_var_tols[var].max(self.row_noise[var - self.num_vars])
        }
    }

    /// The tolerance a non-basic column's reduced cost is priced with: `EPS`,
    /// or the reduced cost's own round-off floor when that is larger —
    /// [`REBUILD_NOISE_FLOOR`] times the magnitude of the terms it is computed
    /// from, `|c_j| + Σᵢ |aᵢⱼ yᵢ|`, with `|y|` from the last exact
    /// recomputation. A reduced cost, like a basic value, is only as accurate
    /// as the magnitudes behind it: on a column with large coefficients an
    /// exact recomputation can show a `1e-10` infeasibility that is pure
    /// round-off, and pricing it forces a degenerate pivot that moves the
    /// vertex for nothing (measured on miplib/gt2 as a re-routed search).
    fn col_tol(&self, var: usize) -> f64 {
        let mut magnitude = self.orig_obj_coeffs[var].abs();
        //guaranteed to be a valid index
        for (r, &coeff) in self.orig_constraints_csc.outer_view(var).unwrap().iter() {
            magnitude += (coeff * self.multiplier_abs[r]).abs();
        }
        EPS.max(REBUILD_NOISE_FLOOR * magnitude)
    }

    /// Re-derive every row's round-off floor from the current values (and the
    /// tolerances that depend on it), and return the largest row residual
    /// `|a·x − b|` as a multiple of that row's tolerance — the terminal
    /// rebuild gate, see [`REBUILD_RESIDUAL_FACTOR`]. A basic solution
    /// satisfies every row exactly in exact arithmetic, so a residual beyond
    /// the tolerance is drift of the incremental pivot updates. Non-finite
    /// values report `INFINITY`. O(nnz).
    fn measure_rows(&mut self) -> f64 {
        let mut worst: f64 = 0.0;
        for (r, row) in self.orig_constraints.outer_iterator().enumerate() {
            let mut activity = 0.0;
            let mut magnitude = 0.0;
            for (var, &coeff) in row.iter() {
                let term = coeff * *self.get_value(var);
                activity += term;
                magnitude += term.abs();
            }
            let rhs = self.orig_rhs[r];
            let residual = (activity - rhs).abs();
            if !residual.is_finite() {
                worst = f64::INFINITY;
                continue;
            }
            self.row_noise[r] = REBUILD_NOISE_FLOOR * (1.0 + rhs.abs() + magnitude);
            worst = worst.max(residual / self.var_tol(self.num_vars + r));
        }
        for i in 0..self.basic_vars.len() {
            self.basic_var_tols[i] = self.var_tol(self.basic_vars[i]);
        }
        worst
    }

    /// A phase that has used up [`MAX_TERMINAL_RESTARTS`] ends on the point
    /// it has. Say so loudly when that point still fails the terminal residual
    /// gate: the drift a rebuild could not clear then reaches the caller.
    fn warn_if_terminal_residual_remains(&mut self, phase: &str, iter: u64) {
        let excess = self.measure_rows();
        if excess > REBUILD_RESIDUAL_FACTOR {
            log::warn!(
                "{} iter {}: ending the phase after {} terminal re-examinations with a row \
                 residual still {:.1}x its tolerance",
                phase,
                iter,
                MAX_TERMINAL_RESTARTS,
                excess,
            );
        }
    }

    pub(crate) fn get_value(&self, var: usize) -> &f64 {
        match self.var_states[var] {
            VarState::Basic(idx) => &self.basic_var_vals[idx],
            VarState::NonBasic(idx) => &self.nb_var_vals[idx],
        }
    }

    /// Check `values` (one entry per structural var) against every ORIGINAL
    /// constraint row, within the ABSOLUTE tolerance `tol` floored at the
    /// row's round-off ([`row_tolerance`]). Bounds are not checked here. Each
    /// row's sense is encoded by its slack var's bounds
    /// (lhs + s = rhs with s in [smin, smax]  ⇔  rhs - smax ≤ lhs ≤ rhs - smin);
    /// slack bounds are never touched by branching, so this always reflects the
    /// user's original rows.
    ///
    /// Rows carry an internal power-of-two equilibration factor; the
    /// tolerance is multiplied by that same factor, which is algebraically
    /// equivalent to applying `tol` to the unscaled user row. It is deliberately
    /// NOT scaled by the row's coefficient magnitude: this check exists for
    /// the big-M trap, where a violation that is tiny RELATIVE to huge row
    /// coefficients (e.g. 5.0 on a 1e9-scale row) is decisive in absolute
    /// terms, and any coefficient-relative tolerance would be blind to exactly
    /// the violations this guard is for. The only relaxation is the round-off
    /// floor of the activity itself, below which double precision cannot tell
    /// a violation from a feasible point at all.
    pub(crate) fn check_constraints(&self, values: &[f64], tol: f64) -> bool {
        for (r, row) in self.orig_constraints.outer_iterator().enumerate() {
            let rhs = self.orig_rhs[r];
            let mut lhs = 0.0;
            let mut magnitude = 0.0;
            for (v, &coeff) in row.iter() {
                if v < self.num_vars {
                    let term = coeff * values[v];
                    lhs += term;
                    magnitude += term.abs();
                }
            }
            if !lhs.is_finite() {
                return false;
            }
            let slack = self.num_vars + r;
            let (smin, smax) = (self.orig_var_mins[slack], self.orig_var_maxs[slack]);
            let lo = if smax.is_finite() {
                rhs - smax
            } else {
                f64::NEG_INFINITY
            };
            let hi = if smin.is_finite() {
                rhs - smin
            } else {
                f64::INFINITY
            };
            let scaled_tol = row_tolerance(tol * self.row_scales[r], rhs, magnitude);
            if lhs < lo - scaled_tol || lhs > hi + scaled_tol {
                return false;
            }
        }
        true
    }

    /// Objective value (internal minimize space) of an explicit structural-var
    /// value vector.
    pub(crate) fn objective_of(&self, values: &[f64]) -> f64 {
        values
            .iter()
            .enumerate()
            .map(|(v, &x)| self.orig_obj_coeffs[v] * x)
            .sum()
    }

    pub(crate) fn get_var_bounds(&self, var: usize) -> (f64, f64) {
        (self.orig_var_mins[var], self.orig_var_maxs[var])
    }

    /// Change a variable's bounds in place. Records the new bounds and repairs the
    /// invariants that depend on them; does NOT run simplex — call [`Self::reoptimize`]
    /// afterwards. Returns `Err(Infeasible)` with state untouched if either bound
    /// is NaN or `min > max`.
    pub(crate) fn set_var_bounds(&mut self, var: usize, min: f64, max: f64) -> Result<(), Error> {
        if min.is_nan() || max.is_nan() || min > max {
            return Err(Error::Infeasible);
        }
        self.orig_var_mins[var] = min;
        self.orig_var_maxs[var] = max;
        match self.var_states[var] {
            VarState::Basic(row) => {
                self.basic_var_mins[row] = min;
                self.basic_var_maxs[row] = max;
                let val = self.basic_var_vals[row];
                let tol = self.basic_var_tols[row];
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
                }
                self.nb_var_states[col] = NonBasicVarState {
                    at_min: float_eq(new_val, min),
                    at_max: float_eq(new_val, max),
                };
                // A var at a loosened bound may no longer justify its reduced cost.
                self.is_dual_feasible = self.is_dual_feasible
                    && nb_dual_feasible(
                        &self.nb_var_states[col],
                        self.nb_var_obj_coeffs[col],
                        self.nb_var_tols[col],
                    );
            }
        }
        Ok(())
    }

    /// Re-solve after bound changes, an edit, or a basis load: the two phases
    /// alternate (dual simplex while primal feasibility is broken, primal
    /// simplex while reduced costs are dual-infeasible) until both flags hold
    /// as measured at the end of a phase; see [`Self::run_phases`].
    pub(crate) fn reoptimize(&mut self) -> Result<StopReason, Error> {
        self.run_phases()
    }

    /// Alternate the two simplex phases until both feasibility flags hold as
    /// *measured* at the end of a phase: each phase re-measures the flag the
    /// other one owns when it ends, because the Harris ratio tests trade
    /// bounded infeasibility on that side for pivot size. Bounded by
    /// [`MAX_PHASE_ROUNDS`], after which the point is accepted as it stands.
    /// Returns `Limit` as soon as the deadline fires, leaving the honest flags
    /// for a later call to continue from.
    fn run_phases(&mut self) -> Result<StopReason, Error> {
        for _ in 0..MAX_PHASE_ROUNDS {
            if !self.is_primal_feasible && self.restore_feasibility()? == StopReason::Limit {
                return Ok(StopReason::Limit);
            }
            if !self.is_dual_feasible {
                self.recalc_obj_coeffs()?;
                if self.optimize()? == StopReason::Limit {
                    return Ok(StopReason::Limit);
                }
            }
            if self.is_primal_feasible && self.is_dual_feasible {
                return Ok(StopReason::Finished);
            }
        }
        self.accept_exhausted_point()?;
        Ok(StopReason::Finished)
    }

    /// The exit of [`Self::run_phases`] after [`MAX_PHASE_ROUNDS`]: measure
    /// what is left instead of assuming it is tolerance-level. Within
    /// [`PHASE_ACCEPT_FACTOR`] times the tolerances the point is accepted
    /// with a `warn!`; beyond that the phases have not converged and the
    /// caller gets an error rather than a point certified as optimal.
    fn accept_exhausted_point(&mut self) -> Result<(), Error> {
        let primal_excess = self.max_primal_excess();
        let dual_excess = self.max_dual_excess();
        if !(primal_excess <= PHASE_ACCEPT_FACTOR && dual_excess <= PHASE_ACCEPT_FACTOR) {
            return Err(Error::InternalError(format!(
                "simplex phases did not converge in {} rounds: worst primal violation is \
                 {:.1}x its tolerance, worst dual violation {:.1}x",
                MAX_PHASE_ROUNDS, primal_excess, dual_excess,
            )));
        }
        log::warn!(
            "simplex phases still trading tolerance-level infeasibilities after {} rounds \
             (worst primal {:.1}x, worst dual {:.1}x its tolerance); accepting the current point",
            MAX_PHASE_ROUNDS,
            primal_excess,
            dual_excess,
        );
        self.is_primal_feasible = true;
        self.is_dual_feasible = true;
        Ok(())
    }

    /// Largest bound violation of a basic variable as a multiple of that
    /// variable's tolerance (`0` when none is violated; `INFINITY` for a
    /// non-finite value).
    fn max_primal_excess(&self) -> f64 {
        let mut worst: f64 = 0.0;
        for (((&val, &min), &max), &tol) in self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
            .zip(&self.basic_var_tols)
        {
            if !val.is_finite() {
                return f64::INFINITY;
            }
            let violation = (min - val).max(val - max).max(0.0);
            worst = worst.max(violation / tol);
        }
        worst
    }

    /// Largest dual infeasibility of a non-basic column as a multiple of the
    /// tolerance it is priced with (`0` when every column is dual feasible;
    /// `INFINITY` for a non-finite reduced cost).
    fn max_dual_excess(&self) -> f64 {
        let mut worst: f64 = 0.0;
        for ((&obj_coeff, var_state), &tol) in self
            .nb_var_obj_coeffs
            .iter()
            .zip(&self.nb_var_states)
            .zip(&self.nb_var_tols)
        {
            if !obj_coeff.is_finite() {
                return f64::INFINITY;
            }
            if !nb_dual_feasible(var_state, obj_coeff, tol) {
                worst = worst.max(obj_coeff.abs() / tol);
            }
        }
        worst
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
        self.basic_var_tols.clear();
        self.nb_vars.clear();
        self.nb_var_vals.clear();
        self.nb_var_states.clear();
        self.nb_var_is_fixed.clear();
        // `nb_var_tols` and `nb_var_obj_coeffs` are rebuilt by `recalc_obj_coeffs`.

        for var in 0..n {
            match basis.0[var] {
                VarStatus::Basic => {
                    self.var_states[var] = VarState::Basic(self.basic_vars.len());
                    self.basic_vars.push(var);
                    self.basic_var_mins.push(self.orig_var_mins[var]);
                    self.basic_var_maxs.push(self.orig_var_maxs[var]);
                    self.basic_var_tols.push(self.orig_var_tols[var]);
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
                    self.nb_var_states.push(NonBasicVarState {
                        at_min: float_eq(val, min),
                        at_max: float_eq(val, max),
                    });
                    self.nb_var_is_fixed.push(false);
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
        self.measure_rows();

        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        self.is_dual_feasible = self.calc_dual_infeasibility().0 == 0;
        Ok(())
    }

    pub(crate) fn fix_var(&mut self, var: usize, val: f64) -> Result<StopReason, Error> {
        if val < self.orig_var_mins[var] || val > self.orig_var_maxs[var] {
            return Err(Error::Infeasible);
        }

        let col = match self.var_states[var] {
            VarState::Basic(row) => {
                // if var was basic, remove it.
                self.calc_row_coeffs(row);
                let pivot_info = self.choose_entering_col_dual(row, val)?;
                self.calc_col_coeffs(pivot_info.col);
                self.pivot(&pivot_info)?;
                pivot_info.col
            }

            VarState::NonBasic(col) => {
                self.calc_col_coeffs(col);

                let diff = val - self.nb_var_vals[col];
                for (r, coeff) in self.col_coeffs.iter() {
                    self.basic_var_vals[r] -= diff * coeff;
                }
                self.cur_obj_val += diff * self.nb_var_obj_coeffs[col];
                self.nb_var_vals[col] = val;

                col
            }
        };

        self.nb_var_states[col] = NonBasicVarState {
            at_min: true,
            at_max: true,
        };
        self.nb_var_is_fixed[col] = true;

        self.is_primal_feasible = false;
        self.reoptimize()
    }

    /// Return whether the var was really unset and whether reoptimization
    /// finished within the active deadline.
    pub(crate) fn unfix_var(&mut self, var: usize) -> Result<(bool, StopReason), Error> {
        if let VarState::NonBasic(col) = self.var_states[var] {
            if !std::mem::replace(&mut self.nb_var_is_fixed[col], false) {
                return Ok((false, StopReason::Finished));
            }

            let cur_val = self.nb_var_vals[col];
            self.nb_var_states[col] = NonBasicVarState {
                at_min: float_eq(cur_val, self.orig_var_mins[var]),
                at_max: float_eq(cur_val, self.orig_var_maxs[var]),
            };

            self.is_dual_feasible = false;
            let stop = self.reoptimize()?;
            Ok((true, stop))
        } else {
            Ok((false, StopReason::Finished))
        }
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

        if self.run_phases()? == StopReason::Limit {
            return Ok(StopReason::Limit);
        }

        // Disable updates of primal sq. norms, because lengthy primal simplex runs
        // are unlikely after the initial solve.
        self.enable_primal_steepest_edge = false;

        Ok(StopReason::Finished)
    }

    fn optimize(&mut self) -> Result<StopReason, Error> {
        debug_assert!(
            !self.artificial_obj,
            "optimize runs on real reduced costs; recalc_obj_coeffs installs them"
        );
        // Terminal valve, as in `restore_feasibility`: optimality is declared
        // only on reduced costs recomputed from the original data and on basic
        // values that still satisfy the rows. Bounded per phase, and the
        // reduced-cost recomputation is skipped when nothing pivoted since the
        // last exact one (the caller installs exact reduced costs on entry).
        let mut terminal_restarts = 0;
        for iter in 0.. {
            if iter % DEADLINE_CHECK_INTERVAL == 0 {
                if check_deadline(&self.deadline) == StopReason::Limit {
                    return Ok(StopReason::Limit);
                }

                let (num_vars, infeasibility) = self.calc_dual_infeasibility();
                debug!(
                    "optimize iter {}: obj.: {}, non-optimal coeffs: {} ({})",
                    iter, self.cur_obj_val, num_vars, infeasibility,
                );
            }

            if let Some(pivot_info) = self.choose_pivot()? {
                self.pivot(&pivot_info)?;
                self.lp_iterations += 1;
            } else {
                if terminal_restarts >= MAX_TERMINAL_RESTARTS {
                    self.warn_if_terminal_residual_remains("optimize", iter);
                } else {
                    let excess = self.measure_rows();
                    let re_examine = if excess > REBUILD_RESIDUAL_FACTOR {
                        debug!(
                            "optimize iter {}: terminal residual is {:.1}x a row's tolerance \
                             (limit {}); rebuilding before declaring optimality",
                            iter, excess, REBUILD_RESIDUAL_FACTOR,
                        );
                        self.rebuild()?;
                        true
                    } else if self.reduced_cost_updates > 0 {
                        // Reduced costs drift the same way the values do, and
                        // recomputing them through the current factorization
                        // is one transposed solve: re-examine on exact ones
                        // before declaring optimality.
                        self.recalc_obj_coeffs()?;
                        true
                    } else {
                        false
                    };
                    if re_examine {
                        terminal_restarts += 1;
                        continue;
                    }
                }
                debug!(
                    "found optimum in {} iterations, obj.: {}",
                    iter + 1,
                    self.cur_obj_val,
                );
                break;
            }
        }

        self.is_dual_feasible = true;
        // Primal simplex moves through vertices: report where it actually
        // ended instead of assuming primal feasibility survived.
        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        Ok(StopReason::Finished)
    }

    fn restore_feasibility(&mut self) -> Result<StopReason, Error> {
        let obj_str = if self.is_dual_feasible {
            "obj."
        } else {
            "artificial obj."
        };

        // Numerics valve, armed once per stall: neither an infeasibility
        // declaration nor a "feasible" termination is allowed to stand on
        // incrementally-updated values — both first get the basis
        // refactorized and the values recomputed from the original data
        // (see below). Any successful pivot re-arms it, and the terminal
        // re-examinations are bounded per phase, so neither path can loop.
        let mut rebuilt_since_pivot = false;
        let mut terminal_restarts = 0;

        for iter in 0.. {
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
                let pivot_info = match self.choose_entering_col_dual(row, leaving_new_val) {
                    Ok(pivot_info) => pivot_info,
                    Err(Error::Infeasible) if !rebuilt_since_pivot => {
                        // "No eligible entering column" is a proof of primal
                        // infeasibility only in exact arithmetic. This deep
                        // in an eta-file chain, the leaving row can be a
                        // *phantom* violation — basic values drifted by
                        // accumulated round-off — whose (equally drifted)
                        // pivot row then blocks every candidate; declaring
                        // infeasibility here is a wrong answer (netlib/brandy
                        // did exactly this). Rebuild the factorization and
                        // the basic values from the original data and
                        // re-examine: a phantom dissolves, a real
                        // infeasibility survives the refresh and the next
                        // declaration stands.
                        debug!(
                            "restore feasibility iter {}: no entering column for row {}; \
                             refreshing basis before declaring infeasibility",
                            iter, row,
                        );
                        self.rebuild()?;
                        rebuilt_since_pivot = true;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                self.calc_col_coeffs(pivot_info.col);
                self.pivot(&pivot_info)?;
                self.lp_iterations += 1;
                // Any successful pivot is progress: re-arm the valve.
                rebuilt_since_pivot = false;
            } else {
                // The bound checks above look at basic values, not at the
                // rows they were derived from: a pivot on a small element can
                // leave every value inside its bounds yet off the basis
                // equations by far more than `EPS` (issue #44). End the phase
                // only on a point a fresh factorization would also produce.
                if !rebuilt_since_pivot && terminal_restarts >= MAX_TERMINAL_RESTARTS {
                    self.warn_if_terminal_residual_remains("restore feasibility", iter);
                } else if !rebuilt_since_pivot {
                    let excess = self.measure_rows();
                    if excess > REBUILD_RESIDUAL_FACTOR {
                        debug!(
                            "restore feasibility iter {}: terminal residual is {:.1}x a row's \
                             tolerance (limit {}); rebuilding the basic values before ending the phase",
                            iter, excess, REBUILD_RESIDUAL_FACTOR,
                        );
                        self.rebuild()?;
                        terminal_restarts += 1;
                        rebuilt_since_pivot = true;
                        continue;
                    }
                    // `measure_rows` re-derived the round-off floors from the
                    // current values; a basic value the stale tolerance accepted
                    // can lie outside the fresh one. Price again before ending.
                    if self.calc_primal_infeasibility().0 != 0 {
                        debug!(
                            "restore feasibility iter {}: fresh tolerances expose a violation; \
                             continuing",
                            iter,
                        );
                        terminal_restarts += 1;
                        continue;
                    }
                }
                debug!(
                    "restored feasibility in {} iterations, {}: {}",
                    iter + 1,
                    obj_str,
                    self.cur_obj_val,
                );
                break;
            }
        }

        // Measured on the tolerances now in force: the loop above exits only
        // when pricing finds nothing, but its re-examinations are bounded.
        self.is_primal_feasible = self.calc_primal_infeasibility().0 == 0;
        // The Harris ratio test trades bounded dual infeasibility for pivot
        // size: report what is actually there instead of assuming the flag
        // still holds — on reduced costs recomputed from the original data,
        // since the incremental ones drift exactly like the values do. (The
        // artificial objective is never dual feasible.)
        if self.reduced_cost_updates > 0 && !self.artificial_obj {
            self.recalc_obj_coeffs()?;
        }
        self.is_dual_feasible = self.is_dual_feasible && self.calc_dual_infeasibility().0 == 0;
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

        let Some(PreparedRow {
            mut coeffs,
            rhs,
            row_scale,
            slack_var_min,
            slack_var_max,
        }) = prepare_row(coeffs, cmp_op, rhs)?
        else {
            return Ok(StopReason::Finished);
        };

        let slack_var = self.num_total_vars();

        self.orig_obj_coeffs.push(0.0);
        self.orig_var_mins.push(slack_var_min);
        self.orig_var_maxs.push(slack_var_max);
        let tol = contract_tol(self.feasibility_tol, row_scale);
        self.orig_var_tols.push(tol);
        self.row_noise.push(0.0);
        self.var_states.push(VarState::Basic(self.basic_vars.len()));
        self.basic_vars.push(slack_var);
        self.basic_var_mins.push(slack_var_min);
        self.basic_var_maxs.push(slack_var_max);
        self.basic_var_tols.push(tol);

        let mut lhs_val = 0.0;
        for (var, &coeff) in coeffs.iter() {
            let val = match self.var_states[var] {
                VarState::Basic(idx) => self.basic_var_vals[idx],
                VarState::NonBasic(idx) => self.nb_var_vals[idx],
            };
            lhs_val += val * coeff;
        }
        self.basic_var_vals.push(rhs - lhs_val);

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
        // The new row's slack is basic with zero cost: its multiplier is zero.
        self.multiplier_abs.push(0.0);

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

        self.measure_rows();
        self.is_primal_feasible = false;
        self.reoptimize()
    }

    /// Number of infeasible basic vars and sum of their infeasibilities.
    fn calc_primal_infeasibility(&self) -> (usize, f64) {
        let mut num_vars = 0;
        let mut infeasibility = 0.0;
        for (((&val, &min), &max), &tol) in self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
            .zip(&self.basic_var_tols)
        {
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
        for ((&obj_coeff, var_state), &tol) in self
            .nb_var_obj_coeffs
            .iter()
            .zip(&self.nb_var_states)
            .zip(&self.nb_var_tols)
        {
            if !nb_dual_feasible(var_state, obj_coeff, tol) {
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

    fn choose_pivot(&mut self) -> Result<Option<PivotInfo>, Error> {
        let entering_c = {
            let filtered_obj_coeffs = self
                .nb_var_obj_coeffs
                .iter()
                .zip(&self.nb_var_states)
                .zip(&self.nb_var_tols)
                .enumerate()
                .filter_map(|(col, ((&obj_coeff, var_state), &tol))| {
                    // Choose only among non-basic vars that can be changed
                    // with objective decreasing.
                    if nb_dual_feasible(var_state, obj_coeff, tol) {
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
                return Ok(None);
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

        // First, we determine the max change in entering variable so that basic variables
        // remain feasible using relaxed bounds.
        let mut max_step = (entering_other_val - entering_cur_val).abs();
        for (r, &coeff) in self.col_coeffs.iter() {
            let coeff_abs = coeff.abs();
            if coeff_abs < EPS {
                continue;
            }

            // By which amount can we change the entering variable so that the limit on this
            // basic var is not violated. The var with the minimum such amount becomes leaving.
            let cur_step = (get_leaving_var_step(r, coeff) + self.basic_var_tols[r]) / coeff_abs;
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
            if coeff_abs < EPS {
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

            let entering_diff = (self.basic_var_vals[row] - leaving_new_val) / pivot_coeff;
            let entering_new_val = entering_cur_val + entering_diff;

            Ok(Some(PivotInfo {
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

            Ok(Some(PivotInfo {
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
            .zip(&self.basic_var_tols)
            .enumerate()
            .filter_map(|(r, (((&val, &min), &max), &tol))| {
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

    fn choose_entering_col_dual(
        &self,
        row: usize,
        leaving_new_val: f64,
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

        let is_eligible_var = |coeff: f64, var_state: &NonBasicVarState| -> bool {
            let entering_diff_sign = if coeff >= EPS {
                !leaving_diff_sign
            } else if coeff <= -EPS {
                leaving_diff_sign
            } else {
                return false;
            };

            if entering_diff_sign {
                !var_state.at_max
            } else {
                !var_state.at_min
            }
        };

        // Harris rule. See e.g.
        // Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989).
        // A practical anti-cycling procedure for linearly constrained optimization.
        // Mathematical Programming, 45(1-3), 437-474.
        //
        // https://link.springer.com/content/pdf/10.1007/BF01589114.pdf

        // First, we determine the max step (change in the leaving variable obj. coeff that still
        // leaves us with a dual-feasible state) using relaxed bounds.
        let mut max_step = f64::INFINITY;
        for (c, &coeff) in self.row_coeffs.iter() {
            let var_state = &self.nb_var_states[c];
            if !is_eligible_var(coeff, var_state) {
                continue;
            }

            let obj_coeff = clamp_obj_coeff(self.nb_var_obj_coeffs[c], var_state);
            let cur_step = (obj_coeff.abs() + self.nb_var_tols[c]) / coeff.abs();
            if cur_step < max_step {
                max_step = cur_step;
            }
        }

        // Second, we choose among the variables satisfying the relaxed step bound
        // the one with the biggest pivot coefficient. This allows for a much more
        // numerically stable basis at the price of slight infeasibility in dual
        // variables (within each column's own tolerance).
        let mut entering_c = None;
        let mut pivot_coeff_abs = f64::NEG_INFINITY;
        let mut pivot_coeff = 0.0;
        for (c, &coeff) in self.row_coeffs.iter() {
            let var_state = &self.nb_var_states[c];
            if !is_eligible_var(coeff, var_state) {
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
            Err(Error::Infeasible)
        }
    }

    fn pivot(&mut self, pivot_info: &PivotInfo) -> Result<(), Error> {
        self.cur_obj_val += self.nb_var_obj_coeffs[pivot_info.col] * pivot_info.entering_diff;

        let entering_var = self.nb_vars[pivot_info.col];

        if pivot_info.elem.is_none() {
            // "entering" var is still non-basic, it just changes value from one limit
            // to the other.
            self.nb_var_vals[pivot_info.col] = pivot_info.entering_new_val;
            for (r, coeff) in self.col_coeffs.iter() {
                self.basic_var_vals[r] -= pivot_info.entering_diff * coeff;
            }
            let var_state = &mut self.nb_var_states[pivot_info.col];
            var_state.at_min = float_eq(
                pivot_info.entering_new_val,
                self.orig_var_mins[entering_var],
            );
            var_state.at_max = float_eq(
                pivot_info.entering_new_val,
                self.orig_var_maxs[entering_var],
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
        self.basic_var_tols[pivot_elem.row] = self.var_tol(entering_var);

        if self.enable_dual_steepest_edge {
            self.update_dual_sq_norms(pivot_elem.row, pivot_coeff);
        }

        // Update non-basic vars stuff

        let leaving_var = self.basic_vars[pivot_elem.row];

        self.nb_var_vals[pivot_info.col] = pivot_elem.leaving_new_val;
        self.nb_var_tols[pivot_info.col] = self.col_tol(leaving_var);
        let leaving_var_state = &mut self.nb_var_states[pivot_info.col];
        leaving_var_state.at_min =
            float_eq(pivot_elem.leaving_new_val, self.orig_var_mins[leaving_var]);
        leaving_var_state.at_max =
            float_eq(pivot_elem.leaving_new_val, self.orig_var_maxs[leaving_var]);

        let pivot_obj = self.nb_var_obj_coeffs[pivot_info.col] / pivot_coeff;
        for (c, &coeff) in self.row_coeffs.iter() {
            if c == pivot_info.col {
                self.nb_var_obj_coeffs[c] = -pivot_obj;
            } else {
                self.nb_var_obj_coeffs[c] -= pivot_obj * coeff;
            }
        }
        self.reduced_cost_updates += 1;

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
        let eta_matrices_nnz = self.basis_solver.eta_matrices.coeff_cols.nnz();
        if eta_matrices_nnz < self.basis_solver.lu_factors.nnz() {
            self.basis_solver
                .push_eta_matrix(&self.col_coeffs, pivot_elem.row, pivot_coeff);
        } else if self.reduced_cost_updates >= REDUCED_COST_UPDATE_LIMIT {
            // Refactorizing anyway: also replace the incrementally-updated
            // basic values — and, past `REDUCED_COST_UPDATE_LIMIT`, the
            // reduced costs — with ones computed from the original data, so
            // neither kind of drift outlives a long incremental chain.
            self.rebuild()?;
        } else {
            self.refresh_values()?;
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

    /// Refactorize the current basis and recompute the basic values from the
    /// original data, discarding the drift the incremental pivot updates
    /// accumulated. The reduced costs are left alone here: [`Self::rebuild`]
    /// recomputes both, and `pivot` chooses between the two by
    /// [`REDUCED_COST_UPDATE_LIMIT`].
    fn refresh_values(&mut self) -> Result<(), Error> {
        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars)?;
        self.recalc_basic_var_vals()?;
        self.measure_rows();
        // The incrementally updated objective is refreshed only when it has
        // drifted materially. It feeds the branch-and-bound's node bounds and
        // pseudocosts, and on a degenerate tree replacing it with an exact
        // value that differs only in the last bits changes tie-breaking enough
        // to change the whole search: measured on miplib/gt2 as 75 ms → 3.9 min
        // when refreshed unconditionally, while the largest discrepancy seen
        // over 78k refactorizations there was 1.6e-10 — round-off, five orders
        // below any pruning slack. Real drift is caught; noise is left alone.
        if !self.artificial_obj {
            let exact = self.exact_obj_val();
            if (exact - self.cur_obj_val).abs() > OBJECTIVE_DRIFT_TOL * (1.0 + exact.abs()) {
                self.cur_obj_val = exact;
            }
        }
        Ok(())
    }

    /// Refactorize and recompute the basic values — and, unless the phase-1
    /// artificial objective is in force, the reduced costs — from the original
    /// data: at both phases' terminal stalls, the phase-1 stall valve, and
    /// refactorizations past [`REDUCED_COST_UPDATE_LIMIT`]. Recomputing
    /// reduced costs mid-phase was once avoided: exact ones expose noise-level
    /// dual infeasibilities on large-coefficient columns that a flat `EPS`
    /// then priced into forced degenerate pivots (measured on miplib/gt2 as a
    /// re-routed search, 50 ms to minutes). Each column is now priced with its
    /// own round-off floor ([`Self::col_tol`]), which is what makes exact
    /// reduced costs safe to use anywhere. The objective value is refreshed
    /// only on real drift ([`OBJECTIVE_DRIFT_TOL`]) for the reason given in
    /// [`Self::refresh_values`].
    fn rebuild(&mut self) -> Result<(), Error> {
        self.refresh_values()?;
        if !self.artificial_obj {
            self.recalc_obj_coeffs()?;
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
        Ok(())
    }

    fn recalc_obj_coeffs(&mut self) -> Result<(), Error> {
        // Same as recalc_basic_var_vals: pending etas participate in the
        // (transposed) dense solve instead of forcing a refactorization.
        let multipliers = {
            let mut rhs = vec![0.0; self.num_constraints()];
            for (c, &var) in self.basic_vars.iter().enumerate() {
                rhs[c] = self.orig_obj_coeffs[var];
            }
            self.basis_solver.solve_transp_dense_with_etas(&mut rhs);
            rhs
        };
        self.multiplier_abs.clear();
        self.multiplier_abs
            .extend(multipliers.iter().map(|y| y.abs()));

        self.nb_var_obj_coeffs.clear();
        self.nb_var_tols.clear();
        for &var in &self.nb_vars {
            //guaranteed to be a valid index
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let mut dot_prod = 0.0;
            let mut magnitude = self.orig_obj_coeffs[var].abs();
            for (r, val) in col.iter() {
                let term = val * multipliers[r];
                dot_prod += term;
                magnitude += term.abs();
            }
            self.nb_var_obj_coeffs
                .push(self.orig_obj_coeffs[var] - dot_prod);
            self.nb_var_tols
                .push(EPS.max(REBUILD_NOISE_FLOOR * magnitude));
        }

        self.reduced_cost_updates = 0;
        // Leaving the phase-1 artificial objective, the real objective value
        // is installed; otherwise the incremental one is kept unless it has
        // drifted — see `refresh_values` for why a last-bit change matters.
        let exact = self.exact_obj_val();
        if self.artificial_obj
            || (exact - self.cur_obj_val).abs() > OBJECTIVE_DRIFT_TOL * (1.0 + exact.abs())
        {
            self.cur_obj_val = exact;
        }
        self.artificial_obj = false;
        Ok(())
    }

    /// Objective value of the current point from the original coefficients.
    fn exact_obj_val(&self) -> f64 {
        let mut obj = 0.0;
        for (r, &var) in self.basic_vars.iter().enumerate() {
            obj += self.orig_obj_coeffs[var] * self.basic_var_vals[r];
        }
        for (c, &var) in self.nb_vars.iter().enumerate() {
            obj += self.orig_obj_coeffs[var] * self.nb_var_vals[c];
        }
        obj
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

#[derive(Debug)]
struct PivotInfo {
    col: usize,
    entering_new_val: f64,
    entering_diff: f64,

    /// Contains info about the intersection between pivot row and column.
    /// If it is None, objective can be decreased without changing the basis
    /// (simply by changing the value of non-basic variable chosen as entering)
    elem: Option<PivotElem>,
}

#[derive(Debug)]
struct PivotElem {
    row: usize,
    coeff: f64,
    leaving_new_val: f64,
}

/// Stuff related to inversion of the basis matrix
#[derive(Clone)]
struct BasisSolver {
    lu_factors: LUFactors,
    lu_factors_transp: LUFactors,
    scratch: ScratchSpace,
    eta_matrices: EtaMatrices,
    rhs: ScatteredVec,
}

impl BasisSolver {
    fn push_eta_matrix(&mut self, col_coeffs: &SparseVec, r_leaving: usize, pivot_coeff: f64) {
        let coeffs = col_coeffs.iter().map(|(r, &coeff)| {
            let val = if r == r_leaving {
                1.0 - 1.0 / pivot_coeff
            } else {
                coeff / pivot_coeff
            };
            (r, val)
        });
        self.eta_matrices.push(r_leaving, coeffs);
    }

    fn reset(&mut self, orig_constraints_csc: &CsMat, basic_vars: &[usize]) -> Result<(), Error> {
        self.scratch.clear_sparse(basic_vars.len());
        self.eta_matrices.clear_and_resize(basic_vars.len());
        self.rhs.clear_and_resize(basic_vars.len());
        self.lu_factors = lu_factorize(
            basic_vars.len(),
            |c| {
                orig_constraints_csc
                    .outer_view(basic_vars[c])
                    //guaranteed to be a valid index
                    .unwrap()
                    .into_raw_storage()
            },
            LU_STABILITY_THRESHOLD,
            &mut self.scratch,
        )?;
        self.lu_factors_transp = self.lu_factors.transpose();
        Ok(())
    }

    fn solve<'a>(&mut self, rhs: impl Iterator<Item = (usize, &'a f64)>) -> &ScatteredVec {
        self.rhs.set(rhs);
        self.lu_factors.solve(&mut self.rhs, &mut self.scratch);

        // apply eta matrices (Vanderbei p.139)
        for idx in 0..self.eta_matrices.len() {
            let r_leaving = self.eta_matrices.leaving_rows[idx];
            let coeff = *self.rhs.get(r_leaving);
            for (r, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                *self.rhs.get_mut(r) -= coeff * val;
            }
        }

        &mut self.rhs
    }

    /// Dense counterpart of [`Self::solve`]: LU solve plus the forward eta
    /// application, so callers with dense right-hand sides (the recalcs) no
    /// longer need a full refactorization just because etas are pending.
    fn solve_dense_with_etas(&mut self, rhs: &mut [f64]) {
        self.lu_factors.solve_dense(rhs, &mut self.scratch);
        for idx in 0..self.eta_matrices.len() {
            let coeff = rhs[self.eta_matrices.leaving_rows[idx]];
            if coeff != 0.0 {
                for (r, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                    rhs[r] -= coeff * val;
                }
            }
        }
    }

    /// Dense counterpart of [`Self::solve_transp`]: the reverse eta
    /// application, then the transposed LU solve.
    fn solve_transp_dense_with_etas(&mut self, rhs: &mut [f64]) {
        for idx in (0..self.eta_matrices.len()).rev() {
            let mut coeff = 0.0;
            for (i, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                coeff += val * rhs[i];
            }
            rhs[self.eta_matrices.leaving_rows[idx]] -= coeff;
        }
        self.lu_factors_transp.solve_dense(rhs, &mut self.scratch);
    }

    /// Pass right-hand side via self.rhs
    fn solve_transp<'a>(&mut self, rhs: impl Iterator<Item = (usize, &'a f64)>) -> &ScatteredVec {
        self.rhs.set(rhs);
        // apply eta matrices in reverse (Vanderbei p.139)
        for idx in (0..self.eta_matrices.len()).rev() {
            let mut coeff = 0.0;
            // eta col `dot` rhs_transp
            for (i, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                coeff += val * self.rhs.get(i);
            }
            let r_leaving = self.eta_matrices.leaving_rows[idx];
            *self.rhs.get_mut(r_leaving) -= coeff;
        }

        self.lu_factors_transp
            .solve(&mut self.rhs, &mut self.scratch);
        &mut self.rhs
    }
}

#[derive(Clone, Debug)]
struct EtaMatrices {
    leaving_rows: Vec<usize>,
    coeff_cols: SparseMat,
}

impl EtaMatrices {
    fn new(n_rows: usize) -> EtaMatrices {
        EtaMatrices {
            leaving_rows: vec![],
            coeff_cols: SparseMat::new(n_rows),
        }
    }

    fn len(&self) -> usize {
        self.leaving_rows.len()
    }

    fn clear_and_resize(&mut self, n_rows: usize) {
        self.leaving_rows.clear();
        self.coeff_cols.clear_and_resize(n_rows);
    }

    fn push(&mut self, leaving_row: usize, coeffs: impl Iterator<Item = (usize, f64)>) {
        self.leaving_rows.push(leaving_row);
        self.coeff_cols.append_col(coeffs);
    }
}

fn into_resized(vec: CsVec, len: usize) -> CsVec {
    let (mut indices, mut data) = vec.into_raw_storage();

    while let Some(&i) = indices.last() {
        if i < len {
            // TODO: binary search
            break;
        }

        indices.pop();
        data.pop();
    }

    CsVec::new(len, indices, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{assert_matrix_eq, to_sparse};
    use crate::{OptimizationDirection, Problem};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn initialize() {
        init();
        let sol = Solver::try_new(
            &[2.0, 1.0],
            &[f64::NEG_INFINITY, 5.0],
            &[0.0, f64::INFINITY],
            &[
                (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 6.0),
                (to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 8.0),
                (to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 2.0),
                (to_sparse(&[0.0, 1.0]), ComparisonOp::Eq, 3.0),
            ],
            &[VarDomain::Real, VarDomain::Real],
            Default::default(),
        )
        .unwrap();

        assert_eq!(sol.num_vars, 2);
        assert!(!sol.is_primal_feasible);
        assert!(!sol.is_dual_feasible);

        assert_eq!(&sol.orig_obj_coeffs, &[2.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(
            &sol.orig_var_mins,
            &[f64::NEG_INFINITY, 5.0, 0.0, 0.0, f64::NEG_INFINITY, 0.0,]
        );
        assert_eq!(
            &sol.orig_var_maxs,
            &[0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0, 0.0]
        );

        // Equilibration scales the second constraint (max structural
        // coefficient 2) and its rhs by 1/2; the slack column stays 1. The
        // unit-coefficient rows are unchanged.
        let orig_constraints_ref = vec![
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.5, 1.0, 0.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        assert_matrix_eq(&sol.orig_constraints, &orig_constraints_ref);

        assert_eq!(&sol.orig_rhs, &[6.0, 4.0, 2.0, 3.0]);

        assert_eq!(&sol.basic_vars, &[2, 3, 4, 5]);
        assert_eq!(&sol.basic_var_vals, &[1.0, -1.0, -3.0, -2.0]);
        assert_eq!(&sol.dual_edge_sq_norms, &[1.0, 1.0, 1.0, 1.0]);

        assert_eq!(&sol.nb_vars, &[0, 1]);
        assert_eq!(&sol.nb_var_obj_coeffs, &[-1.0, 1.0]);
        assert_eq!(&sol.nb_var_vals, &[0.0, 5.0]);
        assert_eq!(&sol.primal_edge_sq_norms, &[3.25, 5.0]);

        assert_eq!(sol.cur_obj_val, 0.0);
    }

    #[test]
    fn try_new_rejects_nan_bound() {
        init();
        // A NaN bound used to slip past the `min > max` guard (every
        // comparison against NaN is false, including this one), so it was
        // accepted as an ordinary bound. Downstream, the simplex loop's own
        // bound comparisons against that NaN never resolve either, so the
        // solve hangs forever instead of reporting Infeasible up front (the
        // same thing set_var_bounds already guards against for edits).
        let res = Solver::try_new(
            &[1.0],
            &[f64::NAN],
            &[10.0],
            &[],
            &[VarDomain::Real],
            Default::default(),
        );
        assert_eq!(res.unwrap_err(), Error::Infeasible);
    }

    /// Dense recalculations with pending etas must match a fresh factorization
    /// of the same basis. Values are compared per variable because reloading
    /// may reorder basis positions.
    #[test]
    fn recalcs_with_pending_etas_match_a_fresh_factorization() {
        init();
        // minimize x + y + z, pairwise sums >= 2, boxes [0, 10]: optimum
        // x = y = z = 1. A bound tightening then forces dual pivots, which
        // push etas.
        let mut solver = Solver::try_new(
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[10.0, 10.0, 10.0],
            &[
                (to_sparse(&[1.0, 1.0, 0.0]), ComparisonOp::Ge, 2.0),
                (to_sparse(&[0.0, 1.0, 1.0]), ComparisonOp::Ge, 2.0),
                (to_sparse(&[1.0, 0.0, 1.0]), ComparisonOp::Ge, 2.0),
            ],
            &[VarDomain::Real, VarDomain::Real, VarDomain::Real],
            None,
        )
        .unwrap();
        assert_eq!(solver.initial_solve().unwrap(), StopReason::Finished);
        solver.set_var_bounds(2, 0.0, 0.25).unwrap();
        assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
        assert!(
            solver.basis_solver.eta_matrices.len() > 0,
            "fixture must leave etas pending to exercise eta-aware recalculation"
        );

        // Recalculate through the eta-aware dense solves.
        solver.recalc_basic_var_vals().unwrap();
        solver.recalc_obj_coeffs().unwrap();
        let by_var = |s: &Solver| -> Vec<(usize, f64)> {
            let mut v: Vec<(usize, f64)> = s
                .basic_vars
                .iter()
                .zip(&s.basic_var_vals)
                .map(|(&var, &val)| (var, val))
                .collect();
            v.sort_by_key(|&(var, _)| var);
            v
        };
        let rc_by_var = |s: &Solver| -> Vec<(usize, f64)> {
            let mut v: Vec<(usize, f64)> = s
                .nb_vars
                .iter()
                .zip(&s.nb_var_obj_coeffs)
                .map(|(&var, &rc)| (var, rc))
                .collect();
            v.sort_by_key(|&(var, _)| var);
            v
        };
        let eta_vals = by_var(&solver);
        let eta_rcs = rc_by_var(&solver);
        let eta_obj = solver.cur_obj_val;

        // Reloading the solver's own snapshot refactorizes from scratch and
        // reruns the recalcs eta-free — the ground truth.
        let basis = solver.snapshot_basis();
        solver.load_basis(&basis).unwrap();
        assert_eq!(solver.basis_solver.eta_matrices.len(), 0);
        for ((va, a), (vb, b)) in eta_vals.iter().zip(by_var(&solver).iter()) {
            assert_eq!(va, vb);
            assert!((a - b).abs() < 1e-9, "basic val of var {va}: {a} vs {b}");
        }
        for ((va, a), (vb, b)) in eta_rcs.iter().zip(rc_by_var(&solver).iter()) {
            assert_eq!(va, vb);
            assert!((a - b).abs() < 1e-9, "reduced cost of var {va}: {a} vs {b}");
        }
        assert!((eta_obj - solver.cur_obj_val).abs() < 1e-9);
    }

    #[test]
    fn solve_integer_singular_var() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Ge, 90.0);
        assert!(
            (problem
                .solve()
                .unwrap()
                .into_solution()
                .unwrap()
                .objective()
                - 3.0)
                .abs()
                < EPS
        );

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Ge, 91.0);
        assert!(
            (problem
                .solve()
                .unwrap()
                .into_solution()
                .unwrap()
                .objective()
                - 4.0)
                .abs()
                < EPS
        );

        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Le, 90.0);
        assert!(
            (problem
                .solve()
                .unwrap()
                .into_solution()
                .unwrap()
                .objective()
                - 3.0)
                .abs()
                < EPS
        );

        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Le, 91.0);
        assert!(
            (problem
                .solve()
                .unwrap()
                .into_solution()
                .unwrap()
                .objective()
                - 3.0)
                .abs()
                < EPS
        );
    }

    #[test]
    fn solve_powers_integer() {
        init();
        let n = 15626;
        // return (a,b,c) such that 2^a * 3^b * 5^c >= n and is minimized given a,b,c € N
        let logn = (n as f64).log2();
        let log2 = 2_f64.log2();
        let log3 = 3_f64.log2();
        let log5 = 5_f64.log2();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let p2 = problem.add_integer_var(log2, (0, 100));
        let p3 = problem.add_integer_var(log3, (0, 100));
        let p5 = problem.add_integer_var(log5, (0, 100));
        problem.add_constraint(
            &[(p2, log2), (p3, log3), (p5, log5)],
            ComparisonOp::Ge,
            logn,
        );
        let sol = problem.solve().unwrap().into_solution().unwrap();
        assert_eq!(sol.objective().round() as i64, 14);
    }

    #[test]
    fn initial_solve() {
        init();
        let mut sol = Solver::try_new(
            &[-3.0, -4.0],
            &[f64::NEG_INFINITY, 5.0],
            &[20.0, f64::INFINITY],
            &[
                (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 20.0),
                (to_sparse(&[-1.0, 4.0]), ComparisonOp::Le, 20.0),
            ],
            &[VarDomain::Real, VarDomain::Real],
            Default::default(),
        )
        .unwrap();
        sol.initial_solve().unwrap();

        assert!(sol.is_primal_feasible);
        assert!(sol.is_dual_feasible);

        assert_eq!(&sol.basic_vars, &[0, 1]);
        assert_eq!(&sol.basic_var_vals, &[12.0, 8.0]);
        assert_eq!(&sol.nb_vars, &[2, 3]);
        assert_eq!(&sol.nb_var_vals, &[0.0, 0.0]);
        // The optimum (x=12, y=8, obj -68) is unchanged by equilibration; only
        // the second constraint's slack reduced cost is scaled: that row
        // (-x+4y<=20, max coeff 4) is equilibrated by 1/4, so its dual scales
        // up 4x, 0.2 -> 0.8.
        assert_eq!(&sol.nb_var_obj_coeffs, &[3.2, 0.8]);
        assert_eq!(sol.cur_obj_val, -68.0);

        let infeasible = Solver::try_new(
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[f64::INFINITY, f64::INFINITY],
            &[
                (to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 10.0),
                (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 5.0),
            ],
            &[VarDomain::Real, VarDomain::Real],
            Default::default(),
        )
        .unwrap()
        .initial_solve();
        assert_eq!(infeasible.unwrap_err(), Error::Infeasible);
    }

    #[test]
    fn set_var_bounds_tighten_matches_fresh_solve() {
        init();
        // minimize 2x + 3y s.t. x + y >= 4, 0 <= x,y <= 10. Optimum: x=4, y=0, obj 8.
        let coeffs = [2.0, 3.0];
        let mins = [0.0, 0.0];
        let maxs = [10.0, 10.0];
        let cons = [(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 4.0)];
        let domains = [VarDomain::Real, VarDomain::Real];

        let mut warm = Solver::try_new(&coeffs, &mins, &maxs, &cons, &domains, None).unwrap();
        warm.initial_solve().unwrap();
        assert!(float_eq(warm.cur_obj_val, 8.0));

        // Tighten x to [0, 2] and re-solve warm: optimum becomes x=2, y=2, obj 10.
        warm.set_var_bounds(0, 0.0, 2.0).unwrap();
        assert_eq!(warm.reoptimize().unwrap(), StopReason::Finished);
        assert!(warm.is_primal_feasible && warm.is_dual_feasible);
        assert!(float_eq(warm.cur_obj_val, 10.0));
        assert!(float_eq(*warm.get_value(0), 2.0));
        assert!(float_eq(*warm.get_value(1), 2.0));

        // Fresh solve of the tightened problem must agree.
        let mut fresh =
            Solver::try_new(&coeffs, &mins, &[2.0, 10.0], &cons, &domains, None).unwrap();
        fresh.initial_solve().unwrap();
        assert!(float_eq(fresh.cur_obj_val, warm.cur_obj_val));
    }

    #[test]
    fn set_var_bounds_loosen_and_retighten() {
        init();
        // maximize x + y (internally minimize -x - y) s.t. x + y <= 4, 0 <= x,y <= 3.
        let mut solver = Solver::try_new(
            &[-1.0, -1.0],
            &[0.0, 0.0],
            &[3.0, 3.0],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        assert!(float_eq(solver.cur_obj_val, -4.0));

        // Tighten x to [0, 0.5]: optimum x=0.5, y=3, obj -3.5.
        solver.set_var_bounds(0, 0.0, 0.5).unwrap();
        assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
        assert!(float_eq(solver.cur_obj_val, -3.5));

        // Loosen x back to [0, 3]: optimum returns to -4.
        solver.set_var_bounds(0, 0.0, 3.0).unwrap();
        assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
        assert!(float_eq(solver.cur_obj_val, -4.0));

        assert!(solver.lp_iterations > 0);
    }

    #[test]
    fn set_var_bounds_crossing_is_infeasible_and_leaves_state_untouched() {
        init();
        let mut solver = Solver::try_new(
            &[1.0],
            &[0.0],
            &[10.0],
            &[(to_sparse(&[1.0]), ComparisonOp::Ge, 1.0)],
            &[VarDomain::Real],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        let obj_before = solver.cur_obj_val;
        assert_eq!(
            solver.set_var_bounds(0, 2.0, 1.0).unwrap_err(),
            Error::Infeasible
        );
        assert_eq!(solver.get_var_bounds(0), (0.0, 10.0)); // untouched
        assert!(float_eq(solver.cur_obj_val, obj_before));
    }

    #[test]
    fn set_var_bounds_nan_is_infeasible_and_leaves_state_untouched() {
        let mut original =
            Solver::try_new(&[1.0], &[0.0], &[10.0], &[], &[VarDomain::Real], None).unwrap();
        assert_eq!(original.initial_solve().unwrap(), StopReason::Finished);

        for (min, max) in [(f64::NAN, 10.0), (0.0, f64::NAN)] {
            let mut solver = original.clone();
            let bounds_before = solver.get_var_bounds(0);
            let value_before = *solver.get_value(0);
            let objective_before = solver.cur_obj_val;
            let primal_before = solver.is_primal_feasible;
            let dual_before = solver.is_dual_feasible;

            assert_eq!(solver.set_var_bounds(0, min, max), Err(Error::Infeasible));
            assert_eq!(solver.get_var_bounds(0), bounds_before);
            assert_eq!(*solver.get_value(0), value_before);
            assert_eq!(solver.cur_obj_val, objective_before);
            assert_eq!(solver.is_primal_feasible, primal_before);
            assert_eq!(solver.is_dual_feasible, dual_before);
        }

        let mut solver = original;
        assert_eq!(
            solver.set_var_bounds(0, f64::NEG_INFINITY, f64::INFINITY),
            Ok(())
        );
        assert_eq!(solver.get_var_bounds(0), (f64::NEG_INFINITY, f64::INFINITY));
    }

    /// Issue #44 at the engine level. The dual phase pivots `x2` into the
    /// basis on the `-1.19e-8` coefficient, multiplying round-off by ~1e8;
    /// the final basis is perfectly conditioned and its exact solution is
    /// `x1 = x2 = 0`, but the incrementally-updated values said `x2 = 2^-26`
    /// — inside its bounds, off its row by `1.5e-8`. A phase must end on
    /// values that satisfy the rows, not merely sit inside their bounds.
    #[test]
    fn phase_ends_on_values_that_satisfy_the_rows() {
        init();
        let tiny = -1.190834764418229e-8;
        let mut solver = Solver::try_new(
            &[1.0, 0.0, 0.0],
            &[-1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[
                (
                    to_sparse(&[0.0, 8.510167104926385, 0.0]),
                    ComparisonOp::Eq,
                    0.0,
                ),
                (to_sparse(&[-1.0, tiny, 0.0]), ComparisonOp::Eq, 0.0),
            ],
            &[VarDomain::Real, VarDomain::Real, VarDomain::Integer],
            None,
        )
        .unwrap();
        assert_eq!(solver.initial_solve().unwrap(), StopReason::Finished);

        let excess = solver.measure_rows();
        assert!(
            excess <= REBUILD_RESIDUAL_FACTOR,
            "terminal residual is {excess:.1}x a row's tolerance (limit {REBUILD_RESIDUAL_FACTOR})"
        );
        for var in 0..3 {
            let value = *solver.get_value(var);
            assert!(
                value.abs() < 1e-12,
                "var {var} = {value:e}; the only feasible point is 0"
            );
        }
        assert!(solver.is_primal_feasible && solver.is_dual_feasible);
    }

    /// The terminal valve in isolation, with no refactorization in play:
    /// plant drift into a basic value that stays inside its bounds and let the
    /// dual phase end. The bound checks see nothing; the residual check must.
    #[test]
    fn terminal_stall_rebuilds_drifted_basic_values() {
        init();
        // minimize -x - 2y, x, y in [0, 3], x + y <= 4: optimum y = 3, x = 1 (basic).
        let mut solver = Solver::try_new(
            &[-1.0, -2.0],
            &[0.0, 0.0],
            &[3.0, 3.0],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
        )
        .unwrap();
        assert_eq!(solver.initial_solve().unwrap(), StopReason::Finished);
        assert_eq!(solver.basic_vars, vec![0]);
        assert!((solver.basic_var_vals[0] - 1.0).abs() < 1e-12);
        assert!(solver.measure_rows() <= REBUILD_RESIDUAL_FACTOR);

        // Far beyond EPS, still inside [0, 3]: invisible to the bound checks.
        solver.basic_var_vals[0] -= 1e-6;
        assert!(
            solver.measure_rows() > REBUILD_RESIDUAL_FACTOR,
            "planted drift must show up as a row residual"
        );
        solver.is_primal_feasible = false;
        assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);

        assert!(solver.measure_rows() <= REBUILD_RESIDUAL_FACTOR);
        assert!(
            (solver.basic_var_vals[0] - 1.0).abs() < 1e-12,
            "drifted value {} was not rebuilt",
            solver.basic_var_vals[0]
        );
        assert!((solver.cur_obj_val + 7.0).abs() < 1e-12);

        // A residual of a few tolerances — above what the bound checks and
        // the validation guard allow, but inside the window a gate one order
        // above the tolerance would have let through — must be rebuilt too.
        let tol = solver.basic_var_tols[0];
        assert_eq!(tol, EPS, "a well-scaled row's slack is held to EPS");
        solver.basic_var_vals[0] -= 3.0 * tol;
        assert!(solver.measure_rows() > REBUILD_RESIDUAL_FACTOR);
        solver.is_primal_feasible = false;
        assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
        assert!(
            (solver.basic_var_vals[0] - 1.0).abs() < 1e-14,
            "a {}x-tolerance residual was not rebuilt: value {}",
            3.0,
            solver.basic_var_vals[0]
        );
    }

    /// The validation guard is absolute, floored at the row's round-off: a
    /// violation below a few hundred ulps of the activity is accepted (no
    /// double-precision point could do better), while the big-M trap — a
    /// violation that is tiny relative to the coefficients but large in
    /// absolute terms — is still rejected.
    #[test]
    fn check_constraints_floors_at_the_rows_round_off() {
        init();
        // 1e9 (x - b) == 10 with x, b continuous here: the guard only reads rows.
        let solver = Solver::try_new(
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[f64::INFINITY, 1.0],
            &[(to_sparse(&[1e9, -1e9]), ComparisonOp::Eq, 10.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
        )
        .unwrap();
        let tol = DEFAULT_FEASIBILITY_TOL;
        // Exact: 1e9 * (1 + 1e-8) - 1e9 = 10.
        assert!(solver.check_constraints(&[1.0 + 1e-8, 1.0], tol));
        // Off by one ulp of x: the activity moves by ~2e-7 in user units,
        // above the absolute tolerance but far below the row's round-off
        // floor (1e-13 x 2e9 = 2e-4), so it is accepted.
        let x = f64::from_bits((1.0f64 + 1e-8).to_bits() + 1);
        assert!(solver.check_constraints(&[x, 1.0], tol));
        // The big-M trap: b rounded from 5e-7 to 0 moves the row by 500.
        assert!(!solver.check_constraints(&[510.0 / 1e9 + 1.0, 0.0], tol));
        assert!(!solver.check_constraints(&[1.0 + 1e-8 + 1e-3 / 1e9 * 1e3, 1.0], tol));
    }

    /// Phase-round exhaustion accepts only what the Harris relaxations can
    /// leave behind: a violation a few tolerances wide is accepted (with a
    /// warning), a large one — primal or dual — is an error rather than a
    /// point certified optimal. Ablation: with the measurement removed both
    /// large violations would be accepted.
    #[test]
    fn exhausted_phases_accept_only_tolerance_level_violations() {
        init();
        // minimize -x - 2y, x, y in [0, 3], x + y <= 4: optimum y = 3, x = 1 (basic).
        let build = || {
            let mut solver = Solver::try_new(
                &[-1.0, -2.0],
                &[0.0, 0.0],
                &[3.0, 3.0],
                &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0)],
                &[VarDomain::Real, VarDomain::Real],
                None,
            )
            .unwrap();
            assert_eq!(solver.initial_solve().unwrap(), StopReason::Finished);
            assert_eq!(solver.basic_vars, vec![0]);
            solver
        };

        let mut solver = build();
        let tol = solver.basic_var_tols[0];
        solver.basic_var_vals[0] = solver.basic_var_maxs[0] + 3.0 * tol;
        assert!((solver.max_primal_excess() - 3.0).abs() < 1e-6);
        solver.is_primal_feasible = false;
        solver.accept_exhausted_point().unwrap();
        assert!(solver.is_primal_feasible && solver.is_dual_feasible);

        let mut solver = build();
        solver.basic_var_vals[0] = solver.basic_var_maxs[0] + 100.0 * tol;
        assert!(matches!(
            solver.accept_exhausted_point(),
            Err(Error::InternalError(_))
        ));

        let mut solver = build();
        // A reduced cost of the wrong sign for a column at its lower bound.
        let col = solver
            .nb_var_states
            .iter()
            .position(|st| st.at_min)
            .expect("a non-basic column at its lower bound");
        solver.nb_var_obj_coeffs[col] = -3.0 * EPS;
        assert!((solver.max_dual_excess() - 3.0).abs() < 1e-6);
        solver.accept_exhausted_point().unwrap();
        let mut solver = build();
        solver.nb_var_obj_coeffs[col] = -100.0 * EPS;
        assert!(matches!(
            solver.accept_exhausted_point(),
            Err(Error::InternalError(_))
        ));
    }

    /// A column's reduced cost is priced with `EPS` or its own round-off
    /// floor, whichever is larger: `1e-13` times the magnitude of the terms it
    /// is computed from. A wrong-signed reduced cost below that floor is
    /// noise, not a dual infeasibility, so pricing leaves it alone.
    #[test]
    fn column_tolerance_follows_the_reduced_cost_magnitude() {
        init();
        // minimize 1e4 x + y, x + y >= 1, x, y in [0, 10]: y = 1 basic, x at 0.
        let mut solver = Solver::try_new(
            &[1e4, 1.0],
            &[0.0, 0.0],
            &[10.0, 10.0],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 1.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
        )
        .unwrap();
        assert_eq!(solver.initial_solve().unwrap(), StopReason::Finished);
        assert_eq!(solver.basic_vars, vec![1]);
        for (c, &var) in solver.nb_vars.iter().enumerate() {
            assert_eq!(solver.nb_var_tols[c], solver.col_tol(var));
        }
        let col_x = solver.nb_vars.iter().position(|&v| v == 0).unwrap();
        let col_s = solver.nb_vars.iter().position(|&v| v == 2).unwrap();
        // The phase ended on exact reduced costs: the multiplier of the row
        // is y's cost, 1, so x's reduced cost is 1e4 - 1, computed from
        // magnitudes 1e4 + 1, and the slack's from 0 + 1.
        assert!((solver.nb_var_obj_coeffs[col_x] - 9999.0).abs() < 1e-9);
        assert!((solver.nb_var_tols[col_x] - REBUILD_NOISE_FLOOR * 10001.0).abs() < 1e-25);
        assert_eq!(solver.nb_var_tols[col_s], EPS);

        // Wrong-signed by 5 EPS but under the column's floor: not priced.
        assert!(solver.nb_var_states[col_x].at_min);
        solver.nb_var_obj_coeffs[col_x] = -5.0 * EPS;
        assert_eq!(solver.calc_dual_infeasibility().0, 0);
        assert_eq!(solver.max_dual_excess(), 0.0);
        assert!(solver.choose_pivot().unwrap().is_none());
        // Five times the floor: a real dual infeasibility.
        solver.nb_var_obj_coeffs[col_x] = -5.0 * solver.nb_var_tols[col_x];
        assert_eq!(solver.calc_dual_infeasibility().0, 1);
        assert!((solver.max_dual_excess() - 5.0).abs() < 1e-9);
        // The same magnitude on the slack column, whose floor is EPS, is
        // priced.
        solver.nb_var_obj_coeffs[col_x] = 9999.0;
        let wrong_sign = if solver.nb_var_states[col_s].at_min {
            -1.0
        } else {
            assert!(solver.nb_var_states[col_s].at_max);
            1.0
        };
        solver.nb_var_obj_coeffs[col_s] = wrong_sign * 5.0 * EPS;
        assert_eq!(solver.calc_dual_infeasibility().0, 1);
    }

    /// Each row's slack is held to the user's absolute tolerance expressed in
    /// the row's own equilibrated units, capped at `EPS`, and never below the
    /// row's own round-off floor.
    #[test]
    fn slack_tolerance_follows_the_row_scale() {
        init();
        let rows = |c: f64| (to_sparse(&[c]), ComparisonOp::Le, 1.0);
        let mut solver = Solver::try_new(
            &[1.0],
            &[0.0],
            &[1.0],
            &[rows(1.0), rows(1.0e2), rows(1.0e4), rows(1.0e9)],
            &[VarDomain::Real],
            None,
        )
        .unwrap();
        // structural var, then one slack per row: scales 1, 2^-6, 2^-13, 2^-29
        assert_eq!(solver.orig_var_tols[0], EPS);
        assert_eq!(solver.orig_var_tols[1], EPS);
        assert_eq!(
            solver.orig_var_tols[2], EPS,
            "1e-7 * 2^-6 exceeds EPS: capped"
        );
        assert_eq!(solver.orig_var_tols[3], 1e-7 * 2f64.powi(-13));
        assert_eq!(solver.orig_var_tols[4], 1e-7 * 2f64.powi(-29));
        // What is applied never drops below the row's round-off floor: the
        // 1e9 row's contract (1.9e-16) is unreachable, its floor is not.
        let floor_1e9 = solver.var_tol(4);
        assert!(
            floor_1e9 >= REBUILD_NOISE_FLOOR && floor_1e9 < 1e-12,
            "1e9 row held to its round-off floor, got {floor_1e9:e}"
        );
        assert_eq!(solver.var_tol(3), 1e-7 * 2f64.powi(-13));
        for (i, &var) in solver.basic_vars.iter().enumerate() {
            assert_eq!(solver.basic_var_tols[i], solver.var_tol(var));
        }

        solver.set_feasibility_tolerance(1e-9);
        assert_eq!(solver.orig_var_tols[0], EPS);
        assert_eq!(solver.orig_var_tols[2], 1e-9 * 2f64.powi(-6));
        assert_eq!(solver.orig_var_tols[3], 1e-9 * 2f64.powi(-13));
        assert!(solver.var_tol(3) >= REBUILD_NOISE_FLOOR);
        for (i, &var) in solver.basic_vars.iter().enumerate() {
            assert_eq!(solver.basic_var_tols[i], solver.var_tol(var));
        }
    }

    #[test]
    fn check_constraints_rejects_non_finite_activity() {
        let solver = Solver::try_new(
            &[0.0],
            &[0.0],
            &[f64::INFINITY],
            &[(to_sparse(&[1.0e308]), ComparisonOp::Eq, f64::INFINITY)],
            &[VarDomain::Real],
            None,
        )
        .unwrap();

        assert!(!solver.check_constraints(&[1.0e308], 1.0e-7));
    }

    #[test]
    fn basis_snapshot_load_roundtrip() {
        init();
        // This bounded fixture has objective -68 at (12, 8), with both
        // structural variables basic and both slacks non-basic. Its basis is
        // therefore non-trivial and differs from the slack basis.
        let mut solver = Solver::try_new(
            &[-3.0, -4.0],
            &[f64::NEG_INFINITY, 5.0],
            &[20.0, f64::INFINITY],
            &[
                (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 20.0),
                (to_sparse(&[-1.0, 4.0]), ComparisonOp::Le, 20.0),
            ],
            &[VarDomain::Real, VarDomain::Real],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        let obj = solver.cur_obj_val;
        let vals: Vec<f64> = (0..2).map(|v| *solver.get_value(v)).collect();
        let basis = solver.snapshot_basis();

        // Wreck the state by loading the all-slack basis…
        let slack = solver.slack_basis();
        solver.load_basis(&slack).unwrap();

        // …then reload the optimal basis: objective and values must round-trip.
        solver.load_basis(&basis).unwrap();
        assert!(solver.is_primal_feasible && solver.is_dual_feasible);
        assert!(float_eq(solver.cur_obj_val, obj));
        for v in 0..2 {
            assert!(float_eq(*solver.get_value(v), vals[v]));
        }
    }

    #[test]
    fn slack_basis_load_then_reoptimize_reaches_optimum() {
        init();
        // minimize 2x + 3y s.t. x + y >= 4, 0 <= x,y <= 10 → obj 8.
        let mut solver = Solver::try_new(
            &[2.0, 3.0],
            &[0.0, 0.0],
            &[10.0, 10.0],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 4.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        assert!(float_eq(solver.cur_obj_val, 8.0));

        let slack = solver.slack_basis();
        solver.load_basis(&slack).unwrap();
        assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
        assert!(float_eq(solver.cur_obj_val, 8.0));
    }

    #[test]
    fn load_basis_rejects_wrong_shape() {
        init();
        let mut solver = Solver::try_new(
            &[1.0],
            &[0.0],
            &[1.0],
            &[(to_sparse(&[1.0]), ComparisonOp::Le, 1.0)],
            &[VarDomain::Real],
            None,
        )
        .unwrap();
        solver.initial_solve().unwrap();
        // 2 total vars (1 structural + 1 slack); a basis with zero Basic entries is invalid.
        let bad = Basis(vec![VarStatus::AtLower, VarStatus::AtLower]);
        assert!(solver.load_basis(&bad).is_err());
        // Solver must still be usable via the slack-basis fallback path.
        let slack = solver.slack_basis();
        solver.load_basis(&slack).unwrap();
        assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    }
}
