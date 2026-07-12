use core::panic;

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
/// Deliberately tight (upstream minilp uses `1e-8`); the whole correctness
/// suite — including the big-M models, whose branch & bound relies on node
/// LPs resolving basic integer values sharply onto their bounds — is
/// calibrated against this value. Loosening it (globally or just for
/// bound-violation candidacy) lets basic values legally sit ~1e-8 off their
/// bounds, which 1e9-scale big-M rows amplify past the MIP layer's
/// rounded-incumbent feasibility guard and the branch-and-bound tree
/// explodes. The flip side of running this tight — round-off noise being
/// promoted into phantom infeasibilities — is handled where it bites, by
/// the refresh valve in [`Solver::restore_feasibility`].
pub const EPS: f64 = 1e-10;

/// How often (in simplex iterations) the primal/dual loops in `optimize` and
/// `restore_feasibility` check the deadline and emit a progress `debug!` log.
/// Checking every iteration would make the deadline check itself a
/// significant fraction of the per-iteration cost on easy problems; checking
/// too rarely would make a time limit overshoot by a visible amount on hard
/// ones. 1000 keeps the check overhead negligible while still bounding the
/// worst-case overshoot to about a thousand pivots.
pub(crate) const DEADLINE_CHECK_INTERVAL: u64 = 1000;

/// A basic integer variable only seeds a Gomory mixed-integer cut when its
/// fractional part is at least this far from BOTH 0 and 1. Near-integral
/// values are dominated by tableau round-off — the cut's `f0/(1−f0)` ratios
/// would amplify that noise into garbage coefficients — and cutting a
/// hair's width off the relaxation is worthless anyway.
const GMI_FRAC_MIN: f64 = 0.01;

/// GMI cut coefficients below this magnitude are relaxed into the rhs using
/// the variable's bound (never dropped: dropping a term from a `Ge` row
/// strengthens it, i.e. makes the cut invalid). Values this small are
/// tableau round-off, not information.
const GMI_COEF_EPS: f64 = 1e-11;

/// A GMI cut whose coefficient magnitudes span more than this ratio is
/// discarded outright: such rows make the basis ill-conditioned, and the
/// extreme coefficients usually trace back to amplified round-off in the
/// tableau row rather than real structure.
const GMI_DYNAMISM_CAP: f64 = 1e7;

/// Threshold-pivoting stability coefficient passed to [`lu_factorize`] for
/// every LU (re)factorization the simplex performs: a candidate pivot is
/// accepted only if its magnitude is at least this fraction of the column's
/// largest eligible entry. 0.1 is the standard textbook default for
/// Gilbert-Peierls sparse LU (see `lu_factorize`'s doc reference) — it
/// balances numerical stability (higher would refuse more marginal pivots,
/// at the cost of extra fill-in) against sparsity (lower risks amplifying
/// rounding error through a poorly-conditioned pivot).
pub(crate) const LU_STABILITY_THRESHOLD: f64 = 0.1;

pub(crate) fn float_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}
pub(crate) fn float_ne(a: f64, b: f64) -> bool {
    !float_eq(a, b)
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
    /// Total number of simplex pivots performed across all solves/reoptimizes on this instance.
    pub(crate) lp_iterations: u64,

    orig_obj_coeffs: Vec<f64>,
    orig_var_mins: Vec<f64>,
    orig_var_maxs: Vec<f64>,
    pub(crate) orig_var_domains: Vec<VarDomain>,
    orig_constraints: CsMat, // excluding rhs
    orig_constraints_csc: CsMat,
    orig_rhs: Vec<f64>,

    enable_primal_steepest_edge: bool,
    enable_dual_steepest_edge: bool,

    is_primal_feasible: bool,
    is_dual_feasible: bool,

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
            if min > max {
                return Err(Error::Infeasible);
            }

            // initially all user-created variables are non-basic
            var_states.push(VarState::NonBasic(nb_vars.len()));
            nb_vars.push(v);

            // Try to choose values to achieve dual feasibility.
            let init_val = if float_eq(min, max) {
                // Fixed variable, the obj. coeff doesn't matter.
                min
            } else if min.is_infinite() && max.is_infinite() {
                // Free variable, if we are lucky and obj. coeff is zero, then dual-feasible.
                if float_ne(obj_coeffs[v], 0.0) {
                    //TODO should this use float_eq?
                    is_dual_feasible = false;
                }
                0.0
            } else if obj_coeffs[v] > 0.0 {
                // We need a finite value and prefer min for dual feasibility.
                if min.is_finite() {
                    min
                } else {
                    is_dual_feasible = false;
                    max
                }
            } else if obj_coeffs[v] < 0.0 {
                // We need a finite value and prefer max for dual feasibility.
                if max.is_finite() {
                    max
                } else {
                    is_dual_feasible = false;
                    min
                }
            } else if min.is_finite() {
                // Obj. coeff is zero, just take any finite value,
                // dual feasibility will be satisfied.
                min
            } else {
                max
            };

            nb_var_vals.push(init_val);
            obj_val += init_val * obj_coeffs[v];

            nb_var_states.push(NonBasicVarState {
                at_min: float_eq(init_val, min),
                at_max: float_eq(init_val, max),
            });
        }

        let mut constraint_coeffs = vec![];
        let mut orig_rhs = vec![];

        // Initially, all slack vars are basic.
        let mut basic_vars = vec![];
        let mut basic_var_vals = vec![];
        let mut basic_var_mins = vec![];
        let mut basic_var_maxs = vec![];

        for (coeffs, cmp_op, rhs) in constraints {
            let rhs = *rhs;

            if coeffs.indices().is_empty() {
                let is_tautological = match cmp_op {
                    ComparisonOp::Eq => float_eq(rhs, 0.0),
                    ComparisonOp::Le => 0.0 <= rhs,
                    ComparisonOp::Ge => 0.0 >= rhs,
                };

                if is_tautological {
                    continue;
                } else {
                    return Err(Error::Infeasible);
                }
            }

            constraint_coeffs.push(coeffs.clone());
            orig_rhs.push(rhs);

            let (slack_var_min, slack_var_max) = match cmp_op {
                ComparisonOp::Le => (0.0, f64::INFINITY),
                ComparisonOp::Ge => (f64::NEG_INFINITY, 0.0),
                ComparisonOp::Eq => (0.0, 0.0),
            };

            orig_var_mins.push(slack_var_min);
            orig_var_maxs.push(slack_var_max);

            basic_var_mins.push(slack_var_min);
            basic_var_maxs.push(slack_var_max);

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

        let res = Self {
            num_vars,
            orig_obj_coeffs,
            orig_var_mins,
            orig_var_maxs,
            orig_constraints,
            orig_constraints_csc,
            orig_rhs,
            deadline,
            lp_iterations: 0,
            orig_var_domains: var_domains.to_vec(),
            enable_primal_steepest_edge,
            enable_dual_steepest_edge,
            is_primal_feasible,
            is_dual_feasible,
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

        Ok(res)
    }

    pub(crate) fn get_value(&self, var: usize) -> &f64 {
        match self.var_states[var] {
            VarState::Basic(idx) => &self.basic_var_vals[idx],
            VarState::NonBasic(idx) => &self.nb_var_vals[idx],
        }
    }

    /// Check `values` (one entry per structural var) against every ORIGINAL
    /// constraint row, within the ABSOLUTE tolerance `tol`. Bounds are not
    /// checked here. Each row's sense is encoded by its slack var's bounds
    /// (lhs + s = rhs with s in [smin, smax]  ⇔  rhs - smax ≤ lhs ≤ rhs - smin);
    /// slack bounds are never touched by branching, so this always reflects the
    /// user's original rows.
    ///
    /// The tolerance is deliberately NOT scaled by the row magnitude: this
    /// check exists for the big-M trap, where a violation that is tiny
    /// RELATIVE to huge row coefficients (e.g. 5.0 on a 1e9-scale row) is
    /// decisive in absolute terms. Any row-scale-relative tolerance would be
    /// blind to exactly the violations this guard is for.
    pub(crate) fn check_constraints(&self, values: &[f64], tol: f64) -> bool {
        for (r, row) in self.orig_constraints.outer_iterator().enumerate() {
            let rhs = self.orig_rhs[r];
            let mut lhs = 0.0;
            for (v, &coeff) in row.iter() {
                if v < self.num_vars {
                    lhs += coeff * values[v];
                }
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
            if lhs < lo - tol || lhs > hi + tol {
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

    /// Reduced cost and bound-side status of a NONBASIC var, `None` for basic
    /// vars (their reduced cost is zero by definition). Only meaningful right
    /// after a completed solve: the maintained `nb_var_obj_coeffs` are true
    /// reduced costs exactly when the solver is dual-feasible on the real
    /// objective, which `reoptimize`/`initial_solve` guarantee on `Finished`.
    pub(crate) fn nb_reduced_cost(&self, var: usize) -> Option<(f64, bool, bool)> {
        match self.var_states[var] {
            VarState::Basic(_) => None,
            VarState::NonBasic(col) => Some((
                self.nb_var_obj_coeffs[col],
                self.nb_var_states[col].at_min,
                self.nb_var_states[col].at_max,
            )),
        }
    }

    /// Change a variable's bounds in place. Records the new bounds and repairs the
    /// invariants that depend on them; does NOT run simplex — call [`Self::reoptimize`]
    /// afterwards. Returns `Err(Infeasible)` (state untouched) if `min > max`.
    pub(crate) fn set_var_bounds(&mut self, var: usize, min: f64, max: f64) -> Result<(), Error> {
        if min > max {
            return Err(Error::Infeasible);
        }
        self.orig_var_mins[var] = min;
        self.orig_var_maxs[var] = max;
        match self.var_states[var] {
            VarState::Basic(row) => {
                self.basic_var_mins[row] = min;
                self.basic_var_maxs[row] = max;
                let val = self.basic_var_vals[row];
                if val < min - EPS || val > max + EPS {
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
                    && (self.nb_var_states[col].at_min && self.nb_var_obj_coeffs[col] > -EPS
                        || self.nb_var_states[col].at_max && self.nb_var_obj_coeffs[col] < EPS
                        || self.nb_var_obj_coeffs[col].abs() < EPS);
            }
        }
        Ok(())
    }

    /// Re-solve after bound changes or a basis load: dual simplex to restore primal
    /// feasibility, then primal simplex if reduced costs became dual-infeasible
    /// (only happens after loosening bounds or a numerically imperfect basis load).
    pub(crate) fn reoptimize(&mut self) -> Result<StopReason, Error> {
        if !self.is_primal_feasible && self.restore_feasibility()? == StopReason::Limit {
            return Ok(StopReason::Limit);
        }
        if !self.is_dual_feasible {
            self.recalc_obj_coeffs()?;
            if self.optimize()? == StopReason::Limit {
                return Ok(StopReason::Limit);
            }
            // Primal simplex may have moved through vertices; make sure primal holds too.
            if !self.is_primal_feasible && self.restore_feasibility()? == StopReason::Limit {
                return Ok(StopReason::Limit);
            }
        }
        Ok(StopReason::Finished)
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
    /// are recomputed honestly, so half-pivoted pre-load state is fully discarded.
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
        self.nb_var_is_fixed.clear();

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
        self.restore_feasibility()
    }

    /// Return true if the var was really unset.
    pub(crate) fn unfix_var(&mut self, var: usize) -> bool {
        if let VarState::NonBasic(col) = self.var_states[var] {
            if !std::mem::replace(&mut self.nb_var_is_fixed[col], false) {
                return false;
            }

            let cur_val = self.nb_var_vals[col];
            self.nb_var_states[col] = NonBasicVarState {
                at_min: float_eq(cur_val, self.orig_var_mins[var]),
                at_max: float_eq(cur_val, self.orig_var_maxs[var]),
            };

            // Shouldn't result in error, presumably problem was solvable before this variable
            // was fixed.
            self.is_dual_feasible = false;
            //TODO check unwrap
            self.optimize().unwrap();
            true
        } else {
            false
        }
    }

    pub(crate) fn add_gomory_cut(&mut self, var: usize) -> Result<StopReason, Error> {
        if let VarState::Basic(row) = self.var_states[var] {
            self.calc_row_coeffs(row);

            let mut cut_coeffs = SparseVec::new();
            for (col, &coeff) in self.row_coeffs.iter() {
                let var = self.nb_vars[col];
                cut_coeffs.push(var, coeff.floor() - coeff);
            }

            let cut_bound = self.basic_var_vals[row].floor() - self.basic_var_vals[row];
            let num_total_vars = self.num_total_vars();
            self.add_constraint(
                cut_coeffs.into_csvec(num_total_vars),
                ComparisonOp::Le,
                cut_bound,
            )
        } else {
            panic!("var {:?} is not basic!", var);
        }
    }

    /// Generate (without adding) a Gomory MIXED-integer cut from `var`'s
    /// tableau row, as a `Ge` inequality `(coeffs, rhs)` over TOTAL variable
    /// space (structural + slacks — slacks are ordinary bounded variables to
    /// `add_constraints`, which spares the error-prone substitution back
    /// into structural space). `None` when any soundness or numerics guard
    /// rejects. Unlike the legacy [`Self::add_gomory_cut`] above (the
    /// PURE-integer fractional cut, valid only when every nonbasic in the
    /// row is integer and sits at a zero lower bound — preconditions the
    /// caller of that public API owns), this handles the bounded mixed case:
    /// nonbasics are shifted to their active bound (`t = x − l` at lower,
    /// `t = u − x` at upper), integer coefficient treatment is applied only
    /// where integrality is PROVEN, and everything else conservatively uses
    /// the continuous formula, which is valid for integer variables too —
    /// just weaker.
    ///
    /// The caller must ensure the LP is solved to optimality (fresh
    /// factorization, real values) and that `var` is basic with a fractional
    /// value; `domains` is the STRUCTURAL domain slice.
    ///
    /// Not wired into the search: the root-cut driver hookup was measured
    /// and reverted (see `mip::run_root_cuts` — on this solver's node-LP
    /// economics dense GMI rows cost more per node than their bound gain
    /// saves in tree size, even at 92% gap closure on gt2). Kept, with its
    /// validity tests, for when node LPs get cheaper; generation is correct.
    #[allow(dead_code)]
    pub(crate) fn gmi_cut(&mut self, var: usize, domains: &[VarDomain]) -> Option<(CsVec, f64)> {
        // Numeric slack for "this data value is an integer" checks on
        // bounds and row coefficients (data, not LP values).
        const DATA_INT_EPS: f64 = 1e-9;
        let is_data_int = |x: f64| (x - x.round()).abs() <= DATA_INT_EPS;

        let row = match self.var_states[var] {
            VarState::Basic(row) => row,
            VarState::NonBasic(_) => return None,
        };
        let x_star = self.basic_var_vals[row];
        let f0 = x_star - x_star.floor();
        if !(GMI_FRAC_MIN..=1.0 - GMI_FRAC_MIN).contains(&f0) {
            return None;
        }

        // The mutable tableau read happens before the immutable-borrowing
        // closure below (it only fills the row_coeffs scratch).
        self.calc_row_coeffs(row);

        // Is total var `v`'s value provably integral whenever every integer
        // variable is integral? Structural: its domain. Slack of row k:
        // integer row data over integer-domained vars only.
        let slack_is_int = |k: usize| -> bool {
            if !is_data_int(self.orig_rhs[k]) {
                return false;
            }
            for (v, &c) in self.orig_constraints.outer_view(k).unwrap().iter() {
                if v >= self.num_vars {
                    continue; // the row's own slack, coefficient 1
                }
                if c == 0.0 {
                    continue;
                }
                if !is_data_int(c)
                    || !matches!(
                        self.orig_var_domains[v],
                        crate::VarDomain::Integer | crate::VarDomain::Boolean
                    )
                {
                    return false;
                }
            }
            true
        };

        let mut terms: Vec<(usize, f64)> = Vec::new();
        let mut rhs = f0;
        for (col, &a_bar) in self.row_coeffs.iter() {
            if a_bar == 0.0 {
                continue;
            }
            let v = self.nb_vars[col];
            let state = &self.nb_var_states[col];
            if state.at_min && state.at_max {
                continue; // fixed: t ≡ 0, the term is exactly zero
            }
            if !state.at_min && !state.at_max {
                return None; // free nonbasic: t ≥ 0 does not hold, no valid shift
            }
            let at_lower = state.at_min;
            let bound = self.nb_var_vals[col];
            // Row in t-space: x_i + Σ ã_j t_j = x_i*, t_j ≥ 0.
            let a_tilde = if at_lower { a_bar } else { -a_bar };

            let t_is_int = if v < self.num_vars {
                matches!(
                    domains[v],
                    crate::VarDomain::Integer | crate::VarDomain::Boolean
                ) && is_data_int(bound)
            } else {
                slack_is_int(v - self.num_vars) && is_data_int(bound)
            };

            // GMI coefficient on t_j (always ≥ 0 by construction).
            let g = if t_is_int {
                let f_j = a_tilde - a_tilde.floor();
                if f_j <= f0 {
                    f_j
                } else {
                    f0 * (1.0 - f_j) / (1.0 - f0)
                }
            } else if a_tilde >= 0.0 {
                a_tilde
            } else {
                -f0 * a_tilde / (1.0 - f0)
            };
            if g == 0.0 {
                continue; // integer coefficient: the term drops exactly
            }

            // Back to x-space: t = x − l keeps +g and shifts the rhs up by
            // g·l; t = u − x flips to −g and shifts the rhs down by g·u.
            if at_lower {
                terms.push((v, g));
                rhs += g * bound;
            } else {
                terms.push((v, -g));
                rhs -= g * bound;
            }
        }

        // Numerics gate: relax near-zero coefficients away using the var's
        // bound (dropping a term from a Ge row STRENGTHENS it — invalid);
        // an infinite bound there, any non-finite value, or excessive
        // dynamism discards the whole cut.
        let mut kept: Vec<(usize, f64)> = Vec::with_capacity(terms.len());
        for (v, c) in terms {
            if !c.is_finite() {
                return None;
            }
            if c.abs() >= GMI_COEF_EPS {
                kept.push((v, c));
                continue;
            }
            let relax_bound = if c > 0.0 {
                self.orig_var_maxs[v]
            } else {
                self.orig_var_mins[v]
            };
            if !relax_bound.is_finite() {
                return None;
            }
            rhs -= c * relax_bound;
        }
        if kept.is_empty() || !rhs.is_finite() {
            return None;
        }
        let (mut lo_mag, mut hi_mag) = (f64::INFINITY, 0.0f64);
        for &(_, c) in &kept {
            lo_mag = lo_mag.min(c.abs());
            hi_mag = hi_mag.max(c.abs());
        }
        if hi_mag / lo_mag > GMI_DYNAMISM_CAP {
            return None;
        }

        kept.sort_by_key(|&(v, _)| v);
        let n = self.num_total_vars();
        let (indices, data): (Vec<usize>, Vec<f64>) = kept.into_iter().unzip();
        Some((CsVec::new(n, indices, data), rhs))
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

        if !self.is_primal_feasible && self.restore_feasibility()? == StopReason::Limit {
            return Ok(StopReason::Limit);
        }

        if !self.is_dual_feasible {
            self.recalc_obj_coeffs()?;
            if self.optimize()? == StopReason::Limit {
                return Ok(StopReason::Limit);
            }
        }

        // Disable updates of primal sq. norms, because lengthy primal simplex runs
        // are unlikely after the initial solve.
        self.enable_primal_steepest_edge = false;

        Ok(StopReason::Finished)
    }

    fn optimize(&mut self) -> Result<StopReason, Error> {
        for iter in 0.. {
            self.lp_iterations += 1;
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
            } else {
                debug!(
                    "found optimum in {} iterations, obj.: {}",
                    iter + 1,
                    self.cur_obj_val,
                );
                break;
            }
        }

        self.is_dual_feasible = true;
        Ok(StopReason::Finished)
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
                let pivot_info = match self.choose_entering_col_dual(row, leaving_new_val) {
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
                        // the basic values from the original data and
                        // re-examine: a phantom dissolves, a real
                        // infeasibility survives the refresh and the next
                        // declaration stands.
                        debug!(
                            "restore feasibility iter {}: no entering column for row {}; \
                             refreshing basis before declaring infeasibility",
                            iter, row,
                        );
                        self.recalc_basic_var_vals()?;
                        refreshed_since_pivot = true;
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                self.calc_col_coeffs(pivot_info.col);
                self.pivot(&pivot_info)?;
                // Any successful pivot is progress: re-arm the valve.
                refreshed_since_pivot = false;
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

        self.is_primal_feasible = true;
        Ok(StopReason::Finished)
    }

    pub(crate) fn add_constraint(
        &mut self,
        coeffs: CsVec,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<StopReason, Error> {
        self.add_constraints(vec![(coeffs, cmp_op, rhs)])
    }

    /// Add a batch of rows to a solved LP and dual-simplex back to
    /// optimality. The whole batch costs ONE matrix rebuild, ONE basis
    /// refactorization and ONE feasibility restore — the per-row versions of
    /// those dominated `add_constraint` loops (the root cut loop adds tens
    /// of rows per round; measured on BIP_easy, 32 per-row refactorizations
    /// of a 2900-row basis were most of the loop's cost). Rows may reference
    /// any variable that exists BEFORE the batch, never another batch row's
    /// slack.
    pub(crate) fn add_constraints(
        &mut self,
        new_rows: Vec<(CsVec, ComparisonOp, f64)>,
    ) -> Result<StopReason, Error> {
        assert!(self.is_primal_feasible);
        assert!(self.is_dual_feasible);

        let mut rows = Vec::with_capacity(new_rows.len());
        for (coeffs, cmp_op, rhs) in new_rows {
            if coeffs.indices().is_empty() {
                let is_tautological = match cmp_op {
                    ComparisonOp::Eq => float_eq(rhs, 0.0),
                    ComparisonOp::Le => 0.0 <= rhs,
                    ComparisonOp::Ge => 0.0 >= rhs,
                };
                if is_tautological {
                    continue;
                }
                return Err(Error::Infeasible);
            }
            rows.push((coeffs, cmp_op, rhs));
        }
        if rows.is_empty() {
            return Ok(StopReason::Finished);
        }

        // Every new row's slack enters the basis at the row's current
        // activity slack; the basis matrix stays square and gains a
        // block-triangular slack border, so existing tableau rows are
        // unchanged (relied on by the steepest-edge update below).
        let first_slack = self.num_total_vars();
        let new_num_total_vars = first_slack + rows.len();
        for (i, (coeffs, cmp_op, rhs)) in rows.iter().enumerate() {
            let (slack_var_min, slack_var_max) = match cmp_op {
                ComparisonOp::Le => (0.0, f64::INFINITY),
                ComparisonOp::Ge => (f64::NEG_INFINITY, 0.0),
                ComparisonOp::Eq => (0.0, 0.0),
            };
            self.orig_obj_coeffs.push(0.0);
            self.orig_var_mins.push(slack_var_min);
            self.orig_var_maxs.push(slack_var_max);
            self.var_states.push(VarState::Basic(self.basic_vars.len()));
            self.basic_vars.push(first_slack + i);
            self.basic_var_mins.push(slack_var_min);
            self.basic_var_maxs.push(slack_var_max);

            let mut lhs_val = 0.0;
            for (var, &coeff) in coeffs.iter() {
                debug_assert!(
                    var < first_slack,
                    "batch rows must not reference batch slacks"
                );
                let val = match self.var_states[var] {
                    VarState::Basic(idx) => self.basic_var_vals[idx],
                    VarState::NonBasic(idx) => self.nb_var_vals[idx],
                };
                lhs_val += val * coeff;
            }
            self.basic_var_vals.push(rhs - lhs_val);
            self.orig_rhs.push(*rhs);
        }

        let mut new_orig_constraints = CsMat::empty(CompressedStorage::CSR, new_num_total_vars);
        for row in self.orig_constraints.outer_iterator() {
            new_orig_constraints =
                new_orig_constraints.append_outer_csvec(resized_view(&row, new_num_total_vars));
        }
        for (i, (coeffs, _, _)) in rows.into_iter().enumerate() {
            let mut coeffs = into_resized(coeffs, new_num_total_vars);
            coeffs.append(first_slack + i, 1.0);
            new_orig_constraints = new_orig_constraints.append_outer_csvec(coeffs.view());
        }

        self.orig_constraints = new_orig_constraints;
        self.orig_constraints_csc = self.orig_constraints.to_csc();

        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars)?;

        if self.enable_primal_steepest_edge || self.enable_dual_steepest_edge {
            // Existing tableau rows didn't change (slack border), so only
            // the new rows contribute to the sq. norms.
            for r in (self.num_constraints() - (new_num_total_vars - first_slack))
                ..self.num_constraints()
            {
                self.calc_row_coeffs(r);

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
        }

        self.is_primal_feasible = false;
        self.restore_feasibility()
    }

    /// Number of infeasible basic vars and sum of their infeasibilities.
    fn calc_primal_infeasibility(&self) -> (usize, f64) {
        let mut num_vars = 0;
        let mut infeasibility = 0.0;
        for ((&val, &min), &max) in self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
        {
            if val < min - EPS {
                num_vars += 1;
                infeasibility += min - val;
            } else if val > max + EPS {
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
        for (&obj_coeff, var_state) in self.nb_var_obj_coeffs.iter().zip(&self.nb_var_states) {
            if !(var_state.at_min && obj_coeff > -EPS || var_state.at_max && obj_coeff < EPS) {
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
                .enumerate()
                .filter_map(|(col, (&obj_coeff, var_state))| {
                    // Choose only among non-basic vars that can be changed
                    // with objective decreasing.
                    if (var_state.at_min && obj_coeff > -EPS)
                        || (var_state.at_max && obj_coeff < EPS)
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
            let cur_step = (get_leaving_var_step(r, coeff) + EPS) / coeff_abs;
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
            .enumerate()
            .filter_map(|(r, ((&val, &min), &max))| {
                if val < min - EPS {
                    Some((r, min - val))
                } else if val > max + EPS {
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
            let cur_step = (obj_coeff.abs() + EPS) / coeff.abs();
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
        // TODO: periodically (say, every 1000 pivots) recalc basic vars and object coeffs
        // from scratch for numerical stability.

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

        if self.enable_dual_steepest_edge {
            self.update_dual_sq_norms(pivot_elem.row, pivot_coeff);
        }

        // Update non-basic vars stuff

        let leaving_var = self.basic_vars[pivot_elem.row];

        self.nb_var_vals[pivot_info.col] = pivot_elem.leaving_new_val;
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
        } else {
            self.basis_solver
                .reset(&self.orig_constraints_csc, &self.basic_vars)?;
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

        if self.basis_solver.eta_matrices.len() > 0 {
            self.basis_solver
                .reset(&self.orig_constraints_csc, &self.basic_vars)?;
        }

        self.basis_solver
            .lu_factors
            .solve_dense(&mut cur_vals, &mut self.basis_solver.scratch);
        self.basic_var_vals = cur_vals;
        Ok(())
    }

    fn recalc_obj_coeffs(&mut self) -> Result<(), Error> {
        if self.basis_solver.eta_matrices.len() > 0 {
            self.basis_solver
                .reset(&self.orig_constraints_csc, &self.basic_vars)?;
        }

        let multipliers = {
            let mut rhs = vec![0.0; self.num_constraints()];
            for (c, &var) in self.basic_vars.iter().enumerate() {
                rhs[c] = self.orig_obj_coeffs[var];
            }
            self.basis_solver
                .lu_factors_transp
                .solve_dense(&mut rhs, &mut self.basis_solver.scratch);
            rhs
        };

        self.nb_var_obj_coeffs.clear();
        for &var in &self.nb_vars {
            //guaranteed to be a valid index
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let dot_prod: f64 = col.iter().map(|(r, val)| val * multipliers[r]).sum();
            self.nb_var_obj_coeffs
                .push(self.orig_obj_coeffs[var] - dot_prod);
        }

        self.cur_obj_val = 0.0;
        for (r, &var) in self.basic_vars.iter().enumerate() {
            self.cur_obj_val += self.orig_obj_coeffs[var] * self.basic_var_vals[r];
        }
        for (c, &var) in self.nb_vars.iter().enumerate() {
            self.cur_obj_val += self.orig_obj_coeffs[var] * self.nb_var_vals[c];
        }
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

        let orig_constraints_ref = vec![
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        assert_matrix_eq(&sol.orig_constraints, &orig_constraints_ref);

        assert_eq!(&sol.orig_rhs, &[6.0, 8.0, 2.0, 3.0]);

        assert_eq!(&sol.basic_vars, &[2, 3, 4, 5]);
        assert_eq!(&sol.basic_var_vals, &[1.0, -2.0, -3.0, -2.0]);
        assert_eq!(&sol.dual_edge_sq_norms, &[1.0, 1.0, 1.0, 1.0]);

        assert_eq!(&sol.nb_vars, &[0, 1]);
        assert_eq!(&sol.nb_var_obj_coeffs, &[-1.0, 1.0]);
        assert_eq!(&sol.nb_var_vals, &[0.0, 5.0]);
        assert_eq!(&sol.primal_edge_sq_norms, &[4.0, 8.0]);

        assert_eq!(sol.cur_obj_val, 0.0);
    }

    /// Hand-derived GMI fixture. minimize −3x − 2y (i.e. max 3x + 2y),
    /// 4x + 3y ≤ 10, x,y integer in [0,2]: the LP vertex is x = 2 (nonbasic
    /// at upper), y = 2/3 (basic, f0 = 2/3), slack s = 0 (nonbasic at
    /// lower). Tableau row: y − (4/3)t_x + (1/3)t_s = 2/3 with t_x = 2 − x.
    /// Integer formulas (x's bound and the row's data are integral):
    /// f_x = frac(−4/3) = 2/3 ≤ f0 → g_x = 2/3; f_s = 1/3 ≤ f0 → g_s = 1/3.
    /// Cut (2/3)t_x + (1/3)t_s ≥ 2/3, in x-space −(2/3)x + (1/3)s ≥ −2/3 —
    /// equivalent to 2x + y ≤ 4, which every integer point satisfies while
    /// the vertex (2, 2/3) violates it by exactly f0.
    #[test]
    fn gmi_cut_matches_hand_derivation() {
        init();
        let domains = [VarDomain::Integer, VarDomain::Integer];
        let mut solver = Solver::try_new(
            &[-3.0, -2.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            &[(to_sparse(&[4.0, 3.0]), ComparisonOp::Le, 10.0)],
            &domains,
            None,
        )
        .unwrap();
        assert_eq!(solver.initial_solve().unwrap(), StopReason::Finished);
        assert!((*solver.get_value(0) - 2.0).abs() < 1e-9);
        assert!((*solver.get_value(1) - 2.0 / 3.0).abs() < 1e-9);

        let (coeffs, rhs) = solver.gmi_cut(1, &domains).expect("y is basic fractional");
        let terms: Vec<(usize, f64)> = coeffs.iter().map(|(v, &c)| (v, c)).collect();
        assert_eq!(terms.len(), 2, "terms: {:?}", terms);
        assert_eq!(terms[0].0, 0); // x
        assert!((terms[0].1 + 2.0 / 3.0).abs() < 1e-9, "{:?}", terms);
        assert_eq!(terms[1].0, 2); // the row's slack
        assert!((terms[1].1 - 1.0 / 3.0).abs() < 1e-9, "{:?}", terms);
        assert!((rhs + 2.0 / 3.0).abs() < 1e-9, "rhs {}", rhs);

        // Adding it and re-solving must land the bound exactly on the
        // integer optimum: with 2x + y ≤ 4 the LP vertex moves to (1, 2),
        // which is integral with objective 7 — one cut closes this fixture's
        // whole integrality gap.
        assert_eq!(
            solver
                .add_constraint(coeffs, ComparisonOp::Ge, rhs)
                .unwrap(),
            StopReason::Finished
        );
        assert!(
            (solver.cur_obj_val + 7.0).abs() < 1e-6,
            "bound after cut: {}",
            solver.cur_obj_val
        );
    }

    /// The killer test for GMI soundness, mirroring the cover-cut
    /// enumeration test: on seeded random MIXED instances, every generated
    /// cut must hold at every mixed-feasible point. Integer assignments are
    /// enumerated outright; for each one the cut's minimum over the
    /// continuous completions is computed with a fresh LP (the simplex core
    /// is the trusted, suite-validated oracle here — the code under test is
    /// only the cut generation). Slack variables in the cut are substituted
    /// via their definition s_k = rhs_k − Σ a_k·x, turning the check into a
    /// pure-structural LP.
    #[test]
    fn gmi_cuts_never_cut_mixed_feasible_points() {
        init();
        let mut rng_state: u64 = 0x51ce_b00c_5eed;
        let mut rng = move || {
            rng_state = rng_state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let mut cuts_checked = 0usize;
        for _trial in 0..240 {
            let n_int = 2 + (rng() % 2) as usize; // 2..=3
            let n_cont = (rng() % 3) as usize; // 0..=2
            let n = n_int + n_cont;
            let int_hi = 1 + (rng() % 2) as u64; // boxes [0,1] or [0,2]
            let mins = vec![0.0; n];
            let mut maxs = Vec::new();
            let mut domains = Vec::new();
            for v in 0..n {
                if v < n_int {
                    maxs.push(int_hi as f64);
                    domains.push(VarDomain::Integer);
                } else {
                    maxs.push(3.0);
                    domains.push(VarDomain::Real);
                }
            }
            // Anchor point (kept feasible by rhs construction below).
            let anchor: Vec<f64> = (0..n)
                .map(|v| {
                    if v < n_int {
                        (rng() % (int_hi + 1)) as f64
                    } else {
                        (rng() % 4) as f64 * 0.75
                    }
                })
                .collect();
            let m = 2 + (rng() % 2) as usize;
            let mut rows = Vec::new();
            for _ in 0..m {
                let coeffs: Vec<f64> = (0..n)
                    .map(|_| {
                        if rng() % 5 == 0 {
                            0.0
                        } else {
                            ((rng() % 9) as i64 - 4) as f64
                        }
                    })
                    .collect();
                if coeffs.iter().all(|&c| c == 0.0) {
                    continue;
                }
                let at_anchor: f64 = coeffs.iter().zip(&anchor).map(|(c, a)| c * a).sum();
                let (op, rhs) = if rng() % 2 == 0 {
                    (ComparisonOp::Le, at_anchor + (rng() % 4) as f64)
                } else {
                    (ComparisonOp::Ge, at_anchor - (rng() % 4) as f64)
                };
                rows.push((to_sparse(&coeffs), op, rhs));
            }
            if rows.is_empty() {
                continue;
            }
            let obj: Vec<f64> = (0..n).map(|_| ((rng() % 7) as i64 - 3) as f64).collect();

            let mut solver = match Solver::try_new(&obj, &mins, &maxs, &rows, &domains, None) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if !matches!(solver.initial_solve(), Ok(StopReason::Finished)) {
                continue;
            }

            // Generate a cut from every eligible basic integer var.
            let mut generated: Vec<(Vec<(usize, f64)>, f64)> = Vec::new();
            for v in 0..n_int {
                let val = *solver.get_value(v);
                let frac = val - val.floor();
                if !(0.01..=0.99).contains(&frac) {
                    continue;
                }
                if let Some((coeffs, rhs)) = solver.gmi_cut(v, &domains) {
                    generated.push((coeffs.iter().map(|(i, &c)| (i, c)).collect(), rhs));
                }
            }
            if generated.is_empty() {
                continue;
            }

            // Substitute slacks out of each cut: s_k = rhs_k − Σ a_k·x.
            let structural_cuts: Vec<(Vec<f64>, f64)> = generated
                .iter()
                .map(|(terms, rhs_ge)| {
                    let mut c_struct = vec![0.0; n];
                    let mut constant = 0.0;
                    for &(tv, c) in terms {
                        if tv < n {
                            c_struct[tv] += c;
                        } else {
                            let k = tv - n;
                            let (row, _, row_rhs) = &rows[k];
                            constant += c * row_rhs;
                            for (sv, &a) in row.iter() {
                                c_struct[sv] -= c * a;
                            }
                        }
                    }
                    // Σ c_struct·x + constant ≥ rhs_ge must hold mixed-wide.
                    (c_struct, rhs_ge - constant)
                })
                .collect();
            cuts_checked += structural_cuts.len();

            // Enumerate integer assignments; LP-minimize each cut over the
            // continuous completions.
            let assignments = (int_hi + 1).pow(n_int as u32);
            for code in 0..assignments {
                let mut c = code;
                let mut fixed_mins = mins.clone();
                let mut fixed_maxs = maxs.clone();
                for fm in fixed_mins.iter_mut().take(n_int).zip(fixed_maxs.iter_mut()) {
                    let val = (c % (int_hi + 1)) as f64;
                    c /= int_hi + 1;
                    *fm.0 = val;
                    *fm.1 = val;
                }
                for (c_struct, rhs_ge) in &structural_cuts {
                    let mut oracle = match Solver::try_new(
                        c_struct,
                        &fixed_mins,
                        &fixed_maxs,
                        &rows,
                        &domains,
                        None,
                    ) {
                        Ok(s) => s,
                        Err(_) => continue, // infeasible assignment: vacuous
                    };
                    match oracle.initial_solve() {
                        Ok(StopReason::Finished) => {}
                        _ => continue,
                    }
                    // Infeasible assignments error out of initial_solve and
                    // were skipped above; a finite minimum below the cut's
                    // rhs is a feasible mixed point the cut wrongly excludes.
                    assert!(
                        oracle.cur_obj_val >= rhs_ge - 1e-7,
                        "GMI cut cuts a feasible completion: min {} < rhs {}\n\
                         rows {:?}\nassignment code {}",
                        oracle.cur_obj_val,
                        rhs_ge,
                        rows,
                        code,
                    );
                }
            }
        }
        assert!(
            cuts_checked > 30,
            "only {cuts_checked} GMI cuts generated across all trials"
        );
    }

    #[test]
    fn solve_integer_singular_var() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Ge, 90.0);
        assert!((problem.solve().unwrap().objective() - 3.0).abs() < EPS);

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Ge, 91.0);
        assert!((problem.solve().unwrap().objective() - 4.0).abs() < EPS);

        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Le, 90.0);
        assert!((problem.solve().unwrap().objective() - 3.0).abs() < EPS);

        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_integer_var(1.0, (0, 10));
        problem.add_constraint([(x, 30.0)], ComparisonOp::Le, 91.0);
        assert!((problem.solve().unwrap().objective() - 3.0).abs() < EPS);
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
        let sol = problem.solve().unwrap();
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
        assert_eq!(&sol.nb_var_obj_coeffs, &[3.2, 0.2]);
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
    fn basis_snapshot_load_roundtrip() {
        init();
        // NOTE: deviates from the task brief's literal fixture, which used obj [2.0, 1.0]
        // with var 0 in (-inf, 0] and var 1 in [5, inf) under x+y<=6, x+2y<=8. That LP is
        // unbounded (fix y at its min of 5: both constraints reduce to upper bounds on x,
        // and x has no lower bound, so x -> -inf drives 2x+y -> -inf), so
        // `initial_solve().unwrap()` panics with `Unbounded` before any basis code runs —
        // confirmed empirically and by hand. Swapped in the fixture from the adjacent
        // `initial_solve` test below (proven feasible+bounded, obj -68.0 at x=12, y=8,
        // with both structural vars basic and both slacks non-basic), so the round trip
        // exercises a non-trivial basis that actually differs from the slack basis.
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
