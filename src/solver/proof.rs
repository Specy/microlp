//! The proof behind every `Infeasible` the dual simplex returns: a
//! certificate of primal infeasibility checked on the model's own data, as
//! HiGHS checks one before its dual simplex may declare a model infeasible
//! (`HEkk::proofOfPrimalInfeasibility`, highs/simplex/HEkk.cpp, called from
//! `HEkkDual::assessPossiblyDualUnbounded`, highs/simplex/HEkkDual.cpp) and
//! SCIP checks the Farkas proof its LP solver hands it (`SCIPlpGetDualfarkas`
//! with `lp/checkfarkas`, scip/src/scip/lp.c).
//!
//! When the dual ratio test finds no var to repair a violated row, the row
//! of the basis inverse `rho = B^-T e_r` suggests a certificate: the rows
//! combined with the multipliers `rho` give one equation, `p^T z = beta`
//! over every var `z` (structural vars and slacks, basic or not), in which
//! the leaving var has the coefficient one and every other basic var zero.
//! If the largest value `p^T z` can take over the box of the bounds falls
//! short of `beta` (or the smallest exceeds it), no point satisfies the rows
//! and the bounds together. The multipliers come from the factorization, in
//! `f64`, refined twice and kept to twice that precision, with a bound on
//! the error each keeps ([`Solver::refined_basis_inverse_row`]), and are only
//! a guess; the certificate is valid for whatever multipliers it is built
//! with, so it is built from the model's data ([`Solver::orig_constraints`],
//! the right-hand sides and the bounds) in compensated arithmetic
//! ([`CDouble`]) and judged as a statement about the model, not about the
//! tableau. What the tableau row said (its entries
//! computed through the eta file and summed in `f64`, entries below a
//! round-off floor counted as zero) decides nothing. Every coefficient of
//! the certificate counts at its var's bound where that is finite, however
//! small, and makes the range unbounded where that is infinite; the one
//! concession is for a coefficient on an infinite bound that the error of
//! the multipliers could make zero, which counts as zero, as HiGHS and SCIP
//! count one below a fixed tolerance ([`Solver::proof_of`]).
//!
//! The certificate proves infeasibility when its gap exceeds the tolerance
//! the engine holds the leaving var to (`var_tols`, the one by which the
//! dual simplex found its row violated), with the other terms exact: HiGHS
//! compares its gap with its primal feasibility tolerance, the one its
//! simplex holds every var to. The proof then says that no point the
//! simplex could accept for that var exists, the same test as the one that
//! made the row a candidate to leave. A proof with any other tolerance
//! would leave a band where a row is violated for the simplex yet not
//! infeasible for the proof, and nothing that can act on it: with the
//! user's absolute tolerance as a floor (the engine's is half of it for a
//! row, and capped by its rows' reserve for a var), two MILPs of the
//! property stress failed loudly, one of them at a node that puts a var
//! `1e-8` past its bound, where their nodes had been pruned (HiGHS prunes
//! them too); with every term's contract tolerance summed, so that a
//! proof would say that no point within the contract exists, so did the
//! MIPs whose node rows disagree by less than their tolerances together
//! (`a_cleanup_dual_phase_shifts_a_boxed_var_rather_than_flipping_it` in
//! `src/tests/magnitudes.rs`), which HiGHS proves infeasible too. The price
//! is the engine's: a model whose only points lie between the engine's
//! share of the contract and the contract itself is reported infeasible
//! (a row missed by `6e-8`, beyond half of the `1e-7` contract), as it was
//! before the proof; under the full contract of R4 the engine's tolerance
//! is the contract's, and so is the proof's. The engine scales rows and
//! columns by powers of two, so the scaled data the proof reads are the
//! user's data exactly, and the tolerances, converted into the scaled units
//! as everywhere in the engine, are the user's.

use super::{Solver, ROUNDOFF_FLOOR};

/// A compensated double: the unevaluated sum `hi + lo` of two `f64`, which
/// carries about twice the precision of one (HiGHS's `HighsCDouble`,
/// highs/util/HighsCDouble.h, after Rump, "High precision evaluation of
/// nonlinear functions", 2005). Only accumulation is needed here: a sum of
/// exact products, whose error is about `n · 2^-104` of the sum of the
/// terms' magnitudes rather than `n · 2^-53` of it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct CDouble {
    hi: f64,
    lo: f64,
}

impl CDouble {
    /// `a + b = x + y` exactly, with `x = fl(a + b)` (Knuth's two-sum).
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let x = a + b;
        let z = x - a;
        (x, (a - (x - z)) + (b - z))
    }

    /// `a = x + y` exactly, each half with at most 26 significant bits
    /// (Dekker's split).
    fn split(a: f64) -> (f64, f64) {
        const FACTOR: f64 = 134_217_729.0; // 2^27 + 1
        let c = FACTOR * a;
        let x = c - (c - a);
        (x, a - x)
    }

    /// `a · b = x + y` exactly, with `x = fl(a · b)` (Dekker's two-product,
    /// without a fused multiply-add, so the result is the same on every
    /// target, WASM included).
    fn two_product(a: f64, b: f64) -> (f64, f64) {
        let x = a * b;
        let (a1, a2) = Self::split(a);
        let (b1, b2) = Self::split(b);
        (x, a2 * b2 - (((x - a1 * b1) - a2 * b1) - a1 * b2))
    }

    /// `self += v` (`HighsCDouble::operator+=(double)`).
    fn add(&mut self, v: f64) {
        let (hi, c) = Self::two_sum(v, self.hi);
        self.hi = hi;
        self.lo += c;
    }

    /// `self += a · b`, the product taken exactly.
    pub(super) fn add_product(&mut self, a: f64, b: f64) {
        let (p, e) = Self::two_product(a, b);
        self.add(p);
        self.lo += e;
    }

    /// `self += c · b` for a compensated `c`: its high part's product exact,
    /// its low part's rounded (a relative error of about `2^-106`).
    pub(super) fn add_scaled(&mut self, c: CDouble, b: f64) {
        self.add_product(c.hi, b);
        self.lo += c.lo * b;
    }

    /// `-self`, exactly.
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    /// The nearest `f64`.
    pub(super) fn value(self) -> f64 {
        self.hi + self.lo
    }
}

impl From<f64> for CDouble {
    fn from(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }
}

/// The allowance for the error of the compensated sums, per term and per
/// unit of the terms' magnitude: sixteen times `2^-104`. The data are exact
/// and so are the products (but for those of a compensated multiplier's low
/// part, rounded at `2^-106` of the term); what is left is the error of
/// accumulating the low parts in `f64`. That grows with the square of the
/// count of terms in the worst case, `n^2 · 2^-106` of their magnitude for
/// Ogita, Rump and Oishi's `Dot2` ("Accurate sum and dot product", 2005),
/// which these sums are, so this covers the worst case only up to 64 terms
/// and beyond is an allowance for the typical error, not a bound. Either way
/// it matters only where the magnitude of the terms is some `1e22` times
/// the tolerance.
const COMPENSATED_ERROR: f64 = 16.0 * 4.930_380_657_631_324e-32; // 16 · 2^-104

/// LAPACK's test of a converged refinement: a correction of at most
/// `EPS = DLAMCH('Epsilon')`, `2^-53`, of the solution in the max norm
/// (`DX_X <= EPS`, `DLA_GERFSX_EXTENDED`). Only a refinement whose second
/// correction passes it bounds the error of the multipliers
/// ([`multipliers_error`]): a solve that leaves more is too inaccurate to
/// measure its own error, and a bound it gave could make a coefficient of
/// the model's own count as zero.
const REFINEMENT_CONVERGED: f64 = f64::EPSILON / 2.0;

/// The bound on the error of each multiplier of the refined row `rho` that
/// the refinement's `second` correction gives
/// ([`Solver::refine_basis_inverse_row`]): that correction, where the
/// refinement converged, the correction at most [`REFINEMENT_CONVERGED`] of
/// the row in the max norm (zero for a row solved exactly); `None` where it
/// did not, or where the row or the correction is not finite.
pub(super) fn multipliers_error(rho: &[CDouble], second: &[f64]) -> Option<Vec<f64>> {
    let rho_norm = finite_max_norm(rho.iter().map(|r| r.value()))?;
    let second_norm = finite_max_norm(second.iter().copied())?;
    (second_norm <= REFINEMENT_CONVERGED * rho_norm)
        .then(|| second.iter().map(|c| c.abs()).collect())
}

/// The largest magnitude of `values`; `None` if one is not finite.
fn finite_max_norm(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut norm = 0.0f64;
    for value in values {
        if !value.is_finite() {
            return None;
        }
        norm = norm.max(value.abs());
    }
    Some(norm)
}

/// What [`Solver::prove_infeasible`] found, for the log and the tests.
#[derive(Clone, Copy, Debug)]
pub(super) struct Proof {
    /// Whether the certificate proves the model infeasible.
    pub(super) holds: bool,
    /// How far `beta` lies beyond the range `p^T z` can take over the box of
    /// the bounds (positive: out of reach), in the leaving var's scaled
    /// units; `-inf` when a var with an infinite bound in the certificate's
    /// direction makes the range unbounded, and NaN when a number of the
    /// certificate is not finite, which voids it.
    pub(super) gap: f64,
    /// The leaving var's tolerance the gap must exceed (see the module doc),
    /// plus the allowance for the error of the compensated sums
    /// ([`COMPENSATED_ERROR`]).
    pub(super) allowance: f64,
}

/// The row of the basis inverse the proof reads, refined, with a bound on
/// the error each of its multipliers keeps
/// ([`Solver::refined_basis_inverse_row`]).
pub(super) struct RefinedRow {
    /// Per row, its multiplier, compensated.
    pub(super) rho: Vec<CDouble>,
    /// Per row, a bound on the error of its multiplier, its last correction;
    /// `None` where the refinement did not converge, which leaves no bound,
    /// so that only the round-off of the certificate's own sums counts as
    /// zero ([`Solver::proof_of`]).
    pub(super) error: Option<Vec<f64>>,
}

/// The certificate the row of the basis inverse suggests for a violated
/// row (see the module doc): `p^T z = beta`, summed from the model's data.
struct Certificate {
    /// Per total var, its coefficient `p`, compensated.
    coeffs: Vec<CDouble>,
    /// Per total var, the magnitude `Σ |y_i a_ij|` its coefficient was
    /// summed from (see [`Certificate::is_round_off`]), and how many terms.
    magnitudes: Vec<f64>,
    terms_of: Vec<u32>,
    /// Per total var, how far the error of the multipliers can carry its
    /// coefficient, `Σ e_i |a_ij|` over the rows the certificate keeps (`e_i`
    /// the bound of [`RefinedRow::error`]); `None` where there is no bound.
    errors: Option<Vec<f64>>,
    beta: CDouble,
    /// The magnitude of every term summed so far, and their count, for the
    /// allowance for the error of the compensated sums.
    magnitude: f64,
    terms: usize,
    /// How many rows the certificate combines.
    rows: usize,
}

impl Certificate {
    /// The coefficient of `var`, to the nearest `f64`.
    fn coeff(&self, var: usize) -> f64 {
        self.coeffs[var].value()
    }

    /// Whether the coefficient `p` of `var` is round-off: at most
    /// `ROUNDOFF_FLOOR` of the magnitude of the terms it was summed from,
    /// below which the engine trusts nothing it computes from such terms in
    /// `f64` (`tolerances.rs`), the tableau entry of `var` included. The
    /// certificate is exact for its multipliers all the same, so this says
    /// what the rest of the engine can see of the coefficient, not whether
    /// it is there ([`Solver::round_off_repair`]).
    fn is_round_off(&self, var: usize, p: f64) -> bool {
        p.abs() <= ROUNDOFF_FLOOR * self.magnitudes[var]
    }

    /// How far from zero the coefficient of `var` can come out where it is
    /// zero for the exact multipliers: what the error of the multipliers
    /// carries into it, plus the error of its compensated sum
    /// ([`COMPENSATED_ERROR`] per term); where the multipliers' error has no
    /// bound, the error of the sum alone. A coefficient within that counts as
    /// zero on an infinite bound, and nothing the model's data leave above it
    /// can be one (see [`Solver::proof_of`]).
    fn error_of_zero(&self, var: usize) -> f64 {
        let multipliers_error = self.errors.as_ref().map_or(0.0, |errors| errors[var]);
        let sum_error =
            (f64::from(self.terms_of[var]) + 1.0) * COMPENSATED_ERROR * self.magnitudes[var];
        multipliers_error + sum_error
    }
}

impl Solver {
    /// The row of the basis inverse for `row`, `rho = B^-T e_row`, solved
    /// for on the current factorization as [`Self::calc_row_coeffs`] solves
    /// for it (the same transposed solve, so the same row as the ratio test
    /// read), not taken from `inv_basis_row_coeffs`, which holds whatever
    /// row was computed last, and refined ([`Self::refine_basis_inverse_row`]).
    pub(super) fn refined_basis_inverse_row(&mut self, row: usize) -> RefinedRow {
        let mut rho = vec![CDouble::default(); self.num_constraints()];
        let solved = self.basis_solver.solve_transp(std::iter::once((row, &1.0)));
        for (r, &value) in solved.iter() {
            rho[r] = CDouble::from(value);
        }
        self.refine_basis_inverse_row(row, rho)
    }

    /// `rho`, the row of the basis inverse for `row` as solved for, refined
    /// twice, with a bound on the error each multiplier keeps.
    ///
    /// Each step is HiGHS's `HEkk::unitBtranIterativeRefinement` (and
    /// `unitBtranResidual`, highs/simplex/HEkk.cpp), which HiGHS applies to a
    /// pivot row whose pivot it doubts, and which its proof offers too (off
    /// by default): the residual `B^T rho - e_row`, summed in compensated
    /// arithmetic over the basic columns, is solved for and taken off
    /// ([`Self::basis_inverse_row_correction`]). In exact arithmetic a basic
    /// var's certificate coefficient is zero (it is that residual's entry),
    /// but through an ill-conditioned basis the computed residual exceeds any
    /// fixed fraction of the terms it is summed from, and times a huge bound
    /// it alone can close the certificate's gap: a basic var's `8.4e-16`,
    /// `1.8e-14` of its terms, times its scaled bound of `3.6e8` closed an
    /// infeasible MILP node's gap of `3.05e-7` (found by the property
    /// stress).
    ///
    /// The corrections are taken off exactly, into the trailing bits of a
    /// compensated row, as LAPACK's extra-precise iterative refinement keeps
    /// its solution in doubled precision (`DLA_GERFSX_EXTENDED`, which adds
    /// each correction into `(Y, Y_TAIL)` with `DLA_WWADDW`), where HiGHS
    /// rounds the refined row to `f64`. Rounded, the row keeps the error of
    /// its own representation, about `1e-17` of the terms of each basic
    /// column, whatever the solve achieved, and times a scaled bound of `1e9`
    /// to `6e10` that error closed the gaps of three infeasible MILP nodes of
    /// the property stress, whose rows no `f64` multipliers combine exactly
    /// (their multipliers are fifths and tenths). Kept, the basic
    /// coefficients are the residual the solves leave.
    ///
    /// The second step is also what bounds the error of the multipliers, as
    /// LAPACK bounds the error of its refined solution by its last
    /// correction (`ERR_BNDS_NORM`, `FINAL_DX_X / (1 - DXRATMAX)`): the
    /// second correction is the error the first step left, measured, and
    /// once it is taken off each multiplier is taken to be within it (see
    /// [`Self::proof_of`] for what that bound decides). One step leaves the
    /// residuals the next would correct: in the big-M bases of
    /// `issue_48_switchable_dc_network_solves`, where the first correction
    /// reaches `2.6e-3` on multipliers of `7.4e6`, it leaves basic
    /// coefficients of up to `1.6e-26` of their terms, and the second
    /// `1e-33`. The bound holds only where the refinement converged, in
    /// LAPACK's sense ([`REFINEMENT_CONVERGED`]); a second correction above
    /// that leaves no bound. Dense.
    pub(super) fn refine_basis_inverse_row(
        &mut self,
        row: usize,
        mut rho: Vec<CDouble>,
    ) -> RefinedRow {
        let Some(first) = self.basis_inverse_row_correction(row, &rho) else {
            return RefinedRow { rho, error: None };
        };
        for (r, correction) in rho.iter_mut().zip(&first) {
            r.add(-correction);
        }
        let Some(second) = self.basis_inverse_row_correction(row, &rho) else {
            return RefinedRow { rho, error: None };
        };
        for (r, correction) in rho.iter_mut().zip(&second) {
            r.add(-correction);
        }
        let error = multipliers_error(&rho, &second);
        RefinedRow { rho, error }
    }

    /// The correction of one step of [`Self::refine_basis_inverse_row`]:
    /// `B^-T (B^T rho - e_row)`, the residual summed in compensated
    /// arithmetic over the basic columns and scaled by a power of two into
    /// `[1, 2)` for the solve, as HiGHS scales it, which is exact; zero where
    /// the residual is, and `None` where it is not finite.
    fn basis_inverse_row_correction(&mut self, row: usize, rho: &[CDouble]) -> Option<Vec<f64>> {
        let mut residual = vec![0.0; self.num_constraints()];
        let mut norm = 0.0f64;
        for (i, res) in residual.iter_mut().enumerate() {
            let mut value = CDouble::default();
            if i == row {
                value.add(-1.0);
            }
            //guaranteed to be a valid index
            let col = self
                .orig_constraints_csc
                .outer_view(self.basic_vars[i])
                .unwrap();
            for (r, &a) in col.iter() {
                value.add_scaled(rho[r], a);
            }
            *res = value.value();
            if !res.is_finite() {
                return None;
            }
            norm = norm.max(res.abs());
        }
        if norm == 0.0 {
            return Some(residual);
        }
        let scale = 2f64.powi(-(norm.log2().floor() as i32));
        for res in &mut residual {
            *res *= scale;
        }
        self.basis_solver
            .solve_transp_dense_with_etas(&mut residual);
        for res in &mut residual {
            *res /= scale;
        }
        Some(residual)
    }

    /// The certificate of the multipliers of `refined` (one per row) for a
    /// leaving var that is to move up to its lower bound (`leaving_up`) or
    /// down to its upper one (see [`Self::proof_of`]).
    fn certificate(&self, refined: &RefinedRow, leaving_up: bool) -> Certificate {
        let total = self.num_total_vars();
        let mut cert = Certificate {
            coeffs: vec![CDouble::default(); total],
            magnitudes: vec![0.0; total],
            terms_of: vec![0; total],
            errors: refined.error.as_ref().map(|_| vec![0.0; total]),
            beta: CDouble::default(),
            magnitude: 0.0,
            terms: 0,
            rows: 0,
        };
        for (r, &rho) in refined.rho.iter().enumerate() {
            let y = if leaving_up { rho.neg() } else { rho };
            let y_value = y.value();
            let slack = self.num_vars + r;
            let needs_infinite_side = if y_value > 0.0 {
                self.orig_var_maxs[slack] == f64::INFINITY
            } else {
                self.orig_var_mins[slack] == f64::NEG_INFINITY
            };
            if y_value == 0.0 || needs_infinite_side {
                continue;
            }
            cert.rows += 1;
            cert.beta.add_scaled(y, self.orig_rhs[r]);
            cert.magnitude += (y_value * self.orig_rhs[r]).abs();
            cert.terms += 1;
            let error = refined.error.as_ref().map(|error| error[r]);
            //guaranteed to be a valid index
            for (var, &a) in self.orig_constraints.outer_view(r).unwrap().iter() {
                cert.coeffs[var].add_scaled(y, a);
                cert.magnitudes[var] += (y_value * a).abs();
                cert.terms_of[var] += 1;
                cert.terms += 1;
                if let (Some(errors), Some(error)) = (&mut cert.errors, error) {
                    errors[var] += error * a.abs();
                }
            }
        }
        cert
    }

    /// Whether the row of the basis inverse for `row`, whose basic var is to
    /// move up to its lower bound (`leaving_up`) or down to its upper one,
    /// proves the model infeasible (see the module doc): the row, on a
    /// fresh factorization, as HiGHS proves only after a fresh rebuild,
    /// solved for and refined ([`Self::refined_basis_inverse_row`]), then
    /// judged by [`Self::proof_of`].
    pub(super) fn prove_infeasible(&mut self, row: usize, leaving_up: bool) -> Proof {
        let refined = self.refined_basis_inverse_row(row);
        self.proof_of(&refined, row, leaving_up)
    }

    /// Whether the multipliers of `refined` prove the model infeasible for
    /// the violated `row` (see [`Self::prove_infeasible`]).
    ///
    /// As `HEkk::proofOfPrimalInfeasibility`: the multipliers take the sign
    /// that makes the leaving var's coefficient `+1` when it lies above its
    /// upper bound (`-1` below its lower one), so the certificate fails to
    /// hold exactly when its largest value over the box reaches `beta`; a
    /// multiplier whose row would need its slack's infinite bound is set to
    /// zero, which leaves a certificate of the other rows (HiGHS drops such
    /// rows: "make sure infinite sides are not used"); and the combination
    /// `p = y^T [A I]` and `beta = y^T b` are summed in compensated
    /// arithmetic from the model's data (HiGHS's `productTransposeQuad` and
    /// `HighsCDouble` sums).
    ///
    /// A coefficient counts as it is wherever its var's bound in the
    /// certificate's direction is finite, however far below its terms it
    /// cancelled: the certificate is exact for its multipliers, and such a
    /// coefficient can be the data's own. `x + y + z + q >= 3` and
    /// `t - (1 + 2^-50) z - q >= 1e-6`, combined with ones, leave `z` the
    /// coefficient `-2^-50`, and with `z >= -2e9` it makes room for the rows
    /// (`x = y = t = 1`, `z = -2e9` is feasible); counted as zero, as HiGHS
    /// counts the entries of its proof row below an absolute `1e-14` and
    /// SCIP's `lp/checkfarkas` those within its dual feasibility tolerance,
    /// it proved that feasible model infeasible. A basic var's coefficient,
    /// zero for the exact row of the basis inverse, counts so too: what the
    /// refinement leaves of it is part of the certificate at a finite bound
    /// (unrefined, a basic var's `1.45e-17` times its scaled bound of `4.2e12`
    /// closed an infeasible MILP node's gap of `6.1e-5`; refined, such a
    /// coefficient is zero or the refinement's noise).
    ///
    /// On an infinite bound a coefficient makes the range unbounded and the
    /// proof void, however small it is, unless the error of the multipliers
    /// could make it zero ([`Certificate::error_of_zero`]): the bound
    /// the refinement puts on the error of each multiplier
    /// ([`Self::refine_basis_inverse_row`]), carried through the var's
    /// entries in the rows the certificate keeps, plus the error of its
    /// compensated sum. Such a coefficient counts as zero, the one
    /// concession: it covers what the multipliers' error leaves of a
    /// coefficient that is zero for the exact multipliers, and nothing
    /// above that. A basic var's is one (for the exact row of the basis
    /// inverse its column combines to zero), and so is a non-basic var's
    /// whose tableau entry is zero, and any whose rows' exact multipliers are
    /// zero. The differential fuzz has the first two: its refined rows leave
    /// vars on infinite bounds residuals of up to `1.5e-32` of their terms,
    /// all basic but one, the non-basic `x3` of `int_lp-1145`, whose entry is
    /// zero and which the second correction, spread over the row by its
    /// solve, leaves `3.6e-33` of its terms; without the concession 41
    /// solves of 28 of its models failed loudly (38 on infeasible models, 3
    /// at MILP nodes).
    /// The switchable networks of `src/tests/regressions.rs` have the third:
    /// a free bus angle whose rows have multipliers that are zero for the
    /// exact row, and come out as the refinement's own noise, gets that noise
    /// times its entries, all of its terms, so that no floor relative to its
    /// terms can cover it.
    ///
    /// No fixed threshold tells such a residual from a coefficient the data
    /// leave. HiGHS counts every coefficient on an infinite bound as zero
    /// while their sum stays below its `small_matrix_value` of `1e-9`, and
    /// with `z` unbounded below that proved the model above infeasible. A
    /// concession for basic vars below `ROUNDOFF_FLOOR` of their terms proved
    /// a feasible model infeasible through a row left out: in
    /// `L + j + w = 1.000001`, `-(1 + 2^-50) j - w = 0` and `j + v <= 0` over
    /// `L, v ∈ [0, 1]` and `j, w` free, feasible at `L = 1` and
    /// `j ≈ -1.13e9`, the basis of `j`, `w` and the second row's slack has
    /// the row of its inverse `(1, 1, 2^-50)`, exact, whose third multiplier,
    /// the entry of that row's non-basic slack, needs the slack's infinite
    /// side; the third row is left out, and the free basic `j` keeps the
    /// model's own `-2^-50`, `4.4e-16` of its terms, which was counted as
    /// zero (and the model's MILP pruned at its root node). With the error
    /// bound it counts: that row is exact, the bound is the round-off of the
    /// compensated sums, and the range is unbounded. Floors below that fail
    /// the other way: one at the round-off of the doubled precision, `1e-28`
    /// of the terms, on the row refined once (whose basic coefficients reach
    /// `1.6e-26` of their terms in the big-M bases of
    /// `issue_48_switchable_dc_network_solves`) failed that model's node
    /// proofs, and the round-off of the compensated sums alone, on the row
    /// refined twice, which the fuzz never exceeds, failed 12 of 480 solves
    /// of random switchable networks like issue 48's, where the error bound
    /// fails none. The bound follows the basis: some `1e-30` of the terms in
    /// the fuzz, up to `1.3e-19` in issue 48's big-M bases, the noise itself
    /// where a coefficient's terms are all noise. The concession would still
    /// prove a feasible model infeasible through a coefficient the data leave
    /// that lies within that bound, which the arithmetic cannot tell from
    /// zero, and only where the var times it makes the room the gap lacks.
    ///
    /// The error of a multiplier is taken to be at most its own last
    /// correction, rather than, as LAPACK's normwise bound has it, the largest
    /// of all: in a big-M basis of issue 48 whose multipliers reach `1.4e3`
    /// and whose largest second correction is `2.9e-29`, that largest,
    /// carried through the entries of a var whose rows have multipliers near
    /// `7e-16`, would allow `3.9e-14` of its terms, ninety times the `-2^-50`
    /// above, where its own corrections allow `1.3e-19`. A multiplier whose
    /// error exceeded its correction would only have a bound too small, which
    /// fails a proof (a loud error), never makes one.
    ///
    /// A number of the certificate that is not finite (an overflow: a huge
    /// multiplier or coefficient, or Dekker's split of one beyond about
    /// `1.3e300`) voids it too, as non-finite data fail the dual ratio test.
    /// HiGHS also zeroes multipliers below an absolute `1e-9` of their row;
    /// no multiplier needs that here, since any multipliers give a valid
    /// certificate and the infinite sides are left out by sign.
    pub(super) fn proof_of(&self, refined: &RefinedRow, row: usize, leaving_up: bool) -> Proof {
        let cert = self.certificate(refined, leaving_up);
        let (mut magnitude, mut terms) = (cert.magnitude, cert.terms);
        // Whether every number of the certificate is finite (see above).
        let mut finite = cert.beta.value().is_finite();
        // The largest value of `p^T z` over the box.
        let mut implied_max = CDouble::default();
        let mut unbounded = false;
        // How many coefficients count as zero (see above), for the log.
        let mut zero_within_error = 0;
        for var in 0..self.num_total_vars() {
            let p = cert.coeff(var);
            if p == 0.0 {
                continue;
            }
            if !p.is_finite() || !cert.magnitudes[var].is_finite() {
                finite = false;
                break;
            }
            let bound = if p > 0.0 {
                self.orig_var_maxs[var]
            } else {
                self.orig_var_mins[var]
            };
            if bound.is_infinite() {
                // The concession (see above), or no finite range; an error
                // bound that is not finite voids the proof.
                let error = cert.error_of_zero(var);
                if !error.is_finite() {
                    finite = false;
                    break;
                }
                if p.abs() <= error {
                    zero_within_error += 1;
                    continue;
                }
                unbounded = true;
                break;
            }
            implied_max.add_scaled(cert.coeffs[var], bound);
            // The coefficient's own error, relative to the terms it was
            // summed from, not to what they left, is carried by the bound.
            magnitude += cert.magnitudes[var] * bound.abs();
            terms += 1;
        }
        let gap = if !finite {
            f64::NAN
        } else if unbounded {
            f64::NEG_INFINITY
        } else {
            let mut gap = cert.beta;
            gap.add_scaled(implied_max, -1.0);
            // A sum that overflowed voids the proof too.
            let gap = gap.value();
            if gap.is_finite() {
                gap
            } else {
                f64::NAN
            }
        };
        // The tolerance the engine holds the leaving var to at the bound it
        // violates (see the module doc), plus the allowance for the error of
        // the compensated sums.
        let leaving = self.basic_vars[row];
        let allowance = self.var_tols[leaving].toward(!leaving_up)
            + (terms as f64 + 1.0) * COMPENSATED_ERROR * magnitude;
        let proof = Proof {
            holds: gap > allowance,
            gap,
            allowance,
        };
        debug!(
            "infeasibility proof for row {row} (basic var {leaving}, {} rows combined, {} \
             coefficients zero within the multipliers' error{}): gap {:e}, allowance {:e}: {}",
            cert.rows,
            zero_within_error,
            if cert.errors.is_some() {
                ""
            } else {
                ", which has no bound"
            },
            proof.gap,
            proof.allowance,
            if proof.holds { "proven" } else { "not proven" },
        );
        proof
    }

    /// The last resort of the dual simplex for a violated `row` that no var
    /// can repair, that does not prove the model infeasible, and on which
    /// SoPlex's instable leave made no pivot (see [`Self::dual_phase`]):
    /// the non-basic vars whose entries are below round-off and that can
    /// close the gap move, flipped to their other bound, or (the last one,
    /// if a flip would carry the leaving var past its other bound) only
    /// partway, where it stays non-basic between its bounds. The moves are
    /// applied through the vars' columns, and kept only if the leaving var
    /// then lands within its tolerance, as the certificate says it does;
    /// returns whether it did (the moves are undone if not).
    ///
    /// The moves are read off the certificate, whose failure to hold says
    /// that the bounds of its vars leave room for the row, not off the
    /// tableau row: a var takes part only where its coefficient is finite
    /// and not round-off ([`Certificate::is_round_off`]), in the direction it
    /// helps and with a finite bound there, and how far it carries the
    /// leaving var is taken to be its coefficient times its move. That is
    /// its tableau entry times its move only where the certificate keeps
    /// every row of the var: a row the certificate leaves out (its
    /// multiplier needs its slack's infinite side, see [`Self::proof_of`])
    /// takes its term from the coefficient but not from the entry, and a
    /// move computed from such a coefficient misses the bound; the landing
    /// check, on the values the moves give through the columns, then undoes
    /// the moves, and the row is left to the loud error. The proof counts a
    /// round-off coefficient on a finite bound, since the room it makes may
    /// be real ([`Self::proof_of`]); a move cannot rely on it, since the rest
    /// of the engine does not see such an entry (the round-off floors of the
    /// ratio tests hide it), and the primal step takes the move back: moved
    /// by its `-2^-50`, the far-bound model of
    /// `src/tests/magnitudes.rs` puts `z` at `-2e9`, the primal step lifts it
    /// again, and the phases alternate until `MAX_PHASE_ROUNDS`. A row
    /// whose room lies only there is left to the loud error of
    /// [`Self::dual_phase`].
    ///
    /// No reference has this. HiGHS refuses such entries and fails on the
    /// models that need it without presolve, and SoPlex flips such a var
    /// only together with a stable pivot, which their rows lack (the
    /// `wide_chain` fixtures of `src/tests/magnitudes.rs`, whose coefficient
    /// chains span twelve to fifteen orders of magnitude and whose only
    /// repair moves a var through an entry of `1.5e-15` beside `1.95`, to its
    /// other bound or only partway).
    pub(super) fn round_off_repair(&mut self, row: usize) -> bool {
        let refined = self.refined_basis_inverse_row(row);
        let leaving_new_val = self.leaving_target(row);
        let Some(moves) = self.round_off_moves(&refined, row, leaving_new_val) else {
            return false;
        };
        let undo: Vec<(usize, f64)> = moves
            .iter()
            .map(|&(c, _)| (c, self.nb_var_vals[c]))
            .collect();
        self.apply_flips(&moves);
        if !self.basic_var_within_bounds(row) {
            self.apply_flips(&undo);
            return false;
        }
        trace!(
            "dual round-off repair: {} vars moved, var {} of row {row} lands at {:e}",
            moves.len(),
            self.basic_vars[row],
            self.basic_var_vals[row],
        );
        for &(c, _) in &moves {
            // No dual step went with the move, so the var's reduced cost may
            // have the wrong sign where it landed: its cost is shifted until
            // it has the right one, and the cleanup gives the shift back.
            self.shift_to_dual_feasible(c);
        }
        true
    }

    /// The moves of [`Self::round_off_repair`] that bring the basic var of
    /// `row` to `leaving_new_val`, within its tolerance, read off the
    /// certificate of the multipliers of `refined`; `None` if the vars that
    /// can help cannot.
    fn round_off_moves(
        &self,
        refined: &RefinedRow,
        row: usize,
        leaving_new_val: f64,
    ) -> Option<Vec<(usize, f64)>> {
        let val = self.basic_var_vals[row];
        let leaving_up = leaving_new_val > val;
        let cert = self.certificate(refined, leaving_up);
        let tols = self.var_tols[self.basic_vars[row]];
        // How far the leaving var must move, the tolerance within which it
        // counts as there, and how far past it a move may carry it (to its
        // other bound, within that bound's tolerance).
        let need = (val - leaving_new_val).abs();
        let (target_tol, room_past_target) = if leaving_up {
            (
                tols.lo,
                self.basic_var_maxs[row] - leaving_new_val + tols.hi,
            )
        } else {
            (
                tols.hi,
                leaving_new_val - self.basic_var_mins[row] + tols.lo,
            )
        };
        // The vars that can help, as (col, coefficient, bound it moves to,
        // how far moving there carries the leaving var).
        let mut helpers: Vec<(usize, f64, f64, f64)> = self
            .nb_vars
            .iter()
            .enumerate()
            .filter_map(|(c, &var)| {
                let p = cert.coeff(var);
                if !p.is_finite() || cert.is_round_off(var, p) {
                    return None;
                }
                // A var helps by moving up where its coefficient is
                // positive, down where it is negative.
                let state = &self.nb_var_states[c];
                let (blocked, bound) = if p > 0.0 {
                    (state.at_max, self.orig_var_maxs[var])
                } else {
                    (state.at_min, self.orig_var_mins[var])
                };
                let capacity = p.abs() * (bound - self.nb_var_vals[c]).abs();
                (!blocked && capacity.is_finite() && capacity > 0.0)
                    .then_some((c, p, bound, capacity))
            })
            .collect();
        helpers.sort_by(|a, b| b.3.total_cmp(&a.3));
        let mut moved = 0.0;
        let mut moves = Vec::new();
        for (c, p, bound, capacity) in helpers {
            let rest = need - moved;
            if rest <= target_tol {
                break;
            }
            if capacity <= rest + room_past_target {
                moved += capacity;
                moves.push((c, bound));
            } else {
                let cur = self.nb_var_vals[c];
                let step = rest / p.abs();
                moves.push((c, if bound > cur { cur + step } else { cur - step }));
                moved = need;
            }
        }
        (need - moved <= target_tol).then_some(moves)
    }
}
