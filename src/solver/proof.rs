//! The proof behind every `Infeasible` the dual simplex returns: a
//! certificate of primal infeasibility (a Farkas proof) checked on the
//! model's own data.
//!
//! When the dual ratio test finds no var to repair a violated row, the row of
//! the basis inverse `rho = B^-T e_r` suggests multipliers: the rows combined
//! with them give one equation `p^T z = beta` over every var `z` (structural
//! vars and slacks), in which the leaving var has the coefficient one and
//! every other basic var zero. If the largest value `p^T z` can take over the
//! box of the bounds falls short of `beta` (or the smallest exceeds it), no
//! point satisfies the rows and the bounds together. The multipliers are only
//! a guess ([`Solver::refined_basis_inverse_row`]), but the certificate is
//! valid for whatever multipliers it is built with, so it is summed from the
//! model's data in compensated arithmetic ([`CDouble`]) and the tableau row
//! decides nothing. Rows and columns are scaled by powers of two, so those
//! data are the user's exactly.
//!
//! Every coefficient counts at the bound of its var that raises `p^T z` where
//! that bound is finite, however small the coefficient, and makes the range
//! unbounded where it is infinite, with one concession for the error of the
//! multipliers ([`Solver::proof_of`]).
//!
//! The certificate proves infeasibility when its gap exceeds the room the
//! engine can give its terms, plus the round-off allowance of the compensated
//! sums: the leaving var the tolerance by which the dual simplex found the
//! row violated (`var_tols`: for a slack, its row's contract less the
//! round-off of evaluating the row, and less the cut of
//! [`Solver::leaving_slack_cut`]; for a structural var, its whole bound
//! contract, none for an integer var its bounds fix), and every other term
//! the room by which the widening can move its bound in the direction
//! that helps ([`Solver::widening_room`]: a structural var's tolerance, a
//! slack's less the evaluation margin the widened side keeps, none for a
//! bound already widened or a var the engine keeps on its bounds), each up
//! to the farthest value `f64` holds within it, times its coefficient. So a
//! proof says that no point has its terms within the room the engine can
//! reach, and a row the proof leaves open is the last resorts', the last of
//! which ([`Solver::certificate_repair`]) can widen the same bounds by the
//! same room: rows that agree only within their tolerances together are
//! solved, not reported infeasible. The price: a model whose points all need
//! more room than that is reported infeasible.

use super::{band_edge, implied_slack_error, slack_margin, Solver, ROUNDOFF_FLOOR, UNIT_ROUNDOFF};

/// A compensated double: the unevaluated sum `hi + lo` of two `f64`, which
/// carries about twice the precision of one. The proof accumulates products
/// with it ([`compensated_error`] is the allowance for their round-off),
/// presolve the bound of a row it rewrites, and the test oracle a row's
/// residual.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CDouble {
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

    /// `self += v`.
    pub(crate) fn add(&mut self, v: f64) {
        let (hi, c) = Self::two_sum(v, self.hi);
        self.hi = hi;
        self.lo += c;
    }

    /// `self += a · b`, the product taken exactly.
    pub(crate) fn add_product(&mut self, a: f64, b: f64) {
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

    /// `self`, exactly, its high part the nearest `f64` and its low part
    /// within half an ulp of it.
    pub(super) fn normalized(self) -> Self {
        let (hi, lo) = Self::two_sum(self.hi, self.lo);
        Self { hi, lo }
    }

    /// The nearest `f64` at or above `|self|`.
    fn abs_up(self) -> f64 {
        if self.value() < 0.0 { self.neg() } else { self }.rounded(true)
    }

    /// The nearest `f64`.
    pub(crate) fn value(self) -> f64 {
        self.hi + self.lo
    }

    /// The nearest `f64` at or above the sum (`up`), or at or below it.
    pub(crate) fn rounded(self, up: bool) -> f64 {
        let (value, error) = Self::two_sum(self.hi, self.lo);
        if up && error > 0.0 {
            value.next_up()
        } else if !up && error < 0.0 {
            value.next_down()
        } else {
            value
        }
    }
}

impl From<f64> for CDouble {
    fn from(hi: f64) -> Self {
        Self { hi, lo: 0.0 }
    }
}

/// `a + b`, rounded up: the nearest `f64` at or above the exact sum.
fn add_up(a: f64, b: f64) -> f64 {
    let (sum, error) = CDouble::two_sum(a, b);
    if error > 0.0 {
        sum.next_up()
    } else {
        sum
    }
}

/// `a · b` for `a, b >= 0`, rounded up.
fn mul_up(a: f64, b: f64) -> f64 {
    let (product, error) = CDouble::two_product(a, b);
    if error > 0.0 {
        product.next_up()
    } else {
        product
    }
}

/// `|a - b|`, rounded up.
fn distance_up(a: f64, b: f64) -> f64 {
    let (hi, lo) = CDouble::two_sum(a, -b);
    CDouble { hi, lo }.abs_up()
}

/// The allowance for the round-off of compensated sums, per term and per unit
/// of the terms' magnitude, where no sum has more than six terms:
/// `16 · 2^-104` ([`compensated_error`]). It matters only where the terms are
/// some `1e22` times the tolerance.
const COMPENSATED_ERROR: f64 = 16.0 * 4.930_380_657_631_324e-32; // 16 · 2^-104

/// The allowance for the round-off of compensated sums of `terms` terms in
/// all, the longest of `longest` terms, over terms of `magnitude` in all:
/// `(terms + 1) max(1, (longest + 2) / 8)` times [`COMPENSATED_ERROR`] times
/// `magnitude`. A sum adds three low parts per term (the two-sum error of a
/// product's high part, its low part, and the rounded product of a
/// multiplier's low part, which [`CDouble::normalized`] keeps within half an
/// ulp of its high part) and accumulates them in `f64`, their partial sums
/// growing by an ulp of the magnitude per term: a sum of `n` terms errs by at
/// most `(1.5 n^2 + 7.5 n + 1) 2^-106` of its magnitude, to first order, and a
/// certificate's gap, which adds each coefficient's error and low part times
/// its bound and has the leaving var's room taken off, by at most
/// `(6 n^2 + 24 n + 14) 2^-106` of its magnitude, `n` its longest sum. The
/// allowance bounds both whatever the count.
pub(super) fn compensated_error(terms: usize, longest: usize, magnitude: f64) -> f64 {
    let scale = ((longest as f64 + 2.0) / 8.0).max(1.0);
    (terms as f64 + 1.0) * scale * COMPENSATED_ERROR * magnitude
}

/// The test of a converged refinement: a correction of at most `2^-53` of the
/// solution in the max norm. Only a refinement whose second correction passes
/// it bounds the error of its solution ([`refinement_error`]): a solve that
/// leaves more is too inaccurate to measure its own error, and a bound it gave
/// could make a coefficient or an entry of the model's own count as zero.
const REFINEMENT_CONVERGED: f64 = f64::EPSILON / 2.0;

/// The bound on the error of each entry of a solution refined twice, `x`:
/// the magnitude of its entry of the refinement's `second` correction, where
/// the refinement converged ([`REFINEMENT_CONVERGED`]); `None` where it did
/// not, or where the solution or the correction is not finite. The solutions
/// are the row of the basis inverse the proof reads, whose entries are the
/// multipliers ([`Solver::refine_basis_inverse_row`]), and a pivot column of
/// the primal ratio test (`Solver::refine_pivot_col`). Each entry gets its own
/// correction, not the largest of all, which carried through the entries of a
/// var whose rows have multipliers far below the rest would let its
/// coefficient count as zero far above the error those multipliers carry. An
/// entry whose error exceeds its correction only has too small a bound, which
/// can fail a proof, or bound a step that an exact zero would not, but never
/// make a proof or an unbounded ray.
pub(super) fn refinement_error(x: &[CDouble], second: &[f64]) -> Option<Vec<f64>> {
    let x_norm = finite_max_norm(x.iter().map(|r| r.value()))?;
    let second_norm = finite_max_norm(second.iter().copied())?;
    (second_norm <= REFINEMENT_CONVERGED * x_norm).then(|| second.iter().map(|c| c.abs()).collect())
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
    /// The room of the certificate's terms the gap must exceed (see the
    /// module doc), plus the allowance for the error of the compensated sums
    /// ([`compensated_error`]), rounded up.
    pub(super) allowance: f64,
    /// The round-off withheld from the room of a leaving slack
    /// ([`Solver::leaving_slack_cut`]); zero for a structural var, and where
    /// the gap is not finite.
    pub(super) cut: f64,
    /// How many coefficients counted as zero within their error (the
    /// concession of [`Solver::proof_of`]).
    pub(super) zero_within_error: usize,
}

/// The row of the basis inverse the proof reads, refined, with a bound on
/// the error each of its multipliers keeps
/// ([`Solver::refined_basis_inverse_row`]).
pub(super) struct RefinedRow {
    /// Per row, its multiplier, compensated.
    pub(super) rho: Vec<CDouble>,
    /// Per row, a bound on the error of its multiplier
    /// ([`refinement_error`]); `None` where the refinement did not converge,
    /// and then a coefficient counts as zero only within the round-off of the
    /// certificate's own sums ([`Solver::proof_of`]).
    pub(super) error: Option<Vec<f64>>,
    /// Per row, the second correction, taken off its multiplier; empty where
    /// a correction is not finite. Carried through a column's entries, it is
    /// how far the second step moved that column's coefficient, against which
    /// the dual ratio test weighs the coefficient (`Solver::refine_pivot_row`).
    pub(super) second: Vec<f64>,
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
    /// `ROUNDOFF_FLOOR` of the magnitude of the terms it was summed from.
    /// That says what the rest of the engine can see of the coefficient (its
    /// tableau entry included), not whether it is there: the certificate is
    /// exact for its multipliers ([`Solver::certificate_repair`]).
    fn is_round_off(&self, var: usize, p: f64) -> bool {
        p.abs() <= ROUNDOFF_FLOOR * self.magnitudes[var]
    }

    /// How far from zero the coefficient of `var` can come out where it is
    /// zero for the exact multipliers: what the error of the multipliers
    /// carries into it, plus the round-off of its compensated sum
    /// ([`compensated_error`]), or that round-off alone where the
    /// multipliers' error has no bound. On an infinite bound a coefficient
    /// within this counts as zero ([`Solver::proof_of`]).
    fn error_of_zero(&self, var: usize) -> f64 {
        let multipliers_error = self.errors.as_ref().map_or(0.0, |errors| errors[var]);
        let terms = self.terms_of[var] as usize;
        multipliers_error + compensated_error(terms, terms, self.magnitudes[var])
    }
}

impl Solver {
    /// The row of the basis inverse for `row`, `rho = B^-T e_row`, solved for
    /// on the current factorization by the same transposed solve as
    /// [`Self::calc_row_coeffs`] (so the row the ratio test read, not
    /// whatever row `inv_basis_row_coeffs` holds): dense, and compensated for
    /// a refinement's corrections to go into.
    fn basis_inverse_row(&mut self, row: usize) -> Vec<CDouble> {
        let mut rho = vec![CDouble::default(); self.num_constraints()];
        let solved = self.basis_solver.solve_transp(std::iter::once((row, &1.0)));
        for (r, &value) in solved.iter() {
            rho[r] = CDouble::from(value);
        }
        rho
    }

    /// The row of the basis inverse for `row` ([`Self::basis_inverse_row`]),
    /// refined ([`Self::refine_basis_inverse_row`]).
    pub(super) fn refined_basis_inverse_row(&mut self, row: usize) -> RefinedRow {
        let rho = self.basis_inverse_row(row);
        self.refine_basis_inverse_row(row, rho)
    }

    /// `rho`, the row of the basis inverse for `row` as solved for, refined
    /// twice, with a bound on the error each multiplier keeps.
    ///
    /// Each step solves for the residual `B^T rho - e_row`, summed in
    /// compensated arithmetic over the basic columns, and takes the solution
    /// off ([`Self::basis_inverse_row_correction`]). In exact arithmetic a
    /// basic var's certificate coefficient is zero (it is that residual's
    /// entry), but through an ill-conditioned basis the computed residual
    /// exceeds any fixed fraction of the terms it is summed from, and times a
    /// huge bound it alone can close the certificate's gap. The corrections
    /// are taken off exactly, into the trailing bits of the compensated row:
    /// rounded to `f64`, the row would keep the error of its own
    /// representation (about `1e-17` of each basic column's terms, whatever
    /// the solve achieved), which times a huge bound can close the gap too.
    ///
    /// The second correction is the error the first step left, measured: once
    /// it is taken off, each multiplier is taken to be within it
    /// ([`refinement_error`]), where the refinement converged
    /// ([`REFINEMENT_CONVERGED`]); [`Self::proof_of`] says what that bound
    /// decides. A single step would leave residuals the second removes, and
    /// no measure of its own error
    /// (`an_infeasible_switchable_network_whose_proof_needs_the_twice_refined_row`
    /// in `src/tests/regressions.rs` needs both). Both steps are dense.
    pub(super) fn refine_basis_inverse_row(
        &mut self,
        row: usize,
        mut rho: Vec<CDouble>,
    ) -> RefinedRow {
        let broken = |rho| RefinedRow {
            rho,
            error: None,
            second: Vec::new(),
        };
        let Some(first) = self.basis_inverse_row_correction(row, &rho) else {
            return broken(rho);
        };
        for (r, correction) in rho.iter_mut().zip(&first) {
            r.add(-correction);
        }
        let Some(second) = self.basis_inverse_row_correction(row, &rho) else {
            return broken(rho);
        };
        for (r, correction) in rho.iter_mut().zip(&second) {
            r.add(-correction);
        }
        let error = refinement_error(&rho, &second);
        RefinedRow { rho, error, second }
    }

    /// The correction of one step of [`Self::refine_basis_inverse_row`]:
    /// `B^-T (B^T rho - e_row)`, the residual summed in compensated
    /// arithmetic over the basic columns and scaled for the solve, exactly,
    /// by the power of two that puts its largest entry in `[1, 2)`; zero
    /// where the residual is, and `None` where it is not finite.
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
            let y = if leaving_up { rho.neg() } else { rho }.normalized();
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
    /// proves the model infeasible (see the module doc). The caller runs it
    /// on a fresh factorization, on which the row is solved for and refined
    /// ([`Self::refined_basis_inverse_row`]); [`Self::proof_of`] judges it.
    pub(super) fn prove_infeasible(&mut self, row: usize, leaving_up: bool) -> Proof {
        let refined = self.refined_basis_inverse_row(row);
        self.proof_of(&refined, row, leaving_up)
    }

    /// Whether the multipliers of `refined` prove the model infeasible for
    /// the violated `row` (see [`Self::prove_infeasible`]).
    ///
    /// The multipliers take the sign that makes the leaving var's coefficient
    /// `+1` when it lies above its upper bound (`-1` below its lower one), so
    /// the certificate fails to hold exactly when its largest value over the
    /// box reaches `beta`. A multiplier whose row would need its slack's
    /// infinite bound is set to zero, which leaves a certificate of the other
    /// rows.
    ///
    /// Where its var's bound in the certificate's direction is finite, a
    /// coefficient counts as it is, however far below its terms it cancelled:
    /// the certificate is exact for its multipliers, and such a coefficient
    /// can be the model's own, and the only room a feasible model has
    /// (`room_through_a_round_off_coefficient` in `src/tests/magnitudes.rs`).
    /// So does a basic var's coefficient, zero for the exact multipliers:
    /// what the refinement leaves of it is part of the certificate.
    ///
    /// Where that bound is infinite, a coefficient makes the range unbounded
    /// and the proof void, however small it is, unless it lies within
    /// [`Certificate::error_of_zero`]: the refinement's bound on each
    /// multiplier's error carried through the var's entries in the rows the
    /// certificate keeps, plus the round-off of its compensated sum (that
    /// round-off alone where the refinement did not converge). Such a
    /// coefficient counts as zero, the one concession. It covers what the
    /// multipliers' error leaves of a coefficient that is zero for the exact
    /// multipliers (a basic var's, a non-basic var's whose tableau entry is
    /// zero, or one whose rows' exact multipliers are zero), and nothing
    /// above that. No fixed threshold tells such a residual from a
    /// coefficient the data leave: where a row of its column is left out, a
    /// basic var keeps the model's own coefficient, however small beside its
    /// terms (`a_basic_vars_coefficient_in_a_row_left_out_voids_the_proof`),
    /// and a var whose rows' exact multipliers are zero has nothing but the
    /// refinement's noise for terms. The concession can still prove a
    /// feasible model infeasible through a coefficient the data leave within
    /// the error bound, which the arithmetic cannot tell from zero, and only
    /// where the var times it makes the room the gap lacks. So can a
    /// coefficient within the round-off of its sum of zero whose computed sign
    /// picks a finite bound where its exact sign picks an infinite one: it
    /// counts at the finite bound, where the exact certificate is void.
    ///
    /// A number of the certificate that is not finite (an overflow: a huge
    /// multiplier or coefficient, or Dekker's split of one beyond about
    /// `1.3e300`) voids it, with a NaN gap. The gap must exceed the room of
    /// every term at the bound it counts at (the leaving var's tolerance, a
    /// slack's less its cut, any other's widening room) plus the round-off
    /// allowance of the sums (see the module doc).
    pub(super) fn proof_of(&self, refined: &RefinedRow, row: usize, leaving_up: bool) -> Proof {
        let cert = self.certificate(refined, leaving_up);
        let proof = self.judge(&cert, row);
        debug!(
            "infeasibility proof for row {row} (basic var {}, {} rows combined, {} \
             coefficients zero within the multipliers' error{}): gap {:e}, allowance {:e} \
             (cut {:e}): {}",
            self.basic_vars[row],
            cert.rows,
            proof.zero_within_error,
            if cert.errors.is_some() {
                ""
            } else {
                ", which has no bound"
            },
            proof.gap,
            proof.allowance,
            proof.cut,
            if proof.holds { "proven" } else { "not proven" },
        );
        proof
    }

    /// [`Self::proof_of`] on the certificate `cert` of the violated `row`.
    fn judge(&self, cert: &Certificate, row: usize) -> Proof {
        let leaving = self.basic_vars[row];
        let (mut magnitude, mut terms) = (cert.magnitude, cert.terms);
        // Whether every number of the certificate is finite (see above).
        let mut finite = cert.beta.value().is_finite();
        // The largest value of `p^T z` over the box.
        let mut implied_max = CDouble::default();
        // How far past that the room of the terms but the leaving var carries
        // it, and the leaving var's room, which ends at `leaving_edge`, each
        // rounded up from the exact one.
        let mut room = 0.0;
        let (mut leaving_room, mut leaving_edge) = (0.0, 0.0);
        let mut unbounded = false;
        // How many coefficients count as zero (see above), for the log.
        let mut zero_within_error = 0;
        // The most terms a sum of the gap has: `beta`'s, a coefficient's,
        // `implied_max`'s.
        let (mut longest, mut bounded) = (cert.rows, 0);
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
            // The farthest value of the var's room: the leaving var's
            // tolerance past its bound, any other's widening room, none for a
            // var the engine keeps on its bounds.
            let up = p > 0.0;
            let edge = if var == leaving {
                leaving_edge = band_edge(bound, self.var_tols[var].toward(up), up);
                leaving_edge
            } else if self.holds_on_bounds(var) {
                bound
            } else {
                band_edge(bound, self.widening_room(var, up), up)
            };
            let width = distance_up(edge, bound);
            let reach = mul_up(cert.coeffs[var].abs_up(), width);
            if var == leaving {
                leaving_room = reach;
            } else {
                room = add_up(room, reach);
            }
            // The coefficient's own error, relative to the terms it was
            // summed from, not to what they left, is carried by the farthest
            // value of its var.
            magnitude += cert.magnitudes[var] * (bound.abs() + width);
            terms += 1;
            bounded += 1;
            longest = longest.max(cert.terms_of[var] as usize).max(bounded);
        }
        // `beta` less the largest value of `p^T z` over the box, compensated.
        let gap_sum = (finite && !unbounded).then(|| {
            let mut gap = cert.beta;
            gap.add_scaled(implied_max, -1.0);
            gap
        });
        let gap = match gap_sum {
            Some(sum) if sum.value().is_finite() => sum.value(),
            // A sum that overflowed voids the proof too.
            Some(_) => f64::NAN,
            None if unbounded => f64::NEG_INFINITY,
            None => f64::NAN,
        };
        let sums = compensated_error(terms, longest, magnitude);
        let cut = if leaving >= self.num_vars && gap.is_finite() {
            self.leaving_slack_cut(row, leaving_edge, sums, gap)
        } else {
            0.0
        };
        // Every term's room (see the module doc), a leaving slack's less its
        // cut, plus the round-off allowance of the compensated sums, weighed
        // so that rounding errs against the proof: the rooms rounded up, the
        // leaving var's taken off the compensated gap, which is then rounded
        // down, and the rest, less the cut, rounded up. A proof then holds
        // only where the exact gap exceeds the exact rooms. Of the leaving
        // var's room only its own rounding up enters the verdict, an ulp at
        // most, none where its coefficient is one and its bound zero, as a
        // slack's in a certificate of its row alone
        // ([`Self::leaving_slack_cut`]).
        let others = add_up(room, sums);
        let (holds, allowance) = match gap_sum {
            Some(mut excess) if gap.is_finite() && leaving_room > cut => {
                let rest = add_up(others, -cut);
                excess.add(-leaving_room);
                let holds = gap > others && excess.rounded(false) > rest;
                (holds, add_up(rest, leaving_room))
            }
            _ => (gap > others, others),
        };
        Proof {
            holds,
            gap,
            allowance,
            cut,
            zero_within_error,
        }
    }

    /// The round-off `E` the proof of `row` withholds from the room of its
    /// leaving slack, whose room ends at `edge`, for a certificate whose sums
    /// have the allowance `sums` and whose gap came out `gap`:
    /// `E = min(4 u min(|s|, |edge|) + 4 (n + 2)^2 u^2 m + 2 sums + u |gap|, δ)`,
    /// `s` the engine's value of the slack, `n`, `m` and `δ` the terms,
    /// magnitude and evaluation margin ([`slack_margin`]) of its row at the
    /// current values, `u = 2^-53`.
    ///
    /// The engine found the slack beyond its edge: by a plain sum, more than
    /// twice the row's evaluation margin past it, so the exact slack lies past
    /// it too, or by a compensated one ([`Self::row_slack`]), which errs by at
    /// most the first two terms ([`implied_slack_error`]). They take the slack
    /// at most at the edge: a value less its error, `s - u |s|`, grows with
    /// `s`, so a compensated value past the edge puts the exact slack past it
    /// less the error there. The gap errs by `sums`, which the allowance counts
    /// once more; the verdict weighs the compensated gap, so `u |gap|`, the
    /// rounding of its value, is spare. So where the certificate is the slack's
    /// row alone, as when no basic structural var lies in that row, the
    /// computed gap of a slack the engine finds violated exceeds its room, its
    /// tolerance `α` less `E`, plus `sums`: the proof holds unless other terms'
    /// room closes the gap, which the repair then spends
    /// ([`Self::certificate_repair`], whose target for the slack is `α - E`),
    /// and no such row is left neither reached nor proven. The cap leaves that
    /// so: the errors the cut covers there come to at most `2 u m` and terms in
    /// `u^2`, within `δ`. It keeps the slack at least `α - δ`, the room every
    /// other slack of a certificate gets, where a certificate of several rows
    /// has an allowance beyond its row's round-off. The price: the proof can
    /// hold where a point needs the slack within `E` of the edge of its room.
    fn leaving_slack_cut(&self, row: usize, edge: f64, sums: f64, gap: f64) -> f64 {
        let r = self.basic_vars[row] - self.num_vars;
        let (_, magnitude) = self.row_activity(r, |v| self.internal_value(v));
        let (tol, terms) = (self.feasibility * self.row_scales[r], self.row_terms(r));
        let slack = self.basic_var_vals[row].abs().min(edge.abs());
        let cut =
            implied_slack_error(slack, magnitude, terms) + 2.0 * sums + UNIT_ROUNDOFF * gap.abs();
        cut.min(slack_margin(tol, magnitude, terms))
    }

    /// The last of the dual simplex's last resorts for a violated `row` that
    /// no var can repair and that does not prove the model infeasible (see
    /// [`Self::dual_phase`]): the non-basic vars that can close the gap and
    /// sit at their bound in the direction that helps move past it. Largest
    /// reach first, the bound each sits at is widened by its room
    /// ([`Self::widening_room`]) for the rest of the solve ([`Self::widen`])
    /// and the var moves onto it, until the leaving var reaches its target or
    /// none is left; a widened var stays at the bound it sat at, dual
    /// feasible, and the dual simplex goes on over the bounds in force. That
    /// is the room the proof grants every term but the leaving var
    /// ([`Self::proof_of`]), so a row it leaves open has a widening until the
    /// room of every var that can help it is spent, and a phase widens
    /// finitely often. The target is the leaving var's tolerance, a slack's
    /// less the proof's cut ([`Self::leaving_slack_cut`]): where the engine's
    /// value of the slack is compensated, near its edge, the cut covers the
    /// round-off of predicting the slack from it and of the engine's next
    /// verdict on it. Returns whether it widened a bound.
    ///
    /// The widening is the room of a row every var that could move sits at
    /// the bound that blocks, short of its bound by a tolerance, alone or with
    /// other rows (`the_repair_moves_vars_past_their_bounds_within_their_tolerances`).
    /// A var that could move toward its other bound is the dual ratio test's,
    /// down to the smallest genuine entry of a refined row
    /// ([`Self::genuine_entry_step`]), the last resort before this. A var the
    /// engine keeps on its bounds ([`Self::holds_on_bounds`]) is never
    /// widened: an integer var's bounds are where presolve's rewrite of a
    /// big-M row holds exactly (past them, the row as written is tighter than
    /// the rewritten one by the cut coefficient times the move), and a fixed
    /// structural var's value is where presolve checks the rows it
    /// substituted it out of.
    ///
    /// The moves are read off the certificate, not off the tableau row: a var
    /// takes part only where its coefficient is finite and not round-off
    /// beside its own terms ([`Certificate::is_round_off`]), in the direction
    /// it helps and with a finite bound there, and it carries the leaving var
    /// by its coefficient times its move. That is its tableau entry times its
    /// move only where the certificate keeps every row of the var; where it
    /// does not, a widening leaves the row to the dual simplex.
    pub(super) fn certificate_repair(&mut self, row: usize) -> bool {
        let refined = self.refined_basis_inverse_row(row);
        let leaving_new_val = self.leaving_target(row);
        let val = self.basic_var_vals[row];
        let leaving_up = leaving_new_val > val;
        let cert = self.certificate(&refined, leaving_up);
        let tols = self.var_tols[self.basic_vars[row]];
        // How far the leaving var must move, and the tolerance within which it
        // counts as there (see above).
        let need = (val - leaving_new_val).abs();
        let tol = if leaving_up { tols.lo } else { tols.hi };
        let target_tol = (tol - self.judge(&cert, row).cut).max(0.0);
        // The vars that can help past the bound they sit at, as (col, var,
        // up, the widened bound, how far its move carries the leaving var). A
        // var helps by moving up where its coefficient is positive, down where
        // it is negative.
        let mut band: Vec<(usize, usize, bool, f64, f64)> = self
            .nb_vars
            .iter()
            .enumerate()
            .filter_map(|(c, &var)| {
                let p = cert.coeff(var);
                if !p.is_finite() || cert.is_round_off(var, p) || self.holds_on_bounds(var) {
                    return None;
                }
                let up = p > 0.0;
                let state = &self.nb_var_states[c];
                let (at_bound, bound) = if up {
                    (state.at_max, self.orig_var_maxs[var])
                } else {
                    (state.at_min, self.orig_var_mins[var])
                };
                let room = self.widening_room(var, up);
                if !at_bound || !bound.is_finite() || room <= 0.0 {
                    return None;
                }
                let edge = band_edge(bound, room, up);
                let reach = p * (edge - self.nb_var_vals[c]);
                (reach > 0.0 && reach.is_finite()).then_some((c, var, up, edge, reach))
            })
            .collect();
        band.sort_by(|a, b| b.4.total_cmp(&a.4));
        let mut moved = 0.0;
        let mut moves = Vec::new();
        for (c, var, up, edge, reach) in band {
            if need - moved <= target_tol {
                break;
            }
            moved += reach;
            self.widen(var, up, edge);
            moves.push((c, edge));
        }
        if moves.is_empty() {
            return false;
        }
        self.apply_flips(&moves);
        trace!(
            "dual certificate repair: vars moved onto widened bounds {:?}, var {} of row {row} at \
             {:e}",
            moves
                .iter()
                .map(|&(c, val)| (self.nb_vars[c], val))
                .collect::<Vec<_>>(),
            self.basic_vars[row],
            self.basic_var_vals[row],
        );
        true
    }
}
