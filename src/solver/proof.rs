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
//! round-off of evaluating the row; for a structural var, its whole bound
//! contract, none for an integer var its bounds fix), and every other term
//! the room by which the last resort can widen its bound in the direction
//! that helps ([`Solver::widening_room`]: a structural var's tolerance, a
//! slack's less the evaluation margin the widened side keeps, none for a
//! bound already widened or a var the engine keeps on its bounds), each up
//! to the farthest value `f64` holds within it, times its coefficient. So a
//! proof says that no point has its terms within the room the engine can
//! reach, and a row the proof leaves open is the last resort's
//! ([`Solver::certificate_repair`]), which can widen the same bounds by the
//! same room: rows that agree only within their tolerances together are
//! solved, not reported infeasible. The price: a model whose points all need
//! more room than that is reported infeasible.

use super::{band_edge, Solver, ROUNDOFF_FLOOR};

/// A compensated double: the unevaluated sum `hi + lo` of two `f64`, which
/// carries about twice the precision of one. The proof accumulates products
/// with it ([`COMPENSATED_ERROR`] is the allowance for their round-off),
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

/// The allowance for the round-off of the compensated sums, per term and per
/// unit of the terms' magnitude: `16 · 2^-104`. The products are exact but
/// for a compensated multiplier's low part (rounded at `2^-106` of the term);
/// accumulating the low parts in `f64` errs by up to `n^2 · 2^-106` of the
/// terms' magnitude (`Dot2`, Ogita–Rump–Oishi 2005), so this is a bound up to
/// 64 terms and an allowance for the typical error beyond. It matters only
/// where the terms are some `1e22` times the tolerance.
const COMPENSATED_ERROR: f64 = 16.0 * 4.930_380_657_631_324e-32; // 16 · 2^-104

/// The test of a converged refinement: a correction of at most `2^-53` of the
/// solution in the max norm. Only a refinement whose second correction passes
/// it bounds the error of the multipliers ([`multipliers_error`]): a solve
/// that leaves more is too inaccurate to measure its own error, and a bound
/// it gave could make a coefficient of the model's own count as zero.
const REFINEMENT_CONVERGED: f64 = f64::EPSILON / 2.0;

/// The bound on the error of each multiplier of the refined row `rho`: the
/// magnitude of its entry of the refinement's `second` correction
/// ([`Solver::refine_basis_inverse_row`]), where the refinement converged
/// ([`REFINEMENT_CONVERGED`]); `None` where it did not, or where the row or
/// the correction is not finite. Each multiplier gets its own correction, not
/// the largest of all, which carried through the entries of a var whose rows
/// have multipliers far below the rest would let its coefficient count as
/// zero far above the error those multipliers carry. A multiplier whose error
/// exceeds its correction only has too small a bound, which can fail a proof
/// but never make one.
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
    /// The room of the certificate's terms the gap must exceed (see the
    /// module doc), plus the allowance for the error of the compensated sums
    /// ([`COMPENSATED_ERROR`]).
    pub(super) allowance: f64,
}

/// The row of the basis inverse the proof reads, refined, with a bound on
/// the error each of its multipliers keeps
/// ([`Solver::refined_basis_inverse_row`]).
pub(super) struct RefinedRow {
    /// Per row, its multiplier, compensated.
    pub(super) rho: Vec<CDouble>,
    /// Per row, a bound on the error of its multiplier
    /// ([`multipliers_error`]); `None` where the refinement did not converge,
    /// and then a coefficient counts as zero only within the round-off of the
    /// certificate's own sums ([`Solver::proof_of`]).
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
    /// ([`COMPENSATED_ERROR`] per term), or that round-off alone where the
    /// multipliers' error has no bound. On an infinite bound a coefficient
    /// within this counts as zero ([`Solver::proof_of`]).
    fn error_of_zero(&self, var: usize) -> f64 {
        let multipliers_error = self.errors.as_ref().map_or(0.0, |errors| errors[var]);
        let sum_error =
            (f64::from(self.terms_of[var]) + 1.0) * COMPENSATED_ERROR * self.magnitudes[var];
        multipliers_error + sum_error
    }
}

impl Solver {
    /// The row of the basis inverse for `row`, `rho = B^-T e_row`, solved for
    /// on the current factorization by the same transposed solve as
    /// [`Self::calc_row_coeffs`] (so the row the ratio test read, not
    /// whatever row `inv_basis_row_coeffs` holds), and refined
    /// ([`Self::refine_basis_inverse_row`]).
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
    /// ([`multipliers_error`]), where the refinement converged
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
    /// where the var times it makes the room the gap lacks.
    ///
    /// A number of the certificate that is not finite (an overflow: a huge
    /// multiplier or coefficient, or Dekker's split of one beyond about
    /// `1.3e300`) voids it, with a NaN gap. The gap must exceed the room of
    /// every term at the bound it counts at (the leaving var's tolerance, any
    /// other's widening room) plus the round-off allowance of the sums (see
    /// the module doc).
    pub(super) fn proof_of(&self, refined: &RefinedRow, row: usize, leaving_up: bool) -> Proof {
        let cert = self.certificate(refined, leaving_up);
        let leaving = self.basic_vars[row];
        let (mut magnitude, mut terms) = (cert.magnitude, cert.terms);
        // Whether every number of the certificate is finite (see above).
        let mut finite = cert.beta.value().is_finite();
        // The largest value of `p^T z` over the box.
        let mut implied_max = CDouble::default();
        // How far past that the terms' room carries it.
        let mut room = 0.0;
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
            if var == leaving {
                let tol = self.var_tols[var].toward(p > 0.0);
                room += p.abs() * (band_edge(bound, tol, p > 0.0) - bound).abs();
            } else if !self.holds_on_bounds(var) {
                let reach = self.widening_room(var, p > 0.0);
                room += p.abs() * (band_edge(bound, reach, p > 0.0) - bound).abs();
            }
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
        // Every term's room (see the module doc), plus the round-off
        // allowance of the compensated sums.
        let allowance = room + (terms as f64 + 1.0) * COMPENSATED_ERROR * magnitude;
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
    /// can repair, not even by a pivot on an unstable entry, and that does
    /// not prove the model infeasible (see [`Self::dual_phase`]): the
    /// non-basic vars that can close the gap move. First they flip to their
    /// bound in the direction that helps (the last one, if a flip would carry
    /// the leaving var past its other bound, only partway, where it stays
    /// non-basic between its bounds), kept, if `flips_allowed`, when that
    /// lands the leaving var within its tolerance, computed through their
    /// columns. Otherwise, largest reach first, the bound each sits at in
    /// that direction is widened by its room ([`Self::widening_room`]) for
    /// the rest of the solve ([`Self::widen`]) and the var moves onto it,
    /// until the leaving var reaches its target or none is left; a widened
    /// var stays at the bound it sat at, dual feasible, and the dual simplex
    /// goes on over the bounds in force. That is the room the proof grants
    /// every term but the leaving var ([`Self::proof_of`]), so a row it
    /// leaves open has a widening until the room of every var that can help
    /// it is spent, and a phase widens finitely often.
    ///
    /// The `wide_chain` fixtures of `src/tests/magnitudes.rs` need the
    /// flips: their only repair moves a var through an entry of `1.5e-15`
    /// beside `1.95`, below every pivot floor. The widening is the room of a
    /// row every var that could move sits at the bound that blocks, short of
    /// its bound by a tolerance, alone or with other rows
    /// (`the_repair_moves_vars_past_their_bounds_within_their_tolerances`).
    /// A var the engine keeps on its bounds ([`Self::holds_on_bounds`]) is
    /// never widened: an integer var's bounds are where presolve's rewrite of
    /// a big-M row holds exactly (past them, the row as written is tighter
    /// than the rewritten one by the cut coefficient times the move), and a
    /// fixed structural var's value is where presolve checks the rows it
    /// substituted it out of.
    ///
    /// The moves are read off the certificate, not off the tableau row: a var
    /// takes part only where its coefficient is finite and not round-off
    /// beside its own terms ([`Certificate::is_round_off`]), in the direction
    /// it helps and with a finite bound there, and it carries the leaving var
    /// by its coefficient times its move. That is its tableau entry times its
    /// move only where the certificate keeps every row of the var; where it
    /// does not, a flip misses and is undone, and a widening leaves the row to
    /// the dual simplex. A round-off coefficient, which the proof counts on a
    /// finite bound, makes no move: the round-off floors of the ratio tests
    /// hide such an entry from the rest of the engine, so the primal step
    /// would take the move back and the phases would alternate until
    /// `MAX_PHASE_ROUNDS`.
    pub(super) fn certificate_repair(&mut self, row: usize, flips_allowed: bool) -> Repair {
        let refined = self.refined_basis_inverse_row(row);
        let leaving_new_val = self.leaving_target(row);
        let val = self.basic_var_vals[row];
        let leaving_up = leaving_new_val > val;
        let cert = self.certificate(&refined, leaving_up);
        let tols = self.var_tols[self.basic_vars[row]];
        // How far the leaving var must move, the tolerance within which it
        // counts as there, and how far past it a flip may carry it (to its
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
        // whether it sits there already). A var helps by moving up where its
        // coefficient is positive, down where it is negative.
        let helpers: Vec<(usize, f64, f64, bool)> = self
            .nb_vars
            .iter()
            .enumerate()
            .filter_map(|(c, &var)| {
                let p = cert.coeff(var);
                if !p.is_finite() || cert.is_round_off(var, p) {
                    return None;
                }
                let state = &self.nb_var_states[c];
                let (at_bound, bound) = if p > 0.0 {
                    (state.at_max, self.orig_var_maxs[var])
                } else {
                    (state.at_min, self.orig_var_mins[var])
                };
                bound.is_finite().then_some((c, p, bound, at_bound))
            })
            .collect();
        let mut moves = Moves {
            need,
            target_tol,
            room_past_target,
            moved: 0.0,
            list: Vec::new(),
        };
        // First to the bounds.
        moves.take(
            helpers
                .iter()
                .filter(|h| !h.3)
                .map(|&(c, p, bound, _)| (c, p, self.nb_var_vals[c], bound))
                .collect(),
        );
        if moves.landed() {
            if !flips_allowed {
                return Repair::Nothing;
            }
            let undo: Vec<(usize, f64)> = moves
                .list
                .iter()
                .map(|&(c, _)| (c, self.nb_var_vals[c]))
                .collect();
            self.apply_flips(&moves.list);
            if !self.basic_var_within_bounds(row) {
                self.apply_flips(&undo);
                return Repair::Nothing;
            }
            self.finish_repair(row, &moves.list, 0);
            return Repair::Flipped;
        }
        // Then past them: largest reach first, each var's bound in the
        // direction it helps widened by its room, with the var on it.
        let mut band: Vec<(usize, usize, bool, f64, f64)> = helpers
            .iter()
            .filter_map(|&(c, p, bound, _)| {
                let var = self.nb_vars[c];
                let up = p > 0.0;
                let room = self.widening_room(var, up);
                if self.holds_on_bounds(var) || room <= 0.0 {
                    return None;
                }
                let edge = band_edge(bound, room, up);
                // Every var a flip took sits at its bound.
                let from = if moves.list.iter().any(|&(col, _)| col == c) {
                    bound
                } else {
                    self.nb_var_vals[c]
                };
                let reach = p * (edge - from);
                (reach > 0.0 && reach.is_finite()).then_some((c, var, up, edge, reach))
            })
            .collect();
        band.sort_by(|a, b| b.4.total_cmp(&a.4));
        let mut widened = Vec::new();
        for (c, var, up, edge, reach) in band {
            if moves.landed() {
                break;
            }
            moves.moved += reach;
            moves.set(c, edge);
            widened.push((var, up, edge));
        }
        if widened.is_empty() {
            return Repair::Nothing;
        }
        for &(var, up, edge) in &widened {
            self.widen(var, up, edge);
        }
        self.apply_flips(&moves.list);
        self.finish_repair(row, &moves.list, widened.len());
        Repair::Widened
    }

    /// Log the certificate repair's `moves` of `row` (see
    /// [`Self::certificate_repair`]), `widened` of them onto widened bounds,
    /// and shift the costs a flip leaves dual infeasible.
    fn finish_repair(&mut self, row: usize, moves: &[(usize, f64)], widened: usize) {
        trace!(
            "dual certificate repair: vars moved {:?} ({widened} onto widened bounds), var {} of \
             row {row} at {:e}",
            moves
                .iter()
                .map(|&(c, val)| (self.nb_vars[c], val))
                .collect::<Vec<_>>(),
            self.basic_vars[row],
            self.basic_var_vals[row],
        );
        for &(c, _) in moves {
            // No dual step went with a flip, so the var's reduced cost may
            // have the wrong sign where it landed: its cost is shifted until
            // it has the right one, and the cleanup gives the shift back. A
            // widened var stays at the bound it sat at, which needs none.
            self.shift_to_dual_feasible(c);
        }
    }
}

/// What [`Solver::certificate_repair`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Repair {
    /// Its flips landed the row.
    Flipped,
    /// It widened at least one bound.
    Widened,
    /// Nothing: the flips do not land the row or are not allowed, and no
    /// bound is left to widen.
    Nothing,
}

/// The moves [`Solver::certificate_repair`] collects: how far the leaving var
/// must go (`need`), within `target_tol`, and a flip may overshoot
/// (`room_past_target`), how far the moves so far carry it, and the moves,
/// as (col, value).
struct Moves {
    need: f64,
    target_tol: f64,
    room_past_target: f64,
    moved: f64,
    list: Vec<(usize, f64)>,
}

impl Moves {
    /// Whether the moves so far bring the leaving var within its tolerance
    /// of its target.
    fn landed(&self) -> bool {
        self.need - self.moved <= self.target_tol
    }

    /// Take the `candidates` (col, coefficient, from, to), largest reach
    /// first, each all the way or, where that would carry the leaving var
    /// past its other bound, the last one just far enough, until the leaving
    /// var lands. A later move of a col replaces its earlier one.
    fn take(&mut self, candidates: Vec<(usize, f64, f64, f64)>) {
        let mut candidates: Vec<(usize, f64, f64, f64, f64)> = candidates
            .into_iter()
            .filter(|&(_, p, from, to)| (to - from) * p > 0.0)
            .map(|(c, p, from, to)| (c, p, from, to, p.abs() * (to - from).abs()))
            .filter(|&(.., reach)| reach.is_finite())
            .collect();
        candidates.sort_by(|a, b| b.4.total_cmp(&a.4));
        for (c, p, from, to, reach) in candidates {
            if self.landed() {
                break;
            }
            let rest = self.need - self.moved;
            let value = if reach <= rest + self.room_past_target {
                self.moved += reach;
                to
            } else {
                let step = rest / p.abs();
                self.moved = self.need;
                if to > from {
                    from + step
                } else {
                    from - step
                }
            };
            self.set(c, value);
        }
    }

    /// Move `col` to `value`, replacing an earlier move of it.
    fn set(&mut self, col: usize, value: f64) {
        match self.list.iter_mut().find(|(c, _)| *c == col) {
            Some(entry) => entry.1 = value,
            None => self.list.push((col, value)),
        }
    }
}
