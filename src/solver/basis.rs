//! Inversion of the basis matrix, and the description of a single pivot.
//!
//! [`BasisSolver`] owns the LU factorization of the basis plus the eta file of
//! pivots stacked on top of it, and answers `B^{-1} x` / `B^{-T} x` queries.
//! [`PivotChoice`] and friends are the plain-data verdict the simplex loops in
//! [`super::Solver`] produce and then apply.

use crate::{
    lu::{lu_factorize, LUFactors, ScratchSpace},
    sparse::{ScatteredVec, SparseMat, SparseVec},
    CsVec, Error,
};

use super::tolerances::LU_STABILITY_THRESHOLD;
use super::CsMat;

/// Outcome of the primal simplex pivot choice.
#[derive(Debug)]
pub(crate) enum PivotChoice {
    /// No reduced cost can improve the objective: the current basis is optimal.
    Optimal,
    /// Apply this pivot.
    Pivot(PivotInfo),
    /// The pivot element at `row` computed through the row and through the
    /// column disagree (see [`super::PIVOT_AGREEMENT_TOL`]). With pivots stacked on
    /// the factorization it must be rebuilt before choosing again; on a fresh
    /// one the entry is noise if `tiny` (relative to the column) and the row
    /// is then excluded, else the factorization itself is unusable.
    Inconsistent { row: usize, tiny: bool },
    /// The improving column `col` cannot be entered: its only step is
    /// degenerate, blocked by a basic var sitting past its bound within
    /// tolerance, and putting that var onto its bound would move the
    /// entering var by more than its own tolerance, i.e. onto an infeasible
    /// basis. A primal pivot must keep primal feasibility, so the column is
    /// not an improving direction at this vertex.
    Unexploitable(usize),
}

#[derive(Debug)]
pub(crate) struct PivotInfo {
    pub(crate) col: usize,
    pub(crate) entering_new_val: f64,
    pub(crate) entering_diff: f64,

    /// Contains info about the intersection between pivot row and column.
    /// If it is None, objective can be decreased without changing the basis
    /// (simply by changing the value of non-basic variable chosen as entering)
    pub(crate) elem: Option<PivotElem>,
}

#[derive(Debug)]
pub(crate) struct PivotElem {
    pub(crate) row: usize,
    pub(crate) coeff: f64,
    pub(crate) leaving_new_val: f64,
}

/// Stuff related to inversion of the basis matrix
#[derive(Clone)]
pub(crate) struct BasisSolver {
    pub(crate) lu_factors: LUFactors,
    pub(crate) lu_factors_transp: LUFactors,
    pub(crate) scratch: ScratchSpace,
    pub(crate) eta_matrices: EtaMatrices,
    pub(crate) rhs: ScatteredVec,
}

impl BasisSolver {
    /// Record the basis change `B' = B E` as the eta column of `E^{-1}`: the
    /// diagonal entry is `1 / pivot` and the off-diagonal entries are
    /// `alpha_i / pivot`, where `alpha = B^{-1} a_entering` is `col_coeffs`.
    /// The forward application is then `x_r' = x_r / pivot`,
    /// `x_i' = x_i - x_r alpha_i / pivot` (see [`Self::apply_etas`]).
    ///
    /// The diagonal is stored as `1 / pivot` itself, not as `1 - 1 / pivot`
    /// applied by subtraction: for a large pivot `1 - 1 / pivot` rounds to
    /// within an ulp of one, and `x_r - x_r (1 - 1 / pivot)` cancels away all
    /// but a few digits of `x_r / pivot`. Every later solve through that eta,
    /// including a from-scratch recomputation of the basic values, carried
    /// that error (a single-row LP lost nine significant digits at a pivot
    /// of 7e6).
    pub(crate) fn push_eta_matrix(
        &mut self,
        col_coeffs: &SparseVec,
        r_leaving: usize,
        pivot_coeff: f64,
    ) {
        let coeffs = col_coeffs.iter().map(|(r, &coeff)| {
            let val = if r == r_leaving {
                1.0 / pivot_coeff
            } else {
                coeff / pivot_coeff
            };
            (r, val)
        });
        self.eta_matrices.push(r_leaving, coeffs);
    }

    /// Forward eta application (Vanderbei p.139): `x <- E_k^{-1} ... E_1^{-1} x`
    /// on a dense vector.
    fn apply_etas(eta_matrices: &EtaMatrices, x: &mut [f64]) {
        for idx in 0..eta_matrices.len() {
            let r_leaving = eta_matrices.leaving_rows[idx];
            let x_r = x[r_leaving];
            if x_r == 0.0 {
                continue;
            }
            for (r, &val) in eta_matrices.coeff_cols.col_iter(idx) {
                if r == r_leaving {
                    x[r] = x_r * val;
                } else {
                    x[r] -= x_r * val;
                }
            }
        }
    }

    /// Transposed eta application: `y <- E_1^{-T} ... E_k^{-T} y` on a dense
    /// vector, i.e. `y_r' = y_r / pivot - sum_{i != r} (alpha_i / pivot) y_i`.
    fn apply_etas_transp(eta_matrices: &EtaMatrices, y: &mut [f64]) {
        for idx in (0..eta_matrices.len()).rev() {
            let r_leaving = eta_matrices.leaving_rows[idx];
            let mut diag = 0.0;
            let mut acc = 0.0;
            for (i, &val) in eta_matrices.coeff_cols.col_iter(idx) {
                if i == r_leaving {
                    diag = val;
                } else {
                    acc += val * y[i];
                }
            }
            debug_assert!(diag != 0.0, "eta column without its pivot entry");
            y[r_leaving] = diag * y[r_leaving] - acc;
        }
    }

    pub(crate) fn reset(
        &mut self,
        orig_constraints_csc: &CsMat,
        basic_vars: &[usize],
    ) -> Result<(), Error> {
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

    pub(crate) fn solve<'a>(
        &mut self,
        rhs: impl Iterator<Item = (usize, &'a f64)>,
    ) -> &ScatteredVec {
        self.rhs.set(rhs);
        self.lu_factors.solve(&mut self.rhs, &mut self.scratch);

        // Forward eta application on the scattered vector; same arithmetic as
        // `apply_etas`, but through `get_mut` so the nonzero pattern is kept.
        for idx in 0..self.eta_matrices.len() {
            let r_leaving = self.eta_matrices.leaving_rows[idx];
            let x_r = *self.rhs.get(r_leaving);
            if x_r == 0.0 {
                continue;
            }
            for (r, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                if r == r_leaving {
                    *self.rhs.get_mut(r) = x_r * val;
                } else {
                    *self.rhs.get_mut(r) -= x_r * val;
                }
            }
        }

        &mut self.rhs
    }

    /// Dense counterpart of [`Self::solve`]: LU solve plus the forward eta
    /// application, so callers with dense right-hand sides (the recalcs) no
    /// longer need a full refactorization just because etas are pending.
    pub(crate) fn solve_dense_with_etas(&mut self, rhs: &mut [f64]) {
        self.lu_factors.solve_dense(rhs, &mut self.scratch);
        Self::apply_etas(&self.eta_matrices, rhs);
    }

    /// Dense counterpart of [`Self::solve_transp`]: the reverse eta
    /// application, then the transposed LU solve.
    pub(crate) fn solve_transp_dense_with_etas(&mut self, rhs: &mut [f64]) {
        Self::apply_etas_transp(&self.eta_matrices, rhs);
        self.lu_factors_transp.solve_dense(rhs, &mut self.scratch);
    }

    /// Pass right-hand side via self.rhs
    pub(crate) fn solve_transp<'a>(
        &mut self,
        rhs: impl Iterator<Item = (usize, &'a f64)>,
    ) -> &ScatteredVec {
        self.rhs.set(rhs);
        // Transposed eta application on the scattered vector; same arithmetic
        // as `apply_etas_transp`, through `get_mut` for the nonzero pattern.
        for idx in (0..self.eta_matrices.len()).rev() {
            let r_leaving = self.eta_matrices.leaving_rows[idx];
            let mut diag = 0.0;
            let mut acc = 0.0;
            for (i, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                if i == r_leaving {
                    diag = val;
                } else {
                    acc += val * self.rhs.get(i);
                }
            }
            debug_assert!(diag != 0.0, "eta column without its pivot entry");
            let y_r = *self.rhs.get(r_leaving);
            *self.rhs.get_mut(r_leaving) = diag * y_r - acc;
        }

        self.lu_factors_transp
            .solve(&mut self.rhs, &mut self.scratch);
        &mut self.rhs
    }
}

#[derive(Clone, Debug)]
pub(crate) struct EtaMatrices {
    leaving_rows: Vec<usize>,
    pub(crate) coeff_cols: SparseMat,
}

impl EtaMatrices {
    pub(crate) fn new(n_rows: usize) -> EtaMatrices {
        EtaMatrices {
            leaving_rows: vec![],
            coeff_cols: SparseMat::new(n_rows),
        }
    }

    pub(crate) fn len(&self) -> usize {
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

pub(crate) fn into_resized(vec: CsVec, len: usize) -> CsVec {
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
