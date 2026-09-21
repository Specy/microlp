use crate::sparse::{Error, Perm, ScatteredVec, SparseMat, TriangleMat};
use std::cmp::Ordering;

/// A column whose largest eligible pivot is below this fraction of the
/// column's largest transformed entry is treated as singular: the eligible
/// part is then within the round-off of the elimination that produced it,
/// i.e. the column is dependent on the previous ones as far as `f64` can
/// tell. Relative to the column's own scale rather than absolute, so the
/// test does not depend on the units the column happens to be written in,
/// and at round-off level rather than higher so that a basis the ratio tests
/// were willing to pivot into (they accept entries down to the round-off of
/// their row or column) is not refused here; an ill-conditioned but
/// non-singular basis is judged by the verification of the values it
/// produces, not by a threshold on its pivots.
const LU_SINGULAR_REL: f64 = crate::solver::ROUNDOFF_FLOOR;

#[derive(Clone)]
pub struct LUFactors {
    lower: TriangleMat,
    upper: TriangleMat,
    row_perm: Option<Perm>,
    col_perm: Option<Perm>,
}

#[derive(Clone, Debug)]
pub struct ScratchSpace {
    rhs: ScatteredVec,
    dense_rhs: Vec<f64>,
    mark_nonzero: MarkNonzero,
}

impl ScratchSpace {
    pub fn with_capacity(n: usize) -> ScratchSpace {
        ScratchSpace {
            rhs: ScatteredVec::empty(n),
            dense_rhs: vec![0.0; n],
            mark_nonzero: MarkNonzero::with_capacity(n),
        }
    }

    pub(crate) fn clear_sparse(&mut self, size: usize) {
        self.rhs.clear_and_resize(size);
        self.mark_nonzero.clear_and_resize(size);
    }
}

impl std::fmt::Debug for LUFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "L:\n{:?}", self.lower)?;
        writeln!(f, "U:\n{:?}", self.upper)?;
        writeln!(
            f,
            "row_perm.new2orig: {:?}",
            self.row_perm.as_ref().map(|p| &p.new2orig)
        )?;
        writeln!(
            f,
            "col_perm.new2orig: {:?}",
            self.col_perm.as_ref().map(|p| &p.new2orig)
        )?;
        Ok(())
    }
}

impl LUFactors {
    pub fn nnz(&self) -> usize {
        self.lower.nondiag.nnz() + self.upper.nondiag.nnz() + self.lower.cols()
    }

    pub fn solve_dense(&self, rhs: &mut [f64], scratch: &mut ScratchSpace) {
        scratch.dense_rhs.resize(rhs.len(), 0.0);

        if let Some(row_perm) = &self.row_perm {
            for (i, rhs_el) in rhs.iter().enumerate() {
                scratch.dense_rhs[row_perm.orig2new[i]] = *rhs_el;
            }
        } else {
            scratch.dense_rhs.copy_from_slice(rhs);
        }

        tri_solve_dense(&self.lower, Triangle::Lower, &mut scratch.dense_rhs);
        tri_solve_dense(&self.upper, Triangle::Upper, &mut scratch.dense_rhs);

        if let Some(col_perm) = &self.col_perm {
            for i in 0..rhs.len() {
                rhs[col_perm.new2orig[i]] = scratch.dense_rhs[i];
            }
        } else {
            rhs.copy_from_slice(&scratch.dense_rhs);
        }
    }

    pub fn solve(&self, rhs: &mut ScatteredVec, scratch: &mut ScratchSpace) {
        if let Some(row_perm) = &self.row_perm {
            scratch.rhs.clear();
            for &i in &rhs.nonzero {
                let new_i = row_perm.orig2new[i];
                scratch.rhs.nonzero.push(new_i);
                scratch.rhs.is_nonzero[new_i] = true;
                scratch.rhs.values[new_i] = rhs.values[i];
            }
        } else {
            std::mem::swap(&mut scratch.rhs, rhs);
        }

        tri_solve_sparse(&self.lower, scratch);
        tri_solve_sparse(&self.upper, scratch);

        if let Some(col_perm) = &self.col_perm {
            rhs.clear();
            for &i in &scratch.rhs.nonzero {
                let new_i = col_perm.new2orig[i];
                rhs.nonzero.push(new_i);
                rhs.is_nonzero[new_i] = true;
                rhs.values[new_i] = scratch.rhs.values[i];
            }
        } else {
            std::mem::swap(rhs, &mut scratch.rhs);
        }
    }

    pub fn transpose(&self) -> LUFactors {
        LUFactors {
            lower: self.upper.transpose(),
            upper: self.lower.transpose(),
            row_perm: self.col_perm.clone(),
            col_perm: self.row_perm.clone(),
        }
    }
}

pub fn lu_factorize<'a>(
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    stability_coeff: f64,
    scratch: &mut ScratchSpace,
) -> Result<LUFactors, Error> {
    // Implementation of the Gilbert-Peierls algorithm:
    //
    // Gilbert, John R., and Tim Peierls. "Sparse partial pivoting in time
    // proportional to arithmetic operations." SIAM Journal on Scientific and
    // Statistical Computing 9.5 (1988): 862-874.
    //
    // https://ecommons.cornell.edu/bitstream/handle/1813/6623/86-783.pdf

    let mat_nnz = (0..size).map(|c| get_col(c).0.len()).sum::<usize>();
    trace!(
        "lu_factorize: starting, matrix size: {}, nnz: {} (excess: {})",
        size,
        mat_nnz,
        mat_nnz - size,
    );

    let col_perm = super::ordering::order_simple(size, |c| get_col(c).0)?;

    let mut orig_row2elt_count = vec![0; size];
    for col_rows in (0..size).map(|c| get_col(c).0) {
        for &orig_r in col_rows {
            orig_row2elt_count[orig_r] += 1;
        }
    }

    scratch.clear_sparse(size);

    let mut lower = SparseMat::new(size);
    let mut upper = SparseMat::new(size);
    let mut upper_diag = Vec::with_capacity(size);

    let mut new2orig_row = (0..size).collect::<Vec<_>>();
    let mut orig2new_row = new2orig_row.clone();

    for (i_col, &orig_col) in col_perm.new2orig.iter().enumerate() {
        let mat_col = get_col(orig_col);

        // Solve the equation L'_j * x = a_j (x will be in scratch.rhs).
        // L'_j is a sq. matrix with the first j columns of L
        // and columns of identity matrix after j.
        // Part of x above the diagonal (in the new row indices) is the column of U
        // and part of x below the diagonal divided by the pivot value is the column of L

        scratch.rhs.set(mat_col.0.iter().copied().zip(mat_col.1));

        scratch.mark_nonzero.run(
            &mut scratch.rhs,
            |new_i| lower.col_rows(new_i),
            |new_i| new_i < i_col,
            |orig_r| orig2new_row[orig_r],
        );

        // At this point all future nonzero positions of scratch.rhs are marked
        // and the order in which variables depend on each other is determined.
        // rev() because DFS returns vertices in reverse topological order.
        for &orig_i in scratch.mark_nonzero.visited.iter().rev() {
            // values[orig_i] is already fully calculated, diag coeff = 1.0.
            let new_i = orig2new_row[orig_i];
            if new_i < i_col {
                let x_val = scratch.rhs.values[orig_i];
                for (orig_r, coeff) in lower.col_iter(new_i) {
                    scratch.rhs.values[orig_r] -= x_val * coeff;
                }
            }
        }

        // Next we choose a pivot among values of x below the diagonal.
        // Pivoting by choosing the max element is good for stability,
        // but bad for sparseness, so we do threshold pivoting instead.

        let pivot_orig_r = {
            // The largest eligible pivot, and the largest entry of the whole
            // transformed column (eliminated part included) it is judged
            // against: a column whose remaining part is negligible relative
            // to what was already eliminated is linearly dependent on the
            // previous columns up to round-off, whatever its absolute scale.
            let mut max_abs = 0.0;
            let mut col_max = 0.0;
            for &orig_r in &scratch.rhs.nonzero {
                let abs = f64::abs(scratch.rhs.values[orig_r]);
                if abs > col_max {
                    col_max = abs;
                }
                if orig2new_row[orig_r] < i_col {
                    continue;
                }
                if abs > max_abs {
                    max_abs = abs;
                }
            }

            if !max_abs.is_normal() || max_abs < LU_SINGULAR_REL * col_max {
                return Err(Error::SingularMatrix);
            }

            // Choose among eligible pivot rows one with the least elements.
            // Gilbert-Peierls suggest to choose row with least elements *to the right*,
            // but it yielded poor results. Our heuristic is not a huge improvement either,
            // but at least we are less dependent on initial row ordering.
            let mut best_orig_r = None;
            let mut best_elt_count = None;
            for &orig_r in &scratch.rhs.nonzero {
                if orig2new_row[orig_r] < i_col {
                    continue;
                }

                if f64::abs(scratch.rhs.values[orig_r]) >= stability_coeff * max_abs {
                    let elt_count = orig_row2elt_count[orig_r];
                    if best_elt_count.is_none() || best_elt_count.unwrap() > elt_count {
                        best_orig_r = Some(orig_r);
                        best_elt_count = Some(elt_count);
                    }
                }
            }
            best_orig_r.unwrap()
        };

        let pivot_val = scratch.rhs.values[pivot_orig_r];

        {
            // Keep track of row permutations.
            let row = i_col;
            let orig_row = new2orig_row[row];
            let pivot_row = orig2new_row[pivot_orig_r];
            new2orig_row.swap(row, pivot_row);
            orig2new_row.swap(orig_row, pivot_orig_r);
        }

        // Gather the values of x into lower and upper matrices.

        for &orig_r in &scratch.rhs.nonzero {
            let val = scratch.rhs.values[orig_r];

            if val == 0.0 {
                continue;
            }

            let new_r = orig2new_row[orig_r];
            match new_r.cmp(&i_col) {
                Ordering::Less => upper.push(new_r, val),
                Ordering::Equal => upper_diag.push(pivot_val),
                Ordering::Greater => lower.push(orig_r, val / pivot_val),
            }
        }

        upper.seal_column();
        lower.seal_column();
    }

    // permute rows of lower to "new" indices.
    for i_col in 0..lower.cols() {
        for r in lower.col_rows_mut(i_col) {
            *r = orig2new_row[*r];
        }
    }

    let lower_nnz = lower.nnz();
    let upper_nnz = upper.nnz();
    trace!(
        "lu_factorize: done, lower nnz: {} (excess: {}), upper nnz: {} (excess: {}), additional fill-in: {}",
        lower_nnz + size,
        lower_nnz,
        upper_nnz + size,
        upper_nnz,
        (lower_nnz + upper_nnz + size) as isize - mat_nnz as isize,
    );

    let res = LUFactors {
        lower: TriangleMat {
            nondiag: lower,
            diag: None,
        },
        upper: TriangleMat {
            nondiag: upper,
            diag: Some(upper_diag),
        },
        row_perm: Some(Perm {
            orig2new: orig2new_row,
            new2orig: new2orig_row,
        }),
        col_perm: Some(col_perm),
    };

    Ok(res)
}

#[derive(Clone, Debug)]
struct MarkNonzero {
    dfs_stack: Vec<DfsStep>,
    is_visited: Vec<bool>,
    visited: Vec<usize>, // in reverse topological order
}

#[derive(Clone, Debug)]
struct DfsStep {
    orig_i: usize,
    cur_child: usize,
}

impl MarkNonzero {
    fn with_capacity(n: usize) -> MarkNonzero {
        MarkNonzero {
            dfs_stack: Vec::with_capacity(n),
            is_visited: vec![false; n],
            visited: vec![],
        }
    }

    fn clear(&mut self) {
        assert!(self.dfs_stack.is_empty());
        for &i in &self.visited {
            self.is_visited[i] = false;
        }
        self.visited.clear();
    }

    fn clear_and_resize(&mut self, n: usize) {
        self.clear();
        self.dfs_stack.reserve(n);
        self.is_visited.resize(n, false);
    }

    // compute the non-zero elements of the result by dfs traversal
    fn run<'a>(
        &mut self,
        rhs: &mut ScatteredVec,
        get_children: impl Fn(usize) -> &'a [usize] + 'a,
        filter: impl Fn(usize) -> bool,
        orig2new_row: impl Fn(usize) -> usize,
    ) {
        self.clear();

        for &orig_r in &rhs.nonzero {
            let new_r = orig2new_row(orig_r);
            if !filter(new_r) {
                continue;
            }
            if self.is_visited[orig_r] {
                continue;
            }

            self.dfs_stack.push(DfsStep {
                orig_i: orig_r,
                cur_child: 0,
            });
            while !self.dfs_stack.is_empty() {
                let cur_step = self.dfs_stack.last_mut().unwrap();
                let new_i = orig2new_row(cur_step.orig_i);
                let children = if filter(new_i) {
                    get_children(new_i)
                } else {
                    &[]
                };
                if !self.is_visited[cur_step.orig_i] {
                    self.is_visited[cur_step.orig_i] = true;
                } else {
                    cur_step.cur_child += 1;
                }

                while cur_step.cur_child < children.len() {
                    let child_orig_r = children[cur_step.cur_child];
                    if !self.is_visited[child_orig_r] {
                        break;
                    }
                    cur_step.cur_child += 1;
                }

                if cur_step.cur_child < children.len() {
                    let i_child = cur_step.cur_child;
                    self.dfs_stack.push(DfsStep {
                        orig_i: children[i_child],
                        cur_child: 0,
                    });
                } else {
                    self.visited.push(cur_step.orig_i);
                    self.dfs_stack.pop();
                }
            }
        }

        for &i in &self.visited {
            if !rhs.is_nonzero[i] {
                rhs.is_nonzero[i] = true;
                rhs.nonzero.push(i)
            }
        }
    }
}

enum Triangle {
    Lower,
    Upper,
}

fn tri_solve_dense(tri_mat: &TriangleMat, triangle: Triangle, rhs: &mut [f64]) {
    assert_eq!(tri_mat.rows(), rhs.len());
    match triangle {
        Triangle::Lower => {
            for col in 0..tri_mat.cols() {
                tri_solve_process_col(tri_mat, col, rhs);
            }
        }

        Triangle::Upper => {
            for col in (0..tri_mat.cols()).rev() {
                tri_solve_process_col(tri_mat, col, rhs);
            }
        }
    };
}

/// rhs is passed via scratch.visited, scratch.values.
fn tri_solve_sparse(tri_mat: &TriangleMat, scratch: &mut ScratchSpace) {
    assert_eq!(tri_mat.rows(), scratch.rhs.len());

    // compute the non-zero elements of the result by dfs traversal
    scratch.mark_nonzero.run(
        &mut scratch.rhs,
        |col| tri_mat.nondiag.col_rows(col),
        |_| true,
        |orig_i| orig_i,
    );

    // solve for the non-zero values into dense workspace.
    // rev() because DFS returns vertices in reverse topological order.
    for &col in scratch.mark_nonzero.visited.iter().rev() {
        tri_solve_process_col(tri_mat, col, &mut scratch.rhs.values);
    }
}

fn tri_solve_process_col(tri_mat: &TriangleMat, col: usize, rhs: &mut [f64]) {
    // all other variables in this row (multiplied by their coeffs)
    // are already subtracted from rhs[col].
    let x_val = if let Some(diag) = tri_mat.diag.as_ref() {
        rhs[col] / diag[col]
    } else {
        rhs[col]
    };

    rhs[col] = x_val;
    for (r, &coeff) in tri_mat.nondiag.col_iter(col) {
        rhs[r] -= x_val * coeff;
    }
}

#[cfg(test)]
mod tests;
