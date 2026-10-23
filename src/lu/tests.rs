use super::*;
use crate::helpers::{assert_matrix_eq, to_dense, to_sparse};
use sprs::{CsMat, CsVec, TriMat};

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

fn mat_from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut mat = TriMat::with_capacity((rows, cols), triplets.len());
    for (r, c, val) in triplets {
        mat.add_triplet(*r, *c, *val);
    }
    mat.to_csc()
}

#[test]
fn lu_simple() {
    init();
    let mat = mat_from_triplets(
        3,
        4,
        &[
            (0, 1, 2.0),
            (0, 0, 2.0),
            (0, 2, 123.0),
            (1, 2, 456.0),
            (1, 3, 1.0),
            (2, 1, 4.0),
            (2, 0, 3.0),
            (2, 2, 789.0),
            (2, 3, 1.0),
        ],
    );

    let mut scratch = ScratchSpace::with_capacity(mat.rows());
    let (lu, replaced) = lu_factorize(
        mat.rows(),
        |c| mat.outer_view([1, 0, 3][c]).unwrap().into_raw_storage(),
        0.9,
        &mut scratch,
    );
    assert!(replaced.is_empty());
    let lu_transp = lu.transpose();

    let l_nondiag_ref = [
        vec![0.0, 0.0, 0.0],
        vec![0.5, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
    ];
    assert_matrix_eq(&lu.lower.nondiag.to_csmat(), &l_nondiag_ref);
    assert_eq!(lu.lower.diag, None);

    let u_nondiag_ref = [
        vec![0.0, 3.0, 1.0],
        vec![0.0, 0.0, -0.5],
        vec![0.0, 0.0, 0.0],
    ];
    let u_diag_ref = [4.0, 0.5, 1.0];
    assert_matrix_eq(&lu.upper.nondiag.to_csmat(), &u_nondiag_ref);
    assert_eq!(lu.upper.diag.as_ref().unwrap(), &u_diag_ref);

    assert_eq!(lu.row_perm.as_ref().unwrap().new2orig, &[2, 0, 1]);
    assert_eq!(lu.col_perm.as_ref().unwrap().new2orig, &[0, 1, 2]);

    {
        let mut rhs_dense = [6.0, 3.0, 13.0];
        lu.solve_dense(&mut rhs_dense, &mut scratch);
        assert_eq!(&rhs_dense, &[1.0, 2.0, 3.0]);
    }

    {
        let mut rhs_dense_t = [14.0, 11.0, 5.0];
        lu_transp.solve_dense(&mut rhs_dense_t, &mut scratch);
        assert_eq!(&rhs_dense_t, &[1.0, 2.0, 3.0]);
    }

    {
        let mut rhs = ScatteredVec::empty(3);
        rhs.set(to_sparse(&[0.0, -1.0, 0.0]).iter());
        lu.solve(&mut rhs, &mut scratch);
        assert_eq!(to_dense(&rhs.to_csvec()), vec![1.0, -1.0, -1.0]);
    }

    {
        let mut rhs = ScatteredVec::empty(3);
        rhs.set(to_sparse(&[0.0, -1.0, 1.0]).iter());
        lu_transp.solve(&mut rhs, &mut scratch);
        assert_eq!(to_dense(&rhs.to_csvec()), vec![-2.0, 0.0, 1.0]);
    }
}

/// Factorize `mat` (square, columns in their own order) at `stability_coeff`.
fn factorize(mat: &CsMat<f64>, stability_coeff: f64) -> (LUFactors, Vec<(usize, usize)>) {
    let mut scratch = ScratchSpace::with_capacity(mat.rows());
    lu_factorize(
        mat.rows(),
        |c| mat.outer_view(c).unwrap().into_raw_storage(),
        stability_coeff,
        &mut scratch,
    )
}

/// `lu` must factorize `mat` with each column of `replaced` swapped for the
/// unit column of its row: solves with it and its transpose, on small
/// integers, are exact.
fn assert_factorizes_the_repaired_matrix(
    lu: &LUFactors,
    mat: &CsMat<f64>,
    replaced: &[(usize, usize)],
) {
    let size = mat.rows();
    // The repaired matrix, dense, by columns.
    let mut cols = vec![vec![0.0; size]; size];
    for (c, col) in cols.iter_mut().enumerate() {
        match replaced
            .iter()
            .find(|&&(replaced_col, _)| replaced_col == c)
        {
            Some(&(_, r)) => col[r] = 1.0,
            None => {
                for (r, &val) in mat.outer_view(c).unwrap().iter() {
                    col[r] = val;
                }
            }
        }
    }
    let x: Vec<f64> = (1..=size).map(|i| i as f64).collect();
    let mut scratch = ScratchSpace::with_capacity(size);

    let mut rhs = vec![0.0; size];
    for (col, &x_c) in cols.iter().zip(&x) {
        for (r, &val) in col.iter().enumerate() {
            rhs[r] += val * x_c;
        }
    }
    lu.solve_dense(&mut rhs, &mut scratch);
    assert_eq!(rhs, x, "B x = b");

    let mut rhs_t: Vec<f64> = cols
        .iter()
        .map(|col| col.iter().zip(&x).map(|(val, x_r)| val * x_r).sum())
        .collect();
    lu.transpose().solve_dense(&mut rhs_t, &mut scratch);
    assert_eq!(rhs_t, x, "B^T y = c");
}

/// A dependent column is replaced by the unit column of the row no pivot
/// took, and the factors are those of the matrix so repaired. The second
/// and third columns are parallel, symbolically (the last row is empty) and
/// numerically (the third column is 1.5 times the second): the third is
/// taken after the second, its remaining part is zero, and the last row
/// gets no pivot.
#[test]
fn lu_completes_a_singular_matrix_with_unit_columns() {
    init();
    let symbolically_singular =
        mat_from_triplets(3, 3, &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 2.0), (1, 2, 3.0)]);
    let numerically_singular = mat_from_triplets(
        3,
        3,
        &[
            (0, 0, 1.0),
            (1, 0, 1.0),
            (1, 1, 2.0),
            (1, 2, 3.0),
            (2, 0, 2.0),
            (2, 1, 2.0),
            (2, 2, 3.0),
        ],
    );
    for mat in [symbolically_singular, numerically_singular] {
        let (lu, replaced) = factorize(&mat, 0.9);
        assert_eq!(replaced, [(2, 2)]);
        assert_factorizes_the_repaired_matrix(&lu, &mat, &replaced);
    }
}

/// Two dependent columns (the second is twice the first, the fourth four
/// times the third) take the two rows no pivot took, in the order the
/// columns were found dependent.
#[test]
fn lu_completes_a_rank_deficiency_of_two() {
    init();
    let mat = mat_from_triplets(
        4,
        4,
        &[
            (0, 0, 1.0),
            (1, 0, 1.0),
            (0, 1, 2.0),
            (1, 1, 2.0),
            (2, 2, 1.0),
            (2, 3, 4.0),
        ],
    );
    let (lu, replaced) = factorize(&mat, 0.1);
    assert_eq!(replaced, [(3, 1), (1, 3)]);
    assert_factorizes_the_repaired_matrix(&lu, &mat, &replaced);
}

#[test]
fn lu_rand() {
    init();
    let size = 10;

    let mut rng = rand_pcg::Pcg64::seed_from_u64(12345);
    use rand::prelude::*;

    let mut mat = TriMat::new((size, size));
    for r in 0..size {
        for c in 0..size {
            if rng.random_range(0..2) == 0 {
                mat.add_triplet(r, c, rng.random_range(0.0..1.0));
            }
        }
    }
    let mat: CsMat<f64> = mat.to_csc();

    let mut scratch = ScratchSpace::with_capacity(mat.rows());

    // TODO: random permutation?
    let cols: Vec<_> = (0..size).collect();

    let (lu, replaced) = lu_factorize(
        size,
        |c| mat.outer_view(cols[c]).unwrap().into_raw_storage(),
        0.1,
        &mut scratch,
    );
    assert!(replaced.is_empty());
    let lu_transp = lu.transpose();

    let multiplied = &lu.lower.to_csmat() * &lu.upper.to_csmat();
    assert!(multiplied.is_csc());
    for (i, &c) in cols.iter().enumerate() {
        let permuted = {
            let mut is = vec![];
            let mut vs = vec![];
            for (i, &val) in mat.outer_view(c).unwrap().iter() {
                is.push(lu.row_perm.as_ref().unwrap().orig2new[i]);
                vs.push(val);
            }
            CsVec::new_from_unsorted(size, is, vs).unwrap()
        };
        let diff = &multiplied
            .outer_view(lu.col_perm.as_ref().unwrap().orig2new[i])
            .unwrap()
            - &permuted;
        assert!(diff.norm(1.0) < 1e-5);
    }

    let sparse_rhs = {
        let mut res = CsVec::empty(size);
        for i in 0..size {
            if rng.random_range(0..3) == 0 {
                res.append(i, rng.random_range(0.0..1.0));
            }
        }
        res
    };

    {
        let mut rhs = ScatteredVec::empty(size);
        rhs.set(sparse_rhs.iter());
        lu.solve(&mut rhs, &mut scratch);
        let diff = &sparse_rhs - &(&mat * &rhs.to_csvec());
        assert!(diff.norm(1.0) < 1e-5);
    }

    {
        let mut rhs_t = ScatteredVec::empty(size);
        rhs_t.set(sparse_rhs.iter());
        lu_transp.solve(&mut rhs_t, &mut scratch);
        let diff = &sparse_rhs - &(&mat.transpose_view() * &rhs_t.to_csvec());
        assert!(diff.norm(1.0) < 1e-5);
    }
}

/// An empty column is dependent, with trace logging on too: the trace of
/// the matrix's excess entries, `nnz - size`, is negative here and must not
/// underflow (a panic in a debug build).
#[test]
fn lu_completes_an_empty_column_with_trace_logging() {
    init();
    let mat = mat_from_triplets(3, 3, &[(0, 0, 1.0), (1, 1, 1.0)]);
    let previous = log::max_level();
    log::set_max_level(log::LevelFilter::Trace);
    let res = std::panic::catch_unwind(|| factorize(&mat, 0.9));
    log::set_max_level(previous);
    let (lu, replaced) = res.unwrap();
    assert_eq!(replaced, [(2, 2)]);
    assert_factorizes_the_repaired_matrix(&lu, &mat, &replaced);
}

/// The basis of a DC power flow over `num_buses` buses on a ring, with a chord
/// every seventh bus a third of the way round: one angle column per bus over a
/// balance row per bus and a flow row per line, filled up to square with the
/// unit columns of the flow rows. A balance row holds `+b` at one end of each
/// incident line and `-b` at the other, and a flow row `+b` and `-b`, so the
/// angle columns sum to zero in every row; with susceptances that are powers
/// of two, from `2^-6` to `2^10`, they sum to exactly `0.0` in `f64`, and the
/// matrix is singular in the arithmetic, not only up to round-off. A shunt
/// `shunt` on bus 0's diagonal breaks that dependency, and none other.
fn power_flow_basis(num_buses: usize, shunt: f64) -> CsMat<f64> {
    let n = num_buses;
    let mut lines: Vec<(usize, usize)> = (0..n).map(|i| (i, (i + 1) % n)).collect();
    if n >= 9 {
        for i in (0..n).step_by(7) {
            let j = (i + n / 3) % n;
            let (a, b) = (i.min(j), i.max(j));
            if a != b && !lines.iter().any(|&(x, y)| (x.min(y), x.max(y)) == (a, b)) {
                lines.push((a, b));
            }
        }
    }
    let size = n + lines.len();
    let mut triplets = Vec::new();
    for (e, &(i, j)) in lines.iter().enumerate() {
        let b = 2f64.powi(-6 + ((e * 7) % 17) as i32);
        triplets.extend([(i, i, b), (j, j, b), (i, j, -b), (j, i, -b)]);
        triplets.extend([(n + e, i, b), (n + e, j, -b)]);
        triplets.push((n + e, n + e, 1.0));
    }
    triplets.push((0, 0, shunt));
    mat_from_triplets(size, size, &triplets)
}

/// A dependent column's remaining part is the round-off of the eliminations
/// applied to it, which grows with their number: on the power flow bases
/// above it measures between `2e-16` and `2e-14` of the column's largest
/// entry, with no trend in the size, so a floor of `1e-14` alone sits inside
/// that range and took a round-off pivot at 40 buses, after 18 eliminations.
/// Every size has its one dependency found, and with a shunt on bus 0, none.
/// A genuine pivot as small, after one elimination, is still taken: see
/// `a_primal_blocking_entry_small_beside_its_column_bounds_the_step`.
#[test]
fn lu_finds_a_dependency_left_as_the_round_off_of_many_eliminations() {
    init();
    for num_buses in [5, 10, 20, 40, 80, 120, 200] {
        let singular = power_flow_basis(num_buses, 0.0);
        let mut row_sums = vec![0.0f64; singular.rows()];
        for c in 0..num_buses {
            for (r, &val) in singular.outer_view(c).unwrap().iter() {
                row_sums[r] += val;
            }
        }
        assert!(
            row_sums.iter().all(|&s| s == 0.0),
            "{num_buses} buses: the angle columns must sum to exactly zero"
        );
        let (_, replaced) = factorize(&singular, crate::solver::LU_STABILITY_THRESHOLD);
        assert_eq!(
            replaced.len(),
            1,
            "{num_buses} buses, singular: {replaced:?}"
        );

        let shunted = power_flow_basis(num_buses, 0.5);
        let (_, replaced) = factorize(&shunted, crate::solver::LU_STABILITY_THRESHOLD);
        assert!(
            replaced.is_empty(),
            "{num_buses} buses, shunted: {replaced:?}"
        );
    }
}
