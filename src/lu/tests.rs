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
    let lu = lu_factorize(
        mat.rows(),
        |c| mat.outer_view([1, 0, 3][c]).unwrap().into_raw_storage(),
        0.9,
        &mut scratch,
    )
    .unwrap();
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

#[test]
fn lu_singular() {
    init();
    let size = 3;

    {
        let symbolically_singular = mat_from_triplets(
            size,
            size,
            &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 2.0), (1, 2, 3.0)],
        );

        let mut scratch = ScratchSpace::with_capacity(size);
        let err = lu_factorize(
            size,
            |c| {
                symbolically_singular
                    .outer_view(c)
                    .unwrap()
                    .into_raw_storage()
            },
            0.9,
            &mut scratch,
        );
        assert_eq!(err.unwrap_err(), Error::SingularMatrix);
    }

    {
        let numerically_singular = mat_from_triplets(
            size,
            size,
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

        let mut scratch = ScratchSpace::with_capacity(size);
        let err = lu_factorize(
            size,
            |c| {
                numerically_singular
                    .outer_view(c)
                    .unwrap()
                    .into_raw_storage()
            },
            0.9,
            &mut scratch,
        );
        assert_eq!(err.unwrap_err(), Error::SingularMatrix);
    }
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

    let lu = lu_factorize(
        size,
        |c| mat.outer_view(cols[c]).unwrap().into_raw_storage(),
        0.1,
        &mut scratch,
    )
    .unwrap();
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
