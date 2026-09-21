use super::colamd::order_colamd;
use super::matching::{find_block_diag_form, find_diag_matching};
use super::*;
use sprs::{CsMat, TriMat};

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

fn mat_from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize)]) -> CsMat<f64> {
    let mut mat = TriMat::with_capacity((rows, cols), triplets.len());
    for (r, c) in triplets {
        mat.add_triplet(*r, *c, 1.0);
    }
    mat.to_csc()
}

#[test]
fn colamd() {
    init();
    let mat = mat_from_triplets(
        4,
        5,
        &[
            (0, 0),
            (0, 2),
            (1, 0),
            (1, 2),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 4),
            (3, 0),
            (3, 4),
        ],
    );

    let perm = order_colamd(4, |c| {
        mat.outer_view([0, 1, 2, 4][c])
            .unwrap()
            .into_raw_storage()
            .0
    })
    .unwrap();
    assert_eq!(&perm.new2orig, &[1, 0, 2, 3]);
    assert_eq!(&perm.orig2new, &[1, 0, 2, 3]);
}

#[test]
fn colamd_singular() {
    init();
    {
        let empty_col_mat = mat_from_triplets(3, 3, &[(0, 0), (1, 0), (1, 1), (1, 2)]);
        let res = order_colamd(3, |c| {
            empty_col_mat.outer_view(c).unwrap().into_raw_storage().0
        });
        assert_eq!(res.unwrap_err(), Error::SingularMatrix);
    }

    {
        let empty_row_mat = mat_from_triplets(3, 3, &[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]);
        let res = order_colamd(3, |c| {
            empty_row_mat.outer_view(c).unwrap().into_raw_storage().0
        });
        assert_eq!(res.unwrap_err(), Error::SingularMatrix);
    }
}

#[test]
fn order_simple_singular() {
    init();
    // Column 2 has no entries at all (a variable with zero coefficients
    // in every row). order_simple used to compute `len() - 1`
    // unconditionally and panic with a usize underflow instead of
    // reporting SingularMatrix, unlike order_colamd's handling of the
    // same class of input.
    let mat = mat_from_triplets(3, 3, &[(0, 0), (1, 0), (2, 1)]);
    let res = order_simple(3, |c| mat.outer_view(c).unwrap().into_raw_storage().0);
    assert_eq!(res.unwrap_err(), Error::SingularMatrix);
}

#[test]
fn diag_matching() {
    init();
    let size = 3;
    let mat = mat_from_triplets(
        size,
        size,
        &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0)],
    );

    let matching = find_diag_matching(size, |c| mat.outer_view(c).unwrap().into_raw_storage().0);
    assert_eq!(matching, Some(vec![1, 2, 0]));
}

#[test]
fn block_diag_form() {
    init();
    let size = 3;
    let mat = mat_from_triplets(
        size,
        size,
        &[(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2)],
    );

    let bd_form = find_block_diag_form(size, |c| mat.outer_view(c).unwrap().into_raw_storage().0);
    assert_eq!(bd_form.row2col, &[2, 0, 1]);
    assert_eq!(bd_form.block_cols, vec![vec![0, 1], vec![2]]);
}
