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

/// An empty column (a var with no entry in any row) is ordered with the
/// columns of one entry, for the factorization to find dependent, instead
/// of computing `len() - 1` and underflowing.
#[test]
fn order_simple_ranks_an_empty_column_with_the_single_entries() {
    init();
    let mat = mat_from_triplets(3, 3, &[(0, 0), (1, 0), (2, 1)]);
    let perm = order_simple(3, |c| mat.outer_view(c).unwrap().into_raw_storage().0);
    assert_eq!(perm.new2orig, [1, 2, 0]);
    assert_eq!(perm.orig2new, [2, 0, 1]);
}
