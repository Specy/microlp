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
fn order_simple_singular() {
    init();
    // Column 2 has no entries at all (a variable with zero coefficients
    // in every row): order_simple must report SingularMatrix, not
    // underflow computing `len() - 1`.
    let mat = mat_from_triplets(3, 3, &[(0, 0), (1, 0), (2, 1)]);
    let res = order_simple(3, |c| mat.outer_view(c).unwrap().into_raw_storage().0);
    assert_eq!(res.unwrap_err(), Error::SingularMatrix);
}
