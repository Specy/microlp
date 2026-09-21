use super::*;

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn mat_transpose() {
    init();
    let mut mat = SparseMat::new(2);
    mat.push(0, 1.1);
    mat.push(1, 2.2);
    mat.seal_column();
    mat.push(1, 3.3);
    mat.seal_column();
    mat.push(0, 4.4);
    mat.seal_column();

    let transp = mat.transpose();
    assert_eq!(&transp.indptr, &[0, 2, 4]);
    assert_eq!(&transp.indices, &[2, 0, 1, 0]);
    assert_eq!(&transp.data, &[4.4, 1.1, 3.3, 2.2]);
}
