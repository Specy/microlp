use super::*;
use crate::solver::Solver;
use crate::{ComparisonOp, VarDomain};

fn to_sparse(values: &[f64]) -> crate::CsVec {
    let mut indices = vec![];
    let mut data = vec![];
    for (i, &v) in values.iter().enumerate() {
        if v != 0.0 {
            indices.push(i);
            data.push(v);
        }
    }
    crate::CsVec::new(values.len(), indices, data)
}

#[test]
fn most_fractional_var_is_chosen() {
    // minimize -x - y s.t. x + 2y <= 3.2, x <= 1.9; x,y integer-domained.
    // LP optimum: x = 1.9, y = 0.65 → fractional parts 0.9 and 0.65;
    // most-fractional metric |v - round(v)|: x → 0.1, y → 0.35 → picks y (idx 1).
    // With a fresh PseudoCosts (uniform init, no recorded data), the product score
    // est_down·f_down · est_up·f_up is maximized by the most-fractional var too:
    // var 1: 0.65·0.35 ≈ 0.23 beats var 0: 0.9·0.1 = 0.09.
    let mut solver = Solver::try_new(
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[1.9, 10.0],
        &[(to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 3.2)],
        &[VarDomain::Integer, VarDomain::Integer],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    assert!(!is_integral(
        &solver,
        solver.orig_var_domains.clone().as_slice(),
        1e-6
    ));
    let pc = PseudoCosts::new(&[-1.0, -1.0], 2);
    assert_eq!(
        choose_branch_var(
            &solver,
            solver.orig_var_domains.clone().as_slice(),
            1e-6,
            &pc
        ),
        Some(1)
    );
}

#[test]
fn integral_solution_yields_no_branch_var() {
    // minimize x s.t. x >= 2, x integer in [0, 10] → LP optimum x = 2 (integral).
    let mut solver = Solver::try_new(
        &[1.0],
        &[0.0],
        &[10.0],
        &[(to_sparse(&[2.0]), ComparisonOp::Ge, 4.0)],
        &[VarDomain::Integer],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    assert!(is_integral(
        &solver,
        solver.orig_var_domains.clone().as_slice(),
        1e-6
    ));
    let pc = PseudoCosts::new(&[1.0], 1);
    assert_eq!(
        choose_branch_var(
            &solver,
            solver.orig_var_domains.clone().as_slice(),
            1e-6,
            &pc
        ),
        None
    );
}

#[test]
fn pseudocosts_average_and_fall_back_to_init() {
    // 2 vars, obj coeffs 3 and 0 → init estimates 3+1e-6 and 1e-6... clamped by new().
    let mut pc = PseudoCosts::new(&[3.0, 0.0], 2);
    assert!((pc.estimate(0, true) - 3.0).abs() < 1e-3);
    pc.record(0, true, 10.0);
    pc.record(0, true, 20.0);
    assert!((pc.estimate(0, true) - 15.0).abs() < 1e-9); // average of observations
    assert!((pc.estimate(0, false) - 3.0).abs() < 1e-3); // down side still init
}

#[test]
fn pseudocost_selection_prefers_high_degradation_var() {
    // Two fractional int vars; var 1 has recorded huge degradations → must be chosen.
    let mut solver = Solver::try_new(
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[1.5, 10.0],
        &[(to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 3.0)],
        &[VarDomain::Integer, VarDomain::Integer],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    // LP optimum: x=1.5, y=0.75 → both fractional.
    let mut pc = PseudoCosts::new(&[-1.0, -1.0], 2);
    pc.record(1, true, 100.0);
    pc.record(1, false, 100.0);
    let domains = solver.orig_var_domains.clone();
    assert_eq!(choose_branch_var(&solver, &domains, 1e-6, &pc), Some(1));
}
