use super::*;
use crate::helpers::{assert_matrix_eq, to_sparse};
use crate::{OptimizationDirection, Problem};

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn initialize() {
    init();
    let sol = Solver::try_new(
        &[2.0, 1.0],
        &[f64::NEG_INFINITY, 5.0],
        &[0.0, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 6.0),
            (to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 8.0),
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 2.0),
            (to_sparse(&[0.0, 1.0]), ComparisonOp::Eq, 3.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        Default::default(),
        crate::Tolerances::default().feasibility,
    )
    .unwrap();

    assert_eq!(sol.num_vars, 2);
    assert!(!sol.is_primal_feasible);
    assert!(!sol.is_dual_feasible);

    assert_eq!(&sol.orig_obj_coeffs, &[2.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

    assert_eq!(
        &sol.orig_var_mins,
        &[f64::NEG_INFINITY, 5.0, 0.0, 0.0, f64::NEG_INFINITY, 0.0,]
    );
    assert_eq!(
        &sol.orig_var_maxs,
        &[0.0, f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0, 0.0]
    );

    // Equilibration scales the second constraint (max structural
    // coefficient 2) and its rhs by 1/2; the slack column stays 1. The
    // unit-coefficient rows are unchanged.
    let orig_constraints_ref = vec![
        vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.5, 1.0, 0.0, 1.0, 0.0, 0.0],
        vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ];
    assert_matrix_eq(&sol.orig_constraints, &orig_constraints_ref);

    assert_eq!(&sol.orig_rhs, &[6.0, 4.0, 2.0, 3.0]);

    assert_eq!(&sol.basic_vars, &[2, 3, 4, 5]);
    assert_eq!(&sol.basic_var_vals, &[1.0, -1.0, -3.0, -2.0]);
    assert_eq!(&sol.dual_edge_sq_norms, &[1.0, 1.0, 1.0, 1.0]);

    assert_eq!(&sol.nb_vars, &[0, 1]);
    assert_eq!(&sol.nb_var_obj_coeffs, &[-1.0, 1.0]);
    assert_eq!(&sol.nb_var_vals, &[0.0, 5.0]);
    assert_eq!(&sol.primal_edge_sq_norms, &[3.25, 5.0]);

    assert_eq!(sol.cur_obj_val, 0.0);
}

#[test]
fn try_new_rejects_nan_bound() {
    init();
    // A NaN bound used to slip past the `min > max` guard (every
    // comparison against NaN is false, including this one), so it was
    // accepted as an ordinary bound. Downstream, the simplex loop's own
    // bound comparisons against that NaN never resolve either, so the
    // solve hangs forever instead of reporting Infeasible up front (the
    // same thing set_var_bounds already guards against for edits).
    let res = Solver::try_new(
        &[1.0],
        &[f64::NAN],
        &[10.0],
        &[],
        &[VarDomain::Real],
        Default::default(),
        crate::Tolerances::default().feasibility,
    );
    assert_eq!(res.unwrap_err(), Error::Infeasible);
}

/// Dense recalculations with pending etas must match a fresh factorization
/// of the same basis. Values are compared per variable because reloading
/// may reorder basis positions.
#[test]
fn recalcs_with_pending_etas_match_a_fresh_factorization() {
    init();
    // minimize x + y + z, pairwise sums >= 2, boxes [0, 10]: optimum
    // x = y = z = 1. A bound tightening then forces dual pivots, which
    // push etas.
    let mut solver = Solver::try_new(
        &[1.0, 1.0, 1.0],
        &[0.0, 0.0, 0.0],
        &[10.0, 10.0, 10.0],
        &[
            (to_sparse(&[1.0, 1.0, 0.0]), ComparisonOp::Ge, 2.0),
            (to_sparse(&[0.0, 1.0, 1.0]), ComparisonOp::Ge, 2.0),
            (to_sparse(&[1.0, 0.0, 1.0]), ComparisonOp::Ge, 2.0),
        ],
        &[VarDomain::Real, VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(solver.initial_solve().unwrap(), StopReason::Finished);
    solver.set_var_bounds(2, 0.0, 0.25).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(
        solver.basis_solver.eta_matrices.len() > 0,
        "fixture must leave etas pending to exercise eta-aware recalculation"
    );

    // Recalculate through the eta-aware dense solves.
    solver.recalc_basic_var_vals().unwrap();
    solver.recalc_obj_coeffs().unwrap();
    let by_var = |s: &Solver| -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = s
            .basic_vars
            .iter()
            .zip(&s.basic_var_vals)
            .map(|(&var, &val)| (var, val))
            .collect();
        v.sort_by_key(|&(var, _)| var);
        v
    };
    let rc_by_var = |s: &Solver| -> Vec<(usize, f64)> {
        let mut v: Vec<(usize, f64)> = s
            .nb_vars
            .iter()
            .zip(&s.nb_var_obj_coeffs)
            .map(|(&var, &rc)| (var, rc))
            .collect();
        v.sort_by_key(|&(var, _)| var);
        v
    };
    let eta_vals = by_var(&solver);
    let eta_rcs = rc_by_var(&solver);
    let eta_obj = solver.cur_obj_val;

    // Reloading the solver's own snapshot refactorizes from scratch and
    // reruns the recalcs eta-free — the ground truth.
    let basis = solver.snapshot_basis();
    solver.load_basis(&basis).unwrap();
    assert_eq!(solver.basis_solver.eta_matrices.len(), 0);
    for ((va, a), (vb, b)) in eta_vals.iter().zip(by_var(&solver).iter()) {
        assert_eq!(va, vb);
        assert!((a - b).abs() < 1e-9, "basic val of var {va}: {a} vs {b}");
    }
    for ((va, a), (vb, b)) in eta_rcs.iter().zip(rc_by_var(&solver).iter()) {
        assert_eq!(va, vb);
        assert!((a - b).abs() < 1e-9, "reduced cost of var {va}: {a} vs {b}");
    }
    assert!((eta_obj - solver.cur_obj_val).abs() < 1e-9);
}

#[test]
fn solve_integer_singular_var() {
    init();
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_integer_var(1.0, (0, 10));
    problem.add_constraint([(x, 30.0)], ComparisonOp::Ge, 90.0);
    assert!(
        (problem
            .solve()
            .unwrap()
            .into_solution()
            .unwrap()
            .objective()
            - 3.0)
            .abs()
            < EPS
    );

    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_integer_var(1.0, (0, 10));
    problem.add_constraint([(x, 30.0)], ComparisonOp::Ge, 91.0);
    assert!(
        (problem
            .solve()
            .unwrap()
            .into_solution()
            .unwrap()
            .objective()
            - 4.0)
            .abs()
            < EPS
    );

    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_integer_var(1.0, (0, 10));
    problem.add_constraint([(x, 30.0)], ComparisonOp::Le, 90.0);
    assert!(
        (problem
            .solve()
            .unwrap()
            .into_solution()
            .unwrap()
            .objective()
            - 3.0)
            .abs()
            < EPS
    );

    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_integer_var(1.0, (0, 10));
    problem.add_constraint([(x, 30.0)], ComparisonOp::Le, 91.0);
    assert!(
        (problem
            .solve()
            .unwrap()
            .into_solution()
            .unwrap()
            .objective()
            - 3.0)
            .abs()
            < EPS
    );
}

#[test]
fn solve_powers_integer() {
    init();
    let n = 15626;
    // return (a,b,c) such that 2^a * 3^b * 5^c >= n and is minimized given a,b,c € N
    let logn = (n as f64).log2();
    let log2 = 2_f64.log2();
    let log3 = 3_f64.log2();
    let log5 = 5_f64.log2();
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let p2 = problem.add_integer_var(log2, (0, 100));
    let p3 = problem.add_integer_var(log3, (0, 100));
    let p5 = problem.add_integer_var(log5, (0, 100));
    problem.add_constraint(
        &[(p2, log2), (p3, log3), (p5, log5)],
        ComparisonOp::Ge,
        logn,
    );
    let sol = problem.solve().unwrap().into_solution().unwrap();
    assert_eq!(sol.objective().round() as i64, 14);
}

#[test]
fn initial_solve() {
    init();
    let mut sol = Solver::try_new(
        &[-3.0, -4.0],
        &[f64::NEG_INFINITY, 5.0],
        &[20.0, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 20.0),
            (to_sparse(&[-1.0, 4.0]), ComparisonOp::Le, 20.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        Default::default(),
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    sol.initial_solve().unwrap();

    assert!(sol.is_primal_feasible);
    assert!(sol.is_dual_feasible);

    assert_eq!(&sol.basic_vars, &[0, 1]);
    assert_eq!(&sol.basic_var_vals, &[12.0, 8.0]);
    assert_eq!(&sol.nb_vars, &[2, 3]);
    assert_eq!(&sol.nb_var_vals, &[0.0, 0.0]);
    // The optimum (x=12, y=8, obj -68) is unchanged by equilibration; only
    // the second constraint's slack reduced cost is scaled: that row
    // (-x+4y<=20, max coeff 4) is equilibrated by 1/4, so its dual scales
    // up 4x, 0.2 -> 0.8.
    assert_eq!(&sol.nb_var_obj_coeffs, &[3.2, 0.8]);
    assert_eq!(sol.cur_obj_val, -68.0);

    let infeasible = Solver::try_new(
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 10.0),
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 5.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        Default::default(),
        crate::Tolerances::default().feasibility,
    )
    .unwrap()
    .initial_solve();
    assert_eq!(infeasible.unwrap_err(), Error::Infeasible);
}

#[test]
fn set_var_bounds_tighten_matches_fresh_solve() {
    init();
    // minimize 2x + 3y s.t. x + y >= 4, 0 <= x,y <= 10. Optimum: x=4, y=0, obj 8.
    let coeffs = [2.0, 3.0];
    let mins = [0.0, 0.0];
    let maxs = [10.0, 10.0];
    let cons = [(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 4.0)];
    let domains = [VarDomain::Real, VarDomain::Real];

    let mut warm = Solver::try_new(
        &coeffs,
        &mins,
        &maxs,
        &cons,
        &domains,
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    warm.initial_solve().unwrap();
    assert!(float_eq(warm.cur_obj_val, 8.0));

    // Tighten x to [0, 2] and re-solve warm: optimum becomes x=2, y=2, obj 10.
    warm.set_var_bounds(0, 0.0, 2.0).unwrap();
    assert_eq!(warm.reoptimize().unwrap(), StopReason::Finished);
    assert!(warm.is_primal_feasible && warm.is_dual_feasible);
    assert!(float_eq(warm.cur_obj_val, 10.0));
    assert!(float_eq(warm.get_value(0), 2.0));
    assert!(float_eq(warm.get_value(1), 2.0));

    // Fresh solve of the tightened problem must agree.
    let mut fresh = Solver::try_new(
        &coeffs,
        &mins,
        &[2.0, 10.0],
        &cons,
        &domains,
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    fresh.initial_solve().unwrap();
    assert!(float_eq(fresh.cur_obj_val, warm.cur_obj_val));
}

#[test]
fn set_var_bounds_loosen_and_retighten() {
    init();
    // maximize x + y (internally minimize -x - y) s.t. x + y <= 4, 0 <= x,y <= 3.
    let mut solver = Solver::try_new(
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[3.0, 3.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    assert!(float_eq(solver.cur_obj_val, -4.0));

    // Tighten x to [0, 0.5]: optimum x=0.5, y=3, obj -3.5.
    solver.set_var_bounds(0, 0.0, 0.5).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(solver.cur_obj_val, -3.5));

    // Loosen x back to [0, 3]: optimum returns to -4.
    solver.set_var_bounds(0, 0.0, 3.0).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(solver.cur_obj_val, -4.0));

    assert!(solver.lp_iterations > 0);
}

#[test]
fn set_var_bounds_crossing_is_infeasible_and_leaves_state_untouched() {
    init();
    let mut solver = Solver::try_new(
        &[1.0],
        &[0.0],
        &[10.0],
        &[(to_sparse(&[1.0]), ComparisonOp::Ge, 1.0)],
        &[VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    let obj_before = solver.cur_obj_val;
    assert_eq!(
        solver.set_var_bounds(0, 2.0, 1.0).unwrap_err(),
        Error::Infeasible
    );
    assert_eq!(solver.get_var_bounds(0), (0.0, 10.0)); // untouched
    assert!(float_eq(solver.cur_obj_val, obj_before));
}

#[test]
fn set_var_bounds_nan_is_infeasible_and_leaves_state_untouched() {
    let mut original = Solver::try_new(
        &[1.0],
        &[0.0],
        &[10.0],
        &[],
        &[VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(original.initial_solve().unwrap(), StopReason::Finished);

    for (min, max) in [(f64::NAN, 10.0), (0.0, f64::NAN)] {
        let mut solver = original.clone();
        let bounds_before = solver.get_var_bounds(0);
        let value_before = solver.get_value(0);
        let objective_before = solver.cur_obj_val;
        let primal_before = solver.is_primal_feasible;
        let dual_before = solver.is_dual_feasible;

        assert_eq!(solver.set_var_bounds(0, min, max), Err(Error::Infeasible));
        assert_eq!(solver.get_var_bounds(0), bounds_before);
        assert_eq!(solver.get_value(0), value_before);
        assert_eq!(solver.cur_obj_val, objective_before);
        assert_eq!(solver.is_primal_feasible, primal_before);
        assert_eq!(solver.is_dual_feasible, dual_before);
    }

    let mut solver = original;
    assert_eq!(
        solver.set_var_bounds(0, f64::NEG_INFINITY, f64::INFINITY),
        Ok(())
    );
    assert_eq!(solver.get_var_bounds(0), (f64::NEG_INFINITY, f64::INFINITY));
}

#[test]
fn check_constraints_rejects_non_finite_activity() {
    let solver = Solver::try_new(
        &[0.0],
        &[0.0],
        &[f64::INFINITY],
        &[(to_sparse(&[1.0e308]), ComparisonOp::Eq, f64::INFINITY)],
        &[VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();

    assert!(solver.first_violated_row(&[1.0e308]).is_some());
}

#[test]
fn basis_snapshot_load_roundtrip() {
    init();
    // This bounded fixture has objective -68 at (12, 8), with both
    // structural variables basic and both slacks non-basic. Its basis is
    // therefore non-trivial and differs from the slack basis.
    let mut solver = Solver::try_new(
        &[-3.0, -4.0],
        &[f64::NEG_INFINITY, 5.0],
        &[20.0, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 20.0),
            (to_sparse(&[-1.0, 4.0]), ComparisonOp::Le, 20.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    let obj = solver.cur_obj_val;
    let vals: Vec<f64> = (0..2).map(|v| solver.get_value(v)).collect();
    let basis = solver.snapshot_basis();

    // Wreck the state by loading the all-slack basis…
    let slack = solver.slack_basis();
    solver.load_basis(&slack).unwrap();

    // …then reload the optimal basis: objective and values must round-trip.
    solver.load_basis(&basis).unwrap();
    assert!(solver.is_primal_feasible && solver.is_dual_feasible);
    assert!(float_eq(solver.cur_obj_val, obj));
    for v in 0..2 {
        assert!(float_eq(solver.get_value(v), vals[v]));
    }
}

#[test]
fn slack_basis_load_then_reoptimize_reaches_optimum() {
    init();
    // minimize 2x + 3y s.t. x + y >= 4, 0 <= x,y <= 10 → obj 8.
    let mut solver = Solver::try_new(
        &[2.0, 3.0],
        &[0.0, 0.0],
        &[10.0, 10.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 4.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    assert!(float_eq(solver.cur_obj_val, 8.0));

    let slack = solver.slack_basis();
    solver.load_basis(&slack).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(solver.cur_obj_val, 8.0));
}

#[test]
fn load_basis_rejects_wrong_shape() {
    init();
    let mut solver = Solver::try_new(
        &[1.0],
        &[0.0],
        &[1.0],
        &[(to_sparse(&[1.0]), ComparisonOp::Le, 1.0)],
        &[VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    solver.initial_solve().unwrap();
    // 2 total vars (1 structural + 1 slack); a basis with zero Basic entries is invalid.
    let bad = Basis(vec![VarStatus::AtLower, VarStatus::AtLower]);
    assert!(solver.load_basis(&bad).is_err());
    // Solver must still be usable via the slack-basis fallback path.
    let slack = solver.slack_basis();
    solver.load_basis(&slack).unwrap();
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
}
