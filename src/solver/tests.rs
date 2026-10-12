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
    assert!(sol.artificial_obj.is_some());

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

/// `entry_bounds` judges each side of a range on its own: a bound at or
/// beyond 2^52 on its side of infinity is infinite whatever the other bound
/// (a var must not start on a far bound of `f64::MAX`), a bound on the other
/// side is data, a point or an empty range is kept, and NaN stays NaN.
#[test]
fn entry_bounds_judges_each_side_on_its_own() {
    let t = INFINITE_BOUND;
    let (inf, max) = (f64::INFINITY, f64::MAX);
    let cases = [
        ((0.0, max), (0.0, inf)),
        ((-max, 0.0), (-inf, 0.0)),
        ((-max, max), (-inf, inf)),
        ((0.0, t), (0.0, inf)),
        ((-t, 0.0), (-inf, 0.0)),
        ((0.0, t - 1.0), (0.0, t - 1.0)),
        ((1.0 - t, 0.0), (1.0 - t, 0.0)),
        // Ranges beyond the threshold: the far side is infinite, the near side a bound.
        ((5e15, max), (5e15, inf)),
        ((t, 1e20), (t, inf)),
        ((5e15, 1e16), (5e15, inf)),
        ((-max, -5e15), (-inf, -5e15)),
        ((-(i64::MAX as f64), -t), (-inf, -t)),
        ((max, inf), (max, inf)),
        // A point or an empty range is the var's value or a contradiction.
        ((5e15, 5e15), (5e15, 5e15)),
        ((-t, -t), (-t, -t)),
        ((max, max), (max, max)),
        ((6e15, 5e15), (6e15, 5e15)),
        ((-5e15, -6e15), (-5e15, -6e15)),
        ((max, -max), (max, -max)),
        ((inf, inf), (inf, inf)),
    ];
    for ((min, max), want) in cases {
        assert_eq!(entry_bounds(min, max), want, "[{min:e}, {max:e}]");
    }
    // A NaN bound stays NaN for the solve to refuse (`try_new_rejects_nan_bound`).
    assert!(entry_bounds(f64::NAN, f64::MAX).0.is_nan());
    assert!(entry_bounds(-f64::MAX, f64::NAN).1.is_nan());
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
    assert_eq!(sol.calc_dual_infeasibility().0, 0);

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

/// The bound-flipping dual ratio test: `min Σ c_i x_i` over `x ∈ [0, 1]^8`
/// with `Σ x_i ≥ 5` and costs `1..=8` starts dual feasible at zero; the four
/// cheapest vars flip to their upper bound and the fifth enters, so the phase
/// takes one pivot where a ratio test without flips takes five.
#[test]
fn dual_step_flips_the_boxed_vars_it_passes() {
    init();
    let n = 8;
    let costs: Vec<f64> = (1..=n).map(|c| c as f64).collect();
    let mut sol = Solver::try_new(
        &costs,
        &vec![0.0; n],
        &vec![1.0; n],
        &[(to_sparse(&vec![1.0; n]), ComparisonOp::Ge, 5.0)],
        &vec![VarDomain::Real; n],
        Default::default(),
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    sol.initial_solve().unwrap();

    assert_eq!(sol.cur_obj_val, 15.0);
    let values: Vec<f64> = (0..n).map(|v| sol.get_value(v)).collect();
    assert_eq!(values, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    // One pivot, then the check that finds every row within its bounds.
    assert_eq!(sol.lp_iterations, 2);
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
    assert!(warm.is_primal_feasible && warm.calc_dual_infeasibility().0 == 0);
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
        let reduced_costs_before = solver.nb_var_obj_coeffs.clone();

        assert_eq!(solver.set_var_bounds(0, min, max), Err(Error::Infeasible));
        assert_eq!(solver.get_var_bounds(0), bounds_before);
        assert_eq!(solver.get_value(0), value_before);
        assert_eq!(solver.cur_obj_val, objective_before);
        assert_eq!(solver.is_primal_feasible, primal_before);
        assert_eq!(solver.nb_var_obj_coeffs, reduced_costs_before);
    }

    let mut solver = original;
    assert_eq!(
        solver.set_var_bounds(0, f64::NEG_INFINITY, f64::INFINITY),
        Ok(())
    );
    assert_eq!(solver.get_var_bounds(0), (f64::NEG_INFINITY, f64::INFINITY));
}

/// A value or an activity that is not finite violates every bound and row,
/// whatever its sense: it must not pass an infinite bound, nor meet the
/// infinite tolerance an infinite magnitude gives a row (an infinite value of
/// a free var would let a NaN objective through).
#[test]
fn check_constraints_rejects_non_finite_activity() {
    for op in [ComparisonOp::Eq, ComparisonOp::Le, ComparisonOp::Ge] {
        let solver = Solver::try_new(
            &[0.0, 0.0],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY],
            &[f64::INFINITY, f64::INFINITY],
            &[(to_sparse(&[1.0, 1.0]), op, 0.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        for values in [
            [f64::MAX, f64::MAX],
            [-f64::MAX, -f64::MAX],
            [f64::INFINITY, 0.0],
            [f64::NEG_INFINITY, 0.0],
            [f64::NAN, 0.0],
        ] {
            assert!(
                solver.first_violated_row(&values).is_some(),
                "{op:?} row passed at {values:?}"
            );
        }
        for values in [
            [f64::INFINITY, 0.0],
            [0.0, f64::NEG_INFINITY],
            [f64::NAN, 0.0],
        ] {
            assert!(
                solver.first_violated_bound(&values).is_some(),
                "free vars passed at {values:?}"
            );
        }
    }
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
    assert!(solver.is_primal_feasible && solver.calc_dual_infeasibility().0 == 0);
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

/// `load_basis` fails only on a basis of the wrong shape (a singular one loads
/// repaired: `a_singular_basis_loads_repaired`), and then leaves the state
/// untouched: the solver goes on from where it was, with no reload.
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
    let state = format!("{solver:?}");
    use VarStatus::{AtLower, Basic};
    // Two total vars (one structural, one slack) and one row.
    for bad in [vec![AtLower, AtLower], vec![Basic, Basic], vec![Basic]] {
        let result = solver.load_basis(&Basis(bad.clone()));
        assert!(matches!(result, Err(Error::InternalError(_))), "{bad:?}");
        assert_eq!(format!("{solver:?}"), state, "{bad:?}");
    }
    let iterations = solver.lp_iterations;
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!(solver.lp_iterations, iterations);
}

/// A basis the factorization finds singular loads repaired and solves. `x`
/// and `y` have the same column in `x + y <= 4`, `2 x + 2 y <= 10`, so
/// `{x, y}` is singular: the load makes `y` non-basic at its lower bound, the
/// value the slack basis gives it, with the second row's slack in its place,
/// records nothing outside a phase, and the re-solve reaches the optimum `-8`
/// at `y = 4`.
#[test]
fn a_singular_basis_loads_repaired() {
    init();
    let mut solver = Solver::try_new(
        &[-1.0, -2.0],
        &[0.0, 0.0],
        &[10.0, 10.0],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0),
            (to_sparse(&[2.0, 2.0]), ComparisonOp::Le, 10.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    use VarStatus::{AtLower, Basic};
    solver
        .load_basis(&Basis(vec![Basic, Basic, AtLower, AtLower]))
        .unwrap();
    assert!(matches!(solver.var_states[0], VarState::Basic(_)));
    assert!(matches!(solver.var_states[3], VarState::Basic(_)));
    assert!(matches!(solver.var_states[1], VarState::NonBasic(_)));
    assert_eq!(solver.get_value(1), 0.0);
    assert!(!solver.cycle_guard.any_taboo());
    assert_eq!(solver.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(solver.cur_obj_val, -8.0));
    assert!(float_eq(solver.get_value(1), 4.0));
}

/// The optimum of `min -2 x - y` s.t. `x + y <= 4`, `x - y <= 0`,
/// `x <= 10`, `y <= 7`: `-6` at `x = y = 2`, with the basis `{x, y}`.
fn optimum_with_x_and_y_basic() -> Solver {
    let mut sol = Solver::try_new(
        &[-2.0, -1.0],
        &[0.0, 0.0],
        &[10.0, 7.0],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0),
            (to_sparse(&[1.0, -1.0]), ComparisonOp::Le, 0.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    assert!(
        matches!(sol.var_states[0], VarState::Basic(_))
            && matches!(sol.var_states[1], VarState::Basic(_)),
        "fixture: x and y are basic"
    );
    sol
}

/// Plant `y`'s entry in the second row equal to its entry in the first, in the
/// column and the row data: the basis `{x, y}` is singular for the next
/// factorization, and the model is now `x + y <= 4`, `x + y <= 0`, whose
/// optimum is `0` at the origin.
fn plant_y_parallel_to_x(sol: &mut Solver) {
    let a = *sol.orig_constraints_csc.get(0, 1).unwrap();
    *sol.orig_constraints_csc.get_mut(1, 1).unwrap() = a;
    *sol.orig_constraints.get_mut(1, 1).unwrap() = a;
}

/// A repair within a phase makes the change back bad and taboo: the dual
/// simplex may not take the slack's row out, nor the primal simplex put the
/// var back in, and once a rebuild lifts the taboo, the change is taboo again
/// as soon as it is chosen. The plant makes `{x, y}` singular for a rebuild
/// at the optimum.
#[test]
fn a_repair_within_a_phase_makes_the_change_back_taboo() {
    init();
    let mut sol = optimum_with_x_and_y_basic();
    plant_y_parallel_to_x(&mut sol);

    sol.begin_dual_phase(true).unwrap();
    sol.rebuild().unwrap();
    let var = (0..2)
        .find(|&v| matches!(sol.var_states[v], VarState::NonBasic(_)))
        .expect("the repair takes x or y out");
    let (slack, row) = (2..4)
        .find_map(|s| match sol.var_states[s] {
            VarState::Basic(row) => Some((s, row)),
            VarState::NonBasic(_) => None,
        })
        .expect("a slack replaces it");
    assert!(sol.cycle_guard.is_taboo(row) && sol.cycle_guard.is_taboo_entering(var));
    sol.cycle_guard.clear_taboo();
    assert!(!sol.cycle_guard.any_taboo());
    let here = sol.cycle_guard.hash_after(std::iter::empty());
    assert!(sol.cycle_guard.is_bad_change(row, slack, var, here));
    sol.end_dual_phase();
}

/// A var the repair takes out gives back the cost shift of the zero dual step
/// it entered on, as a var that leaves by a pivot does: a shift lives only
/// while its var is basic. Both basic vars carry a shift when the plant makes
/// `{x, y}` singular for a rebuild within a dual phase; the var taken out has
/// none left, and the one still basic keeps its own.
#[test]
fn a_var_the_repair_takes_out_gives_back_its_cost_shift() {
    init();
    let mut sol = optimum_with_x_and_y_basic();
    plant_y_parallel_to_x(&mut sol);
    sol.begin_dual_phase(true).unwrap();
    let shifts = [0.5, 0.25];
    sol.cost_shifts[..2].copy_from_slice(&shifts);
    sol.costs_modified = true;

    sol.rebuild().unwrap();
    let out = (0..2)
        .find(|&v| matches!(sol.var_states[v], VarState::NonBasic(_)))
        .expect("the repair takes x or y out");
    let kept = 1 - out;
    assert!(matches!(sol.var_states[kept], VarState::Basic(_)));
    assert_eq!(sol.cost_shifts[out], 0.0);
    assert_eq!(sol.cost_shifts[kept], shifts[kept]);
    sol.end_dual_phase();
}

/// `verify_primal`'s second rung repairs a basis the fresh factorization finds
/// singular. With the plant, the values no longer satisfy the rows, the
/// refinement through the old factors cannot make them (it takes `y` to 4,
/// nearer its upper bound than its lower), and the fresh factorization finds
/// `{x, y}` singular. The vertex the repair reaches is no verification: the
/// answer is `false`, with the feasibility flags measured on it (`y` leaves
/// at its upper bound, so it is neither primal nor dual feasible), and the
/// re-solve reaches the planted model's optimum.
#[test]
fn verify_primal_repairs_a_basis_its_fresh_factorization_finds_singular() {
    init();
    let mut sol = optimum_with_x_and_y_basic();
    plant_y_parallel_to_x(&mut sol);
    sol.values_dirty = true;

    assert_eq!(
        sol.verify_primal().unwrap(),
        Verdict::Checked { feasible: false }
    );
    assert!(
        (0..2).any(|v| matches!(sol.var_states[v], VarState::NonBasic(_))),
        "the repair takes x or y out"
    );
    assert!(!sol.is_primal_feasible && sol.calc_primal_infeasibility().0 > 0);
    assert!(sol.calc_dual_infeasibility().0 > 0);
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert!(float_eq(sol.cur_obj_val, 0.0));
}

/// `n` rows `d x_j + x_{j-1} + x_{n-1}` of sense `op`, `d` the `diagonal`
/// (no `x_j` term in the last, no `x_{j-1}` in the first), met with equality
/// at `x = 1`, over free vars of zero cost, with every structural var basic.
/// The basis is well conditioned, but a factorization that takes each
/// column's `d` as its pivot beside the `1` below it (eligible at a threshold
/// below `d`, and in the row with fewer entries) multiplies the last column's
/// entries by about `1 / d` per column: its factors represent another matrix,
/// and the values recomputed through them miss the rows.
fn chain_basis(n: usize, diagonal: f64, op: ComparisonOp) -> Solver {
    let mut sol = Solver::try_new(
        &vec![0.0; n],
        &vec![f64::NEG_INFINITY; n],
        &vec![f64::INFINITY; n],
        &chain_rows(n, n, diagonal, op),
        &vec![VarDomain::Real; n],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let statuses = (0..2 * n).map(|v| {
        if v < n {
            VarStatus::Basic
        } else {
            VarStatus::AtLower
        }
    });
    sol.load_basis(&Basis(statuses.collect())).unwrap();
    sol
}

/// The rows of [`chain_basis`] over the first `n` of `cols` columns.
fn chain_rows(
    n: usize,
    cols: usize,
    diagonal: f64,
    op: ComparisonOp,
) -> Vec<(CsVec, ComparisonOp, f64)> {
    (0..n)
        .map(|i| {
            let mut coeffs = vec![0.0; cols];
            if i + 1 < n {
                coeffs[i] = diagonal;
            }
            if i > 0 {
                coeffs[i - 1] = 1.0;
            }
            coeffs[n - 1] = 1.0;
            let rhs = coeffs.iter().sum();
            (to_sparse(&coeffs), op, rhs)
        })
        .collect()
}

/// `verify_primal`'s third rung: a basis whose factors at
/// [`LU_STABILITY_THRESHOLD`] grow by `9^n` (pivots of `0.11` beside a `1`),
/// beyond what refinements through them repair, is refactorized at
/// [`LU_STABLE_THRESHOLD`], which takes the `1`s, and the values recomputed
/// through it satisfy the rows: `x = 1`.
#[test]
fn verify_primal_refactorizes_at_the_stable_threshold() {
    init();
    let mut sol = chain_basis(72, 0.11, ComparisonOp::Eq);
    assert_eq!(
        sol.verify_primal().unwrap(),
        Verdict::Checked { feasible: true }
    );
    for v in 0..72 {
        assert!(
            (sol.get_value(v) - 1.0).abs() < 1e-9,
            "x{v} = {}",
            sol.get_value(v)
        );
    }
}

/// The ladder's last rung: no factorization represents a basis whose factors
/// at both thresholds grow by about `1.8^n` (pivots of `0.55` beside a `1`),
/// so the solve starts over from the slack basis, which here is optimal: the
/// rows, `<=` and met with equality at `x = 1`, hold at `x = 0`.
#[test]
fn a_basis_no_factorization_represents_is_solved_from_the_slack_basis() {
    init();
    let sol = chain_basis(140, 0.55, ComparisonOp::Le);
    assert_eq!(
        sol.clone().verify_primal().unwrap(),
        Verdict::Unrepresentable
    );
    let mut sol = sol;
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    let x: Vec<f64> = (0..140).map(|v| sol.get_value(v)).collect();
    assert!(x.iter().all(|&x| x == 0.0), "{x:?}");
    assert!(sol.first_violated_row(&x).is_none());
}

/// The chain of [`a_basis_no_factorization_represents_is_solved_from_the_slack_basis`]
/// (`n = 140` vars, all basic when `planted`) beside two blocks that need
/// the dual phase's tolerance mechanisms: `x + y >= 2 + 2.5e-7` over
/// `x, y ∈ [0, 1]` at costs `-1` and `-0.5`, whose vertices all miss the
/// row by more than its tolerance, and the equalities `u + v = 1` and
/// `u + v + 1e8 z = 1e8 + 1 + 5e-7` over `u, v ∈ [0, 2]` with `z` fixed at
/// 1, which disagree by less than the second row's tolerance. Planted, `x`
/// and both equalities' slacks are basic, `y` and the first row's slack at
/// their upper bounds; otherwise the slack basis is loaded.
fn chain_beside_a_cover_and_two_equalities(planted: bool) -> Solver {
    let n = 140;
    let cols = n + 5;
    let (x, y, u, v, z) = (n, n + 1, n + 2, n + 3, n + 4);
    let row = |terms: &[(usize, f64)], op, rhs| {
        let mut coeffs = vec![0.0; cols];
        for &(var, coeff) in terms {
            coeffs[var] = coeff;
        }
        (to_sparse(&coeffs), op, rhs)
    };
    let mut rows = chain_rows(n, cols, 0.55, ComparisonOp::Le);
    rows.push(row(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 2.0 + 2.5e-7));
    rows.push(row(&[(u, 1.0), (v, 1.0)], ComparisonOp::Eq, 1.0));
    rows.push(row(
        &[(u, 1.0), (v, 1.0), (z, 1e8)],
        ComparisonOp::Eq,
        1e8 + 1.0 + 5e-7,
    ));
    let mut obj = vec![0.0; cols];
    obj[x] = -1.0;
    obj[y] = -0.5;
    let (mut mins, mut maxs) = (vec![f64::NEG_INFINITY; cols], vec![f64::INFINITY; cols]);
    for (var, min, max) in [
        (x, 0.0, 1.0),
        (y, 0.0, 1.0),
        (u, 0.0, 2.0),
        (v, 0.0, 2.0),
        (z, 1.0, 1.0),
    ] {
        (mins[var], maxs[var]) = (min, max);
    }
    let mut sol = Solver::try_new(
        &obj,
        &mins,
        &maxs,
        &rows,
        &vec![VarDomain::Real; cols],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let basis = if planted {
        let cover_slack = cols + n;
        Basis(
            (0..cols + n + 3)
                .map(|var| {
                    if var <= x || var > cover_slack {
                        VarStatus::Basic
                    } else if var == y || var == cover_slack {
                        VarStatus::AtUpper
                    } else {
                        VarStatus::AtLower
                    }
                })
                .collect(),
        )
    } else {
        sol.slack_basis()
    };
    sol.load_basis(&basis).unwrap();
    sol
}

/// The slack-basis restart starts over from the model's bounds, with no
/// absorption spent. From the planted basis the dual phase widens `x`'s and
/// the cover row's slack's upper bounds and lets the second equality's slack
/// absorb the equalities' disagreement, then the verification finds that
/// no factorization represents the chain. The restart then solves as a solve
/// from the slack basis does: it widens `x`'s and `y`'s bounds and absorbs
/// into that slack again. Restarted on the widened bounds, the dual phase
/// would leave `y` short of its widening; with the absorption spent, it
/// would widen that slack's bound instead.
#[test]
fn the_slack_basis_restart_starts_from_the_models_bounds_with_no_absorption_spent() {
    init();
    let sides = |sol: &Solver| {
        let mut sides: Vec<(usize, bool)> = sol.widened.iter().map(|w| (w.var, w.up)).collect();
        sides.sort();
        sides
    };
    let values = |sol: &Solver| {
        (0..sol.num_vars)
            .map(|v| sol.get_value(v))
            .collect::<Vec<_>>()
    };
    let mut fresh = chain_beside_a_cover_and_two_equalities(false);
    assert_eq!(fresh.reoptimize().unwrap(), StopReason::Finished);

    let planted = chain_beside_a_cover_and_two_equalities(true);
    let mut first = planted.clone();
    assert_eq!(first.phase_rounds().unwrap(), None, "fixture: the restart");
    assert!(
        !first.absorbed.is_empty() && sides(&first) != sides(&fresh),
        "fixture: the first attempt absorbs {:?} and widens {:?}, a solve from the slack basis \
         widens {:?}",
        first.absorbed,
        sides(&first),
        sides(&fresh)
    );

    let mut restarted = planted;
    assert_eq!(restarted.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!(sides(&restarted), sides(&fresh));
    assert_eq!(restarted.absorbed, fresh.absorbed);
    assert_eq!(values(&restarted), values(&fresh));
}

/// The vertex of `a_vertex_whose_exact_values_miss_a_bound_goes_on_from_its_refined_values`
/// (`src/tests/magnitudes.rs`): its computed values meet every bound, and
/// its exact ones put `x3` past its upper bound.
fn vertex_whose_exact_values_miss_a_bound() -> Solver {
    Solver::try_new(
        &[0.0, -2.0, 0.5, -2.0],
        &[67109.0, 0.0, 0.0, 0.0],
        &[67109.0, 1.0, 1.0, 1.0],
        &[
            (
                to_sparse(&[53.3, 0.0, 0.0016, 0.0]),
                ComparisonOp::Eq,
                3576909.7016,
            ),
            (
                to_sparse(&[0.0, -0.0194, -824.0, -0.0011]),
                ComparisonOp::Ge,
                -824.0011,
            ),
            (to_sparse(&[0.0, 0.0155, 0.0, 0.0]), ComparisonOp::Eq, 0.0),
        ],
        &vec![VarDomain::Real; 4],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap()
}

/// The slack-basis restart starts with no bound held: its rounds go past the
/// bounds a first attempt went past on refined values, and reach the
/// optimum. Held, `x3`'s upper bound would keep the computed values of the
/// vertex whose exact values miss it
/// (`refined_values_missing_only_held_bounds_do_not_go_on`).
#[test]
fn the_slack_basis_restart_starts_with_no_bound_held() {
    init();
    let values = |sol: &Solver| sol.reported_values();
    let mut first = vertex_whose_exact_values_miss_a_bound();
    assert_eq!(first.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!(
        first.held,
        [(3, true)],
        "fixture: the refined values go past x3's upper bound"
    );
    assert!(
        (first.cur_obj_val + 1.5).abs() < 1e-8,
        "{}",
        first.cur_obj_val
    );

    let mut restarted = vertex_whose_exact_values_miss_a_bound();
    restarted.held = first.held.clone();
    assert_eq!(
        restarted.phase_rounds_from_slack_basis().unwrap(),
        Some(StopReason::Finished)
    );
    assert_eq!(values(&restarted), values(&first));
}

/// Refined values that miss only held bounds do not go on: with `x3`'s upper
/// bound held, the vertex whose exact values miss it ends the rounds on its
/// computed values, `0.083` worse than the optimum.
#[test]
fn refined_values_missing_only_held_bounds_do_not_go_on() {
    init();
    let mut sol = vertex_whose_exact_values_miss_a_bound();
    sol.held.push((3, true));
    assert_eq!(sol.phase_rounds().unwrap(), Some(StopReason::Finished));
    assert!(
        (sol.cur_obj_val + 1.4288401582615).abs() < 1e-9,
        "{}",
        sol.cur_obj_val
    );
    assert_eq!(sol.held, [(3, true)]);
    assert!(sol.fallback.is_none());
}

/// An excursion on refined values that ends in an error returns to the
/// answer it left, a state the checks accept and the phases go on from: the
/// verification finds the rows held and the bounds met, the dual check the
/// reduced costs, and a solve that verifies its values again goes on from
/// the refined ones to the optimum.
#[test]
fn an_excursion_that_fails_returns_to_the_answer_it_left() {
    init();
    let mut sol = vertex_whose_exact_values_miss_a_bound();
    assert_eq!(sol.phase_rounds().unwrap(), Some(StopReason::Finished));
    let answer = sol
        .fallback
        .as_ref()
        .expect("fixture: the refined values go on")
        .objective;
    assert!((answer + 1.4288401582615).abs() < 1e-9, "{answer}");
    assert!((sol.cur_obj_val + 1.5).abs() < 1e-8, "{}", sol.cur_obj_val);

    let failed = Err(Error::InternalError("the excursion failed".to_string()));
    assert_eq!(
        sol.keep_the_better_answer(failed).unwrap(),
        Some(StopReason::Finished)
    );
    assert_eq!(sol.cur_obj_val, answer);
    assert_eq!(sol.objective_of(&sol.reported_values()), answer);
    sol.values_dirty = true;
    assert_eq!(
        sol.verify_primal().unwrap(),
        Verdict::Checked { feasible: true }
    );
    sol.recalc_obj_coeffs_refined();
    assert_eq!(sol.calc_dual_infeasibility().0, 0);
    sol.values_dirty = true;
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert!((sol.cur_obj_val + 1.5).abs() < 1e-8, "{}", sol.cur_obj_val);
}

/// The vertex of [`vertex_whose_exact_values_miss_a_bound`] after an
/// excursion on its refined values to the optimum, `-1.5`, and the objective
/// of the answer it left, `-1.4288`.
fn after_an_excursion() -> (Solver, f64) {
    let mut sol = vertex_whose_exact_values_miss_a_bound();
    assert_eq!(sol.phase_rounds().unwrap(), Some(StopReason::Finished));
    let answer = sol
        .fallback
        .as_ref()
        .expect("fixture: the refined values go on")
        .objective;
    assert!((sol.cur_obj_val + 1.5).abs() < 1e-8, "{}", sol.cur_obj_val);
    (sol, answer)
}

/// An excursion on refined values that meets the deadline stops on the
/// answer it left, still with `Limit`, and a later call ends there at once:
/// no pivot, and no excursion again.
#[test]
fn an_excursion_that_meets_the_deadline_stops_on_the_answer_it_left() {
    init();
    let (mut sol, answer) = after_an_excursion();
    assert_eq!(
        sol.keep_the_better_answer(Ok(Some(StopReason::Limit)))
            .unwrap(),
        Some(StopReason::Limit)
    );
    assert_eq!(sol.cur_obj_val, answer);
    let (values, iterations) = (sol.reported_values(), sol.lp_iterations);
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!(sol.reported_values(), values);
    assert_eq!(sol.lp_iterations, iterations);
    assert!(sol.held.is_empty());
}

/// An excursion that ends without an answer returns to the answer it left:
/// through a vertex no factorization represents (`Ok(None)`), or with the
/// phase rounds run out.
#[test]
fn an_excursion_without_an_answer_returns_to_the_answer_it_left() {
    init();
    let rounds_out = Error::InternalError(format!(
        "simplex did not reach a verified optimum in {MAX_PHASE_ROUNDS} phase rounds"
    ));
    for outcome in [Ok(None), Err(rounds_out)] {
        let (mut sol, answer) = after_an_excursion();
        assert_eq!(
            sol.keep_the_better_answer(outcome).unwrap(),
            Some(StopReason::Finished)
        );
        assert_eq!(sol.cur_obj_val, answer);
    }
}

/// A return to an answer whose basis a fresh factorization finds singular
/// fails and changes nothing: the excursion's outcome stands. Planted: the
/// answer's basis with `x1`'s column swapped for `x0`'s, which shares the
/// zero third entry of `x2`'s and `x3`'s.
#[test]
fn a_return_to_a_basis_found_singular_leaves_the_excursions_outcome() {
    init();
    let (mut sol, _) = after_an_excursion();
    let ended = (
        sol.cur_obj_val,
        sol.reported_values(),
        sol.basic_vars.clone(),
    );
    let answer = sol.fallback.as_mut().unwrap();
    let pos = answer
        .basic_vars
        .iter()
        .position(|&var| var == 1)
        .expect("fixture: x1 is basic at the answer");
    answer.basic_vars[pos] = 0;
    let failed = Err(Error::InternalError("the excursion failed".to_string()));
    assert!(sol.keep_the_better_answer(failed).is_err());
    assert_eq!(
        (
            sol.cur_obj_val,
            sol.reported_values(),
            sol.basic_vars.clone()
        ),
        ended
    );
    assert!(sol.fallback.is_none());
}

/// The engine of `problem` as written, without presolve.
fn engine_of(problem: &Problem) -> Solver {
    Solver::try_new(
        &problem.obj_coeffs,
        &problem.var_mins,
        &problem.var_maxs,
        &problem.constraints,
        &problem.var_domains,
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap()
}

/// Refined values that gain the objective within the tolerances do not go
/// on: the coupled model of
/// `computed_values_within_their_tolerances_of_the_vertex_stay`
/// (`src/tests/magnitudes.rs`), whose answer's refined values miss a bound
/// and gain `5.8e-9`, `x6` being `5.5e-8` from its refined value, within its
/// tolerance. No bound is held, and the answer is the computed values'.
#[test]
fn refined_values_that_gain_within_the_tolerances_do_not_go_on() {
    init();
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x: Vec<_> = [
        (-3.0, 0.0, 1.0),
        (0.0, 67109.0, 67109.0),
        (-2.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, 67109.0, 67109.0),
        (-2.0, 0.0, 1.0),
        (0.1, 0.0, 1.0),
        (0.0, 67109.0, 67109.0),
        (-2.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 67109.0, 67109.0),
        (-1.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
    ]
    .into_iter()
    .map(|(cost, min, max)| problem.add_var(cost, (min, max)))
    .collect();
    let rows: [(&[(usize, f64)], ComparisonOp, f64); 10] = [
        (
            &[(5, -0.0194), (6, -412.0), (0, -0.0022)],
            ComparisonOp::Le,
            -412.0022,
        ),
        (&[(4, 53.3), (6, 0.0016)], ComparisonOp::Eq, 3576909.7016),
        (&[(1, 53.3), (3, 0.0016)], ComparisonOp::Le, 3576909.7016),
        (
            &[(8, -0.0194), (9, -824.0), (0, -0.0022)],
            ComparisonOp::Ge,
            -824.0022,
        ),
        (
            &[(10, 53.3), (12, 0.0016), (0, -0.5)],
            ComparisonOp::Le,
            3576909.2065999997,
        ),
        (
            &[(2, -0.0194), (3, -412.0), (0, -0.0022)],
            ComparisonOp::Le,
            -412.0022,
        ),
        (
            &[(7, 53.3), (9, 0.0016), (0, -1.0)],
            ComparisonOp::Eq,
            3576908.7016,
        ),
        (
            &[(11, -0.0194), (12, -824.0), (0, -0.0011)],
            ComparisonOp::Ge,
            -824.0011,
        ),
        (&[(11, 0.0155)], ComparisonOp::Eq, 0.0),
        (&[(5, 0.0155)], ComparisonOp::Eq, 0.0),
    ];
    for (terms, op, rhs) in rows {
        let terms: Vec<_> = terms.iter().map(|&(j, a)| (x[j], a)).collect();
        problem.add_constraint(terms, op, rhs);
    }
    let mut sol = engine_of(&problem);
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert!(sol.held.is_empty(), "{:?}", sol.held);
    assert_eq!(sol.cur_obj_val, -6.895199995022267);
    sol.values_dirty = true;
    sol.verify_primal().unwrap();
    assert!(
        sol.undone.is_some(),
        "fixture: the answer's refined values miss a bound"
    );
}

/// Values one ulp apart are no evidence that the computed ones fall short of
/// the vertex: rounding `x = 1e12`, with no bound and a cost of one, moves the
/// objective by `1.2e-4`, far beyond its bound tolerance's `1e-7`, and the
/// allowance covers it.
#[test]
fn values_an_ulp_apart_gain_nothing_beyond_the_allowance() {
    init();
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY));
    let y = problem.add_var(0.0, (0.0, 1.0));
    problem.add_constraint([(x, 1.0), (y, -1.0)], ComparisonOp::Eq, 1e12);
    let mut sol = engine_of(&problem);
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    let VarState::Basic(pos) = sol.var_states[0] else {
        panic!("fixture: x is basic");
    };
    assert_eq!(sol.basic_var_vals[pos], 1e12, "fixture");
    let mut refined = sol.basic_var_vals.clone();
    refined[pos] = refined[pos].next_down();
    let (gain, allowance) = sol.refinement_gain(&refined);
    assert!(
        gain > 1e-4 && gain <= allowance,
        "gain {gain:e}, allowance {allowance:e}"
    );
}

/// A row presolve took over (a validation row) that a finished LP's answer
/// misses is no error: the engine takes the row over and resumes, so the
/// answer holds the model as written. Planted: `max x + y` over `[0, 2]^2`
/// with the engine's row `x + 2 y <= 4` and `x + y <= 2.5` held back, as a
/// wrong reduction would: the engine's optimum `(2, 1)` misses it by `0.5`,
/// and the answer is the model's optimum, `2.5`. Held back with
/// `x + y <= -1`, the model as written is infeasible, and so is the answer.
/// Taken over at a deadline already passed, the re-solve stops before its
/// first step: the answer is interrupted, and resumed, the optimum.
#[test]
fn a_validation_row_the_answer_misses_is_taken_over() {
    init();
    let solve = |held_back: f64, stopped: bool| {
        let mut sol = Solver::try_new(
            &[-1.0, -1.0],
            &[0.0, 0.0],
            &[2.0, 2.0],
            &[(to_sparse(&[1.0, 2.0]), ComparisonOp::Le, 4.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        sol.set_validation_rows(vec![(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, held_back)]);
        let stop = sol.initial_solve().unwrap();
        assert_eq!((sol.get_value(0), sol.get_value(1)), (2.0, 1.0));
        if stopped {
            sol.deadline = Some(Instant::now());
        }
        crate::SolveOutcome::from_lp_stop(
            OptimizationDirection::Maximize,
            2,
            stop,
            Box::new(sol),
            crate::ResumeOptions::default(),
        )
    };
    let solution = solve(2.5, false).unwrap().into_solution().unwrap();
    assert!((solution.objective() - 2.5).abs() < 1e-9, "{solution:?}");
    assert!(solve(-1.0, false).is_err_and(|e| e == Error::Infeasible));
    let interrupted = solve(2.5, true).unwrap();
    assert_eq!(
        interrupted.termination_reason(),
        crate::TerminationReason::TimeLimit
    );
    let solution = interrupted.resume().unwrap().into_solution().unwrap();
    assert!((solution.objective() - 2.5).abs() < 1e-9, "{solution:?}");
}

/// The boundary sums a row in compensated arithmetic, so its verdict does not
/// depend on the order of the terms: `2^53 x0 + 0.75 (x1 + ... + x300) -
/// 2^53 x301 >= 225` holds exactly at `x = 1`, where a plain sum in the
/// written order drops every `0.75` against `2^53` and finds the row `225`
/// short, beyond its contract of `180` (the round-off floor of its magnitude,
/// `1.8e16`).
#[test]
fn the_boundary_check_does_not_depend_on_the_order_of_a_rows_terms() {
    let mut coeffs = vec![0.75; 302];
    coeffs[0] = 2f64.powi(53);
    coeffs[301] = -(2f64.powi(53));
    let row = to_sparse(&coeffs);
    let feasibility = crate::Tolerances::default().feasibility;
    assert_eq!(
        row_violation(&row, ComparisonOp::Ge, 225.0, |_| 1.0, feasibility),
        None
    );
    assert!(row_violation(&row, ComparisonOp::Ge, 226e3, |_| 1.0, feasibility).is_some());
}

/// `x0 + x1 + ... + x300 + y >= 2^50 + 75` over the vars of
/// [`long_row_solver`]. At `y = 0` the row is `37.5` short; a plain sum in
/// the written order rounds each `0.12500001` it adds to `2^50` up to
/// `0.25`, the spacing of `f64` there, and finds it held exactly. The row is
/// too long for that sum's round-off to fit its evaluation margin.
fn long_row() -> (CsVec, ComparisonOp, f64) {
    (
        to_sparse(&[1.0; 302]),
        ComparisonOp::Ge,
        2f64.powi(50) + 75.0,
    )
}

/// The solver of `rows` over `x0` fixed at `2^50`, `x1, ..., x300` fixed at
/// `0.12500001`, and `y ∈ [0, 100]` at cost one.
fn long_row_solver(rows: &[(CsVec, ComparisonOp, f64)]) -> Solver {
    let mut mins = vec![2f64.powi(50)];
    mins.extend([0.12500001; 300]);
    let mut maxs = mins.clone();
    mins.push(0.0);
    maxs.push(100.0);
    let mut costs = vec![0.0; 302];
    costs[301] = 1.0;
    Solver::try_new(
        &costs,
        &mins,
        &maxs,
        rows,
        &vec![VarDomain::Real; 302],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap()
}

/// The constructor evaluates a row as `check_rows` does: the slack of
/// [`long_row`] starts at the row's shortfall, summed in compensated
/// arithmetic, with the value and tolerances `check_rows` computes, bit for
/// bit. So the starting values are verified as they stand: a verification
/// leaves them as they are.
#[test]
fn the_constructor_evaluates_a_long_row_as_check_rows_does() {
    init();
    let sol = long_row_solver(&[long_row()]);
    assert_eq!(sol.row_scales, [1.0], "fixture: the row is not scaled");
    let (lhs, magnitude) = sol.row_activity(0, |v| sol.internal_value(v));
    assert!(
        !plain_sum_resolves(sol.feasibility, magnitude, sol.row_terms(0))
            && sol.orig_rhs[0] - lhs == 0.0,
        "fixture: a row too long for the plain sum, which finds it held"
    );
    let slack = sol.num_vars;
    let value = sol.basic_var_vals[0];
    assert!((value - 37.5).abs() < 1e-5, "the row's shortfall: {value}");
    assert!(!sol.is_primal_feasible);

    let mut checked = sol.clone();
    assert!(checked.check_rows());
    assert_eq!(checked.basic_var_vals[0].to_bits(), value.to_bits());
    assert_eq!(checked.var_tols[slack], sol.var_tols[slack]);

    assert!(!sol.values_dirty);
    let mut verified = sol.clone();
    verified.values_dirty = true;
    assert_eq!(
        verified.verify_primal().unwrap(),
        Verdict::Checked { feasible: false }
    );
    assert_eq!(verified.basic_var_vals[0].to_bits(), value.to_bits());
    assert_eq!(verified.var_tols[slack], sol.var_tols[slack]);
}

/// `add_constraint` evaluates the row it adds as `check_rows` does: added to
/// `y <= 50` at its optimum, [`long_row`] gets its shortfall as its slack's
/// value, with the tolerances `check_rows` computes, bit for bit. With a
/// deadline already passed, the re-solve stops before its first step, on
/// that value.
#[test]
fn add_constraint_evaluates_a_long_row_as_check_rows_does() {
    init();
    let mut y = vec![0.0; 302];
    y[301] = 1.0;
    let mut sol = long_row_solver(&[(to_sparse(&y), ComparisonOp::Le, 50.0)]);
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    sol.deadline = Some(Instant::now());
    let (coeffs, op, rhs) = long_row();
    assert_eq!(
        sol.add_constraint(coeffs, op, rhs).unwrap(),
        StopReason::Limit
    );
    let slack = sol.num_vars + 1;
    let VarState::Basic(pos) = sol.var_states[slack] else {
        panic!("the new slack is basic");
    };
    let value = sol.basic_var_vals[pos];
    assert!((value - 37.5).abs() < 1e-5, "the row's shortfall: {value}");

    let mut checked = sol.clone();
    assert!(checked.check_rows());
    assert_eq!(checked.basic_var_vals[pos].to_bits(), value.to_bits());
    assert_eq!(checked.var_tols[slack], sol.var_tols[slack]);
}

/// A row's slack as the plain sum gives it and as the compensated one does,
/// with how far beyond the slack's edge each puts it and the row's evaluation
/// margin, at the current values of `sol` (row 0).
fn plain_and_compensated_slack(sol: &Solver) -> ((f64, f64), (f64, f64), f64) {
    let (lhs, magnitude) = sol.row_activity(0, |v| sol.internal_value(v));
    let plain = sol.orig_rhs[0] - lhs;
    let row = sol.orig_constraints.outer_view(0).unwrap();
    let terms = row
        .iter()
        .filter(|&(v, _)| v < sol.num_vars)
        .map(|(v, &coeff)| (coeff, sol.internal_value(v)));
    let compensated = implied_slack(sol.orig_rhs[0], terms).0;
    let tols = sol.var_tols[sol.num_vars];
    let margin = slack_margin(
        sol.feasibility * sol.row_scales[0],
        magnitude,
        sol.row_terms(0),
    );
    (
        (plain, sol.slack_excess(0, plain, tols)),
        (compensated, sol.slack_excess(0, compensated, tols)),
        margin,
    )
}

/// A plain sum that puts a row's slack beyond its edge by at most twice the
/// row's evaluation margin is summed again, compensated, and the slack takes
/// that value: `-1125 x0 - 1.125e-7 x1 = -5876999.999999887` at `x0 = 5224`,
/// `x1 = -2` (both fixed), which the plain sum puts `3.7e-10` beyond the
/// row's room, its exact slack `1.02e-11` inside it. A plain acceptance
/// stands: `x0 + x1 + x2 >= 348952339.66500664` at `348952339`, `0.665` and
/// `4.594137084480981e-8` keeps the plain sum's slack, `3.5e-8` inside the
/// room, which the compensated sum puts `2e-21` beyond it.
#[test]
fn a_plain_rejection_near_the_edge_is_summed_again_compensated() {
    init();
    let fixed = |values: &[f64], row: (CsVec, ComparisonOp, f64)| {
        Solver::try_new(
            &vec![0.0; values.len()],
            values,
            values,
            &[row],
            &vec![VarDomain::Real; values.len()],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap()
    };

    let sol = fixed(
        &[5224.0, -2.0],
        (
            to_sparse(&[-1125.0, -1.125e-7]),
            ComparisonOp::Eq,
            -5876999.999999887,
        ),
    );
    let ((plain, plain_beyond), (compensated, compensated_beyond), margin) =
        plain_and_compensated_slack(&sol);
    assert!(
        plain_beyond > 0.0 && plain_beyond <= 2.0 * margin && compensated_beyond < 0.0,
        "fixture: the plain sum {plain:e} rejects the slack near its edge ({plain_beyond:e} \
         beyond, margin {margin:e}), the compensated one {compensated:e} accepts it"
    );
    assert_eq!(sol.basic_var_vals[0].to_bits(), compensated.to_bits());
    assert_eq!(sol.calc_primal_infeasibility().0, 0);
    let mut checked = sol.clone();
    assert!(checked.check_rows());
    assert_eq!(checked.basic_var_vals[0].to_bits(), compensated.to_bits());

    let sol = fixed(
        &[348952339.0, 0.665, 4.594137084480981e-8],
        (
            to_sparse(&[1.0, 1.0, 1.0]),
            ComparisonOp::Ge,
            348952339.66500664,
        ),
    );
    let ((plain, plain_beyond), (compensated, compensated_beyond), _) =
        plain_and_compensated_slack(&sol);
    assert!(
        plain_beyond < 0.0 && compensated_beyond > 0.0,
        "fixture: the plain sum {plain:e} accepts the slack, the compensated one \
         {compensated:e} rejects it"
    );
    assert_eq!(sol.basic_var_vals[0].to_bits(), plain.to_bits());
    assert_eq!(sol.calc_primal_infeasibility().0, 0);
}

/// A row every point satisfies, as an infinite right-hand side on the side
/// the row leaves open makes it, is violated nowhere: the boundary's check
/// agrees with the verdict the row gets before any point is looked at, so
/// an edit keeps an incumbent such a row cannot exclude.
#[test]
fn a_row_with_an_infinite_right_hand_side_on_its_open_side_holds_everywhere() {
    let row = to_sparse(&[1.0, 2.0]);
    let feasibility = crate::Tolerances::default().feasibility;
    assert_eq!(
        row_violation(&row, ComparisonOp::Le, f64::INFINITY, |_| 1.0, feasibility),
        None
    );
    assert_eq!(
        row_violation(
            &row,
            ComparisonOp::Ge,
            f64::NEG_INFINITY,
            |_| 1.0,
            feasibility
        ),
        None
    );
    assert!(row_violation(&row, ComparisonOp::Ge, f64::INFINITY, |_| 1.0, feasibility).is_some());
}

/// The boundary rounds a product whose factor Dekker's split would overflow
/// to a NaN, which holds nothing: `x + y + 1e305 f >= 3` holds at
/// `x = y = 1`, `f = 1e-305`.
#[test]
fn the_boundary_check_rounds_a_product_beyond_the_split_limit() {
    let row = to_sparse(&[1.0, 1.0, 1e305]);
    let value = |v: usize| if v == 2 { 1e-305 } else { 1.0 };
    let feasibility = crate::Tolerances::default().feasibility;
    assert_eq!(
        row_violation(&row, ComparisonOp::Ge, 3.0, value, feasibility),
        None
    );
    assert!(row_violation(&row, ComparisonOp::Ge, 3.1, value, feasibility).is_some());
}

/// `add_constraint` repairs a basis its factorization finds singular, and
/// recomputes the values before the re-solve. With the plant, adding `x <= 3`
/// factorizes `{x, y}` and the new row's slack: singular, and repaired. The
/// re-solve reaches the planted model's optimum within its rows; with a
/// deadline already passed, it stops before its first step, on values that
/// satisfy the rows.
#[test]
fn add_constraint_repairs_a_basis_its_factorization_finds_singular() {
    init();
    for stopped in [false, true] {
        let mut sol = optimum_with_x_and_y_basic();
        plant_y_parallel_to_x(&mut sol);
        if stopped {
            sol.deadline = Some(Instant::now());
        }
        let stop = sol
            .add_constraint(to_sparse(&[1.0, 0.0]), ComparisonOp::Le, 3.0)
            .unwrap();
        assert!(
            (0..2).any(|v| matches!(sol.var_states[v], VarState::NonBasic(_))),
            "the repair takes x or y out"
        );
        if stopped {
            assert_eq!(stop, StopReason::Limit);
            assert!(sol.check_rows(), "the rows miss at the values");
        } else {
            assert_eq!(stop, StopReason::Finished);
            assert!(float_eq(sol.cur_obj_val, 0.0));
            let x = [sol.get_value(0), sol.get_value(1)];
            assert!(sol.first_violated_row(&x).is_none());
        }
    }
}

/// A var basic in two positions makes the basis singular, and a repair would
/// hide the bug: it would take one copy out and leave the var basic and
/// non-basic at once. At the optimum, one basic var is planted in the other's
/// position, and the rebuild fails loudly.
#[test]
#[should_panic(expected = "basis bookkeeping is corrupt")]
fn a_singular_basis_with_a_var_basic_twice_is_not_repaired() {
    init();
    let mut sol = optimum_with_x_and_y_basic();
    sol.basic_vars[1] = sol.basic_vars[0];
    let _ = sol.rebuild();
}

/// The same check on the non-basic side: at the optimum, `y` is made parallel
/// to `x`, a genuine singular basis, and two non-basic positions are swapped,
/// so the slack the repair takes in names a position another var holds.
#[test]
#[should_panic(expected = "basis bookkeeping is corrupt")]
fn a_singular_basis_with_a_slack_out_of_place_is_not_repaired() {
    init();
    let mut sol = optimum_with_x_and_y_basic();
    plant_y_parallel_to_x(&mut sol);
    sol.nb_vars.swap(0, 1);
    let _ = sol.rebuild();
}

/// Tied breakpoints form one Harris group, whose largest entry is the pivot,
/// with nothing flipped: in `8 x1 + x2 / 8 >= 10` (equilibrated to
/// `x1 + x2 / 64 >= 1.25`), with both vars at their lower bound and zero
/// cost, the pivot is `x1` (entry 1), not `x2` (entry 1/64) after flipping
/// `x1` on a zero dual step.
#[test]
fn dual_ratio_test_pivots_on_the_largest_entry_of_a_tied_group() {
    init();
    let new_solver = || {
        Solver::try_new(
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[1.0, 64.0],
            &[(to_sparse(&[8.0, 0.125]), ComparisonOp::Ge, 10.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap()
    };
    let mut sol = new_solver();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    let step = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step();
    assert!(step.flips.is_empty(), "flips {:?}", step.flips);
    let (col, coeff) = step.entering;
    assert_eq!((sol.nb_vars[col], coeff), (0, 1.0));

    let mut sol = new_solver();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    assert_eq!([sol.get_value(0), sol.get_value(1)], [1.0, 16.0]);
}

/// The same row with `x2` costing 1: the first Harris group holds `x1` alone
/// and cannot carry the row (capacity 1 of 1.25), and the next group's only
/// entry, 1/64, is below a tenth of the largest grouped entry, so the pivot
/// steps back to `x1` with nothing flipped. `x1` enters past its other bound,
/// and the next iteration moves the rest of the violation onto `x2`.
#[test]
fn dual_ratio_test_steps_back_to_a_group_with_a_stable_pivot() {
    init();
    let new_solver = || {
        Solver::try_new(
            &[0.0, 1.0],
            &[0.0, 0.0],
            &[1.0, 64.0],
            &[(to_sparse(&[8.0, 0.125]), ComparisonOp::Ge, 10.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap()
    };
    let mut sol = new_solver();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    let step = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step();
    assert!(step.flips.is_empty(), "flips {:?}", step.flips);
    let (col, coeff) = step.entering;
    assert_eq!((sol.nb_vars[col], coeff), (0, 1.0));

    let mut sol = new_solver();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    assert_eq!([sol.get_value(0), sol.get_value(1)], [1.0, 16.0]);
    assert_eq!(sol.cur_obj_val, 16.0);
}

/// The violated row `x0 + x1 + x2 >= 5` of `min x2 - x0`, `x0 ∈ [-1, 0]`,
/// `x1, x2 >= 0`, from the slack basis, with the matrix entries of the three
/// vars in the row planted as `entries` (in the row-wise and the column-wise
/// copy alike, so that every computation of the row agrees): `x0` sits at its
/// upper bound and cannot help, `x1` helps with a reduced cost of zero (its
/// breakpoint comes first) and `x2` with one of one. The row is computed for
/// the dual ratio test; returns the solver, the row and the bound its basic
/// var leaves at.
fn planted_dual_row(entries: [f64; 3]) -> (Solver, usize, f64) {
    let mut sol = Solver::try_new(
        &[-1.0, 0.0, 1.0],
        &[-1.0, 0.0, 0.0],
        &[0.0, f64::INFINITY, f64::INFINITY],
        &[(to_sparse(&[1.0, 1.0, 1.0]), ComparisonOp::Ge, 5.0)],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.col_scales, [1.0; 3], "fixture: no column scaling");
    assert_eq!(sol.row_scales, [1.0], "fixture: no row scaling");
    for (var, &entry) in entries.iter().enumerate() {
        *sol.orig_constraints.get_mut(0, var).unwrap() = entry;
        *sol.orig_constraints_csc.get_mut(0, var).unwrap() = entry;
    }
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    (sol, row, leaving_new_val)
}

/// An entry of the dual ratio test's row at or below the pivot tolerance is
/// no candidate while a larger one can enter, however small the other
/// helping entries are: planted row `[1, 5e-10, 2e-9]`, where `x1`'s entry has
/// the first breakpoint and `x2` enters. A floor relative to the largest
/// helping entry made `x1`'s a candidate, a pivot of `5e-10`. The tolerance is
/// `1e-9` on a fresh factorization, larger the longer the eta file.
#[test]
fn a_dual_entry_at_or_below_the_pivot_tolerance_gives_way_to_a_larger_one() {
    init();
    let (mut sol, row, leaving_new_val) = planted_dual_row([1.0, 5e-10, 2e-9]);
    let (col, coeff) = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step()
        .entering;
    assert_eq!((sol.nb_vars[col], coeff), (2, 2e-9));
    assert_eq!(
        [0, 9, 10, 19, 20].map(dual_pivot_tolerance),
        [1e-9, 1e-9, 3e-8, 3e-8, 1e-6]
    );
    assert_eq!(
        [0, 9, 10, 19, 20].map(primal_pivot_tolerance),
        [1e-9, 1e-9, 1e-8, 1e-8, 1e-7]
    );
}

/// Where no entry above the pivot tolerance can enter and the row proves
/// nothing, an entry below it enters once the refined row shows it genuine:
/// planted row `[1, 5e-10, 0]`, where `x1`'s `5e-10` alone can repair the
/// row (unbounded above, its coefficient voids the proof). The regular test
/// leaves it out; the refined row, summed from the data, has it beyond the
/// error the refinement measured. Planted as `5e-10` in the computed row over
/// data of zero, the entry is round-off, which the refined row does not
/// have: no candidate.
#[test]
fn a_dual_entry_below_the_pivot_tolerance_enters_once_the_refined_row_shows_it_genuine() {
    init();
    let (mut sol, row, leaving_new_val) = planted_dual_row([1.0, 5e-10, 0.0]);
    assert!(matches!(
        sol.choose_entering_col_dual(row, leaving_new_val),
        Ok(DualRatioTest::NoCandidate)
    ));
    let leaving_up = leaving_new_val > sol.basic_var_vals[row];
    assert!(!sol.prove_infeasible(row, leaving_up).holds);
    let (col, coeff) = sol
        .genuine_entry_step(row, leaving_new_val)
        .unwrap()
        .step()
        .entering;
    assert_eq!((sol.nb_vars[col], coeff), (1, 5e-10));

    let (mut sol, row, leaving_new_val) = planted_dual_row([1.0, 0.0, 0.0]);
    let x1 = nb_col(&sol, 1);
    *sol.row_coeffs.get_mut(x1) = 5e-10;
    assert!(matches!(
        sol.choose_entering_col_dual(row, leaving_new_val),
        Ok(DualRatioTest::NoCandidate)
    ));
    assert!(matches!(
        sol.genuine_entry_step(row, leaving_new_val),
        Ok(DualRatioTest::NoCandidate)
    ));
}

/// A fixed var's entry sets no scale for the growth test: planted row
/// `[1e10, 1, 100]` over a fixed `f` and the `y` and `z` of `x + y + z >= 5`,
/// `min z`, where `y`'s breakpoint comes first. Measured against `f`'s
/// `1e10`, `y`'s one would be small and left out, and `z`, whose reduced cost
/// the step takes past zero, would enter in its place.
#[test]
fn a_fixed_vars_entry_sets_no_growth_scale() {
    init();
    let mut sol = Solver::try_new(
        &[0.0, 0.0, 1.0],
        &[0.0, 0.0, 0.0],
        &[0.0, f64::INFINITY, f64::INFINITY],
        &[(to_sparse(&[1.0, 1.0, 1.0]), ComparisonOp::Ge, 5.0)],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.col_scales, [1.0; 3], "fixture: no column scaling");
    assert_eq!(sol.row_scales, [1.0], "fixture: no row scaling");
    for (var, entry) in [1e10, 1.0, 100.0].into_iter().enumerate() {
        *sol.orig_constraints.get_mut(0, var).unwrap() = entry;
        *sol.orig_constraints_csc.get_mut(0, var).unwrap() = entry;
    }
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    let (col, coeff) = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step()
        .entering;
    assert_eq!((sol.nb_vars[col], coeff), (1, 1.0));
}

/// A pivot that fails the growth test (at most `1e-9` of the row's largest
/// entry) is not passed over before the row has been refined and its entries
/// summed again in compensated arithmetic: planted row `[1e3, 1, 0]`, with
/// `x1`'s entry in the computed row then planted as `1e-7`, as if the
/// transposed solve had lost it. The refined row has it back at one, and `x1`
/// enters on it; the refined row of the basis inverse is what the
/// steepest-edge update reads. A floor relative to the largest helping entry
/// took the `1e-7`.
#[test]
fn a_small_dual_pivot_is_refined_before_it_is_passed_over() {
    init();
    let (mut sol, row, leaving_new_val) = planted_dual_row([1e3, 1.0, 0.0]);
    let x1 = nb_col(&sol, 1);
    *sol.row_coeffs.get_mut(x1) = 1e-7;
    let step = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step();
    assert!(step.flips.is_empty());
    assert_eq!(step.entering, (x1, 1.0));
    assert_eq!(*sol.row_coeffs.get(x1), 1.0);
    let rho: Vec<(usize, f64)> = sol
        .inv_basis_row_coeffs
        .iter()
        .map(|(r, &v)| (r, v))
        .collect();
    assert_eq!(rho, [(row, 1.0)]);
}

/// A pivot that still fails the growth test on the refined row is left out,
/// and the ratio test runs again without it: planted row `[1e6, 1e-5, 1]`,
/// whose first breakpoint is `x1`'s, `1e-11` of the row's largest entry.
/// `x2` enters instead, its entry `1e-6` of the largest. A floor relative to
/// the largest helping entry took `x1`'s `1e-5` beside the `1e6`. Where no
/// other can enter, the regular test has no candidate, and the small pivot,
/// genuine on the refined row, is the fallback's: the row `[1e6, 1e-5, 0]`.
#[test]
fn a_dual_pivot_small_beside_its_row_is_left_out_for_the_next() {
    init();
    let (mut sol, row, leaving_new_val) = planted_dual_row([1e6, 1e-5, 1.0]);
    let step = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step();
    assert!(step.flips.is_empty());
    let (col, coeff) = step.entering;
    assert_eq!((sol.nb_vars[col], coeff), (2, 1.0));

    let (mut sol, row, leaving_new_val) = planted_dual_row([1e6, 1e-5, 0.0]);
    assert!(matches!(
        sol.choose_entering_col_dual(row, leaving_new_val),
        Ok(DualRatioTest::NoCandidate)
    ));
    let (col, coeff) = sol
        .genuine_entry_step(row, leaving_new_val)
        .unwrap()
        .step()
        .entering;
    assert_eq!((sol.nb_vars[col], coeff), (1, 1e-5));
}

/// A dual pivot puts the column's element in the eta file, so the updated
/// basis holds the entering column `a_q` exactly (the row's would make it
/// `a_q + (a_row - a_col) b_r`); the primal step and the steepest-edge weight
/// use it too, and only the reduced costs use the row's element. Planted: the
/// row's element `1e-9` (relative) off the column's, within the agreement check.
#[test]
fn dual_pivot_applies_the_column_element_to_the_eta_file() {
    init();
    // min x + y s.t. x + 2y >= 4, 3x + y >= 3, boxes [0, 10]: the start
    // (both at zero) is dual feasible and violates both rows.
    let mut sol = Solver::try_new(
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[10.0, 10.0],
        &[
            (to_sparse(&[1.0, 2.0]), ComparisonOp::Ge, 4.0),
            (to_sparse(&[3.0, 1.0]), ComparisonOp::Ge, 3.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    let step = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step();
    let (col, a_row) = step.entering;
    sol.calc_col_coeffs(col);
    let a_col = sol.col_coeff_at(row);
    assert_eq!(
        a_row, a_col,
        "fixture: a fresh slack basis computes both alike"
    );
    let planted_row = a_col * (1.0 + 1e-9);
    *sol.row_coeffs.get_mut(col) = planted_row;
    let entering_var = sol.nb_vars[col];
    let d_q = sol.nb_var_obj_coeffs[col];
    let rho_sq_norm = sol.inv_basis_row_coeffs.sq_norm();
    let entering_diff = (sol.basic_var_vals[row] - leaving_new_val) / a_col;
    sol.pivot(&PivotInfo {
        col,
        entering_new_val: sol.nb_var_vals[col] + entering_diff,
        entering_diff,
        elem: Some(PivotElem {
            row,
            coeff: a_col,
            dual_step_coeff: planted_row,
            leaving_new_val,
        }),
    })
    .unwrap();
    // The eta file represents the basis the column describes.
    let a_q: Vec<(usize, f64)> = sol
        .orig_constraints_csc
        .outer_view(entering_var)
        .unwrap()
        .iter()
        .map(|(r, &v)| (r, v))
        .collect();
    let num_rows = sol.num_constraints();
    let unit = sol.basis_solver.solve(a_q.iter().map(|(r, v)| (*r, v)));
    for r in 0..num_rows {
        let expected = if r == row { 1.0 } else { 0.0 };
        assert!(
            (unit.get(r) - expected).abs() <= 1e-15,
            "B^-1 a_q at row {r}: {:e}",
            unit.get(r)
        );
    }
    // The leaving var's reduced cost is the dual step along the row.
    assert_eq!(sol.nb_var_obj_coeffs[col], -d_q / planted_row);
    // The pivot row's steepest-edge weight is exact for the new basis.
    assert_eq!(sol.dual_edge_sq_norms[row], rho_sq_norm / (a_col * a_col));
}

/// A genuine blocking entry of the primal ratio test's column bounds the
/// step beside huge entries in rows that cannot block: `max x` is `x = 8`.
/// Basis: the free `y1`, `y2` of the nearly dependent `y1 + y2 + x/8 = 0` and
/// `y1 + (1 + 2^-43) y2 + 8x = 0` (7e13 in `x`'s column, never blocking), and
/// the slack of `x/8 + 8z <= 1`, whose entry, 1/64, is exact (the basis is
/// block diagonal). A floor relative to the column's largest entry took that
/// entry for round-off, and the column for an unbounded ray.
#[test]
fn a_primal_blocking_entry_small_beside_its_column_bounds_the_step() {
    init();
    let eps = 2.0f64.powi(-43);
    let mut sol = Solver::try_new(
        &[0.0, 0.0, -1.0, 0.0],
        &[f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY, f64::INFINITY, 1.0],
        &[
            (to_sparse(&[1.0, 1.0, 0.125, 0.0]), ComparisonOp::Eq, 0.0),
            (
                to_sparse(&[1.0, 1.0 + eps, 8.0, 0.0]),
                ComparisonOp::Eq,
                0.0,
            ),
            (to_sparse(&[0.0, 0.0, 0.125, 8.0]), ComparisonOp::Le, 1.0),
        ],
        &vec![VarDomain::Real; 4],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.col_scales, [1.0; 4], "fixture: no column scaling");
    let basis = Basis(vec![
        VarStatus::Basic,
        VarStatus::Basic,
        VarStatus::AtLower,
        VarStatus::AtLower,
        VarStatus::AtLower,
        VarStatus::AtLower,
        VarStatus::Basic,
    ]);
    sol.load_basis(&basis).unwrap();
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert!(
        (sol.get_value(2) - 8.0).abs() < 1e-9,
        "x = {}",
        sol.get_value(2)
    );
    assert!((sol.cur_obj_val + 8.0).abs() < 1e-9);
}

/// Dual steepest-edge weights are floored: a basis load resets them to one,
/// an underestimate of `||e_1^T B^-1||^2 = 4.61` for `B = [[1, 0], [1.9, 1]]`
/// (the basis `y`, `s1` of `y <= 3`, `1.9 y <= 10`), and the pivot that
/// repairs `y = 3 > 1` would update the second row's weight to
/// `1 - 1.9^2 = -2.61`. A weight that overflows or meets a NaN is an
/// `InternalError`, never floored finite.
#[test]
fn dual_steepest_edge_weights_are_floored_after_a_basis_load() {
    init();
    let new_solver = || {
        let mut sol = Solver::try_new(
            &[-1.0],
            &[0.0],
            &[1.0],
            &[
                (to_sparse(&[1.0]), ComparisonOp::Le, 3.0),
                (to_sparse(&[1.9]), ComparisonOp::Le, 10.0),
            ],
            &[VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        let basis = Basis(vec![VarStatus::Basic, VarStatus::AtLower, VarStatus::Basic]);
        sol.load_basis(&basis).unwrap();
        assert_eq!(sol.dual_edge_sq_norms, [1.0, 1.0]);
        sol
    };
    let mut sol = new_solver();
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!(sol.get_value(0), 1.0);
    assert!(
        sol.dual_edge_sq_norms
            .iter()
            .all(|&w| w >= MIN_DUAL_EDGE_WEIGHT),
        "weights {:?}",
        sol.dual_edge_sq_norms
    );

    let mut sol = new_solver();
    let (row, _) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    sol.calc_col_coeffs(0);
    let entry = |sol: &Solver, r: usize| {
        sol.col_coeffs
            .iter()
            .find_map(|(i, &c)| (i == r).then_some(c))
            .unwrap_or(0.0)
    };
    let pivot_coeff = entry(&sol, row);
    let mut planted = sol.clone();
    assert!(matches!(
        sol.update_dual_sq_norms(row, 1e-200),
        Err(Error::InternalError(_))
    ));
    let other = 1 - row;
    assert_ne!(entry(&planted, other), 0.0);
    planted.dual_edge_sq_norms[other] = f64::NAN;
    assert!(matches!(
        planted.update_dual_sq_norms(row, pivot_coeff),
        Err(Error::InternalError(_))
    ));
}

/// Primal steepest-edge weights are floored at `1 + lambda^2`, a lower bound
/// of the true weight: planted at -5, as cancellation can leave one, `x2`'s
/// weight would update to -2 at the first pivot of `min -x1 - x2` s.t.
/// `x1 + x2 <= 4`, `x1 - x2 <= 2`. A weight that overflows or meets a NaN is
/// an `InternalError`, never floored finite.
#[test]
fn primal_steepest_edge_weights_are_floored() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0),
            (to_sparse(&[1.0, -1.0]), ComparisonOp::Le, 2.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert!(sol.enable_primal_steepest_edge);
    let c2 = sol.nb_vars.iter().position(|&v| v == 1).unwrap();
    sol.primal_edge_sq_norms[c2] = -5.0;
    let PivotChoice::Pivot(info) = primal_pivot_choice(&mut sol, 1.0) else {
        panic!("x1 enters");
    };
    assert_eq!(sol.nb_vars[info.col], 0);
    let r_coeff = *sol.row_coeffs.get(c2);
    let pivot_coeff = info.elem.as_ref().unwrap().coeff;
    sol.pivot(&info).unwrap();
    let lambda = r_coeff / pivot_coeff;
    assert_eq!(sol.primal_edge_sq_norms[c2], 1.0 + lambda * lambda);

    let mut planted = sol.clone();
    assert!(matches!(
        sol.update_primal_sq_norms(info.col, 1e-200),
        Err(Error::InternalError(_))
    ));
    planted.primal_edge_sq_norms[c2] = f64::NAN;
    assert!(matches!(
        planted.update_primal_sq_norms(info.col, pivot_coeff),
        Err(Error::InternalError(_))
    ));
}

/// A reduced cost that is not a number never ends a phase as optimal: it
/// counts as dual infeasible, and its steepest-edge score, a NaN too, would
/// lose every comparison, leaving no entering column and the vertex taken for
/// optimal. Planted: a NaN in a verified optimum's reduced costs, which a
/// primal phase then prices (the verification would recompute them).
#[test]
fn nan_reduced_cost_is_an_error_not_an_optimum() {
    init();
    let mut sol = Solver::try_new(
        &[1.0, 2.0],
        &[0.0, 0.0],
        &[10.0, 10.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 1.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    sol.nb_var_obj_coeffs[0] = f64::NAN;
    assert!(matches!(
        sol.optimize(true, 1.0),
        Err(Error::InternalError(_))
    ));
}

/// The same for a fixed var, dual feasible whatever its reduced cost as long
/// as it is a number: a NaN still counts as dual infeasible, and the pricing
/// reports it. Planted as above.
#[test]
fn nan_reduced_cost_of_a_fixed_var_is_an_error_not_an_optimum() {
    init();
    let mut sol = Solver::try_new(
        &[1.0, 2.0, 3.0],
        &[0.0, 0.0, 1.0],
        &[10.0, 10.0, 1.0],
        &[(to_sparse(&[1.0, 1.0, 1.0]), ComparisonOp::Ge, 2.0)],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    let col = nb_col(&sol, 2);
    sol.nb_var_obj_coeffs[col] = f64::NAN;
    assert!(matches!(
        sol.optimize(true, 1.0),
        Err(Error::InternalError(_))
    ));
}

/// The non-basic column of `var`.
fn nb_col(sol: &Solver, var: usize) -> usize {
    sol.nb_vars.iter().position(|&v| v == var).unwrap()
}

/// The primal simplex's next choice, as its phase makes it: the pricing, then
/// the ratio test for the priced column.
fn primal_pivot_choice(sol: &mut Solver, harris_scale: f64) -> PivotChoice {
    let col = sol.price_primal().unwrap().expect("a var can enter");
    sol.choose_pivot(col, harris_scale).unwrap()
}

/// A dual step never runs backwards: where the Harris pass lets the entering
/// var's reduced cost lie just across zero, the step is zero and the var's
/// cost is shifted. In `8 x0 + x1 >= 1e-4` (equilibrated to
/// `x0 + x1 / 8 >= 1.25e-5`), `x1`'s reduced cost is planted at `-9e-11`,
/// within its tolerance of `1e-10`: a step `-9e-11 / 0.125` would leave the
/// slack a reduced cost of `+7.2e-10`, dual infeasible beyond its tolerance.
#[test]
fn dual_step_is_zero_where_the_entering_reduced_cost_has_crossed_zero() {
    init();
    let new_solver = || {
        let mut sol = Solver::try_new(
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[10.0, f64::INFINITY],
            &[(to_sparse(&[8.0, 1.0]), ComparisonOp::Ge, 1e-4)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        let c1 = nb_col(&sol, 1);
        sol.nb_var_obj_coeffs[c1] = -9e-11;
        sol
    };

    let sol = new_solver();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    let mut sol = sol;
    sol.calc_row_coeffs(row);
    let step = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step();
    assert_eq!(sol.nb_vars[step.entering.0], 1);
    assert!(step.zero_step && step.flips.is_empty());

    // The phase: one pivot, on a zero step. The start is near optimal, so the
    // costs are not perturbed and the ratio test sees the planted reduced cost.
    let mut sol = new_solver();
    let x0_reduced_cost = sol.nb_var_obj_coeffs[nb_col(&sol, 0)];
    sol.begin_dual_phase(false).unwrap();
    assert_eq!(sol.dual_phase().unwrap(), StopReason::Finished);
    assert!(matches!(sol.var_states[1], VarState::Basic(_)));
    assert_eq!(sol.nb_var_obj_coeffs[nb_col(&sol, 2)], 0.0);
    assert_eq!(sol.nb_var_obj_coeffs[nb_col(&sol, 0)], x0_reduced_cost);
    assert_eq!(sol.cost_shifts[1], 9e-11);
    assert_eq!(sol.calc_dual_infeasibility().0, 0);
    // The cleanup gives every shift back; the real costs' reduced costs,
    // recomputed as `run_phases` does, are dual feasible.
    sol.end_dual_phase();
    assert!(!sol.costs_modified && sol.reduced_costs_dirty);
    assert!(sol
        .cost_shifts
        .iter()
        .chain(&sol.cost_perturbation)
        .all(|&s| s == 0.0));
    sol.recalc_obj_coeffs().unwrap();
    assert_eq!(sol.calc_dual_infeasibility().0, 0);

    // The same through the dual phase as `run_phases` runs it.
    let mut sol = new_solver();
    assert_eq!(
        sol.restore_feasibility(false).unwrap(),
        StopReason::Finished
    );
    assert_eq!(sol.nb_var_obj_coeffs[nb_col(&sol, 2)], 0.0);
}

/// A dual phase that starts dual infeasible first corrects its reduced costs:
/// a boxed var moves to the bound where its reduced cost has the right sign,
/// and any other var (a boxed one too, in a cleanup phase) has its cost
/// shifted until the reduced cost is just feasible. Loaded at the lower
/// bounds, `x0` (boxed) and `x3` (bounded below only) have reduced cost -1:
/// the phase moves `x0` to 5 first and gives `x3`'s shift back at its end,
/// for the primal phase to take `x3` to 7.
#[test]
fn dual_phase_starts_by_correcting_its_dual_infeasibilities() {
    init();
    let new_solver = || {
        let mut sol = Solver::try_new(
            &[-1.0, 1.0, 1.0, -1.0],
            &[0.0, 0.0, 0.0, -3.0],
            &[5.0, f64::INFINITY, 10.0, f64::INFINITY],
            &[
                (to_sparse(&[0.0, 0.0, 1.0, 0.0]), ComparisonOp::Ge, 1.0),
                (to_sparse(&[0.0, 0.0, 0.0, 1.0]), ComparisonOp::Le, 7.0),
            ],
            &vec![VarDomain::Real; 4],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        // Solved first, so that the real objective is in force.
        assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
        let at_lower = Basis(vec![
            VarStatus::AtLower,
            VarStatus::AtLower,
            VarStatus::AtLower,
            VarStatus::AtLower,
            VarStatus::Basic,
            VarStatus::Basic,
        ]);
        sol.load_basis(&at_lower).unwrap();
        assert!(!sol.is_primal_feasible && sol.calc_dual_infeasibility().0 > 0);
        sol
    };

    let mut sol = new_solver();
    sol.begin_dual_phase(false).unwrap();
    assert_eq!(sol.get_value(0), 5.0);
    let c3 = nb_col(&sol, 3);
    let tol = sol.dual_tols[3];
    assert_eq!(sol.get_value(3), -3.0);
    assert!(sol.nb_var_obj_coeffs[c3] >= tol && sol.nb_var_obj_coeffs[c3] <= 2.0 * tol);
    assert_eq!(sol.calc_dual_infeasibility().0, 0);
    assert_eq!(sol.dual_phase().unwrap(), StopReason::Finished);
    sol.end_dual_phase();
    assert_eq!(sol.get_value(0), 5.0);
    // With the shift given back, `x3`'s reduced cost is -1 at its lower
    // bound again: the dual infeasibility the primal phase removes.
    sol.recalc_obj_coeffs().unwrap();
    assert_eq!(sol.calc_dual_infeasibility().0, 1);

    // The dual phase as `run_phases` runs it, and the whole re-solve.
    let mut sol = new_solver();
    assert_eq!(
        sol.restore_feasibility(false).unwrap(),
        StopReason::Finished
    );
    assert_eq!(sol.get_value(0), 5.0);
    let mut sol = new_solver();
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!(
        (0..4).map(|v| sol.get_value(v)).collect::<Vec<_>>(),
        [5.0, 0.0, 1.0, 7.0]
    );
    assert_eq!(sol.cur_obj_val, -11.0);

    // A dual phase that cleans up after another corrects by shifts alone: the
    // boxed `x0` keeps its bound and has its cost shifted like any other var.
    let mut sol = new_solver();
    sol.begin_dual_phase(true).unwrap();
    assert_eq!(sol.get_value(0), 0.0);
    let c0 = nb_col(&sol, 0);
    let tol = sol.dual_tols[0];
    assert!(sol.nb_var_obj_coeffs[c0] >= tol && sol.nb_var_obj_coeffs[c0] <= 2.0 * tol);
    assert_eq!(sol.calc_dual_infeasibility().0, 0);
    assert!(sol.cost_perturbation[0] > 0.0 && sol.cost_perturbation[1] == 0.0);
    sol.end_dual_phase();
}

/// The cost perturbation of a dual phase: the base is `5e-7` times the
/// largest cost (its fourth root above 100); a structural var's cost moves by
/// `(1 + r) (|c| + 1) base` towards dual feasibility at the bound it can sit
/// at (up bounded below only, down bounded above only, by the cost's sign
/// when boxed), a free or fixed var's not at all, a slack's by at most `5e-13`.
#[test]
fn cost_perturbation_follows_the_bound_types() {
    init();
    let costs = [1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10_000.0];
    let mut sol = Solver::try_new(
        &costs,
        &[
            0.0,
            f64::NEG_INFINITY,
            0.0,
            0.0,
            f64::NEG_INFINITY,
            2.0,
            0.0,
        ],
        &[f64::INFINITY, 0.0, 1.0, 1.0, f64::INFINITY, 2.0, 1.0],
        &[(to_sparse(&[1.0; 7]), ComparisonOp::Le, 100.0)],
        &vec![VarDomain::Real; 7],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    sol.perturb_costs();
    assert!(sol.costs_modified);
    // max |c| = 1e4 > 100: the base is 5e-7 * 1e4^(1/4) = 5e-6.
    let base = 5e-7 * 10.0;
    let p = &sol.cost_perturbation;
    for (var, sign) in [(0, 1.0), (1, -1.0), (2, -1.0), (3, 1.0), (6, 1.0)] {
        let amount = sign * p[var] / ((costs[var].abs() + 1.0) * base);
        assert!((1.0..2.0).contains(&amount), "var {var}: {}", p[var]);
    }
    // Free (4) and fixed (5) vars keep their costs.
    assert_eq!((p[4], p[5]), (0.0, 0.0));
    assert!(p[7] != 0.0 && p[7].abs() <= 5e-13);

    // An objective of zeros is perturbed as if its largest cost were one.
    let mut sol = Solver::try_new(
        &[0.0],
        &[0.0],
        &[1.0],
        &[(to_sparse(&[1.0]), ComparisonOp::Ge, 0.5)],
        &[VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    sol.perturb_costs();
    assert!((5e-7..1e-6).contains(&sol.cost_perturbation[0]));
}

/// A cost of `1e20` or more takes no part in the cost perturbation: it
/// neither moves nor sets the base, so the other costs move by less than a
/// tenth of `|c| + 1` and every working cost stays finite (perturbed, `-1e300`
/// would move by about `1e-6 * 1e375`, to minus infinity).
#[test]
fn cost_perturbation_leaves_out_infinite_costs() {
    init();
    let costs = [1.0, -1e300, 1e20, 5e19];
    let mut sol = Solver::try_new(
        &costs,
        &[0.0; 4],
        &[1.0; 4],
        &[(to_sparse(&[1.0; 4]), ComparisonOp::Le, 2.0)],
        &vec![VarDomain::Real; 4],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    sol.perturb_costs();
    let p = &sol.cost_perturbation;
    assert_eq!((p[1], p[2]), (0.0, 0.0));
    // The largest cost that takes part is 5e19: the base is
    // 5e-7 * 5e19^(1/4), about 0.042.
    let base = 5e-7 * 5e19f64.sqrt().sqrt();
    for var in [0, 3] {
        let amount = p[var] / ((costs[var].abs() + 1.0) * base);
        assert!((1.0..2.0).contains(&amount), "var {var}: {}", p[var]);
        assert!(
            p[var] < 0.1 * (costs[var].abs() + 1.0),
            "var {var}: {}",
            p[var]
        );
    }
    assert!(p.iter().all(|x| x.is_finite()));
}

/// A dual phase that starts near optimal (dual feasible, every row violated
/// by less than `1e-3`) is not perturbed, and one far from it is.
#[test]
fn dual_phase_is_perturbed_only_far_from_optimal() {
    init();
    for (rhs, perturbed) in [(1e-4, false), (1.0, true)] {
        let mut sol = Solver::try_new(
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[10.0, 10.0],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, rhs)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        sol.begin_dual_phase(false).unwrap();
        assert_eq!(sol.costs_modified, perturbed, "rhs {rhs}");
        assert_eq!(sol.dual_phase().unwrap(), StopReason::Finished);
        sol.end_dual_phase();
        assert!(!sol.costs_modified);
        sol.recalc_obj_coeffs().unwrap();
        assert_eq!(sol.calc_dual_infeasibility().0, 0);
        assert_eq!(sol.get_value(0), rhs);
    }
}

/// The cycling detection of the dual simplex: a basis change leading to a
/// visited vertex is cycling only when the change before it did too; it is
/// then recorded as bad and made taboo, a bad change seen again is taboo again
/// at once, a rebuild lifts the taboo, and a record goes once its row has
/// moved. The vertex includes where the non-basic vars sit, so a basic set
/// reached again after a bound flip is a new vertex.
#[test]
fn vertex_hash_makes_a_repeated_basis_change_taboo() {
    use super::cycling::{
        CycleGuard,
        VertexChange::{Basic, Upper},
    };
    // Basic vars 10 and 11 in rows 0 and 1; vars 0 and 1 non-basic at their
    // lower bound.
    let start = || {
        let mut guard = CycleGuard::default();
        guard.start([Basic(10), Basic(11)].into_iter());
        guard
    };
    let next = |guard: &CycleGuard, leaving: usize, entering: usize| {
        guard.hash_after([Basic(leaving), Basic(entering)].into_iter())
    };
    let pivot = |guard: &mut CycleGuard, leaving: usize, entering: usize| {
        guard.apply(Basic(leaving));
        guard.apply(Basic(entering));
        guard.record_pivot();
    };

    // Var 0 replaces 10 in row 0, then 10 replaces 0: back at the start.
    let mut guard = start();
    let h = next(&guard, 10, 0);
    assert!(!guard.is_bad_change(0, 10, 0, h));
    pivot(&mut guard, 10, 0);
    let h = next(&guard, 0, 10);
    assert!(
        !guard.is_bad_change(0, 0, 10, h),
        "one revisit is not a cycle"
    );
    pivot(&mut guard, 0, 10);
    // Again var 0 in row 0: a visited vertex right after a visited vertex.
    let h = next(&guard, 10, 0);
    assert!(guard.is_bad_change(0, 10, 0, h));
    assert!(guard.is_taboo(0) && !guard.is_taboo(1) && guard.any_taboo());
    guard.clear_taboo();
    assert!(!guard.any_taboo());
    // The recorded change is taboo again as soon as it is chosen again.
    assert!(guard.is_bad_change(0, 10, 0, h));
    assert!(guard.is_taboo(0));
    // Another change of the same row is not.
    guard.clear_taboo();
    let h = next(&guard, 10, 1);
    assert!(!guard.is_bad_change(0, 10, 1, h));
    // Once row 0 has moved, the record is forgotten: after a pivot in row 1
    // (var 1 replaces 11), the change leads to an unvisited vertex and is not bad.
    guard.forget_moved_rows(|r| r == 0);
    pivot(&mut guard, 11, 1);
    let h = next(&guard, 10, 0);
    assert!(!guard.is_bad_change(0, 10, 0, h));

    // The same pivots with var 1 flipped to its upper bound after the first:
    // the basic sets repeat, the vertices do not.
    let mut guard = start();
    pivot(&mut guard, 10, 0);
    guard.apply(Upper(1));
    let h = next(&guard, 0, 10);
    assert!(!guard.is_bad_change(0, 0, 10, h));
    pivot(&mut guard, 0, 10);
    let h = next(&guard, 10, 0);
    assert!(!guard.is_bad_change(0, 10, 0, h));
}

/// A dual phase whose fresh data leave only taboo basis changes fails loudly
/// instead of answering or running on. From the slack basis of `min x0 + 2 x1`
/// s.t. `x0 + x1 >= 1`, the violated row's one change (`x0` in for the slack)
/// is planted as bad, so it is taboo again after each rebuild. Without the
/// record the phase finishes; a time limit turns running on into a failure.
#[test]
fn dual_phase_fails_loudly_when_only_taboo_changes_are_left() {
    init();
    let new_solver = || {
        let mut sol = Solver::try_new(
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[10.0, 10.0],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 1.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        sol.begin_dual_phase(false).unwrap();
        sol
    };

    let mut sol = new_solver();
    assert_eq!(sol.dual_phase().unwrap(), StopReason::Finished);
    sol.end_dual_phase();
    assert!(matches!(sol.var_states[0], VarState::Basic(_)));

    let mut sol = new_solver();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    let (col, _) = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step()
        .entering;
    let (leaving, entering) = (sol.basic_vars[row], sol.nb_vars[col]);
    assert_eq!(entering, 0);
    let here = sol.cycle_guard.hash_after(std::iter::empty());
    assert!(!sol.cycle_guard.is_bad_change(row, leaving, entering, here));
    sol.cycle_guard.record_pivot();
    assert!(sol.cycle_guard.is_bad_change(row, leaving, entering, here));
    // A phase that ran on instead would stop at the limit, and fail below.
    sol.deadline = Some(Instant::now() + Duration::from_secs(10));
    let result = sol.dual_phase();
    sol.end_dual_phase();
    assert!(
        matches!(&result, Err(Error::InternalError(msg)) if msg.contains("cycles")),
        "{result:?}"
    );
}

/// The loud end of a phase that fresh data leave only taboo changes names a
/// singular basis when one of them leads to one. The model of the test above,
/// with its one change planted as the change back of a basis repair.
#[test]
fn dual_phase_names_a_singular_basis_when_only_its_change_is_left() {
    init();
    let mut sol = Solver::try_new(
        &[1.0, 2.0],
        &[0.0, 0.0],
        &[10.0, 10.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 1.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    sol.begin_dual_phase(false).unwrap();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    let (col, _) = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step()
        .entering;
    let (leaving, entering) = (sol.basic_vars[row], sol.nb_vars[col]);
    sol.cycle_guard.add_singular_change(row, leaving, entering);
    // A phase that ran on instead would stop at the limit, and fail below.
    sol.deadline = Some(Instant::now() + Duration::from_secs(10));
    let result = sol.dual_phase();
    sol.end_dual_phase();
    assert!(
        matches!(&result, Err(Error::InternalError(msg))
            if msg.contains("leads to a singular basis")),
        "{result:?}"
    );
}

/// The compensated sums of the infeasibility proof keep what `f64` rounds
/// away: `1e16 + 1 - 1e16` is one, where `f64` gives zero, and a product is
/// exact: `(1 + 2^-30)^2` keeps its `2^-60`.
#[test]
fn compensated_sums_keep_what_f64_rounds_away() {
    let mut sum = proof::CDouble::default();
    sum.add_product(1e16, 1.0);
    sum.add_product(1.0, 1.0);
    sum.add_product(-1e16, 1.0);
    assert_eq!(sum.value(), 1.0);
    assert_eq!(std::hint::black_box(1e16) + 1.0 - 1e16, 0.0);

    let a = 1.0 + 2f64.powi(-30);
    let mut square = proof::CDouble::default();
    square.add_product(a, a);
    square.add_product(-(1.0 + 2f64.powi(-29)), 1.0);
    assert_eq!(square.value(), 2f64.powi(-60));
    assert_eq!(std::hint::black_box(a) * a - (1.0 + 2f64.powi(-29)), 0.0);
}

/// The proof of infeasibility of the row of `x + y >= rhs` over `x ∈ [0, 1]`
/// and `y ∈ [0, y_max]` (both integer if asked), violated at the slack basis.
fn proof_of_a_single_row(rhs: f64, y_max: f64, integer: bool) -> proof::Proof {
    let domain = if integer {
        VarDomain::Integer
    } else {
        VarDomain::Real
    };
    let mut sol = Solver::try_new(
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, y_max],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, rhs)],
        &[domain.clone(), domain],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.prove_infeasible(row, leaving_new_val > sol.basic_var_vals[row])
}

/// The infeasibility proof needs a gap beyond the room the engine can give
/// the certificate's terms: at the slack basis the certificate is the row
/// `x + y + s = rhs`, `s <= 0`, at most 2 over the box, and the row's slack
/// may pass its bound by its tolerance, the user's `1e-7` less the round-off
/// of evaluating the row, while `x` and `y` may each pass their upper bound
/// by their own `1e-7`. So `rhs = 3` and `2 + 4e-7` are proven, but neither
/// `2 + 3e-8`, a miss the dual simplex accepts, nor `2 + 2.5e-7`, which
/// `x = y = 1 + 1e-7` meets within the row's tolerance, nor any `rhs` with
/// `y` unbounded above. Integer `x` and `y` get no room, as the engine keeps
/// them on their bounds: `2 + 2.5e-7` is proven then.
#[test]
fn infeasibility_proof_needs_a_gap_beyond_the_tolerances() {
    init();
    let proof = proof_of_a_single_row(3.0, 1.0, false);
    assert!(proof.holds, "{proof:?}");
    assert!((proof.gap - 1.0).abs() < 1e-15, "{proof:?}");

    let proof = proof_of_a_single_row(2.0 + 3e-8, 1.0, false);
    assert!(!proof.holds, "{proof:?}");
    assert!(proof.gap > 2.9e-8, "{proof:?}");

    let proof = proof_of_a_single_row(2.0 + 2.5e-7, 1.0, false);
    assert!(!proof.holds, "{proof:?}");
    assert!((proof.allowance - 3e-7).abs() < 1e-14, "{proof:?}");

    let proof = proof_of_a_single_row(2.0 + 2.5e-7, 1.0, true);
    assert!(proof.holds, "{proof:?}");
    assert!((proof.allowance - 1e-7).abs() < 1e-14, "{proof:?}");

    let proof = proof_of_a_single_row(2.0 + 4e-7, 1.0, false);
    assert!(proof.holds, "{proof:?}");

    let proof = proof_of_a_single_row(3.0, f64::INFINITY, false);
    assert!(!proof.holds, "{proof:?}");
    assert_eq!(proof.gap, f64::NEG_INFINITY);
}

/// The optimal basis of `min x + y` s.t. `x + 2y >= 4`, `3x + y >= 3`, both
/// structural vars basic, with the row of its inverse for the first row
/// computed: fifths, which no `f64` holds.
fn fifths_basis() -> Solver {
    let mut sol = Solver::try_new(
        &[1.0, 1.0],
        &[0.0, 0.0],
        &[10.0, 10.0],
        &[
            (to_sparse(&[1.0, 2.0]), ComparisonOp::Ge, 4.0),
            (to_sparse(&[3.0, 1.0]), ComparisonOp::Ge, 3.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    assert!(
        sol.basic_vars.iter().all(|&var| var < sol.num_vars),
        "fixture: both structural vars basic, {:?}",
        sol.basic_vars
    );
    sol.calc_row_coeffs(0);
    sol
}

/// `B^T rho - e_row` over the basic columns, in compensated arithmetic: the
/// coefficients the certificate of `rho` gives the basic vars, less the
/// leaving var's one.
fn basic_residual(sol: &Solver, rho: &[proof::CDouble], row: usize) -> Vec<f64> {
    sol.basic_vars
        .iter()
        .enumerate()
        .map(|(i, &var)| {
            let mut value = proof::CDouble::default();
            for (r, &a) in sol.orig_constraints_csc.outer_view(var).unwrap().iter() {
                value.add_scaled(rho[r], a);
            }
            if i == row {
                value.add_product(-1.0, 1.0);
            }
            value.value()
        })
        .collect()
}

/// Planted multipliers for the proof.
fn multipliers(rho: &[f64]) -> Vec<proof::CDouble> {
    rho.iter().map(|&r| proof::CDouble::from(r)).collect()
}

/// Planted multipliers for the proof, taken as exact: their error is bounded
/// by zero.
fn exact_row(rho: &[f64]) -> proof::RefinedRow {
    proof::RefinedRow {
        rho: multipliers(rho),
        error: Some(vec![0.0; rho.len()]),
        second: vec![0.0; rho.len()],
    }
}

/// The row of the basis inverse the proof starts from is refined: its
/// residual `B^T rho - e_row`, summed in compensated arithmetic, is solved
/// for and taken off, twice. Planted: at [`fifths_basis`], the computed row
/// perturbed by `1e-6`; the refinement brings it back to within round-off of
/// the computed row, and converges: the second correction, the bound it
/// reports on each multiplier's error, is some ten orders below the first.
#[test]
fn the_row_of_the_basis_inverse_is_refined_before_the_proof() {
    init();
    let mut sol = fifths_basis();
    let row = 0;
    let mut exact = vec![0.0; sol.num_constraints()];
    for (r, &value) in sol.inv_basis_row_coeffs.iter() {
        exact[r] = value;
    }
    let perturbed: Vec<f64> = exact
        .iter()
        .enumerate()
        .map(|(r, &value)| value + if r == 1 { 1e-6 } else { 0.0 })
        .collect();
    let refined = sol.refine_basis_inverse_row(row, multipliers(&perturbed));
    for (r, (got, &want)) in refined.rho.iter().zip(&exact).enumerate() {
        let got = got.value();
        assert!(
            (got - want).abs() <= 1e-15 * want.abs().max(1.0),
            "rho[{r}] {got:e} vs {want:e}"
        );
    }
    let error = refined.error.expect("the refinement converges");
    assert!(error.iter().all(|&e| e <= 1e-16), "{error:?}");
}

/// The refinement keeps its corrections to twice the precision of an `f64`:
/// at [`fifths_basis`] the refined row combines the basic columns into the
/// unit vector to within `1e-30` and bounds each multiplier's error to that
/// order, whereas rounded to `f64` it keeps its representation error, some
/// `1e-17`, which a far bound turns into a void proof
/// (`multipliers_no_f64_holds_do_not_void_a_nodes_proof` in `src/tests/magnitudes.rs`).
#[test]
fn the_refined_row_keeps_its_correction_to_twice_the_precision() {
    init();
    let mut sol = fifths_basis();
    let row = 0;
    let refined = sol.refined_basis_inverse_row(row);
    let kept = basic_residual(&sol, &refined.rho, row);
    assert!(kept.iter().all(|res| res.abs() <= 1e-30), "{kept:?}");
    let error = refined.error.as_ref().expect("the refinement converges");
    assert!(error.iter().all(|&e| e <= 1e-30), "{error:?}");
    let rounded: Vec<f64> = refined.rho.iter().map(|rho| rho.value()).collect();
    let rounded = basic_residual(&sol, &multipliers(&rounded), row);
    assert!(rounded.iter().any(|res| res.abs() > 1e-20), "{rounded:?}");
}

/// The refinement bounds each multiplier's error by its second correction
/// only where it converged (that correction at most `2^-53` of the refined
/// row in the max norm): a solve that leaves more cannot measure its own
/// error, and its bound could make a coefficient of the model count as zero.
/// A row solved exactly has a zero bound; a row or correction that is not
/// finite has none (`f64::max` would drop a NaN).
#[test]
fn the_refinement_bounds_the_multipliers_error_only_where_it_converged() {
    let row = |values: &[f64]| multipliers(values);
    assert_eq!(
        proof::refinement_error(&row(&[1.0, -2.0]), &[3e-33, -1e-34]),
        Some(vec![3e-33, 1e-34])
    );
    assert_eq!(
        proof::refinement_error(&row(&[1.0, 0.5]), &[0.0, 0.0]),
        Some(vec![0.0, 0.0])
    );
    assert_eq!(
        proof::refinement_error(&row(&[1e20]), &[1e4]),
        Some(vec![1e4])
    );
    assert_eq!(proof::refinement_error(&row(&[1.0]), &[2e-16]), None);
    assert_eq!(proof::refinement_error(&row(&[1e-20]), &[1e-30]), None);
    assert_eq!(
        proof::refinement_error(&row(&[1.0, 1.0]), &[1e-33, f64::NAN]),
        None
    );
    assert_eq!(
        proof::refinement_error(&row(&[f64::INFINITY]), &[0.0]),
        None
    );
}

/// A non-basic var's certificate coefficient counts however far below its
/// terms it cancelled, at a finite bound and as an unbounded range on an
/// infinite one: it is the model's own, and counted as zero it proves
/// feasible models infeasible (the round-off coefficient models of
/// `src/tests/magnitudes.rs`). Planted: `rho = (1, 1)` leaves `z` the
/// coefficient `-2^-50` beside terms of two and the rest out of reach by
/// `1.5`; with `z >= -2^51` the certificate reaches `5`, beyond `beta = 4.5`,
/// and with `z` unbounded below any value, so nothing is proven. (Another
/// combination of the rows proves this model infeasible.)
#[test]
fn a_non_basic_vars_round_off_coefficient_counts_on_every_bound() {
    init();
    let proof = |z_min: f64| {
        let sol = Solver::try_new(
            &[0.0, 0.0, -1.0, 0.0],
            &[0.0, 0.0, z_min, 0.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[
                (to_sparse(&[1.0, 1.0, 1.0, 0.0]), ComparisonOp::Ge, 3.0),
                (
                    to_sparse(&[0.0, 0.0, -(1.0 + 2f64.powi(-50)), 1.0]),
                    ComparisonOp::Ge,
                    1.5,
                ),
            ],
            &vec![VarDomain::Real; 4],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        let row = 0;
        assert_eq!(
            sol.basic_vars[row], sol.num_vars,
            "fixture: the slack basis"
        );
        sol.proof_of(&exact_row(&[1.0, 1.0]), row, false)
    };
    let unproven = proof(-(2f64.powi(51)));
    assert!(!unproven.holds, "{unproven:?}");
    assert!((unproven.gap + 0.5).abs() < 1e-14, "{unproven:?}");

    let unproven = proof(f64::NEG_INFINITY);
    assert!(!unproven.holds, "{unproven:?}");
    assert_eq!(unproven.gap, f64::NEG_INFINITY);
}

/// A number of the certificate that is not finite voids it, rather than count
/// as zero or as a gap. Planted: in `x + y + 1e305 f >= 3` with `f` fixed at
/// `1e-305` (a fixed column keeps the user's units), Dekker's split of `f`'s
/// coefficient overflows to a NaN; counted as zero, it would prove the model,
/// feasible at `x = y = 1`, infeasible.
#[test]
fn non_finite_certificate_data_void_the_proof() {
    init();
    let sol = Solver::try_new(
        &[0.0; 3],
        &[0.0, 0.0, 1e-305],
        &[1.0, 1.0, 1e-305],
        &[(to_sparse(&[1.0, 1.0, 1e305]), ComparisonOp::Ge, 3.0)],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let row = 0;
    assert_eq!(
        sol.basic_vars[row], sol.num_vars,
        "fixture: the slack basis"
    );
    let proof = sol.proof_of(&exact_row(&[1.0]), row, false);
    assert!(!proof.holds, "{proof:?}");
    assert!(proof.gap.is_nan(), "{proof:?}");
}

/// An error bound that is not finite voids the proof too, rather than count
/// every coefficient within it as zero. Planted: the multiplier of
/// `x + y + j >= 3` (`j` free) with an infinite error, as an overflow of
/// `Σ e_i |a_ij|` would leave it; counted as zero, `j`'s coefficient would
/// prove the model, feasible at `j >= 1`, infeasible.
#[test]
fn an_error_bound_that_is_not_finite_voids_the_proof() {
    init();
    let sol = Solver::try_new(
        &[0.0; 3],
        &[0.0, 0.0, f64::NEG_INFINITY],
        &[1.0, 1.0, f64::INFINITY],
        &[(to_sparse(&[1.0, 1.0, 1.0]), ComparisonOp::Ge, 3.0)],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let row = 0;
    assert_eq!(
        sol.basic_vars[row], sol.num_vars,
        "fixture: the slack basis"
    );
    let unbounded_error = proof::RefinedRow {
        rho: multipliers(&[1.0]),
        error: Some(vec![f64::INFINITY]),
        second: vec![f64::INFINITY],
    };
    let proof = sol.proof_of(&unbounded_error, row, false);
    assert!(!proof.holds, "{proof:?}");
    assert!(proof.gap.is_nan(), "{proof:?}");
    let proof = sol.proof_of(&exact_row(&[1.0]), row, false);
    assert!(!proof.holds, "{proof:?}");
    assert_eq!(proof.gap, f64::NEG_INFINITY, "{proof:?}");
}

/// A certificate multiplier whose row would need its slack's infinite bound
/// is dropped with its row, leaving a certificate of the other rows.
/// Planted: `rho` combines `x + y >= 3` over `x, y ∈ [0, 1]` with `1e-3` of
/// `z <= 5`, whose slack has no upper bound; kept, that multiplier would make
/// the range unbounded and leave an infeasible model unproven.
#[test]
fn a_multiplier_on_a_rows_infinite_side_is_dropped() {
    init();
    let sol = Solver::try_new(
        &[0.0; 3],
        &[0.0; 3],
        &[1.0, 1.0, 10.0],
        &[
            (to_sparse(&[1.0, 1.0, 0.0]), ComparisonOp::Ge, 3.0),
            (to_sparse(&[0.0, 0.0, 1.0]), ComparisonOp::Le, 5.0),
        ],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let row = 0;
    assert_eq!(
        sol.basic_vars[row], sol.num_vars,
        "fixture: the slack basis"
    );
    let proof = sol.proof_of(&exact_row(&[1.0, 1e-3]), row, false);
    assert!(proof.holds, "{proof:?}");
    assert!((proof.gap - 1.0).abs() < 1e-15, "{proof:?}");
}

/// A basic var keeps its certificate coefficient when the certificate leaves
/// a row out, and on an infinite bound that makes the range unbounded: only
/// what the multipliers' error could leave counts as zero. The dual simplex
/// ends this feasible model at the basis `{j, w}` and the second row's slack,
/// `1e-6` above its bound; the exact row of the basis inverse,
/// `(1, 1, 2^-50)`, leaves out the third row (its multiplier needs its
/// slack's infinite side), which gives the free `j` the coefficient `-2^-50`,
/// `4.4e-16` of its terms. Counted as zero, it proves the model infeasible.
#[test]
fn a_basic_vars_coefficient_in_a_row_left_out_voids_the_proof() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, 0.0, 0.0, 0.0],
        &[0.0, f64::NEG_INFINITY, f64::NEG_INFINITY, 0.0],
        &[1.0, f64::INFINITY, f64::INFINITY, 1.0],
        &[
            (to_sparse(&[1.0, 1.0, 1.0, 0.0]), ComparisonOp::Eq, 1.000001),
            (
                to_sparse(&[0.0, -(1.0 + 2f64.powi(-50)), -1.0, 0.0]),
                ComparisonOp::Eq,
                0.0,
            ),
            (to_sparse(&[0.0, 1.0, 0.0, 1.0]), ComparisonOp::Le, 0.0),
        ],
        &vec![VarDomain::Real; 4],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let (j, w, s2) = (1, 2, sol.num_vars + 1);
    sol.load_basis(&Basis(vec![
        VarStatus::AtUpper,
        VarStatus::Basic,
        VarStatus::Basic,
        VarStatus::AtLower,
        VarStatus::AtLower,
        VarStatus::Basic,
        VarStatus::AtLower,
    ]))
    .unwrap();
    let row = sol.basic_vars.iter().position(|&var| var == s2).unwrap();
    assert!(
        sol.basic_var_vals[row] > 5e-7,
        "fixture: the slack of the second row above its bound, {:?}",
        sol.basic_var_vals
    );
    assert!(
        [j, w]
            .iter()
            .all(|&var| matches!(sol.var_states[var], VarState::Basic(_))),
        "fixture: j and w basic"
    );
    let refined = sol.refined_basis_inverse_row(row);
    assert_eq!(refined.error, Some(vec![0.0; 3]), "an exact row");
    let proof = sol.prove_infeasible(row, false);
    assert!(!proof.holds, "{proof:?}");
    assert_eq!(proof.gap, f64::NEG_INFINITY, "{proof:?}");
}

/// A coefficient on an infinite bound counts as zero exactly where the
/// multipliers' error could make it zero: within each multiplier's error
/// bound carried through the var's entries, plus its compensated sum's own
/// error. Planted, on an infeasible model at the basis `{j}` and the first
/// row's slack: the multipliers `(1, -(1 + d))` leave the basic `j` and the
/// non-basic `w`, both free, the coefficient `-d` and the rest out of reach
/// by one. `d = 2^-80` is proven with an error bound of `2^-79`, not with
/// `2^-82` or none; `d = 2^-100`, within the sums' own error, is proven
/// either way.
#[test]
fn a_coefficient_on_an_infinite_bound_counts_as_zero_only_within_its_error() {
    init();
    let mut sol = Solver::try_new(
        &[0.0; 5],
        &[0.0, 0.0, 0.0, f64::NEG_INFINITY, f64::NEG_INFINITY],
        &[1.0, 1.0, 1.0, f64::INFINITY, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0, 0.0, 1.0, 1.0]), ComparisonOp::Ge, 3.0),
            (to_sparse(&[0.0, 0.0, 1.0, 1.0, 1.0]), ComparisonOp::Eq, 0.0),
        ],
        &vec![VarDomain::Real; 5],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let (j, w, s0) = (3, 4, sol.num_vars);
    sol.load_basis(&Basis(vec![
        VarStatus::AtLower,
        VarStatus::AtLower,
        VarStatus::AtLower,
        VarStatus::Basic,
        VarStatus::Free,
        VarStatus::Basic,
        VarStatus::AtLower,
    ]))
    .unwrap();
    let row = sol.basic_vars.iter().position(|&var| var == s0).unwrap();
    assert!(
        (sol.basic_var_vals[row] - 3.0).abs() < 1e-12,
        "fixture: the slack of the first row 3 above its bound, {:?}",
        sol.basic_var_vals
    );
    assert!(
        matches!(sol.var_states[j], VarState::Basic(_))
            && matches!(sol.var_states[w], VarState::NonBasic(_)),
        "fixture: j basic, w non-basic"
    );
    let proof = |d: f64, error: Option<f64>| {
        let mut second = proof::CDouble::default();
        second.add_product(-1.0, 1.0);
        second.add_product(-d, 1.0);
        let refined = proof::RefinedRow {
            rho: vec![proof::CDouble::from(1.0), second],
            error: error.map(|e| vec![0.0, e]),
            second: vec![0.0, error.unwrap_or(f64::INFINITY)],
        };
        sol.proof_of(&refined, row, false)
    };
    let proven = proof(2f64.powi(-80), Some(2f64.powi(-79)));
    assert!(proven.holds, "{proven:?}");
    assert!((proven.gap - 1.0).abs() < 1e-14, "{proven:?}");
    for error in [Some(2f64.powi(-82)), None] {
        let unproven = proof(2f64.powi(-80), error);
        assert!(!unproven.holds, "{error:?}: {unproven:?}");
        assert_eq!(unproven.gap, f64::NEG_INFINITY, "{error:?}: {unproven:?}");
    }
    for error in [Some(0.0), None] {
        let proven = proof(2f64.powi(-100), error);
        assert!(proven.holds, "{error:?}: {proven:?}");
    }
}

/// A row that no var can repair and that does not prove infeasibility is
/// recorded as a bad basis change with no entering var: taboo, and
/// unproven, until a rebuild lifts the taboo; the record goes when a step
/// moves its row.
#[test]
fn a_failed_proof_makes_its_row_taboo() {
    let mut guard = CycleGuard::default();
    guard.start(std::iter::empty());
    guard.add_failed_proof(3, 7);
    assert!(guard.is_taboo(3) && guard.is_unproven(3));
    assert!(!guard.is_taboo(2) && !guard.is_unproven(2));
    assert!(!guard.is_taboo_entering(7));
    guard.clear_taboo();
    assert!(!guard.is_taboo(3) && !guard.is_unproven(3));
    guard.add_failed_proof(3, 7);
    assert!(guard.is_unproven(3));
    guard.forget_moved_rows(|r| r == 3);
    assert!(!guard.any_taboo());
}

/// A fixed var's huge coefficient in a row added through the LP edit path:
/// `min y` s.t. `f + y >= 0`, then `f` fixed at zero and `1e17 f + y >= 1`
/// added. The added row leaves the fixed `f` out of its equilibration, so the
/// dual ratio test sees `[1e17, 1]`. A deadline in the past stops the edit's
/// solve before its first iteration, for the tests to resume.
fn c2_06_through_an_edit() -> Solver {
    let mut sol = Solver::try_new(
        &[0.0, 1.0],
        &[0.0, 0.0],
        &[10.0, f64::INFINITY],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 0.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    assert_eq!(sol.fix_var(0, 0.0).unwrap(), StopReason::Finished);
    sol.deadline = Some(Instant::now());
    assert_eq!(
        sol.add_constraint(to_sparse(&[1e17, 1.0]), ComparisonOp::Ge, 1.0)
            .unwrap(),
        StopReason::Limit
    );
    sol.deadline = None;
    sol
}

/// The growth test measures a pivot against the entries of the vars that can
/// move: `f`, fixed by an edit, never enters, so its `1e17` leaves `y`'s exact
/// one a regular pivot, and the phase ends at `y = 1`. That pivot refused
/// (planted as a bad basis change), no other can repair the row, and the
/// phase fails loudly at once, every basis change left returning to visited
/// vertices, instead of retrying until its stall bound; a time limit turns
/// running on into a failure.
#[test]
fn a_fixed_vars_huge_entry_leaves_a_genuine_pivot_regular() {
    init();
    let mut sol = c2_06_through_an_edit();
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!((sol.get_value(0), sol.get_value(1)), (0.0, 1.0));

    let mut sol = c2_06_through_an_edit();
    sol.begin_dual_phase(false).unwrap();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    assert_eq!(row, 1, "fixture: the added row is the violated one");
    sol.calc_row_coeffs(row);
    let (y_col, coeff) = sol
        .choose_entering_col_dual(row, leaving_new_val)
        .unwrap()
        .step()
        .entering;
    assert_eq!((sol.nb_vars[y_col], coeff), (1, 1.0));
    let here = sol.cycle_guard.hash_after(std::iter::empty());
    let (leaving, entering) = (sol.basic_vars[row], 1);
    assert!(!sol.cycle_guard.is_bad_change(row, leaving, entering, here));
    sol.cycle_guard.record_pivot();
    assert!(sol.cycle_guard.is_bad_change(row, leaving, entering, here));
    sol.deadline = Some(Instant::now() + Duration::from_secs(10));
    let iterations = sol.lp_iterations;
    let result = sol.dual_phase();
    sol.end_dual_phase();
    assert!(
        matches!(
            &result,
            Err(Error::InternalError(message))
                if message.contains("every basis change left returns to visited vertices")
        ),
        "{result:?}"
    );
    assert!(sol.lp_iterations - iterations < 20, "{}", sol.lp_iterations);
}

/// `x + y >= 2 + 1.8e-7` over `x, y ∈ [0, 1]` (integer if asked), at the
/// slack basis with `x = y = 1`: the row's slack is `1.8e-7` past its bound,
/// beyond its tolerance of about `1e-7`, with both vars at their upper
/// bounds.
fn row_short_of_its_bounds_by_a_tolerance(integer: bool) -> Solver {
    let domain = if integer {
        VarDomain::Integer
    } else {
        VarDomain::Real
    };
    let sol = Solver::try_new(
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 2.0 + 1.8e-7)],
        &[domain.clone(), domain],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(
        (sol.col_scales.as_slice(), sol.row_scales.as_slice()),
        ([1.0, 1.0].as_slice(), [1.0].as_slice()),
        "fixture: no scaling"
    );
    assert_eq!(sol.nb_var_vals, [1.0, 1.0], "fixture: x = y = 1");
    let slack = sol.num_vars;
    assert_eq!(sol.basic_vars, [slack], "fixture: the slack basis");
    assert_eq!(
        sol.var_tols[slack],
        BoundTols::uniform(slack_tol(1e-7, 4.0 + 1.8e-7, 2)),
        "fixture: the row's whole contract less its round-off"
    );
    sol
}

/// The certificate repair widens, largest reach first, the bound each var
/// that can help sits at by its tolerance and moves the var there, to the
/// farthest value within its tolerance of the model's bound: `x`'s upper
/// bound in force becomes `1 + 1e-7`, which lands the row's slack within its
/// own tolerance. The model's bound is kept for the boundary, and the next
/// solve restores it. An integer var is never widened, so with `x, y` integer
/// the repair does nothing.
#[test]
fn the_repair_moves_vars_past_their_bounds_within_their_tolerances() {
    init();
    let mut sol = row_short_of_its_bounds_by_a_tolerance(false);
    let x_tol = sol.var_tols[0].hi;
    assert_eq!(x_tol, 1e-7, "fixture: x's tolerance is its whole contract");
    assert!(sol.certificate_repair(0));
    let x = sol.nb_var_vals[0];
    assert!(
        x > 1.0 && x - 1.0 <= x_tol && x.next_up() - 1.0 > x_tol,
        "x = {x:e}"
    );
    assert_eq!(sol.orig_var_maxs[0], x, "x's bound in force");
    assert_eq!(sol.var_tols[0].hi, 0.0, "x's bound holds its tolerance");
    assert_eq!(sol.get_var_bounds(0), (0.0, 1.0), "the model's bounds");
    assert!(sol.nb_var_states[0].at_max && !sol.nb_var_states[0].at_min);
    assert_eq!(sol.nb_var_vals[1], 1.0);
    assert_eq!(sol.orig_var_maxs[1], 1.0, "y is not widened");
    assert!(sol.basic_var_within_bounds(0), "{:?}", sol.basic_var_vals);
    sol.restore_model_bounds();
    assert_eq!(
        (sol.orig_var_maxs[0], sol.var_tols[0].hi, sol.nb_var_vals[0]),
        (1.0, x_tol, 1.0),
        "restored"
    );

    let mut sol = row_short_of_its_bounds_by_a_tolerance(true);
    assert!(!sol.certificate_repair(0));
    assert_eq!(sol.nb_var_vals, [1.0, 1.0]);
    assert!(!sol.basic_var_within_bounds(0));
}

/// The certificate repair keeps a fixed structural var at its value, at which
/// presolve checks the rows it substituted the var out of: its tolerance is
/// no room. `x + y >= 2 + 2.5e-7` with `x ∈ [0, 1]` at 1 and `y` fixed at 1,
/// at the slack basis: `x`'s bound is widened by its tolerance, `1e-7`, which
/// leaves the row's slack `1.5e-7` past its bound, beyond its tolerance of
/// about `1e-7`; `y` stays at 1, though its own `1e-7` would land the row.
/// The proof, which grants `y` no room either, then holds.
#[test]
fn the_repair_keeps_a_fixed_var_at_its_value() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, 0.0],
        &[0.0, 1.0],
        &[1.0, 1.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 2.0 + 2.5e-7)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(
        (sol.col_scales.as_slice(), sol.row_scales.as_slice()),
        ([1.0, 1.0].as_slice(), [1.0].as_slice()),
        "fixture: no scaling"
    );
    assert_eq!(sol.nb_var_vals, [1.0, 1.0], "fixture: x = 1, y = 1");
    assert_eq!(sol.basic_vars, [sol.num_vars], "fixture: the slack basis");
    assert_eq!(sol.var_tols[0].hi, 1e-7, "fixture: x's tolerance");
    assert_eq!(sol.var_tols[1].lo, 1e-7, "fixture: y's tolerance");
    assert_eq!(
        sol.var_tols[sol.num_vars],
        BoundTols::uniform(slack_tol(1e-7, 2.0 + 2.5e-7 + 2.0, 2)),
        "fixture: the row's tolerance"
    );
    assert!(sol.certificate_repair(0));
    assert_eq!(sol.nb_var_vals[1], 1.0);
    assert!(!sol.basic_var_within_bounds(0), "{:?}", sol.basic_var_vals);
    let up = sol.leaving_target(0) > sol.basic_var_vals[0];
    assert!(sol.prove_infeasible(0, up).holds);
}

/// The certificate repair lands a leaving slack within its tolerance less the
/// proof's cut. `2^-30 x1 + 2^-31 x2 >= 1.0000000009313221e-7` over
/// `x1, x2 ∈ [-1, 0]`, both at 0 at the slack basis: the row's slack lies
/// past the edge of its room by `x1`'s reach, `2^-30 1e-7`, and the proof
/// fails, both vars having room. Moving `x1` alone lands the slack exactly on
/// the edge, within its tolerance but not within the cut; the repair moves
/// `x2` too. The test pins the rule: the slack is evaluated exactly here, so
/// the edge is accepted too, and the solve ends optimal with either target.
#[test]
fn the_repair_lands_a_slack_within_its_tolerance_less_the_proofs_cut() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, -1.0],
        &[-1.0, -1.0],
        &[0.0, 0.0],
        &[(
            to_sparse(&[2f64.powi(-30), 2f64.powi(-31)]),
            ComparisonOp::Ge,
            1.0000000009313221e-7,
        )],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let slack = sol.num_vars;
    assert_eq!(sol.basic_vars, [slack], "fixture: the slack basis");
    assert_eq!(sol.nb_var_vals, [0.0, 0.0], "fixture: x1 = x2 = 0");
    let tol = sol.var_tols[slack].hi;
    let proof = sol.prove_infeasible(0, false);
    assert!(!proof.holds && proof.cut > 0.0, "{proof:?}");
    // `x1`'s move to the farthest value within its room, in the engine's
    // units, carries the slack by its coefficient times that.
    let reach = sol
        .orig_constraints
        .outer_view(0)
        .unwrap()
        .get(0)
        .copied()
        .unwrap()
        * band_edge(0.0, sol.var_tols[0].hi, true);
    let alone = sol.basic_var_vals[0] - reach;
    assert!(
        alone == tol && proof.cut < tol,
        "fixture: x1 alone lands the slack at {alone:e}, on its tolerance {tol:e}"
    );
    assert!(sol.certificate_repair(0));
    for var in [0, 1] {
        assert!(
            sol.nb_var_vals[var] > 0.0 && sol.nb_var_vals[var] == sol.orig_var_maxs[var],
            "x{} on its widened bound: {:?}",
            var + 1,
            sol.nb_var_vals
        );
    }
    assert!(
        sol.basic_var_vals[0] <= tol - proof.cut,
        "the slack at {:e}",
        sol.basic_var_vals[0]
    );
}

/// The bound perturbation of a primal phase: every finite bound moves outward
/// by `r 5e-7 max(1, |b|)` in user units (a slack's by the magnitude of its
/// row's right-hand side), mapped into the engine's scaled units like its
/// tolerances; a fixed var is never perturbed, basic or not; a non-basic var
/// at a bound moves with it and the basic values follow.
#[test]
fn bound_perturbation_follows_the_bound_types() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, -1.0, 0.0, 0.0],
        &[0.0, -3e6, 2.0, f64::NEG_INFINITY],
        &[10.0, -2e6, 2.0, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0, 1.0, 1.0]), ComparisonOp::Le, 100.0),
            (to_sparse(&[0.0, 0.0, 0.0, 1.0]), ComparisonOp::Eq, 0.0),
        ],
        &vec![VarDomain::Real; 4],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.col_scales, [1.0; 4], "fixture: no column scaling");
    assert_eq!(sol.row_scales, [1.0; 2], "fixture: no row scaling");
    // x0 and x1 start at their upper bounds (negative costs), x2 is fixed,
    // x3 free; both slacks are basic, the second one fixed.
    assert_eq!(sol.nb_var_vals, [10.0, -2e6, 2.0, 0.0]);
    sol.perturb_bounds();
    assert!(sol.bounds_modified);
    let amount =
        |var: usize, mag: f64| random_fraction(var, Stream::Perturbation) * 5e-7 * mag.max(1.0);
    let working = |sol: &Solver, var: usize| sol.working_bounds(var);
    assert_eq!(working(&sol, 0), (-amount(0, 0.0), 10.0 + amount(0, 10.0)));
    assert_eq!(
        working(&sol, 1),
        (-3e6 - amount(1, 3e6), -2e6 + amount(1, 2e6))
    );
    assert_eq!(working(&sol, 2), (2.0, 2.0));
    assert_eq!(working(&sol, 3), (f64::NEG_INFINITY, f64::INFINITY));
    // The first slack's lower bound, by its row's right-hand side of 100;
    // the second slack is fixed, and basic.
    assert_eq!(working(&sol, 4), (-amount(4, 100.0), f64::INFINITY));
    assert_eq!(working(&sol, 5), (0.0, 0.0));
    for var in [0, 1, 4] {
        assert!(amount(var, 1.0) > 0.0, "var {var}: no perturbation drawn");
    }
    // The non-basic vars at a bound sit at the perturbed one, and the basic
    // slack of the first row follows them.
    assert_eq!(sol.nb_var_vals[0], working(&sol, 0).1);
    assert_eq!(sol.nb_var_vals[1], working(&sol, 1).1);
    assert_eq!(sol.nb_var_vals[2], 2.0);
    let activity: f64 = sol.nb_var_vals.iter().sum();
    assert!((sol.basic_var_vals[0] - (100.0 - activity)).abs() < 1e-9);
    assert_eq!(
        (sol.basic_var_mins[0], sol.basic_var_maxs[0]),
        working(&sol, 4)
    );
}

/// A primal phase is perturbed only when it is not a cleanup and starts far
/// from optimal (a thousand or more dual infeasibilities, or one of `1e-3` or
/// more), and its cleanup takes every non-basic var at a moved bound back to
/// the true one: `min c (x0 + x1)`, primal feasible at zero with both reduced
/// costs `c`, ends at the vertex `(3, 1)` exactly, however it was perturbed.
#[test]
fn primal_phase_is_perturbed_only_far_from_optimal_and_cleans_up_exactly() {
    init();
    for (cost, cleanup, perturbed) in [
        (-1.0, false, true),
        (-1.0, true, false),
        (-1e-4, false, false),
    ] {
        let mut sol = Solver::try_new(
            &[cost, cost],
            &[0.0, 0.0],
            &[f64::INFINITY, f64::INFINITY],
            &[
                (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0),
                (to_sparse(&[1.0, -1.0]), ComparisonOp::Le, 2.0),
            ],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        assert!(sol.is_primal_feasible && sol.calc_dual_infeasibility().0 > 0);
        let label = format!("cost {cost}, cleanup {cleanup}");
        sol.begin_primal_phase(cleanup);
        assert_eq!(sol.bounds_modified, perturbed, "{label}");
        assert_eq!(sol.primal_phase(1.0).unwrap().0, StopReason::Finished);
        sol.end_primal_phase().unwrap();
        assert!(!sol.bounds_modified, "{label}");
        assert!(sol.is_primal_feasible, "{label}");
        assert_eq!((sol.get_value(0), sol.get_value(1)), (3.0, 1.0), "{label}");
        for (c, &var) in sol.nb_vars.iter().enumerate() {
            assert_eq!(sol.nb_var_vals[c], 0.0, "{label}: slack {var}");
        }
        for r in 0..2 {
            let var = sol.basic_vars[r];
            assert_eq!(
                (sol.basic_var_mins[r], sol.basic_var_maxs[r]),
                (sol.orig_var_mins[var], sol.orig_var_maxs[var]),
                "{label}"
            );
        }
    }
}

/// A leaving var past the bound it leaves at, within its tolerance, gives a
/// zero step, not a step backwards, and leaves where it sits: the slack of
/// `x0 + x1 <= -3.5e-8` starts at `-3.5e-8` (tolerance about `1e-7`), and a
/// step back to its bound would move `x0` to `-3.5e-8`, out of its bounds.
/// The slack keeps its working bounds and, within its tolerance, stays there
/// through the cleanup.
#[test]
fn a_step_that_would_run_backwards_is_zero() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, 0.0],
        &[0.0, 0.0],
        &[f64::INFINITY, 10.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, -3.5e-8)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.basic_var_vals, [-3.5e-8]);
    assert_eq!(sol.var_tols[2].lo, slack_tol(1e-7, 3.5e-8, 2));
    // `try_new`'s exact test finds the start primal infeasible and sets the
    // phase-1 objective; the primal phase works on the real one, as in `run_phases`.
    sol.artificial_obj = None;
    sol.recalc_obj_coeffs().unwrap();
    sol.begin_primal_phase(true);
    assert!(!sol.bounds_modified, "within its tolerance: no shift");
    let PivotChoice::Pivot(info) = primal_pivot_choice(&mut sol, 1.0) else {
        panic!("x0 enters");
    };
    let elem = info.elem.as_ref().unwrap();
    assert_eq!(sol.nb_vars[info.col], 0);
    assert_eq!(info.entering_diff, 0.0);
    assert_eq!(elem.leaving_new_val, -3.5e-8);
    assert_eq!(sol.primal_phase(1.0).unwrap(), (StopReason::Finished, true));
    assert!(!sol.bounds_modified);
    assert_eq!(sol.get_value(0), 0.0);
    let slack = nb_col(&sol, 2);
    assert_eq!(sol.nb_var_vals[slack], -3.5e-8);
    assert!(sol.nb_var_states[slack].at_min);
    sol.end_primal_phase().unwrap();
    assert_eq!(sol.nb_var_vals[slack], -3.5e-8);
    assert_eq!(sol.get_value(0), 0.0);
}

/// The Harris pass's limit on the step is never below zero: after the first
/// round of `run_phases` the primal ratio test allows a basic var only a
/// hundredth of its tolerance past its bound (`LOOP_HARRIS_REDUCTION`), and a
/// var already past it by more admits no step toward it. Here the second
/// slack sits `2.5e-8` past its bound (tolerance about `1e-7`) and scaling
/// puts `x0`'s entries at `1` and `1.34e-8`: the negative limit
/// `(-2.5e-8 + 1e-9) / 1.34e-8` would pivot on `1.34e-8`, below the pivot
/// floor, instead of on the first row's `1`, at its bound.
#[test]
fn a_var_past_its_bound_beyond_the_allowance_limits_the_step_to_zero() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, 0.0],
        &[0.0, 0.0],
        &[f64::INFINITY, 10.0],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 0.0),
            (to_sparse(&[1e-16, 1.0]), ComparisonOp::Le, -2.5e-8),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.basic_var_vals, [0.0, -2.5e-8]);
    assert_eq!(sol.var_tols[3].lo, slack_tol(1e-7, 2.5e-8, 2));
    // `try_new`'s exact test finds the start primal infeasible and sets the
    // phase-1 objective; the primal phase works on the real one, as in `run_phases`.
    sol.artificial_obj = None;
    sol.recalc_obj_coeffs().unwrap();
    sol.begin_primal_phase(true);
    assert!(!sol.bounds_modified, "within its tolerance: no shift");
    for harris_scale in [1.0, LOOP_HARRIS_REDUCTION] {
        let PivotChoice::Pivot(info) = primal_pivot_choice(&mut sol, harris_scale) else {
            panic!("x0 enters");
        };
        assert_eq!(sol.nb_vars[info.col], 0);
        assert_eq!(
            sol.col_coeffs
                .iter()
                .map(|(r, &a)| (r, a))
                .collect::<Vec<_>>(),
            [(0, 1.0), (1, 1e-16 * 2f64.powi(27))],
            "fixture: the scaled column of x0"
        );
        let elem = info.elem.as_ref().unwrap();
        assert_eq!(
            (elem.row, elem.coeff),
            (0, 1.0),
            "harris_scale {harris_scale}: the row at its bound, with the largest entry, leaves"
        );
        assert_eq!(info.entering_diff, 0.0);
    }
}

/// The working bounds a leaving var holds (`hold_leaving_value`): past its
/// true bound within its tolerance, it keeps them; past a bound the phase
/// moved, that bound moves to where it sits; a fixed var stays fixed at its
/// value within its tolerance of it, and is fixed where it sits beyond. The
/// cleanup takes a var at a moved bound back to its true one, and a slack
/// off its bound beyond its current tolerance onto it.
#[test]
fn a_leaving_var_holds_its_working_bounds_until_the_cleanup() {
    init();
    let mut sol = Solver::try_new(
        &[0.0, 0.0],
        &[0.0, 2.0],
        &[10.0, 2.0],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 100.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let tol = sol.var_tols[1].lo;
    assert!(tol > 0.0);
    sol.hold_leaving_value(0, -1e-9);
    assert!(!sol.bounds_modified, "past its true bound: no move");
    sol.hold_leaving_value(1, 2.0 + tol / 2.0);
    assert!(!sol.bounds_modified, "a fixed var within its tolerance");

    sol.working_mins = sol.orig_var_mins.clone();
    sol.working_maxs = sol.orig_var_maxs.clone();
    sol.bounds_modified = true;
    sol.working_mins[0] = -1e-6;
    sol.hold_leaving_value(0, -1.01e-6);
    assert_eq!(sol.working_bounds(0), (-1.01e-6, 10.0));
    sol.hold_leaving_value(1, 2.0 + 3.0 * tol);
    assert_eq!(sol.working_bounds(1), (2.0 + 3.0 * tol, 2.0 + 3.0 * tol));
    sol.hold_leaving_value(1, 2.0 + tol / 2.0);
    assert_eq!(sol.working_bounds(1), (2.0, 2.0));

    // The cleanup: x0, non-basic at its moved lower bound, goes back to 0,
    // and the slack follows.
    let x0 = nb_col(&sol, 0);
    sol.nb_var_vals[x0] = -1.01e-6;
    sol.nb_var_states[x0] = sol.nonbasic_state(0, -1.01e-6);
    sol.end_primal_phase().unwrap();
    assert!(!sol.bounds_modified);
    assert_eq!(sol.nb_var_vals[x0], 0.0);
    assert_eq!(sol.basic_var_vals[0], 98.0);
}

/// A step also moves the basic vars whose entries the ratio test ignores, and
/// one that ends beyond its tolerance has its bound shifted, so the phase
/// goes on over bounds its point satisfies; the cleanup restores the bound
/// and reports the violation. Here the step toward `x3 = -1`, `x4 = 1000`
/// passes the equality `1e4 x2 - 1e-7 x3 = 0` through an entry at or below
/// the pivot tolerance and takes its slack `3e-12` past its bound, beyond its
/// tolerance. Planted: the vertex at the origin with `x3`, `x4`
/// and the equality's slack basic, from which the primal phase takes that
/// step, the slack halfway to its tolerance, as round-off can leave it, since
/// the step alone takes it to the edge (the model of
/// `an_equality_a_long_step_passes_below_its_floor_settles`).
#[test]
fn a_step_past_a_row_the_ratio_test_ignores_shifts_its_bound() {
    init();
    let mut sol = Solver::try_new(
        &[1.0, 0.0, 1.0, -1.0, -1.0],
        &[0.0, 0.0, 0.0, -1.0, 0.0],
        &[0.0, 0.0, 1.0, 0.0, 256167.0],
        &[
            (
                to_sparse(&[0.0, 0.0, 0.0, -1.0, -0.001]),
                ComparisonOp::Eq,
                -0.0,
            ),
            (
                to_sparse(&[0.0, 0.0, -1e-5, 0.0, -10000.0]),
                ComparisonOp::Le,
                0.0,
            ),
            (
                to_sparse(&[0.0, 0.0, 10000.0, -1e-7, 0.0]),
                ComparisonOp::Eq,
                0.0,
            ),
        ],
        &vec![VarDomain::Real; 5],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    use VarStatus::{AtLower, Basic};
    let vertex = [
        AtLower, AtLower, AtLower, Basic, Basic, AtLower, AtLower, Basic,
    ];
    sol.load_basis(&Basis(vertex.to_vec())).unwrap();
    assert!(sol.is_primal_feasible);
    assert!(sol.calc_dual_infeasibility().0 > 0);
    let slack = sol.num_vars + 2;
    let VarState::Basic(row) = sol.var_states[slack] else {
        panic!("the slack of the equality is basic");
    };
    let tol = sol.var_tols[slack].lo;
    sol.basic_var_vals[row] = -tol / 2.0;

    sol.begin_primal_phase(true);
    assert!(!sol.bounds_modified);
    assert_eq!(sol.primal_phase(1.0).unwrap(), (StopReason::Finished, true));
    assert!(sol.bounds_modified);
    let val = sol.basic_var_vals[row];
    assert!(
        val < -tol,
        "the row is violated beyond its tolerance: {val:e}"
    );
    assert!(
        sol.basic_var_mins[row] < val - tol,
        "and its bound shifted past it"
    );
    assert!(sol.calc_primal_infeasibility().0 == 0);

    sol.end_primal_phase().unwrap();
    assert!(!sol.bounds_modified);
    assert_eq!(sol.basic_var_mins[row], 0.0);
    assert!(!sol.is_primal_feasible);
    assert!(sol.calc_primal_infeasibility().0 == 1);
}

/// The cycling detection of the primal simplex, whose taboo is on the
/// entering var: a phase whose fresh data leave only taboo basis changes
/// fails loudly instead of answering or running on. From the slack basis of
/// `min -x0 - 2 x1` s.t. `x0 + x1 <= 4`, `x1` in for the slack is planted as
/// bad and `x0` made dual feasible, so the only change is taboo again after
/// each rebuild. Without the record the phase finishes; a time limit turns
/// running on into a failure.
#[test]
fn primal_phase_fails_loudly_when_only_taboo_changes_are_left() {
    init();
    let new_solver = || {
        let mut sol = Solver::try_new(
            &[-1.0, -2.0],
            &[0.0, 0.0],
            &[f64::INFINITY, f64::INFINITY],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        sol.begin_primal_phase(true);
        sol
    };

    let mut sol = new_solver();
    assert_eq!(sol.primal_phase(1.0).unwrap(), (StopReason::Finished, true));
    sol.end_primal_phase().unwrap();
    assert!(matches!(sol.var_states[1], VarState::Basic(_)));

    let mut sol = new_solver();
    let PivotChoice::Pivot(info) = primal_pivot_choice(&mut sol, 1.0) else {
        panic!("x1 enters");
    };
    let elem = info.elem.as_ref().unwrap();
    let (row, leaving, entering) = (elem.row, sol.basic_vars[elem.row], sol.nb_vars[info.col]);
    assert_eq!(entering, 1);
    let here = sol.cycle_guard.hash_after(std::iter::empty());
    assert!(!sol.cycle_guard.is_bad_change(row, leaving, entering, here));
    sol.cycle_guard.record_pivot();
    assert!(sol.cycle_guard.is_bad_change(row, leaving, entering, here));
    assert!(sol.cycle_guard.is_taboo_entering(1) && !sol.cycle_guard.is_taboo_entering(0));
    // `x0` would enter once `x1` is taboo; make it dual feasible, so that
    // only the taboo change is left.
    let x0 = nb_col(&sol, 0);
    sol.nb_var_obj_coeffs[x0] = 0.0;
    sol.orig_obj_coeffs[0] = 0.0;
    // A phase that ran on instead would stop at the limit, and fail below.
    sol.deadline = Some(Instant::now() + Duration::from_secs(10));
    let result = sol.primal_phase(1.0);
    sol.end_primal_phase().unwrap();
    assert!(
        matches!(&result, Err(Error::InternalError(msg)) if msg.contains("cycles")),
        "{result:?}"
    );
}

/// The entering var's reduced cost, updated since its last recompute, is
/// checked against the one its column gives before the ratio test (each
/// planted value below is marked updated). In `min x0` s.t. `x0 >= 1`, `x1` (in
/// `[0, ∞)`) is in no row, so its column is empty and its reduced cost is its
/// cost. With cost zero, planted at `-1e-8`, as drift past its tolerance
/// leaves it, it prices `x1`, but the column gives zero: `x1` does not enter,
/// and no ray is found. With cost `2`, planted at `-1`, the column's `2` lies
/// on the other side of zero, and the phase rebuilds: the slack's reduced
/// cost, planted wrong too, is recomputed. With cost `-1` the ray stands.
#[test]
fn an_entering_reduced_cost_is_checked_against_its_column() {
    init();
    let new_solver = |x1_cost: f64| {
        Solver::try_new(
            &[1.0, x1_cost],
            &[0.0, 0.0],
            &[f64::INFINITY, f64::INFINITY],
            &[(to_sparse(&[1.0, 0.0]), ComparisonOp::Ge, 1.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap()
    };
    let run_phase = |sol: &mut Solver| {
        sol.begin_primal_phase(true);
        let result = sol.primal_phase(1.0);
        sol.end_primal_phase().unwrap();
        result.unwrap()
    };

    let mut sol = new_solver(0.0);
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    let x1 = nb_col(&sol, 1);
    sol.nb_var_obj_coeffs[x1] = -1e-8;
    sol.reduced_costs_dirty = true;
    assert!(matches!(
        primal_pivot_choice(&mut sol, 1.0),
        PivotChoice::Drifted { sign_error: false }
    ));
    assert_eq!(sol.nb_var_obj_coeffs[x1], 0.0);
    sol.nb_var_obj_coeffs[x1] = -1e-8;
    assert_eq!(run_phase(&mut sol), (StopReason::Finished, false));
    assert_eq!(sol.get_value(1), 0.0);

    let mut sol = new_solver(2.0);
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    let (x1, slack) = (nb_col(&sol, 1), nb_col(&sol, 2));
    let slack_reduced_cost = sol.nb_var_obj_coeffs[slack];
    assert!(slack_reduced_cost != 0.0);
    sol.nb_var_obj_coeffs[x1] = -1.0;
    sol.reduced_costs_dirty = true;
    assert!(matches!(
        primal_pivot_choice(&mut sol, 1.0),
        PivotChoice::Drifted { sign_error: true }
    ));
    assert_eq!(sol.nb_var_obj_coeffs[x1], 2.0);
    sol.nb_var_obj_coeffs[x1] = -1.0;
    sol.nb_var_obj_coeffs[slack] = 5.0 * slack_reduced_cost;
    assert_eq!(run_phase(&mut sol), (StopReason::Finished, false));
    assert_eq!(sol.nb_var_obj_coeffs[slack], slack_reduced_cost);

    let mut sol = new_solver(-1.0);
    assert!(matches!(sol.initial_solve(), Err(Error::Unbounded)));
}

/// A reduced cost fresh from refined multipliers is not checked against its
/// column, which a factorization computes no better. In the model of
/// [`an_entering_reduced_cost_is_checked_against_its_column`], `x1`'s
/// reduced cost planted at `-1e-8` right after a solve, unmarked, stands for
/// a fresh one: it prices `x1` into a ray, which the phase then declares
/// only on fresh data.
#[test]
fn a_fresh_reduced_cost_is_not_checked_against_its_column() {
    init();
    let mut sol = Solver::try_new(
        &[1.0, 0.0],
        &[0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY],
        &[(to_sparse(&[1.0, 0.0]), ComparisonOp::Ge, 1.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    assert!(!sol.reduced_costs_dirty);
    let x1 = nb_col(&sol, 1);
    sol.nb_var_obj_coeffs[x1] = -1e-8;
    assert!(matches!(
        primal_pivot_choice(&mut sol, 1.0),
        PivotChoice::Unbounded
    ));
}

/// An `Unbounded` verdict is taken on a fresh factorization: a ray found on
/// data updated since the last rebuild is priced again after one. In
/// `min -2 x0 - x1` s.t. `x0 - x1 <= 1`, `x0` enters first; then `x1`'s
/// column blocks nothing, a ray found after a pivot.
#[test]
fn an_unbounded_verdict_is_taken_on_a_fresh_factorization() {
    init();
    let mut sol = Solver::try_new(
        &[-2.0, -1.0],
        &[0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY],
        &[(to_sparse(&[1.0, -1.0]), ComparisonOp::Le, 1.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert!(matches!(sol.initial_solve(), Err(Error::Unbounded)));
    assert!(sol.factorization_is_fresh());
}

/// A primal phase starts only on verified values, so that its `Unbounded`
/// rests on a point the model admits. `x0 >= 2` and `x0 <= 1` admit none,
/// while `x1`, free with cost `-1`, is in no row: a ray. A start taken for
/// primal feasible, as a dual phase ending on drifted values would leave it,
/// is checked against the rows first, and the dual simplex proves the model
/// infeasible.
#[test]
fn a_primal_phase_starts_only_on_verified_values() {
    init();
    let mut sol = Solver::try_new(
        &[0.0, -1.0],
        &[0.0, f64::NEG_INFINITY],
        &[f64::INFINITY, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 0.0]), ComparisonOp::Ge, 2.0),
            (to_sparse(&[1.0, 0.0]), ComparisonOp::Le, 1.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert!(!sol.is_primal_feasible && sol.artificial_obj.is_some());
    sol.is_primal_feasible = true;
    assert!(matches!(sol.run_phases(), Err(Error::Infeasible)));
}

/// Reduced-cost tolerances hold in the user's units as in the engine's: the
/// tolerance of a var whose factor between the two is below one shrinks
/// with it. In `max 0.1 x` s.t. `1e9 x <= 1e9` with `x` free, `x`'s column is
/// scaled by `2^-30`, which puts its cost at `9.3e-11` and its tolerance at
/// `1e-10 * 2^-30`, so `x` does not start dual feasible and the primal
/// phase's edge weights are kept. A slack's reduced cost is in units of its
/// row's activity divided by the row's largest coefficient as written on a
/// var its bounds do not fix: the slack of `1e9 x + 1e12 z <= 1e9 + 1e12`,
/// with `z` fixed at one and a row factor `r`, has the tolerance
/// `1e-10 / (1e9 r)`; `1e-6 y <= 1`, whose `y` the row `y <= 1` gives a
/// column entry of one, is scaled up by its factor, but `1e-6 r` is below
/// one, so the engine's `1e-10` holds its slack.
#[test]
fn reduced_cost_tolerances_hold_in_user_units() {
    init();
    let sol = Solver::try_new(
        &[-0.1, 0.0, 0.0],
        &[f64::NEG_INFINITY, 0.0, 1.0],
        &[f64::INFINITY, 1.0, 1.0],
        &[
            (to_sparse(&[1e9, 0.0, 1e12]), ComparisonOp::Le, 1e9 + 1e12),
            (to_sparse(&[0.0, 1e-6, 0.0]), ComparisonOp::Le, 1.0),
            (to_sparse(&[0.0, 1.0, 0.0]), ComparisonOp::Le, 1.0),
        ],
        &[VarDomain::Real, VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let s = sol.col_scales[0];
    assert!(s < 1e-8, "x's column is scaled down: {s:e}");
    assert_eq!(sol.dual_tols[0], EPS * s);
    assert!(sol.enable_primal_steepest_edge);
    let r = sol.row_scales[0];
    assert_eq!(sol.dual_tols[sol.num_vars], EPS * (1.0 / (r * 1e9)));
    let r = sol.row_scales[1];
    assert!(
        r > 1.0 && 1e-6 * r < 1.0,
        "the second row is scaled up: {r:e}"
    );
    assert_eq!(sol.dual_tols[sol.num_vars + 1], EPS);
}

/// A loaded basis recomputes the primal steepest-edge weights: they belong
/// to the non-basic positions, which the load renumbers. Reloading the slack
/// basis gives the weights it had at construction.
#[test]
fn load_basis_recomputes_the_primal_edge_weights() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, -2.0],
        &[0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 4.0),
            (to_sparse(&[1.0, 3.0]), ComparisonOp::Le, 6.0),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert!(sol.enable_primal_steepest_edge);
    let at_construction = sol.primal_edge_sq_norms.clone();
    sol.primal_edge_sq_norms.iter_mut().for_each(|w| *w = 7.0);
    let basis = sol.slack_basis();
    sol.load_basis(&basis).unwrap();
    assert_eq!(sol.primal_edge_sq_norms, at_construction);
}

/// A pivot element whose row and column computations disagree renews the
/// factorization at a raised threshold first; on a fresh one at the highest
/// threshold it is taken with the column's element, and a basis it leaves
/// singular would be repaired. Planted: `x`'s entry in `x <= 4` is `0.5` in
/// the column data and `1` in the row data, and the primal phase pivots on
/// it.
#[test]
fn a_primal_pivot_whose_computations_disagree_on_fresh_factors_is_taken() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0],
        &[0.0],
        &[f64::INFINITY],
        &[(to_sparse(&[1.0]), ComparisonOp::Le, 4.0)],
        &[VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.col_scales, [1.0], "fixture: no column scaling");
    assert_eq!(sol.row_scales, [1.0], "fixture: no row scaling");
    *sol.orig_constraints_csc.get_mut(0, 0).unwrap() = 0.5;
    sol.begin_primal_phase(true);
    let x = sol.price_primal().unwrap().unwrap();
    assert!(matches!(
        sol.choose_pivot(x, 1.0).unwrap(),
        PivotChoice::Inconsistent { row: 0 }
    ));
    sol.raise_lu_threshold();
    sol.rebuild_primal().unwrap();
    assert_eq!(sol.lu_threshold, LU_STABLE_THRESHOLD);
    let PivotChoice::Pivot(info) = sol.choose_pivot(x, 1.0).unwrap() else {
        panic!("x enters");
    };
    assert_eq!(info.elem.as_ref().unwrap().coeff, 0.5);
    assert_eq!(sol.primal_phase(1.0).unwrap(), (StopReason::Finished, true));
    sol.end_primal_phase().unwrap();
    assert!(matches!(sol.var_states[0], VarState::Basic(_)));
}

/// The dual simplex's counterpart: `x`'s entry in `x >= 1` is `0.5` in the
/// column data and `1` in the row data, and the violated row is repaired by
/// a pivot on it, with the column's element, once the factorization is
/// renewed at the highest threshold.
#[test]
fn a_dual_pivot_whose_computations_disagree_on_fresh_factors_is_taken() {
    init();
    let mut sol = Solver::try_new(
        &[1.0],
        &[0.0],
        &[f64::INFINITY],
        &[(to_sparse(&[1.0]), ComparisonOp::Ge, 1.0)],
        &[VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.col_scales, [1.0], "fixture: no column scaling");
    assert_eq!(sol.row_scales, [1.0], "fixture: no row scaling");
    *sol.orig_constraints_csc.get_mut(0, 0).unwrap() = 0.5;
    sol.begin_dual_phase(true).unwrap();
    let result = sol.dual_phase();
    sol.end_dual_phase();
    assert_eq!(result.unwrap(), StopReason::Finished);
    assert!(matches!(sol.var_states[0], VarState::Basic(_)));
    assert_eq!(sol.lu_threshold, LU_STABLE_THRESHOLD);
}

/// On a fresh factorization at the highest threshold, a dual pivot whose
/// column computation is zero, or of the other sign than its row
/// computation, cannot drive the step: the solve fails loudly instead of
/// dividing by it, or moving the entering var the wrong way past its bound.
/// In `x + y >= 1` with `min x + 2 y`, `x` (the smaller ratio) has its entry
/// planted as `1` in the row data and `0`, or `-0.5`, in the column data.
#[test]
fn a_dual_pivot_the_column_computes_as_zero_or_of_the_other_sign_fails_loudly() {
    init();
    for planted in [0.0, -0.5] {
        let mut sol = Solver::try_new(
            &[1.0, 2.0],
            &[0.0, 0.0],
            &[f64::INFINITY, f64::INFINITY],
            &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, 1.0)],
            &[VarDomain::Real, VarDomain::Real],
            None,
            crate::Tolerances::default().feasibility,
        )
        .unwrap();
        assert_eq!(sol.col_scales, [1.0, 1.0], "fixture: no column scaling");
        assert_eq!(sol.row_scales, [1.0], "fixture: no row scaling");
        *sol.orig_constraints_csc.get_mut(0, 0).unwrap() = planted;
        sol.begin_dual_phase(true).unwrap();
        let result = sol.dual_phase();
        sol.end_dual_phase();
        assert!(
            matches!(&result, Err(Error::InternalError(msg))
                if msg.contains("through the column of a fresh factorization")),
            "planted {planted}: {result:?}"
        );
        assert_eq!(sol.lu_threshold, LU_STABLE_THRESHOLD);
        assert!(matches!(sol.var_states[0], VarState::NonBasic(_)));
        assert_eq!(sol.get_value(0), 0.0);
    }
}

/// The cleanup of a primal phase moves a slack left off its bound beyond its
/// tolerance at the current values onto the bound. Issue #3's model with
/// `z ≤ 1e15`: the phase-1 dual phase starts `z` at 1e15, where the first
/// row's tolerance is taken (4.7 scaled, the row halved), and ends with the
/// row violated by 1.375 within it; the primal phase lets the slack leave
/// there, where its tolerance is now `5e-8` scaled, so the cleanup moves it
/// onto its bound and reports the violation for the dual simplex.
#[test]
fn a_slack_left_off_its_bound_beyond_its_current_tolerance_moves_onto_it() {
    init();
    let mut sol = Solver::try_new(
        &[-50.0, -40.0, -45.0],
        &[2.0, 0.0, 0.0],
        &[f64::INFINITY, 7.0, 1e15],
        &[
            (to_sparse(&[3.0, 2.0, 1.0]), ComparisonOp::Le, 20.0),
            (to_sparse(&[2.0, 1.0, 3.0]), ComparisonOp::Le, 15.0),
        ],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(
        sol.restore_feasibility(false).unwrap(),
        StopReason::Finished
    );
    sol.artificial_obj = None;
    sol.recalc_obj_coeffs().unwrap();
    let slack = sol.num_vars;
    let VarState::Basic(row) = sol.var_states[slack] else {
        panic!("the first row's slack is basic");
    };
    let (val, tol) = (sol.basic_var_vals[row], sol.var_tols[slack].lo);
    assert!(val < -0.5 && val > -tol, "slack {val:e}, tolerance {tol:e}");

    assert_eq!(
        sol.optimize(true, 1.0).unwrap(),
        (StopReason::Finished, true)
    );
    let VarState::NonBasic(col) = sol.var_states[slack] else {
        panic!("the first row's slack has left the basis");
    };
    assert_eq!(sol.nb_var_vals[col], 0.0);
    assert!(sol.var_tols[slack].lo < 1e-7);
    assert!(!sol.is_primal_feasible);
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert!(
        (sol.cur_obj_val + 420.0).abs() < 1e-9,
        "{}",
        sol.cur_obj_val
    );
}

/// Two equalities `x + y = 1` and `x + y = 1 + 1.5e-7` over `x, y ∈ [0, 2]`
/// disagree by `1.5e-7`, within their tolerances together but beyond either
/// alone, so no slack can absorb the miss, and the proof grants both slacks
/// their room. The dual phase's widening moves a slack's bound, and the
/// solve ends within the rows' contract, the model's bounds unchanged for the
/// boundary; the next solve restores them in the engine too.
#[test]
fn a_solve_widens_a_slack_bound_and_the_next_restores_it() {
    init();
    let mut sol = Solver::try_new(
        &[1.0, 0.0],
        &[0.0, 0.0],
        &[2.0, 2.0],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Eq, 1.0),
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Eq, 1.0 + 1.5e-7),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    let values: Vec<f64> = (0..2).map(|v| sol.get_value(v)).collect();
    assert_eq!(sol.first_violated_row(&values), None, "{values:?}");
    let w = *sol
        .widened
        .iter()
        .find(|w| w.var >= sol.num_vars)
        .expect("a slack's bound is widened");
    assert_eq!(sol.model_bounds(w.var), (0.0, 0.0), "the model's bounds");
    assert!(sol.orig_var_mins[w.var] < 0.0 || sol.orig_var_maxs[w.var] > 0.0);

    sol.restore_model_bounds();
    assert!(sol.widened.is_empty());
    assert_eq!(
        (sol.orig_var_mins[w.var], sol.orig_var_maxs[w.var]),
        (0.0, 0.0)
    );
}

/// The slack of `x + 1e-9 z <= 1` over `x ∈ [0, 2]` and `z ∈ [0, 1e12]`, its
/// lower bound widened by its room at the slack basis, where the row's
/// magnitude is its right-hand side's. Moved to `1e12`, `z` raises the
/// magnitude a thousandfold, and with it the round-off of evaluating the
/// row, which the widened side keeps as its tolerance: the bound then moves
/// back to the room the row leaves at those values.
#[test]
fn a_widened_slack_bound_follows_its_rows_room() {
    init();
    let mut sol = Solver::try_new(
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[2.0, 1e12],
        &[(to_sparse(&[1.0, 1e-9]), ComparisonOp::Le, 1.0)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.row_scales, [1.0], "fixture: the row is not scaled");
    let slack = sol.num_vars;
    let room = sol.slack_room(0);
    let edge = band_edge(0.0, room, false);
    sol.widen(slack, false, edge);
    assert_eq!(sol.orig_var_mins[slack], edge);
    assert_eq!(sol.model_bounds(slack), (0.0, f64::INFINITY));
    let margin = slack_margin(1e-7, 1.0, 2);
    assert_eq!(sol.var_tols[slack].lo, margin, "the side keeps the margin");

    let z = nb_col(&sol, 1);
    sol.apply_flips(&[(z, sol.orig_var_maxs[1])]);
    let refit = band_edge(0.0, sol.slack_room(0), false);
    assert!(refit > edge, "the room shrinks: {refit:e} {edge:e}");
    sol.refit_widened_slacks();
    assert_eq!(sol.orig_var_mins[slack], refit);
    let VarState::Basic(row) = sol.var_states[slack] else {
        panic!("the slack is basic");
    };
    assert_eq!(sol.basic_var_mins[row], refit);
}
