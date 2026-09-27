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

/// `entry_bounds` judges each side of a range on its own: a bound at or
/// beyond 2^52 on its side of infinity is infinite, whatever the other
/// bound (a var living beyond the threshold kept a far bound of `f64::MAX`
/// as written, and started on it), and a bound on the other side is data.
/// Only a range that is a point or empty is kept as written, and a NaN
/// bound stays NaN.
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
        // The range lies beyond the threshold: the far side is infinite,
        // the near side a bound.
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

/// `min Σ c_i x_i` over `x ∈ [0, 1]^8` with `Σ x_i ≥ 5` and costs `1..=8`:
/// the start (every var at its lower bound) is dual feasible, and the one
/// violated row is repaired by the four cheapest vars moving to their upper
/// bound and the fifth entering. The dual ratio test passes the breakpoints of
/// the four (flips, no pivot) and pivots only on the fifth, so the phase takes
/// one pivot where a ratio test without bound flipping takes five.
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

/// A point is finite: a value or an activity that is not finite violates
/// every bound and row whatever its sense. It would otherwise pass an
/// infinite bound, or meet the infinite tolerance an infinite magnitude
/// gives a row (an overflowing `Ge` activity used to pass, as did an
/// infinite value of a free var, which let a NaN objective through).
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

/// The single violated row `8 x1 + x2 / 8 >= 10` (equilibrated to
/// `x1 + x2 / 64 >= 1.25`) with both vars at their lower bound and a zero
/// cost: their breakpoints tie at a zero dual step. Flipping `x1` (entry 1,
/// capacity 1) and `x2` (entry 1/64, capacity 1) together carries the row, so
/// they form one Harris group and the pivot is its largest entry, `x1`, with
/// nothing flipped, as HiGHS's `chooseFinal` has it (C2-07). A walk over the
/// breakpoints one at a time flipped `x1` and pivoted on `x2`'s entry, 64
/// times smaller, and made that flip on a zero dual step, which neither
/// HiGHS nor SoPlex does (C2-09). The solve ends at the same point either
/// way.
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
        .choose_entering_col_dual(row, leaving_new_val, &[], false)
        .unwrap()
        .step();
    assert!(step.flips.is_empty(), "flips {:?}", step.flips);
    let (col, coeff) = step.entering;
    assert_eq!((sol.nb_vars[col], coeff), (0, 1.0));

    let mut sol = new_solver();
    assert_eq!(sol.initial_solve().unwrap(), StopReason::Finished);
    assert_eq!([sol.get_value(0), sol.get_value(1)], [1.0, 16.0]);
}

/// The same row with `x2` costing 1: its breakpoint now lies beyond the
/// first Harris group, which holds `x1` alone and cannot carry the row
/// (capacity 1 of 1.25). The groups up to `x2`'s can, but the last group's
/// only entry, 1/64, is below a tenth of the largest grouped entry, so the
/// pivot steps back to the first group, `x1`, and nothing is flipped (HiGHS
/// `chooseFinalLargeAlpha`). `x1` then enters past its other bound, and the
/// next iteration moves the rest of the violation onto `x2`.
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
        .choose_entering_col_dual(row, leaving_new_val, &[], false)
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

/// A column excluded for the row (its entry disagreed with the column
/// computation on a fresh factorization) no longer sets the pivot floor
/// (C2-15). Planted row `x0 + 1e-6 x1 + 1e-14 x2` of the violated row
/// `x0 + x1 + x2 >= 5`: `x0` sits at its upper bound (no help), `x1` is
/// excluded, and `x2` (unbounded above) is the only var left to repair the
/// row. With the floor taken over the excluded `x1`, `x2`'s entry was below
/// it and above the round-off floor, neither a candidate nor a term of the
/// verdict, and the row was declared infeasible.
#[test]
fn excluded_column_does_not_set_the_dual_pivot_floor() {
    init();
    let mut sol = Solver::try_new(
        &[-1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, f64::INFINITY],
        &[(to_sparse(&[1.0, 1.0, 1.0]), ComparisonOp::Ge, 5.0)],
        &vec![VarDomain::Real; 3],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.calc_row_coeffs(row);
    let col_of = |sol: &Solver, var: usize| sol.nb_vars.iter().position(|&v| v == var).unwrap();
    let (c0, c1, c2) = (col_of(&sol, 0), col_of(&sol, 1), col_of(&sol, 2));
    *sol.row_coeffs.get_mut(c0) = 1.0;
    *sol.row_coeffs.get_mut(c1) = 1e-6;
    *sol.row_coeffs.get_mut(c2) = 1e-14;
    let step = sol
        .choose_entering_col_dual(row, leaving_new_val, &[c1], false)
        .unwrap()
        .step();
    assert!(step.flips.is_empty());
    assert_eq!(step.entering, (c2, 1e-14));
}

/// The dual simplex applies a pivot with the element computed through the
/// column (C2-23): the eta file stores the column's own pivot, so the updated
/// factorization maps the entering column exactly onto the unit vector of
/// its row, and the primal step and the steepest-edge weight use it too,
/// while the reduced costs, updated along the row, use the row's element.
/// Planted: the row's element 1e-9 (relative) off the column's, within the
/// agreement check. With the row's element in the eta, the updated basis had
/// the entering column `a_q + (a_row - a_col) b_r` instead of `a_q`.
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
        .choose_entering_col_dual(row, leaving_new_val, &[], false)
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

/// KNOWN LIMITATION (ARCHITECTURE §7; C3-11 of the solver review): the
/// primal ratio test's round-off scale is the largest entry of the whole
/// column, so a genuine blocking entry below its `ENTRY_ROUNDOFF` share, next
/// to huge entries in rows that cannot block, counts as noise. Basis: the free
/// vars `y1`, `y2` of the nearly dependent equalities `y1 + y2 + x/8 = 0` and
/// `y1 + (1 + 2^-43) y2 + 8x = 0`, which put 7e13 into the column of `x` in
/// rows that never block, and the slack of `x/8 + 8z <= 1`, whose entry is
/// 1/64 after equilibration and exact (the basis is block diagonal). Nothing
/// blocks `x` and `max x` comes back `Unbounded` instead of `x = 8`. Taking
/// the scale over the rows that can block instead fixes this model but pivots
/// on round-off elsewhere (an entry of -4.4e-16 in a blocking row beside a
/// huge one, taken with a step of 5.6e15, left a singular basis on an LP
/// HiGHS finds unbounded), since the rows of one column share the round-off
/// of one solve; telling the two apart needs an absolute pivot tolerance and
/// a refined column, not a relative floor. When a change fixes this, the test
/// fails: drop the attribute and keep it as a regression test.
#[test]
#[should_panic(expected = "Unbounded")]
fn primal_ratio_test_scale_hides_a_blocking_entry_is_a_known_limitation() {
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

/// Dual steepest-edge weights are floored (C2-02, C2-03). A basis load
/// resets every weight to one, an underestimate of `||e_1^T B^-1||^2 = 4.61`
/// for `B = [[1, 0], [1.9, 1]]` (the basis `y`, `s1` of `y <= 3`,
/// `1.9 y <= 10`); the dual pivot that repairs `y = 3 > 1` then updated the
/// second row's weight to `1 - 1.9^2 = -2.61`, and a negative weight gives
/// its row a negative score. Also: a weight that overflows, or that a NaN
/// reaches, is the engine's error, not a panic (the floor once made a NaN
/// weight finite, review of R0).
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

/// Primal steepest-edge weights are floored at `1 + lambda^2`, a lower
/// bound of the true weight, as HiGHS does (C3-06): planted, the weight of
/// `x2` an underestimate far below its true value (as cancellation in the
/// update can leave one), which the update of the first pivot of
/// `min -x1 - x2` s.t. `x1 + x2 <= 4`, `x1 - x2 <= 2` left at -2. Also: a
/// weight that overflows, or that a NaN reaches, is the engine's error, not
/// a panic (the floor once made a NaN weight finite, review of R0).
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

/// A reduced cost that is not a number never ends a phase as optimal
/// (C4-07): it counts as dual infeasible, but its steepest-edge score, not
/// a number either, lost every comparison, so the primal simplex found no
/// entering column and the phase reported the vertex optimal. Planted: a
/// NaN in a verified optimum's reduced costs.
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
    assert!(matches!(sol.reoptimize(), Err(Error::InternalError(_))));
}

/// The same for a fixed var, which is dual feasible whatever its reduced
/// cost is, as long as it is a number: one that is not still counts as dual
/// infeasible, so the pricing reports it. When the dual infeasibility of a
/// fixed var was zero for any reduced cost, a NaN passed as feasible and
/// the phase ended optimal. Planted as above.
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
    assert!(matches!(sol.reoptimize(), Err(Error::InternalError(_))));
}

/// The non-basic column of `var`.
fn nb_col(sol: &Solver, var: usize) -> usize {
    sol.nb_vars.iter().position(|&v| v == var).unwrap()
}

/// The primal simplex's choice from the current state, as an iteration of
/// its phase makes it: the pricing, then the ratio test for the priced
/// column with no row excluded.
fn primal_pivot_choice(sol: &mut Solver, harris_scale: f64) -> PivotChoice {
    let col = sol.price_primal().unwrap().expect("a var can enter");
    sol.choose_pivot(col, &[], harris_scale).unwrap()
}

/// A dual step never runs backwards (C2-16): where the Harris pass lets the
/// entering var's reduced cost lie just across zero, the step is zero and
/// the var's cost is shifted, as HiGHS takes `workTheta = 0` and shifts the
/// cost. `min x0` s.t. `8 x0 + x1 >= 1e-4` (equilibrated to
/// `x0 + x1 / 8 >= 1.25e-5`), `x1 >= 0`, with `x1`'s reduced cost planted at
/// `-9e-11`, across zero within its tolerance of `1e-10`: its relaxed ratio
/// is the Harris bound and it enters. The step `d_q / a_q` was
/// `-9e-11 / 0.125`: the leaving slack, at its upper bound, ended with a
/// reduced cost of `+7.2e-10`, dual infeasible beyond its tolerance, where it
/// now ends at zero with every other reduced cost as it was.
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
        .choose_entering_col_dual(row, leaving_new_val, &[], false)
        .unwrap()
        .step();
    assert_eq!(sol.nb_vars[step.entering.0], 1);
    assert!(step.zero_step && step.flips.is_empty());

    // The phase: one pivot, on a zero step. The start violates its row by
    // little (near optimal), so the costs are not perturbed and the planted
    // reduced cost is what the ratio test sees.
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

/// A dual phase that starts dual infeasible first corrects its reduced costs
/// (C2-21, HiGHS `correctDualInfeasibilities`): a boxed var moves to the
/// bound where its reduced cost has the right sign, any other has its cost
/// shifted until the reduced cost is just feasible; in a phase that cleans
/// up after another, the boxed var's cost is shifted too. `min -x0 + x1 + x2 - x3`
/// s.t. `x2 >= 1`, `x3 <= 7`, `x0 in [0, 5]`, `x1 >= 0`, `x2 in [0, 10]`,
/// `x3 >= -3`, loaded with every var at its lower bound: `x0` (boxed) and
/// `x3` (bounded below only) have reduced cost -1 there. The dual phase ran
/// on those infeasibilities and left `x0` at zero for the primal simplex to
/// move; now it moves `x0` to 5 before its first step, and the phase ends
/// with the shift of `x3` given back, for the primal phase to take `x3` to 7.
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
        assert!(!sol.is_primal_feasible && !sol.is_dual_feasible);
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

    // A dual phase that cleans up after another forces phase 2, as HiGHS's
    // cleanup calls do (`pass_force_phase2`): the boxed `x0` keeps its bound
    // and has its cost shifted like any other var.
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

/// The cost perturbation of a dual phase, as HiGHS's `initialiseCost`: the
/// base is `5e-7` times the largest cost (its fourth root above 100); a
/// structural var's cost moves by `(1 + r) (|c| + 1) base` towards dual
/// feasibility at the bound it can sit at (up bounded below only, down
/// bounded above only, by the cost's sign when boxed), a free or fixed var's
/// not at all, and a slack's by at most `5e-13` either way.
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

    // An objective of zeros is perturbed as if its largest cost were one
    // (HiGHS's default path).
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

/// A cost of `1e20` or more (HiGHS's `infinite_cost`, which HiGHS's
/// perturbation never sees) takes no part in the cost perturbation: it
/// neither moves nor sets the base, so the other costs move as if it were
/// not there, by less than a tenth of `|c| + 1`, and every working cost is
/// finite. Before, `-1e300` moved by about `1e-6 * 1e375`: to minus
/// infinity.
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
/// by less than `1e-3`) is not perturbed (HiGHS `HEkkDual::solve`), and one
/// far from it is.
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

/// The cycling detection of the dual simplex (HiGHS `isBadBasisChange`): a
/// basis change leading to a visited vertex is cycling only when the change
/// before it did too; it is then recorded as bad and made taboo, a bad change
/// seen again is taboo again at once, the taboo is lifted at a rebuild, and a
/// record goes once its row has moved. The vertex includes where the
/// non-basic vars sit: a basic set reached again after a bound flip is a new
/// vertex, not a revisit.
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
    // (var 1 replaces 11), the change leads to a vertex never visited and is
    // no longer known as bad.
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

/// A dual phase whose fresh data leave nothing but taboo basis changes fails
/// loudly (HiGHS's `kSolvePhaseTabooBasis`, which returns `kUnknown`),
/// instead of reporting an answer or running on. `min x0 + 2 x1` s.t.
/// `x0 + x1 >= 1`, `x in [0, 10]^2`, from the slack basis: the change of the
/// one violated row (`x0` enters for the slack) is planted as bad, as two
/// consecutive revisits record it. The row is skipped while taboo; the
/// rebuild that follows lifts the taboo, the change is chosen again and is
/// taboo again at once, and nothing else is left. Without the record the
/// same phase makes that change and finishes. The phase runs under a time
/// limit, so that a phase that went on instead fails the test.
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
        .choose_entering_col_dual(row, leaving_new_val, &[], false)
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

/// The compensated sums of the infeasibility proof keep what `f64` rounds
/// away (HiGHS's `HighsCDouble`): `1e16 + 1 - 1e16` is one, where `f64`
/// gives zero, and a product is exact: `(1 + 2^-30)^2` keeps its `2^-60`.
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
/// and `y ∈ [0, y_max]`, violated at the slack basis.
fn proof_of_a_single_row(rhs: f64, y_max: f64) -> proof::Proof {
    let mut sol = Solver::try_new(
        &[0.0, 0.0],
        &[0.0, 0.0],
        &[1.0, y_max],
        &[(to_sparse(&[1.0, 1.0]), ComparisonOp::Ge, rhs)],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    sol.prove_infeasible(row, leaving_new_val > sol.basic_var_vals[row])
}

/// The infeasibility proof judges the model's data (C2-14, C4-33 of the
/// solver review): at the slack basis the certificate is the row itself,
/// `x + y + s = rhs` with `s <= 0`, whose largest value over the box is 2.
/// For `rhs = 3` it is out of reach by one: proven. The gap must exceed the
/// tolerance the engine holds the row's slack to, half the user's `1e-7`:
/// out of reach by `3e-8`, a point the dual simplex accepts misses the row
/// by that much, so nothing is proven; by `2e-7`, it is. With `y` unbounded
/// above the range is unbounded: nothing is proven, however far the row is.
#[test]
fn infeasibility_proof_needs_a_gap_beyond_the_tolerance() {
    init();
    let proof = proof_of_a_single_row(3.0, 1.0);
    assert!(proof.holds, "{proof:?}");
    assert!((proof.gap - 1.0).abs() < 1e-15, "{proof:?}");

    let proof = proof_of_a_single_row(2.0 + 3e-8, 1.0);
    assert!(!proof.holds, "{proof:?}");
    assert!(proof.gap > 2.9e-8 && proof.allowance > 4.9e-8, "{proof:?}");

    let proof = proof_of_a_single_row(2.0 + 2e-7, 1.0);
    assert!(proof.holds, "{proof:?}");

    let proof = proof_of_a_single_row(3.0, f64::INFINITY);
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
    }
}

/// The row of the basis inverse the proof starts from is refined, with
/// HiGHS's `unitBtranIterativeRefinement`: its residual
/// `B^T rho - e_row`, summed in compensated arithmetic, is solved for and
/// taken off, twice. Planted: at [`fifths_basis`], the computed row
/// perturbed by `1e-6`; the refinement brings it back to within round-off of
/// the computed row (and so its basic coefficients, `B^T rho`, to the unit
/// vector), and it converges: the second correction, the bound it reports
/// on the error of each multiplier, is some ten orders below the first.
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

/// The refinement keeps its corrections to twice the precision of an `f64`
/// (as LAPACK's extra-precise refinement keeps its solution), so that the
/// basic coefficients of the certificate vanish where the basis defines
/// multipliers no `f64` holds. At [`fifths_basis`], the refined row as a
/// compensated sum combines the basic columns into the unit vector to within
/// `1e-30`, and the bound it reports on the error of each multiplier is of
/// that order too; rounded to `f64`, as HiGHS rounds it, the same row leaves
/// the error of its representation, some `1e-17`, which times a scaled bound
/// of `1e9` to `6e10` voided the proofs of three infeasible MILP nodes of the
/// property stress (`multipliers_no_f64_holds_do_not_void_a_nodes_proof` in
/// `src/tests/magnitudes.rs`).
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

/// The bound the refinement puts on the error of each multiplier is its
/// second correction, and only where the refinement converged in LAPACK's
/// sense, that correction at most `2^-53` of the refined row in the max
/// norm: a solve that leaves more cannot measure its own error, and a bound
/// taken from it could make a coefficient of the model's own count as zero.
/// With no bound, only the round-off of the certificate's own sums does
/// (`a_coefficient_on_an_infinite_bound_counts_as_zero_only_within_its_error`).
/// A row solved exactly has a zero correction and a zero bound; a row or a
/// correction that is not a number leaves no bound (`f64::max` would drop
/// it).
#[test]
fn the_refinement_bounds_the_multipliers_error_only_where_it_converged() {
    let row = |values: &[f64]| multipliers(values);
    assert_eq!(
        proof::multipliers_error(&row(&[1.0, -2.0]), &[3e-33, -1e-34]),
        Some(vec![3e-33, 1e-34])
    );
    assert_eq!(
        proof::multipliers_error(&row(&[1.0, 0.5]), &[0.0, 0.0]),
        Some(vec![0.0, 0.0])
    );
    assert_eq!(
        proof::multipliers_error(&row(&[1e20]), &[1e4]),
        Some(vec![1e4])
    );
    assert_eq!(proof::multipliers_error(&row(&[1.0]), &[2e-16]), None);
    assert_eq!(proof::multipliers_error(&row(&[1e-20]), &[1e-30]), None);
    assert_eq!(
        proof::multipliers_error(&row(&[1.0, 1.0]), &[1e-33, f64::NAN]),
        None
    );
    assert_eq!(
        proof::multipliers_error(&row(&[f64::INFINITY]), &[0.0]),
        None
    );
}

/// A non-basic var's coefficient of the certificate counts however far
/// below its terms it cancelled, at its bound where that is finite, and as
/// an unbounded range where it is not: it is the model's own. Planted: at
/// the slack basis, `rho` combines `x + y + z >= 3` and
/// `t - (1 + 2^-50) z >= 1.5` over `x, y, t ∈ [0, 1]` and `z <= 1` (which
/// starts at 1, its cost being negative), which leaves `z` the coefficient
/// `-2^-50` beside terms of magnitude two, round-off, and the rest of the
/// certificate out of reach by `1.5`. With `z >= -2^51` the coefficient
/// times the bound takes the certificate to `5`, beyond `beta = 4.5`, and
/// with `z` unbounded below anywhere: nothing is proven either way. Counted
/// as zero, on the finite bound and then on the infinite one, such a
/// coefficient proved a feasible model infeasible
/// (`a_round_off_coefficient_on_a_far_bound_is_no_proof_and_is_a_known_limitation`
/// and its variant on an infinite bound, in `src/tests/magnitudes.rs`).
/// This model is infeasible all the same: another combination of the rows
/// proves it.
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

/// A number of the certificate that is not finite voids it, rather than
/// count as zero or as a gap. Planted: `x + y + 1e305 f >= 3` over
/// `x, y ∈ [0, 1]` and `f` fixed at `1e-305`, feasible at `x = y = 1`, taken
/// alone: a fixed column keeps the user's units, and Dekker's split of
/// `f`'s coefficient overflows, which leaves it not a number. Counted as
/// zero (a NaN is above no round-off threshold), it left `x + y + s = 3`
/// with `s <= 0` out of reach by one: a proof of a feasible model.
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

/// An error bound of a coefficient that is not finite voids the proof, as
/// any number of the certificate that is not finite does, rather than count
/// every coefficient within it as zero. Planted: `x + y + j >= 3` over
/// `x, y ∈ [0, 1]` and `j` free, feasible at `j >= 1`, taken alone with its
/// multiplier's error unbounded (as an overflow of `Σ e_i |a_ij|` would
/// leave it): counted as zero, `j`'s coefficient would leave `x + y + s = 3`
/// with `s <= 0` out of reach by one, a proof of a feasible model.
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
    };
    let proof = sol.proof_of(&unbounded_error, row, false);
    assert!(!proof.holds, "{proof:?}");
    assert!(proof.gap.is_nan(), "{proof:?}");
    let proof = sol.proof_of(&exact_row(&[1.0]), row, false);
    assert!(!proof.holds, "{proof:?}");
    assert_eq!(proof.gap, f64::NEG_INFINITY, "{proof:?}");
}

/// A multiplier of the certificate whose row would need its slack's infinite
/// bound is dropped, with its row, as HiGHS drops it ("make sure infinite
/// sides are not used"): what is left is a certificate of the other rows.
/// Planted: `rho` combines `x + y >= 3` over `x, y ∈ [0, 1]` with `1e-3` of
/// `z <= 5`, whose slack has no upper bound; kept, that multiplier would make
/// the range unbounded, and nothing would be proven (without the rule the
/// differential fuzz's `int_lp-970`, an infeasible LP, failed loudly).
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

/// A basic var of a row the certificate leaves out while the row's
/// multiplier is the model's own (its slack's tableau entry) keeps the
/// model's own coefficient, and on an infinite bound that makes the range
/// unbounded: only what the error of the multipliers could leave counts as
/// zero, and this row of the basis inverse is exact. The model of the review
/// of the concession: `min -L` s.t. `L + j + w = 1.000001`,
/// `-(1 + 2^-50) j - w = 0` and `j + v <= 0` over `L, v ∈ [0, 1]` and
/// `j, w` free, feasible at `L = 1`, `j = -2^50 (fl(1.000001) - 1)`,
/// `w = -(1 + 2^-50) j`, where the dual simplex ends: the basis `{j, w}` and
/// the second row's slack, `L` at its upper bound, which puts that slack
/// `1e-6` above its bound of zero. The row of the basis inverse is
/// `(1, 1, 2^-50)`, solved exactly: both corrections of the refinement are
/// zero, and so is the bound they put on the multipliers' error. The third
/// row's multiplier needs its non-basic slack's infinite side and is left
/// out, which leaves the free basic `j` the coefficient `-2^-50` beside terms
/// of one, `4.4e-16` of them. Conceded as below the `f64` round-off floor of
/// its terms, it left `L + s1 + s2 = 1.000001`, out of reach by `1e-6`: a
/// proof of a feasible model (and of its MILP, whose root node was pruned).
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
/// error of the multipliers could make it zero, whatever its var: within the
/// bound the refinement puts on each multiplier's error, carried through the
/// var's entries in the rows the certificate keeps, plus the error of its
/// compensated sum; beyond that it makes the range unbounded, and with no
/// bound only the error of the sum counts. Planted: `x + y + j + w >= 3` and
/// `j + z + w = 0` over `x, y, z ∈ [0, 1]` and `j, w` free, infeasible
/// (`j + w = -z <= 0`), at the basis `{j}` and the first row's slack, `3`
/// above its bound, with `w` non-basic; the row of the basis inverse is
/// `(1, -1)`. The planted, compensated multipliers `(1, -(1 + d))` leave the
/// basic `j` and the non-basic `w` the coefficient `-d` beside terms of two,
/// and the rest of the certificate out of reach by one. At `d = 2^-80` with
/// the second multiplier's error bounded by `2^-79`, both count as zero:
/// proven. Bounded by `2^-82`, or with no bound, neither does: nothing is
/// proven. At `d = 2^-100`, within the error of the compensated sums, both
/// count as zero with the error bounded by zero, and with no bound.
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
/// recorded as a bad basis change with no entering var (HiGHS's
/// `kFailedInfeasibilityProof`): taboo, and unproven, until a rebuild lifts
/// the taboo; the record goes when a step moves its row.
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

/// The model of C2-06 through the LP edit path, to the point where the dual
/// phase takes over: `min y` s.t. `f + y >= 0` over `f ∈ [0, 10]`, `y >= 0`,
/// then `f` fixed at zero and `1e17 f + y >= 1` added. The added row leaves
/// the fixed `f` out of its equilibration, so its row in the dual ratio test
/// is `[1e17, 1]` with `f`'s entry in the round-off scale, and `y`'s exact
/// entry of one is below the pivot floor: no candidate. The phase is stopped
/// before its first iteration (a deadline in the past) and started again.
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

/// The last resorts of the dual simplex for a row that no var can repair and
/// that does not prove infeasibility (C2-06): the row is taboo and nothing
/// else is left, so SoPlex's instable leave takes `y`'s entry of one, far
/// above its `lowstab`, and the phase ends at `y = 1`, where the verdict read
/// off the tableau row declared the model infeasible. When that pivot is
/// refused (planted here as a bad basis change, as the cycling detection
/// records one), the instable leave is not tried again: the round-off repair
/// has nothing to move (`y` has no upper bound) and the phase fails loudly at
/// once, where a phase that scheduled the instable leave again would go round
/// until its stall bound ended it. The phase runs under a time limit, so that
/// a phase that went on instead fails the test.
#[test]
fn a_row_the_proof_leaves_open_gets_the_instable_leave_once() {
    init();
    let mut sol = c2_06_through_an_edit();
    assert_eq!(sol.reoptimize().unwrap(), StopReason::Finished);
    assert_eq!((sol.get_value(0), sol.get_value(1)), (0.0, 1.0));

    let mut sol = c2_06_through_an_edit();
    sol.begin_dual_phase(false).unwrap();
    let (row, leaving_new_val) = sol.choose_pivot_row_dual().unwrap();
    assert_eq!(row, 1, "fixture: the added row is the violated one");
    sol.calc_row_coeffs(row);
    assert!(matches!(
        sol.choose_entering_col_dual(row, leaving_new_val, &[], false),
        Ok(DualRatioTest::NoCandidate)
    ));
    let (y_col, _) = sol
        .choose_entering_col_dual(row, leaving_new_val, &[], true)
        .unwrap()
        .step()
        .entering;
    assert_eq!(sol.nb_vars[y_col], 1);
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
        matches!(&result, Err(Error::InternalError(msg)) if msg.contains("no var can repair row")),
        "{result:?}"
    );
    assert!(sol.lp_iterations - iterations < 20, "{}", sol.lp_iterations);
}

/// The bound perturbation of a primal phase, as HiGHS's `initialiseBound`
/// in user units: every finite bound moves outward by `r 5e-7 max(1, |b|)`
/// (a slack's by the magnitude of its row's right-hand side), mapped into
/// the engine's scaled units like its tolerances; a fixed var is never
/// perturbed, basic or not; a non-basic var at a bound moves with it and the
/// basic values follow.
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

/// A primal phase is perturbed only when it may be (it is not the cleanup of
/// another phase) and far from optimal (HiGHS `HEkkPrimal::solve`: a
/// thousand or more dual infeasibilities, or one of `1e-3` or more), and its
/// cleanup takes every non-basic var back to the true bound it sits at a
/// moved version of: `min c (x0 + x1)` s.t. `x0 + x1 <= 4`, `x0 - x1 <= 2`,
/// `x >= 0`, starts primal feasible at zero with both reduced costs `c`, and
/// ends at the vertex `(3, 1)` exactly, however it was perturbed.
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
        assert!(sol.is_primal_feasible && !sol.is_dual_feasible);
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

/// A leaving var that sits past the bound it leaves at (within its
/// tolerance, as the Harris pass allows) gives a zero step, not a step
/// backwards, and leaves where it sits (SoPlex's `maxReLeave`): `min -x0`
/// s.t. `x0 + x1 <= -3.5e-8`, `x >= 0`, whose slack starts at `-3.5e-8`,
/// within its tolerance of `5e-8`. The step backwards to the slack's bound
/// would move `x0` to `-3.5e-8`, out of its own bounds; the clamp this
/// replaced left the slack where it sat too (C3-21), but unrecorded, and
/// set a var beyond its tolerance onto bound plus tolerance. Past its true
/// bound the var keeps its working bounds (it is at its bound in the
/// engine's sense) and stays there through the cleanup, since it is within
/// its tolerance of it at the current values.
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
    assert_eq!(sol.var_tols[2].lo, 5e-8);
    // The start is primal infeasible by the exact test `try_new` makes, so
    // it came with the phase-1 objective; the primal phase works on the real
    // one, as `run_phases` has it.
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

/// The Harris pass's limit on the step is never below zero (review of R1b).
/// In a round of `run_phases` after the first, the primal ratio test allows
/// a basic var a hundredth of its tolerance past its bound
/// (`LOOP_HARRIS_REDUCTION`), and a var already past it by more (within its
/// whole tolerance, as `correct_primal` leaves it) admits no step toward it:
/// a limit of zero. `min -x0` s.t. `x0 + x1 <= 0`, `1e-16 x0 + x1 <=
/// -2.5e-8`, `x0 >= 0`, `x1` in `[0, 10]`: the first slack sits at its
/// bound, the second `2.5e-8` past it, within its tolerance of `5e-8`, and
/// the scaling puts `x0`'s entries at `1` and `1.34e-8`. With the allowance
/// cut, the second row's limit was `(-2.5e-8 + 5e-10) / 1.34e-8`, negative,
/// and only that row's own step was within it: the zero step pivoted on
/// `1.34e-8`, below the pivot floor, and passed over the first row's `1`, at
/// its bound. Clamped to zero, the limit lets every row at or past its bound
/// compete by its entry, in every round alike.
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
    assert_eq!(sol.var_tols[3].lo, 5e-8);
    // The start is primal infeasible by the exact test `try_new` makes, so
    // it came with the phase-1 objective; the primal phase works on the real
    // one, as `run_phases` has it.
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

/// The working bounds a leaving var holds (see `hold_leaving_value`): past
/// its true bound, within its tolerance, it keeps them; past a bound the
/// phase moved, that bound moves on to where it sits; a fixed var is fixed
/// at its value when it leaves within its tolerance of it, and where it sits
/// otherwise. The cleanup then takes a var at a moved bound back to its true
/// one, and a slack off its bound beyond its tolerance at the current values
/// onto it.
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

/// A step moves the basic vars whose entries are too small for the ratio
/// test to count, and one of them can end outside its bounds beyond its
/// tolerance: its bound is shifted, and the phase goes on over bounds its
/// point satisfies (HiGHS `phase2UpdatePrimal`, C3-23); the cleanup restores
/// the bound and reports the violation, for the dual simplex. The model is
/// `lp_wide_3` of the solver review's stress run: from the vertex its dual
/// phase ends at, `x0` enters, its entry in `-1e4 x2 + 1e-3 x3 >= 0` is a
/// genuine `3e-15`, below the round-off floor of its column (whose largest
/// entry is 4), and the step of 32 takes the row's slack `9.5e-14` past its
/// bound, twice its tolerance. (The whole solve still fails: the dual
/// simplex repairs the row by moving `x5` across its range through another
/// such entry, which gives the primal simplex its step back, round after
/// round; C2-10 of the review, for the infeasibility proof of step R3.)
#[test]
fn a_step_past_a_row_the_ratio_test_ignores_shifts_its_bound() {
    init();
    let mut sol = Solver::try_new(
        &[0.0, 0.0, 0.0, 1.0, 1.0, -1.0],
        &[-2.0, 0.0, 0.0, -1.0, 0.0, -1.0],
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        &[
            (
                to_sparse(&[1.0, 0.0, 0.0, -0.01, 0.0, 1.0]),
                ComparisonOp::Eq,
                0.0,
            ),
            (
                to_sparse(&[0.0, 0.0, -10000.0, 0.001, 0.0, 0.0]),
                ComparisonOp::Ge,
                0.0,
            ),
            (
                to_sparse(&[-1.0, 0.0, -1e-8, 10000.0, 0.0, 0.0]),
                ComparisonOp::Eq,
                0.0,
            ),
        ],
        &vec![VarDomain::Real; 6],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(
        sol.restore_feasibility(false).unwrap(),
        StopReason::Finished
    );
    assert!(sol.verify_primal().unwrap());
    sol.recalc_obj_coeffs().unwrap();
    assert!(sol.calc_dual_infeasibility().0 > 0);
    let slack = sol.num_vars + 1;
    let VarState::Basic(row) = sol.var_states[slack] else {
        panic!("the slack of the second row is basic");
    };

    sol.begin_primal_phase(true);
    assert!(!sol.bounds_modified);
    assert_eq!(sol.primal_phase(1.0).unwrap(), (StopReason::Finished, true));
    assert!(sol.bounds_modified);
    let tol = sol.var_tols[slack].hi;
    let val = sol.basic_var_vals[row];
    assert!(
        val > tol,
        "the row is violated beyond its tolerance: {val:e}"
    );
    assert!(
        sol.basic_var_maxs[row] > val + tol,
        "and its bound shifted past it"
    );
    assert!(sol.calc_primal_infeasibility().0 == 0);

    sol.end_primal_phase().unwrap();
    assert!(!sol.bounds_modified);
    assert_eq!(sol.basic_var_maxs[row], 0.0);
    assert!(!sol.is_primal_feasible);
    assert!(sol.calc_primal_infeasibility().0 == 1);
}

/// The cycling detection of the primal simplex (HiGHS `isBadBasisChange`,
/// with the taboo on the entering var, `applyTabooVariableIn`): a primal
/// phase whose fresh data leave nothing but taboo basis changes fails
/// loudly (HiGHS's `kSolvePhaseTabooBasis`, which returns `kUnknown`)
/// instead of reporting an answer or running on. `min -x0 - 2 x1` s.t.
/// `x0 + x1 <= 4`, `x >= 0`, from the slack basis: the change `x1` in for
/// the slack is planted as bad, as two consecutive revisits record it, and
/// `x0`'s reduced cost is made feasible, so that `x1` is the only var that
/// can enter. The pricing skips it while it is taboo; the rebuild that
/// follows lifts the taboo, the change is chosen again and is taboo again at
/// once, and nothing else is left. Without the record the same phase makes
/// that change and finishes. The phase runs under a time limit, so that a
/// phase that went on instead fails the test.
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

/// The rows a primal phase excludes as round-off are excluded for the
/// column they were judged in only (review of R1b): once another var enters,
/// its own entries in those rows bound its step. `min -2 x0 - x1` s.t.
/// `x0 + x1 <= 1e-6`, `x0 - x1 <= 1e4`, `x >= 0`, with `x0`'s entry in the
/// first row planted as `1e-9` in the column data and `2e-9` in the row
/// data, so that its column and row computations disagree as round-off's
/// do: the first row, with the smallest limit, is `x0`'s pivot row, its
/// entry disagrees on a fresh factorization, and it is excluded for `x0`.
/// The pivot on the second row that `x0` takes then is planted as a bad
/// basis change, so `x0` becomes taboo and `x1` is priced with no pivot
/// between. The first row is `x1`'s only blocking row, and the exclusions,
/// reset only after a pivot, hid it: the phase took `x1`'s ray for an
/// unbounded one. Now `x1` enters on that row, the step moves the second row
/// and so drops the planted record, and `x0` enters later.
#[test]
fn rows_excluded_as_round_off_for_one_column_block_another() {
    init();
    let mut sol = Solver::try_new(
        &[-2.0, -1.0],
        &[0.0, 0.0],
        &[f64::INFINITY, f64::INFINITY],
        &[
            (to_sparse(&[1.0, 1.0]), ComparisonOp::Le, 1e-6),
            (to_sparse(&[1.0, -1.0]), ComparisonOp::Le, 1e4),
        ],
        &[VarDomain::Real, VarDomain::Real],
        None,
        crate::Tolerances::default().feasibility,
    )
    .unwrap();
    assert_eq!(sol.col_scales, [1.0, 1.0], "fixture: no column scaling");
    assert_eq!(sol.row_scales, [1.0, 1.0], "fixture: no row scaling");
    *sol.orig_constraints_csc.get_mut(0, 0).unwrap() = 1e-9;
    *sol.orig_constraints.get_mut(0, 0).unwrap() = 2e-9;
    sol.begin_primal_phase(true);
    let x0 = sol.price_primal().unwrap().unwrap();
    assert_eq!(sol.nb_vars[x0], 0, "fixture: x0 is priced first");
    assert!(
        matches!(
            sol.choose_pivot(x0, &[], 1.0).unwrap(),
            PivotChoice::Inconsistent { row: 0, tiny: true }
        ),
        "fixture: x0's entry in the first row is found round-off"
    );
    let PivotChoice::Pivot(info) = sol.choose_pivot(x0, &[0], 1.0).unwrap() else {
        panic!("x0 enters");
    };
    assert_eq!(
        info.elem.as_ref().unwrap().row,
        1,
        "fixture: with the first row excluded, x0 pivots on the second"
    );
    // The change `x0` in for the slack of the second row (var 3) is planted
    // as bad, as two consecutive revisits record it, and left not taboo, as
    // a rebuild leaves it.
    let here = sol.cycle_guard.hash_after(std::iter::empty());
    assert!(!sol.cycle_guard.is_bad_change(1, 3, 0, here));
    sol.cycle_guard.record_pivot();
    assert!(sol.cycle_guard.is_bad_change(1, 3, 0, here));
    sol.cycle_guard.clear_taboo();
    assert!(!sol.cycle_guard.is_taboo_entering(0));
    assert_eq!(sol.primal_phase(1.0).unwrap(), (StopReason::Finished, true));
    assert!(matches!(sol.var_states[0], VarState::Basic(_)));
    assert!(matches!(sol.var_states[1], VarState::Basic(_)));
}

/// The cleanup of a primal phase moves a slack left off its bound beyond its
/// tolerance at the current values onto the bound. Issue #3's model with
/// `z ≤ 1e15`: `max 50x + 40y + 45z` s.t. `3x + 2y + z ≤ 20`,
/// `2x + y + 3z ≤ 15`, `x ≥ 2`, `y ≤ 7`. The phase-1 dual phase starts `z`
/// at 1e15, where the first row's tolerance is taken (2.5 in its scaled
/// units, where the row is halved), and ends with that row violated by 1.375
/// (0.6875 scaled) within it; the primal phase that follows, unperturbed
/// here, lets the row's slack leave the basis where it sits (a step
/// backwards would be zero). Its tolerance at the current values is
/// `2.5e-8` (`5e-8` in user units): the cleanup moves it onto its bound,
/// recomputes the values, and reports the violation that moves into `x` for
/// the dual simplex, where the slack used to stay and the boundary reported
/// the row violated.
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
