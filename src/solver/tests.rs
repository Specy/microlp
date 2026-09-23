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
        .choose_entering_col_dual(row, leaving_new_val, &[], &[])
        .unwrap();
    assert!(step.flips.is_empty(), "flips {:?}", step.flips);
    let (col, coeff) = step.entering.unwrap();
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
        .choose_entering_col_dual(row, leaving_new_val, &[], &[])
        .unwrap();
    assert!(step.flips.is_empty(), "flips {:?}", step.flips);
    let (col, coeff) = step.entering.unwrap();
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
        .choose_entering_col_dual(row, leaving_new_val, &[c1], &[])
        .unwrap();
    assert!(step.flips.is_empty());
    assert_eq!(step.entering, Some((c2, 1e-14)));
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
        .choose_entering_col_dual(row, leaving_new_val, &[], &[])
        .unwrap();
    let (col, a_row) = step.entering.unwrap();
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
