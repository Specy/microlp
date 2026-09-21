use super::search::{try_adopt_incumbent, try_warm_start};
use super::state::Incumbent;
use super::*;
use crate::{ComparisonOp, OptimizationDirection, Problem, StopReason};
use core::time::Duration;

fn int_2var_problem() -> Problem {
    // minimize 3a + 4b s.t. a + 2b >= 5, 3a + b >= 4; a,b integer in [0,10].
    // LP relaxation: a=0.6, b=2.2, obj 10.6. Integer optimum: a=1, b=2, obj 11.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let a = p.add_integer_var(3.0, (0, 10));
    let b = p.add_integer_var(4.0, (0, 10));
    p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
    p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
    p
}

fn binary_knapsack() -> Problem {
    // maximize 8x + 11y + 6z + 4w s.t. 5x + 7y + 4z + 3w <= 14, binaries.
    // Optimum: y + z + w = 21 (weight 14).
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_binary_var(8.0);
    let y = p.add_binary_var(11.0);
    let z = p.add_binary_var(6.0);
    let w = p.add_binary_var(4.0);
    p.add_constraint(
        &[(x, 5.0), (y, 7.0), (z, 4.0), (w, 3.0)],
        ComparisonOp::Le,
        14.0,
    );
    p
}

fn incumbent_obj(state: &MipState) -> f64 {
    state.incumbent.as_ref().unwrap().objective
}

#[test]
fn driver_finds_integer_optimum() {
    let run = run(&int_2var_problem(), SolveOptions::default()).unwrap();
    assert_eq!(run.reason, TerminationReason::ProvenOptimal);
    // Internal space == user space for Minimize.
    assert!((incumbent_obj(&run.state) - 11.0).abs() < 1e-6);
    let inc = run.state.incumbent.as_ref().unwrap();
    assert!((inc.values[0] - 1.0).abs() < 1e-6);
    assert!((inc.values[1] - 2.0).abs() < 1e-6);
    assert!(run.state.stats.nodes_solved > 0);
}

#[test]
fn driver_binary_knapsack_maximize() {
    let run = run(&binary_knapsack(), SolveOptions::default()).unwrap();
    assert_eq!(run.reason, TerminationReason::ProvenOptimal);
    // Maximize is negated internally: internal optimum is -21.
    assert!((incumbent_obj(&run.state) + 21.0).abs() < 1e-6);
}

#[test]
fn driver_no_integer_point_is_infeasible() {
    // 2x == 1 with x integer in [0,10]: LP-feasible (x=0.5), integer-infeasible.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(1.0, (0, 10));
    p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
    assert_eq!(
        run(&p, SolveOptions::default()).unwrap_err(),
        crate::Error::Infeasible
    );
}

#[test]
fn driver_exact_node_exhaustion_reports_infeasible_not_interrupted() {
    // Same infeasible fixture (2x == 1, x int in [0,10]): the root LP is
    // fractional (x=0.5) and branches into x<=0 and x>=1, both LP-infeasible.
    // The tree is therefore exactly two nodes: node_limit=1 interrupts and
    // node_limit=2 exhausts it. At the exact exhaustion count, the empty-open
    // check precedes the node-limit check and must report Infeasible.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(1.0, (0, 10));
    p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
    let mut options = SolveOptions::default();
    options.node_limit = Some(2);
    assert_eq!(run(&p, options).unwrap_err(), crate::Error::Infeasible);
}

#[test]
fn driver_node_limit_equal_to_exhaustion_count_reports_optimal() {
    // int_2var_problem is proven optimal in exactly 2 B&B nodes (deterministic:
    // the unlimited solve reports nodes_solved == 2; node_limit=1 -> Interrupted,
    // node_limit=2 -> Optimal). Setting the node limit to that exact count must
    // report Optimal, not Interrupted: when the tree empties on the same
    // iteration the limit would fire, the empty-`open` check at the loop top
    // wins and the finished proof is reported honestly.
    assert_eq!(
        run(&int_2var_problem(), SolveOptions::default())
            .unwrap()
            .state
            .stats
            .nodes_solved,
        2,
        "node count must stay deterministic; update the limit below if it changes"
    );
    let mut options = SolveOptions::default();
    options.node_limit = Some(2);
    let r = run(&int_2var_problem(), options).unwrap();
    assert_eq!(r.reason, TerminationReason::ProvenOptimal);
    assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
}

#[test]
fn driver_node_limit_interrupts_and_resumes_to_same_optimum() {
    let mut options = SolveOptions::default();
    options.node_limit = Some(1);
    let mut r = run(&int_2var_problem(), options).unwrap();
    let mut guard = 0;
    while r.reason != TerminationReason::ProvenOptimal {
        guard += 1;
        assert!(guard < 10_000, "resume loop did not terminate");
        r.reason = resume_run(&mut r.state, ResumeOptions::default()).unwrap();
    }
    assert!(guard >= 1, "node_limit=1 should interrupt at least once");
    assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
}

#[test]
fn open_children_have_branch_metadata() {
    let mut options = SolveOptions::default();
    options.node_limit = Some(0);
    let run = run(&int_2var_problem(), options).unwrap();

    assert_eq!(run.reason, TerminationReason::NodeLimit);
    assert_eq!(run.state.open.len(), 2);
    assert!(run.state.open.iter().all(|node| node.branch_var.is_some()));
}

#[test]
fn driver_zero_time_limit_interrupts_cleanly_then_resumes() {
    let mut options = SolveOptions::default();
    options.time_limit = Some(Duration::ZERO);
    let mut r = run(&binary_knapsack(), options).unwrap();
    assert_eq!(r.reason, TerminationReason::TimeLimit);
    assert!(r.state.incumbent.is_none());
    let resume_options = ResumeOptions {
        time_limit: Some(Duration::from_secs(10)),
        ..ResumeOptions::default()
    };
    let reason = resume_run(&mut r.state, resume_options).unwrap();
    assert_eq!(reason, TerminationReason::ProvenOptimal);
    assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
}

#[test]
fn optimal_solve_reports_zero_gap_and_matching_bound() {
    let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
    assert_eq!(r.reason, TerminationReason::ProvenOptimal);
    assert_eq!(r.state.stats.gap, Some(0.0));
    // User space == internal for Minimize.
    assert!((r.state.stats.best_bound.unwrap() - 11.0).abs() < 1e-6);
}

#[test]
fn unsolved_root_with_incumbent_has_no_proven_bound_or_gap() {
    let problem = binary_knapsack();
    let mut state = build_state(&problem, SolveOptions::default()).unwrap();
    state.incumbent = Some(Incumbent {
        values: vec![0.0; problem.obj_coeffs.len()],
        objective: 0.0,
    });
    state.root_solved = false;
    state.open.clear();

    fill_bound_stats(&mut state);

    assert_eq!(state.stats.best_bound, None);
    assert_eq!(state.stats.gap, None);
}

#[test]
fn maximize_bound_is_in_user_space() {
    let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
    assert_eq!(r.reason, TerminationReason::ProvenOptimal);
    // Internally -21; user-facing bound must be +21.
    assert!((r.state.stats.best_bound.unwrap() - 21.0).abs() < 1e-6);
}

#[test]
fn mip_gap_stops_early_with_consistent_bound() {
    let mut options = SolveOptions::default();
    options.mip_gap = 0.5;
    let r = run(&binary_knapsack(), options).unwrap();
    assert_eq!(r.reason, TerminationReason::MipGap);
    let inc = -incumbent_obj(&r.state); // user space (Maximize)
    let bound = r.state.stats.best_bound.unwrap();
    // Incumbent within 50% of the proven bound, and never better than it.
    assert!(inc <= bound + 1e-9);
    assert!((bound - inc) / bound.abs().max(1e-10) <= 0.5 + 1e-9);
}

#[test]
fn feasible_interrupt_reports_gap() {
    let mut options = SolveOptions::default();
    options.node_limit = Some(2);
    let mut r = run(&binary_knapsack(), options).unwrap();
    // Resume with node budget until an incumbent exists but the search isn't done.
    let mut guard = 0;
    while r.reason == TerminationReason::NodeLimit && r.state.incumbent.is_none() {
        guard += 1;
        assert!(guard < 10_000);
        r.reason = resume_run(
            &mut r.state,
            ResumeOptions {
                node_limit: Some(2),
                ..ResumeOptions::default()
            },
        )
        .unwrap();
    }
    if r.reason == TerminationReason::NodeLimit {
        // Feasible-but-unproven: a gap must be reported.
        assert!(r.state.stats.gap.unwrap() >= 0.0);
        assert!(r.state.stats.best_bound.is_some());
    }
}

#[test]
fn plunge_and_jump_selection_preserves_optima() {
    // Same optima as plain DFS on all driver test problems, plus interrupt/resume.
    let r = run(&int_2var_problem(), SolveOptions::default()).unwrap();
    assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);

    let r = run(&binary_knapsack(), SolveOptions::default()).unwrap();
    assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);

    let mut options = SolveOptions::default();
    options.node_limit = Some(1);
    let mut r = run(&binary_knapsack(), options).unwrap();
    let mut guard = 0;
    while r.reason != TerminationReason::ProvenOptimal {
        guard += 1;
        assert!(guard < 10_000);
        r.reason = resume_run(&mut r.state, ResumeOptions::default()).unwrap();
    }
    assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
    // After a best-bound jump the pop is NOT the last-pushed node at least once
    // on this instance; correctness above is the real assertion.
}

#[test]
fn tolerances_default_matches_documented_values() {
    let t = Tolerances::default();
    assert_eq!(
        t.feasibility, 1e-7,
        "see Tolerances::feasibility's doc default"
    );
    assert_eq!(
        t.integrality_rounding, 1e-5,
        "see Tolerances::integrality_rounding's doc default"
    );
    assert_eq!(
        t.prune_epsilon, 1e-9,
        "see Tolerances::prune_epsilon's doc default"
    );
}

#[test]
fn try_adopt_incumbent_respects_custom_feasibility_tolerance() {
    // Derived from the big-M fixture `tests_general::solve_big_m` (same
    // m = 1e9 shape: `x - m*b == 10`, minimize x). Pin b to a value that
    // is integral-within-`int_tol` (5e-7, well inside the default 1e-6)
    // but not exactly 0; the rounded-incumbent guard then re-checks the
    // ROUNDED point (b -> 0) against the ORIGINAL row, which is off by
    // exactly m * 5e-7 = 500 — precisely the "big-M trap"
    // `tolerances.feasibility` exists to catch, in absolute terms. The guard
    // applies the tolerance the search was built with (the engine and the
    // guard share one contract), so each tolerance gets its own search.
    let m = 1.0e9;
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, f64::INFINITY));
    let b = p.add_binary_var(0.0);
    p.add_constraint(&[(x, 1.0), (b, -m)], ComparisonOp::Eq, 10.0);

    let pin_and_adopt = |feasibility: f64| -> (f64, bool) {
        let mut options = SolveOptions::default();
        options.tolerances.feasibility = feasibility;
        let mut solved = run(&p, options).unwrap();
        let state = &mut solved.state;
        // Force the relaxation to a specific near-zero fractional b: pin its
        // bounds to [5e-7, 5e-7] and re-solve. The equality row then forces
        // x to 10 + m*5e-7 = 510, deterministically.
        state.solver.set_var_bounds(b.idx(), 5e-7, 5e-7).unwrap();
        assert_eq!(
            state.solver.reoptimize().unwrap(),
            crate::StopReason::Finished
        );
        let x_value = state.solver.get_value(x.idx());
        (x_value, try_adopt_incumbent(state).unwrap())
    };

    // Default tolerance (1e-7): the rounded point (x=510, b=0) misses the
    // original `x - m*b == 10` row by 500 — must be rejected.
    let (x_value, adopted) = pin_and_adopt(Tolerances::default().feasibility);
    assert!((x_value - 510.0).abs() < 1e-6);
    assert!(
        !adopted,
        "a 500-unit rounding-induced violation must be rejected at the default feasibility tolerance"
    );

    // Absurdly loosened tolerance: the same 500-unit violation is now
    // within bounds — the guard must accept it.
    let (_, adopted) = pin_and_adopt(1e6);
    assert!(
        adopted,
        "the same violation must be accepted once tolerances.feasibility is loosened past it"
    );
}

#[test]
fn incumbent_feasible_row_tolerance_is_absolute_not_relative_to_rhs() {
    // `incumbent_feasible` uses the same absolute feasibility tolerance
    // as the rounded-incumbent guard, regardless of row magnitude.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, f64::INFINITY));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1000.0);
    let fixed = std::collections::BTreeMap::new();
    let tolerances = Tolerances::default();

    // Within the absolute tolerance (5e-8 < 1e-7): accepted.
    assert!(incumbent_feasible(
        &p,
        &fixed,
        &[1000.0 + 5e-8],
        &tolerances
    ));

    // A `5e-5` violation exceeds the `1e-7` absolute tolerance even though
    // it is small relative to this row's right-hand side.
    assert!(!incumbent_feasible(
        &p,
        &fixed,
        &[1000.0 + 5e-5],
        &tolerances
    ));
}

#[test]
fn candidate_validation_rejects_non_finite_and_malformed_values() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    problem.add_integer_var(1.0, (0, 10));
    let fixed = BTreeMap::new();
    let tolerances = Tolerances::default();

    assert!(!incumbent_feasible(&problem, &fixed, &[], &tolerances));
    assert!(!incumbent_feasible(
        &problem,
        &fixed,
        &[f64::NAN],
        &tolerances
    ));
    assert!(!incumbent_feasible(
        &problem,
        &fixed,
        &[f64::INFINITY],
        &tolerances
    ));

    let mut overflowing_row = Problem::new(OptimizationDirection::Minimize);
    let x = overflowing_row.add_var(0.0, (0.0, f64::INFINITY));
    overflowing_row.add_constraint(&[(x, 1.0e308)], ComparisonOp::Le, 1.0e308);
    assert!(!incumbent_feasible(
        &overflowing_row,
        &fixed,
        &[1.0e308],
        &tolerances
    ));
}

#[test]
fn valid_candidate_completes_unbounded_classification() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    problem.add_integer_var(1.0, (0, 10));
    let mut state = build_state(&problem, SolveOptions::default()).unwrap();
    assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
    state.classifying_unbounded = true;

    assert_eq!(try_adopt_incumbent(&mut state), Err(Error::Unbounded));
}

#[test]
fn warm_start_restores_bounds_before_unbounded_verdict() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_integer_var(0.0, (0, 10));
    // The hint is injected below the options plumbing that disables
    // dual fixing for a warm start; presolve would otherwise pin the
    // cost-free, row-free x at 0 and the hint would fail its bounds.
    let options = SolveOptions {
        presolve: false,
        ..SolveOptions::default()
    };
    let mut state = build_state(&problem, options).unwrap();
    assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
    state.classifying_unbounded = true;

    assert_eq!(
        try_warm_start(&mut state, &[(x, 5.0)]),
        Err(Error::Unbounded)
    );
    assert_eq!(state.solver.get_var_bounds(x.idx()), (0.0, 10.0));
}
