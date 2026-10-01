use super::search::{try_adopt_incumbent, try_warm_start};
use super::state::Incumbent;
use super::*;
use crate::{ComparisonOp, OptimizationDirection, Problem, StopReason, Variable};
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
    // m = 1e9 shape: `x - m*b == 10`, minimize x). The candidate is the
    // engine's point as computed: pinned at b = 5e-7, integral within
    // `int_tol` but not exactly 0, and x = 510, it meets the row exactly and
    // is adopted unrounded (rounded, b = 0 would miss the row by 500). The
    // guard applies the tolerance the search was built with (the engine and
    // the guard share one contract): with x pinned at 510.5, the row is off
    // by 0.5, which 1e6 accepts and the default tolerance refuses, in the
    // engine, which holds the row. The next test has the guard refuse a row
    // the engine does not hold.
    let m = 1.0e9;
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, f64::INFINITY));
    let b = p.add_binary_var(0.0);
    p.add_constraint(&[(x, 1.0), (b, -m)], ComparisonOp::Eq, 10.0);

    let pin_and_adopt = |feasibility: f64, x_pin: f64| -> Result<Option<Vec<f64>>, Error> {
        let mut options = SolveOptions::default();
        options.tolerances.feasibility = feasibility;
        let mut solved = run(&p, options).unwrap();
        let state = &mut solved.state;
        state.solver.set_var_bounds(b.idx(), 5e-7, 5e-7).unwrap();
        state.solver.set_var_bounds(x.idx(), x_pin, x_pin).unwrap();
        assert_eq!(state.solver.reoptimize()?, crate::StopReason::Finished);
        state.incumbent = None;
        let adopted = try_adopt_incumbent(state).unwrap();
        Ok(adopted.then(|| state.incumbent.as_ref().unwrap().values.clone()))
    };

    let default = Tolerances::default().feasibility;
    assert_eq!(
        pin_and_adopt(default, 510.0),
        Ok(Some(vec![510.0, 5e-7])),
        "the point is adopted as computed"
    );
    assert_eq!(
        pin_and_adopt(default, 510.5),
        Err(Error::Infeasible),
        "a 0.5 violation is beyond the default tolerance"
    );
    assert_eq!(
        pin_and_adopt(1e6, 510.5),
        Ok(Some(vec![510.5, 5e-7])),
        "the same violation must be accepted once tolerances.feasibility is loosened past it"
    );
}

#[test]
fn try_adopt_incumbent_holds_a_rewritten_row_to_the_custom_feasibility_tolerance() {
    // Presolve rewrites `x - 1000 b <= 0` over `x ∈ [0, 10]` to about
    // `x - 10 b <= 0`, which the engine holds, and the guard checks the row
    // as written. Pinned at b = -5e-9, integral within `int_tol`, and x = 0,
    // the point meets the rewritten row within 5e-8 and misses the row as
    // written by 5e-6, which the default tolerance refuses and 1e-5 accepts.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(-1.0, (0.0, 10.0));
    let b = p.add_binary_var(1.0);
    p.add_constraint(&[(x, 1.0), (b, -1000.0)], ComparisonOp::Le, 0.0);

    let pin_and_adopt = |feasibility: f64| -> bool {
        let mut options = SolveOptions::default();
        options.tolerances.feasibility = feasibility;
        let mut solved = run(&p, options).unwrap();
        let state = &mut solved.state;
        state.solver.set_var_bounds(b.idx(), -5e-9, -5e-9).unwrap();
        state.solver.set_var_bounds(x.idx(), 0.0, 0.0).unwrap();
        assert_eq!(
            state.solver.reoptimize(),
            Ok(StopReason::Finished),
            "the engine holds the rewritten row"
        );
        state.incumbent = None;
        try_adopt_incumbent(state).unwrap()
    };

    assert!(
        !pin_and_adopt(Tolerances::default().feasibility),
        "a 5e-6 violation of the row as written is beyond the default tolerance"
    );
    assert!(
        pin_and_adopt(1e-5),
        "the same violation must be accepted once tolerances.feasibility is loosened past it"
    );
}

#[test]
fn incumbent_feasible_row_tolerance_is_absolute_not_relative_to_rhs() {
    // `incumbent_feasible` uses the same absolute feasibility tolerance
    // as the incumbent guard, regardless of row magnitude.
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

/// The prefilter judges the bounds the solve works with, where a bound from
/// 2^52 on is infinite, so the answer `x = 1e16` beyond `x <= 2^53` (the row
/// alone stops `x`) passes it and survives as a warm-start hint after an edit.
#[test]
fn incumbent_feasible_takes_a_bound_from_2_pow_52_as_infinite() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_var(1.0, (0.0, 2.0f64.powi(53)));
    let n = p.add_integer_var(0.0, (0, 1));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1e16);
    let fixed = BTreeMap::new();
    let tolerances = Tolerances::default();
    let solution = p.solve().unwrap().into_solution().unwrap();
    assert_eq!(solution[x], 1e16);
    assert!(incumbent_feasible(
        &p,
        &fixed,
        &[solution[x], solution[n]],
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

/// `min -x - y - t - z - w` s.t. `x + y + z + q + 3w >= 3` and
/// `t - (1 + 2^-50) z - q >= 1e-6` over `x, y, t ∈ [0, 1]`, `z ∈ [-2e9, 1]`,
/// `q ∈ [-1e10, 1e10]` and a binary `w`: the optimum is `-5`, at `w = 1`. With
/// `w = 0` the LP is the one of
/// `a_round_off_coefficient_on_a_far_bound_is_no_proof_and_is_a_known_limitation`
/// in `src/tests/magnitudes.rs`, on which the engine fails loudly.
fn round_off_room_unless_w_is_one() -> (Problem, Variable) {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(-1.0, (0.0, 1.0));
    let y = p.add_var(-1.0, (0.0, 1.0));
    let t = p.add_var(-1.0, (0.0, 1.0));
    let z = p.add_var(-1.0, (-2e9, 1.0));
    let q = p.add_var(0.0, (-1e10, 1e10));
    let w = p.add_binary_var(-1.0);
    p.add_constraint(
        [(x, 1.0), (y, 1.0), (z, 1.0), (q, 1.0), (w, 3.0)],
        ComparisonOp::Ge,
        3.0,
    );
    p.add_constraint(
        [(t, 1.0), (z, -(1.0 + 2f64.powi(-50))), (q, -1.0)],
        ComparisonOp::Ge,
        1e-6,
    );
    (p, w)
}

/// A hint whose LP completion fails with an engine error is dropped, as an
/// infeasible one is: the solve returns what it returns without the hint.
#[test]
fn warm_start_drops_a_hint_whose_completion_fails() {
    let (problem, w) = round_off_room_unless_w_is_one();
    for presolve in [true, false] {
        let options = SolveOptions {
            presolve,
            ..SolveOptions::default()
        };
        let hinted = SolveOptions {
            warm_start: Some(vec![(w, 0.0)]),
            ..options.clone()
        };

        // The completion, the root LP re-solved with `w = 0`, fails loudly.
        let mut state = build_state(&problem, hinted.clone()).unwrap();
        assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
        state.solver.set_var_bounds(w.idx(), 0.0, 0.0).unwrap();
        assert!(matches!(
            state.solver.reoptimize(),
            Err(Error::InternalError(_))
        ));

        let without = problem.solve_with(options).unwrap();
        let with = match problem.solve_with(hinted) {
            Ok(outcome) => outcome,
            Err(err) => panic!("the hinted solve returned {err:?} (presolve {presolve})"),
        };
        let (with, without) = (with.solution().unwrap(), without.solution().unwrap());
        assert!((without.objective() + 5.0).abs() < 1e-6);
        assert_eq!(with.status(), SolutionStatus::Optimal);
        assert_eq!(with.objective(), without.objective());
        assert_eq!(
            with.iter().collect::<Vec<_>>(),
            without.iter().collect::<Vec<_>>()
        );
    }
}

/// The restore after a warm start re-solves the root basis, which the load
/// repairs when a fresh factorization finds it singular. `min -x + y + z` s.t.
/// `x + y <= 1`, `x + (1 + 2^-48) y >= 1`, `x >= 0`, `y` free and `z` binary:
/// the root LP ends at its optimum, `-1` at `(1, 0, 0)`, with the basis
/// `{x, y}` on the eta file. The restore's fresh factorization finds it
/// singular and repairs it (`y` is added first, so that `x` goes out) into
/// `(0, 1, 0)`, a feasible integral point of objective `1`. The hint `x = 0`
/// completes to about `1`, so a root node built without the re-solve would
/// close at that incumbent.
#[test]
fn warm_start_re_solves_a_root_basis_its_restore_repairs() {
    for presolve in [true, false] {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let y = problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY));
        let x = problem.add_var(-1.0, (0.0, f64::INFINITY));
        let z = problem.add_binary_var(1.0);
        problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 1.0);
        let slanted = 1.0 + 2f64.powi(-48);
        problem.add_constraint([(x, 1.0), (y, slanted)], ComparisonOp::Ge, 1.0);
        let options = SolveOptions {
            presolve,
            warm_start: Some(vec![(x, 0.0)]),
            ..SolveOptions::default()
        };
        let solution = match problem.solve_with(options) {
            Ok(outcome) => outcome.into_solution().unwrap(),
            Err(err) => panic!("presolve {presolve}: {err:?}"),
        };
        assert_eq!(solution.status(), SolutionStatus::Optimal);
        let (objective, x, y, z) = (solution.objective(), solution[x], solution[y], solution[z]);
        assert!(
            (objective + 1.0).abs() < 1e-9 && (x - 1.0).abs() < 1e-9 && y.abs() < 1e-9 && z == 0.0,
            "presolve {presolve}: objective {objective} at ({x}, {y}, {z})"
        );
    }
}
