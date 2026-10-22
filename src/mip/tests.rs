use super::search::{solve_node_lp, try_adopt_incumbent, try_warm_start};
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
    // int_2var_problem is proven optimal in exactly 1 B&B node (deterministic:
    // the unlimited solve reports nodes_solved == 1; node_limit=0 -> Interrupted,
    // node_limit=1 -> Optimal): the first child finds the optimum 11, and the
    // second, whose stored bound 10.6 cannot reach a better whole value, is
    // pruned before any LP work. Setting the node limit to that exact count
    // must report Optimal, not Interrupted: the free prune and then the
    // empty-`open` check at the loop top come before the limit, and the
    // finished proof is reported honestly.
    assert_eq!(
        run(&int_2var_problem(), SolveOptions::default())
            .unwrap()
            .state
            .stats
            .nodes_solved,
        1,
        "node count must stay deterministic; update the limit below if it changes"
    );
    let mut options = SolveOptions::default();
    options.node_limit = Some(1);
    let r = run(&int_2var_problem(), options).unwrap();
    assert_eq!(r.reason, TerminationReason::ProvenOptimal);
    assert!((incumbent_obj(&r.state) - 11.0).abs() < 1e-6);
}

#[test]
fn driver_node_limit_interrupts_and_resumes_to_same_optimum() {
    let mut options = SolveOptions::default();
    options.node_limit = Some(1);
    let mut r = run(&binary_knapsack(), options).unwrap();
    let mut guard = 0;
    while r.reason != TerminationReason::ProvenOptimal {
        guard += 1;
        assert!(guard < 10_000, "resume loop did not terminate");
        r.reason = resume_run(&mut r.state, ResumeOptions::default()).unwrap();
    }
    assert!(guard >= 1, "node_limit=1 should interrupt at least once");
    assert!((incumbent_obj(&r.state) + 21.0).abs() < 1e-6);
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

/// `max cost x + cost y` s.t. `2x + 2y <= 7`, `x - y <= 0.5`, `x` and `y`
/// integer over the `i32` range: the LP bound stays `3.5 cost` at every node
/// along a face the integer bounds leave open, and the optimum is `3 cost`.
fn deep_two_var_problem(cost: f64) -> Problem {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var(cost, (i32::MIN, i32::MAX));
    let y = p.add_integer_var(cost, (i32::MIN, i32::MAX));
    p.add_constraint(&[(x, 2.0), (y, 2.0)], ComparisonOp::Le, 7.0);
    p.add_constraint(&[(x, 1.0), (y, -1.0)], ComparisonOp::Le, 0.5);
    p
}

/// [`deep_two_var_problem`] at a cost of one with a continuous var in
/// `[0, 1]` of cost `-1`, which keeps the objective off any lattice: the
/// search finds the incumbent `3` within a few nodes and branches down the
/// face until its budget runs out.
fn deep_two_var_problem_off_any_lattice() -> Problem {
    let mut p = deep_two_var_problem(1.0);
    p.add_var(-1.0, (0.0, 1.0));
    p
}

#[test]
fn a_deep_search_stores_one_bound_change_per_var_in_each_node() {
    // The LP bound stays 3.5 against the incumbent 3 along a face the
    // integer bounds leave open, and nothing prunes the face off any
    // lattice: the search branches x and y again and again until its budget
    // runs out.
    let p = deep_two_var_problem_off_any_lattice();
    let mut options = SolveOptions::default();
    options.node_limit = Some(200);
    let run = run(&p, options).unwrap();

    assert_eq!(run.state.objective_lattice, None);
    assert_eq!(
        run.reason,
        TerminationReason::NodeLimit,
        "the search should stop at its node limit"
    );
    assert!(
        (incumbent_obj(&run.state) + 3.0).abs() < 1e-6,
        "the incumbent should be 3"
    );
    assert!(
        run.state.open.iter().any(|node| node.depth >= 100),
        "the search should branch each var again and again"
    );
    for node in &run.state.open {
        let changes = &node.bound_changes;
        assert!(
            changes.len() <= 2,
            "a node should hold at most one entry per var"
        );
        assert!(
            changes.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "a node's entries should be in var order, one per var"
        );
    }
}

/// Solves `p` under a node limit, warm-started with `warm_start`, with
/// presolve on and off as listed in `presolve`, expecting the search to prove
/// the optimum `expected` (user space) within `max_nodes` nodes.
fn assert_proves(
    p: &Problem,
    presolve: &[bool],
    warm_start: Option<Vec<(Variable, f64)>>,
    expected: f64,
    max_nodes: u64,
) {
    for &presolve in presolve {
        let options = SolveOptions {
            presolve,
            node_limit: Some(2000),
            warm_start: warm_start.clone(),
            ..SolveOptions::default()
        };
        let run = run(p, options).unwrap();
        let objective = to_user_space(p.direction, incumbent_obj(&run.state));
        let nodes = run.state.stats.nodes_solved;
        assert_eq!(
            run.reason,
            TerminationReason::ProvenOptimal,
            "presolve {presolve}: objective {objective} after {nodes} nodes"
        );
        assert!(
            (objective - expected).abs() < 1e-6,
            "presolve {presolve}: objective {objective}, expected {expected}"
        );
        assert!(
            nodes <= max_nodes,
            "presolve {presolve}: {nodes} nodes, expected at most {max_nodes}"
        );
    }
}

/// The deep model's objective takes whole values at its integral points, so
/// a node whose bound is `3.5` cannot beat the incumbent `3`: the search
/// proves it optimal instead of branching down the face.
#[test]
fn a_bound_short_of_the_next_whole_value_cannot_beat_the_incumbent() {
    assert_proves(&deep_two_var_problem(1.0), &[true, false], None, 3.0, 5);
}

/// Costs of `-2` and `-4` put the objective's values at integral points on
/// the even numbers, where the LP bound `-7` of every node along the face of
/// `2x + 4y <= 7` cannot beat the incumbent `-6`, which a step of one would
/// not tell; costs of `0.5` put them on the multiples of a half.
#[test]
fn the_lattice_step_is_the_greatest_common_divisor_of_the_costs() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(-2.0, (i32::MIN, i32::MAX));
    let y = p.add_integer_var(-4.0, (i32::MIN, i32::MAX));
    p.add_constraint(&[(x, 2.0), (y, 4.0)], ComparisonOp::Le, 7.0);
    p.add_constraint(&[(x, 1.0), (y, -1.0)], ComparisonOp::Le, 0.5);
    assert_proves(&p, &[true, false], None, -6.0, 5);

    assert_proves(&deep_two_var_problem(0.5), &[true, false], None, 1.5, 5);
}

/// A continuous var fixed at `0.3` with a cost of `0.7` adds `0.21` to every
/// point's objective: the lattice is offset by it.
#[test]
fn a_fixed_continuous_var_with_a_cost_offsets_the_lattice() {
    let mut p = deep_two_var_problem(1.0);
    p.add_var(0.7, (0.3, 0.3));
    assert_proves(&p, &[true, false], None, 3.21, 5);
}

/// The lattice is the one of the bounds the search starts from: an integer
/// var `z` in `[0, 3]` with a cost of `0.5` and the row `z >= 3`, which
/// presolve turns into a fixing, adds `1.5` to the deep model's values, a
/// whole step apart, where no node beats the incumbent `4.5`. Without
/// presolve the values are a half apart, the LP bound `5` is one of them,
/// and the search branches down the face until its budget runs out.
#[test]
fn a_var_presolve_fixes_joins_the_offset_of_the_lattice() {
    let mut p = deep_two_var_problem(1.0);
    let z = p.add_integer_var(0.5, (0, 3));
    p.add_constraint(&[(z, 1.0)], ComparisonOp::Ge, 3.0);
    let options = |presolve| SolveOptions {
        presolve,
        node_limit: Some(200),
        ..SolveOptions::default()
    };

    let presolved = run(&p, options(true)).unwrap();
    assert_eq!(presolved.reason, TerminationReason::ProvenOptimal);
    assert!((incumbent_obj(&presolved.state) + 4.5).abs() < 1e-6);

    let as_written = run(&p, options(false)).unwrap();
    assert_eq!(as_written.reason, TerminationReason::NodeLimit);
    assert!((incumbent_obj(&as_written.state) + 4.5).abs() < 1e-6);
}

/// Nodes whose LP vertices have reduced costs within their tolerance on their
/// wrong side, over vars whose ranges span the `i32` range or `1e9`: an
/// exactly feasible point of the node can lie a step below its LP objective,
/// and rounding that objective up pruned the optimum. `min 3a - 3b - 2c`
/// s.t. `-2a + 2b + c = -17` and a row of the same vars whose coefficients
/// are `3` perturbed by `1e-11` to `2e-10` has its optimum `20` at
/// `(0, -14, 11)`, and the search proved `21`; `max 2x - y` s.t.
/// `-2x + (1 + 6e-11) y >= -2.99` has `3` at `(500000001, 999999999)`, and
/// it proved `2`; `max -2x - 2y + 3z` s.t.
/// `(1 + 4e-12) x - (2 + 8e-11) y + 3z <= -7.01` has `-7` at
/// `(0, 1500000002, 999999999)`, and it proved `-8` with presolve.
#[test]
fn the_dual_tolerance_over_a_wide_range_does_not_prune_the_optimum() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let a = p.add_integer_var(3.0, (0, 1_000_000_000));
    let b = p.add_integer_var(-3.0, (i32::MIN, i32::MAX));
    let c = p.add_integer_var(-2.0, (0, 1_000_000_000));
    p.add_constraint(&[(a, -2.0), (b, 2.0), (c, 1.0)], ComparisonOp::Eq, -17.0);
    p.add_constraint(
        &[
            (a, 3.0000000000240004),
            (b, -3.00000000018),
            (c, -3.000000000009),
        ],
        ComparisonOp::Ge,
        8.99,
    );
    assert_proves(&p, &[true, false], None, 20.0, 20);

    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var(2.0, (i32::MIN, i32::MAX));
    let y = p.add_integer_var(-1.0, (0, 1_000_000_000));
    p.add_constraint(&[(x, -2.0), (y, 1.00000000006)], ComparisonOp::Ge, -2.99);
    assert_proves(&p, &[true, false], None, 3.0, 20);

    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var(-2.0, (0, i32::MAX));
    let y = p.add_integer_var(-2.0, (i32::MIN, i32::MAX));
    let z = p.add_integer_var(3.0, (0, 1_000_000_000));
    p.add_constraint(
        &[(x, 1.000000000004), (y, -2.00000000008), (z, 3.0)],
        ComparisonOp::Le,
        -7.01,
    );
    assert_proves(&p, &[true], None, -7.0, 20);
}

/// `min x + 2^-1000 y` s.t. `x >= 20000000.5`, `x` in `[0, 1e8]` and `y` in
/// `[0, 10]` integer, warm-started at `x = 3e7`: the costs' step, `2^-1000`,
/// put the bound of every node more steps from the lattice's origin than
/// `f64` holds, rounding it up gave infinity, and every node was pruned by
/// the hint, proved optimal. A step within the margin gives no lattice.
#[test]
fn a_cost_tiny_beside_the_others_leaves_the_bounds_finite() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(1.0, (0, 100_000_000));
    let y = p.add_integer_var(2f64.powi(-1000), (0, 10));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Ge, 20000000.5);
    let hint = vec![(x, 3e7), (y, 0.0)];
    assert_proves(&p, &[true, false], Some(hint), 20000001.0, 5);
}

/// The lattice prunes a node only where none of its integral points would
/// beat the incumbent by more than the pruning epsilon: with the incumbent
/// `11 + 4e-7` above its lattice value (an integer var within `int_tol` of
/// its integer, say), a node whose bound is below `11` could hold a point at
/// `11`, which beats it; at `11 - 4e-7` it could not. A bound at the
/// incumbent prunes whatever its integral bound.
#[test]
fn an_incumbent_off_its_lattice_value_prunes_only_what_cannot_beat_it() {
    let mut state = build_state(&int_2var_problem(), SolveOptions::default()).unwrap();
    assert!(state.objective_lattice.is_some());
    let mut with_incumbent = |objective: f64| {
        state.incumbent = Some(Incumbent {
            values: vec![1.0, 2.0],
            objective,
        });
        [9.5, 10.0, 10.5, 11.0, objective].map(|bound| dominated(&state, bound, bound))
    };
    assert_eq!(with_incumbent(11.0), [false, false, true, true, true]);
    assert_eq!(
        with_incumbent(11.0 - 4e-7),
        [false, false, true, true, true]
    );
    assert_eq!(
        with_incumbent(11.0 + 4e-7),
        [false, false, false, false, true]
    );
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
    // The deep model off any lattice finds its incumbent within a few nodes
    // and never proves it.
    let mut options = SolveOptions::default();
    options.node_limit = Some(2);
    let mut r = run(&deep_two_var_problem_off_any_lattice(), options).unwrap();
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
    // Feasible-but-unproven: a gap must be reported.
    assert_eq!(r.reason, TerminationReason::NodeLimit);
    assert!(r.state.stats.gap.unwrap() > 0.0);
    assert!(r.state.stats.best_bound.is_some());
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
    assert_eq!(
        r.reason,
        TerminationReason::NodeLimit,
        "fixture: the node limit interrupts the search"
    );
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
        &tolerances,
        SolveOptions::default().int_tol
    ));

    // A `5e-5` violation exceeds the `1e-7` absolute tolerance even though
    // it is small relative to this row's right-hand side.
    assert!(!incumbent_feasible(
        &p,
        &fixed,
        &[1000.0 + 5e-5],
        &tolerances,
        SolveOptions::default().int_tol
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
        &tolerances,
        SolveOptions::default().int_tol
    ));
}

#[test]
fn candidate_validation_rejects_non_finite_and_malformed_values() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    problem.add_integer_var(1.0, (0, 10));
    let fixed = BTreeMap::new();
    let tolerances = Tolerances::default();

    assert!(!incumbent_feasible(
        &problem,
        &fixed,
        &[],
        &tolerances,
        SolveOptions::default().int_tol
    ));
    assert!(!incumbent_feasible(
        &problem,
        &fixed,
        &[f64::NAN],
        &tolerances,
        SolveOptions::default().int_tol
    ));
    assert!(!incumbent_feasible(
        &problem,
        &fixed,
        &[f64::INFINITY],
        &tolerances,
        SolveOptions::default().int_tol
    ));

    let mut overflowing_row = Problem::new(OptimizationDirection::Minimize);
    let x = overflowing_row.add_var(0.0, (0.0, f64::INFINITY));
    overflowing_row.add_constraint(&[(x, 1.0e308)], ComparisonOp::Le, 1.0e308);
    assert!(!incumbent_feasible(
        &overflowing_row,
        &fixed,
        &[1.0e308],
        &tolerances,
        SolveOptions::default().int_tol
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

/// A switchable network like issue 48's (8 buses, 10 lines, at most one
/// off; optimum `29.4617904`, with line 9 off) whose LP with the switch of
/// line 8 open, `on[8] = 0`, is infeasible: with at most one line off, every
/// other switch is then closed, and that network cannot carry its demand.
fn network_with_line_8_off_infeasible() -> (Problem, Variable) {
    let lines = [
        (1, 0, 0.132, 0.724),
        (2, 0, 206.0, 0.763),
        (3, 2, 51.900000000000006, 0.993),
        (4, 2, 1.71, 1.1400000000000001),
        (5, 3, 0.0407, 0.801),
        (6, 4, 7.0600000000000005, 1.12),
        (7, 3, 0.0335, 0.721),
        (4, 5, 939.0, 0.974),
        (4, 6, 1.76, 1.04),
        (3, 6, 6.33, 0.404),
    ];
    let demand = [
        0.905,
        0.742,
        0.659,
        0.294,
        0.343,
        0.528,
        0.913,
        0.8130000000000001,
    ];
    let gen_max = [
        0.263,
        1.32,
        1.26,
        1.1400000000000001,
        0.335,
        0.581,
        0.212,
        1.28,
    ];
    let gen_cost = [9.67, 4.16, 3.48, 8.0, 9.61, 3.25, 9.97, 6.21];
    let problem = crate::tests::regressions::regression_tests::switchable_dc_network(
        &lines, &demand, &gen_max, &gen_cost, 1,
    );
    // The vars: 8 bus angles, 8 generators, 10 flows, then the 10 switches.
    (problem, Variable(8 + 8 + 10 + 8))
}

/// The warm-start completion of [`network_with_line_8_off_infeasible`] with
/// line 8 off, the root LP re-solved with `on[8] = 0`, is proven infeasible,
/// with presolve on and off (on: the root LP of the presolved model). Along
/// the way the only repair of a row is a slack unbounded in its direction,
/// through an entry of `1.2e-15` that the plain row computes as `7.1e-15`,
/// and that coefficient leaves the row's certificate unbounded: the first
/// last resort pivots on it, genuine on the refined row, and a later row
/// proves the LP infeasible. Taken for round-off because the plain row was
/// off, the entry left no pivot and no proof, and the solve ended in "no var
/// can repair row".
#[test]
fn a_network_completion_with_a_line_off_is_proven_infeasible() {
    let (problem, on_8) = network_with_line_8_off_infeasible();
    for presolve in [true, false] {
        let hinted = SolveOptions {
            presolve,
            warm_start: Some(vec![(on_8, 0.0)]),
            ..SolveOptions::default()
        };
        let mut state = build_state(&problem, hinted.clone()).unwrap();
        assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
        state.solver.set_var_bounds(on_8.idx(), 0.0, 0.0).unwrap();
        assert_eq!(
            state.solver.reoptimize(),
            Err(Error::Infeasible),
            "presolve {presolve}"
        );

        let solution = problem.solve_with(hinted).unwrap().into_solution().unwrap();
        assert!((solution.objective() - 29.4617904).abs() < 1e-6);
    }
}

/// The completion of [`network_with_line_8_off_infeasible`] with line 2's
/// two big-M rows written `2^-8` times smaller, the same problem: re-solved
/// from the warm start, the dual phase ends at a row no var can repair and
/// that proves nothing, and the solve starts over from the slack basis,
/// which proves the node infeasible, with presolve on and off.
#[test]
fn a_network_completion_ending_at_an_unrepaired_row_starts_over_from_the_slack_basis() {
    let (mut problem, on_8) = network_with_line_8_off_infeasible();
    for row in [12, 13] {
        let (coeffs, _, rhs) = &mut problem.constraints[row];
        coeffs.map_inplace(|c| c / 256.0);
        *rhs /= 256.0;
    }
    for presolve in [true, false] {
        let hinted = SolveOptions {
            presolve,
            warm_start: Some(vec![(on_8, 0.0)]),
            ..SolveOptions::default()
        };
        let mut state = build_state(&problem, hinted.clone()).unwrap();
        assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
        state.solver.set_var_bounds(on_8.idx(), 0.0, 0.0).unwrap();
        assert_eq!(
            state.solver.reoptimize(),
            Err(Error::Infeasible),
            "presolve {presolve}"
        );
        let solution = problem.solve_with(hinted).unwrap().into_solution().unwrap();
        assert!((solution.objective() - 29.4617904).abs() < 1e-6);
    }
}

/// `min 0.4 x0 + 0.4 x1 - 100 z` s.t. three nearly dependent rows, the first
/// with `+ s`, and `-z <= s <= z`, `x0 >= 0`, `x1` free, `s` in `[-1, 1]`,
/// `z` binary: the optimum is `-100 - 0.4 * 19.25 / 3.95` at `x0 = 0`,
/// `z = 1`, and with `z = 0`, which holds `s` at zero, the LP is the pinned
/// model of
/// `an_infeasible_lp_whose_dependency_has_no_boxed_var_is_a_known_limitation`
/// in `src/tests/magnitudes.rs`, whose solve fails loudly. Returns the
/// problem and `z`.
fn milp_whose_lp_with_z_off_fails() -> (Problem, Variable) {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x0 = problem.add_var(0.4, (0.0, f64::INFINITY));
    let x1 = problem.add_var(0.4, (f64::NEG_INFINITY, f64::INFINITY));
    let s = problem.add_var(0.0, (-1.0, 1.0));
    let z = problem.add_binary_var(-100.0);
    problem.add_constraint(
        [(x0, 70.9800000000001), (x1, -35.945), (s, 1.0)],
        ComparisonOp::Eq,
        175.17600000000053,
    );
    problem.add_constraint([(x0, -7.8), (x1, 3.95)], ComparisonOp::Eq, -19.25);
    problem.add_constraint(
        [(x0, 17.16000000372529), (x1, -8.690000000000001)],
        ComparisonOp::Le,
        42.549000018626444,
    );
    problem.add_constraint([(s, 1.0), (z, -1.0)], ComparisonOp::Le, 0.0);
    problem.add_constraint([(s, 1.0), (z, 1.0)], ComparisonOp::Ge, 0.0);
    (problem, z)
}

/// A hint whose LP completion fails with an engine error is dropped, as an
/// infeasible one is: the solve returns what it returns without the hint.
#[test]
fn warm_start_drops_a_hint_whose_completion_fails() {
    let (problem, z) = milp_whose_lp_with_z_off_fails();
    for presolve in [true, false] {
        let options = SolveOptions {
            presolve,
            ..SolveOptions::default()
        };
        let hinted = SolveOptions {
            warm_start: Some(vec![(z, 0.0)]),
            ..options.clone()
        };

        // The completion, the root LP re-solved with `z = 0`, fails loudly.
        let mut state = build_state(&problem, hinted.clone()).unwrap();
        assert_eq!(state.solver.initial_solve().unwrap(), StopReason::Finished);
        state.solver.set_var_bounds(z.idx(), 0.0, 0.0).unwrap();
        assert!(
            matches!(state.solver.reoptimize(), Err(Error::InternalError(_))),
            "fixture: the completion fails loudly (presolve {presolve})"
        );

        let without = problem.solve_with(options).unwrap();
        let with = match problem.solve_with(hinted) {
            Ok(outcome) => outcome,
            Err(err) => panic!("the hinted solve returned {err:?} (presolve {presolve})"),
        };
        let (with, without) = (with.solution().unwrap(), without.solution().unwrap());
        assert!((without.objective() + 100.0 + 0.4 * 19.25 / 3.95).abs() < 1e-6);
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

/// An edit keeps an incumbent a row every point satisfies cannot exclude:
/// `x <= +inf` holds at `x = 5`, so the post-edit prefilter accepts the point
/// and the edit's search starts from it.
#[test]
fn incumbent_feasible_is_not_excluded_by_a_row_with_an_infinite_right_hand_side() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var(1.0, (0, 10));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 5.0);
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Le, f64::INFINITY);
    let fixed = BTreeMap::new();
    assert!(incumbent_feasible(
        &p,
        &fixed,
        &[5.0],
        &Tolerances::default(),
        SolveOptions::default().int_tol
    ));
}

/// The post-edit prefilter decides integrality with `int_tol`, as the search
/// does: `x = 2.00005` is integral at `int_tol = 1e-4`, not at the default.
#[test]
fn incumbent_feasible_decides_integrality_with_int_tol() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(1.0, (0, 10));
    let y = p.add_var(0.0, (0.0, 10.0));
    p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 2.00005);
    let fixed = BTreeMap::new();
    let tolerances = Tolerances::default();
    assert!(incumbent_feasible(
        &p,
        &fixed,
        &[2.00005, 0.0],
        &tolerances,
        1e-4
    ));
    assert!(!incumbent_feasible(
        &p,
        &fixed,
        &[2.00005, 0.0],
        &tolerances,
        1e-6
    ));
}

/// A node LP the engine answers `Unbounded`, which the search's nodes do only
/// numerically, is solved once more from the slack basis before it is
/// reported, as an internal error. This LP, unbounded from every basis,
/// stands in for a node whose ray one basis takes past an entry the
/// refinement cannot resolve: the retry runs, then the error.
#[test]
fn an_unbounded_node_lp_is_solved_again_from_the_slack_basis() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_var(-1.0, (0.0, f64::INFINITY));
    let y = problem.add_var(-1.0, (0.0, f64::INFINITY));
    problem.add_constraint([(x, 1.0), (y, -1.0)], ComparisonOp::Le, 1.0);
    let options = SolveOptions {
        presolve: false,
        ..SolveOptions::default()
    };
    let once = {
        let mut state = build_state(&problem, options.clone()).unwrap();
        assert_eq!(state.solver.reoptimize(), Err(Error::Unbounded));
        state.solver.lp_iterations
    };
    let mut state = build_state(&problem, options).unwrap();
    assert!(matches!(
        solve_node_lp(&mut state),
        Err(Error::InternalError(ref message))
            if message.contains("bounded B&B node reported unbounded")
    ));
    assert!(
        state.solver.lp_iterations > once,
        "the node LP is solved again"
    );
}
