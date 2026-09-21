//! `SolveOptions::warm_start`: a usable hint is adopted, and every kind of
//! unusable hint is ignored rather than trusted.

use super::common::{int_2var_problem, solution};
use crate::*;

#[test]
fn warm_start_with_optimal_hint_is_accepted() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
    let sol = solution(p.solve_with(options).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_with_infeasible_hint_is_ignored() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    // a=0, b=0 violates both constraints — hint must be dropped, solve still exact.
    options.warm_start = Some(vec![(a, 0.0), (b, 0.0)]);
    let sol = solution(p.solve_with(options).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_out_of_bounds_hint_is_ignored() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 99.0), (b, 2.0)]); // 99 > upper bound 10
    let sol = solution(p.solve_with(options).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_non_finite_hint_is_ignored() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (p, a, _) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.warm_start = Some(vec![(a, invalid)]);
        let sol = solution(p.solve_with(options).unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }
}

#[test]
fn warm_start_nan_for_nonbasic_variable_is_ignored() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_integer_var(1.0, (0, 10));
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(x, f64::NAN)]);

    let solution = solution(problem.solve_with(options).unwrap());

    assert_eq!(solution.status(), SolutionStatus::Optimal);
    assert_eq!(solution.var_value(x), 0.0);
    assert_eq!(solution.objective(), 0.0);
}

#[test]
fn warm_start_partial_hint_completes_via_lp() {
    // minimize 3a + 4b, a + 2b >= 5, 3a + b >= 4; hint only a=1 → LP completes b,
    // and if the completion is integral (b=2) it seeds the incumbent.
    let (p, a, _) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 1.0)]);
    let sol = solution(p.solve_with(options).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn warm_start_liveness_hint_seeds_incumbent_before_any_node() {
    // node_limit = 0: the search loop exits before solving a single node, so
    // the ONLY way to have an incumbent is the warm-start hint. This test
    // fails if the hint wiring is ever disconnected.
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.node_limit = Some(0);
    let cold = p.solve_with(options.clone()).unwrap();
    assert!(cold.solution().is_none());
    assert_eq!(cold.termination_reason(), TerminationReason::NodeLimit);

    let mut options = SolveOptions::default();
    options.node_limit = Some(0);
    options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
    let hinted = solution(p.solve_with(options).unwrap());
    assert_eq!(hinted.status(), SolutionStatus::Feasible);
    assert_eq!(hinted.termination_reason(), TerminationReason::NodeLimit);
    assert!((hinted.objective() - 11.0).abs() < 1e-6);
    assert_eq!(hinted.var_value(a), 1.0);
    assert_eq!(hinted.var_value(b), 2.0);
    assert_eq!(hinted.stats().nodes_solved, 0);
}

#[test]
fn warm_start_prunes_immediately_when_hint_is_optimal() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
    let with_hint = solution(p.solve_with(options).unwrap());
    let without = solution(p.solve().unwrap());
    // Correctness identical; the hinted run must not explore MORE nodes.
    assert!(with_hint.stats().nodes_solved <= without.stats().nodes_solved);
    assert_eq!(with_hint.objective(), without.objective());
}
