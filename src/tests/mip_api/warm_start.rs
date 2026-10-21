//! `SolveOptions::warm_start`: a usable hint is adopted, every kind of
//! unusable hint is ignored rather than trusted, and an invalid one (a value
//! that is not finite, a variable outside the problem) is refused.

use super::common::{binary_knapsack, int_2var_problem, solution};
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

/// A hint value that is NaN or infinite is invalid input, refused before
/// anything is solved, on a MILP and on an LP alike.
#[test]
fn warm_start_non_finite_hint_is_an_invalid_operation() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let (p, a, b) = int_2var_problem();
        let mut lp = Problem::new(OptimizationDirection::Minimize);
        let x = lp.add_var(1.0, (0.0, 10.0));
        for (problem, hint) in [(p, vec![(a, 1.0), (b, invalid)]), (lp, vec![(x, invalid)])] {
            let mut options = SolveOptions::default();
            options.warm_start = Some(hint.clone());
            let index = hint.last().unwrap().0.idx();
            assert_eq!(
                problem.solve_with(options).map(|_| "solved"),
                Err(Error::InvalidOperation(format!(
                    "invalid warm-start value {invalid} of variable {index}: expected a finite \
                     value"
                ))),
                "{hint:?}"
            );
        }
    }
}

#[test]
fn warm_start_nan_for_nonbasic_variable_is_an_invalid_operation() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_integer_var(1.0, (0, 10));
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(x, f64::NAN)]);

    assert_eq!(
        problem.solve_with(options).map(|_| "solved"),
        Err(Error::InvalidOperation(
            "invalid warm-start value NaN of variable 0: expected a finite value".to_string()
        )),
    );
}

/// A hint naming a variable of another problem, whose index this one does not
/// have, is invalid input too: refused before anything is solved, the
/// variable before its value.
#[test]
fn warm_start_variable_outside_the_problem_is_an_invalid_operation() {
    let mut other = Problem::new(OptimizationDirection::Minimize);
    let foreign: Vec<Variable> = (0..3).map(|_| other.add_var(1.0, (0.0, 1.0))).collect();
    let (p, a, _) = int_2var_problem();
    let mut lp = Problem::new(OptimizationDirection::Minimize);
    lp.add_var(1.0, (0.0, 10.0));
    for (problem, num_vars) in [(p, 2), (lp, 1)] {
        let mut options = SolveOptions::default();
        options.warm_start = Some(vec![(a, 1.0), (foreign[2], f64::NAN)]);
        assert_eq!(
            problem.solve_with(options).map(|_| "solved"),
            Err(Error::InvalidOperation(format!(
                "invalid warm-start hint: variable index 2 is out of range 0..{num_vars}"
            ))),
        );
    }
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

    // The objective takes whole values at integral points, and the root's
    // bound 10.6 cannot beat the hint's 11: both children are pruned before
    // any LP work, and the hint is proven optimal.
    let mut options = SolveOptions::default();
    options.node_limit = Some(0);
    options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
    let hinted = solution(p.solve_with(options).unwrap());
    assert_eq!(hinted.status(), SolutionStatus::Optimal);
    assert_eq!(
        hinted.termination_reason(),
        TerminationReason::ProvenOptimal
    );
    assert!((hinted.objective() - 11.0).abs() < 1e-6);
    assert_eq!(hinted.var_value(a), 1.0);
    assert_eq!(hinted.var_value(b), 2.0);
    assert_eq!(hinted.stats().nodes_solved, 0);
}

/// A hint the root cannot prove optimal seeds the incumbent of a search the
/// node limit stops before any node: the knapsack's hint `x = y = 1` is worth
/// `19`, and its root LP bound `22` leaves room for better.
#[test]
fn warm_start_hint_seeds_a_feasible_incumbent_before_any_node() {
    let p = binary_knapsack();
    let mut options = SolveOptions::default();
    options.node_limit = Some(0);
    options.warm_start = Some(vec![
        (Variable(0), 1.0),
        (Variable(1), 1.0),
        (Variable(2), 0.0),
        (Variable(3), 0.0),
    ]);
    let hinted = solution(p.solve_with(options).unwrap());
    assert_eq!(hinted.status(), SolutionStatus::Feasible);
    assert_eq!(hinted.termination_reason(), TerminationReason::NodeLimit);
    assert!((hinted.objective() - 19.0).abs() < 1e-6);
    assert_eq!(hinted.stats().nodes_solved, 0);
}

#[test]
fn warm_start_prunes_immediately_when_hint_is_optimal() {
    let (p, a, b) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
    let with_hint = solution(p.solve_with(options).unwrap());
    let without = solution(p.solve().unwrap());
    // Correctness identical, up to the integrality tolerance of the values
    // (costs 3 and 4); the hinted run must not explore MORE nodes.
    assert!(with_hint.stats().nodes_solved <= without.stats().nodes_solved);
    let int_tol = SolveOptions::default().int_tol;
    assert!((with_hint.objective() - without.objective()).abs() <= int_tol * 7.0);
}

#[test]
fn a_hint_completion_outside_the_models_bounds_is_not_adopted() {
    // The hint fixes x at -5e-10 and w at -1e-9, within the hint's slack
    // below their bounds. Its completion holds `x - 100.3 w - 0.5 z = 0` with
    // x basic at 100.3 w = -1.003e-7: within x's tolerance of the value the
    // hint fixed it at, but beyond the contract of its bound 0. Checked
    // against the bounds the hint set, that point was adopted and reported.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(0.0, (0.0, 1.0));
    let w = p.add_var(1.0, (0.0, 1.0));
    let z = p.add_integer_var(-1.0, (0, 1));
    let u = p.add_var(0.0, (0.0, 1.0));
    p.add_constraint(&[(x, 1.0), (w, -100.3), (z, -0.5)], ComparisonOp::Eq, 0.0);
    p.add_constraint(&[(z, 1.0), (u, 1.0)], ComparisonOp::Le, 0.7);
    for presolve in [false, true] {
        let mut options = SolveOptions::default();
        options.presolve = presolve;
        options.warm_start = Some(vec![(x, -5e-10), (w, -1e-9), (z, 0.0), (u, 0.0)]);
        let sol = solution(p.solve_with(options).unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        for var in [x, w, z, u] {
            let value = sol.var_value(var);
            assert!(
                !crate::solver::outside_bounds(value, 0.0, 1.0, 1e-7),
                "presolve {presolve}: {var:?} = {value:e} is outside [0, 1]"
            );
        }
        assert!(sol.objective().abs() <= 1e-6, "presolve {presolve}");
    }
}
