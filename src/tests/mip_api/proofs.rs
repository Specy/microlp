//! Optimality proofs are never short-circuited: a rounded candidate, an
//! infeasible model or an unbounded relaxation must each be reported honestly.

use super::common::solution;
use crate::*;
use core::time::Duration;

#[test]
fn rounded_root_candidate_does_not_bypass_optimality_proof() {
    // The root LP is b=5e-7, y=1, z=0 with objective -3. The default
    // integrality tolerance considers b integral and rounding it to zero
    // yields the feasible point (0, 1, 0), but that point has objective -1.
    // The true integer optimum is (0, 0, 1), objective -2.
    //
    // Feasibility of the rounded point therefore cannot justify returning
    // Optimal: because rounding changed the objective away from the LP
    // bound, branch-and-bound still has proof work to do.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let b = p.add_binary_var(-4_000_000.0);
    let y = p.add_binary_var(-1.0);
    let z = p.add_binary_var(-2.0);
    p.add_constraint(&[(b, 1.0)], ComparisonOp::Le, 5e-7);
    p.add_constraint(&[(b, 2_000_000.0), (z, 1.0)], ComparisonOp::Le, 1.0);
    p.add_constraint(&[(y, 1.0), (z, 1.0)], ComparisonOp::Le, 1.0);

    let sol = solution(p.solve().unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - -2.0).abs() < 1e-9);
    assert_eq!(sol.var_value(b), 0.0);
    assert_eq!(sol.var_value(y), 0.0);
    assert_eq!(sol.var_value(z), 1.0);
}

#[test]
fn rounded_node_candidate_does_not_bypass_its_subtree_proof() {
    // a >= 0.5 forces an ordinary root branch. In the feasible a=1
    // child, the remaining relaxation is the near-integral root fixture
    // above. The child must branch on b rather than treat
    // the feasible rounded point as proof that its whole subtree is done.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let a = p.add_binary_var(0.0);
    let b = p.add_binary_var(-4_000_000.0);
    let y = p.add_binary_var(-1.0);
    let z = p.add_binary_var(-2.0);
    p.add_constraint(&[(a, 1.0)], ComparisonOp::Ge, 0.5);
    p.add_constraint(&[(b, 1.0)], ComparisonOp::Le, 5e-7);
    p.add_constraint(&[(b, 2_000_000.0), (z, 1.0)], ComparisonOp::Le, 1.0);
    p.add_constraint(&[(y, 1.0), (z, 1.0)], ComparisonOp::Le, 1.0);

    let sol = solution(p.solve().unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - -2.0).abs() < 1e-9);
    assert_eq!(sol.var_value(a), 1.0);
    assert_eq!(sol.var_value(b), 0.0);
    assert_eq!(sol.var_value(y), 0.0);
    assert_eq!(sol.var_value(z), 1.0);
}

#[test]
fn milp_infeasible_is_an_error() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(1.0, (0, 10));
    p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
    assert_eq!(p.solve().unwrap_err(), Error::Infeasible);
}

#[test]
fn unbounded_relaxation_does_not_mask_integer_infeasibility() {
    // The relaxation is unbounded through y, but x is forced to 0.5 and
    // has an integer domain, so the actual MILP has no feasible point.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(0.0, (0, 1));
    let _y = p.add_var(-1.0, (0.0, f64::INFINITY));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Eq, 0.5);
    assert_eq!(p.solve().unwrap_err(), Error::Infeasible);
}

#[test]
fn unbounded_relaxation_classification_resumes_after_root_interrupt() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(0.0, (0, 1));
    let _y = p.add_var(-1.0, (0.0, f64::INFINITY));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Eq, 0.5);
    let mut options = SolveOptions::default();
    options.time_limit = Some(Duration::ZERO);
    // Presolve proves this model infeasible before the search starts;
    // the engine's own classification of an unbounded relaxation is
    // what is under test here.
    options.presolve = false;

    let interrupted = p.solve_with(options).unwrap();
    assert!(interrupted.solution().is_none());
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::TimeLimit
    );
    assert_eq!(
        interrupted
            .resume_with(ResumeOptions::default())
            .unwrap_err(),
        Error::Infeasible
    );
}

#[test]
fn interrupted_unbounded_classification_hides_the_working_point() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(7.0, (0, 1));
    let _y = p.add_var(-1.0, (0.0, f64::INFINITY));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Eq, 0.5);

    let mut options = SolveOptions::default();
    options.node_limit = Some(0);
    options.presolve = false; // see the test above

    let outcome = p.solve_with(options).unwrap();
    assert!(outcome.solution().is_none());
    assert_eq!(outcome.termination_reason(), TerminationReason::NodeLimit);
    assert_eq!(outcome.stats().nodes_solved, 0);
}
