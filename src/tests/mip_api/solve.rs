//! Solving a MILP: status, objective sign, and the limits that interrupt a solve.

use super::common::{binary_knapsack, int_2var_problem, solution};
use crate::*;
use core::time::Duration;

#[test]
fn exact_milp_returns_an_optimal_solution_with_proof_reason() {
    let (p, _, _) = int_2var_problem();
    let outcome = p.solve().unwrap();

    assert!(outcome.is_optimal());
    assert_eq!(
        outcome.termination_reason(),
        TerminationReason::ProvenOptimal
    );
    let solution = outcome
        .into_solution()
        .expect("an exact solve must contain a solution");
    assert_eq!(solution.status(), SolutionStatus::Optimal);
    assert_eq!(
        solution.termination_reason(),
        TerminationReason::ProvenOptimal
    );
}

#[test]
fn zero_time_limit_without_incumbent_is_typed_as_interrupted() {
    let (mut p, _, _) = int_2var_problem();
    p.set_time_limit(Duration::ZERO);
    let outcome = p.solve().unwrap();

    assert!(outcome.solution().is_none());
    assert_eq!(outcome.termination_reason(), TerminationReason::TimeLimit);
    let interrupted = outcome.into_solution().unwrap_err();
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::TimeLimit
    );
}

#[test]
fn milp_solve_reports_optimal_and_rounds_values() {
    let (p, a, b) = int_2var_problem();
    let sol = solution(p.solve().unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
    assert_eq!(sol.var_value(a), 1.0);
    assert_eq!(sol.var_value(b), 2.0);
    assert!(sol.stats().nodes_solved > 0);
    assert!(sol.stats().lp_iterations > 0);
}

#[test]
fn milp_maximize_sign_is_correct() {
    // maximize 8x + 11y + 6z + 4w, 5x + 7y + 4z + 3w <= 14, binaries → 21.
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
    let sol = solution(p.solve().unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 21.0).abs() < 1e-6);
    assert_eq!(sol.var_value(x), 0.0);
    assert_eq!(sol.var_value(y), 1.0);
}

#[test]
fn zero_time_limit_is_interrupted_then_resume_finishes() {
    let (mut p, _, _) = int_2var_problem();
    p.set_time_limit(Duration::ZERO);
    let outcome = p.solve().unwrap();
    assert!(outcome.solution().is_none());
    assert_eq!(outcome.termination_reason(), TerminationReason::TimeLimit);
    let sol = solution(outcome.resume_with(ResumeOptions::default()).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn interrupted_outcome_exposes_only_reason_stats_and_resume() {
    let (mut p, _, _) = int_2var_problem();
    p.set_time_limit(Duration::ZERO);
    let outcome = p.solve().unwrap();
    assert!(outcome.solution().is_none());
    assert_eq!(outcome.termination_reason(), TerminationReason::TimeLimit);
    assert_eq!(outcome.stats().nodes_solved, 0);

    let sol = solution(outcome.resume_with(ResumeOptions::default()).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn node_limit_interrupts_deterministically_and_resumes() {
    let (p, _, _) = int_2var_problem();
    let mut options = SolveOptions::default();
    options.node_limit = Some(1);
    let mut outcome = p.solve_with(options).unwrap();
    let mut resumes = 0;
    while !outcome.is_optimal() {
        resumes += 1;
        assert!(resumes < 10_000);
        outcome = outcome.resume().unwrap();
    }
    let sol = solution(outcome);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
}

#[test]
fn positive_mip_gap_returns_a_feasible_solution_with_gap_reason() {
    let p = binary_knapsack();
    let mut options = SolveOptions::default();
    options.mip_gap = 0.5;

    let sol = solution(p.solve_with(options).unwrap());

    assert_eq!(sol.status(), SolutionStatus::Feasible);
    assert_eq!(sol.termination_reason(), TerminationReason::MipGap);
    assert!(sol.gap().unwrap() <= 0.5 + 1e-9);
}
