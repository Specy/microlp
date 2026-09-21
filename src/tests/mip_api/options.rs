//! `SolveOptions` / `ResumeOptions`: what resuming carries over, what it
//! replaces, and which values are rejected.

use super::common::{binary_knapsack, solution};
use crate::*;
use core::time::Duration;

#[test]
fn resume_with_replaces_the_time_limit_and_applies_its_mip_gap() {
    let p = binary_knapsack();
    let mut options = SolveOptions::default();
    options.time_limit = Some(Duration::ZERO);
    options.mip_gap = 0.5;

    let interrupted = p.solve_with(options).unwrap();
    assert!(interrupted.solution().is_none());
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::TimeLimit
    );

    let resumed = solution(
        interrupted
            .resume_with(ResumeOptions {
                mip_gap: Some(0.5),
                ..ResumeOptions::default()
            })
            .unwrap(),
    );
    assert_eq!(resumed.status(), SolutionStatus::Feasible);
    assert_eq!(resumed.termination_reason(), TerminationReason::MipGap);
    assert!(resumed.gap().unwrap() <= 0.5 + 1e-9);
}

#[test]
fn plain_resume_preserves_the_configured_time_limit() {
    let p = binary_knapsack();
    let mut options = SolveOptions::default();
    options.time_limit = Some(Duration::ZERO);

    let interrupted = p.solve_with(options).unwrap();
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::TimeLimit
    );

    let resumed = interrupted.resume().unwrap();
    assert_eq!(resumed.termination_reason(), TerminationReason::TimeLimit);
}

#[test]
fn resume_with_replaces_old_call_options_and_resume_reuses_the_replacements() {
    let p = binary_knapsack();
    let mut initial = SolveOptions::default();
    initial.time_limit = Some(Duration::ZERO);
    initial.node_limit = Some(17);
    initial.mip_gap = 0.5;

    let interrupted = p.solve_with(initial).unwrap();
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::TimeLimit
    );

    let replacements = ResumeOptions {
        time_limit: None,
        node_limit: Some(0),
        mip_gap: None,
    };
    let interrupted = interrupted.resume_with(replacements.clone()).unwrap();
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::NodeLimit
    );
    assert_eq!(interrupted.last_resume_options(), replacements);

    let resumed = interrupted.resume().unwrap();
    assert_eq!(resumed.termination_reason(), TerminationReason::NodeLimit);
    assert_eq!(resumed.last_resume_options(), replacements);
}

#[test]
fn resuming_a_gap_satisfied_solution_without_new_options_keeps_the_gap() {
    let p = binary_knapsack();
    let mut options = SolveOptions::default();
    options.mip_gap = 0.5;
    let outcome = p.solve_with(options).unwrap();
    let nodes_before = outcome.stats().nodes_solved;

    let resumed = outcome.resume().unwrap();

    let sol = solution(resumed);
    assert_eq!(sol.status(), SolutionStatus::Feasible);
    assert_eq!(sol.termination_reason(), TerminationReason::MipGap);
    assert_eq!(sol.stats().nodes_solved, nodes_before);
}

#[test]
fn resume_with_no_gap_target_continues_from_gap_to_exact_proof() {
    let p = binary_knapsack();
    let mut options = SolveOptions::default();
    options.mip_gap = 0.5;
    let outcome = p.solve_with(options).unwrap();

    let exact = solution(outcome.resume_with(ResumeOptions::default()).unwrap());

    assert_eq!(exact.status(), SolutionStatus::Optimal);
    assert_eq!(exact.termination_reason(), TerminationReason::ProvenOptimal);
    assert!((exact.objective() - 21.0).abs() < 1e-6);
}

#[test]
fn exact_proof_wins_over_a_positive_gap_when_the_root_is_integral() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_binary_var(1.0);
    let mut options = SolveOptions::default();
    options.mip_gap = 0.5;

    let sol = solution(p.solve_with(options).unwrap());

    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert_eq!(sol.termination_reason(), TerminationReason::ProvenOptimal);
    assert_eq!(sol.var_value(x), 1.0);
}

#[test]
fn an_exact_bound_wins_over_a_positive_gap_with_an_open_tree() {
    // The objective is fixed at 1. The triangle-cover relaxation is
    // fractional, so root initialization leaves an open tree, while the
    // warm start supplies an integer incumbent with that same objective.
    // Equality of incumbent and global bound is already an exact proof.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    p.add_var(1.0, (1.0, 1.0));
    let x = p.add_binary_var(0.0);
    let y = p.add_binary_var(0.0);
    let z = p.add_binary_var(0.0);
    p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 1.0);
    p.add_constraint([(x, 1.0), (z, 1.0)], ComparisonOp::Ge, 1.0);
    p.add_constraint([(y, 1.0), (z, 1.0)], ComparisonOp::Ge, 1.0);

    let mut options = SolveOptions::default();
    options.mip_gap = 0.5;
    options.warm_start = Some(vec![(x, 1.0), (y, 1.0), (z, 1.0)]);

    let sol = solution(p.solve_with(options).unwrap());

    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert_eq!(sol.termination_reason(), TerminationReason::ProvenOptimal);
    assert_eq!(sol.objective(), 1.0);
    assert_eq!(sol.stats().best_bound, Some(1.0));
    assert_eq!(sol.gap(), Some(0.0));
    assert_eq!(sol.stats().nodes_solved, 0);
}

#[test]
fn invalid_resume_gap_is_rejected_without_changing_the_search() {
    let mut p = binary_knapsack();
    p.set_time_limit(Duration::ZERO);
    let outcome = p.solve().unwrap();

    let err = outcome
        .resume_with(ResumeOptions {
            mip_gap: Some(f64::NAN),
            ..ResumeOptions::default()
        })
        .unwrap_err();

    assert!(
        matches!(err, Error::InvalidOptions(message) if message.contains("ResumeOptions.mip_gap"))
    );
}

#[test]
fn invalid_resume_gap_is_rejected_even_after_optimality_is_proven() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    p.add_binary_var(1.0);
    let outcome = p.solve().unwrap();
    assert!(outcome.is_optimal());

    let err = outcome
        .resume_with(ResumeOptions {
            mip_gap: Some(f64::NAN),
            ..ResumeOptions::default()
        })
        .unwrap_err();

    assert!(
        matches!(err, Error::InvalidOptions(message) if message.contains("ResumeOptions.mip_gap"))
    );
}

#[test]
fn invalid_integrality_tolerances_are_rejected() {
    // Negative/NaN tolerances make exact integers look fractional; a
    // tolerance of 0.5 or more can classify every real value as integral
    // and makes the rounding-based branch split skip part of the domain.
    // The node limit bounds any accidental unchanged-branch loop while
    // each invalid tolerance is being rejected.
    for int_tol in [-1.0, f64::NAN, 0.5, f64::INFINITY] {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        p.add_integer_var(1.0, (0, 1));
        let mut options = SolveOptions::default();
        options.int_tol = int_tol;
        options.node_limit = Some(1);

        let err = p.solve_with(options).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidOptions(message) if message.contains("int_tol")),
            "unexpected error for int_tol={int_tol}: {err}"
        );
    }
}

#[test]
fn invalid_gap_and_expert_tolerances_are_rejected() {
    fn assert_invalid(field: &str, options: SolveOptions) {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        p.add_integer_var(1.0, (0, 1));
        let err = p.solve_with(options).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidOptions(message) if message.contains(field)),
            "unexpected error for {field}: {err}"
        );
    }

    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let mut options = SolveOptions::default();
        options.mip_gap = value;
        assert_invalid("mip_gap", options);
    }
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let mut options = SolveOptions::default();
        options.tolerances.feasibility = value;
        assert_invalid("tolerances.feasibility", options);
    }
    for value in [-1.0, f64::NAN, 0.5, f64::INFINITY] {
        let mut options = SolveOptions::default();
        options.tolerances.integrality_rounding = value;
        assert_invalid("tolerances.integrality_rounding", options);
    }
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let mut options = SolveOptions::default();
        options.tolerances.prune_epsilon = value;
        assert_invalid("tolerances.prune_epsilon", options);
    }
}
