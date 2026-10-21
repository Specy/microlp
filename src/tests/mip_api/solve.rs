//! Solving a MILP: status, objective sign, variable domains, and the limits
//! that interrupt a solve.

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

/// An integer var's value is reported as computed, within `int_tol` of its
/// integer.
#[test]
fn milp_solve_reports_optimal_and_near_integral_values() {
    let (p, a, b) = int_2var_problem();
    let sol = solution(p.solve().unwrap());
    let int_tol = SolveOptions::default().int_tol;
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 11.0).abs() < 1e-6);
    assert!((sol.var_value(a) - 1.0).abs() <= int_tol);
    assert!((sol.var_value(b) - 2.0).abs() <= int_tol);
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

/// `Problem::var_domain` returns each var's domain as it was added.
#[test]
fn problem_var_domain_returns_each_variables_domain() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, 1.0));
    let n = p.add_integer_var(1.0, (0, 3));
    let b = p.add_binary_var(1.0);
    assert_eq!(p.var_domain(x), VarDomain::Real);
    assert_eq!(p.var_domain(n), VarDomain::Integer);
    assert_eq!(p.var_domain(b), VarDomain::Boolean);
}

/// `Problem::var_domain` panics on a handle whose index the problem does not
/// have, as `Solution::var_value` does.
#[test]
#[should_panic(expected = "assertion failed: var.0 <")]
fn problem_var_domain_panics_on_a_variable_outside_the_problem() {
    let mut other = Problem::new(OptimizationDirection::Minimize);
    other.add_var(1.0, (0.0, 1.0));
    let foreign = other.add_var(1.0, (0.0, 1.0));
    let mut p = Problem::new(OptimizationDirection::Minimize);
    p.add_var(1.0, (0.0, 1.0));
    p.var_domain(foreign);
}

/// `Solution::var_domain` returns each var's domain on a MILP solution, the
/// problem's, before and after each kind of edit; the values the search
/// found round to integers on the integer and boolean vars.
#[test]
fn milp_solution_var_domain_returns_each_variables_domain_across_edits() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_var(1.0, (0.0, 2.5));
    let n = p.add_integer_var(2.0, (0, 10));
    let b = p.add_binary_var(1.0);
    p.add_constraint(&[(x, 1.0), (n, 1.0), (b, 1.0)], ComparisonOp::Le, 7.5);
    let domains = [
        (x, VarDomain::Real),
        (n, VarDomain::Integer),
        (b, VarDomain::Boolean),
    ];
    let int_tol = SolveOptions::default().int_tol;
    let check = |sol: &Solution| {
        for (var, domain) in domains {
            assert_eq!(sol.var_domain(var), domain);
            assert_eq!(sol.var_domain(var), p.var_domain(var));
            if domain != VarDomain::Real {
                let value = sol.var_value(var);
                assert!((value - value.round()).abs() <= int_tol, "{value}");
            }
        }
    };

    let sol = solution(p.solve().unwrap());
    check(&sol);
    let sol = solution(
        sol.add_constraint(&[(n, 1.0)], ComparisonOp::Le, 5.0)
            .unwrap(),
    );
    check(&sol);
    let sol = solution(sol.fix_var(b, 0.0).unwrap());
    check(&sol);
    let (outcome, was_fixed) = sol.unfix_var(b).unwrap();
    assert!(was_fixed);
    check(&solution(outcome));
}
