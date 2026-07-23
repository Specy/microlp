#[cfg(test)]
mod tests_mip_api {
    use crate::*;
    use core::time::Duration;

    fn int_2var_problem() -> (Problem, Variable, Variable) {
        // minimize 3a + 4b s.t. a + 2b >= 5, 3a + b >= 4; a,b int in [0,10] → a=1,b=2, obj 11.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let a = p.add_integer_var(3.0, (0, 10));
        let b = p.add_integer_var(4.0, (0, 10));
        p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
        p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
        (p, a, b)
    }

    fn binary_knapsack() -> Problem {
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

    fn solution(outcome: SolveOutcome) -> Solution {
        outcome
            .into_solution()
            .expect("this solve must return a usable solution")
    }

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

    #[test]
    fn plain_resume_after_time_limit_preserves_the_configured_mip_gap() {
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

        let resume_options = ResumeOptions {
            mip_gap: Some(0.5),
            ..ResumeOptions::default()
        };
        let resumed = solution(interrupted.resume_with(resume_options).unwrap());
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
        assert_eq!(
            resumed.termination_reason(),
            TerminationReason::TimeLimit
        );
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
    fn an_explicit_resume_gap_can_continue_from_gap_to_exact_proof() {
        let p = binary_knapsack();
        let mut options = SolveOptions::default();
        options.mip_gap = 0.5;
        let outcome = p.solve_with(options).unwrap();

        let exact = solution(
            outcome
                .resume_with(ResumeOptions {
                    mip_gap: Some(0.0),
                    ..ResumeOptions::default()
                })
                .unwrap(),
        );

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

        let interrupted = p.solve_with(options).unwrap();
        assert!(interrupted.solution().is_none());
        assert_eq!(
            interrupted.termination_reason(),
            TerminationReason::TimeLimit
        );
        assert_eq!(
            interrupted.resume_with(ResumeOptions::default()).unwrap_err(),
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

        let outcome = p.solve_with(options).unwrap();
        assert!(outcome.solution().is_none());
        assert_eq!(outcome.termination_reason(), TerminationReason::NodeLimit);
        assert_eq!(outcome.stats().nodes_solved, 0);
    }

    #[test]
    fn lp_path_still_solves_and_edits_incrementally() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 4.0));
        let y = p.add_var(2.0, (0.0, 3.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 5.0);
        let sol = solution(p.solve().unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert!((sol.objective() - 8.0).abs() < 1e-6);
        // Live-basis incremental add on the LP path.
        let sol = solution(
            sol.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1.0)
                .unwrap(),
        );
        assert!((sol.objective() - 7.0).abs() < 1e-6);
        assert!((sol[x] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn optimal_lp_stats_report_bound_and_zero_gap() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let _x = p.add_var(2.0, (0.0, 3.0));
        let sol = solution(p.solve().unwrap());
        let stats = sol.stats();
        assert_eq!(stats.best_bound, Some(6.0));
        assert_eq!(stats.gap, Some(0.0));
    }

    #[test]
    fn lp_edit_uses_the_most_recent_resume_time_budget() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 10.0));
        p.set_time_limit(Duration::ZERO);

        let interrupted = p.solve().unwrap();
        assert_eq!(
            interrupted.termination_reason(),
            TerminationReason::TimeLimit
        );

        let resumed = solution(interrupted.resume_with(ResumeOptions::default()).unwrap());
        assert_eq!(resumed.objective(), 10.0);

        let edited = solution(
            resumed
                .add_constraint([(x, 1.0)], ComparisonOp::Le, 4.0)
                .unwrap(),
        );
        assert_eq!(edited.status(), SolutionStatus::Optimal);
        assert_eq!(edited.objective(), 4.0);
    }

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

    #[test]
    fn milp_add_constraint_resolves_on_base_problem() {
        let (p, a, b) = int_2var_problem();
        let sol = solution(p.solve().unwrap());
        assert!((sol.objective() - 11.0).abs() < 1e-6); // a=1, b=2

        // Cut off the incumbent: a + b >= 4. From-scratch optimum of the edited
        // problem, over points on the binding line a+b=4: (a=0,b=4): cons1 8>=5,
        // cons2 4>=4, obj 16; (a=1,b=3): 7>=5, 6>=4, obj 15; (a=2,b=2): 6>=5,
        // 8>=4, obj 14; (a=3,b=1): cons1 5>=5, cons2 10>=4, obj 13; (a=4,b=0)
        // violates a+2b>=5. → unique optimum 13 at (3,1).
        let sol = solution(
            sol.add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
                .unwrap(),
        );
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert!((sol.objective() - 13.0).abs() < 1e-6);
        assert_eq!(sol.var_value(a), 3.0);
        assert_eq!(sol.var_value(b), 1.0);

        // Must equal a from-scratch solve of the edited problem.
        let (mut p2, a2, b2) = (int_2var_problem().0, Variable(0), Variable(1));
        p2.add_constraint(&[(a2, 1.0), (b2, 1.0)], ComparisonOp::Ge, 4.0);
        let fresh = solution(p2.solve().unwrap());
        assert!((fresh.objective() - sol.objective()).abs() < 1e-6);
    }

    #[test]
    fn milp_fix_and_unfix_var_roundtrip() {
        let (p, a, b) = int_2var_problem();
        let sol = solution(p.solve().unwrap());

        // Fix a=3: then b >= 1 (cons1: 3+2b>=5) → obj 9+4=13 at (3,1).
        let sol = solution(sol.fix_var(a, 3.0).unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert!((sol.objective() - 13.0).abs() < 1e-6);
        assert_eq!(sol.var_value(a), 3.0);
        assert_eq!(sol.var_value(b), 1.0);

        // Unfix restores the original optimum and reports it was fixed.
        let (sol, was_fixed) = sol.unfix_var(a).unwrap();
        let sol = solution(sol);
        assert!(was_fixed);
        assert!((sol.objective() - 11.0).abs() < 1e-6);

        // Unfixing a never-fixed var is a no-op with `false`.
        let (sol, was_fixed) = sol.unfix_var(b).unwrap();
        let sol = solution(sol);
        assert!(!was_fixed);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn milp_fix_var_outside_bounds_is_infeasible_error() {
        let (p, a, _) = int_2var_problem();
        let sol = solution(p.solve().unwrap());
        assert!(matches!(sol.fix_var(a, 99.0), Err(Error::Infeasible)));
    }

    #[test]
    fn milp_fix_var_non_finite_is_infeasible_error() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (p, a, _) = int_2var_problem();
            let sol = solution(p.solve().unwrap());
            assert!(
                matches!(sol.fix_var(a, invalid), Err(Error::Infeasible)),
                "non-finite fix {invalid} was not rejected"
            );
        }
    }

    #[test]
    fn milp_edit_after_pause_completes_correctly() {
        let (p, a, b) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.node_limit = Some(1); // pause almost immediately
        let mut outcome = p.solve_with(options).unwrap();
        // Interrupted searches are deliberately not editable. Resume to a
        // validated solution before changing the original model.
        while outcome.solution().is_none() {
            outcome = outcome.resume().unwrap();
        }
        let sol = solution(outcome);
        let edited = sol
            .add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
            .unwrap();
        let sol = solution(if edited.is_optimal() {
            edited
        } else {
            edited.resume().unwrap()
        });
        assert!((sol.objective() - 13.0).abs() < 1e-6);
    }

    #[test]
    fn milp_infeasible_edit_is_an_error() {
        let (p, a, _) = int_2var_problem();
        let sol = solution(p.solve().unwrap());
        // a <= -1 crosses a's [0,10] bounds → infeasible.
        assert!(matches!(
            sol.add_constraint(&[(a, 1.0)], ComparisonOp::Le, -1.0),
            Err(Error::Infeasible)
        ));
    }

    #[test]
    fn lp_stats_report_nonzero_elapsed_time() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 10.0));
        p.add_constraint([(x, 1.0)], ComparisonOp::Ge, 3.0);

        let sol = solution(p.solve().unwrap());
        let initial_elapsed = sol.stats().elapsed;
        assert!(initial_elapsed > Duration::ZERO);

        let edited = solution(
            sol.add_constraint([(x, 1.0)], ComparisonOp::Ge, 4.0)
                .unwrap(),
        );
        assert!(edited.stats().elapsed >= initial_elapsed);
    }

    #[test]
    fn lp_edit_gets_a_fresh_time_budget() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 10.0));
        p.add_constraint([(x, 1.0)], ComparisonOp::Ge, 3.0);
        let mut options = SolveOptions::default();
        options.time_limit = Some(Duration::from_millis(50));

        let sol = solution(p.solve_with(options).unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        std::thread::sleep(Duration::from_millis(75));

        let edited = solution(
            sol.add_constraint([(x, 1.0)], ComparisonOp::Ge, 4.0)
                .unwrap(),
        );
        assert_eq!(edited.status(), SolutionStatus::Optimal);
        assert_eq!(edited.objective(), 4.0);
    }

    #[test]
    fn lp_unfix_reports_an_interrupted_reoptimization() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(-1.0, (0.0, 10.0));
        let solved = solution(p.solve().unwrap());
        let mut fixed = solution(solved.fix_var(x, 0.0).unwrap());
        match &mut fixed.state {
            SolveState::Lp(solver) => {
                solver.operation_time_limit = Some(Duration::ZERO);
            }
            SolveState::Mip(_) => unreachable!(),
        }

        let (unfixed, was_fixed) = fixed.unfix_var(x).unwrap();
        assert!(was_fixed);
        assert!(unfixed.solution().is_none());
        assert_eq!(unfixed.termination_reason(), TerminationReason::TimeLimit);
    }

    /// An interrupted pure-LP edit must resume to the *edited* model's optimum —
    /// the same answer as solving that model from scratch. The model change is
    /// durable and the cut-off reoptimization simply continues on resume. This
    /// is the resume counterpart to `lp_unfix_reports_an_interrupted_reoptimization`,
    /// which only checks that the interruption is reported.
    #[test]
    fn interrupted_lp_edit_resumes_to_the_edited_models_optimum() {
        // minimize x + 2y + 3z  s.t.  x + y + z >= 6,  each var in [0, 10].
        // The cheapest unit is x, so the base optimum is (6, 0, 0), objective 6.
        // build() is deterministic, so variable indices match across instances.
        let build = || {
            let mut p = Problem::new(OptimizationDirection::Minimize);
            let x = p.add_var(1.0, (0.0, 10.0));
            let y = p.add_var(2.0, (0.0, 10.0));
            let z = p.add_var(3.0, (0.0, 10.0));
            p.add_constraint([(x, 1.0), (y, 1.0), (z, 1.0)], ComparisonOp::Ge, 6.0);
            (p, x, y, z)
        };

        let (base, x, y, z) = build();
        let mut sol = solution(base.solve().unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        let base_obj = sol.objective();
        assert!((base_obj - 6.0).abs() < 1e-6);

        // Starve the next edit of time so its reoptimization is cut off at the
        // first deadline check, before any pivot: the row is appended but no
        // feasibility restoration runs.
        match &mut sol.state {
            SolveState::Lp(solver) => solver.operation_time_limit = Some(Duration::ZERO),
            SolveState::Mip(_) => unreachable!("a continuous model stays pure-LP"),
        }

        // `x <= 2` cuts off the incumbent x = 6, so the edit genuinely needs to
        // reoptimize — work that the zero budget defers entirely to resume.
        let interrupted = sol
            .add_constraint([(x, 1.0)], ComparisonOp::Le, 2.0)
            .unwrap();
        assert!(interrupted.solution().is_none());
        assert_eq!(
            interrupted.termination_reason(),
            TerminationReason::TimeLimit
        );

        // An ample resume budget finishes the edited solve.
        let resumed = solution(interrupted.resume_with(ResumeOptions::default()).unwrap());
        assert_eq!(resumed.status(), SolutionStatus::Optimal);

        // Oracle: the same edited model solved from scratch → (2, 4, 0), obj 10.
        let (mut edited, ..) = build();
        edited.add_constraint([(x, 1.0)], ComparisonOp::Le, 2.0);
        let fresh = solution(edited.solve().unwrap());
        assert_eq!(fresh.status(), SolutionStatus::Optimal);

        // Resume and the fresh solve agree on objective and the (unique) vertex,
        // and both differ from the pre-edit incumbent — proving resume ran the
        // reoptimization rather than returning stale state.
        assert!((resumed.objective() - fresh.objective()).abs() < 1e-6);
        assert!((resumed.objective() - 10.0).abs() < 1e-6);
        assert!(resumed.objective() > base_obj + 0.5);
        assert!(resumed.var_value(x) <= 2.0 + 1e-6);
        for var in [x, y, z] {
            assert!((resumed.var_value(var) - fresh.var_value(var)).abs() < 1e-6);
        }
    }
}
