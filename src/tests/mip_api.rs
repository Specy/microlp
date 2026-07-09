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

    #[test]
    fn milp_solve_reports_optimal_and_rounds_values() {
        let (p, a, b) = int_2var_problem();
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Optimal);
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
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 21.0).abs() < 1e-6);
        assert_eq!(sol.var_value(x), 0.0);
        assert_eq!(sol.var_value(y), 1.0);
    }

    #[test]
    fn zero_time_limit_is_interrupted_then_resume_finishes() {
        let (mut p, _, _) = int_2var_problem();
        p.set_time_limit(Duration::ZERO);
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Interrupted);
        assert!(sol.gap().is_none());
        let sol = sol.resume(None).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "interrupted before finding a usable solution")]
    fn interrupted_objective_panics_with_clear_message() {
        let (mut p, _, _) = int_2var_problem();
        p.set_time_limit(Duration::ZERO);
        let sol = p.solve().unwrap();
        let _ = sol.objective(); // must panic, not return garbage
    }

    #[test]
    fn node_limit_interrupts_deterministically_and_resumes() {
        let (p, _, _) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.node_limit = Some(1);
        let mut sol = p.solve_with(options).unwrap();
        let mut resumes = 0;
        while sol.status() != Status::Optimal {
            resumes += 1;
            assert!(resumes < 10_000);
            sol = sol.resume(None).unwrap();
        }
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn milp_infeasible_is_an_error() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 1.0);
        assert_eq!(p.solve().unwrap_err(), Error::Infeasible);
    }

    #[test]
    fn lp_path_still_solves_and_edits_incrementally() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 4.0));
        let y = p.add_var(2.0, (0.0, 3.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 5.0);
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 8.0).abs() < 1e-6);
        // Live-basis incremental add on the LP path.
        let sol = sol
            .add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1.0)
            .unwrap();
        assert!((sol.objective() - 7.0).abs() < 1e-6);
        assert!((sol[x] - 1.0).abs() < 1e-6);
    }
}
