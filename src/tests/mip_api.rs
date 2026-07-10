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
    fn interrupted_reads_expose_current_state_without_panicking() {
        let (mut p, x, y) = int_2var_problem();
        p.set_time_limit(Duration::ZERO);
        let sol = p.solve().unwrap();
        assert_eq!(sol.status(), Status::Interrupted);
        // Reading an interrupted solution is allowed: the accessors expose
        // the search's current working point (possibly fractional integer
        // values, possibly infeasible) rather than panicking. Interpreting
        // them as the answer requires checking status() first — that is the
        // caller's contract.
        assert!(sol.objective().is_finite());
        assert!(sol.var_value_raw(x).is_finite());
        assert!(sol.var_value(y).is_finite());
        assert!(sol[x].is_finite());
        assert_eq!(sol.iter().count(), 2);
        assert!(sol.iter().all(|(_, v)| v.is_finite()));
        // Resuming turns the working point into a real answer.
        let sol = sol.resume(None).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
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

    #[test]
    fn warm_start_with_optimal_hint_is_accepted() {
        let (p, a, b) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
        let sol = p.solve_with(options).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn warm_start_with_infeasible_hint_is_ignored() {
        let (p, a, b) = int_2var_problem();
        let mut options = SolveOptions::default();
        // a=0, b=0 violates both constraints — hint must be dropped, solve still exact.
        options.warm_start = Some(vec![(a, 0.0), (b, 0.0)]);
        let sol = p.solve_with(options).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn warm_start_out_of_bounds_hint_is_ignored() {
        let (p, a, b) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.warm_start = Some(vec![(a, 99.0), (b, 2.0)]); // 99 > upper bound 10
        let sol = p.solve_with(options).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn warm_start_partial_hint_completes_via_lp() {
        // minimize 3a + 4b, a + 2b >= 5, 3a + b >= 4; hint only a=1 → LP completes b,
        // and if the completion is integral (b=2) it seeds the incumbent.
        let (p, a, _) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.warm_start = Some(vec![(a, 1.0)]);
        let sol = p.solve_with(options).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
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
        assert_eq!(cold.status(), Status::Interrupted);

        let mut options = SolveOptions::default();
        options.node_limit = Some(0);
        options.warm_start = Some(vec![(a, 1.0), (b, 2.0)]);
        let hinted = p.solve_with(options).unwrap();
        assert_eq!(hinted.status(), Status::Feasible);
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
        let with_hint = p.solve_with(options).unwrap();
        let without = p.solve().unwrap();
        // Correctness identical; the hinted run must not explore MORE nodes.
        assert!(with_hint.stats().nodes_solved <= without.stats().nodes_solved);
        assert_eq!(with_hint.objective(), without.objective());
    }

    #[test]
    fn milp_add_constraint_resolves_on_base_problem() {
        let (p, a, b) = int_2var_problem();
        let sol = p.solve().unwrap();
        assert!((sol.objective() - 11.0).abs() < 1e-6); // a=1, b=2

        // Cut off the incumbent: a + b >= 4. From-scratch optimum of the edited
        // problem, over points on the binding line a+b=4: (a=0,b=4): cons1 8>=5,
        // cons2 4>=4, obj 16; (a=1,b=3): 7>=5, 6>=4, obj 15; (a=2,b=2): 6>=5,
        // 8>=4, obj 14; (a=3,b=1): cons1 5>=5, cons2 10>=4, obj 13; (a=4,b=0)
        // violates a+2b>=5. → unique optimum 13 at (3,1).
        let sol = sol
            .add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
            .unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 13.0).abs() < 1e-6);
        assert_eq!(sol.var_value(a), 3.0);
        assert_eq!(sol.var_value(b), 1.0);

        // Must equal a from-scratch solve of the edited problem.
        let (mut p2, a2, b2) = (int_2var_problem().0, Variable(0), Variable(1));
        p2.add_constraint(&[(a2, 1.0), (b2, 1.0)], ComparisonOp::Ge, 4.0);
        let fresh = p2.solve().unwrap();
        assert!((fresh.objective() - sol.objective()).abs() < 1e-6);
    }

    #[test]
    fn milp_fix_and_unfix_var_roundtrip() {
        let (p, a, b) = int_2var_problem();
        let sol = p.solve().unwrap();

        // Fix a=3: then b >= 1 (cons1: 3+2b>=5) → obj 9+4=13 at (3,1).
        let sol = sol.fix_var(a, 3.0).unwrap();
        assert_eq!(sol.status(), Status::Optimal);
        assert!((sol.objective() - 13.0).abs() < 1e-6);
        assert_eq!(sol.var_value(a), 3.0);
        assert_eq!(sol.var_value(b), 1.0);

        // Unfix restores the original optimum and reports it was fixed.
        let (sol, was_fixed) = sol.unfix_var(a).unwrap();
        assert!(was_fixed);
        assert!((sol.objective() - 11.0).abs() < 1e-6);

        // Unfixing a never-fixed var is a no-op with `false`.
        let (sol, was_fixed) = sol.unfix_var(b).unwrap();
        assert!(!was_fixed);
        assert!((sol.objective() - 11.0).abs() < 1e-6);
    }

    #[test]
    fn milp_fix_var_outside_bounds_is_infeasible_error() {
        let (p, a, _) = int_2var_problem();
        let sol = p.solve().unwrap();
        assert!(matches!(sol.fix_var(a, 99.0), Err(Error::Infeasible)));
    }

    #[test]
    fn milp_edit_after_pause_completes_correctly() {
        let (p, a, b) = int_2var_problem();
        let mut options = SolveOptions::default();
        options.node_limit = Some(1); // pause almost immediately
        let sol = p.solve_with(options).unwrap();
        // Whatever the paused status, editing must work on the ORIGINAL problem + edit.
        let sol = sol
            .add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 4.0)
            .unwrap();
        let sol = if sol.status() == Status::Optimal {
            sol
        } else {
            sol.resume(None).unwrap()
        };
        assert!((sol.objective() - 13.0).abs() < 1e-6);
    }

    #[test]
    fn milp_infeasible_edit_is_an_error() {
        let (p, a, _) = int_2var_problem();
        let sol = p.solve().unwrap();
        // a <= -1 crosses a's [0,10] bounds → infeasible.
        assert!(matches!(
            sol.add_constraint(&[(a, 1.0)], ComparisonOp::Le, -1.0),
            Err(Error::Infeasible)
        ));
    }

    #[test]
    fn milp_gomory_cut_is_rejected_with_internal_error() {
        let (p, a, _) = int_2var_problem();
        let sol = p.solve().unwrap();
        assert!(matches!(
            sol.add_gomory_cut(a),
            Err(Error::InternalError(_))
        ));
    }

    #[test]
    fn lp_interrupted_add_gomory_cut_errors_not_panics() {
        // Pure-LP problem (continuous vars). A zero-duration budget interrupts the
        // solve before it completes, so the solution is Status::Interrupted. Before
        // the guard, add_gomory_cut reached solver internals that assume a finished
        // solve and panicked; it must now return an InternalError, matching the
        // guard already on add_constraint/fix_var.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 10.0));
        let y = p.add_var(1.0, (0.0, 10.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 3.0);
        let mut options = SolveOptions::default();
        options.time_limit = Some(Duration::ZERO);
        let sol = p.solve_with(options).unwrap();
        assert_eq!(sol.status(), Status::Interrupted);
        assert!(matches!(
            sol.add_gomory_cut(x),
            Err(Error::InternalError(_))
        ));
    }
}
