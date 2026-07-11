//! API-level presolve tests: the observable behavior of `Problem::solve` /
//! `Solution` must be identical with presolve on and off, on models that
//! exercise each reduction, including the post-solve edit paths that make
//! baked-in reductions risky in the first place.

#[cfg(test)]
mod tests_presolve_api {
    use crate::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn solve_both(problem: &Problem) -> (Result<Solution, Error>, Result<Solution, Error>) {
        let with = problem.solve_with(SolveOptions::default());
        let without = problem.solve_with(SolveOptions {
            presolve: false,
            ..SolveOptions::default()
        });
        (with, without)
    }

    fn assert_same_objective(problem: &Problem) -> Solution {
        let (with, without) = solve_both(problem);
        let with = with.unwrap();
        let without = without.unwrap();
        assert_eq!(with.status(), Status::Optimal);
        assert_eq!(without.status(), Status::Optimal);
        assert!(
            (with.objective() - without.objective()).abs()
                <= 1e-6 * (1.0 + without.objective().abs()),
            "objective diverged: presolve {} vs raw {}",
            with.objective(),
            without.objective()
        );
        with
    }

    /// LP with a redundant row, a singleton row and a forcing structure: the
    /// presolved solver ends with zero rows, and the optimum must not move.
    #[test]
    fn lp_reductions_preserve_optimum() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 3.0));
        let y = p.add_var(2.0, (0.0, f64::INFINITY));
        p.add_constraint(&[(y, 2.0)], ComparisonOp::Le, 8.0); // singleton: y <= 4
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 100.0); // redundant
        let sol = assert_same_objective(&p);
        assert_eq!(sol.objective(), 11.0); // x = 3, y = 4
        assert_eq!(sol[x], 3.0);
        assert_eq!(sol[y], 4.0);
    }

    /// The doc example, with presolve explicitly on and off.
    #[test]
    fn doc_example_unchanged() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        let y = problem.add_var(2.0, (0.0, 3.0));
        problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);
        let sol = assert_same_objective(&problem);
        assert_eq!(sol.objective(), 7.0);
        assert_eq!(sol[x], 1.0);
        assert_eq!(sol[y], 3.0);
    }

    /// MILP where presolve rounds fractional integer bounds and tightens.
    #[test]
    fn milp_fractional_integer_bounds() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        // Integer x with an LP-loose row 2x <= 7 (x <= 3.5 -> x <= 3).
        let x = p.add_integer_var(1.0, (0, 100));
        let y = p.add_var(0.5, (0.0, 10.0));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Le, 7.0);
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 12.0);
        let sol = assert_same_objective(&p);
        assert_eq!(sol.objective(), 7.5); // x = 3 (from 2x <= 7), y = 9: 3 + 4.5
        assert_eq!(sol.var_value(x), 3.0);
    }

    /// Infeasibility detected by presolve must surface as the same public
    /// error the raw solver reports.
    #[test]
    fn infeasible_lp_and_milp_report_infeasible() {
        init();
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 5.0));
        let y = p.add_var(1.0, (0.0, 5.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 11.0);
        let (with, without) = solve_both(&p);
        assert_eq!(with.unwrap_err(), Error::Infeasible);
        assert_eq!(without.unwrap_err(), Error::Infeasible);

        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 5));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 3.0); // x = 1.5: no integer
        let (with, without) = solve_both(&p);
        assert_eq!(with.unwrap_err(), Error::Infeasible);
        assert_eq!(without.unwrap_err(), Error::Infeasible);
    }

    /// Unboundedness must remain the simplex's verdict (dual fixing skips
    /// infinite bounds rather than misclassifying).
    #[test]
    fn unbounded_stays_unbounded() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let _x = p.add_var(1.0, (0.0, f64::INFINITY));
        let y = p.add_var(0.0, (0.0, 1.0));
        p.add_constraint(&[(y, 1.0)], ComparisonOp::Le, 1.0);
        let (with, without) = solve_both(&p);
        assert_eq!(with.unwrap_err(), Error::Unbounded);
        assert_eq!(without.unwrap_err(), Error::Unbounded);
    }

    /// LP live edits on a presolved solution: rows were dropped from the live
    /// solver, later edits must still give the exact from-scratch answer.
    #[test]
    fn lp_add_constraint_after_presolved_solve() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 10.0));
        let y = p.add_var(1.0, (0.0, 10.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 100.0); // redundant, dropped
        let sol = p.solve().unwrap();
        assert_eq!(sol.objective(), 20.0);
        // Edit: now make the dropped-row world tighter than bounds.
        let sol = sol
            .add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 12.0)
            .unwrap();
        assert_eq!(sol.objective(), 12.0);
    }

    /// LP fix_var against a presolve-tightened bound: fixing to a value that
    /// presolve proved infeasible must report Infeasible (that is the correct
    /// verdict — no feasible point has that value).
    #[test]
    fn lp_fix_var_outside_tightened_bound_is_infeasible() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 10.0));
        let y = p.add_var(0.0, (0.0, 10.0));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Le, 8.0); // x <= 4 (singleton)
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 20.0);
        let sol = p.solve().unwrap();
        assert_eq!(sol.objective(), 4.0);
        // x = 5 violates the (presolved-away) row: Infeasible either way.
        assert_eq!(sol.fix_var(x, 5.0).unwrap_err(), Error::Infeasible);

        // And a fix INSIDE the tightened range keeps working incrementally.
        let sol2 = p.solve().unwrap().fix_var(x, 2.0).unwrap();
        assert_eq!(sol2.objective(), 2.0);
        let (sol3, was_fixed) = sol2.unfix_var(x).unwrap();
        assert!(was_fixed);
        assert_eq!(sol3.objective(), 4.0);
    }

    /// MILP post-solve edits re-solve from the untouched base problem, so a
    /// presolved first solve must not leak into edit results.
    #[test]
    fn milp_edits_after_presolved_solve() {
        init();
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
        assert_eq!(sol.objective(), 21.0); // y + z + w
        let sol = sol.fix_var(y, 0.0).unwrap();
        assert_eq!(sol.objective(), 18.0); // x + z + w = 8 + 6 + 4
        let (sol, was_fixed) = sol.unfix_var(y).unwrap();
        assert!(was_fixed);
        assert_eq!(sol.objective(), 21.0);
        let sol = sol
            .add_constraint(&[(y, 1.0), (z, 1.0)], ComparisonOp::Le, 1.0)
            .unwrap();
        assert_eq!(sol.objective(), 19.0); // x + y + w = 8 + 11 ... 23 > weight 15 no; y+w+x: 5+7+3=15 > 14; best: y + x = 12 w=3 -> 8+11+4=23? weight 5+7+3=15 infeasible; x+y=12 obj 19 feasible
    }

    /// A MILP whose objective-only variables get dual-fixed: same optimum,
    /// and warm-started (hinted) solves — where dual fixing is disabled —
    /// agree too.
    #[test]
    fn milp_dual_fixing_and_warm_start_agree() {
        init();
        let mut p = Problem::new(OptimizationDirection::Minimize);
        // Penalty vars that only appear in <=-rows with positive coeffs:
        // dual-fixable to their lower bounds.
        let a = p.add_integer_var(3.0, (0, 10));
        let b = p.add_integer_var(4.0, (0, 10));
        let pen = p.add_integer_var(5.0, (0, 8)); // only in a <=-row: fix at 0
        p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
        p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
        p.add_constraint(&[(pen, 1.0), (a, 1.0)], ComparisonOp::Le, 50.0);
        let plain = assert_same_objective(&p);
        assert_eq!(plain.objective(), 11.0);
        assert_eq!(plain.var_value(pen), 0.0);

        // Hinted solve: a feasible non-optimal hint must not change the answer.
        let hinted = p
            .solve_with(SolveOptions {
                warm_start: Some(vec![(a, 5.0), (b, 0.0), (pen, 1.0)]),
                ..SolveOptions::default()
            })
            .unwrap();
        assert_eq!(hinted.status(), Status::Optimal);
        assert_eq!(hinted.objective(), 11.0);
    }

    /// Big-M rows: coefficient tightening must preserve the optimum of a
    /// fixed-charge model (and beat the raw formulation to it).
    #[test]
    fn milp_fixed_charge_big_m() {
        init();
        // Two facilities with capacities and fixed opening costs, one demand.
        // minimize 100 o1 + 150 o2 + 2 s1 + 3 s2
        //   s1 + s2 >= 10; s1 <= 1000 o1 (big-M); s2 <= 1000 o2; s in [0, 1000].
        // Optimum: open facility 1 only, s1 = 10: 100 + 20 = 120.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let o1 = p.add_binary_var(100.0);
        let o2 = p.add_binary_var(150.0);
        let s1 = p.add_var(2.0, (0.0, 1000.0));
        let s2 = p.add_var(3.0, (0.0, 1000.0));
        p.add_constraint(&[(s1, 1.0), (s2, 1.0)], ComparisonOp::Ge, 10.0);
        p.add_constraint(&[(s1, 1.0), (o1, -1000.0)], ComparisonOp::Le, 0.0);
        p.add_constraint(&[(s2, 1.0), (o2, -1000.0)], ComparisonOp::Le, 0.0);
        let sol = assert_same_objective(&p);
        assert_eq!(sol.objective(), 120.0);
        assert_eq!(sol.var_value(o1), 1.0);
        assert_eq!(sol.var_value(o2), 0.0);
    }

    /// Interrupt/resume on a presolved MILP: the resumable state is built on
    /// the presolved problem, resuming must reach the same optimum.
    #[test]
    fn milp_interrupt_resume_with_presolve() {
        init();
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
        let mut sol = p
            .solve_with(SolveOptions {
                node_limit: Some(1),
                ..SolveOptions::default()
            })
            .unwrap();
        let mut guard = 0;
        while sol.status() != Status::Optimal {
            guard += 1;
            assert!(guard < 10_000, "resume loop did not terminate");
            sol = sol.resume(None).unwrap();
        }
        assert_eq!(sol.objective(), 21.0);
    }

    /// Equality-heavy model: substitution chains (x fixed -> row becomes
    /// singleton -> next var fixed ...) must keep MILP answers exact.
    #[test]
    fn milp_substitution_chain() {
        init();
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 100));
        let y = p.add_integer_var(1.0, (0, 100));
        let z = p.add_integer_var(1.0, (0, 100));
        p.add_constraint(&[(x, 1.0)], ComparisonOp::Eq, 7.0);
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Eq, 12.0);
        p.add_constraint(&[(y, 1.0), (z, 1.0)], ComparisonOp::Ge, 9.0);
        let sol = assert_same_objective(&p);
        assert_eq!(sol.objective(), 16.0); // x=7, y=5, z=4
        assert_eq!(sol.var_value(x), 7.0);
        assert_eq!(sol.var_value(y), 5.0);
        assert_eq!(sol.var_value(z), 4.0);
    }

    /// Maximize-direction MILP with negative coefficients: sign handling in
    /// dual fixing / tightening must respect the internal negation.
    #[test]
    fn milp_maximize_negative_coeffs() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_integer_var(-2.0, (-5, 5));
        let y = p.add_integer_var(3.0, (-5, 5));
        p.add_constraint(&[(x, -1.0), (y, 2.0)], ComparisonOp::Le, 7.0);
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, -3.0);
        let sol = assert_same_objective(&p);
        // maximize -2x + 3y: x = -3, y = 2 hits row1 exactly (3 + 4 = 7) and
        // row2 loosely (-1 >= -3): objective 6 + 6 = 12. Pushing x lower
        // caps y harder (x = -4 -> y <= 1, obj 11; x = -5 crosses row2).
        assert_eq!(sol.objective(), 12.0);
    }
}
