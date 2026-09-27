//! API-level presolve tests: `Problem::solve` and `Solution` must behave the
//! same with presolve on and off, on models exercising each reduction and the
//! post-solve edit paths.

#[cfg(test)]
mod tests_presolve_api {
    use crate::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn solve_both(problem: &Problem) -> (Result<SolveOutcome, Error>, Result<SolveOutcome, Error>) {
        let with = problem.solve_with(SolveOptions::default());
        let without = problem.solve_with(SolveOptions {
            presolve: false,
            ..SolveOptions::default()
        });
        (with, without)
    }

    fn solution(outcome: Result<SolveOutcome, Error>) -> Solution {
        outcome
            .unwrap()
            .into_solution()
            .unwrap_or_else(|i| panic!("expected a solution, got {i:?}"))
    }

    fn assert_same_objective(problem: &Problem) -> Solution {
        let (with, without) = solve_both(problem);
        let with = solution(with);
        let without = solution(without);
        assert_eq!(with.status(), SolutionStatus::Optimal);
        assert_eq!(without.status(), SolutionStatus::Optimal);
        assert!(
            (with.objective() - without.objective()).abs()
                <= 1e-6 * (1.0 + without.objective().abs()),
            "objective diverged: presolve {} vs raw {}",
            with.objective(),
            without.objective()
        );
        with
    }

    /// The answer satisfies every row as the user wrote it on every path: the
    /// MIP search, the pure-LP solve and an LP resumed after an interrupt.
    /// `-10 x3 >= 0` sees x3's bound tolerance ten times over, more than the
    /// row can take, so presolve must keep it as a row rather than a bound.
    #[test]
    fn answers_satisfy_the_rows_presolve_dropped_on_every_path() {
        init();
        let build = |milp: bool| {
            let mut p = Problem::new(OptimizationDirection::Maximize);
            if milp {
                p.add_integer_var(0.0, (0, 0));
            }
            let x1 = p.add_var(0.0, (-8388610.0, 0.0));
            let _x2 = p.add_var(0.0, (0.0, 0.0));
            let x3 = p.add_var(0.0, (-1.0, 0.0));
            p.add_constraint([(x3, -10.0)], ComparisonOp::Ge, -0.0);
            p.add_constraint(
                [(x1, -1.0), (x3, 1.0745249307680744)],
                ComparisonOp::Eq,
                0.0,
            );
            p.add_constraint([(x1, 1.0), (x3, -1.0)], ComparisonOp::Eq, 0.0);
            (p, x3)
        };
        let check = |what: &str, result: Result<SolveOutcome, Error>, x3: Variable| {
            let sol = solution(result);
            assert_eq!(sol.status(), SolutionStatus::Optimal, "{what}");
            let activity = -10.0 * sol[x3];
            assert!(
                activity >= -1e-7,
                "{what}: the row -10 x3 >= 0 has activity {activity:e}"
            );
        };
        for milp in [true, false] {
            let (p, x3) = build(milp);
            check(&format!("milp {milp}"), p.solve(), x3);
        }
        let (p, x3) = build(false);
        let interrupted = p
            .solve_with(SolveOptions {
                time_limit: Some(std::time::Duration::ZERO),
                ..SolveOptions::default()
            })
            .unwrap();
        assert!(interrupted.solution().is_none());
        check(
            "resumed LP",
            interrupted.resume_with(ResumeOptions::default()),
            x3,
        );
    }

    /// An LP with a singleton row and a redundant row: presolve must not move
    /// the optimum.
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

    /// Edits to a presolved LP solution, whose live solver lacks the dropped
    /// rows, still give the from-scratch answer.
    #[test]
    fn lp_add_constraint_after_presolved_solve() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 10.0));
        let y = p.add_var(1.0, (0.0, 10.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 100.0); // redundant, dropped
        let sol = solution(p.solve());
        assert_eq!(sol.objective(), 20.0);
        // Edit: a row that, unlike the dropped one, binds.
        let sol = solution(sol.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 12.0));
        assert_eq!(sol.objective(), 12.0);
    }

    /// `fix_var` to a value no feasible point has is `Infeasible` after a
    /// presolved LP solve, and a fix inside the range still works
    /// incrementally.
    #[test]
    fn lp_fix_var_outside_tightened_bound_is_infeasible() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 10.0));
        let y = p.add_var(0.0, (0.0, 10.0));
        p.add_constraint(&[(x, 2.0)], ComparisonOp::Le, 8.0); // x <= 4 (singleton)
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 20.0);
        let sol = solution(p.solve());
        assert_eq!(sol.objective(), 4.0);
        // x = 5 violates 2x <= 8.
        assert_eq!(sol.fix_var(x, 5.0).unwrap_err(), Error::Infeasible);

        // And a fix INSIDE the tightened range keeps working incrementally.
        let sol2 = solution(solution(p.solve()).fix_var(x, 2.0));
        assert_eq!(sol2.objective(), 2.0);
        let (outcome, was_fixed) = sol2.unfix_var(x).unwrap();
        assert!(was_fixed);
        assert_eq!(solution(Ok(outcome)).objective(), 4.0);
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
        let sol = solution(p.solve());
        assert_eq!(sol.objective(), 21.0); // y + z + w
        let sol = solution(sol.fix_var(y, 0.0));
        assert_eq!(sol.objective(), 18.0); // x + z + w = 8 + 6 + 4
        let (outcome, was_fixed) = sol.unfix_var(y).unwrap();
        assert!(was_fixed);
        let sol = solution(Ok(outcome));
        assert_eq!(sol.objective(), 21.0);
        let sol = solution(sol.add_constraint(&[(y, 1.0), (z, 1.0)], ComparisonOp::Le, 1.0));
        // With y + z <= 1: {x, y} weighs 12 and scores 19, the best of the
        // remaining subsets.
        assert_eq!(sol.objective(), 19.0);
    }

    /// Dual fixing of objective-only vars keeps the optimum, and a hinted
    /// solve, where dual fixing is off, agrees.
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
        let hinted = solution(p.solve_with(SolveOptions {
            warm_start: Some(vec![(a, 5.0), (b, 0.0), (pen, 1.0)]),
            ..SolveOptions::default()
        }));
        assert_eq!(hinted.status(), SolutionStatus::Optimal);
        assert_eq!(hinted.objective(), 11.0);
    }

    /// Big-M rows: coefficient tightening must preserve the optimum of a
    /// fixed-charge model.
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
        // Approximate: the continuous supply vars carry path-dependent
        // round-off. The integer values stay exact.
        assert!(
            (sol.objective() - 120.0).abs() < 1e-6,
            "{}",
            sol.objective()
        );
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
        let mut outcome = p
            .solve_with(SolveOptions {
                node_limit: Some(1),
                ..SolveOptions::default()
            })
            .unwrap();
        let mut guard = 0;
        while !outcome.is_optimal() {
            guard += 1;
            assert!(guard < 10_000, "resume loop did not terminate");
            outcome = outcome.resume().unwrap();
        }
        assert_eq!(solution(Ok(outcome)).objective(), 21.0);
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

    /// An implied bound computed by cancellation (`rest = activity − own
    /// term`) is relaxed for the round-off of the whole activity: beside a
    /// bound of `i32::MIN` (an integer var's "no bound") the rest carries 2e-7
    /// of it, which `floor` would turn into a whole unit, making
    /// `x ≤ 3.4 − 0.4 y` with `y ≥ 1` into `x ≤ 2` (and the equality
    /// `Infeasible`).
    #[test]
    fn implied_integer_bound_survives_cancellation_against_a_huge_bound() {
        init();
        for op in [ComparisonOp::Le, ComparisonOp::Eq] {
            let mut p = Problem::new(OptimizationDirection::Maximize);
            let x = p.add_integer_var(1.0, (i32::MIN, 10));
            let y = p.add_integer_var(0.0, (1, 5));
            p.add_constraint(&[(x, 1.0), (y, 0.4)], op, 3.4);
            let sol = assert_same_objective(&p);
            assert_eq!(sol.objective(), 3.0, "{op:?}");
            assert_eq!(sol.var_value(x), 3.0, "{op:?}");
        }
    }

    /// A forcing row may fix its vars at their extremes only when that cuts
    /// nothing a var could resolve: the row's minimum falls 1.5e-5 short of
    /// its bound, within the round-off of a magnitude of 4e9, but that sliver
    /// is a range of 0.015 for `y` (optimum 15.02, not 0).
    #[test]
    fn forcing_row_does_not_cut_a_range_a_var_can_resolve() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(0.0, (-2e9, 0.0));
        let y = p.add_var(1000.0, (0.0, 1000.0));
        p.add_constraint(&[(x, 1.0), (y, 0.001)], ComparisonOp::Le, -2e9 + 1.5e-5);
        let sol = assert_same_objective(&p);
        assert!(
            (sol.objective() - 15.02).abs() < 1e-3,
            "{}",
            sol.objective()
        );
    }

    /// A singleton row is a bound only as loose as the row: `0.01 x = 0.01 +
    /// 1e-9` accepts `x` within 1e-5 of 1.0000001, and fixing `x` there made
    /// the row `x = 1`, which the model meets within 1e-9, look inconsistent
    /// in one row order.
    #[test]
    fn singleton_row_keeps_the_tolerance_of_the_row() {
        init();
        for swap in [false, true] {
            let mut p = Problem::new(OptimizationDirection::Minimize);
            let x = p.add_var(1.0, (0.0, 10.0));
            let rows = [(0.01, 0.01 + 1e-9), (1.0, 1.0)];
            let order = if swap { [1, 0] } else { [0, 1] };
            for i in order {
                let (a, b) = rows[i];
                p.add_constraint(&[(x, a)], ComparisonOp::Eq, b);
            }
            let sol = assert_same_objective(&p);
            assert!((sol[x] - 1.0).abs() < 1e-7, "swap {swap}: x = {}", sol[x]);
        }
    }

    /// A singleton row implied by its var's bound is kept when the bound
    /// tolerance, times the row's coefficient, breaks it: x1 may sit 1e-8
    /// above its bound of 0, which `-20 x1 >= 0` sees as 2e-7 against 1e-7.
    /// (The model is feasible at the origin.)
    #[test]
    fn singleton_row_tighter_than_its_bound_tolerance_is_kept() {
        init();
        for obj in [0.0, 1.0] {
            let mut p = Problem::new(OptimizationDirection::Maximize);
            let x0 = p.add_var(-obj, (-1.0, 1.0));
            let x1 = p.add_var(obj, (-1.0, 0.0));
            p.add_constraint(&[(x0, -1.0), (x1, -1.0)], ComparisonOp::Le, 1.0);
            p.add_constraint(&[(x0, -1e-8), (x1, -1.0)], ComparisonOp::Eq, -0.0);
            p.add_constraint(&[(x1, -20.0)], ComparisonOp::Ge, -0.0);
            let sol = assert_same_objective(&p);
            assert!(-20.0 * sol[x1] >= -1e-7, "obj {obj}: x1 = {:e}", sol[x1]);
        }
    }

    /// Integer points are held to the full contract (the incumbent guard's),
    /// so presolve's integer rounding may not cut one that meets it: `x + y
    /// = 4 + 6e-8` holds within 1e-7 at (4, 0), but rounding against half
    /// the tolerance moved each bound by a unit per pass until they crossed.
    #[test]
    fn integer_rounding_keeps_points_the_guard_accepts() {
        init();
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var(1.0, (0, 10));
        let y = p.add_integer_var(1.0, (0, 10));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Eq, 4.0 + 6e-8);
        let sol = assert_same_objective(&p);
        assert_eq!(sol.objective(), 4.0);
    }

    /// A row whose vars are all fixed is judged at the magnitude the engine
    /// sees (the rhs as written plus the fixed terms), not at the rhs left
    /// after substitution: `x + y = 2e9 + 1.5e-5` at `x = y = 1e9` is within
    /// the round-off of 4e9.
    #[test]
    fn all_substituted_row_is_judged_at_its_full_magnitude() {
        init();
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (1e9, 1e9));
        let y = p.add_var(1.0, (1e9, 1e9));
        let z = p.add_var(1.0, (0.0, 1.0));
        p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Eq, 2e9 + 1.5e-5);
        p.add_constraint(&[(z, 1.0), (x, 1e-9)], ComparisonOp::Ge, 1.0);
        let sol = assert_same_objective(&p);
        assert_eq!(sol[x], 1e9);
    }

    /// An equality at 1.34e8 with a 1e-8 coefficient is feasible; presolve
    /// declared it infeasible.
    #[test]
    fn equality_at_1e8_with_a_tiny_coefficient_is_feasible() {
        init();
        for obj in [0.0, 1.0] {
            let mut p = Problem::new(OptimizationDirection::Maximize);
            let x0 = p.add_var(-obj, (13421773.0, 13421774.0));
            let x1 = p.add_var(0.0, (-268.0, 1.0));
            let _x2 = p.add_var(obj, (0.0, 0.0));
            p.add_constraint(&[(x1, 1.0)], ComparisonOp::Le, 0.0);
            p.add_constraint(&[(x0, -10.0), (x1, 1e-8)], ComparisonOp::Eq, -134217730.0);
            let sol = assert_same_objective(&p);
            assert!((sol[x0] - 13421773.0).abs() < 1e-6, "x0 = {}", sol[x0]);
        }
    }

    /// A row whose vars presolve fixed is judged against the full contract,
    /// as an empty input row is, not against the engine's half: the forcing
    /// row `x1 + 100 x4 = 8388700` fixes `x1` at `-2^-30` and `x4` at 83887,
    /// leaving `-100 x1 + x4 / 1000 = 83.887` a residual of 9.3e-8, within
    /// 1e-7 but not 5e-8.
    #[test]
    fn all_substituted_row_is_judged_at_the_full_contract() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x1 = p.add_var(-1.0, (-2.0f64.powi(-30), 1.0));
        let x4 = p.add_var(1.0, (83887.0, 83888.0));
        p.add_constraint(&[(x1, -1.0), (x4, -100.0)], ComparisonOp::Eq, -8388700.0);
        p.add_constraint(&[(x1, -100.0), (x4, 1e-3)], ComparisonOp::Eq, 83.887);
        let sol = assert_same_objective(&p);
        assert_eq!(sol[x4], 83887.0);
    }

    /// Singleton rows go first in each wave: taken first, `-100 x1 = 0` pins
    /// `x1` at zero and `x1 + 100 x4 = 8388700` becomes a singleton in `x4`.
    /// In index order, that forcing row fixed `x1` at `-2^-30`, which the
    /// singleton then declared infeasible.
    #[test]
    fn singleton_row_is_processed_before_a_forcing_row() {
        init();
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let _x0 = p.add_integer_var(0.0, (0, 0));
        let x1 = p.add_var(-1.0, (-2.0f64.powi(-30), 1.0));
        let _x2 = p.add_var(0.0, (0.0, 0.0));
        let _x3 = p.add_var(-1.0, (0.0, 0.0));
        let x4 = p.add_var(1.0, (83887.0, 83888.0));
        p.add_constraint(&[(x1, -1.0), (x4, -100.0)], ComparisonOp::Eq, -8388700.0);
        p.add_constraint(&[(x1, -100.0)], ComparisonOp::Eq, -0.0);
        let sol = solution(p.solve_with(SolveOptions::default()));
        assert_eq!((sol[x1], sol[x4]), (0.0, 83887.0));
        assert_same_objective(&p);
    }
}
