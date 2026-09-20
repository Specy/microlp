//! Special ordered sets through the public API, checked against brute-force
//! enumeration: an SOS1 optimum is the best over "one member free, the rest
//! fixed to zero"; an SOS2 optimum is the best over adjacent member pairs.

#[cfg(test)]
mod tests_sos {
    use crate::*;
    use core::time::Duration;

    const TOL: f64 = 1e-6;

    fn solution(outcome: SolveOutcome) -> Solution {
        outcome
            .into_solution()
            .expect("this solve must return a usable solution")
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= TOL * (1.0 + a.abs().max(b.abs()))
    }

    /// Objective of `build()` with every member outside `free` fixed to zero,
    /// or `None` when that restriction is infeasible. The restricted problem
    /// carries no SOS, so it is solved as a plain LP / MILP.
    fn restricted_objective(
        build: &dyn Fn(&mut Problem) -> Vec<Variable>,
        direction: OptimizationDirection,
        free: &[usize],
    ) -> Option<f64> {
        let mut p = Problem::new(direction);
        let members = build(&mut p);
        for (i, &m) in members.iter().enumerate() {
            if !free.contains(&i) {
                p.add_constraint([(m, 1.0)], ComparisonOp::Eq, 0.0);
            }
        }
        match p.solve() {
            Ok(outcome) => Some(solution(outcome).objective()),
            Err(Error::Infeasible) => None,
            Err(e) => panic!("restricted solve failed: {e:?}"),
        }
    }

    fn best(direction: OptimizationDirection, values: impl Iterator<Item = f64>) -> Option<f64> {
        values.fold(None, |acc, v| match acc {
            None => Some(v),
            Some(b) => Some(match direction {
                OptimizationDirection::Minimize => b.min(v),
                OptimizationDirection::Maximize => b.max(v),
            }),
        })
    }

    /// Brute-force SOS1 optimum: best single free member (or none free).
    fn sos1_oracle(
        build: &dyn Fn(&mut Problem) -> Vec<Variable>,
        direction: OptimizationDirection,
        n: usize,
    ) -> Option<f64> {
        let mut candidates = vec![restricted_objective(build, direction, &[])];
        for i in 0..n {
            candidates.push(restricted_objective(build, direction, &[i]));
        }
        best(direction, candidates.into_iter().flatten())
    }

    /// Brute-force SOS2 optimum: best adjacent pair (covers singles and none).
    fn sos2_oracle(
        build: &dyn Fn(&mut Problem) -> Vec<Variable>,
        direction: OptimizationDirection,
        n: usize,
    ) -> Option<f64> {
        let mut candidates = vec![];
        for i in 0..n.saturating_sub(1) {
            candidates.push(restricted_objective(build, direction, &[i, i + 1]));
        }
        best(direction, candidates.into_iter().flatten())
    }

    fn assert_sos1_holds(sol: &Solution, members: &[Variable]) {
        let nonzero = members
            .iter()
            .filter(|&&m| sol.var_value(m).abs() > 1e-7)
            .count();
        assert!(nonzero <= 1, "SOS1 violated: {nonzero} nonzero members");
    }

    fn assert_sos2_holds(sol: &Solution, members: &[Variable]) {
        let nonzero: Vec<usize> = (0..members.len())
            .filter(|&i| sol.var_value(members[i]).abs() > 1e-7)
            .collect();
        assert!(
            nonzero.len() <= 2 && nonzero.windows(2).all(|w| w[1] == w[0] + 1),
            "SOS2 violated: nonzero positions {nonzero:?}"
        );
    }

    // ------------------------------------------------------------ SOS1

    /// Three continuous members share a capacity; the LP relaxation spreads
    /// the value over all of them, the SOS1 optimum keeps the best one.
    fn sos1_capacity(p: &mut Problem) -> Vec<Variable> {
        let x = p.add_var(3.0, (0.0, 4.0));
        let y = p.add_var(5.0, (0.0, 3.0));
        let z = p.add_var(4.0, (0.0, 5.0));
        p.add_constraint([(x, 1.0), (y, 1.0), (z, 1.0)], ComparisonOp::Le, 6.0);
        // A side constraint that makes y alone worse than it looks: y ≤ z + 1.
        p.add_constraint([(y, 1.0), (z, -1.0)], ComparisonOp::Le, 1.0);
        vec![x, y, z]
    }

    #[test]
    fn sos1_continuous_matches_enumeration() {
        let direction = OptimizationDirection::Maximize;
        let mut p = Problem::new(direction);
        let members = sos1_capacity(&mut p);
        let relaxed = solution(p.clone().solve().unwrap()).objective();
        p.add_sos1_constraint(members.iter().enumerate().map(|(i, &m)| (m, i as f64)));

        let sol = solution(p.solve().unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_sos1_holds(&sol, &members);
        let want = sos1_oracle(&sos1_capacity, direction, 3).unwrap();
        assert!(
            close(sol.objective(), want),
            "{} vs {}",
            sol.objective(),
            want
        );
        // The SOS is binding: the relaxation is strictly better.
        assert!(relaxed > want + 1.0);
        // z alone at 5 (obj 20) beats x alone at 4 (12) and y alone at 1 (5).
        assert!(close(sol.objective(), 20.0));
        assert_eq!(sol.var_value(members[2]), 5.0);
    }

    #[test]
    fn sos1_minimize_direction() {
        let direction = OptimizationDirection::Minimize;
        let build = |p: &mut Problem| -> Vec<Variable> {
            let x = p.add_var(1.0, (0.0, 10.0));
            let y = p.add_var(2.0, (0.0, 10.0));
            let z = p.add_var(3.0, (0.0, 10.0));
            // Demand forces 4 units; the relaxation uses x for everything but
            // x is capped by a second row unless y is zero.
            p.add_constraint([(x, 1.0), (y, 1.0), (z, 1.0)], ComparisonOp::Ge, 4.0);
            p.add_constraint([(x, 1.0), (y, 3.0)], ComparisonOp::Le, 3.0);
            vec![x, y, z]
        };
        let mut p = Problem::new(direction);
        let members = build(&mut p);
        p.add_sos1_constraint(members.iter().map(|&m| (m, 1.0)));
        let sol = solution(p.solve().unwrap());
        assert_sos1_holds(&sol, &members);
        let want = sos1_oracle(&build, direction, 3).unwrap();
        assert!(
            close(sol.objective(), want),
            "{} vs {}",
            sol.objective(),
            want
        );
        assert!(close(want, 12.0)); // z = 4 alone
    }

    // ------------------------------------------------------------ SOS2

    /// Piecewise-linear λ-formulation of a NONCONVEX function with
    /// breakpoints t = 0, 1, 2, 3 and values f = 0, 3, 1, 4, evaluated at
    /// x = 1.5 and minimised. The LP relaxation takes the convex hull
    /// (λ_0 = 0.25, λ_2 = 0.75 → 0.75); SOS2 forces the segment [1, 2]
    /// (λ_1 = λ_2 = 0.5 → 2).
    fn pwl(p: &mut Problem) -> Vec<Variable> {
        let t = [0.0, 1.0, 2.0, 3.0];
        let f = [0.0, 3.0, 1.0, 4.0];
        let lam: Vec<Variable> = (0..4).map(|i| p.add_var(f[i], (0.0, 1.0))).collect();
        p.add_constraint(lam.iter().map(|&l| (l, 1.0)), ComparisonOp::Eq, 1.0);
        p.add_constraint(
            lam.iter().zip(t).map(|(&l, ti)| (l, ti)),
            ComparisonOp::Eq,
            1.5,
        );
        lam
    }

    #[test]
    fn sos2_piecewise_linear_matches_enumeration() {
        let direction = OptimizationDirection::Minimize;
        let mut p = Problem::new(direction);
        let lam = pwl(&mut p);
        let relaxed = solution(p.clone().solve().unwrap()).objective();
        p.add_sos2_constraint(lam.iter().enumerate().map(|(i, &l)| (l, i as f64)));

        let sol = solution(p.solve().unwrap());
        assert_eq!(sol.status(), SolutionStatus::Optimal);
        assert_sos2_holds(&sol, &lam);
        let want = sos2_oracle(&pwl, direction, 4).unwrap();
        assert!(
            close(sol.objective(), want),
            "{} vs {}",
            sol.objective(),
            want
        );
        assert!(close(relaxed, 0.75), "relaxation {relaxed}");
        assert!(close(want, 2.0));
        assert!(close(sol.var_value(lam[1]), 0.5));
        assert!(close(sol.var_value(lam[2]), 0.5));
    }

    /// Weights given out of order: adjacency follows the weights, not the
    /// insertion order. Inserting λ_3 first must not make {λ_3, λ_0} adjacent.
    #[test]
    fn sos2_adjacency_follows_weights_not_insertion_order() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let lam = pwl(&mut p);
        let mut shuffled: Vec<(Variable, f64)> = vec![(lam[3], 3.0), (lam[0], 0.0)];
        shuffled.push((lam[2], 2.0));
        shuffled.push((lam[1], 1.0));
        p.add_sos2_constraint(shuffled);
        let sol = solution(p.solve().unwrap());
        assert_sos2_holds(&sol, &lam);
        assert!(close(sol.objective(), 2.0));
    }

    // ------------------------------------------------------ mixed models

    #[test]
    fn sos1_with_integer_variables_matches_enumeration() {
        let direction = OptimizationDirection::Maximize;
        let build = |p: &mut Problem| -> Vec<Variable> {
            let n = p.add_integer_var(7.0, (0, 3));
            let a = p.add_var(4.0, (0.0, 2.5));
            let b = p.add_var(5.0, (0.0, 2.5));
            let c = p.add_var(6.0, (0.0, 2.5));
            // 2n + a + b + c ≤ 6.5 and c ≤ n (c only pays with n)
            p.add_constraint(
                [(n, 2.0), (a, 1.0), (b, 1.0), (c, 1.0)],
                ComparisonOp::Le,
                6.5,
            );
            p.add_constraint([(c, 1.0), (n, -1.0)], ComparisonOp::Le, 0.0);
            vec![a, b, c]
        };
        let mut p = Problem::new(direction);
        let members = build(&mut p);
        p.add_sos1_constraint([(members[0], 1.0), (members[1], 2.0), (members[2], 3.0)]);
        let sol = solution(p.solve().unwrap());
        assert_sos1_holds(&sol, &members);
        let want = sos1_oracle(&build, direction, 3).unwrap();
        assert!(
            close(sol.objective(), want),
            "{} vs {}",
            sol.objective(),
            want
        );
    }

    /// Integer members of an SOS1: three binaries with a packing objective
    /// but no packing row — only the set keeps them apart.
    #[test]
    fn sos1_over_binaries_without_packing_row() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_binary_var(5.0);
        let y = p.add_binary_var(7.0);
        let z = p.add_binary_var(6.0);
        p.add_sos1_constraint([(x, 1.0), (y, 2.0), (z, 3.0)]);
        let sol = solution(p.solve().unwrap());
        assert_eq!(sol.objective(), 7.0);
        assert_eq!(sol.var_value(y), 1.0);
        assert_eq!(sol.var_value(x) + sol.var_value(z), 0.0);
    }

    #[test]
    fn two_sets_sharing_a_member() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let a = p.add_var(1.0, (0.0, 1.0));
        let b = p.add_var(3.0, (0.0, 1.0));
        let c = p.add_var(1.0, (0.0, 1.0));
        // b is the best member but belongs to both sets: taking it excludes
        // a and c; taking a and c together yields 2 < 3.
        p.add_sos1_constraint([(a, 1.0), (b, 2.0)]);
        p.add_sos1_constraint([(b, 1.0), (c, 2.0)]);
        let sol = solution(p.solve().unwrap());
        assert_eq!(sol.objective(), 3.0);
        assert_eq!(sol.var_value(b), 1.0);
    }

    /// A member whose LP value is below the zero threshold but load-bearing in
    /// a row (breakpoints 10 apart: λ_2 = 1e-7 moves the abscissa row by 2e-6)
    /// must still be adopted: snapping it to zero would break the row, so the
    /// exact point is kept instead of failing with an internal error.
    #[test]
    fn sub_tolerance_member_that_a_row_depends_on_is_kept() {
        for (t, x) in [
            ([0.0, 10.0, 20.0], 10.0 + 1e-6),
            ([0.0, 1000.0, 2000.0], 1000.0 + 1e-5),
        ] {
            let f = [0.0, 1.0, 5.0];
            let mut p = Problem::new(OptimizationDirection::Minimize);
            let lam: Vec<Variable> = (0..3).map(|i| p.add_var(f[i], (0.0, 1.0))).collect();
            p.add_constraint(lam.iter().map(|&l| (l, 1.0)), ComparisonOp::Eq, 1.0);
            p.add_constraint(
                lam.iter().zip(t).map(|(&l, ti)| (l, ti)),
                ComparisonOp::Eq,
                x,
            );
            p.add_sos2_constraint(lam.iter().zip(t).map(|(&l, ti)| (l, ti)));
            let sol = solution(p.solve().unwrap_or_else(|e| panic!("x = {x}: {e:?}")));
            assert_sos2_holds(&sol, &lam);
            // λ_1 ≈ 1, λ_2 = (x − t_1) / (t_2 − t_1): objective 1 + 4 λ_2.
            let lam2 = (x - t[1]) / (t[2] - t[1]);
            assert!(
                close(sol.objective(), 1.0 + 4.0 * lam2),
                "{}",
                sol.objective()
            );
        }
    }

    // ------------------------------------------------- infeasibility / bounds

    #[test]
    fn sos1_infeasible_when_two_members_are_forced_nonzero() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (1.0, 2.0));
        let y = p.add_var(1.0, (0.0, 2.0));
        p.add_constraint([(y, 1.0)], ComparisonOp::Ge, 0.5);
        p.add_sos1_constraint([(x, 1.0), (y, 2.0)]);
        assert!(matches!(p.solve(), Err(Error::Infeasible)));
    }

    #[test]
    fn member_with_bounds_excluding_zero_forces_the_others_to_zero() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (1.0, 2.0));
        let y = p.add_var(10.0, (0.0, 2.0));
        let z = p.add_var(10.0, (0.0, 2.0));
        p.add_sos1_constraint([(x, 1.0), (y, 2.0), (z, 3.0)]);
        let sol = solution(p.solve().unwrap());
        assert_eq!(sol.objective(), 2.0);
        assert_eq!(sol.var_value(x), 2.0);
        assert_eq!(sol.var_value(y), 0.0);
        assert_eq!(sol.var_value(z), 0.0);
    }

    #[test]
    fn negative_bounds_members_count_nonzero_on_either_side() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (-3.0, 3.0));
        let y = p.add_var(1.0, (-3.0, 3.0));
        p.add_sos1_constraint([(x, 1.0), (y, 2.0)]);
        let sol = solution(p.solve().unwrap());
        // Only one may be at −3.
        assert_eq!(sol.objective(), -3.0);
        assert_sos1_holds(&sol, &[x, y]);
    }

    // ------------------------------------------------------- trivial sets

    #[test]
    fn trivial_sets_are_ignored_and_keep_the_lp_path() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 1.0));
        let y = p.add_var(1.0, (0.0, 1.0));
        p.add_sos1_constraint([(x, 1.0)]);
        p.add_sos2_constraint([(x, 1.0), (y, 2.0)]);
        p.add_sos1_constraint(LinearExpr::empty());
        assert!(!p.needs_branching());
        let sol = solution(p.solve().unwrap());
        assert_eq!(sol.objective(), 2.0);
        assert_eq!(sol.stats().nodes_solved, 0);
    }

    // ------------------------------------------------------- panics

    #[test]
    #[should_panic(expected = "repeats a variable")]
    fn repeated_member_panics() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 1.0));
        let y = p.add_var(1.0, (0.0, 1.0));
        p.add_sos1_constraint([(x, 1.0), (y, 2.0), (x, 3.0)]);
    }

    #[test]
    #[should_panic(expected = "finite bounds")]
    fn unbounded_member_panics() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, f64::INFINITY));
        let y = p.add_var(1.0, (0.0, 1.0));
        p.add_sos1_constraint([(x, 1.0), (y, 2.0)]);
    }

    #[test]
    #[should_panic(expected = "not finite")]
    fn non_finite_weight_panics() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 1.0));
        let y = p.add_var(1.0, (0.0, 1.0));
        p.add_sos1_constraint([(x, f64::NAN), (y, 2.0)]);
    }

    #[test]
    #[should_panic(expected = "distinct")]
    fn sos2_repeated_weights_panic() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 1.0));
        let y = p.add_var(1.0, (0.0, 1.0));
        let z = p.add_var(1.0, (0.0, 1.0));
        p.add_sos2_constraint([(x, 1.0), (y, 1.0), (z, 2.0)]);
    }

    #[test]
    #[should_panic(expected = "but the problem has")]
    fn foreign_variable_panics() {
        let mut other = Problem::new(OptimizationDirection::Minimize);
        let _ = other.add_var(1.0, (0.0, 1.0));
        let foreign = other.add_var(1.0, (0.0, 1.0));
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 1.0));
        p.add_sos1_constraint([(x, 1.0), (foreign, 2.0)]);
    }

    // --------------------------------------------- edits, warm start, resume

    #[test]
    fn edits_on_an_sos_solution_match_hand_computed_optima() {
        let direction = OptimizationDirection::Maximize;
        let mut p = Problem::new(direction);
        let members = sos1_capacity(&mut p);
        p.add_sos1_constraint(members.iter().map(|&m| (m, 1.0)));
        let sol = solution(p.clone().solve().unwrap());
        assert!(close(sol.objective(), 20.0));

        // Fixing z to 0 leaves x alone (obj 12) as the best set choice.
        let sol = solution(sol.fix_var(members[2], 0.0).unwrap());
        assert_sos1_holds(&sol, &members);
        assert!(close(sol.objective(), 12.0));
        assert_eq!(sol.var_value(members[0]), 4.0);

        // Adding x ≤ 1 makes y alone (y ≤ z + 1 = 1 → obj 5) vs x = 1 (3): y wins.
        let sol = solution(
            sol.add_constraint([(members[0], 1.0)], ComparisonOp::Le, 1.0)
                .unwrap(),
        );
        assert_sos1_holds(&sol, &members);
        assert!(close(sol.objective(), 5.0));

        // Unfixing z restores z alone at 5 (obj 20) under the added row.
        let (sol, was_fixed) = sol.unfix_var(members[2]).unwrap();
        assert!(was_fixed);
        let sol = solution(sol);
        assert_sos1_holds(&sol, &members);
        assert!(close(sol.objective(), 20.0));
    }

    /// With a zero node budget only the root and the hint are evaluated, so
    /// the outcome shows whether the hint was adopted: a hint violating the
    /// set leaves no incumbent (interrupted), a valid one becomes the
    /// feasible incumbent even though it is not optimal.
    #[test]
    fn warm_start_hint_violating_the_set_is_ignored_and_a_valid_one_adopted() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let members = sos1_capacity(&mut p);
        p.add_sos1_constraint(members.iter().map(|&m| (m, 1.0)));

        let options = SolveOptions {
            warm_start: Some(vec![
                (members[0], 2.0),
                (members[1], 2.0),
                (members[2], 2.0),
            ]),
            node_limit: Some(0),
            ..SolveOptions::default()
        };
        let outcome = p.solve_with(options).unwrap();
        assert!(
            outcome.solution().is_none(),
            "violating hint must not be adopted"
        );

        let options = SolveOptions {
            warm_start: Some(vec![
                (members[0], 4.0),
                (members[1], 0.0),
                (members[2], 0.0),
            ]),
            node_limit: Some(0),
            ..SolveOptions::default()
        };
        let outcome = p.solve_with(options).unwrap();
        let sol = outcome.solution().expect("valid hint must be adopted");
        assert_eq!(sol.status(), SolutionStatus::Feasible);
        assert!(close(sol.objective(), 12.0));
        // Resuming without the budget reaches the true optimum.
        let sol = solution(outcome.resume_with(ResumeOptions::default()).unwrap());
        assert!(close(sol.objective(), 20.0));
    }

    #[test]
    fn node_limit_interrupts_then_resume_finishes_the_proof() {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let lam = pwl(&mut p);
        p.add_sos2_constraint(lam.iter().enumerate().map(|(i, &l)| (l, i as f64)));

        let options = SolveOptions {
            node_limit: Some(1),
            ..SolveOptions::default()
        };
        let mut outcome = p.solve_with(options).unwrap();
        let mut steps = 0;
        while !outcome.is_optimal() {
            steps += 1;
            assert!(steps < 100, "resume did not converge");
            outcome = outcome.resume().unwrap();
        }
        let sol = solution(outcome);
        assert_sos2_holds(&sol, &lam);
        assert!(close(sol.objective(), 2.0));
    }

    #[test]
    fn zero_time_limit_then_resume_reaches_the_sos_optimum() {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let members = sos1_capacity(&mut p);
        p.add_sos1_constraint(members.iter().map(|&m| (m, 1.0)));
        p.set_time_limit(Duration::ZERO);
        let outcome = p.solve().unwrap();
        assert!(!outcome.is_optimal());
        let sol = solution(outcome.resume_with(ResumeOptions::default()).unwrap());
        assert!(close(sol.objective(), 20.0));
        assert_sos1_holds(&sol, &members);
    }

    /// Randomised cross-check against enumeration on small dense models.
    #[test]
    fn random_small_models_match_enumeration() {
        let mut seed: u64 = 0x5eed_5eed;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut rnd = move |lo: f64, hi: f64| lo + (next() % 1000) as f64 / 1000.0 * (hi - lo);
        for trial in 0..40 {
            let n = 3 + (trial % 4); // 3..=6 members
            let rows = 1 + (trial % 3);
            let sos2 = trial % 2 == 1;
            let direction = if trial % 3 == 0 {
                OptimizationDirection::Minimize
            } else {
                OptimizationDirection::Maximize
            };
            let obj: Vec<f64> = (0..n).map(|_| rnd(-5.0, 5.0)).collect();
            let rowdata: Vec<(Vec<f64>, f64)> = (0..rows)
                .map(|_| ((0..n).map(|_| rnd(0.0, 3.0)).collect(), rnd(2.0, 8.0)))
                .collect();
            let build = |p: &mut Problem| -> Vec<Variable> {
                let vars: Vec<Variable> = obj.iter().map(|&c| p.add_var(c, (0.0, 4.0))).collect();
                for (coeffs, rhs) in &rowdata {
                    let terms: Vec<(Variable, f64)> =
                        vars.iter().zip(coeffs).map(|(&v, &c)| (v, c)).collect();
                    p.add_constraint(terms, ComparisonOp::Le, *rhs);
                }
                vars
            };
            let mut p = Problem::new(direction);
            let members = build(&mut p);
            let weighted: Vec<(Variable, f64)> = members
                .iter()
                .enumerate()
                .map(|(i, &m)| (m, i as f64))
                .collect();
            let want = if sos2 {
                p.add_sos2_constraint(weighted);
                sos2_oracle(&build, direction, n)
            } else {
                p.add_sos1_constraint(weighted);
                sos1_oracle(&build, direction, n)
            };
            let sol = solution(p.solve().unwrap());
            if sos2 {
                assert_sos2_holds(&sol, &members);
            } else {
                assert_sos1_holds(&sol, &members);
            }
            let want = want.expect("bounded nonneg models are feasible");
            assert!(
                close(sol.objective(), want),
                "trial {trial}: got {} want {want}",
                sol.objective()
            );
        }
    }
}
