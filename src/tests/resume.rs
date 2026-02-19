#[cfg(test)]
mod tests_resume {
    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    use crate::{solver::float_eq, *};

    /// Deterministic pseudo-random number generator for reproducible test data.
    /// Uses a simple xorshift64 algorithm.
    struct SimpleRng {
        state: u64,
    }

    impl SimpleRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        /// Returns a value in [lo, hi] (inclusive).
        fn next_range(&mut self, lo: u64, hi: u64) -> u64 {
            lo + (self.next_u64() % (hi - lo + 1))
        }
    }

    /// Builds a large multi-dimensional knapsack problem that is hard enough
    /// to take several seconds for the branch-and-bound MIP solver.
    ///
    /// Parameters are tuned so the LP relaxation has many fractional variables
    /// and the solver must explore a large B&B tree.
    fn build_complex_knapsack() -> (Problem, Vec<Variable>) {
        let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_1234);

        let num_items = 110;
        let num_knapsack_constraints = 20;

        let mut problem = Problem::new(OptimizationDirection::Maximize);

        // Create binary variables with pseudo-random objective coefficients.
        // Use a wide range so the LP relaxation is loose and many variables are fractional.
        let mut vars = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            let coeff = rng.next_range(10, 500) as f64;
            vars.push(problem.add_binary_var(coeff));
        }

        // Add many knapsack constraints with pseudo-random weights.
        // Capacity is set to ~28% of total weight — tight enough to force heavy branching.
        for _ in 0..num_knapsack_constraints {
            let weights: Vec<f64> = (0..num_items)
                .map(|_| rng.next_range(1, 80) as f64)
                .collect();
            let total_weight: f64 = weights.iter().sum();
            let capacity = (total_weight * 0.28).floor();

            let entries: Vec<(Variable, f64)> = vars
                .iter()
                .zip(weights.iter())
                .map(|(v, w)| (*v, *w))
                .collect();
            problem.add_constraint(&entries, ComparisonOp::Le, capacity);
        }

        // Add "conflict" constraints: pairs of items that can't both be selected.
        // These create many disjunctions in the B&B tree.
        for i in (0..num_items - 1).step_by(2) {
            let j = (i + 1 + (rng.next_range(1, 8) as usize)) % num_items;
            if i != j {
                problem.add_constraint(&[(vars[i], 1.0), (vars[j], 1.0)], ComparisonOp::Le, 1.0);
            }
        }

        // Add set-packing constraints over overlapping groups of 5-7 variables.
        // At most 2 from each group can be selected — creates many fractional relaxations.
        for start in (0..num_items - 7).step_by(5) {
            let group_size = 5 + (rng.next_range(0, 2) as usize); // 5, 6, or 7
            let end = (start + group_size).min(num_items);
            let entries: Vec<(Variable, f64)> = (start..end).map(|idx| (vars[idx], 1.0)).collect();
            problem.add_constraint(&entries, ComparisonOp::Le, 2.0);
        }

        // Add "coverage" constraints: at least 1 out of every group of 10 must be picked.
        // This conflicts with tight capacity and forces exploration of more branches.
        for start in (0..num_items - 10).step_by(12) {
            let entries: Vec<(Variable, f64)> =
                (start..start + 10).map(|idx| (vars[idx], 1.0)).collect();
            problem.add_constraint(&entries, ComparisonOp::Ge, 1.0);
        }

        (problem, vars)
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "test is too slow in debug mode")]
    fn resume_produces_same_result_as_unlimited() {
        init();

        // ── 1. Solve without any time limit ──────────────────────────────────
        let (problem_unlimited, vars_unlimited) = build_complex_knapsack();

        let t0 = std::time::Instant::now();
        let sol_unlimited = problem_unlimited.solve().unwrap();
        let elapsed_unlimited = t0.elapsed();

        assert_eq!(
            sol_unlimited.stop_reason(),
            &StopReason::Finished,
            "Unlimited solve should finish"
        );

        let values_unlimited: Vec<f64> = vars_unlimited
            .iter()
            .map(|v| sol_unlimited.var_value(*v))
            .collect();
        let obj_unlimited = sol_unlimited.objective();

        // ── 2. Solve the same problem with repeated 1-second time limits ─────
        let (mut problem_limited, vars_limited) = build_complex_knapsack();
        problem_limited.set_time_limit(Duration::from_secs(1));

        let t1 = std::time::Instant::now();
        let mut sol_limited = problem_limited.solve().unwrap();

        let mut resume_count = 0u32;
        while *sol_limited.stop_reason() == StopReason::Limit {
            resume_count += 1;
            sol_limited = sol_limited.resume(Some(Duration::from_secs(1))).unwrap();
        }
        let elapsed_limited = t1.elapsed();

        assert_eq!(
            sol_limited.stop_reason(),
            &StopReason::Finished,
            "Resumed solve should eventually finish"
        );

        println!(
            "Unlimited solve duration: {:.3}s",
            elapsed_unlimited.as_secs_f64()
        );
        println!(
            "Resumed solve duration: {:.3}s",
            elapsed_limited.as_secs_f64()
        );

        let values_limited: Vec<f64> = vars_limited
            .iter()
            .map(|v| sol_limited.var_value(*v))
            .collect();
        let obj_limited = sol_limited.objective();
        // ── 3. Compare results ───────────────────────────────────────────────
        assert!(
            float_eq(obj_unlimited, obj_limited),
            "Objectives differ! unlimited = {}, resumed = {}",
            obj_unlimited,
            obj_limited
        );

        for (i, (a, b)) in values_unlimited
            .iter()
            .zip(values_limited.iter())
            .enumerate()
        {
            assert!(
                float_eq(*a, *b),
                "Variable {} differs: unlimited = {}, resumed = {}",
                i,
                a,
                b
            );
        }

        assert!(
            resume_count >= 1,
            "Expected at least 1 resume call, got {}",
            resume_count
        );
    }

    /// Builds a large dense LP (no integer variables) that is hard enough
    /// to take several seconds for the simplex solver.
    ///
    /// Uses many variables and dense constraints with pseudo-random coefficients
    /// so the simplex method must perform many pivots.
    fn build_large_lp() -> (Problem, Vec<Variable>) {
        let mut rng = SimpleRng::new(0xCAFE_BABE_1337_7331);

        let num_vars = 1500;
        let num_constraints = 1200;

        let mut problem = Problem::new(OptimizationDirection::Maximize);

        // Create continuous variables with pseudo-random objective coefficients
        // and bounded ranges.
        let mut vars = Vec::with_capacity(num_vars);
        for _ in 0..num_vars {
            let coeff = (rng.next_range(1, 1000) as f64) / 100.0;
            let upper = (rng.next_range(5, 50) as f64) / 10.0;
            vars.push(problem.add_var(coeff, (0.0, upper)));
        }

        // Add dense constraints with pseudo-random coefficients.
        // Each constraint involves all variables (dense) to maximise pivot work.
        // Capacity is set tight enough that the LP isn't trivial.
        for _ in 0..num_constraints {
            let coeffs: Vec<f64> = (0..num_vars)
                .map(|_| (rng.next_range(0, 200) as f64) / 100.0)
                .collect();
            let total: f64 = coeffs.iter().sum();
            let capacity = (total * 0.15).floor();

            let entries: Vec<(Variable, f64)> = vars
                .iter()
                .zip(coeffs.iter())
                .filter(|(_, c)| **c > 0.0)
                .map(|(v, c)| (*v, *c))
                .collect();
            problem.add_constraint(&entries, ComparisonOp::Le, capacity);
        }

        (problem, vars)
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "test is too slow in debug mode")]
    fn resume_real_variables_produces_same_result_as_unlimited() {
        init();

        // ── 1. Solve without any time limit ──────────────────────────────────
        let (problem_unlimited, vars_unlimited) = build_large_lp();

        let t0 = std::time::Instant::now();
        let sol_unlimited = problem_unlimited.solve().unwrap();
        let elapsed_unlimited = t0.elapsed();

        assert_eq!(
            sol_unlimited.stop_reason(),
            &StopReason::Finished,
            "Unlimited solve should finish"
        );

        let values_unlimited: Vec<f64> = vars_unlimited
            .iter()
            .map(|v| sol_unlimited.var_value(*v))
            .collect();
        let obj_unlimited = sol_unlimited.objective();

        println!(
            "LP unlimited solve: objective = {:.6}, time = {:.3}s",
            obj_unlimited,
            elapsed_unlimited.as_secs_f64()
        );

        // ── 2. Solve the same problem with repeated short time limits ────────
        let (mut problem_limited, vars_limited) = build_large_lp();
        problem_limited.set_time_limit(Duration::from_millis(100));

        let t1 = std::time::Instant::now();
        let mut sol_limited = problem_limited.solve().unwrap();

        let mut resume_count = 0u32;
        while *sol_limited.stop_reason() == StopReason::Limit {
            resume_count += 1;
            sol_limited = sol_limited
                .resume(Some(Duration::from_millis(100)))
                .unwrap();
        }
        let elapsed_limited = t1.elapsed();

        assert_eq!(
            sol_limited.stop_reason(),
            &StopReason::Finished,
            "Resumed LP solve should eventually finish"
        );

        let values_limited: Vec<f64> = vars_limited
            .iter()
            .map(|v| sol_limited.var_value(*v))
            .collect();
        let obj_limited = sol_limited.objective();

        println!(
            "LP resumed solve:  objective = {:.6}, time = {:.3}s, resumes = {}",
            obj_limited,
            elapsed_limited.as_secs_f64(),
            resume_count
        );

        // ── 3. Compare results ───────────────────────────────────────────────
        assert!(
            float_eq(obj_unlimited, obj_limited),
            "LP objectives differ! unlimited = {}, resumed = {}",
            obj_unlimited,
            obj_limited
        );

        for (i, (a, b)) in values_unlimited
            .iter()
            .zip(values_limited.iter())
            .enumerate()
        {
            assert!(
                float_eq(*a, *b),
                "LP variable {} differs: unlimited = {}, resumed = {}",
                i,
                a,
                b
            );
        }

        assert!(
            resume_count >= 1,
            "Expected at least 1 resume call, got {}",
            resume_count
        );
    }
}
