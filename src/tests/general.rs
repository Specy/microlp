#[cfg(test)]
mod tests_general {
    use crate::solver::float_eq;
    use crate::*;

    use core::time::Duration;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn optimize() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(3.0, (12.0, f64::INFINITY));
        let v2 = problem.add_var(4.0, (5.0, f64::INFINITY));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 20.0);
        problem.add_constraint([(v2, -4.0), (v1, 1.0)], ComparisonOp::Ge, -20.0);

        let sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 12.0);
        assert_eq!(sol[v2], 8.0);
        assert_eq!(sol.objective(), 68.0);
    }

    #[test]
    fn empty_expr_constraints() {
        init();
        let trivial = [
            (LinearExpr::empty(), ComparisonOp::Eq, 0.0),
            (LinearExpr::empty(), ComparisonOp::Ge, -1.0),
            (LinearExpr::empty(), ComparisonOp::Le, 1.0),
        ];

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let _ = problem.add_var(1.0, (0.0, f64::INFINITY));
        for (expr, op, b) in trivial.iter().cloned() {
            problem.add_constraint(expr, op, b);
        }
        assert_eq!(problem.solve().map(|s| s.objective()), Ok(0.0));

        {
            let mut sol = problem.solve().unwrap();
            for (expr, op, b) in trivial.iter().cloned() {
                sol = sol.add_constraint(expr, op, b).unwrap();
            }
            assert_eq!(sol.objective(), 0.0);
        }

        let infeasible = [
            (LinearExpr::empty(), ComparisonOp::Eq, 12.0),
            (LinearExpr::empty(), ComparisonOp::Ge, 34.0),
            (LinearExpr::empty(), ComparisonOp::Le, -56.0),
        ];

        for (expr, op, b) in infeasible.iter().cloned() {
            let mut cloned = problem.clone();
            cloned.add_constraint(expr, op, b);
            assert_eq!(cloned.solve().map(|_| "solved"), Err(Error::Infeasible));
        }

        for (expr, op, b) in infeasible.iter().cloned() {
            let sol = problem.solve().unwrap().add_constraint(expr, op, b);
            assert_eq!(sol.map(|_| "solved"), Err(Error::Infeasible));
        }

        let _ = problem.add_var(-1.0, (0.0, f64::INFINITY));
        assert_eq!(problem.solve().map(|_| "solved"), Err(Error::Unbounded));
    }

    #[test]
    fn free_variables() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(1.0, (0.0, f64::INFINITY));
        let v2 = problem.add_var(2.0, (f64::NEG_INFINITY, f64::INFINITY));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 2.0);
        problem.add_constraint([(v1, 1.0), (v2, -1.0)], ComparisonOp::Ge, 0.0);

        let sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 2.0);
        assert_eq!(sol[v2], 2.0);
        assert_eq!(sol.objective(), 6.0);
    }

    #[test]
    fn fix_unfix_var() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(1.0, (0.0, 3.0));
        let v2 = problem.add_var(2.0, (0.0, 3.0));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 1.0);

        let orig_sol = problem.solve().unwrap();

        {
            let mut sol = orig_sol.clone().fix_var(v1, 0.5).unwrap();
            assert_eq!(sol[v1], 0.5);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 6.5);

            sol = sol.unfix_var(v1).0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }

        {
            let mut sol = orig_sol.clone().fix_var(v2, 2.5).unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 2.5);
            assert_eq!(sol.objective(), 6.5);

            sol = sol.unfix_var(v2).0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }
    }

    #[test]
    fn add_constraint() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let v1 = problem.add_var(2.0, (0.0, f64::INFINITY));
        let v2 = problem.add_var(1.0, (0.0, f64::INFINITY));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 2.0);

        let orig_sol = problem.solve().unwrap();

        {
            let sol = orig_sol
                .clone()
                .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                .unwrap();

            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 1.0);
            assert_eq!(sol.objective(), 3.0);
        }

        {
            let sol = orig_sol
                .clone()
                .fix_var(v2, 1.5)
                .unwrap()
                .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                .unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 1.5);
            assert_eq!(sol.objective(), 4.5);
        }

        {
            let sol = orig_sol
                .clone()
                .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Ge, 3.0)
                .unwrap();

            assert_eq!(sol[v1], 0.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 3.0);
        }
    }

    #[test]
    fn gomory_cut() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let v1 = problem.add_var(0.0, (0.0, f64::INFINITY));
        let v2 = problem.add_var(-1.0, (0.0, f64::INFINITY));
        problem.add_constraint([(v1, 3.0), (v2, 2.0)], ComparisonOp::Le, 6.0);
        problem.add_constraint([(v1, -3.0), (v2, 2.0)], ComparisonOp::Le, 0.0);

        let mut sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 1.0);
        assert_eq!(sol[v2], 1.5);
        assert_eq!(sol.objective(), -1.5);

        sol = sol.add_gomory_cut(v2).unwrap();
        assert!(f64::abs(sol[v1] - 2.0 / 3.0) < 1e-8);
        assert_eq!(sol[v2], 1.0);
        assert_eq!(sol.objective(), -1.0);

        sol = sol.add_gomory_cut(v1).unwrap();
        assert!(f64::abs(sol[v1] - 1.0) < 1e-8);
        assert_eq!(sol[v2], 1.0);
        assert_eq!(sol.objective(), -1.0);
    }

    fn cast_result_to_integers(vec: Vec<f64>) -> Vec<i64> {
        vec.into_iter()
            .map(|x| {
                let val = x.round() as i64;
                assert!(
                    f64::abs(x - val as f64) < 1e-5,
                    "Expected integer, got {}",
                    x
                );
                val
            })
            .collect()
    }

    #[test]
    fn knapsack_solve() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let weights = [10, 60, 30, 40, 30, 20, 20, 2];
        let values = [1, 10, 15, 40, 60, 90, 100, 15];
        let capacity = 102;
        let mut vars = vec![];
        for i in 0..weights.len() {
            let var = problem.add_binary_var(values[i] as f64);
            vars.push(var);
        }
        let entries = vars
            .iter()
            .map(|v| (*v, weights[v.0] as f64))
            .collect::<Vec<_>>();
        problem.add_constraint(&entries, ComparisonOp::Le, capacity as f64);
        let sol = problem.solve().unwrap();

        let values = vars.iter().map(|v| sol.var_value(*v)).collect::<Vec<_>>();
        assert_eq!(
            cast_result_to_integers(values),
            vec![0, 0, 1, 0, 1, 1, 1, 1]
        );
        assert_eq!(sol.objective(), 280.0);
    }

    #[test]
    fn dominating_set_solve() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let vars = [
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
            problem.add_binary_var(1.0),
        ];
        let rows = vec![
            vec![1, 1, 0, 1, 1, 0],
            vec![1, 1, 1, 1, 0, 0],
            vec![0, 1, 1, 1, 0, 0],
            vec![1, 1, 1, 1, 0, 0],
            vec![1, 0, 0, 0, 1, 0],
            vec![1, 0, 0, 0, 0, 1],
        ];
        for row in rows {
            problem.add_constraint(
                row.iter()
                    .enumerate()
                    .map(|(i, v)| (vars[i], *v as f64))
                    .collect::<Vec<_>>(),
                ComparisonOp::Ge,
                1.0,
            );
        }
        let sol = problem.solve().unwrap();
        let values = vars.iter().map(|v| sol.var_value(*v)).collect::<Vec<_>>();
        assert_eq!(cast_result_to_integers(values), vec![1, 0, 1, 0, 0, 0]);
        assert_eq!(sol.objective(), 2.0);
    }

    #[test]
    fn solve_milp() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);

        // Define variables with their objective coefficients
        let x = problem.add_var(50.0, (2.0, f64::INFINITY)); // x ≥ 0
        let y = problem.add_var(40.0, (0.0, 7.0)); // y ≥ 0
        let z = problem.add_integer_var(45.0, (0, i32::MAX)); // z ≥ 0 and integer
                                                              // Machine time constraint: 3x + 2y + z ≤ 20
        problem.add_constraint(&[(x, 3.0), (y, 2.0), (z, 1.0)], ComparisonOp::Le, 20.0);

        // Labor time constraint: 2x + y + 3z ≤ 15
        problem.add_constraint(&[(x, 2.0), (y, 1.0), (z, 3.0)], ComparisonOp::Le, 15.0);

        let sol = problem.solve().unwrap();

        assert_eq!(
            [sol.var_value(x), sol.var_value(y), sol.var_value(z)],
            [2.0, 6.5, 1.0]
        );
        assert_eq!(sol.objective(), 405.0);
    }

    #[test]
    fn solve_production_planning() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);

        // Number of time periods
        const PERIODS: usize = 4;

        // Production costs per unit for each period
        let prod_costs = [10.0, 12.0, 11.0, 14.0];

        // Holding costs per unit at the end of each period
        let holding_costs = [2.0, 2.0, 2.0, 2.0];

        // Setup costs for production in each period
        let setup_costs = [100.0, 100.0, 100.0, 100.0];

        // Demand for each period
        let demand = [50.0, 70.0, 90.0, 60.0];

        // Maximum production capacity per period
        let capacity = 120.0;

        // Production variables - amount to produce in each period
        let mut production = Vec::with_capacity(PERIODS);
        for i in 0..PERIODS {
            production.push(problem.add_var(prod_costs[i], (0.0, capacity)));
        }

        // Inventory variables - amount to hold at the end of each period
        let mut inventory = Vec::with_capacity(PERIODS);
        for i in 0..PERIODS {
            inventory.push(problem.add_var(holding_costs[i], (0.0, f64::INFINITY)));
        }

        // Setup variables - whether there is production in a period
        let mut setup = Vec::with_capacity(PERIODS);
        for i in 0..PERIODS {
            setup.push(problem.add_binary_var(setup_costs[i]));
        }

        // Initial inventory is 0
        let mut prev_inventory = problem.add_var(0.0, (0.0, 0.0));

        // Flow balance constraints and production-setup linking
        for i in 0..PERIODS {
            // Flow balance: prev_inventory + production[i] = demand[i] + inventory[i]
            problem.add_constraint(
                &[
                    (prev_inventory, 1.0),
                    (production[i], 1.0),
                    (inventory[i], -1.0),
                ],
                ComparisonOp::Eq,
                demand[i],
            );

            // Link production to setup: production[i] <= capacity * setup[i]
            problem.add_constraint(
                &[(production[i], 1.0), (setup[i], -capacity)],
                ComparisonOp::Le,
                0.0,
            );

            prev_inventory = inventory[i]
        }

        let sol = problem.solve().unwrap();

        assert!(
            float_eq(sol.objective(), 3440.0),
            "Expected 3440.0, got {}",
            sol.objective()
        );
    }

    #[test]
    fn time_limit_zero_returns_limit_error() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        let y = problem.add_var(2.0, (0.0, 3.0));
        problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

        // A zero duration guarantees the deadline is already passed before solving starts.
        problem.set_time_limit(Duration::ZERO);
        let result = problem.solve().unwrap();
        assert_eq!(result.stop_reason(), &StopReason::Limit);
    }

    #[test]
    fn time_limit_generous_succeeds() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        let y = problem.add_var(2.0, (0.0, 3.0));
        problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

        // A generous time limit should let a small problem solve without issue.
        problem.set_time_limit(Duration::from_secs(60));
        let sol = problem.solve().unwrap();
        assert_eq!(sol.objective(), 7.0);
        assert_eq!(sol[x], 1.0);
        assert_eq!(sol[y], 3.0);
    }

    #[test]
    fn time_limit_zero_integer_returns_limit_error() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let weights = [10, 60, 30, 40, 30, 20, 20, 2];
        let values = [1, 10, 15, 40, 60, 90, 100, 15];
        let capacity = 102;
        let mut vars = vec![];
        for i in 0..weights.len() {
            let var = problem.add_binary_var(values[i] as f64);
            vars.push(var);
        }
        let entries = vars
            .iter()
            .map(|v| (*v, weights[v.0] as f64))
            .collect::<Vec<_>>();
        problem.add_constraint(&entries, ComparisonOp::Le, capacity as f64);

        problem.set_time_limit(Duration::ZERO);
        let result = problem.solve().unwrap();
        assert_eq!(result.stop_reason(), &StopReason::Limit);
    }

    #[test]
    fn solve_big_m() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Minimize);

        let m = 1.0e9;

        // Define variables with their objective coefficients
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        let b = problem.add_binary_var(-1.0);

        problem.add_constraint([(x, 1.0)], ComparisonOp::Ge, 5.0);
        problem.add_constraint([(b, -m), (x, 1.0)], ComparisonOp::Le, 10.0);
        problem.add_constraint([(b, -m), (x, 1.0)], ComparisonOp::Ge, -m + 10.0);

        let sol = problem.solve().unwrap();

        assert_eq!([sol.var_value(x), sol.var_value(b)], [5.0, 0.0]);
        assert_eq!(sol.objective().round(), 5.0);
    }
}
