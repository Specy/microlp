#[cfg(test)]
mod tests_general {
    use crate::solver::float_eq;
    use crate::*;

    use core::time::Duration;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn solution(outcome: SolveOutcome) -> Solution {
        outcome
            .into_solution()
            .expect("an unlimited bounded feasible solve must return a solution")
    }

    #[test]
    fn optimize() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(3.0, (12.0, f64::INFINITY));
        let v2 = problem.add_var(4.0, (5.0, f64::INFINITY));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 20.0);
        problem.add_constraint([(v2, -4.0), (v1, 1.0)], ComparisonOp::Ge, -20.0);

        let sol = solution(problem.solve().unwrap());
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
        assert_eq!(solution(problem.solve().unwrap()).objective(), 0.0);

        {
            let mut sol = solution(problem.solve().unwrap());
            for (expr, op, b) in trivial.iter().cloned() {
                sol = solution(sol.add_constraint(expr, op, b).unwrap());
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
            let sol = solution(problem.solve().unwrap()).add_constraint(expr, op, b);
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

        let sol = solution(problem.solve().unwrap());
        assert_eq!(sol[v1], 2.0);
        assert_eq!(sol[v2], 2.0);
        assert_eq!(sol.objective(), 6.0);
    }

    /// Rows whose data alone decide them get the same verdict on every path
    /// into the engine (pure LP and MIP, presolve on and off, and added to a
    /// solved LP), under the one contract: an empty row has activity zero
    /// and is held to the feasibility tolerance like any row; an infinite
    /// right-hand side on the side a row leaves open is no constraint, and
    /// on the side it constrains, one no point meets; a coefficient or
    /// right-hand side that is not a number, or an infinite coefficient,
    /// makes a row no point satisfies. (Presolve used to drop NaN rows and
    /// rows bounded by an infinity, the engine accepted `x ≤ -∞` under an
    /// infinite tolerance, and the two disagreed on empty rows.)
    #[test]
    fn rows_whose_data_decide_them_get_one_verdict_on_every_path() {
        init();
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        // (coefficient of x, or none for an empty row; sense; rhs; holds)
        let cases = [
            (None, ComparisonOp::Eq, 1e-9, true),
            (None, ComparisonOp::Eq, 0.1 + 0.2 - 0.3, true),
            (None, ComparisonOp::Le, -5e-8, true),
            (None, ComparisonOp::Ge, 5e-8, true),
            (None, ComparisonOp::Eq, 1e-6, false),
            (None, ComparisonOp::Le, -1e-6, false),
            (Some(1.0), ComparisonOp::Le, inf, true),
            (Some(1.0), ComparisonOp::Ge, -inf, true),
            (Some(1.0), ComparisonOp::Le, -inf, false),
            (Some(1.0), ComparisonOp::Ge, inf, false),
            (Some(1.0), ComparisonOp::Eq, inf, false),
            (Some(1.0), ComparisonOp::Le, nan, false),
            (Some(nan), ComparisonOp::Le, 1.0, false),
            (Some(inf), ComparisonOp::Le, 1.0, false),
        ];
        for (coeff, op, rhs, holds) in cases {
            let expected = if holds {
                Ok(0.0)
            } else {
                Err(Error::Infeasible)
            };
            let row = |x: Variable| -> LinearExpr {
                let mut expr = LinearExpr::empty();
                if let Some(c) = coeff {
                    expr.add(x, c);
                }
                expr
            };
            for milp in [false, true] {
                for presolve in [true, false] {
                    let mut problem = Problem::new(OptimizationDirection::Minimize);
                    let x = problem.add_var(1.0, (0.0, 10.0));
                    if milp {
                        problem.add_integer_var(1.0, (0, 0));
                    }
                    problem.add_constraint(row(x), op, rhs);
                    let got = problem
                        .solve_with(SolveOptions {
                            presolve,
                            ..SolveOptions::default()
                        })
                        .map(|outcome| solution(outcome).objective());
                    assert_eq!(
                        got, expected,
                        "{coeff:?} x {op:?} {rhs}: milp {milp}, presolve {presolve}"
                    );
                }
            }
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(1.0, (0.0, 10.0));
            let got = solution(problem.solve().unwrap())
                .add_constraint(row(x), op, rhs)
                .map(|outcome| solution(outcome).objective());
            assert_eq!(
                got, expected,
                "{coeff:?} x {op:?} {rhs}: added to a solved LP"
            );
        }

        // A bound that is not a number: the engine's verdict with presolve too.
        for milp in [false, true] {
            for presolve in [true, false] {
                let mut problem = Problem::new(OptimizationDirection::Minimize);
                let x = problem.add_var(1.0, (nan, 10.0));
                if milp {
                    problem.add_integer_var(1.0, (0, 0));
                }
                problem.add_constraint([(x, 1.0)], ComparisonOp::Eq, 3.0);
                let got = problem.solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                });
                assert_eq!(
                    got.map(|_| "solved"),
                    Err(Error::Infeasible),
                    "NaN bound: milp {milp}, presolve {presolve}"
                );
            }
        }
    }

    /// A free var whose objective coefficient is zero is optimal wherever it
    /// sits: no move of it changes the objective. Counting it as dual
    /// infeasible made the primal simplex enter it along a direction that
    /// improves nothing, and with no bound in the way the step was infinite:
    /// a free var in no row made a bounded model `Unbounded`, one in a row
    /// drove the values to infinity (`InternalError`), and two of them in one
    /// row swapped in and out of the basis forever.
    #[test]
    fn free_variables_with_a_zero_cost_are_optimal_anywhere() {
        init();
        let inf = f64::INFINITY;
        {
            // In no row: the model is bounded by y alone.
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let _x = problem.add_var(0.0, (-inf, inf));
            let y = problem.add_var(1.0, (1.0, 10.0));
            problem.add_constraint([(y, 1.0)], ComparisonOp::Ge, 2.0);
            assert_eq!(solution(problem.solve().unwrap()).objective(), 2.0);
        }
        {
            // In a row whose slack is unbounded in the direction x can move.
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(0.0, (-inf, inf));
            let y = problem.add_var(1.0, (1.0, 10.0));
            problem.add_constraint([(x, 1.0), (y, -1.0)], ComparisonOp::Le, 0.0);
            let sol = solution(problem.solve().unwrap());
            assert_eq!(sol.objective(), 1.0);
            assert!(sol[x].is_finite() && sol[x] <= sol[y] + 1e-9);
        }
        {
            // Linked to another free var by an equality.
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x = problem.add_var(0.0, (-inf, inf));
            let y = problem.add_var(1.0, (0.0, 5.0));
            let z = problem.add_var(0.0, (-inf, inf));
            problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 7.0);
            problem.add_constraint([(x, 1.0), (z, -1.0)], ComparisonOp::Eq, 0.0);
            assert_eq!(solution(problem.solve().unwrap()).objective(), 5.0);
        }
        {
            // Two of them alone in a row: this one never returned.
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x = problem.add_var(0.0, (-inf, inf));
            let y = problem.add_var(0.0, (-inf, inf));
            problem.add_constraint([(x, -4.0), (y, -3.0)], ComparisonOp::Le, 9.0);
            let mut options = SolveOptions::default();
            options.time_limit = Some(Duration::from_secs(10));
            let sol = solution(problem.solve_with(options).unwrap());
            assert_eq!(sol.objective(), 0.0);
            assert!(-4.0 * sol[x] - 3.0 * sol[y] <= 9.0 + 1e-9);
        }
    }

    /// An unbounded LP (HiGHS agrees) whose improving ray had exactly one
    /// row that could "block" it, through a round-off entry of 1.4e-15 of the
    /// column's largest. The primal ratio test took it as the pivot, with a
    /// step of 1.6e15, into a basis the factorization then refused
    /// (`InternalError("Singular matrix")`): its row and column computations
    /// were 2% apart, which the agreement check let through as agreement
    /// within round-off although the element was itself at that level.
    #[test]
    fn unbounded_ray_is_not_blocked_by_a_round_off_entry() {
        init();
        let inf = f64::INFINITY;
        for presolve in [true, false] {
            let mut p = Problem::new(OptimizationDirection::Minimize);
            let x1 = p.add_var(2.75, (-inf, 2.009));
            let x2 = p.add_var(0.0, (-inf, 0.68));
            let x3 = p.add_var(0.0, (-3.697, inf));
            let x4 = p.add_var(0.0, (-inf, inf));
            let x5 = p.add_var(0.0, (-inf, inf));
            let x7 = p.add_var(0.0, (-inf, inf));
            p.add_constraint(
                [(x1, 9.67), (x2, -2.43), (x7, 5.17)],
                ComparisonOp::Le,
                -19.978259999999995,
            );
            p.add_constraint(
                [
                    (x2, -6.77),
                    (x3, -0.6),
                    (x4, -6.05),
                    (x5, -0.12),
                    (x7, 8.58),
                ],
                ComparisonOp::Ge,
                -11.405089999999994,
            );
            p.add_constraint([(x3, -6.9)], ComparisonOp::Ge, 17.6093);
            p.add_constraint(
                [(x3, -3.62), (x4, -4.87), (x5, -0.14)],
                ComparisonOp::Le,
                4.095199999999999,
            );
            p.add_constraint(
                [
                    (x1, -8.75),
                    (x2, 2.39),
                    (x3, -0.75),
                    (x4, 6.74),
                    (x5, 6.99),
                    (x7, 1.13),
                ],
                ComparisonOp::Le,
                13.954310000000001,
            );
            p.add_constraint(
                [(x1, 4.3), (x2, 0.33), (x4, 1.94), (x5, 7.15)],
                ComparisonOp::Ge,
                14.202540000000003,
            );
            let got = p.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            });
            assert_eq!(
                got.map(|_| "solved"),
                Err(Error::Unbounded),
                "presolve {presolve}"
            );
        }
    }

    /// Found by differential fuzzing against HiGHS: an improving column whose
    /// only blocking row had its basic var sitting past its bound within
    /// tolerance. The engine refused the column as "not an improving
    /// direction", since setting the leaving var onto its bound would have
    /// moved the entering var backwards out of its own bounds, and reported
    /// Optimal 3e-3 short of the optimum (-6.19413 for -6.19720). In a
    /// degenerate step the leaving var now leaves where it sits.
    #[test]
    fn a_degenerate_step_past_a_bound_does_not_stop_the_simplex() {
        init();
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x0 = p.add_var(0.57, (f64::NEG_INFINITY, -1.979));
        let x1 = p.add_var(-2.65, (0.125, 1.125));
        let x2 = p.add_var(2.62, (0.848, 2.848));
        let x3 = p.add_var(0.36, (-2.887, 1.113));
        let x4 = p.add_var(-1.54, (-5.2620000000000005, -2.262));
        let x5 = p.add_var(0.0, (-1.906, -0.9059999999999999));
        let x6 = p.add_var(-2.03, (0.7839999999999998, 3.784));
        p.add_constraint(
            [(x0, -6.076433694637136), (x3, 0.08670910052272117)],
            ComparisonOp::Le,
            32.09094329291193,
        );
        p.add_constraint(
            [
                (x0, -88.79547469052551),
                (x1, 551.2716307241976),
                (x2, -0.0017178835049954455),
                (x3, -0.07756659223302026),
                (x4, 0.0710803378234343),
                (x6, -0.015061521859049312),
            ],
            ComparisonOp::Le,
            512.9487571961115,
        );
        p.add_constraint(
            [
                (x0, 0.0002492743006467558),
                (x2, 0.7628838231891608),
                (x4, 0.00011242224615839288),
            ],
            ComparisonOp::Eq,
            0.6454300462006779,
        );
        p.add_constraint(
            [
                (x0, 0.008507916026172499),
                (x1, 0.00012974266836798682),
                (x5, 0.00028865653400840866),
                (x6, -0.024549509136579904),
            ],
            ComparisonOp::Eq,
            -0.13579021798740526,
        );
        for presolve in [true, false] {
            let sol = solution(
                p.solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                })
                .unwrap(),
            );
            assert!(
                (sol.objective() - -6.1972020386339866).abs() < 1e-6,
                "presolve {presolve}: objective {}",
                sol.objective()
            );
        }
        let _ = (x0, x1, x2, x3, x4, x5, x6);
    }

    /// The same on the MIP path, where it matters twice: in the node LPs, and
    /// in the zero-objective search that classifies an unbounded relaxation,
    /// which makes every var a zero-cost one.
    #[test]
    fn free_variables_with_a_zero_cost_in_a_milp() {
        init();
        let inf = f64::INFINITY;
        {
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let n = problem.add_integer_var(1.0, (0, 10));
            let _f = problem.add_var(0.0, (-inf, inf));
            problem.add_constraint([(n, 2.0)], ComparisonOp::Ge, 3.0);
            assert_eq!(solution(problem.solve().unwrap()).objective(), 2.0);
        }
        {
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let z = problem.add_integer_var(1.0, (0, 10));
            let w = problem.add_var(0.0, (-inf, inf));
            problem.add_constraint([(w, 1.0), (z, -1.0)], ComparisonOp::Le, 1.0);
            problem.add_constraint([(z, 1.0)], ComparisonOp::Le, 2.5);
            assert_eq!(solution(problem.solve().unwrap()).objective(), 2.0);
        }
        {
            // The relaxation is unbounded (y is free and pushes down), and any
            // integer point proves the MILP unbounded.
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_binary_var(0.0);
            let y = problem.add_var(1.0, (-inf, inf));
            problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 1.0);
            assert_eq!(problem.solve().map(|_| "solved"), Err(Error::Unbounded));
        }
    }

    #[test]
    fn fix_unfix_var() {
        init();
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var(1.0, (0.0, 3.0));
        let v2 = problem.add_var(2.0, (0.0, 3.0));
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 1.0);

        let orig_sol = solution(problem.solve().unwrap());

        {
            let mut sol = solution(orig_sol.clone().fix_var(v1, 0.5).unwrap());
            assert_eq!(sol[v1], 0.5);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 6.5);

            sol = solution(sol.unfix_var(v1).unwrap().0);
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }

        {
            let mut sol = solution(orig_sol.clone().fix_var(v2, 2.5).unwrap());
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 2.5);
            assert_eq!(sol.objective(), 6.5);

            sol = solution(sol.unfix_var(v2).unwrap().0);
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }
    }

    #[test]
    fn fix_var_rejects_non_finite_values() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(1.0, (0.0, 10.0));
            let solution = solution(problem.solve().unwrap());
            assert!(
                matches!(solution.fix_var(x, invalid), Err(Error::Infeasible)),
                "non-finite fix {invalid} was not rejected"
            );
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

        let orig_sol = solution(problem.solve().unwrap());

        {
            let sol = solution(
                orig_sol
                    .clone()
                    .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                    .unwrap(),
            );

            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 1.0);
            assert_eq!(sol.objective(), 3.0);
        }

        {
            let fixed = solution(orig_sol.clone().fix_var(v2, 1.5).unwrap());
            let sol = solution(
                fixed
                    .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                    .unwrap(),
            );
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 1.5);
            assert_eq!(sol.objective(), 4.5);
        }

        {
            let sol = solution(
                orig_sol
                    .clone()
                    .add_constraint([(v1, -1.0), (v2, 1.0)], ComparisonOp::Ge, 3.0)
                    .unwrap(),
            );

            assert_eq!(sol[v1], 0.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 3.0);
        }
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
        let sol = solution(problem.solve().unwrap());

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
        let sol = solution(problem.solve().unwrap());
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

        let sol = solution(problem.solve().unwrap());

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

        let sol = solution(problem.solve().unwrap());

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
        assert!(matches!(&result, SolveOutcome::Interrupted(_)));
        assert_eq!(result.termination_reason(), TerminationReason::TimeLimit);
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
        let sol = solution(problem.solve().unwrap());
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
        assert!(matches!(&result, SolveOutcome::Interrupted(_)));
        assert_eq!(result.termination_reason(), TerminationReason::TimeLimit);
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

        let sol = solution(problem.solve().unwrap());

        assert_eq!([sol.var_value(x), sol.var_value(b)], [5.0, 0.0]);
        assert_eq!(sol.objective().round(), 5.0);
    }

    /// A model from `(objective, lower, upper, integer)` vars and
    /// `(terms, sense, rhs)` rows, as the differential fuzzing writes them.
    fn fuzz_model(
        direction: OptimizationDirection,
        vars: &[(f64, f64, f64, bool)],
        rows: &[(&[(usize, f64)], ComparisonOp, f64)],
    ) -> Problem {
        let mut problem = Problem::new(direction);
        let vars: Vec<Variable> = vars
            .iter()
            .map(|&(obj, lo, hi, integer)| {
                if integer {
                    problem.add_integer_var(obj, (lo as i32, hi as i32))
                } else {
                    problem.add_var(obj, (lo, hi))
                }
            })
            .collect();
        for &(terms, op, rhs) in rows {
            let terms: Vec<(Variable, f64)> = terms.iter().map(|&(v, a)| (vars[v], a)).collect();
            problem.add_constraint(&terms[..], op, rhs);
        }
        problem
    }

    /// Infeasible models (HiGHS agrees) whose dual simplex reaches a row
    /// whose only helping entry is round-off: 1.1e-16, the residue of two
    /// parallel rows cancelling, beside a slack's entry of 1.05 that cannot
    /// help. Every entry of a row comes from the same transposed solve, so
    /// the round-off scale of the row is taken over the entries that cannot
    /// help too; taken over the helping entries alone, or dropped from the
    /// fallback to vars that absorb the move, it let the residue through as
    /// a pivot, with a step of 4.5e15, into a basis the factorization
    /// refused (`InternalError("Singular matrix")`).
    #[test]
    fn infeasible_row_is_not_repaired_through_a_round_off_entry() {
        init();
        let inf = f64::INFINITY;
        use ComparisonOp::{Eq, Ge, Le};
        let models = [
            fuzz_model(
                OptimizationDirection::Maximize,
                &[
                    (3.0, -0.20697761674377824, 1.913871936096928, false),
                    (0.0, -20.234403244297035, -9.584280022528944, false),
                    (0.0, -9.270817452147654, 9.896834943606258, false),
                ],
                &[
                    (&[(0, 2.302285)], Ge, 2.0886425656280156),
                    (&[(1, -8.0), (2, 3.8)], Ge, 41.44513386207047),
                    (
                        &[(0, 9.917117), (1, -7.564712), (2, 0.0)],
                        Le,
                        81.49916805098938,
                    ),
                    (&[(0, 4.60457)], Ge, 4.177285131256031),
                    (&[(0, 8.734751)], Le, 7.454648431432122),
                    (&[(1, -4.0), (2, 1.9)], Ge, 20.722566931035235),
                ],
            ),
            fuzz_model(
                OptimizationDirection::Maximize,
                &[
                    (-3.0, -inf, -2.0, false),
                    (-3.0, 0.0, 14.0, false),
                    (2.0, 0.0, 18.0, false),
                    (-5.0, -inf, inf, false),
                    (-4.0, -inf, inf, false),
                    (1.0, 2.0, 3.0, true),
                ],
                &[
                    (&[(0, 3.0), (2, 2.0), (3, 5.0), (4, 2.0)], Le, 8.0),
                    (&[(0, -3.0), (1, 3.0), (5, -5.0)], Ge, -8.0),
                    (
                        &[(0, 5.0), (2, -4.0), (3, 4.0), (4, -3.0), (5, 2.0)],
                        Eq,
                        0.0,
                    ),
                    (&[(0, 1.0), (1, -3.0), (3, 5.0), (4, -2.0)], Le, -3.0),
                    (&[(0, -2.0), (4, -5.0)], Eq, 17.0),
                    (&[(4, 2.0)], Ge, 7.0),
                ],
            ),
            fuzz_model(
                OptimizationDirection::Minimize,
                &[
                    (-2.0, 0.0, 6.0, false),
                    (1.0, 4.0, 7.0, false),
                    (-4.0, 0.0, inf, false),
                    (3.0, 0.0, inf, false),
                    (-4.0, 0.0, 18.0, false),
                    (-1.0, 0.0, inf, false),
                    (-5.0, 0.0, 4.0, false),
                    (3.0, 0.0, 13.0, false),
                ],
                &[
                    (
                        &[
                            (0, -5.0),
                            (1, -3.0),
                            (2, -5.0),
                            (3, -5.0),
                            (4, 2.0),
                            (5, 1.0),
                            (7, -5.0),
                        ],
                        Le,
                        5.0,
                    ),
                    (
                        &[
                            (0, -5.0),
                            (1, -5.0),
                            (2, 1.0),
                            (3, -2.0),
                            (4, 2.0),
                            (5, -3.0),
                        ],
                        Eq,
                        10.0,
                    ),
                    (
                        &[(0, 5.0), (2, -5.0), (3, -2.0), (4, -2.0), (5, -3.0)],
                        Ge,
                        2.0,
                    ),
                    (
                        &[(1, -2.0), (3, 2.0), (4, -1.0), (5, 5.0), (6, 1.0), (7, 2.0)],
                        Ge,
                        3.0,
                    ),
                    (
                        &[
                            (0, -1.0),
                            (1, 3.0),
                            (2, -3.0),
                            (3, 2.0),
                            (5, -4.0),
                            (6, 2.0),
                            (7, -2.0),
                        ],
                        Eq,
                        12.0,
                    ),
                    (
                        &[(1, -5.0), (3, 1.0), (5, -3.0), (6, -5.0), (7, 5.0)],
                        Le,
                        -8.0,
                    ),
                ],
            ),
        ];
        for (k, problem) in models.iter().enumerate() {
            for presolve in [true, false] {
                let got = problem.solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                });
                assert_eq!(
                    got.map(|_| "solved"),
                    Err(Error::Infeasible),
                    "model {k}, presolve {presolve}"
                );
            }
        }
    }

    /// An infeasible LP of the differential fuzz (`int_lp-155`, HiGHS
    /// agrees) whose certificate, without presolve, combines three rows and
    /// leaves the free basic `x2` a residual of the multipliers, zero for the
    /// exact row of the basis inverse: `1.2e-32` of its terms from the row
    /// refined once. On an infinite bound no count of it exists, and any
    /// count but zero leaves the certificate's range unbounded: counted as it
    /// was, it voided the proof, and the solve failed loudly. The second step
    /// of the refinement leaves it exactly zero, and the concession for a
    /// coefficient the multipliers' error could make zero would cover it
    /// too. With presolve the row that proves it alone, `-4 x0 - 4 x1 >= 4`
    /// over `x0, x1 >= 0`, is found before the engine runs.
    #[test]
    fn a_basic_free_vars_residual_does_not_void_a_proof() {
        init();
        let inf = f64::INFINITY;
        use ComparisonOp::{Eq, Ge, Le};
        let problem = fuzz_model(
            OptimizationDirection::Maximize,
            &[
                (4.0, 0.0, inf, false),
                (3.0, 0.0, inf, false),
                (2.0, -inf, inf, false),
                (3.0, -inf, 0.0, false),
            ],
            &[
                (&[(2, 3.0), (3, -1.0)], Eq, -6.0),
                (&[(0, 5.0), (2, 5.0), (3, 1.0)], Ge, 6.0),
                (&[(0, -4.0), (1, -4.0)], Ge, 4.0),
                (&[(0, -4.0), (2, -5.0), (3, 2.0)], Le, 20.0),
            ],
        );
        for presolve in [true, false] {
            let got = problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            });
            assert_eq!(
                got.map(|_| "solved"),
                Err(Error::Infeasible),
                "presolve {presolve}"
            );
        }
    }

    /// An infeasible LP of the differential fuzz (`int_lp-970`, HiGHS
    /// agrees) whose certificate leaves out a row whose slack is non-basic:
    /// that slack's entry in the leaving row is zero, and the refined row of
    /// the basis inverse carries it as noise (`-7.9e-31` refined once), whose
    /// sign asks for the slack's infinite side. With the row left out, as
    /// HiGHS leaves it out (without that the proof failed), the basic `x0`,
    /// unbounded above, keeps a residual: `3.7e-31` of its terms from the row
    /// refined once, zero from the row refined twice. A rule that withdrew
    /// the concession from every var of a row left out while its slack is
    /// non-basic, taking such a slack's multiplier for the model's own, left
    /// the proof on the once-refined row void and the solve a loud error,
    /// with presolve on and off; whether a coefficient counts as zero is
    /// judged by the error of the multipliers instead, which such a noise
    /// lies within.
    #[test]
    fn a_residual_left_by_a_row_left_out_does_not_void_a_proof() {
        init();
        let inf = f64::INFINITY;
        use ComparisonOp::{Ge, Le};
        let problem = fuzz_model(
            OptimizationDirection::Minimize,
            &[
                (5.0, 0.0, inf, false),
                (-1.0, -inf, inf, false),
                (1.0, 4.0, 4.0, false),
                (4.0, 4.0, 7.0, false),
            ],
            &[
                (&[(0, -1.0), (1, 3.0), (2, -2.0)], Le, -6.0),
                (&[(0, 4.0), (1, 3.0), (2, -2.0), (3, -5.0)], Ge, 12.0),
                (&[(0, -1.0), (1, -1.0)], Ge, 19.0),
                (&[(0, -3.0), (1, 1.0)], Le, 3.0),
                (&[(0, -2.0), (1, -2.0), (3, -3.0)], Le, 16.0),
            ],
        );
        for presolve in [true, false] {
            let got = problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            });
            assert_eq!(
                got.map(|_| "solved"),
                Err(Error::Infeasible),
                "presolve {presolve}"
            );
        }
    }

    /// An infeasible LP of the differential fuzz (`int_lp-1145`, HiGHS
    /// agrees) whose proof needs a non-basic var's coefficient on an infinite
    /// bound to count as zero. Once refined, the row of the basis inverse
    /// that proves it gives the non-basic `x3` (`x3 >= 0`, unbounded above),
    /// whose tableau entry is zero, the coefficient zero; the second
    /// correction, spread over the row by its solve, leaves it `3.6e-33` of
    /// its terms. That lies within the bound the refinement puts on the
    /// multipliers' error, and counts as zero, as a basic var's residual
    /// does: a concession for basic vars only voided the proof, and the
    /// solve failed loudly, with presolve on and off.
    #[test]
    fn a_non_basic_vars_residual_does_not_void_a_proof() {
        init();
        let inf = f64::INFINITY;
        use ComparisonOp::{Eq, Ge, Le};
        let problem = fuzz_model(
            OptimizationDirection::Minimize,
            &[
                (-4.0, 0.0, 17.0, false),
                (1.0, 0.0, 17.0, false),
                (2.0, -3.0, 4.0, false),
                (5.0, 0.0, inf, false),
                (4.0, 0.0, 9.0, false),
                (-2.0, 0.0, inf, false),
                (-4.0, 0.0, inf, false),
                (2.0, 0.0, inf, false),
            ],
            &[
                (
                    &[(1, 3.0), (2, -4.0), (3, -4.0), (5, 5.0), (6, -4.0)],
                    Ge,
                    5.0,
                ),
                (
                    &[
                        (0, -4.0),
                        (1, 5.0),
                        (2, -3.0),
                        (3, 1.0),
                        (4, 5.0),
                        (5, 2.0),
                        (6, 5.0),
                        (7, -2.0),
                    ],
                    Eq,
                    -6.0,
                ),
                (&[(0, -3.0), (1, 4.0), (4, 4.0)], Le, 11.0),
                (
                    &[
                        (1, -4.0),
                        (2, 4.0),
                        (3, -2.0),
                        (4, -5.0),
                        (6, -3.0),
                        (7, 2.0),
                    ],
                    Ge,
                    14.0,
                ),
                (
                    &[
                        (1, -3.0),
                        (2, 4.0),
                        (4, -5.0),
                        (5, -2.0),
                        (6, -5.0),
                        (7, -1.0),
                    ],
                    Ge,
                    -5.0,
                ),
                (
                    &[(0, 2.0), (1, 2.0), (3, 1.0), (4, 2.0), (6, -5.0), (7, -3.0)],
                    Eq,
                    14.0,
                ),
            ],
        );
        for presolve in [true, false] {
            let got = problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            });
            assert_eq!(
                got.map(|_| "solved"),
                Err(Error::Infeasible),
                "presolve {presolve}"
            );
        }
    }

    /// A MILP whose relaxation is unbounded (HiGHS agrees), where the
    /// improving column meets the one row whose basic var could block it
    /// through a round-off entry of -4.4e-16, next to a genuine entry in a
    /// row that cannot block. The primal ratio test's round-off scale is the
    /// whole column: taken over the rows that can block alone, the entry was
    /// a pivot with a step of 5.6e15 into a singular basis.
    #[test]
    fn unbounded_relaxation_is_not_blocked_by_a_round_off_entry() {
        init();
        let inf = f64::INFINITY;
        use ComparisonOp::{Eq, Ge};
        let problem = fuzz_model(
            OptimizationDirection::Minimize,
            &[
                (1.0, -2.0, 6.0, false),
                (-5.0, 0.0, inf, false),
                (3.0, 0.0, 10.0, false),
                (2.0, 0.0, inf, false),
                (1.0, 0.0, inf, false),
                (2.0, 5.0, 5.0, false),
                (3.0, -4.0, 1.0, true),
            ],
            &[
                (
                    &[
                        (0, -2.0),
                        (1, 5.0),
                        (2, 4.0),
                        (3, -2.0),
                        (4, -4.0),
                        (6, -3.0),
                    ],
                    Eq,
                    13.0,
                ),
                (
                    &[(0, 2.0), (2, 4.0), (3, -2.0), (5, 1.0), (6, 4.0)],
                    Ge,
                    16.0,
                ),
            ],
        );
        for presolve in [true, false] {
            let got = problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            });
            assert_eq!(
                got.map(|_| "solved"),
                Err(Error::Unbounded),
                "presolve {presolve}"
            );
        }
    }
}
