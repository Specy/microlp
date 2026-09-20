//! Tests reproducing reported issues. The closed ones guard against the bug
//! returning; the open ones (#44, #47, #48) fail until it is fixed.

#[cfg(test)]
mod regression_tests {
    use crate::{ComparisonOp, OptimizationDirection, Problem};

    /// <https://github.com/Specy/microlp/issues/3>: a huge but *finite* variable
    /// bound (`f64::MAX`, `f32::MAX`, `i64::MAX`, …) must behave like
    /// `f64::INFINITY`. Such a bound used to be seeded into the simplex tableau
    /// as a literal value, which swamped the problem data (the rhs and
    /// coefficients lost all significance against it) and returned a wrong vertex
    /// — e.g. `[2, 0, 0]` (or a NaN objective) instead of `[2, 6.2, 1.6]`.
    #[test]
    fn issue_3_huge_upper_bound_behaves_like_infinity() {
        // Every one of these upper bounds must give the same answer as infinity.
        for upper in [f64::MAX, f32::MAX as f64, i64::MAX as f64, f64::INFINITY] {
            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x = problem.add_var(50.0, (2.0, f64::INFINITY));
            let y = problem.add_var(40.0, (0.0, 7.0));
            let z = problem.add_var(45.0, (0.0, upper));
            problem.add_constraint(&[(x, 3.0), (y, 2.0), (z, 1.0)], ComparisonOp::Le, 20.0);
            problem.add_constraint(&[(x, 2.0), (y, 1.0), (z, 3.0)], ComparisonOp::Le, 15.0);

            let sol = problem
                .solve()
                .unwrap()
                .into_solution()
                .expect("an unlimited bounded solve must return a solution");

            assert!(
                (sol.var_value(x) - 2.0).abs() < 1e-6,
                "x wrong for upper={upper:e}"
            );
            assert!(
                (sol.var_value(y) - 6.2).abs() < 1e-6,
                "y wrong for upper={upper:e}"
            );
            assert!(
                (sol.var_value(z) - 1.6).abs() < 1e-6,
                "z wrong for upper={upper:e}"
            );
            assert!(
                (sol.objective() - 420.0).abs() < 1e-6,
                "objective wrong for upper={upper:e}: got {}",
                sol.objective()
            );
        }
    }

    /// <https://github.com/Specy/microlp/issues/42>: an always-feasible integer
    /// model must solve no matter how large `x`'s upper bound is. The only
    /// feasible point of `3x + 2y = 5` with `x` integer and `y ∈ [0, 1]` is
    /// `x = 1, y = 1` (objective 1). In 0.4.0, upper bounds of the form `2^k + 2`
    /// made `solve()` return `Err` for even `k` (a bound-magnitude rounding
    /// interaction), even though the feasible point never changes.
    #[test]
    fn issue_42_large_integer_bound_stays_feasible() {
        // The reported bounds are 2^k + 2 for k = 20..=30 (which alternately
        // failed); sweep those and their immediate neighbours as a guard.
        for k in 20..=30u32 {
            for max in [(1i64 << k), (1i64 << k) + 1, (1i64 << k) + 2] {
                let max = max as i32;
                let mut problem = Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_integer_var(1.0, (0, max));
                let y = problem.add_var(0.0, (0.0, 1.0));
                problem.add_constraint([(x, 3.0), (y, 2.0)], ComparisonOp::Eq, 5.0);

                let sol = problem
                    .solve()
                    .unwrap_or_else(|e| panic!("max={max} (~2^{k}) must be feasible, got {e:?}"))
                    .into_solution()
                    .unwrap_or_else(|interrupted| {
                        panic!("max={max} (~2^{k}) must finish without limits, got {interrupted:?}")
                    });

                assert!(
                    (sol.var_value(x) - 1.0).abs() < 1e-6,
                    "x wrong for max={max}"
                );
                assert!(
                    (sol.var_value(y) - 1.0).abs() < 1e-6,
                    "y wrong for max={max}"
                );
                assert!(
                    (sol.objective() - 1.0).abs() < 1e-6,
                    "objective wrong for max={max}: got {}",
                    sol.objective()
                );
            }
        }
    }

    /// One variant of the model reported in issue #44: maximise `-x1` subject
    /// to `8.510167104926385 x2 == 0` (when `first_eq`) and `-x1 + tiny x2 ==
    /// 0`, with `x1` in `[-1, 0]`, `x2` in `[0, 1]` and, when `integer_var`, an
    /// integer variable fixed at zero by its own bounds that appears in neither
    /// the objective nor a row.
    struct Issue44 {
        tiny: f64,
        integer_var: bool,
        first_eq: bool,
    }

    /// Solves an issue-44 variant and returns `(objective, x1, x2)`.
    fn solve_issue_44(variant: Issue44) -> (f64, f64, f64) {
        let Issue44 {
            tiny,
            integer_var,
            first_eq,
        } = variant;
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x1 = problem.add_var(-1.0, (-1.0, 0.0));
        let x2 = problem.add_var(0.0, (0.0, 1.0));
        if integer_var {
            problem.add_integer_var(0.0, (0, 0));
        }
        if first_eq {
            problem.add_constraint([(x2, 8.510167104926385)], ComparisonOp::Eq, 0.0);
        }
        problem.add_constraint([(x1, -1.0), (x2, tiny)], ComparisonOp::Eq, 0.0);
        let sol = problem
            .solve()
            .unwrap_or_else(|e| panic!("tiny={tiny:e}: must solve, got {e:?}"))
            .into_solution()
            .unwrap_or_else(|i| panic!("tiny={tiny:e}: must finish without limits, got {i:?}"));
        (sol.objective(), sol.var_value(x1), sol.var_value(x2))
    }

    /// <https://github.com/Specy/microlp/issues/44>: `solve()` returns
    /// `InternalError("exactly integral solution failed feasibility validation
    /// after slack-basis retry")` on a feasible, bounded MILP. `x1 = x2 = 0`
    /// satisfies both equalities and every bound, and it is the only such
    /// point, so the answer is `Ok` with objective `0`.
    ///
    /// The report rules out a magnitude threshold in the small coefficient: the
    /// same model solves at `-1e-7`, `-1e-8`, `-1e-9` and `-1e-12`, and the
    /// reported coefficient itself solves once either the integer variable or
    /// the first row is dropped.
    #[test]
    fn issue_44_feasible_bounded_milp_solves() {
        let tiny = -1.190834764418229e-8;

        // The reported model, and the coefficients around it that already
        // solve. In each the only feasible point is x1 = x2 = 0.
        for t in [tiny, -1e-7, -1e-8, -1e-9, -1e-12] {
            let (obj, x1, x2) = solve_issue_44(Issue44 {
                tiny: t,
                integer_var: true,
                first_eq: true,
            });
            assert!(
                obj.abs() < 1e-9 && x1.abs() < 1e-9 && x2.abs() < 1e-9,
                "tiny={t:e}: got obj={obj:e} x1={x1:e} x2={x2:e}, expected all zero"
            );
        }

        // The same LP down the pure-LP path, with no integer variable.
        let (obj, x1, x2) = solve_issue_44(Issue44 {
            tiny,
            integer_var: false,
            first_eq: true,
        });
        assert!(
            obj.abs() < 1e-9 && x1.abs() < 1e-9 && x2.abs() < 1e-9,
            "pure LP: got obj={obj:e} x1={x1:e} x2={x2:e}, expected all zero"
        );

        // Without the first row, x2 is free to reach its upper bound and the
        // optimum is -tiny, reached at x1 = tiny, x2 = 1. Asserting it guards
        // against "fixing" the model above by dropping small coefficients from
        // the ratio tests, which would answer 0 here.
        let (obj, x1, x2) = solve_issue_44(Issue44 {
            tiny,
            integer_var: true,
            first_eq: false,
        });
        assert!(
            (obj + tiny).abs() < 1e-14 && (x1 - tiny).abs() < 1e-14 && (x2 - 1.0).abs() < 1e-9,
            "no first row: got obj={obj:e} x1={x1:e} x2={x2:e}, expected obj={:e} x1={tiny:e} x2=1",
            -tiny
        );
    }

    /// <https://github.com/Specy/microlp/issues/47>: `Solution::fix_var`
    /// returns `Infeasible` when a variable that appears in an equality row is
    /// fixed to the value the solve just reported for it. The row pins `x` to
    /// its rhs, `solve` returns that value, and fixing `x` to it only restates
    /// what the row already forces, so the edited model has the same feasible
    /// point as the original.
    ///
    /// The `Le` row is the control: the same value through a row that is not an
    /// equality is accepted. The third case moves the fixed value off the
    /// variable's bounds and away from zero, so neither explains the failure.
    #[test]
    fn issue_47_fix_var_accepts_the_value_solve_returned() {
        for (op, rhs, hi) in [
            (ComparisonOp::Eq, 0.0, 1.0),
            (ComparisonOp::Le, 0.0, 1.0),
            (ComparisonOp::Eq, 1.0, 2.0),
        ] {
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(-1.0, (0.0, hi));
            problem.add_constraint([(x, 1.0)], op, rhs);

            let solution = problem
                .solve()
                .unwrap_or_else(|e| panic!("x in [0, {hi}], x {op:?} {rhs}: must solve, got {e:?}"))
                .into_solution()
                .unwrap_or_else(|i| {
                    panic!("x in [0, {hi}], x {op:?} {rhs}: must finish without limits, got {i:?}")
                });

            // Minimising -x drives x up to the row, so the reported value is
            // the rhs in all three models.
            let value = solution.var_value(x);
            assert!(
                (value - rhs).abs() < 1e-9,
                "x in [0, {hi}], x {op:?} {rhs}: solve gave x = {value:?}, expected {rhs:?}"
            );

            let refixed = solution
                .fix_var(x, value)
                .unwrap_or_else(|e| {
                    panic!("x in [0, {hi}], x {op:?} {rhs}: fix_var(x, {value:?}) gave {e:?}")
                })
                .into_solution()
                .unwrap_or_else(|i| {
                    panic!("x in [0, {hi}], x {op:?} {rhs}: fix_var must finish, got {i:?}")
                });

            let fixed_value = refixed.var_value(x);
            assert!(
                (fixed_value - value).abs() < 1e-9,
                "x in [0, {hi}], x {op:?} {rhs}: after fix_var(x, {value:?}) x = {fixed_value:?}"
            );
        }
    }

    /// <https://github.com/Specy/microlp/issues/48>: `solve()` returns
    /// `InternalError("Singular matrix")` on a small, feasible, bounded MILP —
    /// a DC network with switchable lines (8 buses, 10 lines, at most one
    /// switched out), whose big-M on/off rows spread the coefficients from
    /// 4e-2 to 1.1e3. HiGHS and SCIP prove the same model optimal at
    /// 23.339329.
    ///
    /// The report traces it to the dual ratio test accepting a pivot of
    /// |alpha| = 2.2e-10 — just above the `1e-10` floor, and zero in exact
    /// arithmetic — after which the refactorization is singular and the
    /// slack-basis retry re-derives a singular basis of its own.
    #[test]
    fn issue_48_switchable_dc_network_solves() {
        // (from, to, susceptance, rating)
        let lines = [
            (0, 1, 0.97, 0.52),
            (2, 4, 0.24, 1.11),
            (3, 5, 431.79, 0.55),
            (2, 6, 54.08, 0.77),
            (3, 7, 0.06, 1.22),
            (6, 1, 63.12, 0.84),
            (4, 1, 1108.6, 1.17),
            (1, 3, 0.34, 0.48),
            (5, 1, 538.81, 0.71),
            (3, 4, 0.04, 0.83),
        ];
        let demand = [0.4, 0.4, 0.85, 0.92, 0.19, 0.15, 0.52, 0.6];
        let gen_max = [0.53, 1.36, 0.95, 0.3, 0.5, 0.92, 0.22, 0.67];
        let gen_cost = [1.57, 8.12, 8.82, 8.04, 3.59, 2.49, 9.28, 8.04];
        let max_off = 1;

        let n = demand.len();
        let angle_max = 0.52;
        let big = angle_max * n as f64;
        let mut problem = Problem::new(OptimizationDirection::Minimize);

        // Bus angles: free, except the reference bus.
        let theta: Vec<_> = (0..n)
            .map(|i| match i {
                0 => problem.add_var(0.0, (0.0, 0.0)),
                _ => problem.add_var(0.0, (f64::NEG_INFINITY, f64::INFINITY)),
            })
            .collect();
        let gen: Vec<_> = (0..n)
            .map(|i| problem.add_var(gen_cost[i], (0.0, gen_max[i])))
            .collect();
        let flow: Vec<_> = lines
            .iter()
            .map(|&(_, _, _, r)| problem.add_var(0.0, (-r, r)))
            .collect();
        let on: Vec<_> = lines.iter().map(|_| problem.add_binary_var(0.0)).collect();

        for (e, &(i, j, b, r)) in lines.iter().enumerate() {
            let m = b * big;
            // |flow - b (theta_i - theta_j)| <= m (1 - on)
            problem.add_constraint(
                [(flow[e], 1.0), (theta[i], -b), (theta[j], b), (on[e], m)],
                ComparisonOp::Le,
                m,
            );
            problem.add_constraint(
                [(flow[e], 1.0), (theta[i], -b), (theta[j], b), (on[e], -m)],
                ComparisonOp::Ge,
                -m,
            );
            // |flow| <= r on
            problem.add_constraint([(flow[e], 1.0), (on[e], -r)], ComparisonOp::Le, 0.0);
            problem.add_constraint([(flow[e], 1.0), (on[e], r)], ComparisonOp::Ge, 0.0);
            // |theta_i - theta_j| <= angle_max + big (1 - on)
            problem.add_constraint(
                [(theta[i], 1.0), (theta[j], -1.0), (on[e], big)],
                ComparisonOp::Le,
                big + angle_max,
            );
            problem.add_constraint(
                [(theta[i], 1.0), (theta[j], -1.0), (on[e], -big)],
                ComparisonOp::Ge,
                -big - angle_max,
            );
        }
        for i in 0..n {
            // generation - demand = flow out - flow in
            let mut terms = vec![(gen[i], 1.0)];
            for (e, &(from, to, _, _)) in lines.iter().enumerate() {
                if from == i {
                    terms.push((flow[e], -1.0));
                } else if to == i {
                    terms.push((flow[e], 1.0));
                }
            }
            problem.add_constraint(terms, ComparisonOp::Eq, demand[i]);
        }
        let in_service: Vec<_> = on.iter().map(|&z| (z, 1.0)).collect();
        problem.add_constraint(in_service, ComparisonOp::Ge, (lines.len() - max_off) as f64);

        let sol = problem
            .solve()
            .unwrap_or_else(|e| panic!("a feasible, bounded MILP must solve, got {e:?}"))
            .into_solution()
            .unwrap_or_else(|i| panic!("must finish without limits, got {i:?}"));

        assert!(
            (sol.objective() - 23.339329).abs() < 1e-4,
            "objective {} does not match the 23.339329 HiGHS and SCIP prove optimal",
            sol.objective()
        );
    }
}
