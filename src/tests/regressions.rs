//! Regression tests reproducing reported issues that have since been fixed.

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

    /// One variant of the model from issue #44.
    struct Issue44 {
        tiny: f64,
        integer_var: bool,
        first_eq: bool,
    }

    /// Solve an issue-44 variant and return `(objective, x1, x2)`.
    fn solve_issue_44(v: Issue44) -> (f64, f64, f64) {
        let Issue44 {
            tiny,
            integer_var,
            first_eq,
        } = v;
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

    /// <https://github.com/Specy/microlp/issues/44>: a feasible, bounded MILP
    /// returned `InternalError`. The dual simplex pivoted `x2` into the basis
    /// on the `-1.19e-8` coefficient (accepted: it is above the `1e-10` pivot
    /// floor), which multiplied round-off by ~1e8 and left `x2 = 2^-26` where
    /// the final — perfectly conditioned — basis says exactly `0`. The phase
    /// ended because every basic value sat inside its bounds; nothing compared
    /// the values against the rows, so the drifted point reached the MIP
    /// layer's independent feasibility guard and was rejected. The engine now
    /// recomputes basic values from the original data at every
    /// refactorization and before ending a phase on a point whose row
    /// residual a fresh factorization would not produce.
    #[test]
    fn issue_44_reported_model_solves_to_the_only_feasible_point() {
        let tiny = -1.190834764418229e-8;
        // The report's model: the only feasible point is x1 = x2 = 0.
        for t in [tiny, -1e-7, -1e-8, -1e-9, -1e-12] {
            let (obj, x1, x2) = solve_issue_44(Issue44 {
                tiny: t,
                integer_var: true,
                first_eq: true,
            });
            assert!(
                obj.abs() < 1e-9 && x1.abs() < 1e-9 && x2.abs() < 1e-9,
                "tiny={t:e}: got obj={obj:e} x1={x1:e} x2={x2:e}"
            );
        }
        // The same LP through the pure-LP path (no integer variable).
        let (obj, x1, x2) = solve_issue_44(Issue44 {
            tiny,
            integer_var: false,
            first_eq: true,
        });
        assert!(
            obj.abs() < 1e-9 && x1.abs() < 1e-9 && x2.abs() < 1e-9,
            "pure LP: got obj={obj:e} x1={x1:e} x2={x2:e}"
        );
        // Without the first row x2 is free to reach 1 and the optimum is
        // -tiny. This guards the tempting non-fix of ignoring small
        // coefficients in the ratio tests, which silently returns 0 here.
        let (obj, x1, x2) = solve_issue_44(Issue44 {
            tiny,
            integer_var: true,
            first_eq: false,
        });
        assert!(
            (obj + tiny).abs() < 1e-15 && (x1 - tiny).abs() < 1e-15 && (x2 - 1.0).abs() < 1e-9,
            "no first row: got obj={obj:e} x1={x1:e} x2={x2:e}, expected obj={:e}",
            -tiny
        );
    }

    /// The same shape with generic coefficients. Before the fix roughly one
    /// in ten full-mantissa first-row coefficients of magnitude ≥ 4, paired
    /// with any `tiny` in `(1e-10, 1e-8)`, reproduced the failure — a band,
    /// not a point — so pin it with a deterministic sweep. Short-mantissa
    /// coefficients such as `8` or `8.5` never failed: their arithmetic is
    /// exact, which is why the report's value looked "magic".
    #[test]
    fn issue_44_coefficient_band_solves_to_the_only_feasible_point() {
        use rand::prelude::*;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(44);
        for _ in 0..200 {
            // log-uniform in [1e-10, 1e-8]; [4, 1024) with a full mantissa
            let u: f64 = rng.random();
            let tiny = -1e-10 * 100f64.powf(u);
            let e: f64 = rng.random();
            let m: f64 = rng.random();
            let big = 2f64.powi(2 + (e * 8.0) as i32) * (1.0 + m);

            let mut problem = Problem::new(OptimizationDirection::Maximize);
            let x1 = problem.add_var(-1.0, (-1.0, 0.0));
            let x2 = problem.add_var(0.0, (0.0, 1.0));
            problem.add_integer_var(0.0, (0, 0));
            problem.add_constraint([(x2, big)], ComparisonOp::Eq, 0.0);
            problem.add_constraint([(x1, -1.0), (x2, tiny)], ComparisonOp::Eq, 0.0);
            let sol = problem
                .solve()
                .unwrap_or_else(|e| panic!("tiny={tiny:e} big={big:e}: must solve, got {e:?}"))
                .into_solution()
                .unwrap_or_else(|i| panic!("tiny={tiny:e} big={big:e}: got {i:?}"));
            let (obj, v1, v2) = (sol.objective(), sol.var_value(x1), sol.var_value(x2));
            assert!(
                obj.abs() < 1e-9 && v1.abs() < 1e-9 && v2.abs() < 1e-9,
                "tiny={tiny:e} big={big:e}: got obj={obj:e} x1={v1:e} x2={v2:e}"
            );
        }
    }
}
