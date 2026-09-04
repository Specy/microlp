//! Regression tests reproducing reported issues that have since been fixed.

#[cfg(test)]
mod regression_tests {
    use crate::{ComparisonOp, OptimizationDirection, Problem, SolveOptions};

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

    /// Build the second model from issue #44 (reported on the fix PR): `x`
    /// is pinned to zero by `-2e-4·k·x − 1e3·k·y = 0` with `y` fixed at zero.
    fn issue_44_scaled_row_model(
        k: f64,
        integer: bool,
    ) -> (Problem, crate::Variable, crate::Variable) {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(-1.0, (-1.0, 0.0));
        let y = problem.add_var(-1.0, (0.0, 0.0));
        if integer {
            problem.add_integer_var(-1.0, (0, 0));
        }
        problem.add_constraint([(x, -1e4)], ComparisonOp::Le, 1.0);
        problem.add_constraint([(x, -2e-4 * k), (y, -1e3 * k)], ComparisonOp::Eq, 0.0);
        (problem, x, y)
    }

    /// `x = y = i = 0` satisfies every bound and both rows for every `k`,
    /// and each `k` only scales the second row, so all five are the same
    /// problem. The engine equilibrates rows by a power of two and compared
    /// basic values to their bounds with a flat `EPS`, so its effective
    /// tolerance in the user's units grew with the row's largest coefficient:
    /// the vertex `x = -1e-4` violates the second row by `2e-8·k`, which the
    /// engine could not see, the MIP guard rejected for `k ≥ 10`
    /// (`InternalError`), and the LP path returned silently. Slacks now carry
    /// a tolerance derived from `Tolerances::feasibility` and the row's scale.
    #[test]
    fn issue_44_scaled_row_solves_for_every_scale() {
        for k in [1.0, 2.0, 0.5, 10.0, 1000.0] {
            for integer in [true, false] {
                let (problem, x, y) = issue_44_scaled_row_model(k, integer);
                let sol = problem
                    .solve()
                    .unwrap_or_else(|e| {
                        panic!("k={k} integer={integer}: {e:?} on a feasible model")
                    })
                    .into_solution()
                    .unwrap();
                let row2 = (-2e-4 * k * sol[x] - 1e3 * k * sol[y]).abs();
                assert!(
                    row2 <= 1e-7,
                    "k={k} integer={integer}: second row off by {row2:e} (x={:e})",
                    sol[x]
                );
                // Once the wrong vertex violates the row by more than the
                // tolerance, only the exact one is left.
                if 2e-4 * k * 1e-4 > 1e-7 {
                    assert!(
                        sol[x].abs() < 1e-9 && sol.objective().abs() < 1e-9,
                        "k={k} integer={integer}: x={:e} objective={:e}, expected 0",
                        sol[x],
                        sol.objective()
                    );
                }
            }
        }
    }

    /// The residual class behind issue #44 once the engine held its rows
    /// honestly: a row with coefficients around `1e8` and an activity near
    /// `6e9`, whose exact vertex is not representable in double precision.
    /// The validation guard demanded an absolute `1e-7` there — below one ulp
    /// of the activity — so no returned point could pass and the solve ended
    /// in `InternalError` (5 of 300 such random models on the fix branch, 6 on
    /// master). The guard is now floored at the row's round-off, which is
    /// also the tolerance the engine holds the row to. A brute-force
    /// enumeration of the two integer variables is the oracle.
    #[test]
    fn huge_coefficient_row_validates_within_round_off() {
        let (a0, a1, a2) = (233961702.1092298, -38659774.0504899, -132943028.20165081);
        let rhs = 5.839427392501522e9;
        let (c0, c1, c2) = (-0.5093205118777648, -1.9993457673342485, -0.9845099835842941);
        let x1_max = 68.6087643341415;

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_integer_var(c0, (0, 68));
        let x1 = problem.add_var(c1, (0.0, x1_max));
        let x2 = problem.add_integer_var(c2, (0, 68));
        problem.add_constraint([(x0, a0), (x1, a1), (x2, a2)], ComparisonOp::Eq, rhs);
        let sol = problem
            .solve()
            .unwrap_or_else(|e| panic!("feasible model (60, 47, 48) must solve, got {e:?}"))
            .into_solution()
            .unwrap();

        // Oracle: for each integer pair the row pins x1.
        let mut best = f64::INFINITY;
        for i0 in 0..=68 {
            for i2 in 0..=68 {
                let v1 = (rhs - a0 * f64::from(i0) - a2 * f64::from(i2)) / a1;
                if (0.0..=x1_max).contains(&v1) {
                    best = best.min(c0 * f64::from(i0) + c1 * v1 + c2 * f64::from(i2));
                }
            }
        }
        assert!(best.is_finite());
        assert!(
            (sol.objective() - best).abs() <= 1e-9 * best.abs(),
            "objective {} vs brute force {}",
            sol.objective(),
            best
        );
        let activity = a0 * sol[x0] + a1 * sol[x1] + a2 * sol[x2];
        let magnitude = (a0 * sol[x0]).abs() + (a1 * sol[x1]).abs() + (a2 * sol[x2]).abs();
        assert!(
            (activity - rhs).abs() <= 1e-13 * (1.0 + rhs.abs() + magnitude),
            "row off by {:e} at activity magnitude {:e}",
            activity - rhs,
            magnitude
        );
    }

    /// `Tolerances::feasibility` is the tolerance the engine works to. At the
    /// default `1e-7` the `k = 1` model may legitimately stop at `x = -1e-4`
    /// (its row is off by `2e-8`); asking for `1e-9` yields the exact vertex.
    #[test]
    fn feasibility_tolerance_drives_the_engine() {
        let (problem, x, y) = issue_44_scaled_row_model(1.0, false);
        let sol = problem.solve().unwrap().into_solution().unwrap();
        assert!((-2e-4 * sol[x] - 1e3 * sol[y]).abs() <= 1e-7);

        let mut options = SolveOptions::default();
        options.tolerances.feasibility = 1e-9;
        let (problem, x, _) = issue_44_scaled_row_model(1.0, false);
        let sol = problem
            .solve_with(options)
            .unwrap()
            .into_solution()
            .unwrap();
        assert!(
            sol[x].abs() < 1e-9 && sol.objective().abs() < 1e-9,
            "feasibility 1e-9: x={:e} objective={:e}, expected the exact vertex 0",
            sol[x],
            sol.objective()
        );
    }
}
