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

    /// <https://github.com/Specy/microlp/issues/47>: fixing a variable to the
    /// value `solve` just reported for it must keep the model feasible. It
    /// used to return `Infeasible` whenever the variable appeared in an
    /// equality row: the LP path forced the (basic) variable out of the basis
    /// with a dual pivot, but the only other column of the row was the
    /// equality slack, itself fixed at zero and therefore never eligible to
    /// enter. The same row as `Le` was accepted.
    #[test]
    fn issue_47_fix_var_to_reported_value_on_equality_row() {
        for (op, rhs, hi) in [
            (ComparisonOp::Eq, 0.0, 1.0),
            (ComparisonOp::Le, 0.0, 1.0),
            (ComparisonOp::Eq, 1.0, 2.0),
        ] {
            let mut problem = Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(-1.0, (0.0, hi));
            problem.add_constraint([(x, 1.0)], op, rhs);
            let solution = problem.solve().unwrap().into_solution().unwrap();
            let value = solution.var_value(x);
            assert!(
                (value - rhs).abs() < 1e-9,
                "{op:?} {rhs}: solve gave {value}"
            );

            let fixed = solution
                .fix_var(x, value)
                .unwrap_or_else(|e| {
                    panic!("x in [0, {hi}], x {op:?} {rhs}: fix_var(x, {value}) = Err({e:?})")
                })
                .into_solution()
                .unwrap();
            assert!((fixed.var_value(x) - value).abs() < 1e-9);
            assert!((fixed.objective() - -rhs).abs() < 1e-9);

            // Releasing the fix returns to the same optimum.
            let (restored, was_fixed) = fixed.unfix_var(x).unwrap();
            assert!(was_fixed);
            let restored = restored.into_solution().unwrap();
            assert!((restored.var_value(x) - value).abs() < 1e-9);
        }
    }

    /// Companion to the issue #47 fix: an equality row must still reject a
    /// fix to a value it cannot satisfy, on a variable that is basic in the
    /// solved basis.
    #[test]
    fn issue_47_fix_var_off_the_equality_value_is_infeasible() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_var(-1.0, (0.0, 2.0));
        problem.add_constraint([(x, 1.0)], ComparisonOp::Eq, 1.0);
        let solution = problem.solve().unwrap().into_solution().unwrap();
        assert!((solution.var_value(x) - 1.0).abs() < 1e-9);

        assert!(matches!(
            solution.fix_var(x, 0.5),
            Err(crate::Error::Infeasible)
        ));
    }

    /// Companion to the issue #47 fix: fixes are applied as bounds internally,
    /// so re-fixing must still validate against the variable's *original*
    /// bounds (not the pinned ones), and unfixing must restore the original
    /// bounds regardless of how many times the variable was re-fixed.
    #[test]
    fn issue_47_refix_validates_against_original_bounds_and_unfix_restores() {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, 3.0));
        let y = problem.add_var(2.0, (0.0, 3.0));
        problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        let solution = problem.solve().unwrap().into_solution().unwrap();
        assert!((solution.objective() - 7.0).abs() < 1e-9);

        // Fix x = 0.5, then re-fix x = 2.0 (inside the original [0, 3]).
        let fixed = solution.fix_var(x, 0.5).unwrap().into_solution().unwrap();
        assert!((fixed.var_value(x) - 0.5).abs() < 1e-9);
        assert!((fixed.objective() - 6.5).abs() < 1e-9);
        let refixed = fixed.fix_var(x, 2.0).unwrap().into_solution().unwrap();
        assert!((refixed.var_value(x) - 2.0).abs() < 1e-9);
        assert!((refixed.var_value(y) - 2.0).abs() < 1e-9);
        assert!((refixed.objective() - 6.0).abs() < 1e-9);

        // A value outside the original bounds is still rejected while fixed.
        assert!(matches!(
            refixed.clone().fix_var(x, 3.5),
            Err(crate::Error::Infeasible)
        ));

        // Unfix restores [0, 3] and the original optimum in one step.
        let (restored, was_fixed) = refixed.unfix_var(x).unwrap();
        assert!(was_fixed);
        let restored = restored.into_solution().unwrap();
        assert!((restored.var_value(x) - 1.0).abs() < 1e-9);
        assert!((restored.var_value(y) - 3.0).abs() < 1e-9);
        assert!((restored.objective() - 7.0).abs() < 1e-9);
        let (_, was_fixed_again) = restored.unfix_var(x).unwrap();
        assert!(!was_fixed_again);
    }
}
