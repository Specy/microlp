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

            let sol = problem.solve().unwrap();

            assert!((sol.var_value(x) - 2.0).abs() < 1e-6, "x wrong for upper={upper:e}");
            assert!((sol.var_value(y) - 6.2).abs() < 1e-6, "y wrong for upper={upper:e}");
            assert!((sol.var_value(z) - 1.6).abs() < 1e-6, "z wrong for upper={upper:e}");
            assert!(
                (sol.objective() - 420.0).abs() < 1e-6,
                "objective wrong for upper={upper:e}: got {}",
                sol.objective()
            );
        }
    }
}
