#[cfg(test)]
mod broken_tests_set {

    #[allow(unused_imports)]
    use crate::{ComparisonOp, OptimizationDirection, Problem};

    #[test]
    #[should_panic]
    fn broken_test_1() {
        let mut problem = Problem::new(OptimizationDirection::Maximize);

        // Define variables with their objective coefficients
        let x = problem.add_var(50.0, (2.0, f64::INFINITY));
        let y = problem.add_var(40.0, (0.0, 7.0));

        //TODO this fails to find optimal solution with f64::MAX
        let z = problem.add_var(45.0, (0.0, f64::MAX));

        problem.add_constraint(&[(x, 3.0), (y, 2.0), (z, 1.0)], ComparisonOp::Le, 20.0);
        problem.add_constraint(&[(x, 2.0), (y, 1.0), (z, 3.0)], ComparisonOp::Le, 15.0);

        let sol = problem.solve().unwrap();

        assert_eq!(
            [sol.var_value(x), sol.var_value(y), sol.var_value(z)],
            [2.0, 6.2, 1.6]
        );
        assert_eq!(sol.objective(), 420.0);
    }
}
