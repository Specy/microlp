#[cfg(test)]
mod tests_aoc2 {
    use crate::{ComparisonOp, OptimizationDirection, Problem, Variable};
    #[test]
    fn solve_ilp() {
        // Real-world ILP problem: minimize button presses to reach exact targets
        //
        // Variables: x[i] = number of times to press button i (non-negative integers)
        // Objective: minimize sum(x[i])
        // Constraints: for each target j, sum(x[i] where j in button[i]) = target[j]

        let buttons: Vec<Vec<usize>> = vec![
            vec![8],
            vec![0, 1, 3, 4, 6, 8],
            vec![1, 3, 4, 5, 7, 8],
            vec![2, 4],
            vec![0, 1, 3, 5],
            vec![0, 2, 3, 4, 5, 7, 8],
            vec![1, 2, 3],
            vec![0, 2, 5, 7],
            vec![1, 2, 6],
            vec![0, 1, 6],
            vec![1, 2, 4, 5, 6, 7, 8],
        ];

        let target: Vec<i32> = vec![41, 60, 170, 51, 186, 40, 30, 34, 44];

        // Build ILP model
        let mut problem = Problem::new(OptimizationDirection::Minimize);

        // Create integer variables for each button press count (non-negative integers)
        let mut press_counts = Vec::new();
        for _ in 0..buttons.len() {
            press_counts.push(problem.add_integer_var(1.0, (0, i32::MAX)));
        }

        // Add constraints: for each target index, sum of press counts for buttons
        // that affect that target must equal the target value
        for (target_idx, &target_val) in target.iter().enumerate() {
            let mut terms: Vec<(Variable, f64)> = Vec::new();
            for (btn_idx, button) in buttons.iter().enumerate() {
                if button.contains(&target_idx) {
                    terms.push((press_counts[btn_idx], 1.0));
                }
            }
            problem.add_constraint(&terms, ComparisonOp::Eq, target_val as f64);
        }

        // Solve
        let sol = problem.solve().unwrap();

        // Get solution values
        let values: Vec<i64> = press_counts
            .iter()
            .map(|&var| sol.var_value(var) as i64)
            .collect();

        let total: i64 = values.iter().sum();
        println!("Solver returned: {} total presses", total);
        println!("Solution values: {:?}", values);

        // Calculate achieved values
        let mut achieved = vec![0i64; target.len()];
        for (btn_idx, &count) in values.iter().enumerate() {
            for &idx in &buttons[btn_idx] {
                achieved[idx] += count;
            }
        }

        println!("Verification:");
        println!("  Expected: {:?}", target);
        println!("  Achieved: {:?}", achieved);

        // Verify all constraints are satisfied
        for (i, (a, t)) in achieved.iter().zip(&target).enumerate() {
            assert_eq!(
                *a, *t as i64,
                "Constraint {} violated: expected {}, got {}",
                i, t, a
            );
        }
    }
}
