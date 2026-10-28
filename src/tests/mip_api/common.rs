//! Problem builders and assertions shared by the `mip_api` test groups.

use crate::*;

pub(super) fn int_2var_problem() -> (Problem, Variable, Variable) {
    // minimize 3a + 4b s.t. a + 2b >= 5, 3a + b >= 4; a,b int in [0,10] → a=1,b=2, obj 11.
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let a = p.add_integer_var(3.0, (0, 10));
    let b = p.add_integer_var(4.0, (0, 10));
    p.add_constraint(&[(a, 1.0), (b, 2.0)], ComparisonOp::Ge, 5.0);
    p.add_constraint(&[(a, 3.0), (b, 1.0)], ComparisonOp::Ge, 4.0);
    (p, a, b)
}

pub(super) fn binary_knapsack() -> Problem {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_binary_var(8.0);
    let y = p.add_binary_var(11.0);
    let z = p.add_binary_var(6.0);
    let w = p.add_binary_var(4.0);
    p.add_constraint(
        &[(x, 5.0), (y, 7.0), (z, 4.0), (w, 3.0)],
        ComparisonOp::Le,
        14.0,
    );
    p
}

pub(super) fn solution(outcome: SolveOutcome) -> Solution {
    outcome
        .into_solution()
        .expect("this solve must return a usable solution")
}
