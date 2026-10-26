//! Duals and reduced costs ([`Solution::dual`], [`Solution::reduced_cost`]).
//!
//! A simplex dual is one of possibly many optimal duals, so beyond a few
//! models whose duals are known by hand, the tests check what every optimal
//! dual satisfies, in the units the user sees:
//!
//! * the optimality conditions ([`check_optimality_conditions`]): each reduced
//!   cost is the objective coefficient less the column times the duals, a
//!   dual has the sign its row's sense and the direction allow and is zero on
//!   a row that does not bind, a reduced cost has the sign the var's position
//!   allows, and the objective equals the dual objective;
//! * the subgradient inequality ([`check_subgradient`]): the optimal value is
//!   convex in a right-hand side when minimizing (concave when maximizing),
//!   and an optimal dual is a subgradient of it, so re-solving with the
//!   right-hand side moved either way never beats the first-order prediction.
//!   That pins a dual between the one-sided slopes whatever the degeneracy,
//!   and catches a dual off by a row or column factor or by its sign.
//!
//! The random models span row and column factors of `1e-2..1e2`, so the
//! engine's row equilibration and column scaling are both in play.

use crate::{
    ComparisonOp, Constraint, LinearExpr, OptimizationDirection, Problem, Solution, SolveOptions,
    Variable,
};
use rand::prelude::*;

struct Row {
    terms: Vec<(usize, f64)>,
    op: ComparisonOp,
    rhs: f64,
}

struct Lp {
    direction: OptimizationDirection,
    obj: Vec<f64>,
    bounds: Vec<(f64, f64)>,
    rows: Vec<Row>,
}

impl Lp {
    fn build(&self) -> (Problem, Vec<Variable>, Vec<Constraint>) {
        let mut problem = Problem::new(self.direction);
        let vars: Vec<Variable> = self
            .obj
            .iter()
            .zip(&self.bounds)
            .map(|(&c, &b)| problem.add_var(c, b))
            .collect();
        let rows = self
            .rows
            .iter()
            .map(|row| {
                let expr: LinearExpr = row.terms.iter().map(|&(v, a)| (vars[v], a)).collect();
                problem.add_constraint(expr, row.op, row.rhs)
            })
            .collect();
        (problem, vars, rows)
    }

    fn solve(&self, presolve: bool) -> Option<Solution> {
        let (problem, _, _) = self.build();
        let options = SolveOptions {
            presolve,
            ..SolveOptions::default()
        };
        problem.solve_with(options).ok()?.into_solution().ok()
    }

    /// A random feasible, bounded LP of `n` vars and `m` rows, built around a
    /// point inside the box; about a third of the rows bind there.
    fn random(rng: &mut impl Rng, n: usize, m: usize, direction: OptimizationDirection) -> Self {
        fn factor(rng: &mut impl Rng) -> f64 {
            10f64.powf(rng.random_range(-2.0..2.0))
        }
        let col_factors: Vec<f64> = (0..n).map(|_| factor(rng)).collect();
        let mut bounds = Vec::with_capacity(n);
        let mut xstar = Vec::with_capacity(n);
        for &s in &col_factors {
            let lo = rng.random_range(-5.0..0.0) / s;
            let hi = rng.random_range(1.0..5.0) / s;
            bounds.push((lo, hi));
            xstar.push(rng.random_range(lo..hi));
        }
        let obj = (0..n)
            .map(|j| rng.random_range(-10.0..10.0) * col_factors[j])
            .collect();
        let rows = (0..m)
            .map(|_| {
                let row_factor = factor(rng);
                let mut terms = Vec::new();
                for (j, &s) in col_factors.iter().enumerate() {
                    if rng.random_range(0..3) == 0 || terms.is_empty() && j == n - 1 {
                        let a = rng.random_range(0.5..2.0) * if rng.random() { 1.0 } else { -1.0 };
                        terms.push((j, a * s * row_factor));
                    }
                }
                let activity: f64 = terms.iter().map(|&(j, a)| a * xstar[j]).sum();
                let gap = if rng.random_range(0..3) == 0 {
                    0.0
                } else {
                    rng.random_range(0.0..2.0) * row_factor
                };
                let (op, rhs) = match rng.random_range(0..5) {
                    0 => (ComparisonOp::Eq, activity),
                    1 | 2 => (ComparisonOp::Le, activity + gap),
                    _ => (ComparisonOp::Ge, activity - gap),
                };
                Row { terms, op, rhs }
            })
            .collect();
        Lp {
            direction,
            obj,
            bounds,
            rows,
        }
    }
}

/// Whether `a` is at most `b`, allowing `tol`.
fn at_most(a: f64, b: f64, tol: f64) -> bool {
    a <= b + tol
}

/// The optimality conditions of `solution` for `lp` (with `bounds` in force,
/// which a fix changes), within tolerances relative to the magnitudes each
/// quantity is computed from.
fn check_optimality_conditions(lp: &Lp, bounds: &[(f64, f64)], solution: &Solution) {
    let (_, vars, rows) = lp.build();
    let max = lp.direction == OptimizationDirection::Maximize;
    let x: Vec<f64> = vars.iter().map(|&v| solution.var_value(v)).collect();
    let y: Vec<f64> = rows.iter().map(|&r| solution.dual(r).unwrap()).collect();
    assert_eq!(solution.duals().unwrap().len(), lp.rows.len());

    // Reduced costs are the objective less the column times the duals.
    for (j, &var) in vars.iter().enumerate() {
        let d = solution.reduced_cost(var).unwrap();
        let (mut priced, mut magnitude) = (0.0, lp.obj[j].abs());
        for (i, row) in lp.rows.iter().enumerate() {
            for &(v, a) in &row.terms {
                if v == j {
                    priced += a * y[i];
                    magnitude += (a * y[i]).abs();
                }
            }
        }
        let tol = 1e-8 * magnitude + 1e-10;
        assert!(
            (lp.obj[j] - priced - d).abs() <= tol,
            "var {j}: c - A^T y = {} but the reduced cost is {d}",
            lp.obj[j] - priced
        );
        // Its sign by the var's position, in the minimizing sense.
        let d_min = if max { -d } else { d };
        let (lo, hi) = bounds[j];
        let near = |b: f64| (x[j] - b).abs() <= 1e-6 * (1.0 + b.abs());
        if lo == hi {
            continue;
        }
        if !near(lo) {
            assert!(
                at_most(d_min, 0.0, tol),
                "var {j} above its lower bound has d = {d}"
            );
        }
        if !near(hi) {
            assert!(
                at_most(0.0, d_min, tol),
                "var {j} below its upper bound has d = {d}"
            );
        }
    }

    // Duals by sense, and zero on a row that does not bind.
    let mut dual_objective = 0.0;
    let mut magnitude = 0.0;
    for (i, row) in lp.rows.iter().enumerate() {
        let y_min = if max { -y[i] } else { y[i] };
        let activity: f64 = row.terms.iter().map(|&(j, a)| a * x[j]).sum();
        let scale: f64 = row
            .terms
            .iter()
            .map(|&(j, a)| (a * x[j]).abs())
            .sum::<f64>()
            + row.rhs.abs();
        let binds = (activity - row.rhs).abs() <= 1e-6 * (1.0 + scale);
        let tol = 1e-8 * (1.0 + y[i].abs());
        match row.op {
            ComparisonOp::Le => assert!(at_most(y_min, 0.0, tol), "Le row {i}: dual {}", y[i]),
            ComparisonOp::Ge => assert!(at_most(0.0, y_min, tol), "Ge row {i}: dual {}", y[i]),
            ComparisonOp::Eq => {}
        }
        if !binds {
            assert!(
                y[i].abs() <= tol,
                "row {i} does not bind but has dual {}",
                y[i]
            );
        }
        dual_objective += y[i] * row.rhs;
        magnitude += (y[i] * row.rhs).abs();
    }
    // Strong duality: c^T x = b^T y + d^T x, the reduced costs pricing the
    // bounds the vars sit at.
    for (j, &var) in vars.iter().enumerate() {
        let term = solution.reduced_cost(var).unwrap() * x[j];
        dual_objective += term;
        magnitude += term.abs();
    }
    let objective = solution.objective();
    assert!(
        (objective - dual_objective).abs() <= 1e-7 * (1.0 + magnitude + objective.abs()),
        "objective {objective} but dual objective {dual_objective}"
    );
}

/// The subgradient inequality of every dual (see the module docs): moving
/// each right-hand side by `±delta` never beats the dual's prediction.
/// Returns how many one-sided predictions held with equality, to show the
/// test is not vacuous.
fn check_subgradient(lp: &mut Lp, solution: &Solution) -> usize {
    let max = lp.direction == OptimizationDirection::Maximize;
    let v = solution.objective();
    let scale = 1.0 + v.abs();
    let duals = solution.duals().unwrap().to_vec();
    let mut tight = 0;
    for i in 0..lp.rows.len() {
        let rhs = lp.rows[i].rhs;
        let delta = 1e-3 * (1.0 + rhs.abs());
        for step in [delta, -delta] {
            lp.rows[i].rhs = rhs + step;
            if let Some(moved) = lp.solve(true) {
                let predicted = v + duals[i] * step;
                let gain = if max {
                    moved.objective() - predicted
                } else {
                    predicted - moved.objective()
                };
                assert!(
                    gain <= 1e-7 * scale,
                    "row {i} moved by {step}: objective {} beats {predicted} (dual {})",
                    moved.objective(),
                    duals[i]
                );
                if gain.abs() <= 1e-7 * scale {
                    tight += 1;
                }
            }
        }
        lp.rows[i].rhs = rhs;
    }
    tight
}

#[test]
fn a_dispatch_prices_demand_at_the_marginal_unit() {
    // Three generators at 20, 35 and 50, limits 6 and 5 on the first two,
    // demand 14: the third sets the price, and each limit is worth the
    // saving over it.
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let g: Vec<Variable> = [20.0, 35.0, 50.0]
        .iter()
        .map(|&c| problem.add_var(c, (0.0, f64::INFINITY)))
        .collect();
    let demand = problem.add_constraint(
        g.iter().map(|&v| (v, 1.0)).collect::<LinearExpr>(),
        ComparisonOp::Eq,
        14.0,
    );
    let limit0 = problem.add_constraint(&[(g[0], 1.0)], ComparisonOp::Le, 6.0);
    let limit1 = problem.add_constraint(&[(g[1], 1.0)], ComparisonOp::Le, 5.0);
    let solution = problem.solve().unwrap().into_solution().unwrap();
    assert!((solution.objective() - (120.0 + 175.0 + 150.0)).abs() < 1e-9);
    assert!((solution.dual(demand).unwrap() - 50.0).abs() < 1e-9);
    assert!((solution.dual(limit0).unwrap() + 30.0).abs() < 1e-9);
    assert!((solution.dual(limit1).unwrap() + 15.0).abs() < 1e-9);
    for &v in &g {
        assert!(solution.reduced_cost(v).unwrap().abs() < 1e-9);
    }

    // With the dear unit fixed at 4, the second sets the price at 35, and
    // the fix is worth 50 - 35 per unit.
    let solution = solution
        .fix_var(g[2], 4.0)
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((solution.dual(demand).unwrap() - 35.0).abs() < 1e-9);
    assert!((solution.dual(limit0).unwrap() + 15.0).abs() < 1e-9);
    assert!(solution.dual(limit1).unwrap().abs() < 1e-9);
    assert!((solution.reduced_cost(g[2]).unwrap() - 15.0).abs() < 1e-9);
}

#[test]
fn signs_follow_the_direction_and_the_sense() {
    // max 3x + 2y s.t. x + y <= 4, x + 3y <= 6, x >= 1 (as a row), x, y >= 0:
    // optimum at x = 4, y = 0, objective 12. Raising the first right-hand
    // side gains 3; the others do not bind.
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_var(3.0, (0.0, f64::INFINITY));
    let y = problem.add_var(2.0, (0.0, f64::INFINITY));
    let c0 = problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
    let c1 = problem.add_constraint(&[(x, 1.0), (y, 3.0)], ComparisonOp::Le, 6.0);
    let c2 = problem.add_constraint(&[(x, 1.0)], ComparisonOp::Ge, 1.0);
    let solution = problem.solve().unwrap().into_solution().unwrap();
    assert!((solution.objective() - 12.0).abs() < 1e-9);
    assert!((solution.dual(c0).unwrap() - 3.0).abs() < 1e-9);
    assert!(solution.dual(c1).unwrap().abs() < 1e-9);
    assert!(solution.dual(c2).unwrap().abs() < 1e-9);
    // y at its lower bound: moving it up loses 3 - 2 = 1 per unit.
    assert!((solution.reduced_cost(y).unwrap() + 1.0).abs() < 1e-9);
    assert!(solution.reduced_cost(x).unwrap().abs() < 1e-9);

    // min 2x + y s.t. x + y >= 3, x - y = 1: x = 2, y = 1, objective 5.
    // d/d(3) = 3/2, d/d(1) = 1/2.
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_var(2.0, (0.0, f64::INFINITY));
    let y = problem.add_var(1.0, (0.0, f64::INFINITY));
    let ge = problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 3.0);
    let eq = problem.add_constraint(&[(x, 1.0), (y, -1.0)], ComparisonOp::Eq, 1.0);
    let solution = problem.solve().unwrap().into_solution().unwrap();
    assert!((solution.objective() - 5.0).abs() < 1e-9);
    assert!((solution.dual(ge).unwrap() - 1.5).abs() < 1e-9);
    assert!((solution.dual(eq).unwrap() - 0.5).abs() < 1e-9);
}

#[test]
fn duals_are_in_the_units_the_rows_are_written_in() {
    // The dispatch of the first test with its rows and columns written at
    // awkward scales: demand in kW (1e3 per MW of the vars), the first
    // generator in GW and the limit row multiplied by 1e-4. Each dual is the
    // objective per unit of its row as written.
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let g0 = problem.add_var(20.0e3, (0.0, f64::INFINITY)); // GW at 20 $/MW
    let g1 = problem.add_var(50.0, (0.0, f64::INFINITY));
    let demand = problem.add_constraint(&[(g0, 1e6), (g1, 1e3)], ComparisonOp::Eq, 10e3);
    let limit = problem.add_constraint(&[(g0, 1e-1)], ComparisonOp::Le, 6e-4);
    let solution = problem.solve().unwrap().into_solution().unwrap();
    assert!((solution.objective() - 320.0).abs() < 1e-9);
    // 50 $/MW is 0.05 $ per kW of demand; a unit of the limit row is 1e-2
    // GW, worth (50 - 20) * 1e4.
    assert!((solution.dual(demand).unwrap() - 0.05).abs() < 1e-12);
    assert!((solution.dual(limit).unwrap() + 3e5).abs() < 1e-6);
}

#[test]
fn rows_the_engine_does_not_hold_have_zero_duals() {
    for presolve in [true, false] {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_var(1.0, (0.0, 1.0));
        let y = problem.add_var(2.0, (0.0, 1.0));
        let empty = problem.add_constraint(LinearExpr::empty(), ComparisonOp::Le, 1.0);
        let open = problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, f64::INFINITY);
        // Implied by the bounds: presolve drops it.
        let implied = problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 5.0);
        let binding = problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 1.5);
        let options = SolveOptions {
            presolve,
            ..SolveOptions::default()
        };
        let solution = problem
            .solve_with(options)
            .unwrap()
            .into_solution()
            .unwrap();
        assert!((solution.objective() - 2.0).abs() < 1e-9);
        assert_eq!(solution.duals().unwrap().len(), 4);
        assert_eq!(solution.dual(empty), Some(0.0));
        assert_eq!(solution.dual(open), Some(0.0));
        assert!(solution.dual(implied).unwrap().abs() < 1e-12);
        assert!((solution.dual(binding).unwrap() - 2.0).abs() < 1e-9);
        // x at its upper bound, which prices it at 1 - 2.
        assert!((solution.reduced_cost(x).unwrap() + 1.0).abs() < 1e-9);
        assert!(solution.reduced_cost(y).unwrap().abs() < 1e-9);
    }
}

#[test]
fn an_edit_numbers_its_row_after_the_problems() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let cheap = problem.add_var(20.0, (0.0, f64::INFINITY));
    let dear = problem.add_var(50.0, (0.0, f64::INFINITY));
    let demand = problem.add_constraint(&[(cheap, 1.0), (dear, 1.0)], ComparisonOp::Eq, 10.0);
    let solution = problem.solve().unwrap().into_solution().unwrap();
    assert!((solution.dual(demand).unwrap() - 20.0).abs() < 1e-9);
    assert_eq!(solution.duals().unwrap().len(), 1);

    let solution = solution
        .add_constraint(&[(cheap, 1.0)], ComparisonOp::Le, 6.0)
        .unwrap()
        .into_solution()
        .unwrap();
    let duals = solution.duals().unwrap();
    assert_eq!(duals.len(), 2);
    assert!((duals[demand.idx()] - 50.0).abs() < 1e-9);
    assert!((duals[1] + 30.0).abs() < 1e-9);

    // Fixing the cheap unit below its limit frees the limit and leaves the
    // fix to carry the saving.
    let solution = solution
        .fix_var(cheap, 4.0)
        .unwrap()
        .into_solution()
        .unwrap();
    assert!((solution.dual(demand).unwrap() - 50.0).abs() < 1e-9);
    assert!(solution.duals().unwrap()[1].abs() < 1e-9);
    assert!((solution.reduced_cost(cheap).unwrap() + 30.0).abs() < 1e-9);
    let (outcome, was_fixed) = solution.unfix_var(cheap).unwrap();
    assert!(was_fixed);
    let solution = outcome.into_solution().unwrap();
    assert!((solution.duals().unwrap()[1] + 30.0).abs() < 1e-9);
}

#[test]
fn a_mip_has_no_duals() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_var(1.0, (0.0, 2.5));
    let n = problem.add_integer_var(2.0, (0, 10));
    let c = problem.add_constraint(&[(x, 1.0), (n, 1.0)], ComparisonOp::Le, 7.5);
    let solution = problem.solve().unwrap().into_solution().unwrap();
    assert_eq!(solution.dual(c), None);
    assert_eq!(solution.duals(), None);
    assert_eq!(solution.reduced_cost(x), None);
}

#[test]
fn random_scaled_lps_meet_the_optimality_conditions() {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xd0a1);
    let mut solved = 0;
    for case in 0..200 {
        let direction = if case % 2 == 0 {
            OptimizationDirection::Minimize
        } else {
            OptimizationDirection::Maximize
        };
        let n = rng.random_range(2..12);
        let m = rng.random_range(1..12);
        let lp = Lp::random(&mut rng, n, m, direction);
        for presolve in [true, false] {
            let solution = lp
                .solve(presolve)
                .expect("a model built around a point solves");
            check_optimality_conditions(&lp, &lp.bounds, &solution);
            solved += 1;
        }
    }
    assert_eq!(solved, 400);
}

#[test]
fn random_scaled_duals_are_subgradients() {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0x5b9);
    let (mut tight, mut predictions) = (0, 0);
    for case in 0..60 {
        let direction = if case % 2 == 0 {
            OptimizationDirection::Minimize
        } else {
            OptimizationDirection::Maximize
        };
        let n = rng.random_range(2..10);
        let m = rng.random_range(1..10);
        let mut lp = Lp::random(&mut rng, n, m, direction);
        let solution = lp.solve(true).unwrap();
        tight += check_subgradient(&mut lp, &solution);
        predictions += 2 * lp.rows.len();
    }
    // Most moves stay within the basis, where the prediction is exact.
    assert!(
        tight * 2 > predictions,
        "only {tight} of {predictions} predictions exact"
    );
}

#[test]
fn duals_after_a_fix_are_those_of_the_edited_problem() {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xf1c5);
    for case in 0..60 {
        let direction = if case % 2 == 0 {
            OptimizationDirection::Minimize
        } else {
            OptimizationDirection::Maximize
        };
        let n = rng.random_range(2..10);
        let m = rng.random_range(1..10);
        let lp = Lp::random(&mut rng, n, m, direction);
        let (problem, vars, _) = lp.build();
        let solution = problem.solve().unwrap().into_solution().unwrap();
        // Fix a var a little off its value, inside its bounds.
        let j = rng.random_range(0..n);
        let (lo, hi) = lp.bounds[j];
        let val = (solution.var_value(vars[j]) + 0.05 * (hi - lo)).min(hi);
        let Some(fixed) = solution
            .fix_var(vars[j], val)
            .ok()
            .and_then(|o| o.into_solution().ok())
        else {
            continue;
        };
        let mut bounds = lp.bounds.clone();
        bounds[j] = (val, val);
        check_optimality_conditions(&lp, &bounds, &fixed);
        let (unfixed, _) = fixed.unfix_var(vars[j]).unwrap();
        check_optimality_conditions(&lp, &lp.bounds, &unfixed.into_solution().unwrap());
    }
}
