//! What the port of special ordered sets onto the current search adds beyond
//! the set logic itself (`src/tests/sos.rs`, from Specy/microlp#50): presolve
//! leaves integer members alone, a loss model solves as the sets say, and
//! the sets hold across the feasibility tolerance.

use crate::{
    ComparisonOp, Error, OptimizationDirection, Problem, Solution, SolveOptions, Tolerances,
    Variable,
};

fn solve(problem: &Problem) -> Solution {
    problem.solve().unwrap().into_solution().unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-7 * (1.0 + a.abs().max(b.abs()))
}

/// Dual fixing reads only the rows: with nothing in them stopping `x` or `y`
/// from rising, it would fix both at 5, and every child of the set's split
/// would then have crossed bounds, `Infeasible` for a model whose optimum is
/// 5. Every member of a set is locked against it.
#[test]
fn presolve_leaves_integer_members_of_a_set_free() {
    for presolve in [true, false] {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_integer_var(1.0, (0, 5));
        let y = p.add_integer_var(1.0, (0, 5));
        p.add_sos1_constraint([(x, 1.0), (y, 2.0)]);
        let options = SolveOptions {
            presolve,
            ..SolveOptions::default()
        };
        let solution = p.solve_with(options).unwrap().into_solution().unwrap();
        assert_eq!(solution.objective(), 5.0, "presolve {presolve}");
        assert_eq!(solution.var_value(x) * solution.var_value(y), 0.0);
    }
}

/// Two regions joined by an interconnector whose losses follow a convex
/// curve, given by its breakpoints as weights `λ_i` of an SOS2. Region A has
/// a must-run surplus (`gA ≥ 80`, paid to generate) and no load, so it must
/// export over a link limited to 80; region B has a load of 20 and a
/// generator paid to generate too. Both prices are negative, so energy is
/// worth burning: without the set, the LP mixes the first and last
/// breakpoints, putting 80 on the link and losing all of it, a point the
/// curve does not allow.
struct Losses {
    problem: Problem,
    lambda: Vec<Variable>,
    g_a: Variable,
    g_b: Variable,
}

const BREAKPOINTS: [f64; 5] = [0.0, 25.0, 50.0, 75.0, 100.0];

/// The model, with the SOS2 over the weights if `with_set`.
fn losses(with_set: bool) -> Losses {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let g_a = problem.add_var(-10.0, (80.0, 100.0));
    let g_b = problem.add_var(-5.0, (0.0, 100.0));
    let lambda: Vec<Variable> = (0..BREAKPOINTS.len())
        .map(|_| problem.add_var(0.0, (0.0, 1.0)))
        .collect();
    let loss = |t: f64| 0.01 * t * t;
    problem.add_constraint(lambda.iter().map(|&l| (l, 1.0)), ComparisonOp::Eq, 1.0);
    // The link limit, on the flow `Σ t_i λ_i`.
    problem.add_constraint(
        lambda.iter().zip(BREAKPOINTS).map(|(&l, t)| (l, t)),
        ComparisonOp::Le,
        80.0,
    );
    // A: gA = flow.
    let mut a: Vec<(Variable, f64)> = vec![(g_a, 1.0)];
    a.extend(lambda.iter().zip(BREAKPOINTS).map(|(&l, t)| (l, -t)));
    problem.add_constraint(a, ComparisonOp::Eq, 0.0);
    // B: flow - loss + gB = 20.
    let mut b: Vec<(Variable, f64)> = vec![(g_b, 1.0)];
    b.extend(
        lambda
            .iter()
            .zip(BREAKPOINTS)
            .map(|(&l, t)| (l, t - loss(t))),
    );
    problem.add_constraint(b, ComparisonOp::Eq, 20.0);
    if with_set {
        problem.add_sos2_constraint(lambda.iter().zip(BREAKPOINTS).map(|(&l, t)| (l, t)));
    }
    Losses {
        problem,
        lambda,
        g_a,
        g_b,
    }
}

#[test]
fn sos2_losses_stop_the_lp_burning_energy_on_the_link() {
    let relaxed = losses(false);
    let lp = solve(&relaxed.problem);
    let on = |s: &Solution, i: usize| s.var_value(relaxed.lambda[i]) > 1e-7;
    assert!(
        on(&lp, 0) && on(&lp, 4),
        "the LP mixes the outer breakpoints"
    );
    assert!(close(lp.objective(), -900.0), "{}", lp.objective());

    let model = losses(true);
    let solution = solve(&model.problem);
    // 80 on the link, between the breakpoints at 75 and 100: the curve's
    // loss there is 56.25 + 5 * 1.75 = 65, so 15 arrives and gB makes 5.
    assert!(
        close(solution.objective(), -825.0),
        "{}",
        solution.objective()
    );
    let lambda: Vec<f64> = model
        .lambda
        .iter()
        .map(|&l| solution.var_value(l))
        .collect();
    assert!(lambda[..3].iter().all(|x| x.abs() <= 1e-7), "{lambda:?}");
    assert!(close(solution.var_value(model.g_a), 80.0));
    assert!(close(solution.var_value(model.g_b), 5.0));
}

/// A set holds within the bound contract whatever the feasibility tolerance:
/// small piecewise-linear models over nonconvex curves, solved at tolerances
/// from `1e-9` to `1e-5`, match the best adjacent pair of breakpoints, and
/// read no member beyond the tolerance outside its pair.
#[test]
fn sets_hold_across_the_feasibility_tolerance() {
    let mut seed: u64 = 0x50_5502;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 10_000) as f64 / 10_000.0
    };
    for case in 0..30 {
        let n = 4 + case % 4;
        // Breakpoints a curve's width apart, scaled so a member's share of a
        // row spans orders of magnitude across cases.
        let scale = 10f64.powi((case % 5) as i32 - 2);
        let t: Vec<f64> = (0..n).map(|i| i as f64 * scale).collect();
        let f: Vec<f64> = (0..n).map(|_| 10.0 * next() - 5.0).collect();
        let x = (0.2 + 0.6 * next()) * t[n - 1];
        let build = |hold: Option<usize>| {
            let mut p = Problem::new(OptimizationDirection::Minimize);
            let lam: Vec<Variable> = f.iter().map(|&fi| p.add_var(fi, (0.0, 1.0))).collect();
            p.add_constraint(lam.iter().map(|&l| (l, 1.0)), ComparisonOp::Eq, 1.0);
            p.add_constraint(
                lam.iter().zip(&t).map(|(&l, &ti)| (l, ti)),
                ComparisonOp::Eq,
                x,
            );
            match hold {
                Some(i) => {
                    for (j, &l) in lam.iter().enumerate() {
                        if j != i && j != i + 1 {
                            p.add_constraint([(l, 1.0)], ComparisonOp::Eq, 0.0);
                        }
                    }
                }
                None => p.add_sos2_constraint(lam.iter().zip(&t).map(|(&l, &ti)| (l, ti))),
            }
            (p, lam)
        };
        let want = (0..n - 1)
            .filter_map(|i| match build(Some(i)).0.solve() {
                Ok(outcome) => Some(outcome.into_solution().unwrap().objective()),
                Err(Error::Infeasible) => None,
                Err(e) => panic!("case {case}: the held LP failed: {e:?}"),
            })
            .fold(f64::INFINITY, f64::min);
        for feasibility in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5] {
            let (p, lam) = build(None);
            let mut tolerances = Tolerances::default();
            tolerances.feasibility = feasibility;
            let options = SolveOptions {
                tolerances,
                ..SolveOptions::default()
            };
            let solution = p.solve_with(options).unwrap().into_solution().unwrap();
            let on: Vec<usize> = (0..n)
                .filter(|&i| solution.var_value(lam[i]).abs() > feasibility)
                .collect();
            assert!(
                on.len() <= 2 && on.windows(2).all(|w| w[1] == w[0] + 1),
                "case {case} at {feasibility:e}: members {on:?} beyond the tolerance"
            );
            // The tolerance lets each row and member move by about itself;
            // the objective moves by that times the costs and breakpoints.
            let slack = 1e-6 + 100.0 * feasibility * (1.0 + 1.0 / scale);
            assert!(
                (solution.objective() - want).abs() <= slack,
                "case {case} at {feasibility:e}: {} vs {want}",
                solution.objective()
            );
        }
    }
}

/// A member counts as zero within the bound contract at `[0, 0]`, no more:
/// `y` at `5e-4`, the relaxation's value, is a member the set must switch
/// off, while one at `5e-8`, within the default `1e-7`, is zero as a bound
/// at zero holds it, and is reported as computed.
#[test]
fn a_member_counts_as_zero_within_the_bound_tolerance_only() {
    for (y_max, branches) in [(5e-4, true), (5e-8, false)] {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_var(1.0, (0.0, 1.0));
        let y = p.add_var(1.0, (0.0, y_max));
        p.add_sos1_constraint([(x, 1.0), (y, 2.0)]);
        let solution = solve(&p);
        assert_eq!(solution.var_value(x), 1.0);
        if branches {
            assert!(
                solution.var_value(y).abs() <= 1e-7,
                "y {}",
                solution.var_value(y)
            );
            assert!(close(solution.objective(), 1.0));
            assert!(solution.stats().nodes_solved > 0);
        } else {
            assert_eq!(solution.var_value(y), y_max);
            assert_eq!(solution.stats().nodes_solved, 0);
        }
    }
}

/// A problem with a set solves through branch & bound, whose solutions carry
/// no duals yet, even with no integer var.
#[test]
fn a_set_solution_has_no_duals() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_var(1.0, (0.0, 1.0));
    let y = p.add_var(2.0, (0.0, 1.0));
    let row = p.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 1.5);
    p.add_sos1_constraint([(x, 1.0), (y, 2.0)]);
    let solution = solve(&p);
    assert!(close(solution.objective(), 2.0));
    assert_eq!(solution.dual(row), None);
    assert_eq!(solution.duals(), None);
    assert_eq!(solution.reduced_cost(x), None);
}
