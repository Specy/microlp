//! Warm edits of a solution's right-hand sides and objective coefficients
//! ([`Solution::set_rhs`], [`Solution::set_obj_coeff`]) and the handle
//! [`Solution::add_constraint`] returns.
//!
//! An edit must answer what a cold solve of the edited problem answers: the
//! same objective (within the tolerances), or the same error. The random
//! models are those of the duals tests, with row and column factors of
//! `1e-2..1e2`, so the row factor an edited right-hand side is stored at and
//! the column factor an edited cost is stored at are both in play. The other
//! tests pin the rows the engine does not hold, whose edits are enforced at
//! the boundary rather than by the pivots.

use super::duals::{Lp, Row};
use crate::{
    ComparisonOp, Error, OptimizationDirection, Problem, Solution, SolveOptions, SolveOutcome,
};
use rand::prelude::*;

fn solution(outcome: Result<SolveOutcome, Error>) -> Solution {
    outcome.unwrap().into_solution().unwrap()
}

fn options(presolve: bool) -> SolveOptions {
    SolveOptions {
        presolve,
        ..SolveOptions::default()
    }
}

/// A cold solve of `lp`: its objective, or the error.
fn cold(lp: &Lp, presolve: bool) -> Result<f64, Error> {
    let (problem, _, _) = lp.build();
    let outcome = problem.solve_with(options(presolve))?;
    Ok(outcome.into_solution().unwrap().objective())
}

/// Whether a warm edit's answer is the cold solve's.
fn assert_same(warm: &Result<Solution, Error>, cold: &Result<f64, Error>, what: &str) {
    match (warm, cold) {
        (Ok(warm), Ok(cold)) => {
            let scale = 1.0 + cold.abs();
            assert!(
                (warm.objective() - cold).abs() <= 1e-7 * scale,
                "{what}: warm objective {} against cold {cold}",
                warm.objective()
            );
        }
        (Err(warm), Err(cold)) => assert_eq!(warm, cold, "{what}"),
        (warm, cold) => panic!(
            "{what}: warm {:?} against cold {cold:?}",
            warm.as_ref().map(|s| s.objective())
        ),
    }
}

fn direction(case: usize) -> OptimizationDirection {
    if case % 2 == 0 {
        OptimizationDirection::Minimize
    } else {
        OptimizationDirection::Maximize
    }
}

#[test]
fn a_chain_of_rhs_edits_answers_as_a_cold_solve() {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0x3e7);
    let (mut solved, mut infeasible) = (0, 0);
    for case in 0..120 {
        let n = rng.random_range(2..12);
        let m = rng.random_range(1..12);
        let mut lp = Lp::random(&mut rng, n, m, direction(case));
        let presolve = case % 4 < 2;
        let (problem, _, rows) = lp.build();
        let mut current = solution(problem.solve_with(options(presolve)));
        for edit in 0..4 {
            let i = rng.random_range(0..m);
            let old = lp.rows[i].rhs;
            // Mostly small moves, which keep the problem feasible; now and
            // then a large one, which may not.
            let step = if edit == 3 { 50.0 } else { 0.5 };
            let rhs = old + step * rng.random_range(-1.0..1.0) * (1.0 + old.abs());
            lp.rows[i].rhs = rhs;
            let warm = current
                .set_rhs(rows[i], rhs)
                .map(|o| o.into_solution().unwrap());
            assert_same(
                &warm,
                &cold(&lp, presolve),
                &format!("case {case} edit {edit}"),
            );
            match warm {
                Ok(next) => {
                    solved += 1;
                    current = next;
                }
                Err(_) => {
                    infeasible += 1;
                    break;
                }
            }
        }
    }
    assert!(
        solved > 250 && infeasible > 0,
        "{solved} solved, {infeasible} infeasible"
    );
}

#[test]
fn a_chain_of_cost_edits_answers_as_a_cold_solve() {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0xc057);
    for case in 0..120 {
        let n = rng.random_range(2..12);
        let m = rng.random_range(1..12);
        let mut lp = Lp::random(&mut rng, n, m, direction(case));
        let presolve = case % 4 < 2;
        let (problem, vars, _) = lp.build();
        let mut current = solution(problem.solve_with(options(presolve)));
        for edit in 0..4 {
            let j = rng.random_range(0..n);
            // The box is finite, so every objective has an optimum.
            let coeff = rng.random_range(-10.0..10.0) * lp.obj[j].abs().max(1e-2) / 5.0;
            lp.obj[j] = coeff;
            let warm = current
                .set_obj_coeff(vars[j], coeff)
                .map(|o| o.into_solution().unwrap());
            assert_same(
                &warm,
                &cold(&lp, presolve),
                &format!("case {case} edit {edit}"),
            );
            current = warm.unwrap();
        }
    }
}

#[test]
fn warm_edits_take_fewer_pivots_than_cold_solves() {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0x9a7);
    let (mut warm_pivots, mut cold_pivots) = (0, 0);
    for case in 0..40 {
        let mut lp = Lp::random(&mut rng, 30, 25, direction(case));
        let (problem, vars, rows) = lp.build();
        let first = solution(problem.solve());
        let i = rng.random_range(0..lp.rows.len());
        lp.rows[i].rhs += 0.1 * (1.0 + lp.rows[i].rhs.abs());
        let j = rng.random_range(0..lp.obj.len());
        lp.obj[j] *= 1.5;
        let before = first.stats().lp_iterations;
        let Ok(outcome) = first.set_rhs(rows[i], lp.rows[i].rhs) else {
            continue;
        };
        let edited = solution(
            outcome
                .into_solution()
                .unwrap()
                .set_obj_coeff(vars[j], lp.obj[j]),
        );
        warm_pivots += edited.stats().lp_iterations - before;
        let (problem, _, _) = lp.build();
        cold_pivots += solution(problem.solve()).stats().lp_iterations;
    }
    assert!(
        warm_pivots * 3 < cold_pivots,
        "warm edits took {warm_pivots} pivots, cold solves {cold_pivots}"
    );
}

#[test]
fn an_infinite_rhs_switches_a_constraint_off_and_a_finite_one_back_on() {
    for presolve in [true, false] {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, 10.0));
        let y = problem.add_var(1.0, (0.0, 10.0));
        let cap = problem.add_constraint(&[(x, 1.0), (y, 2.0)], ComparisonOp::Le, 4.0);
        let floor = problem.add_constraint(&[(x, 1.0), (y, -1.0)], ComparisonOp::Ge, -2.0);
        let sol = solution(problem.solve_with(options(presolve)));
        assert!((sol.objective() - 4.0).abs() < 1e-9);

        let sol = solution(sol.set_rhs(cap, f64::INFINITY));
        assert!((sol.objective() - 20.0).abs() < 1e-9);
        assert_eq!(sol.dual(cap), Some(0.0));
        let sol = solution(sol.set_rhs(floor, f64::NEG_INFINITY));
        assert!((sol.objective() - 20.0).abs() < 1e-9);

        let sol = solution(sol.set_rhs(cap, 6.0));
        assert!((sol.objective() - 6.0).abs() < 1e-9);
        assert!((sol[x] + 2.0 * sol[y] - 6.0).abs() < 1e-9);
        assert!((sol.dual(cap).unwrap() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn a_row_every_point_satisfies_binds_once_its_rhs_is_finite() {
    for presolve in [true, false] {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(1.0, (0.0, 10.0));
        let open = problem.add_constraint(&[(x, 1.0)], ComparisonOp::Le, f64::INFINITY);
        let empty = problem.add_constraint(&[], ComparisonOp::Le, 1.0);
        let sol = solution(problem.solve_with(options(presolve)));
        assert!((sol[x] - 10.0).abs() < 1e-9);

        let sol = solution(sol.set_rhs(open, 3.0));
        assert!((sol[x] - 3.0).abs() < 1e-9, "presolve {presolve}");
        assert!((sol.dual(open).unwrap() - 1.0).abs() < 1e-9);
        let sol = solution(sol.set_rhs(open, 5.0));
        assert!((sol[x] - 5.0).abs() < 1e-9);

        // `0 <= -1` holds at no point.
        assert_eq!(sol.set_rhs(empty, -1.0).unwrap_err(), Error::Infeasible);
    }
}

#[test]
fn a_row_presolve_dropped_binds_once_its_rhs_cuts_the_box() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_var(1.0, (0.0, 2.0));
    let y = problem.add_var(1.0, (0.0, 2.0));
    // Implied by the bounds, so presolve drops it.
    let implied = problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 10.0);
    let sol = solution(problem.solve());
    assert!((sol.objective() - 4.0).abs() < 1e-9);
    assert_eq!(sol.dual(implied), Some(0.0));

    let sol = solution(sol.set_rhs(implied, 3.0));
    assert!((sol.objective() - 3.0).abs() < 1e-9);
    assert!((sol.dual(implied).unwrap() - 1.0).abs() < 1e-9);
    // Held by the engine now: a second edit goes through the pivots.
    let sol = solution(sol.set_rhs(implied, 1.0));
    assert!((sol.objective() - 1.0).abs() < 1e-9);
    assert_eq!(sol.set_rhs(implied, -1.0).unwrap_err(), Error::Infeasible);
}

#[test]
fn a_constraint_added_after_the_solve_can_be_priced_and_edited() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let a = problem.add_var(20.0, (0.0, f64::INFINITY));
    let b = problem.add_var(50.0, (0.0, f64::INFINITY));
    let demand = problem.add_constraint(&[(a, 1.0), (b, 1.0)], ComparisonOp::Ge, 10.0);
    let sol = solution(problem.solve());
    let (outcome, limit) = sol
        .add_constraint(&[(a, 1.0)], ComparisonOp::Le, 6.0)
        .unwrap();
    assert_eq!(limit.idx(), 1);
    let sol = outcome.into_solution().unwrap();
    assert!((sol.dual(limit).unwrap() + 30.0).abs() < 1e-9);
    assert!((sol.dual(demand).unwrap() - 50.0).abs() < 1e-9);

    let sol = solution(sol.set_rhs(limit, 8.0));
    assert!((sol.objective() - (8.0 * 20.0 + 2.0 * 50.0)).abs() < 1e-9);
    let sol = solution(sol.set_obj_coeff(b, 10.0));
    assert!((sol[b] - 10.0).abs() < 1e-9);
    assert!((sol.dual(limit).unwrap()).abs() < 1e-9);
}

#[test]
fn a_cost_edit_can_make_the_problem_unbounded() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_var(1.0, (0.0, f64::INFINITY));
    let y = problem.add_var(1.0, (0.0, 4.0));
    problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 2.0);
    let sol = solution(problem.solve());
    let sol = solution(sol.set_obj_coeff(y, -1.0));
    assert!((sol.objective() + 4.0).abs() < 1e-9);
    assert_eq!(sol.set_obj_coeff(x, -1.0).unwrap_err(), Error::Unbounded);
}

#[test]
fn edits_refuse_numbers_no_solve_takes() {
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let x = problem.add_var(1.0, (0.0, 4.0));
    let le = problem.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 3.0);
    let ge = problem.add_constraint(&[(x, 1.0)], ComparisonOp::Ge, 1.0);
    let eq = problem.add_constraint(&[(x, 2.0)], ComparisonOp::Eq, 4.0);
    let sol = solution(problem.solve());
    let refused = |result: Result<SolveOutcome, Error>| {
        assert!(
            matches!(result, Err(Error::InvalidOperation(_))),
            "{result:?}"
        );
    };
    refused(sol.clone().set_rhs(le, f64::NAN));
    refused(sol.clone().set_rhs(le, f64::NEG_INFINITY));
    refused(sol.clone().set_rhs(ge, f64::INFINITY));
    refused(sol.clone().set_rhs(eq, f64::INFINITY));
    refused(sol.clone().set_obj_coeff(x, f64::NAN));
    refused(sol.clone().set_obj_coeff(x, f64::INFINITY));
    match sol.set_rhs(eq, f64::NAN) {
        Err(Error::InvalidOperation(message)) => {
            assert!(message.contains("constraint 2"), "{message}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn mip_edits_answer_as_a_cold_solve() {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(0x31f);
    for case in 0..40 {
        let n = rng.random_range(2..8);
        let m = rng.random_range(1..6);
        let mut lp = Lp::random(&mut rng, n, m, direction(case));
        // Integer vars on boxes holding several integers.
        for b in &mut lp.bounds {
            *b = (b.0.floor() - 1.0, b.1.ceil() + 1.0);
        }
        let build = |lp: &Lp| {
            let mut problem = Problem::new(lp.direction);
            let vars: Vec<_> = lp
                .obj
                .iter()
                .zip(&lp.bounds)
                .map(|(&c, &(lo, hi))| problem.add_integer_var(c, (lo as i32, hi as i32)))
                .collect();
            let rows: Vec<_> = lp
                .rows
                .iter()
                .map(|Row { terms, op, rhs }| {
                    let expr: Vec<_> = terms.iter().map(|&(v, a)| (vars[v], a)).collect();
                    problem.add_constraint(&expr[..], *op, *rhs)
                })
                .collect();
            (problem, vars, rows)
        };
        let cold = |lp: &Lp| {
            let (problem, _, _) = build(lp);
            problem
                .solve()
                .map(|o| o.into_solution().unwrap().objective())
        };
        let (problem, vars, rows) = build(&lp);
        let Ok(outcome) = problem.solve() else {
            continue;
        };
        let sol = outcome.into_solution().unwrap();
        let i = rng.random_range(0..m);
        lp.rows[i].rhs += rng.random_range(-1.0..1.0) * (1.0 + lp.rows[i].rhs.abs());
        let warm = sol
            .clone()
            .set_rhs(rows[i], lp.rows[i].rhs)
            .map(|o| o.into_solution().unwrap());
        assert_same(&warm, &cold(&lp), &format!("case {case} rhs"));
        let j = rng.random_range(0..n);
        let Ok(sol) = warm else { continue };
        lp.obj[j] = -lp.obj[j] + 1.0;
        let warm = sol
            .set_obj_coeff(vars[j], lp.obj[j])
            .map(|o| o.into_solution().unwrap());
        assert_same(&warm, &cold(&lp), &format!("case {case} cost"));
    }
}

#[test]
fn a_constraint_added_open_binds_once_its_rhs_is_finite() {
    let mut problem = Problem::new(OptimizationDirection::Maximize);
    let x = problem.add_var(1.0, (0.0, 10.0));
    let y = problem.add_var(1.0, (0.0, 10.0));
    problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 15.0);
    let sol = solution(problem.solve());
    // Every point satisfies it as added, so the engine does not hold it.
    let (outcome, open) = sol
        .add_constraint(&[(x, 1.0), (y, -1.0)], ComparisonOp::Ge, f64::NEG_INFINITY)
        .unwrap();
    let sol = outcome.into_solution().unwrap();
    assert!((sol.objective() - 15.0).abs() < 1e-9);
    let sol = solution(sol.set_rhs(open, 3.0));
    assert!((sol.objective() - 15.0).abs() < 1e-9);
    assert!(sol[x] - sol[y] >= 3.0 - 1e-9);
    let sol = solution(sol.set_rhs(open, 9.0));
    assert!((sol[x] - 10.0).abs() < 1e-9 && (sol[y] - 1.0).abs() < 1e-9);
    assert!((sol.objective() - 11.0).abs() < 1e-9);
}
