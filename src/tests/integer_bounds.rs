//! Integer variables with infinite bounds ([`Problem::add_integer_var_f64`]).
//!
//! `add_integer_var` takes `i32` bounds, and `i32::MAX` is a bound like any
//! other: an objective that improves along an integer ray stops there with a
//! finite "optimum" (the suite's `milp/huge-int-bound` pins that). These tests
//! cover the variable with no bound on a side: the ray is answered
//! `Unbounded`, a free integer bounded by the rows is solved, the search
//! branches on such a variable, and a fractional bound is refused.

use crate::{ComparisonOp, Error, OptimizationDirection, Problem, SolveOptions, SolveOutcome};

const INF: f64 = f64::INFINITY;

fn solve(p: &Problem, presolve: bool) -> Result<SolveOutcome, Error> {
    p.solve_with(SolveOptions {
        presolve,
        node_limit: Some(10_000),
        ..SolveOptions::default()
    })
}

fn objective(p: &Problem, presolve: bool) -> f64 {
    match solve(p, presolve) {
        Ok(SolveOutcome::Solution(s)) => s.objective(),
        other => panic!("presolve {presolve}: expected a solution, got {other:?}"),
    }
}

/// `max n` over an integer `n >= 0` with no upper bound and no row: the
/// objective improves forever.
#[test]
fn an_integer_ray_with_no_row_is_unbounded() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    p.add_integer_var_f64(1.0, (0.0, INF));
    for presolve in [true, false] {
        assert!(
            matches!(solve(&p, presolve), Err(Error::Unbounded)),
            "presolve {presolve}"
        );
    }
}

/// The same ray through the rows, the shape of popso's `unbdmilp` family:
/// `max x + y` with `x - y <= 1.5`, `y - x <= 2.5` over integers `x, y >= 0`
/// has the integral ray `(1, 1)`. With `i32::MAX` bounds the search instead
/// proves a finite optimum near `2^32`; with infinite bounds it is unbounded.
#[test]
fn an_integer_ray_through_rows_is_unbounded() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_integer_var_f64(1.0, (0.0, INF));
    let y = p.add_integer_var_f64(1.0, (0.0, INF));
    p.add_constraint(&[(x, 1.0), (y, -1.0)], ComparisonOp::Le, 1.5);
    p.add_constraint(&[(x, -1.0), (y, 1.0)], ComparisonOp::Le, 2.5);
    for presolve in [true, false] {
        assert!(
            matches!(solve(&p, presolve), Err(Error::Unbounded)),
            "presolve {presolve}"
        );
    }

    // The i32 constructor keeps its finite bound: the contrast this PR documents.
    let mut q = Problem::new(OptimizationDirection::Maximize);
    let x = q.add_integer_var(1.0, (0, i32::MAX));
    let y = q.add_integer_var(1.0, (0, i32::MAX));
    q.add_constraint(&[(x, 1.0), (y, -1.0)], ComparisonOp::Le, 1.5);
    q.add_constraint(&[(x, -1.0), (y, 1.0)], ComparisonOp::Le, 2.5);
    let z = objective(&q, true);
    assert!(z > 4.0e9, "objective {z}");
}

/// A free integer bounded only by rows: `min x` with `x >= -3.5` gives `-3`,
/// `max x` with `x <= 7.25` gives `7`, from both sides of zero.
#[test]
fn a_free_integer_bounded_by_rows_is_solved() {
    for presolve in [true, false] {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var_f64(1.0, (-INF, INF));
        p.add_constraint(&[(x, 1.0)], ComparisonOp::Ge, -3.5);
        assert_eq!(objective(&p, presolve), -3.0, "presolve {presolve}");

        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = p.add_integer_var_f64(1.0, (-INF, INF));
        let y = p.add_var(0.0, (0.0, 1.0));
        p.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Le, 14.5);
        assert_eq!(objective(&p, presolve), 7.0, "presolve {presolve}");
    }
}

/// The search branches on variables with an infinite bound: `min x + y` with
/// `2x + 2y >= 3` over integers `x, y >= 0` has LP optimum `1.5` and integer
/// optimum `2`; `min x - y` with `3x - 2y >= 1.5`, `x <= 10` needs branching
/// on a free integer on both sides.
#[test]
fn the_search_branches_on_integers_with_infinite_bounds() {
    for presolve in [true, false] {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var_f64(1.0, (0.0, INF));
        let y = p.add_integer_var_f64(1.0, (0.0, INF));
        p.add_constraint(&[(x, 2.0), (y, 2.0)], ComparisonOp::Ge, 3.0);
        assert_eq!(objective(&p, presolve), 2.0, "presolve {presolve}");

        // y <= (3x - 1.5) / 2 and x <= 10: x = 10 gives y <= 14.25, so y = 14
        // and x - y = -4.
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_integer_var_f64(1.0, (-INF, 10.0));
        let y = p.add_integer_var_f64(-1.0, (-INF, INF));
        p.add_constraint(&[(x, 3.0), (y, -2.0)], ComparisonOp::Ge, 1.5);
        assert_eq!(objective(&p, presolve), -4.0, "presolve {presolve}");
    }
}

/// Whole-number `f64` bounds give the same variable as `add_integer_var`.
#[test]
fn whole_number_f64_bounds_match_the_i32_constructor() {
    let build = |f64_bounds: bool| {
        let mut p = Problem::new(OptimizationDirection::Maximize);
        let x = if f64_bounds {
            p.add_integer_var_f64(3.0, (-2.0, 5.0))
        } else {
            p.add_integer_var(3.0, (-2, 5))
        };
        let y = p.add_var(1.0, (0.0, INF));
        p.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Le, 9.5);
        p
    };
    for presolve in [true, false] {
        assert_eq!(
            objective(&build(true), presolve),
            objective(&build(false), presolve)
        );
    }
}

/// A fractional or NaN finite bound on an integer variable is refused at
/// solve time, as other numbers no solve can take are.
#[test]
fn a_fractional_integer_bound_is_refused() {
    for bounds in [
        (0.5, 3.0),
        (0.0, 2.5),
        (f64::NAN, 3.0),
        (0.0, f64::NAN),
        (INF, INF),
    ] {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        p.add_integer_var_f64(1.0, bounds);
        assert!(
            matches!(p.solve(), Err(Error::InvalidOperation(_))),
            "bounds {bounds:?}"
        );
    }
}
