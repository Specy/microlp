//! WebAssembly example for microlp: a thin `wasm-bindgen` layer that solves an
//! LP and a MILP and hands the results back to JavaScript. See `test.js`.
//!
//! Each solve sets a time limit, which drives the solver's deadline checks
//! through `web_time::Instant` — the browser/JS clock on `wasm32-unknown-unknown`.
//! That is the whole point of the example: it proves microlp genuinely *runs*
//! (clock and all) under wasm, not merely that it compiles.

use core::time::Duration;
use microlp::{ComparisonOp, OptimizationDirection, Problem};
use wasm_bindgen::prelude::*;

/// Continuous LP (the textbook Reddy Mikks model):
///
/// ```text
/// maximize 3x + 5y
/// subject to  x        <= 4
///                 2y   <= 12
///             3x + 2y  <= 18
///             x, y     >= 0
/// ```
///
/// Unique optimum: objective 36 at (x = 2, y = 6).
/// Returns `[objective, x, y]`.
#[wasm_bindgen]
pub fn solve_lp() -> Vec<f64> {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_var(3.0, (0.0, f64::INFINITY));
    let y = p.add_var(5.0, (0.0, f64::INFINITY));
    p.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 4.0);
    p.add_constraint(&[(y, 2.0)], ComparisonOp::Le, 12.0);
    p.add_constraint(&[(x, 3.0), (y, 2.0)], ComparisonOp::Le, 18.0);
    p.set_time_limit(Duration::from_secs(5));

    let sol = p
        .solve()
        .expect("LP is feasible and bounded")
        .into_solution()
        .expect("the five-second budget must produce an LP solution");
    vec![sol.objective(), sol.var_value(x), sol.var_value(y)]
}

/// Mixed-integer program exercising branch & bound:
///
/// ```text
/// minimize 2x + 3y
/// subject to  x + y >= 10
///             0 <= x, y <= 8,  x, y integer
/// ```
///
/// Unique optimum: objective 22 at (x = 8, y = 2).
/// Returns `[objective, x, y]`.
#[wasm_bindgen]
pub fn solve_milp() -> Vec<f64> {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_integer_var(2.0, (0, 8));
    let y = p.add_integer_var(3.0, (0, 8));
    p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 10.0);
    p.set_time_limit(Duration::from_secs(10));

    let sol = p
        .solve()
        .expect("MILP is feasible and bounded")
        .into_solution()
        .expect("the ten-second budget must produce a MILP solution");
    vec![sol.objective(), sol.var_value(x), sol.var_value(y)]
}
