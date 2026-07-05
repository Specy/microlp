//! Hand-constructed pure-LP cases with independently known answers: unique
//! vertices computed by hand, published classics (Klee-Minty, Beale), and
//! infeasible/unbounded variants.

use super::{Case, Tier};
use crate::model::{Builder, Expected};
use microlp::ComparisonOp::{Eq, Ge, Le};
use microlp::OptimizationDirection::{Maximize, Minimize};

const INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;

fn case(cases: &mut Vec<Case>, name: &str, build: impl Fn() -> (Builder, Expected) + 'static) {
    cases.push(Case::solve(name, Tier::Quick, 10, move || {
        let (b, expected) = build();
        Ok((b.spec, b.problem, expected))
    }));
}

pub fn register(cases: &mut Vec<Case>) {
    case(cases, "lp/readme-example", || {
        // max x + 2y; x >= 0, 0 <= y <= 3; x + y <= 4, 2x + y >= 2.
        // Unique optimum (1, 3), objective 7.
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, INF);
        let y = b.real(2.0, 0.0, 3.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 4.0);
        b.constraint(&[(x, 2.0), (y, 1.0)], Ge, 2.0);
        let e = Expected::unique(7.0, vec![(x, 1.0), (y, 3.0)]);
        (b, e)
    });

    case(cases, "lp/bounds-only-max", || {
        // No constraints: optimum sits on the variable bounds.
        let mut b = Builder::new(Maximize);
        let x = b.real(3.0, 1.0, 4.0);
        let y = b.real(2.0, -2.0, 5.0);
        let e = Expected::unique(22.0, vec![(x, 4.0), (y, 5.0)]);
        (b, e)
    });

    case(cases, "lp/bounds-only-min", || {
        let mut b = Builder::new(Minimize);
        let x = b.real(3.0, 1.0, 4.0);
        let y = b.real(2.0, -2.0, 5.0);
        let e = Expected::unique(-1.0, vec![(x, 1.0), (y, -2.0)]);
        (b, e)
    });

    case(cases, "lp/negative-cost-min", || {
        // min -x - 2y with x + y <= 10: pushes everything into y.
        let mut b = Builder::new(Minimize);
        let x = b.real(-1.0, 0.0, INF);
        let y = b.real(-2.0, 0.0, INF);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 10.0);
        let e = Expected::unique(-20.0, vec![(x, 0.0), (y, 10.0)]);
        (b, e)
    });

    case(cases, "lp/equality-unique-point", || {
        // Three independent equalities in three free variables: the feasible
        // set is exactly the point (3, 2, 1).
        let mut b = Builder::new(Minimize);
        let x = b.real(2.0, NEG_INF, INF);
        let y = b.real(3.0, NEG_INF, INF);
        let z = b.real(4.0, NEG_INF, INF);
        b.constraint(&[(x, 1.0), (y, 1.0), (z, 1.0)], Eq, 6.0);
        b.constraint(&[(x, 1.0), (y, -1.0)], Eq, 1.0);
        b.constraint(&[(x, 1.0), (z, 1.0)], Eq, 4.0);
        let e = Expected::unique(16.0, vec![(x, 3.0), (y, 2.0), (z, 1.0)]);
        (b, e)
    });

    case(cases, "lp/free-var-min", || {
        // y is free; min y with y >= x - 3 and x in [0, 1] lands at (0, -3).
        let mut b = Builder::new(Minimize);
        let x = b.real(0.0, 0.0, 1.0);
        let y = b.real(1.0, NEG_INF, INF);
        b.constraint(&[(y, 1.0), (x, -1.0)], Ge, -3.0);
        let e = Expected::objective(-3.0);
        (b, e)
    });

    case(cases, "lp/degenerate-vertex", || {
        // Five constraints all active at (2, 2): heavy degeneracy.
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, INF);
        let y = b.real(1.0, 0.0, INF);
        b.constraint(&[(x, 1.0)], Le, 2.0);
        b.constraint(&[(y, 1.0)], Le, 2.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 4.0);
        b.constraint(&[(x, 2.0), (y, 1.0)], Le, 6.0);
        b.constraint(&[(x, 1.0), (y, 2.0)], Le, 6.0);
        let e = Expected::unique(4.0, vec![(x, 2.0), (y, 2.0)]);
        (b, e)
    });

    case(cases, "lp/alternate-optima", || {
        // Whole edge x + y = 5 is optimal: only the objective is asserted.
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, 5.0);
        let y = b.real(1.0, 0.0, 5.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 5.0);
        let e = Expected::objective(5.0);
        (b, e)
    });

    case(cases, "lp/redundant-constraints", || {
        let mut b = Builder::new(Minimize);
        let x = b.real(1.0, 1.0, INF);
        let y = b.real(1.0, 1.0, INF);
        b.constraint(&[(x, 1.0), (y, 1.0)], Ge, 2.0); // implied by bounds
        b.constraint(&[(x, 2.0), (y, 2.0)], Ge, 4.0); // scaled duplicate
        let e = Expected::unique(2.0, vec![(x, 1.0), (y, 1.0)]);
        (b, e)
    });

    case(cases, "lp/zero-objective-feasibility", || {
        let mut b = Builder::new(Minimize);
        let x = b.real(0.0, 0.0, 2.0);
        let y = b.real(0.0, 0.0, 2.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Eq, 3.0);
        let e = Expected::objective(0.0);
        (b, e)
    });

    case(cases, "lp/tiny-coefficients", || {
        // min x + y with 1e-6*(x + y) >= 3  =>  objective 3e6.
        let mut b = Builder::new(Minimize);
        let x = b.real(1.0, 0.0, INF);
        let y = b.real(1.0, 0.0, INF);
        b.constraint(&[(x, 1e-6), (y, 1e-6)], Ge, 3.0);
        let e = Expected::objective_tol(
            3e6,
            crate::model::Tol {
                abs: 1e-3,
                rel: 1e-9,
            },
        );
        (b, e)
    });

    case(cases, "lp/huge-coefficients", || {
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, INF);
        b.constraint(&[(x, 1e9)], Le, 5e9);
        let e = Expected::unique(5.0, vec![(x, 5.0)]);
        (b, e)
    });

    case(cases, "lp/negative-rhs", || {
        // Mirrors the crate's classic `optimize` test: optimum (12, 8), 68.
        let mut b = Builder::new(Maximize);
        let x = b.real(3.0, 12.0, INF);
        let y = b.real(4.0, 5.0, INF);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 20.0);
        b.constraint(&[(y, -4.0), (x, 1.0)], Ge, -20.0);
        let e = Expected::unique(68.0, vec![(x, 12.0), (y, 8.0)]);
        (b, e)
    });

    case(cases, "lp/fixed-variable", || {
        // x fixed to 3 through equal bounds.
        let mut b = Builder::new(Maximize);
        let x = b.real(2.0, 3.0, 3.0);
        let y = b.real(1.0, 0.0, INF);
        b.constraint(&[(y, 1.0), (x, -1.0)], Le, 0.0);
        let e = Expected::unique(9.0, vec![(x, 3.0), (y, 3.0)]);
        (b, e)
    });

    case(cases, "lp/equality-with-bounds", || {
        // x + y = 10, x <= 4, y <= 7; max 5x + 4y = x + 40 => x = 4.
        let mut b = Builder::new(Maximize);
        let x = b.real(5.0, 0.0, 4.0);
        let y = b.real(4.0, 0.0, 7.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Eq, 10.0);
        let e = Expected::unique(44.0, vec![(x, 4.0), (y, 6.0)]);
        (b, e)
    });

    // Klee-Minty cubes: the classic worst case for Dantzig-rule simplex.
    // max sum 2^(n-j) x_j s.t. for each i: sum_{j<i} 2^(i-j+1) x_j + x_i <= 5^i.
    // Unique optimum: x_n = 5^n, all other x_j = 0; objective 5^n.
    for n in 3usize..=8 {
        let name = format!("lp/klee-minty-{}", n);
        cases.push(Case::solve(name, Tier::Quick, 10, move || {
            let mut b = Builder::new(Maximize);
            let vars: Vec<_> = (0..n)
                .map(|j| b.real(2f64.powi((n - 1 - j) as i32), 0.0, INF))
                .collect();
            for i in 1..=n {
                // 0-based j here; the textbook 1-based coefficient 2^(i-j+1)
                // becomes 2^(i-j) (e.g. row 2 is 4*x1 + x2 <= 25).
                let mut terms: Vec<_> = (0..i - 1)
                    .map(|j| (vars[j], 2f64.powi((i - j) as i32)))
                    .collect();
                terms.push((vars[i - 1], 1.0));
                b.constraint(&terms, Le, 5f64.powi(i as i32));
            }
            let opt = 5f64.powi(n as i32);
            let mut expected_vars: Vec<_> = vars.iter().map(|&v| (v, 0.0)).collect();
            expected_vars[n - 1].1 = opt;
            let e = Expected::unique(opt, expected_vars);
            Ok((b.spec, b.problem, e))
        }));
    }

    case(cases, "lp/beale-cycling", || {
        // Beale's classic cycling example; optimum -1/20. A solver without
        // anti-cycling safeguards loops forever here (the per-case time limit
        // turns that into a TIMEOUT instead of a hang).
        let mut b = Builder::new(Minimize);
        let a = b.real(-0.75, 0.0, INF);
        let c = b.real(150.0, 0.0, INF);
        let d = b.real(-0.02, 0.0, INF);
        let f = b.real(6.0, 0.0, INF);
        b.constraint(
            &[(a, 0.25), (c, -60.0), (d, -1.0 / 25.0), (f, 9.0)],
            Le,
            0.0,
        );
        b.constraint(&[(a, 0.5), (c, -90.0), (d, -1.0 / 50.0), (f, 3.0)], Le, 0.0);
        b.constraint(&[(d, 1.0)], Le, 1.0);
        let e = Expected::objective(-0.05);
        (b, e)
    });

    // ---- Infeasible LPs ----

    case(cases, "lp/infeasible-single-var", || {
        let mut b = Builder::new(Minimize);
        let x = b.real(1.0, 0.0, INF);
        b.constraint(&[(x, 1.0)], Ge, 5.0);
        b.constraint(&[(x, 1.0)], Le, 3.0);
        (b, Expected::Infeasible)
    });

    case(cases, "lp/infeasible-pair", || {
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, INF);
        let y = b.real(1.0, 0.0, INF);
        b.constraint(&[(x, 1.0), (y, 1.0)], Ge, 10.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 5.0);
        (b, Expected::Infeasible)
    });

    case(cases, "lp/infeasible-equalities", || {
        let mut b = Builder::new(Minimize);
        let x = b.real(1.0, NEG_INF, INF);
        let y = b.real(1.0, NEG_INF, INF);
        b.constraint(&[(x, 1.0), (y, 1.0)], Eq, 3.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Eq, 5.0);
        (b, Expected::Infeasible)
    });

    case(cases, "lp/infeasible-cycle", || {
        // x <= y - 1, y <= z - 1, z <= x - 1: summing gives 0 <= -3.
        let mut b = Builder::new(Minimize);
        let x = b.real(1.0, NEG_INF, INF);
        let y = b.real(0.0, NEG_INF, INF);
        let z = b.real(0.0, NEG_INF, INF);
        b.constraint(&[(x, 1.0), (y, -1.0)], Le, -1.0);
        b.constraint(&[(y, 1.0), (z, -1.0)], Le, -1.0);
        b.constraint(&[(z, 1.0), (x, -1.0)], Le, -1.0);
        (b, Expected::Infeasible)
    });

    case(cases, "lp/infeasible-bounds-vs-eq", || {
        let mut b = Builder::new(Minimize);
        let x = b.real(1.0, 0.0, 1.0);
        let y = b.real(1.0, 0.0, 1.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Eq, 3.0);
        (b, Expected::Infeasible)
    });

    // ---- Unbounded LPs ----

    case(cases, "lp/unbounded-single-var", || {
        let mut b = Builder::new(Maximize);
        let _x = b.real(1.0, 0.0, INF);
        (b, Expected::Unbounded)
    });

    case(cases, "lp/unbounded-ray", || {
        // max x - y with x - y >= 1: the ray (t+1, t) grows forever.
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, INF);
        let y = b.real(-1.0, 0.0, 5.0);
        b.constraint(&[(x, 1.0), (y, -1.0)], Ge, 1.0);
        (b, Expected::Unbounded)
    });

    case(cases, "lp/unbounded-free-var", || {
        // y free and absent from all constraints, min y.
        let mut b = Builder::new(Minimize);
        let x = b.real(0.0, 0.0, 1.0);
        let y = b.real(1.0, NEG_INF, INF);
        b.constraint(&[(x, 1.0)], Le, 1.0);
        let _ = y;
        (b, Expected::Unbounded)
    });

    case(cases, "lp/unbounded-along-equality", || {
        // x = y and min -(x + y): unbounded along the diagonal.
        let mut b = Builder::new(Minimize);
        let x = b.real(-1.0, 0.0, INF);
        let y = b.real(-1.0, 0.0, INF);
        b.constraint(&[(x, 1.0), (y, -1.0)], Eq, 0.0);
        (b, Expected::Unbounded)
    });
}
