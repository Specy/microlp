//! Randomly generated LPs whose optimum is *certified by construction*
//! (a strong-duality argument with all-integer data), plus metamorphic
//! invariants that cross-check the solver against itself.

use super::{Case, Tier};
use crate::model::{Builder, Expected, Tol};
use crate::rng::Rng;
use microlp::ComparisonOp::{Eq, Ge, Le};
use microlp::OptimizationDirection::{Maximize, Minimize};
use microlp::{ComparisonOp, OptimizationDirection, Problem, Variable};

/// Build `max c'x  s.t.  A x <= b, x >= 0` where:
///   - A is n x n, integer, strictly diagonally dominant (hence nonsingular),
///   - x* is a random nonnegative integer point and b := A x*,
///   - c := column sums of A, i.e. c = A' * 1.
///
/// Then y* = 1 is dual feasible with A'y* = c, and complementary slackness
/// certifies x* optimal with objective c'x*. Because every y*_i > 0, any
/// optimal point must satisfy A x = b, and A is nonsingular, so x* is the
/// UNIQUE optimum — variable values can be asserted exactly.
pub(super) struct CertifiedLp {
    pub(super) a: Vec<Vec<i64>>,
    pub(super) x_star: Vec<i64>,
    pub(super) b: Vec<i64>,
    pub(super) c: Vec<i64>,
    pub(super) objective: i64,
}

pub(super) fn certified_instance(n: usize, seed: u64) -> CertifiedLp {
    let mut rng = Rng::new(seed);
    let mut a = vec![vec![0i64; n]; n];
    for i in 0..n {
        let mut off_diag_sum = 0;
        for j in 0..n {
            if i != j && rng.int(0, 3) == 0 {
                a[i][j] = rng.int(1, 3);
                off_diag_sum += a[i][j];
            }
        }
        a[i][i] = off_diag_sum + rng.int(1, 5); // strict diagonal dominance
    }
    let x_star: Vec<i64> = (0..n).map(|_| rng.int(0, 9)).collect();
    let b: Vec<i64> = a
        .iter()
        .map(|row| row.iter().zip(&x_star).map(|(v, x)| v * x).sum())
        .collect();
    let c: Vec<i64> = (0..n).map(|j| (0..n).map(|i| a[i][j]).sum()).collect();
    let objective = c.iter().zip(&x_star).map(|(c, x)| c * x).sum();
    CertifiedLp {
        a,
        x_star,
        b,
        c,
        objective,
    }
}

fn build_certified(inst: &CertifiedLp) -> (Builder, Vec<Variable>) {
    let n = inst.x_star.len();
    let mut bld = Builder::new(Maximize);
    let vars: Vec<_> = (0..n)
        .map(|j| bld.real(inst.c[j] as f64, 0.0, f64::INFINITY))
        .collect();
    for i in 0..n {
        let terms: Vec<_> = (0..n)
            .filter(|&j| inst.a[i][j] != 0)
            .map(|j| (vars[j], inst.a[i][j] as f64))
            .collect();
        bld.constraint(&terms, Le, inst.b[i] as f64);
    }
    (bld, vars)
}

/// A feasible-by-construction random LP for the metamorphic checks (optimum
/// not known a priori; invariants compare solver runs against each other).
struct RandomLp {
    n: usize,
    costs: Vec<f64>,
    rows: Vec<(Vec<(usize, f64)>, ComparisonOp, f64)>,
    upper: f64,
}

fn random_feasible_lp(n: usize, seed: u64) -> RandomLp {
    let mut rng = Rng::new(seed);
    // A random point inside the box guarantees feasibility.
    let x0: Vec<i64> = (0..n).map(|_| rng.int(0, 8)).collect();
    let mut rows = vec![];
    for _ in 0..n + 2 {
        let mut terms: Vec<(usize, f64)> = vec![];
        for j in 0..n {
            if rng.int(0, 2) > 0 {
                terms.push((j, rng.int(-4, 6) as f64));
            }
        }
        if terms.is_empty() {
            continue;
        }
        let at_x0: f64 = terms.iter().map(|(j, c)| c * x0[*j] as f64).sum();
        let slack = rng.int(0, 10) as f64;
        // Randomly emit <= or >= rows, both satisfied at x0.
        if rng.int(0, 1) == 0 {
            rows.push((terms, Le, at_x0 + slack));
        } else {
            rows.push((terms, Ge, at_x0 - slack));
        }
    }
    let costs: Vec<f64> = (0..n).map(|_| rng.int(-9, 9) as f64).collect();
    RandomLp {
        n,
        costs,
        rows,
        upper: 20.0, // box keeps everything bounded
    }
}

fn build_random_lp(
    lp: &RandomLp,
    direction: OptimizationDirection,
    cost_scale: f64,
    duplicate_rows: bool,
) -> (Problem, Vec<Variable>) {
    let mut p = Problem::new(direction);
    let vars: Vec<_> = (0..lp.n)
        .map(|j| p.add_var(lp.costs[j] * cost_scale, (0.0, lp.upper)))
        .collect();
    let reps = if duplicate_rows { 2 } else { 1 };
    for _ in 0..reps {
        for (terms, op, rhs) in &lp.rows {
            let t: Vec<_> = terms.iter().map(|&(j, c)| (vars[j], c)).collect();
            p.add_constraint(&t, *op, *rhs);
        }
    }
    (p, vars)
}

fn solve_obj(mut p: Problem, budget: std::time::Duration) -> Result<f64, String> {
    p.set_time_limit(budget);
    let sol = p.solve().map_err(|e| format!("solver error: {}", e))?;
    if sol.status() != microlp::Status::Optimal {
        return Err("hit time limit".into());
    }
    Ok(sol.objective())
}

pub fn register(cases: &mut Vec<Case>) {
    // Certified-unique-optimum family.
    for (i, &(n, seed)) in [
        (3usize, 101u64),
        (3, 102),
        (4, 103),
        (5, 104),
        (6, 105),
        (7, 106),
        (8, 107),
        (8, 108),
        (10, 109),
        (10, 110),
        (12, 111),
        (14, 112),
    ]
    .iter()
    .enumerate()
    {
        let name = format!("lp/certified/n{:02}-s{:02}", n, i);
        cases.push(Case::solve(name, Tier::Quick, 10, move || {
            let inst = certified_instance(n, seed);
            let (bld, vars) = build_certified(&inst);
            let expected_vars: Vec<_> = vars
                .iter()
                .zip(&inst.x_star)
                .map(|(&v, &x)| (v, x as f64))
                .collect();
            let e = Expected::unique(inst.objective as f64, expected_vars);
            Ok((bld.spec, bld.problem, e))
        }));
    }

    // Metamorphic invariants on random feasible LPs.
    for (i, &(n, seed)) in [(4usize, 201u64), (5, 202), (6, 203), (7, 204), (8, 205)]
        .iter()
        .enumerate()
    {
        let name = format!("lp/metamorphic/n{}-s{:02}", n, i);
        cases.push(Case::custom(name, Tier::Quick, 20, move |budget| {
            let lp = random_feasible_lp(n, seed);
            let per = budget / 4;
            let tol = Tol {
                abs: 1e-6,
                rel: 1e-7,
            };

            let base = solve_obj(build_random_lp(&lp, Minimize, 1.0, false).0, per)?;

            // Scaling the objective by 7 must scale the optimum by 7.
            let scaled = solve_obj(build_random_lp(&lp, Minimize, 7.0, false).0, per)?;
            if !tol.matches(scaled, 7.0 * base) {
                return Err(format!(
                    "objective scaling broke: base {}, scaled/7 {}",
                    base,
                    scaled / 7.0
                ));
            }

            // Duplicating every constraint must not change the optimum.
            let dup = solve_obj(build_random_lp(&lp, Minimize, 1.0, true).0, per)?;
            if !tol.matches(dup, base) {
                return Err(format!(
                    "duplicated constraints changed optimum: {} vs {}",
                    dup, base
                ));
            }

            // max(-c) must equal -min(c).
            let neg = {
                let mut lp_neg_costs = lp.costs.clone();
                for c in &mut lp_neg_costs {
                    *c = -*c;
                }
                let lp2 = RandomLp {
                    costs: lp_neg_costs,
                    rows: lp.rows.clone(),
                    n: lp.n,
                    upper: lp.upper,
                };
                solve_obj(build_random_lp(&lp2, Maximize, 1.0, false).0, per)?
            };
            if !tol.matches(neg, -base) {
                return Err(format!(
                    "min/max symmetry broke: min {} vs -max(-c) {}",
                    base, -neg
                ));
            }

            Ok(())
        }));
    }

    // An equality-row variant of the certified family: replace each row that
    // is tight anyway with an explicit equality. Same unique optimum.
    for (i, &(n, seed)) in [(4usize, 301u64), (6, 302), (9, 303)].iter().enumerate() {
        let name = format!("lp/certified-eq/n{}-s{:02}", n, i);
        cases.push(Case::solve(name, Tier::Quick, 10, move || {
            let inst = certified_instance(n, seed);
            let nvars = inst.x_star.len();
            let mut bld = Builder::new(Maximize);
            let vars: Vec<_> = (0..nvars)
                .map(|j| bld.real(inst.c[j] as f64, 0.0, f64::INFINITY))
                .collect();
            for i in 0..nvars {
                let terms: Vec<_> = (0..nvars)
                    .filter(|&j| inst.a[i][j] != 0)
                    .map(|j| (vars[j], inst.a[i][j] as f64))
                    .collect();
                bld.constraint(&terms, Eq, inst.b[i] as f64);
            }
            let expected_vars: Vec<_> = vars
                .iter()
                .zip(&inst.x_star)
                .map(|(&v, &x)| (v, x as f64))
                .collect();
            let e = Expected::unique(inst.objective as f64, expected_vars);
            Ok((bld.spec, bld.problem, e))
        }));
    }
}
