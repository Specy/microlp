//! Seeded random MILP families verified against exact combinatorial oracles.
//! The instance seeds are fixed constants baked into the case names, so runs
//! are fully reproducible regardless of the runner's --seed (which only
//! shuffles case *selection*).

use super::{Case, Tier};
use crate::model::{Builder, Expected};
use crate::oracles::{self, TinyIlp};
use crate::rng::Rng;
use microlp::ComparisonOp;
use microlp::ComparisonOp::{Eq, Ge, Le};
use microlp::OptimizationDirection::{Maximize, Minimize};

pub fn register(cases: &mut Vec<Case>) {
    knapsack01_cases(cases);
    bounded_knapsack_cases(cases);
    subset_sum_cases(cases);
    assignment_cases(cases);
    ilp_box_cases(cases);
    mixed_cases(cases);
}

fn knapsack01_cases(cases: &mut Vec<Case>) {
    // Plain instances of growing size plus tie-heavy ones (equal values force
    // lots of branching ties).
    for (i, &(n, ties, seed)) in [
        (10usize, false, 1u64),
        (12, false, 2),
        (14, false, 3),
        (16, false, 4),
        (16, false, 5),
        (18, false, 6),
        (18, false, 7),
        (20, false, 8),
        (20, false, 9),
        (22, false, 10),
        (22, false, 11),
        (24, false, 12),
        (12, true, 13),
        (16, true, 14),
        (20, true, 15),
        (24, true, 16),
    ]
    .iter()
    .enumerate()
    {
        let name = format!(
            "oracle/knap01{}/n{:02}-s{:02}",
            if ties { "-ties" } else { "" },
            n,
            i
        );
        cases.push(Case::solve(name, Tier::Standard, 30, move || {
            let mut rng = Rng::new(0xA000 + seed);
            let weights: Vec<i64> = (0..n).map(|_| rng.int(1, 30)).collect();
            let values: Vec<i64> = if ties {
                vec![10; n]
            } else {
                (0..n).map(|_| rng.int(1, 40)).collect()
            };
            let total: i64 = weights.iter().sum();
            let capacity = total * rng.int(40, 60) / 100;
            let best = oracles::knapsack01(&values, &weights, capacity);

            let mut b = Builder::new(Maximize);
            let vars: Vec<_> = values.iter().map(|&v| b.binary(v as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            b.constraint(&terms, Le, capacity as f64);
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

fn bounded_knapsack_cases(cases: &mut Vec<Case>) {
    for (i, &(n, seed)) in [
        (6usize, 21u64),
        (7, 22),
        (8, 23),
        (8, 24),
        (9, 25),
        (10, 26),
        (10, 27),
        (12, 28),
    ]
    .iter()
    .enumerate()
    {
        let name = format!("oracle/knap-bounded/n{:02}-s{:02}", n, i);
        cases.push(Case::solve(name, Tier::Standard, 30, move || {
            let mut rng = Rng::new(0xB000 + seed);
            let weights: Vec<i64> = (0..n).map(|_| rng.int(2, 20)).collect();
            let values: Vec<i64> = (0..n).map(|_| rng.int(1, 25)).collect();
            let counts: Vec<i64> = (0..n).map(|_| rng.int(1, 4)).collect();
            let total: i64 = weights.iter().zip(&counts).map(|(w, c)| w * c).sum();
            let capacity = total * rng.int(35, 55) / 100;
            let best = oracles::knapsack_bounded(&values, &weights, &counts, capacity);

            let mut b = Builder::new(Maximize);
            let vars: Vec<_> = (0..n)
                .map(|j| b.integer(values[j] as f64, 0, counts[j] as i32))
                .collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            b.constraint(&terms, Le, capacity as f64);
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

fn subset_sum_cases(cases: &mut Vec<Case>) {
    // Feasible targets: maximize sum(w x) <= t where t is reachable, so the
    // optimum is exactly t.
    for (i, &(n, seed)) in [(14usize, 31u64), (16, 32), (18, 33), (20, 34), (22, 35)]
        .iter()
        .enumerate()
    {
        let name = format!("oracle/subset-sum/n{:02}-s{:02}", n, i);
        cases.push(Case::solve(name, Tier::Standard, 30, move || {
            let mut rng = Rng::new(0xC000 + seed);
            let weights: Vec<i64> = (0..n).map(|_| rng.int(3, 60)).collect();
            // Build a reachable target from a random subset (roughly half);
            // fall back to a single element if the draw came up empty.
            let mut target: i64 = weights.iter().filter(|_| rng.int(0, 1) == 1).sum();
            if target == 0 {
                target = weights[0];
            }
            assert!(oracles::subset_sum(&weights, target));

            let mut b = Builder::new(Maximize);
            let vars: Vec<_> = weights.iter().map(|&w| b.binary(w as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            b.constraint(&terms, Le, target as f64);
            Ok((b.spec, b.problem, Expected::objective(target as f64)))
        }));
    }

    // Structurally unreachable targets: all-even weights, odd equality target.
    for (i, &(n, seed)) in [(10usize, 41u64), (14, 42), (18, 43)].iter().enumerate() {
        let name = format!("oracle/subset-sum-infeasible/n{:02}-s{:02}", n, i);
        cases.push(Case::solve(name, Tier::Standard, 30, move || {
            let mut rng = Rng::new(0xD000 + seed);
            let weights: Vec<i64> = (0..n).map(|_| 2 * rng.int(2, 30)).collect();
            let target = 2 * rng.int(10, 40) + 1; // odd: no even subset can hit it

            let mut b = Builder::new(Maximize);
            let vars: Vec<_> = weights.iter().map(|&w| b.binary(w as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            b.constraint(&terms, Eq, target as f64);
            Ok((b.spec, b.problem, Expected::Infeasible))
        }));
    }
}

fn assignment_cases(cases: &mut Vec<Case>) {
    for (i, &(n, seed)) in [(5usize, 51u64), (5, 52), (6, 53), (6, 54), (7, 55), (7, 56)]
        .iter()
        .enumerate()
    {
        let name = format!("oracle/assignment/n{}-s{:02}", n, i);
        cases.push(Case::solve(name, Tier::Standard, 30, move || {
            let mut rng = Rng::new(0xE000 + seed);
            let costs: Vec<Vec<i64>> = (0..n)
                .map(|_| (0..n).map(|_| rng.int(1, 50)).collect())
                .collect();
            let best = oracles::assignment_min(&costs);

            let mut b = Builder::new(Minimize);
            let mut grid = vec![];
            for i in 0..n {
                let mut row = vec![];
                for j in 0..n {
                    row.push(b.binary(costs[i][j] as f64));
                }
                grid.push(row);
            }
            for i in 0..n {
                let terms: Vec<_> = (0..n).map(|j| (grid[i][j], 1.0)).collect();
                b.constraint(&terms, Eq, 1.0);
            }
            for j in 0..n {
                let terms: Vec<_> = (0..n).map(|i| (grid[i][j], 1.0)).collect();
                b.constraint(&terms, Eq, 1.0);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

fn ilp_box_cases(cases: &mut Vec<Case>) {
    // Small general ILPs over a box, solved exactly by enumeration. This is
    // the widest net for branch & bound bugs: mixed constraint senses,
    // equalities, negative coefficients and bounds, maximize and minimize.
    // Whatever the oracle says (optimum or infeasible) is what microlp must
    // reproduce.
    for i in 0..20u64 {
        let name = format!("oracle/ilp-box/s{:02}", i);
        cases.push(Case::solve(name, Tier::Standard, 60, move || {
            let mut rng = Rng::new(0xF000 + i);
            let n = rng.usize(4, 7);
            let bounds: Vec<(i64, i64)> = (0..n)
                .map(|_| {
                    let lo = rng.int(-3, 2);
                    let hi = lo + rng.int(1, 4);
                    (lo, hi)
                })
                .collect();
            let n_cons = rng.usize(3, 5);
            let mut constraints = vec![];
            for _ in 0..n_cons {
                let coeffs: Vec<i64> = (0..n)
                    .map(|_| {
                        if rng.int(0, 3) == 0 {
                            0
                        } else {
                            rng.int(-6, 6)
                        }
                    })
                    .collect();
                if coeffs.iter().all(|&c| c == 0) {
                    continue;
                }
                // Anchor the rhs near a random box point so instances are not
                // trivially infeasible; equalities stay rare but present.
                let point: Vec<i64> = bounds
                    .iter()
                    .map(|&(lo, hi)| lo + (rng.int(0, (hi - lo).max(0))))
                    .collect();
                let at_point: i64 = coeffs.iter().zip(&point).map(|(c, x)| c * x).sum();
                let (op, rhs) = match rng.int(0, 5) {
                    0 => (0i8, at_point),                   // equality through a box point
                    1 | 2 => (1, at_point - rng.int(0, 6)), // >=
                    _ => (-1, at_point + rng.int(0, 6)),    // <=
                };
                constraints.push((coeffs, op, rhs));
            }
            let objective: Vec<i64> = (0..n).map(|_| rng.int(-9, 9)).collect();
            let maximize = rng.int(0, 1) == 1;

            let ilp = TinyIlp {
                bounds: bounds.clone(),
                objective: objective.clone(),
                constraints: constraints.clone(),
                maximize,
            };
            let oracle = ilp.brute_force();

            let dir = if maximize { Maximize } else { Minimize };
            let mut b = Builder::new(dir);
            let vars: Vec<_> = (0..n)
                .map(|j| b.integer(objective[j] as f64, bounds[j].0 as i32, bounds[j].1 as i32))
                .collect();
            for (coeffs, op, rhs) in &constraints {
                let terms: Vec<_> = coeffs
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c != 0)
                    .map(|(j, &c)| (vars[j], c as f64))
                    .collect();
                let cmp = match op {
                    -1 => ComparisonOp::Le,
                    0 => ComparisonOp::Eq,
                    _ => ComparisonOp::Ge,
                };
                b.constraint(&terms, cmp, *rhs as f64);
            }
            let expected = match oracle {
                Some(best) => Expected::objective(best as f64),
                None => Expected::Infeasible,
            };
            Ok((b.spec, b.problem, expected))
        }));
    }
}

fn mixed_cases(cases: &mut Vec<Case>) {
    // Integer core (enumerable) plus continuous variables whose optimal value
    // is a known function of the integers: y_k >= max(0, a_k'x + b_k) with
    // positive cost, so at the optimum y_k = max(0, a_k'x + b_k) exactly.
    // The oracle enumerates the integer box and minimizes the full objective.
    for i in 0..8u64 {
        let name = format!("oracle/mixed/s{:02}", i);
        cases.push(Case::solve(name, Tier::Standard, 60, move || {
            let mut rng = Rng::new(0x1A000 + i);
            let n = rng.usize(3, 5); // integers
            let m = rng.usize(1, 3); // continuous
            let bounds: Vec<(i64, i64)> = (0..n)
                .map(|_| {
                    let lo = rng.int(-2, 1);
                    let hi = lo + rng.int(2, 4);
                    (lo, hi)
                })
                .collect();
            let int_cost: Vec<i64> = (0..n).map(|_| rng.int(-5, 5)).collect();
            let links: Vec<(Vec<i64>, i64, i64)> = (0..m)
                .map(|_| {
                    let a: Vec<i64> = (0..n).map(|_| rng.int(-3, 3)).collect();
                    let b = rng.int(-4, 4);
                    let cost = rng.int(1, 4); // strictly positive
                    (a, b, cost)
                })
                .collect();

            // Oracle over the integer box.
            let mut best: Option<i64> = None;
            let mut point: Vec<i64> = bounds.iter().map(|b| b.0).collect();
            loop {
                let mut obj: i64 = int_cost.iter().zip(&point).map(|(c, x)| c * x).sum();
                for (a, b0, cost) in &links {
                    let lin: i64 = a.iter().zip(&point).map(|(c, x)| c * x).sum::<i64>() + b0;
                    obj += cost * lin.max(0);
                }
                best = Some(best.map_or(obj, |b: i64| b.min(obj)));
                let mut k = 0;
                loop {
                    if k == n {
                        break;
                    }
                    if point[k] < bounds[k].1 {
                        point[k] += 1;
                        break;
                    }
                    point[k] = bounds[k].0;
                    k += 1;
                }
                if k == n {
                    break;
                }
            }
            let best = best.unwrap();

            let mut b = Builder::new(Minimize);
            let ints: Vec<_> = (0..n)
                .map(|j| b.integer(int_cost[j] as f64, bounds[j].0 as i32, bounds[j].1 as i32))
                .collect();
            for (a, b0, cost) in &links {
                let y = b.real(*cost as f64, 0.0, f64::INFINITY);
                // y >= a'x + b0  <=>  y - a'x >= b0
                let mut terms = vec![(y, 1.0)];
                for (j, &coef) in a.iter().enumerate() {
                    if coef != 0 {
                        terms.push((ints[j], -(coef as f64)));
                    }
                }
                b.constraint(&terms, Ge, *b0 as f64);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}
