//! Hand-constructed MILP cases: rounding-direction traps, negative integer
//! ranges, diophantine feasibility, big-M implications, boolean gates and
//! small verifiable puzzles.

use super::{Case, Tier};
use crate::model::{Builder, Expected, Tol};
use crate::oracles;
use microlp::ComparisonOp::{Eq, Ge, Le};
use microlp::OptimizationDirection::{Maximize, Minimize};

const INF: f64 = f64::INFINITY;

fn case(cases: &mut Vec<Case>, name: &str, build: impl Fn() -> (Builder, Expected) + 'static) {
    case_tier(cases, name, Tier::Quick, 15, build);
}

fn case_tier(
    cases: &mut Vec<Case>,
    name: &str,
    tier: Tier,
    budget: u64,
    build: impl Fn() -> (Builder, Expected) + 'static,
) {
    cases.push(Case::solve(name, tier, budget, move || {
        let (b, expected) = build();
        Ok((b.spec, b.problem, expected))
    }));
}

pub fn register(cases: &mut Vec<Case>) {
    case(cases, "milp/readme-example", || {
        // The README example: max x + 2y, y integer in [0, 3]; optimum (1, 3).
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, INF);
        let y = b.integer(2.0, 0, 3);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 4.0);
        b.constraint(&[(x, 2.0), (y, 1.0)], Ge, 2.0);
        (b, Expected::unique(7.0, vec![(x, 1.0), (y, 3.0)]))
    });

    case(cases, "milp/floor-trap-max", || {
        // LP relaxation gives 4.5; the integer answer must round DOWN.
        let mut b = Builder::new(Maximize);
        let x = b.integer(1.0, 0, 10);
        b.constraint(&[(x, 2.0)], Le, 9.0);
        (b, Expected::unique(4.0, vec![(x, 4.0)]))
    });

    case(cases, "milp/ceil-trap-min", || {
        // LP relaxation gives 7/3; the integer answer must round UP.
        let mut b = Builder::new(Minimize);
        let x = b.integer(1.0, 0, 10);
        b.constraint(&[(x, 3.0)], Ge, 7.0);
        (b, Expected::unique(3.0, vec![(x, 3.0)]))
    });

    case(cases, "milp/negative-floor-trap", || {
        // max x with 3x <= -7: relaxation -7/3, integer answer -3 (not -2).
        let mut b = Builder::new(Maximize);
        let x = b.integer(1.0, -10, 10);
        b.constraint(&[(x, 3.0)], Le, -7.0);
        (b, Expected::unique(-3.0, vec![(x, -3.0)]))
    });

    case(cases, "milp/negative-range-min", || {
        // min x, x integer in [-5, 5], 10x >= -45: answer -4 (not -5, not -4.5).
        let mut b = Builder::new(Minimize);
        let x = b.integer(1.0, -5, 5);
        b.constraint(&[(x, 10.0)], Ge, -45.0);
        (b, Expected::unique(-4.0, vec![(x, -4.0)]))
    });

    case(cases, "milp/parity-infeasible", || {
        // 2x + 4y = 7 has no integer solutions (left side is even).
        let mut b = Builder::new(Minimize);
        let x = b.integer(1.0, 0, 10);
        let y = b.integer(1.0, 0, 10);
        b.constraint(&[(x, 2.0), (y, 4.0)], Eq, 7.0);
        (b, Expected::Infeasible)
    });

    case(cases, "milp/interval-infeasible", || {
        // 1.2 <= x <= 1.8 contains no integer.
        let mut b = Builder::new(Minimize);
        let x = b.integer(1.0, 0, 10);
        b.constraint(&[(x, 10.0)], Ge, 12.0);
        b.constraint(&[(x, 10.0)], Le, 18.0);
        (b, Expected::Infeasible)
    });

    case(cases, "milp/diophantine-min", || {
        // 3x + 5y = 1 with x, y in [-10, 10]: solutions are (2,-1), (-3,2),
        // (7,-4), (-8,5); min x+y is -3 at (-8, 5).
        let mut b = Builder::new(Minimize);
        let x = b.integer(1.0, -10, 10);
        let y = b.integer(1.0, -10, 10);
        b.constraint(&[(x, 3.0), (y, 5.0)], Eq, 1.0);
        (b, Expected::unique(-3.0, vec![(x, -8.0), (y, 5.0)]))
    });

    case(cases, "milp/int-eq-forcing", || {
        // 7x - 3y = 1, x,y in [0,20]: (1,2), (4,9), (7,16); min x+y = 3.
        let mut b = Builder::new(Minimize);
        let x = b.integer(1.0, 0, 20);
        let y = b.integer(1.0, 0, 20);
        b.constraint(&[(x, 7.0), (y, -3.0)], Eq, 1.0);
        (b, Expected::unique(3.0, vec![(x, 1.0), (y, 2.0)]))
    });

    case(cases, "milp/bigm-implication", || {
        // x >= 7 forces b = 1 through x <= 5 + M*b; min x + 100b = 107.
        let m = 1e6;
        let mut b = Builder::new(Minimize);
        let x = b.real(1.0, 0.0, INF);
        let ind = b.binary(100.0);
        b.constraint(&[(x, 1.0)], Ge, 7.0);
        b.constraint(&[(x, 1.0), (ind, -m)], Le, 5.0);
        (b, Expected::unique(107.0, vec![(x, 7.0), (ind, 1.0)]))
    });

    case(cases, "milp/knapsack-tight-gap", || {
        // Three items of weight 3 into capacity 8: LP says 5.33, truth is 4.
        let mut b = Builder::new(Maximize);
        let items: Vec<_> = (0..3).map(|_| b.binary(2.0)).collect();
        let terms: Vec<_> = items.iter().map(|&v| (v, 3.0)).collect();
        b.constraint(&terms, Le, 8.0);
        (b, Expected::objective(4.0))
    });

    case(cases, "milp/knapsack-scaled", || {
        // Same structure with 1e5-scale data: tests numeric robustness of B&B.
        let mut b = Builder::new(Maximize);
        let items: Vec<_> = (0..3).map(|_| b.binary(2e5)).collect();
        let terms: Vec<_> = items.iter().map(|&v| (v, 3e3)).collect();
        b.constraint(&terms, Le, 8e3);
        (
            b,
            Expected::objective_tol(
                4e5,
                Tol {
                    abs: 1e-3,
                    rel: 1e-9,
                },
            ),
        )
    });

    case(cases, "milp/all-fixed-ints", || {
        // Every integer fixed by its bounds; only consistency remains.
        let mut b = Builder::new(Maximize);
        let x = b.integer(3.0, 2, 2);
        let y = b.integer(2.0, -1, -1);
        let z = b.integer(1.0, 5, 5);
        b.constraint(&[(x, 1.0), (y, 1.0), (z, 1.0)], Eq, 6.0);
        (
            b,
            Expected::unique(9.0, vec![(x, 2.0), (y, -1.0), (z, 5.0)]),
        )
    });

    case(cases, "milp/fixed-infeasible", || {
        let mut b = Builder::new(Minimize);
        let x = b.integer(1.0, 2, 2);
        let y = b.integer(1.0, 3, 3);
        b.constraint(&[(x, 1.0), (y, 1.0)], Eq, 6.0);
        (b, Expected::Infeasible)
    });

    case(cases, "milp/unbounded-mixed", || {
        // The continuous part is unbounded even though integers are boxed.
        let mut b = Builder::new(Maximize);
        let x = b.real(1.0, 0.0, INF);
        let y = b.integer(1.0, 0, 5);
        b.constraint(&[(y, 1.0)], Le, 5.0);
        let _ = x;
        (b, Expected::Unbounded)
    });

    case(cases, "milp/fractional-cut-needed", || {
        // max y with 2y - 2x <= 1, 2x <= 5: relaxation (2.5, 3), truth 2.
        let mut b = Builder::new(Maximize);
        let x = b.integer(0.0, 0, 10);
        let y = b.integer(1.0, 0, 10);
        b.constraint(&[(y, 2.0), (x, -2.0)], Le, 1.0);
        b.constraint(&[(x, 2.0)], Le, 5.0);
        (b, Expected::objective(2.0))
    });

    case(cases, "milp/alternate-optima", || {
        let mut b = Builder::new(Maximize);
        let x = b.binary(1.0);
        let y = b.binary(1.0);
        b.constraint(&[(x, 1.0), (y, 1.0)], Le, 1.0);
        (b, Expected::objective(1.0))
    });

    case(cases, "milp/huge-int-bound", || {
        // Integer variable with the full i32 range; relaxation is already
        // integral at the bound.
        let mut b = Builder::new(Maximize);
        let _x = b.integer(1.0, 0, i32::MAX);
        // Default tolerance is relative and would allow +-2147 here; a wrong
        // answer would be off by >= 1, so use a strict absolute tolerance.
        (
            b,
            Expected::objective_tol(i32::MAX as f64, Tol { abs: 0.5, rel: 0.0 }),
        )
    });

    case(cases, "milp/mixed-textbook", || {
        // Ported from the crate's solve_milp test: optimum (2, 6.5, 1), 405.
        let mut b = Builder::new(Maximize);
        let x = b.real(50.0, 2.0, INF);
        let y = b.real(40.0, 0.0, 7.0);
        let z = b.integer(45.0, 0, i32::MAX);
        b.constraint(&[(x, 3.0), (y, 2.0), (z, 1.0)], Le, 20.0);
        b.constraint(&[(x, 2.0), (y, 1.0), (z, 3.0)], Le, 15.0);
        (
            b,
            Expected::unique(405.0, vec![(x, 2.0), (y, 6.5), (z, 1.0)]),
        )
    });

    case(cases, "milp/production-planning", || {
        // Ported from the crate's solve_production_planning test: 3440.
        let periods = 4;
        let prod_costs = [10.0, 12.0, 11.0, 14.0];
        let holding_costs = [2.0, 2.0, 2.0, 2.0];
        let setup_costs = [100.0, 100.0, 100.0, 100.0];
        let demand = [50.0, 70.0, 90.0, 60.0];
        let capacity = 120.0;

        let mut b = Builder::new(Minimize);
        let production: Vec<_> = (0..periods)
            .map(|i| b.real(prod_costs[i], 0.0, capacity))
            .collect();
        let inventory: Vec<_> = (0..periods)
            .map(|i| b.real(holding_costs[i], 0.0, INF))
            .collect();
        let setup: Vec<_> = (0..periods).map(|i| b.binary(setup_costs[i])).collect();
        let mut prev_inventory = b.real(0.0, 0.0, 0.0);
        for i in 0..periods {
            b.constraint(
                &[
                    (prev_inventory, 1.0),
                    (production[i], 1.0),
                    (inventory[i], -1.0),
                ],
                Eq,
                demand[i],
            );
            b.constraint(&[(production[i], 1.0), (setup[i], -capacity)], Le, 0.0);
            prev_inventory = inventory[i];
        }
        (b, Expected::objective(3440.0))
    });

    // A mid-search time limit must return a CLEAN status — never the old
    // is_primal_feasible panic: Optimal if 30 ms happened to suffice, Feasible
    // with a valid incumbent, or Interrupted with none — and resume(None) must
    // then finish at the exact DP optimum. Knapsack sized so 30 ms rarely
    // finishes it. Promoted to the standard tier now that the panic is fixed,
    // so the default (CI) run guards the fix.
    cases.push(Case::custom(
        "milp/time-limit-interrupt",
        Tier::Standard,
        30,
        move |_budget| {
            let mut rng = crate::rng::Rng::new(0xDEAD);
            let n = 40;
            // Preserve the original draw order (value then weight per item) so
            // the instance is byte-identical to the pre-flip case.
            let mut values = vec![];
            let mut weights = vec![];
            for _ in 0..n {
                values.push(rng.int(1, 1000));
                weights.push(rng.int(1, 1000));
            }
            let cap = weights.iter().sum::<i64>() as f64 * 0.5;
            let best = oracles::knapsack01(&values, &weights, cap.floor() as i64) as f64;

            let mut p = microlp::Problem::new(Maximize);
            let vars: Vec<_> = values.iter().map(|&v| p.add_binary_var(v as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            p.add_constraint(&terms, Le, cap);
            p.set_time_limit(std::time::Duration::from_millis(30));

            let sol = p
                .solve()
                .map_err(|e| format!("interrupted solve errored: {}", e))?;
            match sol.status() {
                microlp::Status::Optimal => {
                    if (sol.objective() - best).abs() > 1e-6 {
                        return Err(format!(
                            "finished within 30 ms but objective {} != DP optimum {}",
                            sol.objective(),
                            best
                        ));
                    }
                }
                microlp::Status::Feasible => {
                    // The incumbent must be a feasible 0/1 knapsack point and no
                    // better than the true optimum.
                    let mut weight = 0.0;
                    for (i, &v) in vars.iter().enumerate() {
                        let x = sol.var_value_raw(v);
                        if (x - x.round()).abs() > 1e-6 {
                            return Err(format!("incumbent x{} = {} not integral", i, x));
                        }
                        weight += x.round() * weights[i] as f64;
                    }
                    if weight > cap + 1e-6 {
                        return Err(format!(
                            "incumbent weight {} exceeds capacity {}",
                            weight, cap
                        ));
                    }
                    if sol.objective() > best + 1e-6 {
                        return Err(format!(
                            "incumbent objective {} exceeds DP optimum {}",
                            sol.objective(),
                            best
                        ));
                    }
                }
                microlp::Status::Interrupted => {
                    // No incumbent yet; value accessors would panic. Resume below.
                }
            }
            // Resuming to completion must reach the proven optimum.
            let sol = sol.resume(None).map_err(|e| format!("resume: {}", e))?;
            if sol.status() != microlp::Status::Optimal {
                return Err("resume(None) did not reach Optimal".into());
            }
            if (sol.objective() - best).abs() > 1e-6 {
                return Err(format!(
                    "resumed objective {} != DP optimum {}",
                    sol.objective(),
                    best
                ));
            }
            Ok(())
        },
    ));

    // Boolean gates: z is forced to the gate value by the standard exact
    // linearizations; check all four input combinations in both directions.
    for (gate, idx) in [("and", 0usize), ("or", 1), ("xor", 2)] {
        let name = format!("milp/gate-{}", gate);
        cases.push(Case::custom(name, Tier::Quick, 15, move |budget| {
            for a in 0..=1i32 {
                for bit in 0..=1i32 {
                    let want = match idx {
                        0 => a & bit,
                        1 => a | bit,
                        _ => a ^ bit,
                    } as f64;
                    // Both directions: the constraints must pin z either way.
                    for maximize in [false, true] {
                        let dir = if maximize { Maximize } else { Minimize };
                        let mut p = microlp::Problem::new(dir);
                        let x = p.add_binary_var(0.0);
                        let y = p.add_binary_var(0.0);
                        let z = p.add_binary_var(1.0);
                        p.add_constraint(&[(x, 1.0)], Eq, a as f64);
                        p.add_constraint(&[(y, 1.0)], Eq, bit as f64);
                        match idx {
                            0 => {
                                // AND: z <= x, z <= y, z >= x + y - 1
                                p.add_constraint(&[(z, 1.0), (x, -1.0)], Le, 0.0);
                                p.add_constraint(&[(z, 1.0), (y, -1.0)], Le, 0.0);
                                p.add_constraint(&[(z, 1.0), (x, -1.0), (y, -1.0)], Ge, -1.0);
                            }
                            1 => {
                                // OR: z >= x, z >= y, z <= x + y
                                p.add_constraint(&[(z, 1.0), (x, -1.0)], Ge, 0.0);
                                p.add_constraint(&[(z, 1.0), (y, -1.0)], Ge, 0.0);
                                p.add_constraint(&[(z, 1.0), (x, -1.0), (y, -1.0)], Le, 0.0);
                            }
                            _ => {
                                // XOR: z <= x+y, z >= x-y, z >= y-x, z <= 2-x-y
                                p.add_constraint(&[(z, 1.0), (x, -1.0), (y, -1.0)], Le, 0.0);
                                p.add_constraint(&[(z, 1.0), (x, -1.0), (y, 1.0)], Ge, 0.0);
                                p.add_constraint(&[(z, 1.0), (x, 1.0), (y, -1.0)], Ge, 0.0);
                                p.add_constraint(&[(z, 1.0), (x, 1.0), (y, 1.0)], Le, 2.0);
                            }
                        }
                        p.set_time_limit(budget / 8);
                        let sol = p
                            .solve()
                            .map_err(|e| format!("{}({},{}): {}", gate, a, bit, e))?;
                        if sol.status() != microlp::Status::Optimal {
                            return Err("hit time limit".into());
                        }
                        let got = sol.objective();
                        if (got - want).abs() > 1e-6 {
                            return Err(format!(
                                "{}({},{}) {}: expected z = {}, got {}",
                                gate,
                                a,
                                bit,
                                if maximize { "max" } else { "min" },
                                want,
                                got
                            ));
                        }
                    }
                }
            }
            Ok(())
        }));
    }

    // Coin change vs DP oracle.
    for (denoms, target) in [
        (vec![1i64, 5, 10, 25], 30i64),
        (vec![1, 5, 10, 25], 68),
        (vec![1, 5, 10, 25], 99),
        (vec![1, 3, 4], 6), // greedy trap: 3+3, not 4+1+1
    ] {
        let expected = oracles::coin_change_min(&denoms, target)
            .expect("coin change instance must be solvable");
        let name = format!(
            "milp/coin-change-{}-{}",
            denoms
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("_"),
            target
        );
        cases.push(Case::solve(name, Tier::Quick, 15, move || {
            let mut b = Builder::new(Minimize);
            let counts: Vec<_> = denoms
                .iter()
                .map(|_| b.integer(1.0, 0, target as i32))
                .collect();
            let terms: Vec<_> = counts
                .iter()
                .zip(&denoms)
                .map(|(&v, &d)| (v, d as f64))
                .collect();
            b.constraint(&terms, Eq, target as f64);
            Ok((b.spec, b.problem, Expected::objective(expected as f64)))
        }));
    }

    // 3x3 magic square with digits 1..9 via assignment binaries. Every row,
    // column and diagonal sums to 15 and the center is forced to 5: minimizing
    // and maximizing the center must both give exactly 5.
    for maximize in [false, true] {
        let name = format!(
            "milp/magic-square-center-{}",
            if maximize { "max" } else { "min" }
        );
        case_tier(cases, &name, Tier::Standard, 60, move || {
            let dir = if maximize { Maximize } else { Minimize };
            let mut bld = Builder::new(dir);
            // b[cell][digit] = 1 iff cell holds digit+1; center cell is 4.
            let mut grid = vec![];
            for cell in 0..9 {
                let mut row = vec![];
                for digit in 0..9 {
                    let obj = if cell == 4 { (digit + 1) as f64 } else { 0.0 };
                    row.push(bld.binary(obj));
                }
                grid.push(row);
            }
            for cell in 0..9 {
                let terms: Vec<_> = (0..9).map(|d| (grid[cell][d], 1.0)).collect();
                bld.constraint(&terms, Eq, 1.0);
            }
            for digit in 0..9 {
                let terms: Vec<_> = (0..9).map(|c| (grid[c][digit], 1.0)).collect();
                bld.constraint(&terms, Eq, 1.0);
            }
            let lines: [[usize; 3]; 8] = [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [0, 3, 6],
                [1, 4, 7],
                [2, 5, 8],
                [0, 4, 8],
                [2, 4, 6],
            ];
            for line in lines {
                let mut terms = vec![];
                for &cell in &line {
                    for digit in 0..9 {
                        terms.push((grid[cell][digit], (digit + 1) as f64));
                    }
                }
                bld.constraint(&terms, Eq, 15.0);
            }
            (bld, Expected::objective(5.0))
        });
    }

    // n-queens: the maximum number of non-attacking queens on an n x n board
    // is exactly n.
    for n in [5usize, 6] {
        let name = format!("milp/queens-{}", n);
        case_tier(cases, &name, Tier::Standard, 60, move || {
            let mut bld = Builder::new(Maximize);
            let mut board = vec![vec![]; n];
            for row in board.iter_mut() {
                for _ in 0..n {
                    row.push(bld.binary(1.0));
                }
            }
            for r in 0..n {
                let terms: Vec<_> = (0..n).map(|c| (board[r][c], 1.0)).collect();
                bld.constraint(&terms, Le, 1.0);
            }
            for c in 0..n {
                let terms: Vec<_> = (0..n).map(|r| (board[r][c], 1.0)).collect();
                bld.constraint(&terms, Le, 1.0);
            }
            // Diagonals in both directions.
            for s in 0..(2 * n - 1) {
                let diag1: Vec<_> = (0..n)
                    .filter_map(|r| {
                        let c = s as i64 - r as i64;
                        (c >= 0 && (c as usize) < n).then(|| (board[r][c as usize], 1.0))
                    })
                    .collect();
                if diag1.len() > 1 {
                    bld.constraint(&diag1, Le, 1.0);
                }
                let diag2: Vec<_> = (0..n)
                    .filter_map(|r| {
                        let c = r as i64 + s as i64 - (n as i64 - 1);
                        (c >= 0 && (c as usize) < n).then(|| (board[r][c as usize], 1.0))
                    })
                    .collect();
                if diag2.len() > 1 {
                    bld.constraint(&diag2, Le, 1.0);
                }
            }
            (bld, Expected::objective(n as f64))
        });
    }
}
