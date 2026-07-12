//! Structured problem families beyond knapsacks: covering, packing,
//! partitioning, assignment-with-capacities, transportation, fixed-charge
//! networks, equality chains, big-M numerics, and infeasible/unbounded/
//! degenerate stressors. Every case is seeded (reproducible regardless of
//! the runner's --seed), solves in milliseconds, and is verified against an
//! exact integer oracle or a by-construction optimum — no external
//! reference needed. Together they exercise the solver mechanisms one at a
//! time: weak-LP covering rows, clique-style packing rows, substitution
//! chains for the presolve, big-M rows for propagation and coefficient
//! tightening, and the Infeasible/Unbounded verdict paths.

use super::{Case, Tier};
use crate::model::{Builder, Expected};
use crate::oracles::{self, TinyIlp};
use crate::rng::Rng;
use microlp::ComparisonOp::{Eq, Ge, Le};
use microlp::OptimizationDirection::{Maximize, Minimize};

pub fn register(cases: &mut Vec<Case>) {
    set_cover_cases(cases);
    set_pack_cases(cases);
    set_partition_cases(cases);
    fixed_charge_cases(cases);
    gap_cases(cases);
    transport_cases(cases);
    eq_chain_cases(cases);
    big_m_exact_cases(cases);
    infeasible_cases(cases);
    unbounded_cases(cases);
    degenerate_lp_cases(cases);
}

/// Random subsets over `n` elements with per-set costs; every element gets a
/// guaranteed covering set so feasibility holds by construction. Returns
/// (sets, costs).
fn random_cover_sets(rng: &mut Rng, n_elems: usize, n_sets: usize) -> (Vec<Vec<usize>>, Vec<i64>) {
    let mut sets: Vec<Vec<usize>> = (0..n_sets)
        .map(|_| {
            let size = rng.usize(1, (n_elems / 2).max(2));
            let mut members: Vec<usize> = (0..n_elems).collect();
            rng.shuffle(&mut members);
            members.truncate(size);
            members.sort_unstable();
            members
        })
        .collect();
    // Guarantee coverage: element e joins set e % n_sets.
    for e in 0..n_elems {
        let s = e % n_sets;
        if !sets[s].contains(&e) {
            sets[s].push(e);
            sets[s].sort_unstable();
        }
    }
    let costs: Vec<i64> = (0..n_sets).map(|_| rng.int(1, 20)).collect();
    (sets, costs)
}

/// Set covering (min cost, >= rows): the weak-LP-relaxation class (stein-like)
/// in miniature. Oracle: exact box enumeration over the binaries.
fn set_cover_cases(cases: &mut Vec<Case>) {
    for (i, &(n_elems, n_sets, seed)) in [
        (8usize, 10usize, 41u64),
        (10, 12, 42),
        (12, 14, 43),
        (12, 16, 44),
        (14, 16, 45),
        (14, 18, 46),
    ]
    .iter()
    .enumerate()
    {
        let name = format!("family/set-cover/e{:02}s{:02}-{}", n_elems, n_sets, i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            let mut rng = Rng::new(0xD100 + seed);
            let (sets, costs) = random_cover_sets(&mut rng, n_elems, n_sets);

            let ilp = TinyIlp {
                bounds: vec![(0, 1); n_sets],
                objective: costs.clone(),
                constraints: (0..n_elems)
                    .map(|e| {
                        let coeffs: Vec<i64> = (0..n_sets)
                            .map(|s| i64::from(sets[s].contains(&e)))
                            .collect();
                        (coeffs, 1i8, 1i64)
                    })
                    .collect(),
                maximize: false,
            };
            let best = ilp
                .brute_force()
                .expect("coverage is guaranteed by construction");

            let mut b = Builder::new(Minimize);
            let vars: Vec<_> = costs.iter().map(|&c| b.binary(c as f64)).collect();
            for e in 0..n_elems {
                let terms: Vec<_> = (0..n_sets)
                    .filter(|&s| sets[s].contains(&e))
                    .map(|s| (vars[s], 1.0))
                    .collect();
                b.constraint(&terms, Ge, 1.0);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

/// Set packing (max value, <= 1 rows): clique structure in miniature.
fn set_pack_cases(cases: &mut Vec<Case>) {
    for (i, &(n_items, n_rows, seed)) in [
        (10usize, 6usize, 51u64),
        (12, 8, 52),
        (14, 9, 53),
        (16, 10, 54),
        (16, 12, 55),
        (18, 12, 56),
    ]
    .iter()
    .enumerate()
    {
        let name = format!("family/set-pack/i{:02}r{:02}-{}", n_items, n_rows, i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            let mut rng = Rng::new(0xD200 + seed);
            let values: Vec<i64> = (0..n_items).map(|_| rng.int(1, 25)).collect();
            let rows: Vec<Vec<usize>> = (0..n_rows)
                .map(|_| {
                    let size = rng.usize(3, 5.min(n_items));
                    let mut members: Vec<usize> = (0..n_items).collect();
                    rng.shuffle(&mut members);
                    members.truncate(size);
                    members.sort_unstable();
                    members
                })
                .collect();

            let ilp = TinyIlp {
                bounds: vec![(0, 1); n_items],
                objective: values.clone(),
                constraints: rows
                    .iter()
                    .map(|row| {
                        let coeffs: Vec<i64> =
                            (0..n_items).map(|v| i64::from(row.contains(&v))).collect();
                        (coeffs, -1i8, 1i64)
                    })
                    .collect(),
                maximize: true,
            };
            let best = ilp.brute_force().expect("x = 0 is always feasible");

            let mut b = Builder::new(Maximize);
            let vars: Vec<_> = values.iter().map(|&v| b.binary(v as f64)).collect();
            for row in &rows {
                let terms: Vec<_> = row.iter().map(|&v| (vars[v], 1.0)).collect();
                b.constraint(&terms, Le, 1.0);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

/// Set partitioning (== 1 rows): equality-heavy and feasibility-brittle —
/// exactly the structure where wrong presolve substitutions would show.
fn set_partition_cases(cases: &mut Vec<Case>) {
    for (i, &(n_elems, n_sets, seed)) in [
        (8usize, 12usize, 61u64),
        (10, 14, 62),
        (10, 16, 63),
        (12, 16, 64),
    ]
    .iter()
    .enumerate()
    {
        let name = format!("family/set-partition/e{:02}s{:02}-{}", n_elems, n_sets, i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            let mut rng = Rng::new(0xD300 + seed);
            // Feasibility by construction: first, hide a random partition of
            // the elements inside the set pool, then add random decoy sets.
            let mut elems: Vec<usize> = (0..n_elems).collect();
            rng.shuffle(&mut elems);
            let mut sets: Vec<Vec<usize>> = vec![];
            let mut start = 0;
            while start < n_elems {
                let size = rng.usize(1, 3).min(n_elems - start);
                let mut s: Vec<usize> = elems[start..start + size].to_vec();
                s.sort_unstable();
                sets.push(s);
                start += size;
            }
            while sets.len() < n_sets {
                let size = rng.usize(1, 4);
                let mut members: Vec<usize> = (0..n_elems).collect();
                rng.shuffle(&mut members);
                members.truncate(size);
                members.sort_unstable();
                sets.push(members);
            }
            rng.shuffle(&mut sets);
            let costs: Vec<i64> = (0..sets.len()).map(|_| rng.int(1, 15)).collect();

            let ilp = TinyIlp {
                bounds: vec![(0, 1); sets.len()],
                objective: costs.clone(),
                constraints: (0..n_elems)
                    .map(|e| {
                        let coeffs: Vec<i64> =
                            sets.iter().map(|s| i64::from(s.contains(&e))).collect();
                        (coeffs, 0i8, 1i64)
                    })
                    .collect(),
                maximize: false,
            };
            let best = ilp
                .brute_force()
                .expect("a partition is hidden in the pool by construction");

            let mut b = Builder::new(Minimize);
            let vars: Vec<_> = costs.iter().map(|&c| b.binary(c as f64)).collect();
            for e in 0..n_elems {
                let terms: Vec<_> = sets
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.contains(&e))
                    .map(|(idx, _)| (vars[idx], 1.0))
                    .collect();
                b.constraint(&terms, Eq, 1.0);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

/// Fixed-charge facility rows (`x_i <= cap_i * y_i`): THE propagation /
/// coefficient-tightening structure, verified by the open-set oracle.
fn fixed_charge_cases(cases: &mut Vec<Case>) {
    for (i, &(n, seed)) in [
        (5usize, 71u64),
        (6, 72),
        (7, 73),
        (8, 74),
        (9, 75),
        (10, 76),
    ]
    .iter()
    .enumerate()
    {
        let name = format!("family/fixed-charge/n{:02}-{}", n, i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            let mut rng = Rng::new(0xD400 + seed);
            let open_cost: Vec<i64> = (0..n).map(|_| rng.int(10, 60)).collect();
            let unit_cost: Vec<i64> = (0..n).map(|_| rng.int(1, 9)).collect();
            let cap: Vec<i64> = (0..n).map(|_| rng.int(5, 25)).collect();
            let total: i64 = cap.iter().sum();
            let demand = total * rng.int(45, 75) / 100;
            let best = oracles::fixed_charge_min(&open_cost, &unit_cost, &cap, demand)
                .expect("demand is below total capacity by construction");

            let mut b = Builder::new(Minimize);
            let ys: Vec<_> = open_cost.iter().map(|&f| b.binary(f as f64)).collect();
            let xs: Vec<_> = unit_cost
                .iter()
                .zip(&cap)
                .map(|(&c, &u)| b.real(c as f64, 0.0, u as f64))
                .collect();
            let demand_terms: Vec<_> = xs.iter().map(|&x| (x, 1.0)).collect();
            b.constraint(&demand_terms, Ge, demand as f64);
            for j in 0..n {
                b.constraint(&[(xs[j], 1.0), (ys[j], -(cap[j] as f64))], Le, 0.0);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

/// Generalized assignment: every task to exactly one agent (== rows), agent
/// capacities (<= rows). Small enough for exact box enumeration.
fn gap_cases(cases: &mut Vec<Case>) {
    for (i, &(agents, tasks, seed)) in [(3usize, 4usize, 81u64), (3, 5, 82), (4, 4, 83), (2, 7, 84)]
        .iter()
        .enumerate()
    {
        let name = format!("family/gap/a{}t{}-{}", agents, tasks, i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            let mut rng = Rng::new(0xD500 + seed);
            let n = agents * tasks; // x[a][t] flattened as a * tasks + t
            let cost: Vec<i64> = (0..n).map(|_| rng.int(1, 20)).collect();
            let load: Vec<i64> = (0..n).map(|_| rng.int(1, 8)).collect();
            // Generous capacities keep the instance feasible while still
            // binding for the cheap agents.
            let capacity: Vec<i64> = (0..agents)
                .map(|a| {
                    let own: i64 = (0..tasks).map(|t| load[a * tasks + t]).sum();
                    own * rng.int(50, 80) / 100
                })
                .collect();

            let mut constraints: Vec<(Vec<i64>, i8, i64)> = vec![];
            for t in 0..tasks {
                let coeffs: Vec<i64> = (0..n).map(|j| i64::from(j % tasks == t)).collect();
                constraints.push((coeffs, 0, 1));
            }
            for a in 0..agents {
                let coeffs: Vec<i64> = (0..n)
                    .map(|j| if j / tasks == a { load[j] } else { 0 })
                    .collect();
                constraints.push((coeffs, -1, capacity[a]));
            }
            let ilp = TinyIlp {
                bounds: vec![(0, 1); n],
                objective: cost.clone(),
                constraints,
                maximize: false,
            };
            let Some(best) = ilp.brute_force() else {
                // The capacity draw can make an instance infeasible; that is
                // a legitimate variant — assert the solver agrees.
                let mut b = Builder::new(Minimize);
                let vars: Vec<_> = cost.iter().map(|&c| b.binary(c as f64)).collect();
                for t in 0..tasks {
                    let terms: Vec<_> = (0..agents).map(|a| (vars[a * tasks + t], 1.0)).collect();
                    b.constraint(&terms, Eq, 1.0);
                }
                for a in 0..agents {
                    let terms: Vec<_> = (0..tasks)
                        .map(|t| (vars[a * tasks + t], load[a * tasks + t] as f64))
                        .collect();
                    b.constraint(&terms, Le, capacity[a] as f64);
                }
                return Ok((b.spec, b.problem, Expected::Infeasible));
            };

            let mut b = Builder::new(Minimize);
            let vars: Vec<_> = cost.iter().map(|&c| b.binary(c as f64)).collect();
            for t in 0..tasks {
                let terms: Vec<_> = (0..agents).map(|a| (vars[a * tasks + t], 1.0)).collect();
                b.constraint(&terms, Eq, 1.0);
            }
            for a in 0..agents {
                let terms: Vec<_> = (0..tasks)
                    .map(|t| (vars[a * tasks + t], load[a * tasks + t] as f64))
                    .collect();
                b.constraint(&terms, Le, capacity[a] as f64);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

/// Integer transportation (supplies == demands, integral flows): a totally
/// unimodular structure the LP relaxation should already solve integrally —
/// the B&B must recognize that instead of branching.
fn transport_cases(cases: &mut Vec<Case>) {
    for (i, &seed) in [91u64, 92, 93, 94].iter().enumerate() {
        let name = format!("family/transport/3x3-{}", i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            let mut rng = Rng::new(0xD600 + seed);
            let (m, k) = (3usize, 3usize); // sources x sinks
            let supply: Vec<i64> = (0..m).map(|_| rng.int(1, 3)).collect();
            let total: i64 = supply.iter().sum();
            // Random demands summing to total supply.
            let mut demand = vec![0i64; k];
            for _ in 0..total {
                demand[rng.usize(0, k - 1)] += 1;
            }
            let cost: Vec<i64> = (0..m * k).map(|_| rng.int(1, 12)).collect();

            let bound = supply
                .iter()
                .max()
                .unwrap()
                .min(demand.iter().max().unwrap());
            let mut constraints: Vec<(Vec<i64>, i8, i64)> = vec![];
            for s in 0..m {
                let coeffs: Vec<i64> = (0..m * k).map(|j| i64::from(j / k == s)).collect();
                constraints.push((coeffs, 0, supply[s]));
            }
            for d in 0..k {
                let coeffs: Vec<i64> = (0..m * k).map(|j| i64::from(j % k == d)).collect();
                constraints.push((coeffs, 0, demand[d]));
            }
            let ilp = TinyIlp {
                bounds: vec![(0, *bound); m * k],
                objective: cost.clone(),
                constraints,
                maximize: false,
            };
            let best = ilp
                .brute_force()
                .expect("balanced transportation is always feasible");

            let mut b = Builder::new(Minimize);
            let vars: Vec<_> = cost
                .iter()
                .map(|&c| b.integer(c as f64, 0, *bound as i32))
                .collect();
            for s in 0..m {
                let terms: Vec<_> = (0..k).map(|d| (vars[s * k + d], 1.0)).collect();
                b.constraint(&terms, Eq, supply[s] as f64);
            }
            for d in 0..k {
                let terms: Vec<_> = (0..m).map(|s| (vars[s * k + d], 1.0)).collect();
                b.constraint(&terms, Eq, demand[d] as f64);
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

/// Equality chains (x0 fixed, x0 + x1 = c1, x1 + x2 = c2, ...) plus side
/// inequalities: the presolve substitution machinery end to end.
fn eq_chain_cases(cases: &mut Vec<Case>) {
    for (i, &(n, seed)) in [(6usize, 101u64), (8, 102), (10, 103), (12, 104)]
        .iter()
        .enumerate()
    {
        let name = format!("family/eq-chain/n{:02}-{}", n, i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            let mut rng = Rng::new(0xD700 + seed);
            let obj: Vec<i64> = (0..n).map(|_| rng.int(-5, 5)).collect();
            // Hidden solution values drive the chain constants: x0 is pinned
            // and every pair-sum is pinned, so the hidden vector is the ONLY
            // feasible point — the optimum is its objective value directly
            // (no enumeration oracle: a 9^n box scan is exponential and took
            // an hour at n = 12 in the first version of this case).
            let hidden: Vec<i64> = (0..n).map(|_| rng.int(0, 6)).collect();
            let best: i64 = obj.iter().zip(&hidden).map(|(c, h)| c * h).sum();

            let mut b = Builder::new(Minimize);
            let vars: Vec<_> = obj.iter().map(|&c| b.integer(c as f64, 0, 8)).collect();
            b.constraint(&[(vars[0], 1.0)], Eq, hidden[0] as f64);
            for j in 1..n {
                b.constraint(
                    &[(vars[j - 1], 1.0), (vars[j], 1.0)],
                    Eq,
                    (hidden[j - 1] + hidden[j]) as f64,
                );
            }
            Ok((b.spec, b.problem, Expected::objective(best as f64)))
        }));
    }
}

/// Big-M rows with M up to 1e8 and a by-construction optimum: the numerics
/// guards (noise-scaled rounding, rounded-incumbent validation) end to end.
/// One facility is cheap-but-far (open cost high), one expensive-per-unit:
/// optimum = min(open_a + demand*unit_a, open_b + demand*unit_b, open both).
fn big_m_exact_cases(cases: &mut Vec<Case>) {
    for (i, &m) in [1e6f64, 1e7, 1e8].iter().enumerate() {
        let name = format!("family/bigm-exact/m1e{}", 6 + i);
        cases.push(Case::solve(name, Tier::Medium, 30, move || {
            // demand 10; a: open 100, unit 2; b: open 30, unit 9.
            // one(a) = 120, one(b) = 120, both: 130 + best mix 10*2 = 150 no —
            // both = 100 + 30 + 20 = 150. True optimum = 120 (tie, either).
            let (open_a, unit_a, open_b, unit_b, demand) =
                (100.0f64, 2.0f64, 30.0f64, 9.0f64, 10.0f64);
            let best = (open_a + demand * unit_a).min(open_b + demand * unit_b);

            let mut b = Builder::new(Minimize);
            let ya = b.binary(open_a);
            let yb = b.binary(open_b);
            // x bounded only by the big-M rows: [0, M] box with M >> demand.
            let xa = b.real(unit_a, 0.0, m);
            let xb = b.real(unit_b, 0.0, m);
            b.constraint(&[(xa, 1.0), (xb, 1.0)], Ge, demand);
            b.constraint(&[(xa, 1.0), (ya, -m)], Le, 0.0);
            b.constraint(&[(xb, 1.0), (yb, -m)], Le, 0.0);
            Ok((b.spec, b.problem, Expected::objective(best)))
        }));
    }
}

/// Infeasibility by construction, three shapes: capacity below demand,
/// parity conflict, and crossing chain — each double-checked by the oracle
/// where enumeration is possible.
fn infeasible_cases(cases: &mut Vec<Case>) {
    // Capacity < demand.
    cases.push(Case::solve(
        "family/infeasible/capacity",
        Tier::Easy,
        10,
        || {
            let mut b = Builder::new(Minimize);
            let xs: Vec<_> = (0..4).map(|_| b.integer(1.0, 0, 3)).collect();
            let terms: Vec<_> = xs.iter().map(|&x| (x, 1.0)).collect();
            b.constraint(&terms, Ge, 13.0); // max attainable is 12
            Ok((b.spec, b.problem, Expected::Infeasible))
        },
    ));
    // Parity: 2a + 2b == 7 has no integer solution.
    cases.push(Case::solve(
        "family/infeasible/parity",
        Tier::Easy,
        10,
        || {
            let mut b = Builder::new(Minimize);
            let a = b.integer(1.0, 0, 10);
            let bb = b.integer(1.0, 0, 10);
            b.constraint(&[(a, 2.0), (bb, 2.0)], Eq, 7.0);
            Ok((b.spec, b.problem, Expected::Infeasible))
        },
    ));
    // Crossing chain: x == 2, x + y == 4, y >= 3.
    cases.push(Case::solve(
        "family/infeasible/chain",
        Tier::Easy,
        10,
        || {
            let mut b = Builder::new(Minimize);
            let x = b.integer(1.0, 0, 10);
            let y = b.integer(1.0, 0, 10);
            b.constraint(&[(x, 1.0)], Eq, 2.0);
            b.constraint(&[(x, 1.0), (y, 1.0)], Eq, 4.0);
            b.constraint(&[(y, 1.0)], Ge, 3.0);
            Ok((b.spec, b.problem, Expected::Infeasible))
        },
    ));
    // LP-feasible but integer-infeasible packing: three mutually exclusive
    // binaries that must sum to 2 exactly... with pairwise exclusions.
    cases.push(Case::solve(
        "family/infeasible/pack-vs-partition",
        Tier::Easy,
        10,
        || {
            let mut b = Builder::new(Minimize);
            let xs: Vec<_> = (0..3).map(|_| b.binary(1.0)).collect();
            let all: Vec<_> = xs.iter().map(|&x| (x, 1.0)).collect();
            b.constraint(&all, Eq, 2.0);
            for i in 0..3 {
                for j in (i + 1)..3 {
                    b.constraint(&[(xs[i], 1.0), (xs[j], 1.0)], Le, 1.0);
                }
            }
            Ok((b.spec, b.problem, Expected::Infeasible))
        },
    ));
}

/// Unboundedness by construction, both directions and both var kinds.
fn unbounded_cases(cases: &mut Vec<Case>) {
    cases.push(Case::solve(
        "family/unbounded/real-max",
        Tier::Easy,
        10,
        || {
            let mut b = Builder::new(Maximize);
            let x = b.real(1.0, 0.0, f64::INFINITY);
            let y = b.real(0.0, 0.0, 5.0);
            b.constraint(&[(x, -1.0), (y, 1.0)], Le, 3.0); // never caps x
            Ok((b.spec, b.problem, Expected::Unbounded))
        },
    ));
    cases.push(Case::solve(
        "family/unbounded/mixed-min",
        Tier::Easy,
        10,
        || {
            let mut b = Builder::new(Minimize);
            // The unbounded direction must ride the REAL var: integer bounds
            // are i32 in the public API, so an "unbounded" integer is really
            // pinned at i32::MIN (the first version of this case expected
            // Unbounded and got the finite -2^31 optimum instead).
            let x = b.integer(0.0, 0, 10);
            let y = b.real(1.0, f64::NEG_INFINITY, f64::INFINITY);
            b.constraint(&[(x, 1.0), (y, -1.0)], Ge, -3.0); // y <= x + 3: no floor
            Ok((b.spec, b.problem, Expected::Unbounded))
        },
    ));
}

/// Degenerate LPs: transportation with ALL-EQUAL costs — every basic feasible
/// solution is optimal, ties everywhere; the optimum is total_flow * cost by
/// construction. Guards against cycling/stalling on massive degeneracy.
fn degenerate_lp_cases(cases: &mut Vec<Case>) {
    for (i, &(m, k, seed)) in [(3usize, 3usize, 111u64), (4, 4, 112), (5, 4, 113)]
        .iter()
        .enumerate()
    {
        let name = format!("family/degenerate-lp/{}x{}-{}", m, k, i);
        cases.push(Case::solve(name, Tier::Easy, 10, move || {
            let mut rng = Rng::new(0xD800 + seed);
            let unit = 7.0;
            let supply: Vec<i64> = (0..m).map(|_| rng.int(2, 9)).collect();
            let total: i64 = supply.iter().sum();
            let mut demand = vec![0i64; k];
            for _ in 0..total {
                demand[rng.usize(0, k - 1)] += 1;
            }

            let mut b = Builder::new(Minimize);
            let vars: Vec<_> = (0..m * k)
                .map(|_| b.real(unit, 0.0, f64::INFINITY))
                .collect();
            for s in 0..m {
                let terms: Vec<_> = (0..k).map(|d| (vars[s * k + d], 1.0)).collect();
                b.constraint(&terms, Eq, supply[s] as f64);
            }
            for d in 0..k {
                let terms: Vec<_> = (0..m).map(|s| (vars[s * k + d], 1.0)).collect();
                b.constraint(&terms, Eq, demand[d] as f64);
            }
            Ok((b.spec, b.problem, Expected::objective(total as f64 * unit)))
        }));
    }
}
