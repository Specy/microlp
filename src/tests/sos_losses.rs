//! Synthetic interconnector-loss models, for the search's handling of many
//! special ordered sets at once.
//!
//! Five regions are joined by six links, in the shape of a radial market:
//! two parallel links between regions 0 and 1, one between 1 and 2, two
//! between 2 and 3, and one between 2 and 4. Each link's flow `t` lies in
//! `[-F, F]`, and its loss `c t²` is shared equally between its two ends.
//! Flow and loss are written through breakpoint weights `λ_i` (`Σ λ = 1`),
//! with one SOS2 per link. Some regions have generation offered at negative
//! prices, so energy is worth burning: without the sets, the LP mixes
//! far-apart breakpoints to create losses the loss curve does not allow.
//!
//! `cargo test --release sos_losses_sweep -- --ignored --nocapture` prints
//! the node count and time per seed (seeds `SWEEP_FIRST..SWEEP_FIRST +
//! SWEEP_SEEDS`, set by environment variables of those names).

use crate::{
    ComparisonOp, OptimizationDirection, Problem, SolveOptions, TerminationReason, Variable,
};
use rand::prelude::*;

const LINKS: [(usize, usize); 6] = [(0, 1), (0, 1), (1, 2), (2, 3), (2, 3), (2, 4)];
const REGIONS: usize = 5;

pub(crate) struct LossModel {
    pub problem: Problem,
    /// Each link's breakpoint weights, in breakpoint order.
    pub lambda: Vec<Vec<Variable>>,
}

/// The model of `seed`, with one SOS2 per link if `with_sets`.
pub(crate) fn loss_model(seed: u64, with_sets: bool) -> LossModel {
    let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let mut rows: Vec<Vec<(Variable, f64)>> = vec![Vec::new(); REGIONS];
    let mut load = [0.0; REGIONS];
    for r in 0..REGIONS {
        load[r] = rng.random_range(200.0..1500.0);
        for _ in 0..rng.random_range(3..7) {
            let negative = rng.random_range(0.0..1.0) < 0.35;
            let cost = if negative {
                rng.random_range(-1000.0..-1.0)
            } else {
                rng.random_range(0.0..300.0)
            };
            let cap = rng.random_range(100.0..900.0);
            let g = p.add_var(cost, (0.0, cap));
            rows[r].push((g, 1.0));
        }
        // Deficit and surplus, priced far above any offer.
        let deficit = p.add_var(15000.0, (0.0, f64::INFINITY));
        let surplus = p.add_var(15000.0, (0.0, f64::INFINITY));
        rows[r].push((deficit, 1.0));
        rows[r].push((surplus, -1.0));
    }
    let mut lambda = Vec::new();
    for &(from, to) in &LINKS {
        let cap: f64 = rng.random_range(300.0..1500.0);
        let k: usize = rng.random_range(60..121);
        let c = rng.random_range(0.03..0.10) / cap;
        // Symmetric in exact arithmetic, so the middle breakpoint of an odd
        // `k` is exactly 0, not a round-off speck.
        let t: Vec<f64> = (0..k)
            .map(|i| cap * (2 * i) as f64 / (k - 1) as f64 - cap)
            .map(|t| if t.abs() < 1e-9 * cap { 0.0 } else { t })
            .collect();
        let l: Vec<Variable> = (0..k).map(|_| p.add_var(0.0, (0.0, 1.0))).collect();
        p.add_constraint(l.iter().map(|&v| (v, 1.0)), ComparisonOp::Eq, 1.0);
        for (i, &v) in l.iter().enumerate() {
            let loss = c * t[i] * t[i];
            rows[from].push((v, -t[i] - 0.5 * loss));
            rows[to].push((v, t[i] - 0.5 * loss));
        }
        if with_sets {
            p.add_sos2_constraint(l.iter().zip(&t).map(|(&v, &ti)| (v, ti)));
        }
        lambda.push(l);
    }
    for r in 0..REGIONS {
        p.add_constraint(rows[r].clone(), ComparisonOp::Eq, load[r]);
    }
    LossModel { problem: p, lambda }
}

/// How many links have nonzero weights more than one segment apart.
fn violated(lambda: &[Vec<Variable>], value: impl Fn(Variable) -> f64) -> usize {
    lambda
        .iter()
        .filter(|l| {
            let nz: Vec<usize> = (0..l.len()).filter(|&i| value(l[i]).abs() > 1e-7).collect();
            nz.len() >= 2 && nz[nz.len() - 1] - nz[0] >= 2
        })
        .count()
}

/// Seed 5 breaks four of the six sets at the LP optimum. Splitting the first
/// violated set had not finished after 50,000 nodes; splitting the most
/// violated one proves the optimum in under a hundred.
#[test]
fn many_loss_curves_are_solved_in_a_small_tree() {
    let lp = loss_model(5, false);
    let relaxed = lp.problem.solve().unwrap().into_solution().unwrap();
    assert_eq!(violated(&lp.lambda, |v| relaxed.var_value(v)), 4);

    let m = loss_model(5, true);
    let mut options = SolveOptions::default();
    options.node_limit = Some(1_000);
    let outcome = m.problem.solve_with(options).unwrap();
    assert_eq!(
        outcome.termination_reason(),
        TerminationReason::ProvenOptimal
    );
    let solution = outcome.into_solution().unwrap();
    assert_eq!(violated(&m.lambda, |v| solution.var_value(v)), 0);
    assert!(solution.objective() > relaxed.objective());
}

/// A hint that holds at zero the weights outside the segments a solve chose
/// is completed by an LP whose point meets every set, and adopted before the
/// first node: a solve with no node budget returns it.
#[test]
fn a_hint_of_zeros_outside_the_segments_seeds_the_incumbent() {
    let m = loss_model(5, true);
    let optimum = m.problem.solve().unwrap().into_solution().unwrap();
    let hint: Vec<(Variable, f64)> = m
        .lambda
        .iter()
        .flatten()
        .filter(|&&v| optimum.var_value(v).abs() <= 1e-9)
        .map(|&v| (v, 0.0))
        .collect();
    let mut options = SolveOptions::default();
    options.node_limit = Some(0);
    options.warm_start = Some(hint);
    let outcome = m.problem.solve_with(options).unwrap();
    let solution = outcome
        .into_solution()
        .expect("the hint's completion is adopted");
    assert_eq!(violated(&m.lambda, |v| solution.var_value(v)), 0);
    assert!((solution.objective() - optimum.objective()).abs() <= 1e-6 * optimum.objective().abs());
    // Without the hint, the same budget finds nothing.
    let mut cold = SolveOptions::default();
    cold.node_limit = Some(0);
    assert!(m.problem.solve_with(cold).unwrap().into_solution().is_err());
}

#[test]
#[ignore = "a measurement, not a check: prints nodes and time per seed"]
fn sos_losses_sweep() {
    let env = |name: &str, default: u64| {
        std::env::var(name)
            .ok()
            .map_or(default, |s| s.parse().unwrap())
    };
    let first = env("SWEEP_FIRST", 0);
    let seeds = env("SWEEP_SEEDS", 24);
    let node_limit = std::env::var("NODE_LIMIT").ok().map(|s| s.parse().unwrap());
    let (mut nodes, mut secs) = (0, 0.0);
    for seed in first..first + seeds {
        let lp = loss_model(seed, false);
        let relaxed = lp.problem.solve().unwrap().into_solution().unwrap();
        let m = loss_model(seed, true);
        let mut options = SolveOptions::default();
        options.node_limit = node_limit;
        let start = web_time::Instant::now();
        let outcome = m.problem.solve_with(options).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let stats = outcome.stats();
        let objective = outcome.into_solution().map_or(f64::NAN, |s| s.objective());
        nodes += stats.nodes_solved;
        secs += elapsed;
        println!(
            "seed {seed:3}  violated {}  lp {:14.4}  sos {objective:14.4}  nodes {:7}  {elapsed:8.3} s",
            violated(&lp.lambda, |v| relaxed.var_value(v)),
            relaxed.objective(),
            stats.nodes_solved,
        );
    }
    println!("total nodes {nodes}  {secs:.2} s");
}
