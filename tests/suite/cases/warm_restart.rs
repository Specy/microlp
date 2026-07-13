//! Warm-restart and node-limit-stepping integration cases on REAL problems.
//!
//! Unlike `incr::resume-*` (which continues the SAME persistent search state
//! via `Solution::resume`), these cases rebuild a brand-new `Problem` from
//! scratch every round and hand the previous round's incumbent in purely
//! through `SolveOptions::warm_start` — the "periodically checkpoint and
//! restart cold" workflow, as opposed to "keep the process alive and resume".
//!
//! Two families, sharing the same restart loop (`run_restart_loop`):
//!   - `milp/warm-restart-*`    : each round gets a wall-clock slice.
//!   - `milp/nodelimit-steps-*` : each round gets a node budget — the
//!     deterministic twin, with no wall-clock sensitivity at all.
//!
//! ## Why the per-round budget GROWS instead of staying fixed
//!
//! A fixed per-round budget does not converge. Root cause, confirmed
//! empirically on all three real/oracle instances below: this driver's
//! branch & bound is fully deterministic, and a "restart from scratch"
//! round discards
//! *all* search state except the incumbent value (unlike `Solution::resume`,
//! which keeps the open-node frontier and therefore always advances). Once
//! the warm-start hint stops improving (typically within the first couple of
//! rounds — finding a good incumbent is cheap), every subsequent round with
//! an identical hint and an identical fixed budget re-explores the exact same
//! node sequence and gets cut off at the exact same point — making *zero*
//! forward progress, forever. This is not specific to a weak instance: for
//! `stein27`, `mod008` and the hard knapsack, proving optimality with the
//! true optimal solution handed in as the hint up front still costs
//! 66-99% of the nodes a cold, hint-free solve needs (finding a good
//! solution is cheap; *proving* no better one exists is not) — so no small
//! fixed budget can ever finish the proof, regardless of how good the hint
//! is. Growing the budget a fixed, deterministic amount each round (doubling;
//! no randomness, no wall-clock dependence for the node-limit family) fixes
//! this: it costs some repeated work in the early rounds, but guarantees the
//! round eventually gets far enough to close the proof — and the growth is
//! clamped (see `geometric`) so a regression fails in bounded time.
//! Every other property the task asks for is unchanged and still strictly
//! asserted: fresh `Problem` each round, `warm_start`-only hint carry-over,
//! monotone non-worsening incumbents, and the loop must reach `Status::Optimal`
//! to succeed (exhausting the round guard is a hard failure, not tolerated).
//!
//! Real problems exercised: the MIPLIB3 instances `stein27` (optimum 18) and
//! `mod008` (optimum 307), read through the suite's own MPS reader (same
//! files/reader as `cases::miplib`), plus one oracle-backed random MILP: a
//! strongly-correlated 0/1 knapsack (value = weight + 10) — classically much
//! harder for branch & bound than uncorrelated instances (weak LP bound,
//! lots of near-ties) — checked against the exact DP oracle
//! `oracles::knapsack01`.

use super::{locate, read_instance};
use super::{Case, Tier};
use crate::model::{Builder, ModelSpec, Tol};
use crate::oracles;
use crate::rng::Rng;
use crate::verify;
use microlp::ComparisonOp::Le;
use microlp::OptimizationDirection::{Maximize, Minimize};
use microlp::{OptimizationDirection, Problem, SolveOptions, Status, Variable};
use std::time::Duration;

/// Objective tolerance for the known optima checked here (all exact integers
/// from published MIPLIB values or an exact DP oracle) — matches
/// `cases::miplib`'s `EXACT`.
const EXACT: Tol = Tol {
    abs: 1e-4,
    rel: 1e-9,
};

// ------------------------------------------------------------- real problems

/// Parse a MIPLIB3 instance fresh from its `.mps` file (Minimize; not
/// relaxed) — the same file and reader `cases::miplib` uses, called anew so
/// every round gets a genuinely fresh `Problem`/shadow-model pair, just like
/// a caller reloading a saved problem definition from disk would.
fn build_from_mps(name: &str) -> Result<(ModelSpec, Problem), String> {
    let path = locate("miplib3", &format!("{}.mps", name)).0;
    let text = read_instance(&path)?;
    let parsed = crate::mps_milp::parse(&text, OptimizationDirection::Minimize, false)?;
    if parsed.obj_offset != 0.0 {
        // None of the vendored files carries an objective constant (see
        // cases::miplib::load); guard against a future data change silently
        // shifting the expected optimum.
        return Err(format!(
            "{} has objective offset {}; expected optimum needs adjusting",
            name, parsed.obj_offset
        ));
    }
    Ok((parsed.spec, parsed.problem))
}

/// Frozen data for a strongly-correlated 0/1 knapsack (value = weight + 10):
/// classically much harder for branch & bound than uncorrelated instances
/// (weak LP bound, lots of symmetric near-ties) — an uncorrelated knapsack up
/// to n=70 with this same RNG solved in under 3 ms on the dev machine, too
/// easy to exercise multi-round restart behavior. n=40, seed fixed for
/// reproducibility; measured full (unlimited) solve ~1.8 s / ~107,000 nodes
/// — comparable in scale to mod008.
fn hard_knapsack_data() -> (Vec<i64>, Vec<i64>, i64) {
    let n = 40usize;
    let mut rng = Rng::new(0xC0FFEE + n as u64);
    let weights: Vec<i64> = (0..n).map(|_| rng.int(10, 1000)).collect();
    let values: Vec<i64> = weights.iter().map(|&w| w + 10).collect();
    let capacity = (weights.iter().sum::<i64>() as f64 * 0.5).floor() as i64;
    (weights, values, capacity)
}

fn build_knapsack(weights: &[i64], values: &[i64], capacity: i64) -> (ModelSpec, Problem) {
    let mut b = Builder::new(Maximize);
    let vars: Vec<_> = values.iter().map(|&v| b.binary(v as f64)).collect();
    let terms: Vec<_> = vars
        .iter()
        .zip(weights)
        .map(|(&v, &w)| (v, w as f64))
        .collect();
    b.constraint(&terms, Le, capacity as f64);
    (b.spec, b.problem)
}

// ------------------------------------------------------------- restart loop

enum RoundBudget {
    Time(Duration),
    Nodes(u64),
}

/// True if `new_obj` is no worse than `hint_obj` (within `tol`'s slack around
/// the hint), given the optimization direction: minimizing, no worse means
/// not greater; maximizing, no worse means not smaller.
fn no_worse(direction: OptimizationDirection, new_obj: f64, hint_obj: f64, tol: Tol) -> bool {
    let slack = tol.slack(hint_obj);
    match direction {
        Minimize => new_obj <= hint_obj + slack,
        Maximize => new_obj >= hint_obj - slack,
    }
}

/// Restart-from-scratch loop shared by the warm-restart (time-limit) and
/// node-limit-stepping case families. Each round rebuilds a FRESH `Problem`
/// via `build`, solves it under `schedule(round)` with the previous round's
/// incumbent passed in through `SolveOptions::warm_start`, and folds the
/// outcome:
/// - `Optimal`: validate the incumbent against the shadow model and the
///   known optimum, then stop (the loop's only success path — see the
///   return type).
/// - `Feasible`: validate the incumbent against the shadow model, assert it
///   is no worse than the previous round's hint (warm start must never
///   regress the incumbent), then carry it forward as the next hint.
/// - `Interrupted`: no incumbent this round; the previous hint (if any)
///   carries forward unchanged.
///
/// `schedule` maps a 1-based round number to that round's budget; see the
/// module doc for why it grows rather than staying fixed. Returns the number
/// of rounds taken to reach `Optimal`. Exhausting `max_rounds` without doing
/// so is a hard failure (a real non-convergence, never tolerated) — this is
/// what makes "the loop terminated with Optimal, not the guard" an assertion
/// rather than an assumption.
fn run_restart_loop<B, S>(
    build: B,
    direction: OptimizationDirection,
    optimum: f64,
    tol: Tol,
    schedule: S,
    max_rounds: u32,
) -> Result<u32, String>
where
    B: Fn() -> Result<(ModelSpec, Problem), String>,
    S: Fn(u32) -> RoundBudget,
{
    let mut hint: Option<Vec<(Variable, f64)>> = None;
    let mut hint_obj: Option<f64> = None;

    for round in 1..=max_rounds {
        let (spec, problem) = build()?;
        let mut options = SolveOptions::default();
        match schedule(round) {
            RoundBudget::Time(d) => options.time_limit = Some(d),
            RoundBudget::Nodes(n) => options.node_limit = Some(n),
        }
        options.warm_start = hint.clone();

        let sol = problem
            .solve_with(options)
            .map_err(|e| format!("round {}: solve errored: {}", round, e))?;

        match sol.status() {
            Status::Optimal => {
                verify::validate_incumbent(&spec, &sol)?;
                if !tol.matches(sol.objective(), optimum) {
                    return Err(format!(
                        "round {}: expected optimum {}, got {} (diff {:.3e})",
                        round,
                        optimum,
                        sol.objective(),
                        (sol.objective() - optimum).abs()
                    ));
                }
                return Ok(round);
            }
            Status::Feasible => {
                verify::validate_incumbent(&spec, &sol)?;
                let obj = sol.objective();
                if let Some(prev) = hint_obj {
                    if !no_worse(direction, obj, prev, tol) {
                        return Err(format!(
                            "round {}: warm-started incumbent {} is WORSE than the previous \
                             round's hint {} — warm start must never regress the incumbent \
                             (solver bug)",
                            round, obj, prev
                        ));
                    }
                }
                hint = Some(sol.iter().collect());
                hint_obj = Some(obj);
            }
            Status::Interrupted => {
                // No incumbent this round; keep the previous hint (if any).
            }
        }
    }

    Err(format!(
        "did not reach Optimal within the {}-round guard (last hint objective: {:?})",
        max_rounds, hint_obj
    ))
}

/// A geometric schedule `start * growth^(round-1)`, exponent clamped at 8 so
/// a convergence regression stalls at a bounded plateau instead of growing
/// forever — a hung regression must become a bounded FAIL (the round guard),
/// never a hung suite.
///
/// Worst-case totals if a case never converged, with `MAX_ROUNDS` = 40: the
/// plateau `growth^8` is reached at round 9, leaving 31 plateau rounds, so
/// for growth 2 the summed budget is `start * (2^0 + .. + 2^8 + 31 * 2^8)`
/// = `start * 8447`:
/// - warm-restart-stein27: 15 ms * 8447 ~ 2.1 min
/// - warm-restart-mod008: 100 ms * 8447 ~ 14.1 min (the largest; still
///   inside the ~15 min review bound)
/// - warm-restart-knap-hard: 60 ms * 8447 ~ 8.4 min
/// - nodelimit-steps-stein27: 250 * 8447 ~ 2.1M nodes ~ 2 min at the
///   measured ~19k nodes/s
/// - nodelimit-steps-knap-hard: 1,000 * 8447 ~ 8.4M nodes ~ 2.4 min at the
///   measured ~59k nodes/s (its growth was tightened from 3 to 2 for exactly
///   this bound: at growth 3 the plateau alone is 3^8 ~ 6.6M nodes/round,
///   a ~60 min worst case)
///
/// Today's convergence (5-8 rounds, budgets well below the plateau) is
/// unaffected by the clamp.
fn geometric(start: f64, growth: f64, round: u32) -> f64 {
    start * growth.powi(round.saturating_sub(1).min(8) as i32)
}

/// Round guard for `run_restart_loop`: far above today's 5-8 round
/// convergence, low enough that a total failure to converge exhausts the
/// `geometric` worst cases above in minutes, not hours.
const MAX_ROUNDS: u32 = 40;

// ------------------------------------------------------------------- cases

pub fn register(cases: &mut Vec<Case>) {
    warm_restart_time_cases(cases);
    nodelimit_step_cases(cases);
    mwis_warm_start_cases(cases);
}

fn warm_restart_time_cases(cases: &mut Vec<Case>) {
    // stein27 (MIPLIB, optimum 18, Minimize). Cold full solve measured at
    // ~530 ms / ~10,200 nodes; proving optimality even with the true optimum
    // pre-seeded still costs ~9,850 nodes / ~500 ms (see module doc). Slice
    // starts at 15 ms (~3% of the cold solve — reliably interrupts round 1,
    // following incr::resume-midway-milp's small-fraction-of-full-solve
    // approach to sizing) and doubles each round.
    cases.push(Case::custom(
        "milp/warm-restart-stein27",
        Tier::Medium,
        30,
        |_budget| {
            run_restart_loop(
                || build_from_mps("stein27"),
                Minimize,
                18.0,
                EXACT,
                |round| RoundBudget::Time(Duration::from_secs_f64(geometric(0.015, 2.0, round))),
                MAX_ROUNDS,
            )?;
            Ok(())
        },
    ));

    // mod008 (MIPLIB, optimum 307, Minimize). Cold full solve ~2.3 s /
    // ~39,500 nodes; proving-with-optimal-hint ~26,100 nodes / ~1.3 s. Kept
    // Hard tier like its plain miplib/mod008/int case (same instance, same
    // reason: too slow for the default run). Slice starts at 100 ms (~4% of
    // the cold solve) and doubles each round.
    cases.push(Case::custom(
        "milp/warm-restart-mod008",
        Tier::Hard,
        120,
        |_budget| {
            run_restart_loop(
                || build_from_mps("mod008"),
                Minimize,
                307.0,
                EXACT,
                |round| RoundBudget::Time(Duration::from_secs_f64(geometric(0.1, 2.0, round))),
                MAX_ROUNDS,
            )?;
            Ok(())
        },
    ));

    // Oracle-backed random MILP: the hard (strongly-correlated) knapsack.
    // Cold full solve ~1.8 s / ~107,000 nodes; proving-with-optimal-hint
    // ~106,000 nodes / ~1.6 s (barely less — the LP bound is weak by
    // construction). Slice starts at 60 ms (~3% of the cold solve) and
    // doubles each round. Hard tier: heavier than the Standard-tier norm.
    cases.push(Case::custom(
        "milp/warm-restart-knap-hard",
        Tier::Hard,
        60,
        |_budget| {
            let (weights, values, capacity) = hard_knapsack_data();
            let best = oracles::knapsack01(&values, &weights, capacity) as f64;
            run_restart_loop(
                || Ok(build_knapsack(&weights, &values, capacity)),
                Maximize,
                best,
                EXACT,
                |round| RoundBudget::Time(Duration::from_secs_f64(geometric(0.06, 2.0, round))),
                MAX_ROUNDS,
            )?;
            Ok(())
        },
    ));
}

fn nodelimit_step_cases(cases: &mut Vec<Case>) {
    // Deterministic twin of milp/warm-restart-stein27: node_limit instead of
    // time_limit, so the round count and every intermediate outcome are
    // exactly reproducible (no wall-clock sensitivity at all). K starts at
    // 250 nodes (~2.5% of the ~9,850-node proof cost) and doubles each round.
    cases.push(Case::custom(
        "milp/nodelimit-steps-stein27",
        Tier::Medium,
        30,
        |_budget| {
            run_restart_loop(
                || build_from_mps("stein27"),
                Minimize,
                18.0,
                EXACT,
                |round| RoundBudget::Nodes(geometric(250.0, 2.0, round) as u64),
                MAX_ROUNDS,
            )?;
            Ok(())
        },
    ));

    // Deterministic twin of milp/warm-restart-knap-hard. K starts at 1,000
    // nodes (~1% of the ~106,000-node proof cost) and doubles each round:
    // round 8's 128,000-node budget exceeds even the 107,028-node COLD proof
    // cost, so convergence there is guaranteed whatever the hint. (An earlier
    // draft tripled instead, converging in 6 rounds, but with the growth^8
    // plateau its never-converges worst case was ~60 min — doubling keeps it
    // at ~2.4 min; see `geometric`.)
    cases.push(Case::custom(
        "milp/nodelimit-steps-knap-hard",
        Tier::Hard,
        60,
        |_budget| {
            let (weights, values, capacity) = hard_knapsack_data();
            let best = oracles::knapsack01(&values, &weights, capacity) as f64;
            run_restart_loop(
                || Ok(build_knapsack(&weights, &values, capacity)),
                Maximize,
                best,
                EXACT,
                |round| RoundBudget::Nodes(geometric(1000.0, 2.0, round) as u64),
                MAX_ROUNDS,
            )?;
            Ok(())
        },
    ));
}

// ----------------------------------------------------------------- MWIS helpers

mod mwis {
    use super::Rng;
    use bit_set::BitSet;
    use microlp::{ComparisonOp::Le, OptimizationDirection::Maximize, Problem, SolveOptions,
        Status, Variable};

    pub type Node = usize;
    pub type Edge = (Node, Node);

    #[derive(Debug, Clone)]
    pub struct Graph {
        neighborhoods: Vec<BitSet>,
    }

    impl Graph {
        pub fn new(size: usize) -> Self {
            Self {
                neighborhoods: vec![BitSet::with_capacity(size); size],
            }
        }

        pub fn size(&self) -> usize {
            self.neighborhoods.len()
        }

        pub fn neighbors(&self, v: Node) -> &BitSet {
            &self.neighborhoods[v]
        }

        pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
            self.neighborhoods
                .iter()
                .enumerate()
                .flat_map(|(v, neighbors)| neighbors.iter().map(move |w: usize| (v, w)))
                .filter(|(v, w)| v < w)
        }

        pub fn insert_edge(&mut self, v: Node, w: Node) {
            self.neighborhoods[v].insert(w);
            self.neighborhoods[w].insert(v);
        }
    }

    pub fn is_independent_set(graph: &Graph, set: &BitSet) -> bool {
        set.iter()
            .all(|v| graph.neighbors(v).intersection(set).count() == 0)
    }

    /// Build a random maximal independent set by scanning nodes in a random
    /// permutation: add a node if none of its neighbors were already selected.
    pub fn random_independent_set(graph: &Graph, rng: &mut Rng) -> BitSet {
        let n = graph.size();
        let mut perm: Vec<usize> = (0..n).collect();
        rng.shuffle(&mut perm);
        let mut choosable: BitSet = (0..n).collect();
        let mut selected = BitSet::new();
        for node in perm {
            if !choosable.contains(node) {
                continue;
            }
            choosable.remove(node);
            selected.insert(node);
            choosable.difference_with(graph.neighbors(node));
        }
        selected
    }

    /// Build an Erdős–Rényi random graph: each edge appears independently with
    /// probability `edge_pct / 100`.
    pub fn random_graph(size: usize, edge_pct: u32, rng: &mut Rng) -> Graph {
        let mut graph = Graph::new(size);
        for v in 0..size {
            for w in v + 1..size {
                if rng.int(0, 99) < edge_pct as i64 {
                    graph.insert_edge(v, w);
                }
            }
        }
        graph
    }

    /// Solve MWIS with microlp, optionally warm-starting from `hint`.
    /// Returns (selected set, optimal objective) on success.
    pub fn solve_microlp(
        graph: &Graph,
        valuation: &[f64],
        hint: Option<&BitSet>,
    ) -> Result<(BitSet, f64), String> {
        let mut problem = Problem::new(Maximize);
        let vars: Vec<Variable> = (0..graph.size())
            .map(|v| problem.add_binary_var(valuation[v]))
            .collect();
        for (v, w) in graph.edges() {
            problem.add_constraint(&[(vars[v], 1.0), (vars[w], 1.0)], Le, 1.0);
        }
        let mut opts = SolveOptions::default();
        opts.warm_start = hint.map(|set| {
            (0..graph.size())
                .map(|v| (vars[v], if set.contains(v) { 1.0 } else { 0.0 }))
                .collect()
        });
        let sol = problem
            .solve_with(opts)
            .map_err(|e| format!("microlp MWIS solve failed: {}", e))?;
        match sol.status() {
            Status::Optimal => {}
            s => {
                return Err(format!(
                    "microlp MWIS did not reach Optimal (status: {:?})",
                    s
                ))
            }
        }
        let selected: BitSet = (0..graph.size())
            .filter(|&v| sol[vars[v]] > 0.5)
            .collect();
        Ok((selected, sol.objective()))
    }

    /// Solve MWIS with HiGHS and return its optimal objective.
    #[cfg(feature = "highs")]
    pub fn solve_highs(graph: &Graph, valuation: &[f64]) -> Result<f64, String> {
        let mut pb = highs::RowProblem::default();
        let cols: Vec<highs::Col> = (0..graph.size())
            .map(|v| pb.add_integer_column(valuation[v], 0.0..=1.0))
            .collect();
        for (v, w) in graph.edges() {
            pb.add_row(..=1.0_f64, vec![(cols[v], 1.0), (cols[w], 1.0)]);
        }
        let mut model = pb.optimise(highs::Sense::Maximise);
        model.set_option("output_flag", false);
        let solved = model.solve();
        match solved.status() {
            highs::HighsModelStatus::Optimal => {
                let values = solved.get_solution().columns().to_vec();
                let obj: f64 = (0..graph.size()).map(|v| valuation[v] * values[v]).sum();
                Ok(obj)
            }
            s => Err(format!("HiGHS MWIS returned status {:?}", s)),
        }
    }
}

// ------------------------------------------------------- MWIS warm-start cases

fn mwis_warm_start_cases(cases: &mut Vec<Case>) {
    // Three reproducible instances at different sizes and densities. Each
    // case runs two (valuation, hint) pairs. Per pair:
    //   1. Generate a random independent set as the warm-start hint.
    //   2. Solve cold (no hint) and warm (arbitrary hint) — both to Optimal.
    //   3. Assert the returned sets are independent and that their reported
    //      objectives match the recomputed sums over the valuation.
    //   4. Assert cold and warm agree on the optimal objective value.
    //   5. Cross-validate against HiGHS when built with --features highs.
    for (seed, n, edge_pct, tier, budget) in [
        (0xABCD_u64, 100_usize,  5_u32, Tier::Medium, 30_u64),
        (0x1234_u64, 100_usize, 15_u32, Tier::Medium, 30_u64),
        (0xDEAD_u64, 100_usize, 25_u32, Tier::Hard,   90_u64),
        (0xBEEF_u64, 100_usize, 50_u32, Tier::Hard,   90_u64),
    ] {
        let name = format!("milp/mwis-warm-start-n{}-s{:x}", n, seed);
        cases.push(Case::custom(name, tier, budget, move |_budget| {
            let mut rng = Rng::new(seed);
            let graph = mwis::random_graph(n, edge_pct, &mut rng);

            for pair in 0..2_u32 {
                let valuation: Vec<f64> = (0..n).map(|_| rng.int(1, 100) as f64).collect();
                let hint = mwis::random_independent_set(&graph, &mut rng);

                if !mwis::is_independent_set(&graph, &hint) {
                    return Err(format!(
                        "pair {}: random_independent_set produced a non-independent set",
                        pair
                    ));
                }

                // Cold solve — no warm-start hint.
                let (cold_set, cold_obj) = mwis::solve_microlp(&graph, &valuation, None)?;
                if !mwis::is_independent_set(&graph, &cold_set) {
                    return Err(format!(
                        "pair {}: cold solve returned a non-independent set",
                        pair
                    ));
                }
                let cold_recomputed: f64 = cold_set.iter().map(|v| valuation[v]).sum();
                if (cold_obj - cold_recomputed).abs() > 1e-6 {
                    return Err(format!(
                        "pair {}: cold objective {:.6} != recomputed sum {:.6}",
                        pair, cold_obj, cold_recomputed
                    ));
                }

                // Warm solve — arbitrary independent set as hint.
                let (warm_set, warm_obj) =
                    mwis::solve_microlp(&graph, &valuation, Some(&hint))?;
                if !mwis::is_independent_set(&graph, &warm_set) {
                    return Err(format!(
                        "pair {}: warm solve returned a non-independent set",
                        pair
                    ));
                }
                let warm_recomputed: f64 = warm_set.iter().map(|v| valuation[v]).sum();
                if (warm_obj - warm_recomputed).abs() > 1e-6 {
                    return Err(format!(
                        "pair {}: warm objective {:.6} != recomputed sum {:.6}",
                        pair, warm_obj, warm_recomputed
                    ));
                }

                // Both solves must agree on the optimum.
                if (cold_obj - warm_obj).abs() > 1e-4 {
                    return Err(format!(
                        "pair {}: cold {:.6} != warm {:.6} — \
                         warm start produced a different optimum (solver bug)",
                        pair, cold_obj, warm_obj
                    ));
                }

                // Cross-validate against HiGHS when compiled in.
                #[cfg(feature = "highs")]
                {
                    let highs_obj = mwis::solve_highs(&graph, &valuation)?;
                    if (cold_obj - highs_obj).abs() > 1e-4 {
                        return Err(format!(
                            "pair {}: microlp {:.6} != HiGHS {:.6} (diff {:.3e})",
                            pair,
                            cold_obj,
                            highs_obj,
                            (cold_obj - highs_obj).abs()
                        ));
                    }
                }
            }
            Ok(())
        }));
    }
}
