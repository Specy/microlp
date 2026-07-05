//! Cases for the incremental Solution APIs: `add_constraint`, `fix_var` /
//! `unfix_var`, `add_gomory_cut` and `resume`.
//!
//! On pure-LP solutions these are incremental (dual simplex) and are checked
//! against external truths (certified optima, from-scratch solves, an ILP
//! enumeration oracle) — this behavior is correct today. On MILP solutions the
//! same APIs are buggy: they act on the branch & bound incumbent leaf (whose
//! solver still carries the branch path's bound-fixings) instead of the
//! original problem, so feasible edits report a false `Infeasible`. The
//! `*-milp` cases assert the correct behavior and therefore fail until the bug
//! is fixed; they are kept in the hard tier, out of the default (CI) run.

use super::lp_random::{certified_instance, CertifiedLp};
use super::netlib::data_path;
use super::{Case, Tier};
use crate::oracles;
use crate::rng::Rng;
use microlp::ComparisonOp::{Ge, Le};
use microlp::OptimizationDirection::{Maximize, Minimize};
use microlp::{MpsFile, OptimizationDirection, Problem, Solution, StopReason, Variable};
use std::io::BufReader;
use std::time::Duration;

const OBJ_TOL: f64 = 1e-6;

fn assert_close(what: &str, got: f64, want: f64) -> Result<(), String> {
    if (got - want).abs() > OBJ_TOL + OBJ_TOL * want.abs() {
        return Err(format!("{}: expected {}, got {}", what, want, got));
    }
    Ok(())
}

/// Build the certified LP with an added [0, 50] box (x* has entries <= 9, and
/// with y = 1 all rows are forced tight at any optimum, so the box changes
/// neither the optimum nor its uniqueness). The box keeps every prefix of the
/// constraint set bounded, which incremental construction needs.
fn boxed_certified(inst: &CertifiedLp) -> (Problem, Vec<Variable>) {
    let n = inst.x_star.len();
    let mut p = Problem::new(Maximize);
    let vars: Vec<_> = (0..n)
        .map(|j| p.add_var(inst.c[j] as f64, (0.0, 50.0)))
        .collect();
    (p, vars)
}

fn row_terms(inst: &CertifiedLp, vars: &[Variable], i: usize) -> Vec<(Variable, f64)> {
    (0..vars.len())
        .filter(|&j| inst.a[i][j] != 0)
        .map(|j| (vars[j], inst.a[i][j] as f64))
        .collect()
}

pub fn register(cases: &mut Vec<Case>) {
    add_constraint_lp(cases);
    fix_unfix_lp(cases);
    gomory(cases);
    resume(cases);
    milp_known_failures(cases);
}

// ---------------------------------------------------------------- LP: add_constraint

fn add_constraint_lp(cases: &mut Vec<Case>) {
    // Solve with half the rows, then feed the rest through
    // Solution::add_constraint one at a time. The final answer must be the
    // certified unique optimum — same truth a from-scratch solve is held to.
    for (i, &(n, seed)) in [(4usize, 401u64), (6, 402), (8, 403), (10, 404)]
        .iter()
        .enumerate()
    {
        let name = format!("incr/add-constraint-cert/n{:02}-s{:02}", n, i);
        cases.push(Case::custom(name, Tier::Quick, 20, move |budget| {
            let inst = certified_instance(n, seed);
            let (mut p, vars) = boxed_certified(&inst);
            let split = n / 2;
            for i in 0..split {
                let terms = row_terms(&inst, &vars, i);
                p.add_constraint(&terms, Le, inst.b[i] as f64);
            }
            p.set_time_limit(budget / 4);
            let mut sol = p.solve().map_err(|e| format!("prefix solve: {}", e))?;
            for i in split..n {
                let terms = row_terms(&inst, &vars, i);
                sol = sol
                    .add_constraint(&terms, Le, inst.b[i] as f64)
                    .map_err(|e| format!("add_constraint row {}: {}", i, e))?;
            }
            assert_close(
                "incremental objective",
                sol.objective(),
                inst.objective as f64,
            )?;
            for (j, &want) in inst.x_star.iter().enumerate() {
                assert_close(&format!("x{}", j), *sol.var_value_raw(vars[j]), want as f64)?;
            }
            Ok(())
        }));
    }

    // Hand-verified staircase: each added constraint moves the optimum to a
    // known value; the last one makes the problem infeasible.
    cases.push(Case::custom(
        "incr/add-constraint-steps",
        Tier::Quick,
        20,
        |budget| {
            let mut p = Problem::new(Maximize);
            let x = p.add_var(1.0, (0.0, 10.0));
            let y = p.add_var(1.0, (0.0, 10.0));
            p.set_time_limit(budget / 4);
            let sol = p.solve().map_err(|e| format!("base: {}", e))?;
            assert_close("base (bounds only)", sol.objective(), 20.0)?;

            let sol = sol
                .add_constraint(&[(x, 1.0), (y, 1.0)], Le, 15.0)
                .map_err(|e| format!("step 1: {}", e))?;
            assert_close("after x+y<=15", sol.objective(), 15.0)?;

            let sol = sol
                .add_constraint(&[(x, 1.0)], Le, 3.0)
                .map_err(|e| format!("step 2: {}", e))?;
            assert_close("after x<=3", sol.objective(), 13.0)?;

            let sol = sol
                .add_constraint(&[(y, 1.0), (x, -1.0)], Le, 0.0)
                .map_err(|e| format!("step 3: {}", e))?;
            assert_close("after y<=x", sol.objective(), 6.0)?;

            match sol.add_constraint(&[(x, 1.0), (y, 1.0)], Ge, 20.0) {
                Err(microlp::Error::Infeasible) => Ok(()),
                Err(e) => Err(format!("expected Infeasible on step 4, got error {}", e)),
                Ok(s) => Err(format!(
                    "expected Infeasible on step 4, got a solution with objective {}",
                    s.objective()
                )),
            }
        },
    ));
}

// ---------------------------------------------------------------- LP: fix/unfix

fn fix_unfix_lp(cases: &mut Vec<Case>) {
    for (i, &(n, seed)) in [(4usize, 501u64), (5, 502), (7, 503), (9, 504)]
        .iter()
        .enumerate()
    {
        let name = format!("incr/fix-unfix-cert/n{:02}-s{:02}", n, i);
        cases.push(Case::custom(name, Tier::Quick, 20, move |budget| {
            let inst = certified_instance(n, seed);

            // Fresh solve of the full problem (original, unboxed formulation).
            let build_full = |fixed: Option<(usize, f64)>| -> Result<Solution, String> {
                let nvars = inst.x_star.len();
                let mut p = Problem::new(Maximize);
                let vars: Vec<_> = (0..nvars)
                    .map(|j| {
                        let (lo, hi) = match fixed {
                            Some((fj, v)) if fj == j => (v, v),
                            _ => (0.0, f64::INFINITY),
                        };
                        p.add_var(inst.c[j] as f64, (lo, hi))
                    })
                    .collect();
                for i in 0..nvars {
                    let terms = row_terms(&inst, &vars, i);
                    p.add_constraint(&terms, Le, inst.b[i] as f64);
                }
                p.set_time_limit(budget / 8);
                p.solve().map_err(|e| format!("fresh solve: {}", e))
            };

            let base = build_full(None)?;
            assert_close("base objective", base.objective(), inst.objective as f64)?;

            // Fix a variable one unit BELOW its optimal value: with all-
            // nonnegative rows tight at x*, moving down is always feasible
            // (moving up may not be), and by uniqueness of the optimum the
            // result must be strictly worse and must match a from-scratch
            // solve with the variable's bounds pinned.
            let Some(j) = (0..inst.x_star.len()).find(|&j| inst.x_star[j] >= 1) else {
                // Degenerate draw (x* = 0): nothing to move down; the seeds
                // used here are chosen to avoid this.
                return Err("instance has x* = 0; pick a different seed".into());
            };
            let off = inst.x_star[j] as f64 - 1.0;
            let vars_handle: Vec<Variable> = base.iter().map(|(v, _)| v).collect();
            let fixed_incr = base
                .clone()
                .fix_var(vars_handle[j], off)
                .map_err(|e| format!("fix_var: {}", e))?;
            let fixed_fresh = build_full(Some((j, off)))?;
            assert_close(
                "fix_var vs fresh bounded solve",
                fixed_incr.objective(),
                fixed_fresh.objective(),
            )?;
            if fixed_incr.objective() > inst.objective as f64 - 1e-9 {
                return Err(format!(
                    "fixing x{} off-optimum should cost strictly more (unique optimum): \
                     got {} vs optimum {}",
                    j,
                    fixed_incr.objective(),
                    inst.objective
                ));
            }

            // Unfix restores the certified optimum; the bool reports whether
            // the variable was really fixed.
            let (restored, was_fixed) = fixed_incr.unfix_var(vars_handle[j]);
            if !was_fixed {
                return Err("unfix_var returned false for a fixed variable".into());
            }
            assert_close("after unfix", restored.objective(), inst.objective as f64)?;
            for (jj, &want) in inst.x_star.iter().enumerate() {
                assert_close(
                    &format!("restored x{}", jj),
                    *restored.var_value_raw(vars_handle[jj]),
                    want as f64,
                )?;
            }
            let (_, was_fixed_again) = restored.unfix_var(vars_handle[j]);
            if was_fixed_again {
                return Err("second unfix_var claimed the variable was still fixed".into());
            }
            Ok(())
        }));
    }
}

// ---------------------------------------------------------------- Gomory cuts

fn gomory(cases: &mut Vec<Case>) {
    // The crate's own textbook example, with every intermediate step asserted
    // (objective -1.5 -> -1.0 across two cuts).
    cases.push(Case::custom(
        "incr/gomory-textbook",
        Tier::Quick,
        20,
        |budget| {
            let mut p = Problem::new(Minimize);
            let v1 = p.add_var(0.0, (0.0, f64::INFINITY));
            let v2 = p.add_var(-1.0, (0.0, f64::INFINITY));
            p.add_constraint(&[(v1, 3.0), (v2, 2.0)], Le, 6.0);
            p.add_constraint(&[(v1, -3.0), (v2, 2.0)], Le, 0.0);
            p.set_time_limit(budget / 4);
            let sol = p.solve().map_err(|e| format!("base: {}", e))?;
            assert_close("relaxation objective", sol.objective(), -1.5)?;
            assert_close("relaxation v2", *sol.var_value_raw(v2), 1.5)?;

            let sol = sol
                .add_gomory_cut(v2)
                .map_err(|e| format!("first cut: {}", e))?;
            assert_close("after first cut", sol.objective(), -1.0)?;
            assert_close("v2 integral", *sol.var_value_raw(v2), 1.0)?;

            let sol = sol
                .add_gomory_cut(v1)
                .map_err(|e| format!("second cut: {}", e))?;
            assert_close("after second cut", sol.objective(), -1.0)?;
            assert_close("v1 integral", *sol.var_value_raw(v1), 1.0)?;
            Ok(())
        },
    ));

    // Oracle-checked cutting planes: on a small all-integer-data LP whose true
    // ILP optimum comes from box enumeration, repeatedly cut fractional basic
    // variables. A valid Gomory cut never removes an integer point, so the LP
    // objective must never drop below the ILP optimum (maximization); if the
    // LP becomes integral it must equal it exactly.
    for (case_idx, seed) in (0..4u64).enumerate() {
        let name = format!("incr/gomory-oracle/s{:02}", case_idx);
        cases.push(Case::custom(name, Tier::Quick, 30, move |budget| {
            let mut rng = Rng::new(0x6000 + seed);
            let n = rng.usize(3, 4);
            let bounds: Vec<(i64, i64)> = (0..n).map(|_| (0, rng.int(3, 5))).collect();
            // Nonnegative rows anchored at a box point keep the ILP feasible.
            let n_cons = rng.usize(2, 3);
            let mut constraints = vec![];
            for _ in 0..n_cons {
                let coeffs: Vec<i64> = (0..n).map(|_| rng.int(1, 5)).collect();
                let point: Vec<i64> = bounds
                    .iter()
                    .map(|&(lo, hi)| lo + rng.int(0, hi - lo))
                    .collect();
                let at_point: i64 = coeffs.iter().zip(&point).map(|(c, x)| c * x).sum();
                constraints.push((coeffs, -1i8, at_point + rng.int(0, 3)));
            }
            let objective: Vec<i64> = (0..n).map(|_| rng.int(1, 9)).collect();

            let ilp = oracles::TinyIlp {
                bounds: bounds.clone(),
                objective: objective.clone(),
                constraints: constraints.clone(),
                maximize: true,
            };
            let ilp_opt = ilp.brute_force().ok_or("oracle: unexpectedly infeasible")? as f64;

            let mut p = Problem::new(Maximize);
            let vars: Vec<_> = (0..n)
                .map(|j| {
                    p.add_var(
                        objective[j] as f64,
                        (bounds[j].0 as f64, bounds[j].1 as f64),
                    )
                })
                .collect();
            for (coeffs, _, rhs) in &constraints {
                let terms: Vec<_> = coeffs
                    .iter()
                    .enumerate()
                    .map(|(j, &c)| (vars[j], c as f64))
                    .collect();
                p.add_constraint(&terms, Le, *rhs as f64);
            }
            p.set_time_limit(budget / 4);
            let mut sol = p.solve().map_err(|e| format!("relaxation: {}", e))?;

            for round in 0..15 {
                if sol.objective() < ilp_opt - 1e-6 {
                    return Err(format!(
                        "cut {} removed integer points: LP objective {} fell below \
                         ILP optimum {}",
                        round,
                        sol.objective(),
                        ilp_opt
                    ));
                }
                // Find a fractional *basic* variable (value strictly between
                // its bounds; add_gomory_cut panics on non-basic variables).
                let mut cut_var = None;
                for (j, &v) in vars.iter().enumerate() {
                    let val = *sol.var_value_raw(v);
                    let fractional = (val - val.round()).abs() > 1e-6;
                    let interior =
                        val > bounds[j].0 as f64 + 1e-7 && val < bounds[j].1 as f64 - 1e-7;
                    if fractional && interior {
                        cut_var = Some(v);
                        break;
                    }
                }
                let Some(v) = cut_var else {
                    break;
                };
                sol = sol
                    .add_gomory_cut(v)
                    .map_err(|e| format!("cut {} failed: {}", round, e))?;
            }

            let all_integral = vars
                .iter()
                .all(|&v| (sol.var_value_raw(v) - sol.var_value_raw(v).round()).abs() <= 1e-6);
            if all_integral {
                assert_close(
                    "integral cut solution vs ILP oracle",
                    sol.objective(),
                    ilp_opt,
                )?;
            }
            Ok(())
        }));
    }
}

// ---------------------------------------------------------------- resume

fn resume(cases: &mut Vec<Case>) {
    // Duration::ZERO guarantees StopReason::Limit before any work; resume(None)
    // must then complete the solve and reach the known optimum. Exercises the
    // resume entry paths deterministically for LP and MILP alike.
    cases.push(Case::custom(
        "incr/resume-zero-lp",
        Tier::Quick,
        20,
        |_budget| {
            let inst = certified_instance(8, 601);
            let (mut p, vars) = boxed_certified(&inst);
            for i in 0..vars.len() {
                let terms = row_terms(&inst, &vars, i);
                p.add_constraint(&terms, Le, inst.b[i] as f64);
            }
            p.set_time_limit(Duration::ZERO);
            let sol = p.solve().map_err(|e| format!("limited solve: {}", e))?;
            if *sol.stop_reason() != StopReason::Limit {
                return Err("zero time limit did not produce StopReason::Limit".into());
            }
            let sol = sol.resume(None).map_err(|e| format!("resume: {}", e))?;
            if *sol.stop_reason() != StopReason::Finished {
                return Err("resume(None) did not finish".into());
            }
            assert_close("resumed objective", sol.objective(), inst.objective as f64)
        },
    ));

    cases.push(Case::custom(
        "incr/resume-zero-milp-knap",
        Tier::Quick,
        30,
        |_budget| {
            let mut rng = Rng::new(0x7000);
            let n = 18;
            let weights: Vec<i64> = (0..n).map(|_| rng.int(1, 30)).collect();
            let values: Vec<i64> = (0..n).map(|_| rng.int(1, 40)).collect();
            let capacity = weights.iter().sum::<i64>() / 2;
            let best = oracles::knapsack01(&values, &weights, capacity) as f64;

            let mut p = Problem::new(Maximize);
            let vars: Vec<_> = values.iter().map(|&v| p.add_binary_var(v as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            p.add_constraint(&terms, Le, capacity as f64);
            p.set_time_limit(Duration::ZERO);
            let sol = p.solve().map_err(|e| format!("limited solve: {}", e))?;
            if *sol.stop_reason() != StopReason::Limit {
                return Err("zero time limit did not produce StopReason::Limit".into());
            }
            let sol = sol.resume(None).map_err(|e| format!("resume: {}", e))?;
            if *sol.stop_reason() != StopReason::Finished {
                return Err("resume(None) did not finish".into());
            }
            assert_close("resumed knapsack vs DP", sol.objective(), best)
        },
    ));

    // LP interrupted mid-simplex by a real (nonzero) deadline, then finished
    // in slices. On a machine fast enough to finish inside the first slice the
    // case degrades to a plain optimum check — every outcome is asserted.
    cases.push(Case::custom(
        "incr/resume-slices-lp",
        Tier::Standard,
        60,
        |_budget| {
            let path = data_path("netlib", "israel.mps");
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
            let file = MpsFile::parse(
                BufReader::new(text.as_bytes()),
                OptimizationDirection::Minimize,
            )
            .map_err(|e| format!("parse: {}", e))?;
            let mut problem = file.problem;
            problem.set_time_limit(Duration::from_millis(1));
            let mut sol = problem.solve().map_err(|e| format!("solve: {}", e))?;
            let mut slices = 0;
            while *sol.stop_reason() == StopReason::Limit {
                slices += 1;
                if slices > 1000 {
                    return Err(
                        "LP resume made no progress after 1000 one-millisecond slices".into(),
                    );
                }
                sol = sol
                    .resume(Some(Duration::from_millis(1)))
                    .map_err(|e| format!("resume slice {}: {}", slices, e))?;
            }
            assert_close(
                "israel objective via slices",
                sol.objective(),
                -8.9664482186e5,
            )
        },
    ));
}

// ------------------------------------------------- MILP incremental edits

fn milp_known_failures(cases: &mut Vec<Case>) {
    // KNOWN FAILURE (solver bug): after a MILP solve, Solution::add_constraint
    // and Solution::fix_var act on the branch & bound incumbent leaf, whose
    // solver still carries the branch path's bound-rows and variable fixings,
    // instead of the original problem. Feasible edits therefore return a false
    // Infeasible (or a wrong/fractional value). These cases assert the correct
    // documented behavior and so fail until the bug is fixed; they live in the
    // hard tier and are excluded from the default (CI) run. add-constraint-mixed
    // happens to land on the right value for its particular instance but is
    // grouped here as it exercises the same buggy path.

    cases.push(Case::custom(
        "incr/add-constraint-milp",
        Tier::Hard,
        30,
        |_budget| {
            // Knapsack, then forbid the two chosen items from coexisting via
            // Solution::add_constraint. DP gives the exact constrained optimum.
            let values = [2i64, 2, 2];
            let weights = [3i64, 3, 3];
            let capacity = 8i64;
            let mut p = Problem::new(Maximize);
            let vars: Vec<_> = values.iter().map(|&v| p.add_binary_var(v as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            p.add_constraint(&terms, Le, capacity as f64);
            let sol = p.solve().map_err(|e| format!("base solve: {}", e))?;
            assert_close("base knapsack", sol.objective(), 4.0)?;

            // With x0 + x1 <= 1 the best is still 4 (e.g. items 0 and 2).
            let sol = sol
                .add_constraint(&[(vars[0], 1.0), (vars[1], 1.0)], Le, 1.0)
                .map_err(|e| format!("add_constraint on MILP solution: {}", e))?;
            assert_close("constrained knapsack", sol.objective(), 4.0)
        },
    ));

    cases.push(Case::custom(
        "incr/fix-var-milp",
        Tier::Hard,
        30,
        |_budget| {
            // max 5x+4y+3z, 4x+3y+2z <= 6, binary. Optimum 8 at (1,0,1).
            // Fixing y = 1 leaves (0,1,1) as the best integer point: 7.
            let mut p = Problem::new(Maximize);
            let x = p.add_binary_var(5.0);
            let y = p.add_binary_var(4.0);
            let z = p.add_binary_var(3.0);
            p.add_constraint(&[(x, 4.0), (y, 3.0), (z, 2.0)], Le, 6.0);
            let sol = p.solve().map_err(|e| format!("base solve: {}", e))?;
            assert_close("base objective", sol.objective(), 8.0)?;
            let sol = sol
                .fix_var(y, 1.0)
                .map_err(|e| format!("fix_var(y, 1) on MILP solution: {}", e))?;
            // The documented contract implies an integer-feasible re-solve.
            let vals = [
                *sol.var_value_raw(x),
                *sol.var_value_raw(y),
                *sol.var_value_raw(z),
            ];
            for (name, v) in ["x", "y", "z"].iter().zip(vals) {
                if (v - v.round()).abs() > 1e-5 {
                    return Err(format!("{} = {} is fractional after fix_var", name, v));
                }
            }
            assert_close("objective with y fixed to 1", sol.objective(), 7.0)
        },
    ));

    // A chain of edits on a knapsack, each step checked against a brute-force
    // oracle over all subsets (n = 12 -> 4096 subsets, exact integer math).
    // This is the TSP-style iterate-and-cut usage pattern.
    cases.push(Case::custom(
        "incr/add-constraint-milp-chain",
        Tier::Hard,
        60,
        |_budget| {
            let mut rng = Rng::new(0x8000);
            let n = 12usize;
            let weights: Vec<i64> = (0..n).map(|_| rng.int(2, 25)).collect();
            let values: Vec<i64> = (0..n).map(|_| rng.int(1, 30)).collect();
            let capacity = weights.iter().sum::<i64>() / 2;

            // conflicts[k] = (i, j) meaning x_i + x_j <= 1.
            let brute = |conflicts: &[(usize, usize)]| -> i64 {
                let mut best = 0i64;
                for mask in 0u32..(1 << n) {
                    let w: i64 = (0..n)
                        .filter(|&i| mask & (1 << i) != 0)
                        .map(|i| weights[i])
                        .sum();
                    if w > capacity {
                        continue;
                    }
                    if conflicts
                        .iter()
                        .any(|&(i, j)| mask & (1 << i) != 0 && mask & (1 << j) != 0)
                    {
                        continue;
                    }
                    let v: i64 = (0..n)
                        .filter(|&i| mask & (1 << i) != 0)
                        .map(|i| values[i])
                        .sum();
                    best = best.max(v);
                }
                best
            };

            let mut p = Problem::new(Maximize);
            let vars: Vec<_> = values.iter().map(|&v| p.add_binary_var(v as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            p.add_constraint(&terms, Le, capacity as f64);
            let mut sol = p.solve().map_err(|e| format!("base solve: {}", e))?;
            assert_close("base", sol.objective(), brute(&[]) as f64)?;

            let mut conflicts: Vec<(usize, usize)> = vec![];
            for &(i, j) in &[(0usize, 1usize), (2, 3), (4, 5)] {
                conflicts.push((i, j));
                sol = sol
                    .add_constraint(&[(vars[i], 1.0), (vars[j], 1.0)], Le, 1.0)
                    .map_err(|e| format!("adding conflict {:?}: {}", (i, j), e))?;
                let want = brute(&conflicts) as f64;
                assert_close(
                    &format!("after conflict {:?}", (i, j)),
                    sol.objective(),
                    want,
                )?;
                // The point itself must respect every conflict so far.
                for &(a, b) in &conflicts {
                    let sum = sol.var_value_raw(vars[a]) + sol.var_value_raw(vars[b]);
                    if sum > 1.0 + 1e-6 {
                        return Err(format!(
                            "solution violates added conflict {:?}: sum = {}",
                            (a, b),
                            sum
                        ));
                    }
                }
            }
            Ok(())
        },
    ));

    // fix -> check against DP with the item forced out -> unfix -> back to
    // the unconstrained DP optimum; the unfix bool contract holds on MILP.
    cases.push(Case::custom(
        "incr/fix-unfix-milp-roundtrip",
        Tier::Hard,
        60,
        |_budget| {
            let mut rng = Rng::new(0x8100);
            let n = 14usize;
            let weights: Vec<i64> = (0..n).map(|_| rng.int(2, 25)).collect();
            let values: Vec<i64> = (0..n).map(|_| rng.int(1, 30)).collect();
            let capacity = weights.iter().sum::<i64>() / 2;
            let base_best = oracles::knapsack01(&values, &weights, capacity) as f64;

            // DP with item 0 forced to stay out.
            let without0 = oracles::knapsack01(&values[1..], &weights[1..], capacity) as f64;

            let mut p = Problem::new(Maximize);
            let vars: Vec<_> = values.iter().map(|&v| p.add_binary_var(v as f64)).collect();
            let terms: Vec<_> = vars
                .iter()
                .zip(&weights)
                .map(|(&v, &w)| (v, w as f64))
                .collect();
            p.add_constraint(&terms, Le, capacity as f64);
            let sol = p.solve().map_err(|e| format!("base solve: {}", e))?;
            assert_close("base vs DP", sol.objective(), base_best)?;

            let sol = sol
                .fix_var(vars[0], 0.0)
                .map_err(|e| format!("fix_var(x0, 0): {}", e))?;
            assert_close("fixed-out vs DP", sol.objective(), without0)?;

            let (sol, was_fixed) = sol.unfix_var(vars[0]);
            if !was_fixed {
                return Err("unfix_var returned false for a fixed variable".into());
            }
            assert_close("after unfix vs DP", sol.objective(), base_best)?;
            let (_, was_fixed_again) = sol.unfix_var(vars[0]);
            if was_fixed_again {
                return Err("second unfix_var claimed the variable was still fixed".into());
            }
            Ok(())
        },
    ));

    // Fixing an integer variable to a fractional value has no integer
    // solutions by definition.
    cases.push(Case::custom(
        "incr/fix-var-milp-fractional",
        Tier::Hard,
        15,
        |_budget| {
            let mut p = Problem::new(Maximize);
            let x = p.add_binary_var(1.0);
            let y = p.add_binary_var(1.0);
            p.add_constraint(&[(x, 1.0), (y, 1.0)], Le, 2.0);
            let sol = p.solve().map_err(|e| format!("base solve: {}", e))?;
            match sol.fix_var(x, 0.5) {
                Err(microlp::Error::Infeasible) => Ok(()),
                Err(e) => Err(format!("expected Infeasible, got error {}", e)),
                Ok(s) => Err(format!(
                    "expected Infeasible fixing a binary to 0.5, got objective {}",
                    s.objective()
                )),
            }
        },
    ));

    // Mixed problem: an edit touching a continuous variable of a MILP must
    // re-solve to the hand-computed optimum (x continuous, z integer).
    cases.push(Case::custom(
        "incr/add-constraint-mixed",
        Tier::Hard,
        30,
        |_budget| {
            // The suite's mixed-textbook case: max 50x+40y+45z, x>=2 cont.,
            // y in [0,7] cont., z >= 0 int; 3x+2y+z<=20, 2x+y+3z<=15.
            // Optimum 405 at (2, 6.5, 1). Adding y <= 5 moves it to
            // (3, 5, 1): z=1 leaves 3x+2y<=19, 2x+y<=12; at y=5 the first
            // gives x<=3, so 150+200+45 = 395 (z=0 gives 366.7, z=2 390).
            let mut p = Problem::new(Maximize);
            let x = p.add_var(50.0, (2.0, f64::INFINITY));
            let y = p.add_var(40.0, (0.0, 7.0));
            let z = p.add_integer_var(45.0, (0, i32::MAX));
            p.add_constraint(&[(x, 3.0), (y, 2.0), (z, 1.0)], Le, 20.0);
            p.add_constraint(&[(x, 2.0), (y, 1.0), (z, 3.0)], Le, 15.0);
            let sol = p.solve().map_err(|e| format!("base solve: {}", e))?;
            assert_close("base objective", sol.objective(), 405.0)?;

            let sol = sol
                .add_constraint(&[(y, 1.0)], Le, 5.0)
                .map_err(|e| format!("adding y <= 5: {}", e))?;
            assert_close("objective with y <= 5", sol.objective(), 395.0)?;
            let zv = *sol.var_value_raw(z);
            if (zv - zv.round()).abs() > 1e-5 {
                return Err(format!("z = {} is fractional after the edit", zv));
            }
            Ok(())
        },
    ));

    // KNOWN FAILURE (time-limit-interrupt bug): interrupting B&B mid-search
    // currently panics, so resuming from a genuine mid-search Limit cannot be
    // exercised yet. numbers chosen so stein27 (~0.9 s to solve) reliably
    // overruns a 50 ms first slice.
    cases.push(Case::custom(
        "incr/resume-midway-milp",
        Tier::Hard,
        120,
        |_budget| {
            let parsed = crate::mps_milp::parse(
                &std::fs::read_to_string(data_path("miplib3", "stein27.mps"))
                    .map_err(|e| format!("read stein27: {}", e))?,
                OptimizationDirection::Minimize,
                false,
            )?;
            let mut problem = parsed.problem;
            problem.set_time_limit(Duration::from_millis(50));
            let mut sol = problem.solve().map_err(|e| format!("solve: {}", e))?;
            let mut slices = 0;
            while *sol.stop_reason() == StopReason::Limit {
                slices += 1;
                if slices > 600 {
                    return Err("MILP resume made no progress after 600 slices".into());
                }
                sol = sol
                    .resume(Some(Duration::from_millis(100)))
                    .map_err(|e| format!("resume slice {}: {}", slices, e))?;
            }
            assert_close("stein27 via interrupted+resumed B&B", sol.objective(), 18.0)
        },
    ));
}
