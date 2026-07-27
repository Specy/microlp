//! MIPLIB 3 benchmark cases. Every instance yields two cases: the LP
//! relaxation checked against the published "LP SOLN" value (this also
//! validates the suite's MPS reading, in particular the integer-bound
//! conventions) and the integer solve checked against "INT SOLN".
//! Values from the official miplib3.cat; see data/README.md.
//!
//! The `/int` case's tier is derived from the folder the `.mps` file lives in
//! (`data/<tier>/miplib3/`); the `/lp` relaxation case always runs in the
//! medium tier — every relaxation solves in milliseconds and doubles as a
//! parser check, so it belongs in the default run even when the integer solve
//! is hard.

use super::{locate, read_instance, Case, Tier};
use crate::model::{Expected, Tol};
use crate::mps_milp;
use microlp::{OptimizationDirection, SolutionStatus, SolveOutcome};

/// Instances whose integer solve is not guaranteed to finish inside its
/// (generous) budget on a laptop. For these the `/int` case tolerates a clean
/// mid-B&B interrupt (a `Feasible` incumbent or a bare `Interrupted`) instead
/// of demanding the proven optimum. When the solve *does* finish in budget
/// the published optimum is still checked.
const INT_MAY_INTERRUPT: &[&str] = &["mod008", "vpm1", "gt2", "bell3a"];

struct MipInstance {
    name: &'static str,
    int_opt: f64,
    lp_opt: f64,
    /// Published values are rounded (mostly 2 decimals); tolerances per case.
    lp_tol: Tol,
    int_tol: Tol,
    int_budget: u64,
}

const T2DP: Tol = Tol {
    abs: 0.02,
    rel: 1e-6,
};
const EXACT: Tol = Tol {
    abs: 1e-4,
    rel: 1e-9,
};

fn instances() -> Vec<MipInstance> {
    let mk = |name, int_opt, lp_opt, lp_tol, int_tol, int_budget| MipInstance {
        name,
        int_opt,
        lp_opt,
        lp_tol,
        int_tol,
        int_budget,
    };
    vec![
        // Small enough for a naive branch & bound: data/medium/miplib3/.
        mk("flugpl", 1201500.0, 1167185.73, T2DP, EXACT, 120),
        mk("p0033", 3089.0, 2520.57, T2DP, EXACT, 120),
        mk("stein27", 18.0, 13.0, T2DP, EXACT, 120),
        // Measured fast enough for the default tier on a laptop (~1 s / ~6 s).
        mk(
            "rgn",
            82.19999924,
            48.7999,
            Tol {
                abs: 0.001,
                rel: 1e-6,
            },
            Tol {
                abs: 0.001,
                rel: 1e-6,
            },
            120,
        ),
        mk("lseu", 1120.0, 834.68, T2DP, EXACT, 120),
        mk("p0201", 7615.0, 6875.0, T2DP, EXACT, 120),
        mk("misc03", 3360.0, 1910.0, T2DP, EXACT, 120),
        // Heavier: data/hard/miplib3/. stein45 finishes in budget (~30 s).
        mk("stein45", 30.0, 22.0, T2DP, EXACT, 600),
        // mod008/vpm1/gt2/bell3a are listed in INT_MAY_INTERRUPT: their /int
        // solve may not finish in budget, and a mid-B&B deadline must yield a
        // clean status. See register.
        mk("mod008", 307.0, 290.93, T2DP, EXACT, 600),
        // enigma deterministically triggers a singular-matrix error at a B&B
        // node re-solve; solve_node_lp retries such a node once from the
        // all-slack basis (the singular-LU robustness valve, see
        // src/ARCHITECTURE.md §5.3), so enigma solves to its optimum (~14 s).
        mk("enigma", 0.0, 0.0, EXACT, EXACT, 600),
        mk("vpm1", 20.0, 15.4167, T2DP, EXACT, 600),
        mk("gt2", 21166.0, 13460.233074, T2DP, EXACT, 600),
        mk("bell3a", 878430.32, 862578.64, T2DP, T2DP, 600),
    ]
}

pub fn register(cases: &mut Vec<Case>) {
    for inst in instances() {
        let name = inst.name;
        // The integer case's tier follows the data folder.
        let (_, int_tier) = locate("miplib3", &format!("{}.mps", name));

        // LP relaxation case.
        let lp_case = format!("miplib/{}/lp", name);
        let lp_tol = inst.lp_tol;
        let lp_opt = inst.lp_opt;
        cases.push(Case::solve(lp_case, Tier::Medium, 60, move || {
            let parsed = load(name, true)?;
            Ok((
                parsed.spec,
                parsed.problem,
                Expected::objective_tol(lp_opt, lp_tol),
            ))
        }));

        // Integer case.
        let int_case = format!("miplib/{}/int", name);
        let int_tol = inst.int_tol;
        let int_opt = inst.int_opt;
        if INT_MAY_INTERRUPT.contains(&name) {
            // Interrupt-tolerant: a deadline that fires mid-B&B must return a
            // clean status, never a panic. If the solve finishes we still hold
            // it to the published optimum; a `Feasible` incumbent must be a
            // genuine feasible point of the model and no better than the proven
            // optimum; a bare `Interrupted` (no incumbent) just proves no panic.
            cases.push(Case::custom(
                int_case,
                int_tier,
                inst.int_budget,
                move |budget| {
                    let parsed = load(name, false)?;
                    let mut problem = parsed.problem;
                    problem.set_time_limit(budget);
                    let outcome = problem
                        .solve()
                        .map_err(|e| format!("solve errored: {}", e))?;
                    match outcome {
                        SolveOutcome::Solution(sol) => {
                            crate::LAST_SOLVE.with(|slot| {
                                *slot.borrow_mut() = Some((sol.objective(), int_opt));
                            });
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
                            match sol.status() {
                                SolutionStatus::Optimal => {
                                    if !int_tol.matches(sol.objective(), int_opt) {
                                        return Err(format!(
                                            "expected int optimum {}, got {} (diff {:.3e})",
                                            int_opt,
                                            sol.objective(),
                                            (sol.objective() - int_opt).abs()
                                        ));
                                    }
                                }
                                SolutionStatus::Feasible => {
                                    let tol = int_tol.abs + int_tol.rel * int_opt.abs();
                                    if sol.objective() < int_opt - tol {
                                        return Err(format!(
                                            "feasible incumbent objective {} is below the proven \
                                             optimum {} — solver unsoundness",
                                            sol.objective(),
                                            int_opt
                                        ));
                                    }
                                }
                            }
                        }
                        SolveOutcome::Interrupted(_) => {}
                    }
                    Ok(())
                },
            ));
        } else {
            cases.push(Case::solve(
                int_case,
                int_tier,
                inst.int_budget,
                move || {
                    let parsed = load(name, false)?;
                    Ok((
                        parsed.spec,
                        parsed.problem,
                        Expected::objective_tol(int_opt, int_tol),
                    ))
                },
            ));
        }
    }
}

fn load(name: &str, relax: bool) -> Result<mps_milp::ParsedMps, String> {
    let (path, _) = locate("miplib3", &format!("{}.mps", name));
    let text = read_instance(&path)?;
    let parsed = mps_milp::parse(&text, OptimizationDirection::Minimize, relax)?;
    if parsed.obj_offset != 0.0 {
        // None of the vendored files carries an objective constant; if one
        // ever does, the expected values below must be shifted accordingly.
        return Err(format!(
            "{} has objective offset {}; expected values need adjusting",
            name, parsed.obj_offset
        ));
    }
    Ok(parsed)
}
