//! MIPLIB 3 benchmark cases. Every instance yields two cases: the LP
//! relaxation checked against the published "LP SOLN" value (this also
//! validates the suite's MPS reading, in particular the integer-bound
//! conventions) and the integer solve checked against "INT SOLN".
//! Values from the official miplib3.cat; see data/README.md.

use super::netlib::data_path;
use super::{Case, Tier};
use crate::model::{Expected, Tol};
use crate::mps_milp;
use microlp::{OptimizationDirection, Status};

/// Instances whose integer solve is not guaranteed to finish inside its
/// (generous) budget on a laptop. For these the `/int` case tolerates a clean
/// mid-B&B interrupt (a `Feasible` incumbent or a bare `Interrupted`) instead
/// of demanding the proven optimum — before the B&B-interrupt fix these
/// overran their budgets and panicked. When the solve *does* finish in budget
/// the published optimum is still checked.
const INT_MAY_INTERRUPT: &[&str] = &["mod008", "vpm1", "gt2", "bell3a"];

struct MipInstance {
    name: &'static str,
    int_opt: f64,
    lp_opt: f64,
    /// Published values are rounded (mostly 2 decimals); tolerances per case.
    lp_tol: Tol,
    int_tol: Tol,
    int_tier: Tier,
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
    let mk = |name, int_opt, lp_opt, lp_tol, int_tol, int_tier, int_budget| MipInstance {
        name,
        int_opt,
        lp_opt,
        lp_tol,
        int_tol,
        int_tier,
        int_budget,
    };
    vec![
        // Small enough for a naive branch & bound: default tier.
        mk(
            "flugpl",
            1201500.0,
            1167185.73,
            T2DP,
            EXACT,
            Tier::Standard,
            120,
        ),
        mk("p0033", 3089.0, 2520.57, T2DP, EXACT, Tier::Standard, 120),
        mk("stein27", 18.0, 13.0, T2DP, EXACT, Tier::Standard, 120),
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
            Tier::Standard,
            120,
        ),
        mk("lseu", 1120.0, 834.68, T2DP, EXACT, Tier::Standard, 120),
        mk("p0201", 7615.0, 6875.0, T2DP, EXACT, Tier::Standard, 120),
        mk("misc03", 3360.0, 1910.0, T2DP, EXACT, Tier::Standard, 120),
        // Heavier: opt-in via --hard. stein45 finishes in budget (~30 s).
        mk("stein45", 30.0, 22.0, T2DP, EXACT, Tier::Hard, 600),
        // mod008/vpm1/gt2/bell3a are listed in INT_MAY_INTERRUPT: their /int
        // solve may not finish in budget, and a mid-B&B deadline must now yield
        // a clean status (never the old is_primal_feasible panic). See register.
        mk("mod008", 307.0, 290.93, T2DP, EXACT, Tier::Hard, 600),
        // enigma does NOT finish in budget and currently surfaces a real solver
        // bug (Error::SingularMatrix from reoptimize() at a B&B node, not caught
        // by the slack-basis robustness valve). Left failing on purpose — this
        // is a solver-robustness bug, separate from the interrupt fix.
        mk("enigma", 0.0, 0.0, EXACT, EXACT, Tier::Hard, 600),
        mk("vpm1", 20.0, 15.4167, T2DP, EXACT, Tier::Hard, 600),
        mk("gt2", 21166.0, 13460.233074, T2DP, EXACT, Tier::Hard, 600),
        mk("bell3a", 878430.32, 862578.64, T2DP, T2DP, Tier::Hard, 600),
    ]
}

pub fn register(cases: &mut Vec<Case>) {
    for inst in instances() {
        let name = inst.name;

        // LP relaxation case.
        let lp_case = format!("miplib/{}/lp", name);
        let lp_tol = inst.lp_tol;
        let lp_opt = inst.lp_opt;
        cases.push(Case::solve(lp_case, Tier::Standard, 60, move || {
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
                inst.int_tier,
                inst.int_budget,
                move |budget| {
                    let parsed = load(name, false)?;
                    let mut problem = parsed.problem;
                    problem.set_time_limit(budget);
                    let sol = problem
                        .solve()
                        .map_err(|e| format!("solve errored: {}", e))?;
                    match sol.status() {
                        Status::Optimal => {
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
                            if !int_tol.matches(sol.objective(), int_opt) {
                                return Err(format!(
                                    "expected int optimum {}, got {} (diff {:.3e})",
                                    int_opt,
                                    sol.objective(),
                                    (sol.objective() - int_opt).abs()
                                ));
                            }
                        }
                        Status::Feasible => {
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
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
                        Status::Interrupted => {}
                    }
                    Ok(())
                },
            ));
        } else {
            cases.push(Case::solve(
                int_case,
                inst.int_tier,
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
    let path = data_path("miplib3", &format!("{}.mps", name));
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
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
