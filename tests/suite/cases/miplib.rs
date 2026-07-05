//! MIPLIB 3 benchmark cases. Every instance yields two cases: the LP
//! relaxation checked against the published "LP SOLN" value (this also
//! validates the suite's MPS reading, in particular the integer-bound
//! conventions) and the integer solve checked against "INT SOLN".
//! Values from the official miplib3.cat; see data/README.md.

use super::netlib::data_path;
use super::{Case, Tier};
use crate::model::{Expected, Tol};
use crate::mps_milp;
use microlp::OptimizationDirection;

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
        // Heavier: opt-in via --hard.
        mk("stein45", 30.0, 22.0, T2DP, EXACT, Tier::Hard, 600),
        // The four below don't finish inside their budgets on a laptop, and
        // because of the time-limit-interrupt solver bug (see
        // milp/time-limit-interrupt) they currently show PANIC instead of
        // TIMEOUT when the deadline fires mid-B&B.
        mk("mod008", 307.0, 290.93, T2DP, EXACT, Tier::Hard, 600),
        mk("enigma", 0.0, 0.0, EXACT, EXACT, Tier::Hard, 600),
        // KNOWN FAILURES (solver bug): when the time limit fires mid-B&B on
        // these instances the solver panics with "assertion failed:
        // self.is_primal_feasible" (solver.rs:787) instead of returning
        // StopReason::Limit; with no deadline they may just run very long.
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
