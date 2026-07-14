//! MILPBench xhard cases: the families beyond microlp's current ceiling.
//!
//! Of the seven classic MILPBench "easy" families, only CFL is within
//! microlp's practical reach (wide, integral LP relaxation — see
//! `cases/milpbench.rs`). The other six — MIS, MVC, SC, CAT, MIKS, BIP —
//! are 20k–60k *constraints*; microlp typically does not reach an integer
//! incumbent (often not even the root LP optimum) within its budget. One
//! instance per family is vendored under `data/xhard/milpbench/` with an
//! **externally certified verdict** from HiGHS — a zero-gap proven optimum,
//! a proven `[lo, hi]` envelope, or a proven unboundedness; see
//! [`Certificate`] and `data/xhard/milpbench/README.md` for the per-instance
//! details.
//!
//! Contract of an xhard case (10-minute default budget, per the tier):
//!
//! * the solve must end in a **clean status** within budget — an `Err` or a
//!   panic at this scale is a solver bug and fails the case;
//! * `Optimal` — the incumbent must validate against the shadow model and
//!   the objective must match the certificate (equal the certified optimum,
//!   or land inside the certified envelope);
//! * `Feasible` — the incumbent must validate and must never be **better**
//!   than the certified optimum / proven bound (direction-aware: that would
//!   mean solver unsoundness or a wrong certificate);
//! * `Interrupted` — pass: no incumbent was found inside the configured budget.
//!
//! As the solver improves, these cases tighten automatically: the moment a
//! run starts returning incumbents (or optima), the certified value starts
//! biting. The registration is data-driven — drop an `.lp` under
//! `data/xhard/milpbench/<family>/`, certify it with a reference solver, and
//! append a row to `CERTIFIED`.

use super::{locate, read_instance, Case};
use crate::lp_format;
use crate::model::Tol;
use microlp::{OptimizationDirection, Status};

struct XhardInstance {
    /// Family sub-directory under `data/xhard/milpbench/`.
    family: &'static str,
    /// File name, e.g. `IS_easy_instance_11.lp` (upstream name, verbatim).
    file: &'static str,
    /// What the reference solver proved about this instance.
    certificate: Certificate,
    /// Per-instance budget; the xhard tier default is 600 s (10 minutes).
    budget_secs: u64,
}

enum Certificate {
    /// Externally proven optimal objective (HiGHS, zero-gap proof).
    Optimum(f64),
    /// The reference solver could not close the gap in bounded local time,
    /// but *proved* that the true optimum lies in `[lo, hi]`: one side is
    /// HiGHS's proven dual bound, the other its best feasible incumbent.
    /// Weaker than an optimum but with real teeth — a microlp "proven
    /// optimum" outside the envelope, or a feasible incumbent beating the
    /// proven bound side, is unsound.
    Envelope { lo: f64, hi: f64 },
    /// Externally proven **unbounded**: the LP relaxation is unbounded along
    /// the continuous variables and the instance is integer-feasible (a
    /// zeroed-objective MILP solve returns Optimal), so the MILP itself is
    /// unbounded. MILPBench's MIKS_easy generator emits such instances — an
    /// empty `Bounds` section leaves the non-binary variables continuous on
    /// `[0, +inf)` with positive maximization coefficients. A solver must
    /// never claim a finite proven optimum here.
    Unbounded,
}

/// Objective comparison tolerance: certified optima here are O(1e3)–O(1e6),
/// so a small absolute plus relative slack covers cross-platform noise.
const XHARD_TOL: Tol = Tol {
    abs: 1e-3,
    rel: 1e-6,
};

/// One instance per family, certified by HiGHS (see the data README for the
/// exact solver version, wall time and dual bound of each certification).
const CERTIFIED: &[XhardInstance] = &[
    XhardInstance {
        family: "BIP_easy",
        file: "BIP_easy_instance_10.lp",
        certificate: Certificate::Optimum(80800.0),
        budget_secs: 600,
    },
    XhardInstance {
        family: "MIKS_easy",
        file: "MIKS_easy_instance_26.lp",
        certificate: Certificate::Unbounded,
        budget_secs: 600,
    },
    // The four graph-structured families resisted zero-gap certification
    // (HiGHS, 2 threads, 3600 s: gaps 7.7%–869%); their envelopes below are
    // HiGHS-proven. Maximization: lo = best incumbent, hi = proven dual
    // bound. Minimization: lo = proven dual bound, hi = best incumbent.
    XhardInstance {
        family: "MIS_easy",
        file: "IS_easy_instance_11.lp",
        certificate: Certificate::Envelope {
            lo: 523.6264397361941,
            hi: 5075.512443744508,
        },
        budget_secs: 600,
    },
    XhardInstance {
        family: "MVC_easy",
        file: "MVC_easy_instance_19.lp",
        certificate: Certificate::Envelope {
            lo: 4937.689925436303,
            hi: 9570.635205426428,
        },
        budget_secs: 600,
    },
    XhardInstance {
        family: "CAT_easy",
        file: "CAT_easy_instance_25.lp",
        certificate: Certificate::Envelope {
            lo: 979.6483854074107,
            hi: 4939.24371571862,
        },
        budget_secs: 600,
    },
    XhardInstance {
        family: "SC_easy",
        file: "SC_easy_instance_6.lp",
        certificate: Certificate::Envelope {
            lo: 3119.5563850909934,
            hi: 3379.6673668697686,
        },
        budget_secs: 600,
    },
];

pub fn register(cases: &mut Vec<Case>) {
    for inst in CERTIFIED {
        let family = inst.family;
        let file = inst.file;
        let certificate = &inst.certificate;
        let name = format!("milpbench/{}/{}", family, file.trim_end_matches(".lp"));
        let (path, tier) = locate(&format!("milpbench/{}", family), file);
        cases.push(Case::custom(name, tier, inst.budget_secs, move |budget| {
            let text = read_instance(&path)?;
            let parsed = lp_format::parse(&text, false)
                .map_err(|e| format!("LP reader failed on {}: {}", file, e))?;
            let mut problem = parsed.problem;
            problem.set_time_limit(budget);
            let result = problem.solve();
            match certificate {
                Certificate::Optimum(optimum) => {
                    let optimum = *optimum;
                    // A solve *error* (not a clean status) at this scale is a
                    // solver bug; surface it loudly.
                    let sol = result
                        .map_err(|e| format!("solve errored (solver bug at scale): {}", e))?;
                    match sol.status() {
                        Status::Optimal => {
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
                            if !XHARD_TOL.matches(sol.objective(), optimum) {
                                return Err(format!(
                                    "solver claims optimal objective {} but the certified \
                                     optimum is {} (diff {:.3e})",
                                    sol.objective(),
                                    optimum,
                                    (sol.objective() - optimum).abs()
                                ));
                            }
                            crate::LAST_SOLVE
                                .with(|slot| *slot.borrow_mut() = Some((sol.objective(), optimum)));
                        }
                        Status::Feasible => {
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
                            let slack = XHARD_TOL.slack(optimum);
                            let better_than_optimal = match parsed.direction {
                                OptimizationDirection::Minimize => {
                                    sol.objective() < optimum - slack
                                }
                                OptimizationDirection::Maximize => {
                                    sol.objective() > optimum + slack
                                }
                            };
                            if better_than_optimal {
                                return Err(format!(
                                    "feasible incumbent {} is better than the certified \
                                     optimum {} — solver unsoundness (or a wrong certificate)",
                                    sol.objective(),
                                    optimum
                                ));
                            }
                            crate::LAST_SOLVE
                                .with(|slot| *slot.borrow_mut() = Some((sol.objective(), optimum)));
                        }
                        // No incumbent inside the budget: a clean interrupt
                        // satisfies the xhard limit-handling contract.
                        Status::Interrupted => {}
                    }
                }
                Certificate::Envelope { lo, hi } => {
                    let (lo, hi) = (*lo, *hi);
                    let slack = XHARD_TOL.slack(lo.abs().max(hi.abs()));
                    let sol = result
                        .map_err(|e| format!("solve errored (solver bug at scale): {}", e))?;
                    match sol.status() {
                        // A claimed proven optimum must land inside the
                        // externally proven envelope.
                        Status::Optimal => {
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
                            if sol.objective() < lo - slack || sol.objective() > hi + slack {
                                return Err(format!(
                                    "solver claims optimal objective {} outside the \
                                     externally proven envelope [{}, {}]",
                                    sol.objective(),
                                    lo,
                                    hi
                                ));
                            }
                        }
                        // A feasible incumbent can be arbitrarily bad, but
                        // never better than the proven dual bound.
                        Status::Feasible => {
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
                            let beats_proven_bound = match parsed.direction {
                                OptimizationDirection::Maximize => sol.objective() > hi + slack,
                                OptimizationDirection::Minimize => sol.objective() < lo - slack,
                            };
                            if beats_proven_bound {
                                return Err(format!(
                                    "feasible incumbent {} beats the externally proven \
                                     bound (envelope [{}, {}]) — solver unsoundness (or a \
                                     wrong certificate)",
                                    sol.objective(),
                                    lo,
                                    hi
                                ));
                            }
                        }
                        Status::Interrupted => {}
                    }
                }
                Certificate::Unbounded => match result {
                    // Detecting the unboundedness is the correct full answer.
                    Err(microlp::Error::Unbounded) => {}
                    // The instance is certified integer-feasible, so
                    // `Infeasible` (or any other error) is a wrong answer.
                    Err(e) => {
                        return Err(format!(
                            "certified-unbounded instance, expected Unbounded / Feasible / \
                             Interrupted but got error: {}",
                            e
                        ))
                    }
                    Ok(sol) => match sol.status() {
                        // A finite "proven optimum" on an unbounded problem is
                        // unsound, full stop.
                        Status::Optimal => {
                            return Err(format!(
                                "solver claims a finite proven optimum {} on a \
                                 certified-unbounded instance — unsound",
                                sol.objective()
                            ))
                        }
                        // An honest, validated incumbent with no optimality
                        // claim is fine (B&B may find one before noticing the
                        // unbounded direction).
                        Status::Feasible => {
                            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
                        }
                        Status::Interrupted => {}
                    },
                },
            }
            Ok(())
        }));
    }
}
