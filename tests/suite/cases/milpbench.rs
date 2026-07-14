//! MILPBench solved-to-optimality cases + LP reader validation.
//!
//! Two groups live here:
//!
//! 1. **Reader validation** (`Tier::Easy`, always runs). Small hand-written
//!    CPLEX-LP strings with a known optimum, solved through the public `Problem`
//!    API via `crate::lp_format`, plus loud-error probes that assert the reader
//!    rejects unsupported syntax rather than misparsing it. These guard the
//!    reader in the default CI run.
//!
//! 2. **Vendored MILPBench instances that microlp proves optimal** (tier
//!    derived from the data folder; the CFL instances live in `medium/`).
//!    See `data/medium/milpbench/README.md` for provenance. The MILPBench
//!    families microlp *cannot* finish live in the xhard tier instead, with
//!    externally certified optima — see `cases/milpbench_xhard.rs`.
//!
//! Selection bar for this file: an instance is vendored here only if the solve
//! returns `Status::Optimal` with `gap() == Some(0)` inside budget. MILPBench
//! ships no certified optima for these families, so the case asserts
//! **solver-proven** optimality — proven-optimal status, a zero gap, and an
//! independent shadow-model validation of the returned point (feasibility,
//! bounds, integrality, objective consistency). The pinned objective is
//! externally certified by a HiGHS zero-gap proof (see the `optimum` field for
//! the certification details and the structural argument that backs it).
//!
//! Only the CFL family cleared this bar: its easy instances are *wide* (~800
//! constraints over ~160 400 variables) with an integral LP relaxation.

use super::{locate, read_instance, Case, Tier};
use crate::lp_format;
use crate::model::{Expected, Tol};
use microlp::Status;

pub fn register(cases: &mut Vec<Case>) {
    register_reader_tests(cases);
    register_bench_instances(cases);
}

// ---------------------------------------------------------------------------
// 1. Reader validation cases (Easy tier).
// ---------------------------------------------------------------------------

/// A hand-written LP whose optimum is known by construction. `relax` mirrors the
/// reader's integer-relaxation switch (unused by these cases; all solve as
/// stated).
struct ReaderCase {
    name: &'static str,
    lp: &'static str,
    expected: f64,
}

fn reader_cases() -> Vec<ReaderCase> {
    vec![
        // Binary knapsack: max 3x+2y+4z s.t. x+y+z<=2. Best pair is {x,z}=7.
        ReaderCase {
            name: "lp-reader/binary-knapsack",
            lp: "\\ tiny binary knapsack\n\
                 Maximize\n  3 x + 2 y + 4 z\n\
                 Subject To\n  c1: x + y + z <= 2\n\
                 Binaries\n  x y z\n\
                 End\n",
            expected: 7.0,
        },
        // Continuous LP with bounds and a >= constraint. On x+2y=4 the objective
        // x+y = x/2 + 2 is minimized at the smallest feasible x (x>=1): 2.5.
        ReaderCase {
            name: "lp-reader/continuous-bounds",
            lp: "Minimize\n  x + y\n\
                 Subject To\n  r1: x + 2 y >= 4\n  r2: x >= 1\n\
                 Bounds\n  0 <= x <= 10\n  y >= 0\n\
                 End\n",
            expected: 2.5,
        },
        // Integrality matters: LP relaxation is 1.5 (x=y=0.75), the binary
        // optimum is 1. A case that only passes if integer domains are wired.
        ReaderCase {
            name: "lp-reader/integrality-gap",
            lp: "Maximize\n  x + y\n\
                 Subject To\n  2 x + 2 y <= 3\n\
                 Binaries\n  x y\n\
                 End\n",
            expected: 1.0,
        },
        // General integer + a negative objective coefficient. Max 2a-3b with
        // b>=0 pushes b=0, then a<=5 binds: 2*5 = 10, a integral.
        ReaderCase {
            name: "lp-reader/general-and-negcoeff",
            lp: "\\ general integer, negative coefficient\n\
                 Maximize\n  2 a - 3 b\n\
                 Subject To\n  a - b <= 5\n  a + b <= 8\n\
                 Generals\n  a\n\
                 Bounds\n  a <= 10\n  b <= 10\n\
                 End\n",
            expected: 10.0,
        },
        // Constant folded from the constraint LHS: x + 2 <= 5 means x <= 3, so
        // max x = 3. Exercises the `expr + c (op) rhs -> expr (op) rhs-c` rule.
        ReaderCase {
            name: "lp-reader/constraint-constant",
            lp: "Maximize\n  x\n\
                 Subject To\n  x + 2 <= 5\n\
                 Bounds\n  x <= 10\n\
                 End\n",
            expected: 3.0,
        },
        // Ranged constraint, upper half binds: `1 <= x <= 3` expands to two
        // rows (x >= 1, x <= 3). With max x and x <= 10 elsewhere, a reader
        // that silently dropped the upper half would report 10, not 3.
        ReaderCase {
            name: "lp-reader/ranged-upper-binds",
            lp: "Maximize\n  x\n\
                 Subject To\n  r: 1 <= x <= 3\n\
                 Bounds\n  x <= 10\n\
                 End\n",
            expected: 3.0,
        },
        // Ranged constraint, lower half binds: with min x (default bounds
        // [0, 10]), a reader that dropped the lower half would report 0.
        ReaderCase {
            name: "lp-reader/ranged-lower-binds",
            lp: "Minimize\n  x\n\
                 Subject To\n  r: 2 <= x <= 5\n\
                 Bounds\n  x <= 10\n\
                 End\n",
            expected: 2.0,
        },
    ]
}

fn register_reader_tests(cases: &mut Vec<Case>) {
    for rc in reader_cases() {
        let ReaderCase { name, lp, expected } = rc;
        cases.push(Case::solve(name, Tier::Easy, 10, move || {
            let parsed = lp_format::parse(lp, false)
                .map_err(|e| format!("reader rejected a valid instance: {}", e))?;
            Ok((parsed.spec, parsed.problem, Expected::objective(expected)))
        }));
    }

    // Objective offset: `obj:` label + a constant term. microlp's Problem has no
    // objective constant, so the reader records it in `obj_offset` (and drops it
    // from the solved objective). This pins that documented behavior: the offset
    // is parsed as 10, and the solved objective is 2*3 = 6 (constant excluded).
    cases.push(Case::custom(
        "lp-reader/objective-offset",
        Tier::Easy,
        10,
        |budget| {
            let lp = "Minimize\n  obj: 2 x + 10\nst\n  x >= 3\nEnd\n";
            let parsed = lp_format::parse(lp, false).map_err(|e| format!("parse failed: {}", e))?;
            if (parsed.obj_offset - 10.0).abs() > 1e-12 {
                return Err(format!(
                    "objective offset: expected 10, parsed {}",
                    parsed.obj_offset
                ));
            }
            let mut problem = parsed.problem;
            problem.set_time_limit(budget);
            let sol = problem
                .solve()
                .map_err(|e| format!("solve errored: {}", e))?;
            if sol.status() != Status::Optimal {
                return Err(format!("expected Optimal, got {:?}", sol.status()));
            }
            if (sol.objective() - 6.0).abs() > 1e-6 {
                return Err(format!(
                    "expected solved objective 6 (offset excluded), got {}",
                    sol.objective()
                ));
            }
            crate::LAST_SOLVE.with(|slot| *slot.borrow_mut() = Some((sol.objective(), 6.0)));
            Ok(())
        },
    ));

    // Loud-error guarantees: the reader must reject, not misparse.
    register_reader_rejections(cases);
}

/// Each probe feeds the reader a construct outside its supported subset and
/// asserts it returns `Err` containing an expected phrase — never a silent
/// (mis)parse. This is the property that lets the suite trust a parsed instance.
fn register_reader_rejections(cases: &mut Vec<Case>) {
    let rejections: &[(&str, &str, &str)] = &[
        (
            // The grammar requires an objective sense; its error names the
            // missing "sense" token.
            "lp-reader/rejects-missing-sense",
            "\\ no Maximize/Minimize header\nSubject To\n  c: x + y <= 5\nBinaries\n  x y\nEnd\n",
            "sense",
        ),
        (
            "lp-reader/rejects-semicontinuous",
            "Maximize\n  x\nSubject To\n  x <= 5\nSemi-Continuous\n  x\nEnd\n",
            "semi-continuous",
        ),
    ];
    for &(name, lp, needle) in rejections {
        cases.push(Case::custom(
            name,
            Tier::Easy,
            5,
            move |_budget| match lp_format::parse(lp, false) {
                Ok(_) => Err(format!(
                    "expected the reader to reject this instance (looking for '{}'), but it parsed",
                    needle
                )),
                Err(e) => {
                    if e.to_lowercase().contains(needle) {
                        Ok(())
                    } else {
                        Err(format!(
                            "reader rejected as expected but message '{}' lacks '{}'",
                            e, needle
                        ))
                    }
                }
            },
        ));
    }
}

// ---------------------------------------------------------------------------
// 2. Vendored MILPBench instances microlp proves optimal (tier from folder).
// ---------------------------------------------------------------------------

/// A vendored instance selected because microlp proves its optimum in budget.
struct BenchInstance {
    /// Family sub-directory under `data/<tier>/milpbench/`.
    family: &'static str,
    /// File name, e.g. `CFL_easy_instance_15.lp`.
    file: &'static str,
    /// Per-instance time budget (seconds). Generous: these solve in ~0.6 s.
    budget_secs: u64,
    /// The certified optimum for this instance.
    ///
    /// MILPBench ships no reference optima for these families, so each value
    /// carries two independent certifications:
    ///
    /// 1. **HiGHS v1.15 zero-gap proof** (2026-07-11, benchmark harness run
    ///    `--solvers microlp,highs --filter CFL`): HiGHS proved optimality
    ///    with `mip_gap = 0` and its optimum matched this value within the
    ///    harness's cross-check tolerance — no conflicting-claims alert. The
    ///    harness independently re-validated both solvers' solutions against
    ///    the instance (bounds, integrality, constraints, objective).
    /// 2. **Structure**: the LP relaxation is integral, so the simplex-proven
    ///    LP optimum is itself an integer-feasible point and therefore the
    ///    true MILP optimum (any integer solution is LP-feasible, so it
    ///    cannot beat the LP bound).
    optimum: f64,
    tol: Tol,
}

/// Instances that cleared the selection bar (Optimal + zero gap in budget).
///
/// Only the CFL family qualifies: its "easy" instances are *wide* (≈800
/// constraints, ≈160 400 variables) with an integral LP relaxation, so the
/// simplex clears them in well under a second. Every other easy family (MIS,
/// MVC, SC, CAT, MIKS, BIP) is 20k–60k constraints and microlp does not reach
/// an integer incumbent — often not even the root LP relaxation — within 60 s;
/// those live in the xhard tier with externally certified optima instead.
///
/// The registration is data-driven: add an instance by dropping its `.lp` into
/// `data/<tier>/milpbench/<family>/` and appending a row here; the case tier
/// follows the folder.
const VENDORED: &[BenchInstance] = &[
    BenchInstance {
        family: "CFL_easy",
        file: "CFL_easy_instance_15.lp",
        budget_secs: 30,
        optimum: 9959.569649350798,
        tol: Tol {
            abs: 1e-3,
            rel: 1e-6,
        },
    },
    BenchInstance {
        family: "CFL_easy",
        file: "CFL_easy_instance_12.lp",
        budget_secs: 30,
        optimum: 10000.194107917776,
        tol: Tol {
            abs: 1e-3,
            rel: 1e-6,
        },
    },
];

fn register_bench_instances(cases: &mut Vec<Case>) {
    for inst in VENDORED {
        let family = inst.family;
        let file = inst.file;
        let budget_secs = inst.budget_secs;
        let optimum = inst.optimum;
        let tol = inst.tol;
        let name = format!("milpbench/{}/{}", family, file.trim_end_matches(".lp"));
        let (path, tier) = locate(&format!("milpbench/{}", family), file);
        cases.push(Case::custom(name, tier, budget_secs, move |budget| {
            let text = read_instance(&path)?;
            let parsed = lp_format::parse(&text, false)
                .map_err(|e| format!("LP reader failed on {}: {}", file, e))?;
            let mut problem = parsed.problem;
            problem.set_time_limit(budget);
            // A solve *error* here (not a clean status) would be a solver bug on
            // a large input; surface it loudly rather than swallowing it.
            let sol = problem.solve().map_err(|e| {
                format!(
                    "solve errored (solver bug — investigate before vendoring): {}",
                    e
                )
            })?;
            // Selection bar: proven optimality only.
            if sol.status() != Status::Optimal {
                return Err(format!(
                    "expected Status::Optimal within budget, got {:?}",
                    sol.status()
                ));
            }
            match sol.gap() {
                Some(g) if g.abs() <= 1e-6 => {}
                other => {
                    return Err(format!(
                        "expected a proven-optimal zero gap, got gap = {:?}",
                        other
                    ))
                }
            }
            // Independent shadow-model validation of the returned point
            // (feasibility, bounds, integrality, objective consistency).
            crate::verify::validate_incumbent(&parsed.spec, &sol)?;
            // Compare the result against the recorded optimum (see `optimum` doc).
            if !tol.matches(sol.objective(), optimum) {
                return Err(format!(
                    "expected optimum {} (microlp-proven), got {} (diff {:.3e})",
                    optimum,
                    sol.objective(),
                    (sol.objective() - optimum).abs()
                ));
            }
            crate::LAST_SOLVE.with(|slot| *slot.borrow_mut() = Some((sol.objective(), optimum)));
            Ok(())
        }));
    }
}
