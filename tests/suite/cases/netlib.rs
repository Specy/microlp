//! Netlib LP benchmark cases, read through the suite's own MPS adapter
//! (`mps_milp`, the same reader `miplib.rs` uses) and checked against the
//! official netlib "BR" values; see data/README.md.
//!
//! Each case's tier is derived from the folder its `.mps` file lives in
//! (`data/<tier>/netlib/`); moving a file re-tiers the case with no code
//! change.

use super::{locate, read_instance, Case};
use crate::model::{Expected, Tol};
use crate::mps_milp;
use microlp::OptimizationDirection;

/// Instances explicitly expected to return `Err(Infeasible)` despite a
/// published feasible optimum. Any other result fails the case so the marker
/// cannot silently bypass normal objective validation.
const KNOWN_INFEASIBLE_BUG: &[&str] = &[];

const NETLIB: &[(&str, f64)] = &[
    ("afiro", -4.6475314286e2),
    ("sc50a", -6.4575077059e1),
    ("sc50b", -7.0000000000e1),
    ("sc105", -5.2202061212e1),
    ("adlittle", 2.2549496316e5),
    ("blend", -3.0812149846e1),
    ("kb2", -1.7499001299e3),
    ("share2b", -4.1573224074e2),
    ("stocfor1", -4.1131976219e4),
    ("scagr7", -2.3313892548e6),
    ("israel", -8.9664482186e5),
    ("brandy", 1.5185098965e3),
    ("beaconfd", 3.3592485807e4),
    ("sctap1", 1.4122500000e3),
];

pub fn register(cases: &mut Vec<Case>) {
    for &(name, expected) in NETLIB {
        let case_name = format!("netlib/{}", name);
        let (_, tier) = locate("netlib", &format!("{}.mps", name));
        if KNOWN_INFEASIBLE_BUG.contains(&name) {
            cases.push(Case::custom(case_name, tier, 60, move |budget| {
                let parsed = load(name)?;
                let mut problem = parsed.problem;
                problem.set_time_limit(budget);
                match problem.solve() {
                    // A marked case accepts only its explicit expected error.
                    Err(microlp::Error::Infeasible) => Ok(()),
                    Ok(outcome) => match outcome.solution() {
                        Some(sol) => Err(format!(
                            "the {} marked failure solved with objective {} \
                             — verify it against the published value {} and remove it from \
                             KNOWN_INFEASIBLE_BUG to enable normal validation",
                            name,
                            sol.objective(),
                            expected
                        )),
                        None => Err(format!(
                            "the {} marked failure was interrupted instead of returning \
                             Err(Infeasible)",
                            name
                        )),
                    },
                    Err(e) => Err(format!(
                        "known failure changed shape: expected Err(Infeasible), got: {}",
                        e
                    )),
                }
            }));
            continue;
        }
        cases.push(Case::solve(case_name, tier, 60, move || {
            let parsed = load(name)?;
            // Netlib official values carry ~11 significant digits.
            let tol = Tol {
                abs: 1e-6,
                rel: 1e-6,
            };
            Ok((
                parsed.spec,
                parsed.problem,
                Expected::objective_tol(expected, tol),
            ))
        }));
    }
}

fn load(name: &str) -> Result<mps_milp::ParsedMps, String> {
    let (path, _) = locate("netlib", &format!("{}.mps", name));
    let text = read_instance(&path)?;
    let parsed = mps_milp::parse(&text, OptimizationDirection::Minimize, false)?;
    if parsed.obj_offset != 0.0 {
        // None of the vendored files carries an objective constant; if one
        // ever does, the expected values above must be shifted accordingly.
        return Err(format!(
            "{} has objective offset {}; expected values need adjusting",
            name, parsed.obj_offset
        ));
    }
    Ok(parsed)
}
