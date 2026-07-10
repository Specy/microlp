//! Netlib LP benchmark cases. The instance is parsed twice: once with
//! microlp's own `MpsFile` (whose `Problem` is the one solved, so the library
//! parser is under test too) and once with the suite's independent reader
//! (whose shadow model is what the solution is validated against). Expected
//! objectives are the official netlib "BR" values; see data/README.md.
//!
//! Each case's tier is derived from the folder its `.mps` file lives in
//! (`data/<tier>/netlib/`); moving a file re-tiers the case with no code
//! change.

use super::{locate, read_instance, Case};
use crate::model::{Expected, Tol};
use crate::mps_milp;
use microlp::{MpsFile, OptimizationDirection};
use std::io::BufReader;

/// Instances pinned EXPECTED-TO-FAIL as `Err(Infeasible)`: such a case
/// passes while its known engine bug reproduces exactly, and fails loudly
/// the moment the engine starts answering — fixing the bug flips the marker
/// deliberately, keeping the hard tier a usable gate in the meantime.
/// (Currently empty; `brandy` lived here until the refresh valve in
/// `Solver::restore_feasibility` fixed its false-infeasible.)
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
        let (path, tier) = locate("netlib", &format!("{}.mps", name));
        if KNOWN_INFEASIBLE_BUG.contains(&name) {
            cases.push(Case::custom(case_name, tier, 60, move |budget| {
                let text = read_instance(&path)?;
                let file = MpsFile::parse(
                    BufReader::new(text.as_bytes()),
                    OptimizationDirection::Minimize,
                )
                .map_err(|e| format!("MpsFile::parse failed: {}", e))?;
                let mut problem = file.problem;
                problem.set_time_limit(budget);
                match problem.solve() {
                    // The bug reproducing exactly is the expected outcome.
                    Err(microlp::Error::Infeasible) => Ok(()),
                    Ok(sol) => Err(format!(
                        "the {} known failure appears HEALED: it solved with objective {} \
                         — check it against the published value {} and remove it from \
                         KNOWN_INFEASIBLE_BUG so it is verified normally from now on",
                        name,
                        sol.objective(),
                        expected
                    )),
                    Err(e) => Err(format!(
                        "known failure changed shape: expected Err(Infeasible), got: {}",
                        e
                    )),
                }
            }));
            continue;
        }
        cases.push(Case::solve(case_name, tier, 60, move || {
            let text = read_instance(&path)?;

            // The problem actually solved comes from the library's parser.
            let file = MpsFile::parse(
                BufReader::new(text.as_bytes()),
                OptimizationDirection::Minimize,
            )
            .map_err(|e| format!("MpsFile::parse failed: {}", e))?;

            // The shadow model comes from the suite's independent reader.
            let shadow = mps_milp::parse(&text, OptimizationDirection::Minimize, true)
                .map_err(|e| format!("suite reader failed: {}", e))?;
            if shadow.spec.vars.len() != file.variables.len() {
                return Err(format!(
                    "parser disagreement: suite reader sees {} variables, MpsFile {}",
                    shadow.spec.vars.len(),
                    file.variables.len()
                ));
            }

            // Netlib official values carry ~11 significant digits.
            let tol = Tol {
                abs: 1e-6,
                rel: 1e-6,
            };
            Ok((
                shadow.spec,
                file.problem,
                Expected::objective_tol(expected, tol),
            ))
        }));
    }
}
