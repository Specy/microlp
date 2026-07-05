//! Netlib LP benchmark cases. The instance is parsed twice: once with
//! microlp's own `MpsFile` (whose `Problem` is the one solved, so the library
//! parser is under test too) and once with the suite's independent reader
//! (whose shadow model is what the solution is validated against). Expected
//! objectives are the official netlib "BR" values; see data/README.md.

use super::{Case, Tier};
use crate::model::{Expected, Tol};
use crate::mps_milp;
use microlp::{MpsFile, OptimizationDirection};
use std::io::BufReader;

const NETLIB: &[(&str, f64, Tier)] = &[
    ("afiro", -4.6475314286e2, Tier::Standard),
    ("sc50a", -6.4575077059e1, Tier::Standard),
    ("sc50b", -7.0000000000e1, Tier::Standard),
    ("sc105", -5.2202061212e1, Tier::Standard),
    ("adlittle", 2.2549496316e5, Tier::Standard),
    ("blend", -3.0812149846e1, Tier::Standard),
    ("kb2", -1.7499001299e3, Tier::Standard),
    ("share2b", -4.1573224074e2, Tier::Standard),
    ("stocfor1", -4.1131976219e4, Tier::Standard),
    ("scagr7", -2.3313892548e6, Tier::Standard),
    ("israel", -8.9664482186e5, Tier::Hard),
    ("brandy", 1.5185098965e3, Tier::Hard),
    ("beaconfd", 3.3592485807e4, Tier::Hard),
    ("sctap1", 1.4122500000e3, Tier::Hard),
];

pub fn data_path(sub: &str, file: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("suite")
        .join("data")
        .join(sub)
        .join(file)
}

pub fn register(cases: &mut Vec<Case>) {
    for &(name, expected, tier) in NETLIB {
        let case_name = format!("netlib/{}", name);
        cases.push(Case::solve(case_name, tier, 60, move || {
            let path = data_path("netlib", &format!("{}.mps", name));
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;

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
