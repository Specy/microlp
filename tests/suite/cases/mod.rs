//! Case registry: every family contributes `Case`s with a stable unique name.

use crate::model::{Expected, ModelSpec};
use microlp::Problem;
use std::time::Duration;

mod incremental;
mod lp_random;
mod lp_unit;
mod milp_random;
mod milp_unit;
mod miplib;
mod netlib;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tier {
    /// Milliseconds each; runs everywhere, including debug CI.
    Quick,
    /// The default release-mode suite; laptop-friendly.
    Standard,
    /// Opt-in via --hard: instances a naive branch & bound may take minutes on.
    Hard,
}

impl Tier {
    pub fn label(self) -> &'static str {
        match self {
            Tier::Quick => "quick",
            Tier::Standard => "standard",
            Tier::Hard => "hard",
        }
    }
}

pub enum CaseRun {
    /// Build (shadow model, problem, expectation); the runner solves+verifies.
    #[allow(clippy::type_complexity)]
    Solve(Box<dyn Fn() -> Result<(ModelSpec, Problem, Expected), String>>),
    /// Self-contained check (e.g. metamorphic invariants). Receives the time
    /// budget so it can set per-solve limits; returns Err(reason) on failure.
    Custom(Box<dyn Fn(Duration) -> Result<(), String>>),
}

pub struct Case {
    pub name: String,
    pub tier: Tier,
    pub budget: Duration,
    pub run: CaseRun,
}

impl Case {
    pub fn solve(
        name: impl Into<String>,
        tier: Tier,
        budget_secs: u64,
        build: impl Fn() -> Result<(ModelSpec, Problem, Expected), String> + 'static,
    ) -> Case {
        Case {
            name: name.into(),
            tier,
            budget: Duration::from_secs(budget_secs),
            run: CaseRun::Solve(Box::new(build)),
        }
    }

    pub fn custom(
        name: impl Into<String>,
        tier: Tier,
        budget_secs: u64,
        run: impl Fn(Duration) -> Result<(), String> + 'static,
    ) -> Case {
        Case {
            name: name.into(),
            tier,
            budget: Duration::from_secs(budget_secs),
            run: CaseRun::Custom(Box::new(run)),
        }
    }
}

pub fn all() -> Vec<Case> {
    let mut cases = vec![];
    lp_unit::register(&mut cases);
    lp_random::register(&mut cases);
    milp_unit::register(&mut cases);
    milp_random::register(&mut cases);
    incremental::register(&mut cases);
    netlib::register(&mut cases);
    miplib::register(&mut cases);

    // Names must be unique: they are the stable handle for --filter and repro.
    let mut seen = std::collections::BTreeSet::new();
    for c in &cases {
        assert!(
            seen.insert(c.name.clone()),
            "duplicate case name {}",
            c.name
        );
    }
    cases
}
