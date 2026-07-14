//! The solvers under comparison. Each contender builds its native model from
//! the neutral [`ModelSpec`](crate::model::ModelSpec) *inside* [`Contender::run`]
//! and solves it there, so every solver is measured over the same work:
//! instance data in memory → answer. File parsing is excluded for everyone.
//!
//! Fairness settings applied to every rival: one thread, the same relative
//! MIP gap as microlp (0 by default — prove exact optimality), everything
//! else at the solver's own defaults. Solvers that time out report the
//! incumbent they were holding, so timed-out instances still compare
//! solution quality; every reported incumbent is independently re-validated
//! by the caller.

use crate::corpus::Instance;
use crate::model::Domain;
#[cfg(any(feature = "highs", feature = "clarabel", feature = "scip"))]
use crate::model::ModelSpec;
use microlp::Status;
use std::time::Duration;

pub enum RunStatus {
    Optimal,
    Feasible,
    Interrupted,
    Infeasible,
    Unbounded,
    Error(String),
}

impl RunStatus {
    pub fn label(&self) -> String {
        match self {
            RunStatus::Optimal => "optimal".into(),
            RunStatus::Feasible => "feasible".into(),
            RunStatus::Interrupted => "interrupted".into(),
            RunStatus::Infeasible => "infeasible".into(),
            RunStatus::Unbounded => "unbounded".into(),
            RunStatus::Error(e) => format!("error: {}", e),
        }
    }
}

pub struct RunOutcome {
    pub status: RunStatus,
    /// Objective of the returned solution (absent when there is none).
    pub objective: Option<f64>,
    /// Variable values in spec order, for independent validation.
    pub values: Option<Vec<f64>>,
    /// Best proven bound on the optimum, when the solver reports one.
    pub bound: Option<f64>,
    /// Solver-reported relative optimality gap, when available.
    pub gap: Option<f64>,
    pub nodes: Option<u64>,
    pub simplex_iters: Option<u64>,
}

impl RunOutcome {
    pub fn bare(status: RunStatus) -> RunOutcome {
        RunOutcome {
            status,
            objective: None,
            values: None,
            bound: None,
            gap: None,
            nodes: None,
            simplex_iters: None,
        }
    }
}

/// One solve request: the shared time budget plus the run-mode knobs.
pub struct SolveTask {
    pub budget: Duration,
    /// Untimed certification solve (used only to obtain a proven optimum for
    /// the report's correction-gap column): the solver may use every thread
    /// it wants, but must prove *exact* optimality — the caller forces
    /// `mip_gap` to 0 on reference solves.
    #[cfg_attr(not(feature = "highs"), allow(dead_code))]
    pub reference: bool,
    /// Relative MIP gap at which a solver may stop and report the optimum
    /// (0 = exact). Applied to every solver that supports it, so "proved
    /// optimum" keeps meaning the same thing for everyone.
    pub mip_gap: f64,
}

pub trait Contender {
    fn name(&self) -> &'static str;
    fn supports_mip(&self) -> bool;
    /// Build the solver's native model from `inst.spec` and solve it within
    /// `task.budget`. The caller times this whole call.
    fn run(&self, inst: &Instance, task: &SolveTask) -> RunOutcome;
}

/// All compiled-in contenders, microlp first.
pub fn all() -> Vec<Box<dyn Contender>> {
    #[allow(unused_mut)]
    let mut v: Vec<Box<dyn Contender>> = vec![Box::new(Microlp)];
    #[cfg(feature = "highs")]
    v.push(Box::new(highs_solver::Highs));
    #[cfg(feature = "scip")]
    v.push(Box::new(good_lp_solver::GoodLp(
        good_lp_solver::Backend::Scip,
    )));
    #[cfg(feature = "clarabel")]
    v.push(Box::new(good_lp_solver::GoodLp(
        good_lp_solver::Backend::Clarabel,
    )));
    v
}

pub fn by_name(name: &str) -> Option<Box<dyn Contender>> {
    all().into_iter().find(|c| c.name() == name)
}

/// Recompute the objective from variable values — rivals report this instead
/// of their internal value so every solver's number comes from one formula.
#[cfg(any(feature = "highs", feature = "clarabel", feature = "scip"))]
pub fn objective_of(spec: &ModelSpec, values: &[f64]) -> f64 {
    spec.vars
        .iter()
        .zip(values)
        .map(|(v, &x)| v.obj_coeff * x)
        .sum()
}

/// Integer bounds in the shadow spec originate from the adapters' i32 clamp,
/// so this conversion is exact; anything else is a corpus bug worth a panic.
fn int_bound(x: f64) -> i32 {
    if x <= i32::MIN as f64 {
        i32::MIN
    } else if x >= i32::MAX as f64 {
        i32::MAX
    } else {
        let r = x.round();
        assert!(
            (x - r).abs() < 1e-9,
            "non-integral bound {} on an integer variable",
            x
        );
        r as i32
    }
}

// ---------------------------------------------------------------------------
// microlp
// ---------------------------------------------------------------------------

struct Microlp;

impl Contender for Microlp {
    fn name(&self) -> &'static str {
        "microlp"
    }

    fn supports_mip(&self) -> bool {
        true
    }

    fn run(&self, inst: &Instance, task: &SolveTask) -> RunOutcome {
        let mut problem = microlp::Problem::new(inst.direction);
        let vars: Vec<microlp::Variable> = inst
            .spec
            .vars
            .iter()
            .map(|v| match v.domain {
                Domain::Real => problem.add_var(v.obj_coeff, (v.min, v.max)),
                Domain::Integer => {
                    problem.add_integer_var(v.obj_coeff, (int_bound(v.min), int_bound(v.max)))
                }
            })
            .collect();
        for c in &inst.spec.constraints {
            problem.add_constraint(c.terms.iter().map(|&(vi, x)| (vars[vi], x)), c.op, c.rhs);
        }

        let mut options = microlp::SolveOptions::default();
        options.time_limit = Some(task.budget);
        options.mip_gap = task.mip_gap;
        let solution = match problem.solve_with(options) {
            Ok(s) => s,
            Err(microlp::Error::Infeasible) => return RunOutcome::bare(RunStatus::Infeasible),
            Err(microlp::Error::Unbounded) => return RunOutcome::bare(RunStatus::Unbounded),
            Err(e) => return RunOutcome::bare(RunStatus::Error(e.to_string())),
        };

        let stats = solution.stats();
        let (status, has_answer) = match solution.status() {
            Status::Optimal => (RunStatus::Optimal, true),
            Status::Feasible => (RunStatus::Feasible, true),
            Status::Interrupted => (RunStatus::Interrupted, false),
        };
        RunOutcome {
            status,
            objective: has_answer.then(|| solution.objective()),
            values: has_answer.then(|| vars.iter().map(|&v| solution.var_value(v)).collect()),
            bound: stats.best_bound,
            gap: stats.gap,
            nodes: Some(stats.nodes_solved),
            simplex_iters: Some(stats.lp_iterations),
        }
    }
}

// ---------------------------------------------------------------------------
// HiGHS (via the `highs` crate; default feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "highs")]
mod highs_solver {
    use super::*;

    pub struct Highs;

    impl Contender for Highs {
        fn name(&self) -> &'static str {
            "highs"
        }

        fn supports_mip(&self) -> bool {
            true
        }

        fn run(&self, inst: &Instance, task: &SolveTask) -> RunOutcome {
            run_highs(inst, task)
        }
    }

    fn run_highs(inst: &Instance, task: &SolveTask) -> RunOutcome {
        let mut pb = highs::RowProblem::default();
        let cols: Vec<highs::Col> = inst
            .spec
            .vars
            .iter()
            .map(|v| match v.domain {
                Domain::Real => pb.add_column(v.obj_coeff, v.min..=v.max),
                Domain::Integer => pb.add_integer_column(v.obj_coeff, v.min..=v.max),
            })
            .collect();
        for c in &inst.spec.constraints {
            let row: Vec<(highs::Col, f64)> =
                c.terms.iter().map(|&(vi, x)| (cols[vi], x)).collect();
            match c.op {
                microlp::ComparisonOp::Le => pb.add_row(..=c.rhs, row),
                microlp::ComparisonOp::Ge => pb.add_row(c.rhs.., row),
                microlp::ComparisonOp::Eq => pb.add_row(c.rhs..=c.rhs, row),
            };
        }
        let sense = match inst.direction {
            microlp::OptimizationDirection::Minimize => highs::Sense::Minimise,
            microlp::OptimizationDirection::Maximize => highs::Sense::Maximise,
        };
        let mut model = pb.optimise(sense);
        model.set_option("time_limit", task.budget.as_secs_f64());
        if task.reference {
            // Certification solve: throw every core at it.
            model.set_option("parallel", "on");
            let threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            model.set_option("threads", threads as i32);
        } else {
            model.set_option("threads", 1);
            model.set_option("parallel", "off");
        }
        model.set_option("output_flag", false);
        model.set_option("mip_rel_gap", task.mip_gap);
        model.set_option("mip_abs_gap", 0.0);

        let solved = model.solve();
        use highs::HighsModelStatus as S;
        match solved.status() {
            S::Optimal => {
                let values = solved.get_solution().columns().to_vec();
                let objective = objective_of(&inst.spec, &values);
                RunOutcome {
                    status: RunStatus::Optimal,
                    objective: Some(objective),
                    values: Some(values),
                    bound: Some(objective),
                    gap: Some(0.0),
                    nodes: None,
                    simplex_iters: None,
                }
            }
            S::Infeasible => RunOutcome::bare(RunStatus::Infeasible),
            S::Unbounded => RunOutcome::bare(RunStatus::Unbounded),
            S::UnboundedOrInfeasible => RunOutcome::bare(RunStatus::Error(
                "unbounded-or-infeasible (HiGHS did not separate the two)".into(),
            )),
            S::ReachedTimeLimit => {
                // A finite mip_gap means HiGHS holds an integer-feasible
                // incumbent (it reports its infinity constant otherwise);
                // return it so timed-out instances still compare solution
                // quality. The caller re-validates the values.
                let gap = solved.mip_gap();
                if inst.is_mip && gap.is_finite() && gap < 1e29 {
                    let values = solved.get_solution().columns().to_vec();
                    let objective = objective_of(&inst.spec, &values);
                    RunOutcome {
                        status: RunStatus::Feasible,
                        objective: Some(objective),
                        values: Some(values),
                        bound: None,
                        gap: Some(gap),
                        nodes: None,
                        simplex_iters: None,
                    }
                } else {
                    RunOutcome::bare(RunStatus::Interrupted)
                }
            }
            other => RunOutcome::bare(RunStatus::Error(format!("HiGHS status {:?}", other))),
        }
    }
}

// ---------------------------------------------------------------------------
// Rivals driven through the good_lp modeling layer (optional features)
// ---------------------------------------------------------------------------

#[cfg(any(feature = "clarabel", feature = "scip"))]
mod good_lp_solver {
    use super::*;
    use good_lp::solvers::{ResolutionError, SolutionStatus, SolverModel as _};
    use good_lp::{variable, variables, Expression};

    #[derive(Clone, Copy)]
    pub enum Backend {
        /// Clarabel: pure-Rust interior-point solver; LP only (its good_lp
        /// backend panics on integer variables, so the harness never sends
        /// it a MILP).
        #[cfg(feature = "clarabel")]
        Clarabel,
        /// SCIP via russcip, using the bundled precompiled library.
        #[cfg(feature = "scip")]
        Scip,
    }

    pub struct GoodLp(pub Backend);

    impl Contender for GoodLp {
        fn name(&self) -> &'static str {
            match self.0 {
                #[cfg(feature = "clarabel")]
                Backend::Clarabel => "clarabel",
                #[cfg(feature = "scip")]
                Backend::Scip => "scip",
            }
        }

        fn supports_mip(&self) -> bool {
            match self.0 {
                #[cfg(feature = "clarabel")]
                Backend::Clarabel => false,
                #[cfg(feature = "scip")]
                Backend::Scip => true,
            }
        }

        fn run(&self, inst: &Instance, task: &SolveTask) -> RunOutcome {
            let spec = &inst.spec;
            let mut vars = variables!();
            let handles: Vec<good_lp::Variable> = spec
                .vars
                .iter()
                .map(|v| {
                    let mut def = variable().min(v.min).max(v.max);
                    if v.domain == Domain::Integer {
                        def = def.integer();
                    }
                    vars.add(def)
                })
                .collect();
            let mut objective = Expression::with_capacity(spec.vars.len());
            for (v, h) in spec.vars.iter().zip(&handles) {
                objective.add_mul(v.obj_coeff, *h);
            }
            let unsolved = match inst.direction {
                microlp::OptimizationDirection::Minimize => vars.minimise(objective),
                microlp::OptimizationDirection::Maximize => vars.maximise(objective),
            };
            match self.0 {
                #[cfg(feature = "clarabel")]
                Backend::Clarabel => {
                    // Clarabel has no time-limit API in good_lp; the
                    // orchestrator's hard process deadline nets a runaway.
                    let _ = task;
                    let mut model = unsolved.using(good_lp::solvers::clarabel::clarabel);
                    for c in &spec.constraints {
                        model = model.with(to_constraint(c, &handles));
                    }
                    finish(model.solve(), inst, &handles)
                }
                #[cfg(feature = "scip")]
                Backend::Scip => {
                    use good_lp::solvers::{WithMipGap as _, WithTimeLimit as _};
                    let mut model = unsolved
                        .using(good_lp::solvers::scip::scip)
                        .with_time_limit(task.budget.as_secs_f64());
                    if task.mip_gap > 0.0 {
                        model = match model.with_mip_gap(task.mip_gap as f32) {
                            Ok(m) => m,
                            Err(e) => {
                                return RunOutcome::bare(RunStatus::Error(format!(
                                    "cannot set the SCIP mip gap: {}",
                                    e
                                )))
                            }
                        };
                    }
                    for c in &spec.constraints {
                        model = model.with(to_constraint(c, &handles));
                    }
                    finish(model.solve(), inst, &handles)
                }
            }
        }
    }

    fn to_constraint(
        c: &crate::model::ConstraintSpec,
        handles: &[good_lp::Variable],
    ) -> good_lp::Constraint {
        let mut e = Expression::with_capacity(c.terms.len());
        for &(vi, x) in &c.terms {
            e.add_mul(x, handles[vi]);
        }
        match c.op {
            microlp::ComparisonOp::Le => e.leq(c.rhs),
            microlp::ComparisonOp::Ge => e.geq(c.rhs),
            microlp::ComparisonOp::Eq => e.eq(c.rhs),
        }
    }

    fn finish<S: good_lp::Solution>(
        res: Result<S, ResolutionError>,
        inst: &Instance,
        handles: &[good_lp::Variable],
    ) -> RunOutcome {
        match res {
            // GapLimit means "proved within the configured mip gap", which
            // is exactly what the other solvers report as Optimal then.
            Ok(sol) => match sol.status() {
                SolutionStatus::Optimal | SolutionStatus::GapLimit => {
                    let values: Vec<f64> = handles.iter().map(|&h| sol.value(h)).collect();
                    let objective = objective_of(&inst.spec, &values);
                    RunOutcome {
                        status: RunStatus::Optimal,
                        objective: Some(objective),
                        values: Some(values),
                        bound: Some(objective),
                        gap: Some(0.0),
                        nodes: None,
                        simplex_iters: None,
                    }
                }
                SolutionStatus::TimeLimit => {
                    // Report the incumbent held at the limit so timed-out
                    // instances still compare solution quality. good_lp's
                    // SCIP wrapper has no "is there a solution?" accessor —
                    // `value()` panicking is its only signal that the solver
                    // held nothing — so a caught panic here means a clean
                    // "timed out with no incumbent", not a bug.
                    let extracted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        handles.iter().map(|&h| sol.value(h)).collect::<Vec<f64>>()
                    }));
                    match extracted {
                        Ok(values) => {
                            let objective = objective_of(&inst.spec, &values);
                            RunOutcome {
                                status: RunStatus::Feasible,
                                objective: Some(objective),
                                values: Some(values),
                                bound: None,
                                gap: None,
                                nodes: None,
                                simplex_iters: None,
                            }
                        }
                        Err(_) => RunOutcome::bare(RunStatus::Interrupted),
                    }
                }
            },
            Err(ResolutionError::Infeasible) => RunOutcome::bare(RunStatus::Infeasible),
            Err(ResolutionError::Unbounded) => RunOutcome::bare(RunStatus::Unbounded),
            Err(e) => RunOutcome::bare(RunStatus::Error(e.to_string())),
        }
    }
}
