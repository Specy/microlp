//! What a solve-like call returns: either a [`Solution`] or, when a limit cut
//! the search short, an [`InterruptedSolve`] that can be resumed.

use crate::solver::Solver;
use crate::{
    mip, timed_lp_call, Error, OptimizationDirection, ResumeOptions, Solution, SolutionStatus,
    SolveState, Stats, StopReason, TerminationReason,
};
use web_time::Instant;

/// The result of a successful solve.
///
/// The outcome either contains a validated [`Solution`] or reports that a
/// configured limit interrupted the call before a usable assignment was found.
/// An interrupted outcome has no objective or variable values, but it can be
/// resumed to keep searching for an incumbent.
#[derive(Clone)]
pub enum SolveOutcome {
    /// A validated assignment that is optimal or feasible without a proof of
    /// optimality.
    Solution(Solution),
    /// A time or node limit was reached before a usable assignment was found.
    Interrupted(InterruptedSolve),
}

impl std::fmt::Debug for SolveOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Solution(solution) => f.debug_tuple("Solution").field(solution).finish(),
            Self::Interrupted(interrupted) => {
                f.debug_tuple("Interrupted").field(interrupted).finish()
            }
        }
    }
}

impl SolveOutcome {
    /// Wraps a pure-LP engine state. A finished one is first held to the rows
    /// presolve dropped or rewrote: the engine takes over a row the answer
    /// misses and resumes, within the call's deadline
    /// ([`Solver::hold_validation_rows`]). It then becomes a [`Solution`] only
    /// if its values meet the problem's rows and bounds in user units (the
    /// [`Tolerances::feasibility`](crate::Tolerances::feasibility) contract,
    /// rows summed in compensated arithmetic). The engine holds its rows and
    /// bounds to that contract less the round-off of its own evaluation, so a
    /// failure here is an internal contradiction, returned as
    /// [`Error::InternalError`], as is an objective of those values that
    /// cannot be evaluated in `f64`, its terms (costs times values) or their
    /// sum overflowing.
    pub(crate) fn from_lp_stop(
        direction: OptimizationDirection,
        num_vars: usize,
        stop: StopReason,
        mut solver: Box<Solver>,
        last_options: ResumeOptions,
    ) -> Result<Self, Error> {
        let stop = match stop {
            StopReason::Finished => {
                let started = Instant::now();
                let held = solver.hold_validation_rows();
                solver.elapsed += started.elapsed();
                held?
            }
            StopReason::Limit => StopReason::Limit,
        };
        match stop {
            StopReason::Finished => {
                let lp_values = solver.reported_values();
                let lp_objective = solver.objective_of(&lp_values);
                if let Some((var, violation)) = solver.first_violated_bound(&lp_values) {
                    return Err(Error::InternalError(format!(
                        "LP solution violates the bounds of variable {var} by {violation:e}"
                    )));
                }
                if let Some((row, violation)) = solver.first_violated_row(&lp_values) {
                    return Err(Error::InternalError(format!(
                        "LP solution violates engine row {row} by {violation:e} (the engine's \
                         numbering, not the problem's: it leaves out the rows every point \
                         satisfies, empty or with an infinite right-hand side on their open \
                         side, and numbers the rows presolve took out last)"
                    )));
                }
                if !lp_objective.is_finite() {
                    return Err(Error::InternalError(
                        "the objective of the LP solution cannot be evaluated in f64: its \
                         terms (costs times values) or their sum overflow"
                            .to_string(),
                    ));
                }
                let (mut lp_duals, mut lp_reduced_costs) = solver.duals();
                if direction == OptimizationDirection::Maximize {
                    for value in lp_duals.iter_mut().chain(&mut lp_reduced_costs) {
                        *value = -*value;
                    }
                }
                Ok(Self::Solution(Solution {
                    direction,
                    num_vars,
                    status: SolutionStatus::Optimal,
                    termination_reason: TerminationReason::ProvenOptimal,
                    lp_values,
                    lp_objective,
                    lp_duals,
                    lp_reduced_costs,
                    state: SolveState::Lp(solver),
                    last_options,
                }))
            }
            StopReason::Limit => Ok(Self::Interrupted(InterruptedSolve {
                direction,
                num_vars,
                termination_reason: TerminationReason::TimeLimit,
                state: SolveState::Lp(solver),
                last_options,
            })),
        }
    }

    pub(crate) fn from_mip_run(
        direction: OptimizationDirection,
        num_vars: usize,
        run: mip::MipRun,
        last_options: ResumeOptions,
    ) -> Self {
        let mip::MipRun { reason, state } = run;
        let has_incumbent = state.incumbent.is_some() && !state.classifying_unbounded;
        match reason {
            TerminationReason::ProvenOptimal => {
                debug_assert!(has_incumbent);
                Self::Solution(Solution {
                    direction,
                    num_vars,
                    status: SolutionStatus::Optimal,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                    lp_values: Vec::new(),
                    lp_objective: 0.0,
                    lp_duals: Vec::new(),
                    lp_reduced_costs: Vec::new(),
                })
            }
            TerminationReason::MipGap => {
                debug_assert!(has_incumbent);
                Self::Solution(Solution {
                    direction,
                    num_vars,
                    status: SolutionStatus::Feasible,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                    lp_values: Vec::new(),
                    lp_objective: 0.0,
                    lp_duals: Vec::new(),
                    lp_reduced_costs: Vec::new(),
                })
            }
            TerminationReason::TimeLimit | TerminationReason::NodeLimit if has_incumbent => {
                Self::Solution(Solution {
                    direction,
                    num_vars,
                    status: SolutionStatus::Feasible,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                    lp_values: Vec::new(),
                    lp_objective: 0.0,
                    lp_duals: Vec::new(),
                    lp_reduced_costs: Vec::new(),
                })
            }
            TerminationReason::TimeLimit | TerminationReason::NodeLimit => {
                Self::Interrupted(InterruptedSolve {
                    direction,
                    num_vars,
                    termination_reason: reason,
                    state: SolveState::Mip(Box::new(state)),
                    last_options,
                })
            }
        }
    }

    /// Borrows the solution contained in this outcome, if one is available.
    ///
    /// Returns `None` for [`SolveOutcome::Interrupted`].
    pub fn solution(&self) -> Option<&Solution> {
        match self {
            Self::Solution(solution) => Some(solution),
            Self::Interrupted(_) => None,
        }
    }

    /// Consumes this outcome and returns its solution.
    ///
    /// If the outcome was interrupted, the [`InterruptedSolve`] value is
    /// returned as the error so it can still be inspected or resumed.
    pub fn into_solution(self) -> Result<Solution, InterruptedSolve> {
        match self {
            Self::Solution(solution) => Ok(solution),
            Self::Interrupted(interrupted) => Err(interrupted),
        }
    }

    /// Returns why the call that produced this outcome stopped.
    pub fn termination_reason(&self) -> TerminationReason {
        match self {
            Self::Solution(solution) => solution.termination_reason(),
            Self::Interrupted(interrupted) => interrupted.termination_reason(),
        }
    }

    /// Returns statistics for the solve history represented by this outcome.
    ///
    /// The values include any resumes performed before this outcome was
    /// produced.
    pub fn stats(&self) -> Stats {
        match self {
            Self::Solution(solution) => solution.stats(),
            Self::Interrupted(interrupted) => interrupted.stats(),
        }
    }

    /// Returns whether this outcome contains a solution with proven optimality.
    pub fn is_optimal(&self) -> bool {
        matches!(
            self,
            Self::Solution(Solution {
                status: SolutionStatus::Optimal,
                ..
            })
        )
    }

    /// Returns the settings that [`SolveOutcome::resume`] will use.
    ///
    /// These are the time limit, node limit, and MIP gap from the immediately
    /// preceding solve, resume, or post-solve edit call.
    pub fn last_resume_options(&self) -> ResumeOptions {
        match self {
            Self::Solution(solution) => solution.last_options.clone(),
            Self::Interrupted(interrupted) => interrupted.last_options.clone(),
        }
    }

    /// Consumes this outcome and continues using the preceding call's settings.
    ///
    /// Time and node limits are applied as fresh budgets for this call. An
    /// already optimal outcome is returned unchanged.
    ///
    /// # Errors
    ///
    /// Returns errors encountered while continuing the solve, such as
    /// [`Error::Infeasible`], [`Error::Unbounded`], or
    /// [`Error::InternalError`].
    pub fn resume(self) -> Result<Self, Error> {
        let options = self.last_resume_options();
        self.resume_with(options)
    }

    /// Consumes this outcome and continues using the supplied settings.
    ///
    /// The fields replace, rather than merge with, the preceding call's
    /// settings. Time and node limits are fresh budgets for this call, and
    /// `None` makes either budget unlimited. A `mip_gap` of `None` requests
    /// exact optimality.
    ///
    /// An already optimal outcome is returned unchanged after validating
    /// `options`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOptions`] when a supplied value is non-finite or
    /// outside its accepted range, and otherwise the errors of
    /// [`SolveOutcome::resume`].
    pub fn resume_with(self, options: ResumeOptions) -> Result<Self, Error> {
        options.validate()?;
        if self.is_optimal() {
            return Ok(self);
        }
        let range = match &self {
            Self::Solution(solution) => solution.state.coefficient_range(),
            Self::Interrupted(interrupted) => interrupted.state.coefficient_range(),
        };
        self.resume_validated(options)
            .map_err(|error| range.annotate(error))
    }

    /// [`SolveOutcome::resume_with`] of an outcome that is not optimal, with
    /// validated `options`.
    fn resume_validated(self, options: ResumeOptions) -> Result<Self, Error> {
        let (direction, num_vars, state) = match self {
            Self::Solution(solution) => {
                debug_assert_eq!(solution.status, SolutionStatus::Feasible);
                (solution.direction, solution.num_vars, solution.state)
            }
            Self::Interrupted(interrupted) => (
                interrupted.direction,
                interrupted.num_vars,
                interrupted.state,
            ),
        };
        match state {
            SolveState::Lp(mut solver) => {
                solver.operation_time_limit = options.time_limit;
                let stop = timed_lp_call(&mut solver, options.time_limit, Solver::initial_solve)?;
                Self::from_lp_stop(direction, num_vars, stop, solver, options)
            }
            SolveState::Mip(mut state) => {
                let reason = mip::resume_run(&mut state, options.clone())?;
                Ok(Self::from_mip_run(
                    direction,
                    num_vars,
                    mip::MipRun {
                        reason,
                        state: *state,
                    },
                    options,
                ))
            }
        }
    }
}

/// An outcome returned when a limit is reached before a usable solution exists.
///
/// It has a termination reason and statistics, but no objective or variable
/// values because no validated assignment is available. It does not mean that
/// the problem is infeasible. Continue it through [`SolveOutcome::resume`] or
/// [`SolveOutcome::resume_with`].
#[derive(Clone)]
pub struct InterruptedSolve {
    pub(crate) direction: OptimizationDirection,
    pub(crate) num_vars: usize,
    pub(crate) termination_reason: TerminationReason,
    pub(crate) state: SolveState,
    pub(crate) last_options: ResumeOptions,
}

impl std::fmt::Debug for InterruptedSolve {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterruptedSolve")
            .field("direction", &self.direction)
            .field("num_vars", &self.num_vars)
            .field("termination_reason", &self.termination_reason)
            .field("stats", &self.stats())
            .finish()
    }
}

impl InterruptedSolve {
    /// Returns the limit that interrupted the most recent call.
    pub fn termination_reason(&self) -> TerminationReason {
        self.termination_reason
    }

    /// Returns statistics accumulated before the interruption.
    pub fn stats(&self) -> Stats {
        match &self.state {
            SolveState::Lp(solver) => Stats {
                lp_iterations: solver.lp_iterations,
                elapsed: solver.elapsed,
                ..Stats::default()
            },
            SolveState::Mip(state) => state.stats,
        }
    }

    /// Returns the settings used by the immediately preceding solve, resume, or
    /// post-solve edit call.
    pub fn last_resume_options(&self) -> ResumeOptions {
        self.last_options.clone()
    }
}
