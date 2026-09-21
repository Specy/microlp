//! The pure-LP path through the same public API: solving, stats, and the
//! time budget post-solve edits get.

use super::common::solution;
use crate::*;
use core::time::Duration;

#[test]
fn lp_path_still_solves_and_edits_incrementally() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_var(1.0, (0.0, 4.0));
    let y = p.add_var(2.0, (0.0, 3.0));
    p.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 5.0);
    let sol = solution(p.solve().unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    assert!((sol.objective() - 8.0).abs() < 1e-6);
    // Live-basis incremental add on the LP path.
    let sol = solution(
        sol.add_constraint(&[(x, 1.0)], ComparisonOp::Le, 1.0)
            .unwrap(),
    );
    assert!((sol.objective() - 7.0).abs() < 1e-6);
    assert!((sol[x] - 1.0).abs() < 1e-6);
}

#[test]
fn optimal_lp_stats_report_bound_and_zero_gap() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let _x = p.add_var(2.0, (0.0, 3.0));
    let sol = solution(p.solve().unwrap());
    let stats = sol.stats();
    assert_eq!(stats.best_bound, Some(6.0));
    assert_eq!(stats.gap, Some(0.0));
}

#[test]
fn lp_edit_uses_the_most_recent_resume_time_budget() {
    let mut p = Problem::new(OptimizationDirection::Maximize);
    let x = p.add_var(1.0, (0.0, 10.0));
    p.set_time_limit(Duration::ZERO);

    let interrupted = p.solve().unwrap();
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::TimeLimit
    );

    let resumed = solution(interrupted.resume_with(ResumeOptions::default()).unwrap());
    assert_eq!(resumed.objective(), 10.0);

    let edited = solution(
        resumed
            .add_constraint([(x, 1.0)], ComparisonOp::Le, 4.0)
            .unwrap(),
    );
    assert_eq!(edited.status(), SolutionStatus::Optimal);
    assert_eq!(edited.objective(), 4.0);
}

#[test]
fn lp_stats_report_nonzero_elapsed_time() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, 10.0));
    p.add_constraint([(x, 1.0)], ComparisonOp::Ge, 3.0);

    let sol = solution(p.solve().unwrap());
    let initial_elapsed = sol.stats().elapsed;
    assert!(initial_elapsed > Duration::ZERO);

    let edited = solution(
        sol.add_constraint([(x, 1.0)], ComparisonOp::Ge, 4.0)
            .unwrap(),
    );
    assert!(edited.stats().elapsed >= initial_elapsed);
}

#[test]
fn lp_edit_gets_a_fresh_time_budget() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(1.0, (0.0, 10.0));
    p.add_constraint([(x, 1.0)], ComparisonOp::Ge, 3.0);
    let mut options = SolveOptions::default();
    options.time_limit = Some(Duration::from_millis(50));

    let sol = solution(p.solve_with(options).unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    std::thread::sleep(Duration::from_millis(75));

    let edited = solution(
        sol.add_constraint([(x, 1.0)], ComparisonOp::Ge, 4.0)
            .unwrap(),
    );
    assert_eq!(edited.status(), SolutionStatus::Optimal);
    assert_eq!(edited.objective(), 4.0);
}

#[test]
fn lp_unfix_reports_an_interrupted_reoptimization() {
    let mut p = Problem::new(OptimizationDirection::Minimize);
    let x = p.add_var(-1.0, (0.0, 10.0));
    let solved = solution(p.solve().unwrap());
    let mut fixed = solution(solved.fix_var(x, 0.0).unwrap());
    match &mut fixed.state {
        SolveState::Lp(solver) => {
            solver.operation_time_limit = Some(Duration::ZERO);
        }
        SolveState::Mip(_) => unreachable!(),
    }

    let (unfixed, was_fixed) = fixed.unfix_var(x).unwrap();
    assert!(was_fixed);
    assert!(unfixed.solution().is_none());
    assert_eq!(unfixed.termination_reason(), TerminationReason::TimeLimit);
}

/// An interrupted pure-LP edit must resume to the *edited* model's optimum —
/// the same answer as solving that model from scratch. The model change is
/// durable and the cut-off reoptimization simply continues on resume. This
/// is the resume counterpart to `lp_unfix_reports_an_interrupted_reoptimization`,
/// which only checks that the interruption is reported.
#[test]
fn interrupted_lp_edit_resumes_to_the_edited_models_optimum() {
    // minimize x + 2y + 3z  s.t.  x + y + z >= 6,  each var in [0, 10].
    // The cheapest unit is x, so the base optimum is (6, 0, 0), objective 6.
    // build() is deterministic, so variable indices match across instances.
    let build = || {
        let mut p = Problem::new(OptimizationDirection::Minimize);
        let x = p.add_var(1.0, (0.0, 10.0));
        let y = p.add_var(2.0, (0.0, 10.0));
        let z = p.add_var(3.0, (0.0, 10.0));
        p.add_constraint([(x, 1.0), (y, 1.0), (z, 1.0)], ComparisonOp::Ge, 6.0);
        (p, x, y, z)
    };

    let (base, x, y, z) = build();
    let mut sol = solution(base.solve().unwrap());
    assert_eq!(sol.status(), SolutionStatus::Optimal);
    let base_obj = sol.objective();
    assert!((base_obj - 6.0).abs() < 1e-6);

    // Starve the next edit of time so its reoptimization is cut off at the
    // first deadline check, before any pivot: the row is appended but no
    // feasibility restoration runs.
    match &mut sol.state {
        SolveState::Lp(solver) => solver.operation_time_limit = Some(Duration::ZERO),
        SolveState::Mip(_) => unreachable!("a continuous model stays pure-LP"),
    }

    // `x <= 2` cuts off the incumbent x = 6, so the edit genuinely needs to
    // reoptimize — work that the zero budget defers entirely to resume.
    let interrupted = sol
        .add_constraint([(x, 1.0)], ComparisonOp::Le, 2.0)
        .unwrap();
    assert!(interrupted.solution().is_none());
    assert_eq!(
        interrupted.termination_reason(),
        TerminationReason::TimeLimit
    );

    // An ample resume budget finishes the edited solve.
    let resumed = solution(interrupted.resume_with(ResumeOptions::default()).unwrap());
    assert_eq!(resumed.status(), SolutionStatus::Optimal);

    // Oracle: the same edited model solved from scratch → (2, 4, 0), obj 10.
    let (mut edited, ..) = build();
    edited.add_constraint([(x, 1.0)], ComparisonOp::Le, 2.0);
    let fresh = solution(edited.solve().unwrap());
    assert_eq!(fresh.status(), SolutionStatus::Optimal);

    // Resume and the fresh solve agree on objective and the (unique) vertex,
    // and both differ from the pre-edit incumbent — proving resume ran the
    // reoptimization rather than returning stale state.
    assert!((resumed.objective() - fresh.objective()).abs() < 1e-6);
    assert!((resumed.objective() - 10.0).abs() < 1e-6);
    assert!(resumed.objective() > base_obj + 0.5);
    assert!(resumed.var_value(x) <= 2.0 + 1e-6);
    for var in [x, y, z] {
        assert!((resumed.var_value(var) - fresh.var_value(var)).abs() < 1e-6);
    }
}
