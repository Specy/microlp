//! Independent validation of solver output against the shadow model, plus the
//! comparison of the outcome against the case's expected result.
//!
//! The point of this module is to never trust the solver's own bookkeeping:
//! the objective is recomputed from raw variable values, and feasibility,
//! bounds and integrality are re-checked against the shadow `ModelSpec`.

use crate::model::{Domain, Expected, ModelSpec};
use microlp::{ComparisonOp, Error, Solution, SolutionStatus, SolveOutcome};

/// Feasibility slack: a constraint `lhs (op) rhs` is accepted when violated by
/// no more than `FEAS_ABS * (1 + |rhs|)`, which scales gracefully for the
/// benchmark instances with rhs values in the 1e5 range.
const FEAS_ABS: f64 = 1e-6;
/// Integrality slack, matching `Solution::var_value`'s own rounding threshold.
const INT_ABS: f64 = 1e-5;
/// Slack for re-computed vs reported objective.
const OBJ_CONSISTENCY: f64 = 1e-6;

pub enum Outcome {
    Pass,
    Fail(String),
    Timeout,
}

pub fn check(
    spec: &ModelSpec,
    expected: &Expected,
    result: Result<SolveOutcome, Error>,
) -> Outcome {
    match result {
        Ok(SolveOutcome::Interrupted(_)) => Outcome::Timeout,
        Ok(SolveOutcome::Solution(sol)) => {
            // A solve that hit a limit (feasible-but-unproven or interrupted
            // before any incumbent) is a timeout for the suite's purposes —
            // the default tier expects every case to prove optimality.
            if sol.status() == SolutionStatus::Feasible {
                return Outcome::Timeout;
            }
            // `Solution::iter` yields variables in insertion order (0..n).
            let values: Vec<f64> = sol.iter().map(|(_, x)| x).collect();
            if values.len() != spec.vars.len() {
                return Outcome::Fail(format!(
                    "solution has {} variables, shadow model has {}",
                    values.len(),
                    spec.vars.len()
                ));
            }

            // Soundness first: whatever the expectation, a returned "optimal"
            // solution must actually be a feasible point of the model and the
            // reported objective must match the variable values.
            if let Err(msg) = validate_solution(spec, &values, sol.objective()) {
                return Outcome::Fail(msg);
            }

            match expected {
                Expected::Objective { value, tol, vars } => {
                    if !tol.matches(sol.objective(), *value) {
                        return Outcome::Fail(format!(
                            "expected objective {}, got {} (diff {:.3e})",
                            value,
                            sol.objective(),
                            (sol.objective() - value).abs()
                        ));
                    }
                    if let Some(expected_vars) = vars {
                        for (var, want) in expected_vars {
                            let got = values[var.idx()];
                            if (got - want).abs() > 1e-6 + 1e-6 * want.abs() {
                                return Outcome::Fail(format!(
                                    "unique optimum: expected x{} = {}, got {}",
                                    var.idx(),
                                    want,
                                    got
                                ));
                            }
                        }
                    }
                    Outcome::Pass
                }
                Expected::Infeasible => Outcome::Fail(format!(
                    "expected Infeasible, but solver returned a solution with objective {} \
                     that satisfies all constraints of the shadow model — either the solver \
                     or the suite's model of this case is wrong",
                    sol.objective()
                )),
                Expected::Unbounded => Outcome::Fail(format!(
                    "expected Unbounded, got a finite solution with objective {}",
                    sol.objective()
                )),
            }
        }
        Err(err) => match (expected, &err) {
            (Expected::Infeasible, Error::Infeasible) => Outcome::Pass,
            (Expected::Unbounded, Error::Unbounded) => Outcome::Pass,
            (Expected::Objective { value, .. }, _) => {
                Outcome::Fail(format!("expected objective {}, got error: {}", value, err))
            }
            (Expected::Infeasible, _) => {
                Outcome::Fail(format!("expected Infeasible, got error: {}", err))
            }
            (Expected::Unbounded, _) => {
                Outcome::Fail(format!("expected Unbounded, got error: {}", err))
            }
        },
    }
}

/// Extract a usable solution for a custom suite case.
pub fn require_solution(outcome: SolveOutcome, context: &str) -> Result<Solution, String> {
    outcome.into_solution().map_err(|interrupted| {
        format!(
            "{context}: no incumbent before {:?}",
            interrupted.termination_reason()
        )
    })
}

/// Validate a returned incumbent (an `Optimal` or `Feasible` solution) against
/// the shadow model, without comparing to any expected objective. Used by the
/// interrupt cases, which accept a clean feasible-but-unproven incumbent but
/// still insist it be a genuine feasible point of the model.
///
pub fn validate_incumbent(spec: &ModelSpec, sol: &Solution) -> Result<(), String> {
    let values: Vec<f64> = sol.iter().map(|(_, x)| x).collect();
    if values.len() != spec.vars.len() {
        return Err(format!(
            "solution has {} variables, shadow model has {}",
            values.len(),
            spec.vars.len()
        ));
    }
    validate_solution(spec, &values, sol.objective())
}

fn validate_solution(spec: &ModelSpec, values: &[f64], reported_obj: f64) -> Result<(), String> {
    // Bounds and integrality.
    for (i, (var, &x)) in spec.vars.iter().zip(values).enumerate() {
        if !x.is_finite() {
            return Err(format!("x{} is not finite: {}", i, x));
        }
        let slack = FEAS_ABS * (1.0 + var.min.abs().max(var.max.abs()).min(1e12));
        if x < var.min - slack || x > var.max + slack {
            return Err(format!(
                "x{} = {} violates bounds [{}, {}]",
                i, x, var.min, var.max
            ));
        }
        if var.domain == Domain::Integer && (x - x.round()).abs() > INT_ABS {
            return Err(format!("x{} = {} is not integral", i, x));
        }
    }

    // Constraints.
    for (ci, c) in spec.constraints.iter().enumerate() {
        let lhs: f64 = c.terms.iter().map(|(vi, coeff)| coeff * values[*vi]).sum();
        let slack = FEAS_ABS * (1.0 + c.rhs.abs());
        let ok = match c.op {
            ComparisonOp::Le => lhs <= c.rhs + slack,
            ComparisonOp::Ge => lhs >= c.rhs - slack,
            ComparisonOp::Eq => (lhs - c.rhs).abs() <= slack,
        };
        if !ok {
            return Err(format!(
                "constraint #{} violated: lhs = {}, should be {} {}",
                ci,
                lhs,
                match c.op {
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Ge => ">=",
                    ComparisonOp::Eq => "==",
                },
                c.rhs
            ));
        }
    }

    // Objective consistency: recompute from raw values.
    let computed: f64 = spec
        .vars
        .iter()
        .zip(values)
        .map(|(v, &x)| v.obj_coeff * x)
        .sum();
    let slack = OBJ_CONSISTENCY * (1.0 + computed.abs());
    if (computed - reported_obj).abs() > slack {
        return Err(format!(
            "reported objective {} disagrees with recomputed value {} from variable values",
            reported_obj, computed
        ));
    }

    Ok(())
}
