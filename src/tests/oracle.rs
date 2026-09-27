//! What a correct answer to a feasible, bounded model must satisfy, shared by
//! the property tests and the regression tests built from their counterexamples.
//!
//! The models are built around a point `x*` chosen first, so `solve()` must
//! return a solution at least as good as `x*`. In the units the user sees:
//!
//! * A row holds within [`crate::Tolerances::feasibility`] (default `1e-7`),
//!   but never finer than the round-off of evaluating it, `ROUNDOFF`, a
//!   hundred ulps of the row's magnitude: a row whose terms are `1e10`
//!   cannot be checked below about `1e-6`.
//! * A variable is within its bounds by the same rule.
//! * The objective is not worse than at `x*` by more than `1e-4` of its extent
//!   over the variable box. The tolerances let every variable move a little,
//!   so this catches gross sub-optimality (a wrongly pruned branch, a wrong
//!   vertex), not tolerance-level differences.
//! * The reported objective is the objective of the reported values.

use crate::{
    ComparisonOp, OptimizationDirection, Problem, SolutionStatus, SolveOptions, Tolerances,
    Variable,
};
use std::time::Duration;

/// Relative round-off allowance on a row's or bound's magnitude: about a
/// hundred ulps of `f64`.
pub(crate) const ROUNDOFF: f64 = 1e-14;

/// Slack on the objective, relative to its extent over the variable box.
pub(crate) const OBJECTIVE_SLACK: f64 = 1e-4;

pub(crate) struct Row {
    pub terms: Vec<(usize, f64)>,
    pub op: ComparisonOp,
    pub rhs: f64,
}

pub(crate) struct Var {
    pub obj: f64,
    pub lo: f64,
    pub hi: f64,
    pub integer: bool,
}

/// The model as data plus the point it was built around.
pub(crate) struct Model {
    pub direction: OptimizationDirection,
    pub vars: Vec<Var>,
    pub rows: Vec<Row>,
    pub xstar: Vec<f64>,
}

impl Model {
    pub(crate) fn build(&self) -> (Problem, Vec<Variable>) {
        let mut problem = Problem::new(self.direction);
        let vars: Vec<Variable> = self
            .vars
            .iter()
            .map(|v| {
                if v.integer {
                    problem.add_integer_var(v.obj, (v.lo as i32, v.hi as i32))
                } else {
                    problem.add_var(v.obj, (v.lo, v.hi))
                }
            })
            .collect();
        for row in &self.rows {
            let expr: Vec<(Variable, f64)> = row.terms.iter().map(|&(j, a)| (vars[j], a)).collect();
            problem.add_constraint(expr, row.op, row.rhs);
        }
        (problem, vars)
    }

    pub(crate) fn objective_at(&self, x: &[f64]) -> f64 {
        self.vars.iter().zip(x).map(|(v, &xj)| v.obj * xj).sum()
    }

    /// The Rust that rebuilds the model, for failure reports.
    pub(crate) fn source(&self) -> String {
        let mut lines = vec![format!(
            "let mut problem = Problem::new(OptimizationDirection::{:?});",
            self.direction
        )];
        for (j, v) in self.vars.iter().enumerate() {
            if v.integer {
                lines.push(format!(
                    "let x{j} = problem.add_integer_var({:?}, ({}, {}));",
                    v.obj, v.lo as i32, v.hi as i32
                ));
            } else {
                lines.push(format!(
                    "let x{j} = problem.add_var({:?}, ({:?}, {:?}));",
                    v.obj, v.lo, v.hi
                ));
            }
        }
        for row in &self.rows {
            let terms: Vec<String> = row
                .terms
                .iter()
                .map(|(j, a)| format!("(x{j}, {a:?})"))
                .collect();
            lines.push(format!(
                "problem.add_constraint([{}], ComparisonOp::{:?}, {:?});",
                terms.join(", "),
                row.op,
                row.rhs
            ));
        }
        lines.push(format!("// x* = {:?}", self.xstar));
        lines.join("\n")
    }
}

/// `Σ a_j x_j - b` and the magnitude `|b| + Σ |a_j x_j|` it was computed from.
fn activity(row: &Row, x: &[f64]) -> (f64, f64) {
    let mut act = 0.0;
    let mut mag = row.rhs.abs();
    for &(j, a) in &row.terms {
        act += a * x[j];
        mag += (a * x[j]).abs();
    }
    (act - row.rhs, mag)
}

/// How far `x` violates `row` (0 when satisfied, infinite when the activity
/// is not a number, which `f64::max` would otherwise turn into 0).
pub(crate) fn row_violation(row: &Row, x: &[f64]) -> f64 {
    let (diff, _) = activity(row, x);
    if diff.is_nan() {
        return f64::INFINITY;
    }
    match row.op {
        ComparisonOp::Eq => diff.abs(),
        ComparisonOp::Le => diff.max(0.0),
        ComparisonOp::Ge => (-diff).max(0.0),
    }
}

/// The largest violation of `row` that still counts as satisfied at `x`.
pub(crate) fn row_tolerance(row: &Row, x: &[f64]) -> f64 {
    let (_, mag) = activity(row, x);
    Tolerances::default().feasibility.max(ROUNDOFF * mag)
}

fn bound_tolerance(v: &Var) -> f64 {
    let mag = [v.lo.abs(), v.hi.abs()]
        .into_iter()
        .filter(|m| m.is_finite())
        .fold(0.0, f64::max);
    Tolerances::default().feasibility.max(ROUNDOFF * mag)
}

/// Whether `x*` satisfies every row of `model` within the tolerance above:
/// true by construction, and assumed by a generator before drawing conclusions.
pub(crate) fn xstar_is_feasible(model: &Model) -> bool {
    model
        .rows
        .iter()
        .all(|row| row_violation(row, &model.xstar) <= row_tolerance(row, &model.xstar))
}

/// Solve `model` with the default options and check the answer against the
/// contract above. Panics with the model's source on any failure.
pub(crate) fn assert_solves_within_contract(model: &Model) {
    check_contract(model, true);
}

/// [`assert_solves_within_contract`] without the comparison with the
/// objective at `x*`, for models whose tolerances leave the objective real
/// freedom (a tiny coefficient lets a var move far within its row's
/// tolerance). The model is still solved with its objective.
pub(crate) fn assert_solves_feasibly_within_contract(model: &Model) {
    check_contract(model, false);
}

/// [`assert_solves_within_contract`] with presolve off: the LP engine on the
/// model as written.
pub(crate) fn assert_solves_within_contract_without_presolve(model: &Model) {
    check_contract_with(model, true, false, None);
}

/// [`assert_solves_within_contract`] with presolve on and off, each solve
/// under `time_limit`, so that a simplex that cycles fails the test instead
/// of hanging it: reaching the limit is the failure.
pub(crate) fn assert_solves_within_contract_in_time(model: &Model, time_limit: Duration) {
    for presolve in [true, false] {
        check_contract_with(model, true, presolve, Some(time_limit));
    }
}

fn check_contract(model: &Model, compare_objective: bool) {
    check_contract_with(model, compare_objective, true, None);
}

fn check_contract_with(
    model: &Model,
    compare_objective: bool,
    presolve: bool,
    time_limit: Option<Duration>,
) {
    let (problem, vars) = model.build();
    let mut options = SolveOptions::default();
    options.presolve = presolve;
    options.time_limit = time_limit;
    let call = if presolve {
        "solve()"
    } else {
        "solve() without presolve"
    };
    let outcome = match problem.solve_with(options) {
        Ok(outcome) => outcome,
        Err(err) => panic!(
            "{call} returned {err:?} on a feasible, bounded model:\n{}",
            model.source()
        ),
    };
    let solution = match outcome.into_solution() {
        Ok(solution) => solution,
        Err(interrupted) => panic!(
            "into_solution() returned {interrupted:?} (time limit {time_limit:?}):\n{}",
            model.source()
        ),
    };
    assert_eq!(
        solution.status(),
        SolutionStatus::Optimal,
        "no limit was reached, so the status must be Optimal:\n{}",
        model.source()
    );

    let x: Vec<f64> = vars.iter().map(|&v| solution.var_value(v)).collect();
    for (j, v) in model.vars.iter().enumerate() {
        let tol = bound_tolerance(v);
        assert!(
            x[j] >= v.lo - tol && x[j] <= v.hi + tol,
            "x{j} = {:e} is outside [{:e}, {:e}] (tolerance {tol:e}):\n{}",
            x[j],
            v.lo,
            v.hi,
            model.source()
        );
    }
    for (i, row) in model.rows.iter().enumerate() {
        let violation = row_violation(row, &x);
        let tol = row_tolerance(row, &x);
        assert!(
            violation <= tol,
            "row {i} is violated by {violation:e} (tolerance {tol:e}) at x = {x:?}:\n{}",
            model.source()
        );
    }

    let objective = solution.objective();
    let recomputed = model.objective_at(&x);
    assert!(
        (objective - recomputed).abs() <= 1e-9 * (1.0 + objective.abs()),
        "reported objective {objective:e} does not match the reported values ({recomputed:e}):\n{}",
        model.source()
    );
    if !compare_objective {
        return;
    }
    let at_star = model.objective_at(&model.xstar);
    let extent: f64 = model
        .vars
        .iter()
        .zip(&model.xstar)
        .map(|(v, &xj)| v.obj.abs() * (xj.abs() + (v.hi - v.lo)))
        .sum();
    let slack = OBJECTIVE_SLACK * (1.0 + extent);
    let worse = match model.direction {
        OptimizationDirection::Minimize => objective > at_star + slack,
        OptimizationDirection::Maximize => objective < at_star - slack,
    };
    assert!(
        !worse,
        "objective {objective:e} is worse than the objective {at_star:e} at the feasible \
         point the model was built around (slack {slack:e}):\n{}",
        model.source()
    );
}
