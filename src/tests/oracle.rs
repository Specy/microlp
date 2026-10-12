//! What a correct answer to a feasible, bounded model must satisfy, shared by
//! the property tests and the regression tests built from their counterexamples.
//!
//! The models are built around a point `x*` chosen first, an integral point
//! (every integer variable at an integer within its bounds, which presolve,
//! the search and the proofs keep: ARCHITECTURE §7.1), so `solve()` must
//! return a solution at least as good as `x*`. In the units the user sees:
//!
//! * A row holds within [`crate::Tolerances::feasibility`] (default `1e-7`),
//!   but never finer than the round-off of evaluating it, `ROUNDOFF`, a
//!   hundred ulps of the row's magnitude: a row whose terms are `1e10`
//!   cannot be checked below about `1e-6`. Its residual is summed in
//!   compensated arithmetic, so the verdict does not depend on the order of
//!   the terms.
//! * A variable is within its bounds by the same rule.
//! * An integer variable is within [`crate::SolveOptions::int_tol`] of an
//!   integer.
//! * The objective is not worse than at `x*` by more than `1e-4` of its extent
//!   over the variable box. The tolerances let every variable move a little,
//!   so this catches gross sub-optimality (a wrongly pruned branch, a wrong
//!   vertex), not tolerance-level differences.
//! * The reported objective is the objective of the reported values.
//!
//! A model outside the recommended coefficient range (`crate::range`) may
//! instead fail loudly, with an `InternalError` that names a constraint or a
//! variable outside it, but never with a wrong answer
//! ([`assert_no_wrong_answer`]).
//!
//! A MILP with no integral point with its fixed variables at their values
//! may be answered `Infeasible` or with a point the contract accepts, an
//! integer variable off its integer or a fixed variable off its value
//! ([`infeasible_or_within_contract`]).

use crate::range::CoefficientRange;
use crate::{
    ComparisonOp, Error, OptimizationDirection, Problem, SolutionStatus, SolveOptions,
    SolveOutcome, Tolerances, Variable,
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

/// `Σ a_j x_j - b`, summed with the products exact in compensated
/// arithmetic, and the magnitude `|b| + Σ |a_j x_j|` it was computed from.
fn activity(row: &Row, x: &[f64]) -> (f64, f64) {
    let mut act = crate::solver::CDouble::from(-row.rhs);
    let mut mag = row.rhs.abs();
    for &(j, a) in &row.terms {
        act.add_product(a, x[j]);
        mag += (a * x[j]).abs();
    }
    (act.value(), mag)
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

/// Whether `x*` is an integral point of `model` that holds its rows: every
/// integer var at an integer within its bounds, and every row within the
/// tolerance above. True by construction, and assumed by a generator before
/// drawing conclusions.
pub(crate) fn xstar_is_feasible(model: &Model) -> bool {
    let integral = model
        .vars
        .iter()
        .zip(&model.xstar)
        .all(|(v, &x)| !v.integer || (x == x.round() && v.lo <= x && x <= v.hi));
    integral && rows_hold(model, &model.xstar)
}

/// Whether every row of `model` holds at `x` within the tolerance above
/// (bounds and integrality not checked).
pub(crate) fn rows_hold(model: &Model, x: &[f64]) -> bool {
    model
        .rows
        .iter()
        .all(|row| row_violation(row, x) <= row_tolerance(row, x))
}

/// Whether `model` lies within the recommended coefficient range: the
/// nonzero coefficient magnitudes of every row, and of every column, within
/// six orders of each other.
pub(crate) fn within_recommended_range(model: &Model) -> bool {
    let (problem, _) = model.build();
    CoefficientRange::of(&problem.constraints).is_within()
}

/// The part of a loud error's message that says the model lies outside the
/// recommended coefficient range (see `CoefficientRange::annotate`).
const OUTSIDE_THE_RANGE: &str = "outside the recommended coefficient range";

/// Solve `model` with the default options and check the answer against the
/// contract above. Panics with the model's source on any failure.
pub(crate) fn assert_solves_within_contract(model: &Model) {
    check_contract(model, true);
}

/// [`assert_solves_feasibly_within_contract`] for a model outside the
/// recommended coefficient range: an `InternalError` that says so is no
/// failure, any other error is (the model is feasible and bounded), and an
/// answer is held to the contract.
pub(crate) fn assert_no_wrong_answer(model: &Model) {
    assert!(
        !within_recommended_range(model),
        "fixture error: the model lies within the recommended range"
    );
    check_contract_with(model, false, true, None, true);
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
    check_contract_with(model, true, false, None, false);
}

/// [`assert_solves_within_contract`] with presolve on and off, each solve
/// under `time_limit`, so that a simplex that cycles fails the test instead
/// of hanging it: reaching the limit is the failure.
pub(crate) fn assert_solves_within_contract_in_time(model: &Model, time_limit: Duration) {
    for presolve in [true, false] {
        check_contract_with(model, true, presolve, Some(time_limit), false);
    }
}

fn check_contract(model: &Model, compare_objective: bool) {
    check_contract_with(model, compare_objective, true, None, false);
}

/// The check of the module doc; `loud_allowed` accepts an `InternalError`
/// that says the model lies outside the recommended coefficient range.
fn check_contract_with(
    model: &Model,
    compare_objective: bool,
    presolve: bool,
    time_limit: Option<Duration>,
    loud_allowed: bool,
) {
    assert!(
        xstar_is_feasible(model),
        "fixture error: x* must be an integral point, its integer vars at integers within \
         their bounds, that holds every row:\n{}",
        model.source()
    );
    let (problem, vars) = model.build();
    let mut options = SolveOptions::default();
    options.presolve = presolve;
    options.time_limit = time_limit;
    let call = solve_call(presolve);
    let outcome = match problem.solve_with(options) {
        Ok(outcome) => outcome,
        Err(Error::InternalError(message))
            if loud_allowed && message.contains(OUTSIDE_THE_RANGE) =>
        {
            return;
        }
        Err(err) => panic!(
            "{call} returned {err:?} on a feasible, bounded model:\n{}",
            model.source()
        ),
    };
    let (_, objective) = assert_answer_within_contract(model, outcome, &vars, call, time_limit);
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
        "{call}: objective {objective:e} is worse than the objective {at_star:e} at the \
         feasible point the model was built around (slack {slack:e}):\n{}",
        model.source()
    );
}

/// The solve call a failure report names.
fn solve_call(presolve: bool) -> &'static str {
    if presolve {
        "solve()"
    } else {
        "solve() without presolve"
    }
}

/// The values and objective of the answer in `outcome`, held to the contract
/// of the module doc but for the comparison with `x*`.
fn assert_answer_within_contract(
    model: &Model,
    outcome: SolveOutcome,
    vars: &[Variable],
    call: &str,
    time_limit: Option<Duration>,
) -> (Vec<f64>, f64) {
    let solution = match outcome.into_solution() {
        Ok(solution) => solution,
        Err(interrupted) => panic!(
            "{call}: into_solution() returned {interrupted:?} (time limit {time_limit:?}):\n{}",
            model.source()
        ),
    };
    assert_eq!(
        solution.status(),
        SolutionStatus::Optimal,
        "{call}: no limit was reached, so the status must be Optimal:\n{}",
        model.source()
    );

    let int_tol = SolveOptions::default().int_tol;
    let x: Vec<f64> = vars.iter().map(|&v| solution.var_value(v)).collect();
    for (j, v) in model.vars.iter().enumerate() {
        let tol = bound_tolerance(v);
        assert!(
            x[j] >= v.lo - tol && x[j] <= v.hi + tol,
            "{call}: x{j} = {:e} is outside [{:e}, {:e}] (tolerance {tol:e}):\n{}",
            x[j],
            v.lo,
            v.hi,
            model.source()
        );
        let off = (x[j] - x[j].round()).abs();
        assert!(
            !v.integer || off <= int_tol,
            "{call}: x{j} = {:e} is {off:e} off its integer (tolerance {int_tol:e}):\n{}",
            x[j],
            model.source()
        );
    }
    for (i, row) in model.rows.iter().enumerate() {
        let violation = row_violation(row, &x);
        let tol = row_tolerance(row, &x);
        assert!(
            violation <= tol,
            "{call}: row {i} is violated by {violation:e} (tolerance {tol:e}) at x = {x:?}:\n{}",
            model.source()
        );
    }

    let objective = solution.objective();
    let recomputed = model.objective_at(&x);
    assert!(
        (objective - recomputed).abs() <= 1e-9 * (1.0 + objective.abs()),
        "{call}: reported objective {objective:e} does not match the reported values \
         ({recomputed:e}):\n{}",
        model.source()
    );
    (x, objective)
}

/// The asymmetric contract on a MILP with no integral point with its fixed
/// vars at their values (ARCHITECTURE §7.1): the solve with `presolve`
/// answers `Infeasible` (`None`), or a point the contract of the module doc
/// accepts, but for the comparison with `x*` (`Some`, its values). Any other
/// answer panics.
pub(crate) fn infeasible_or_within_contract(model: &Model, presolve: bool) -> Option<Vec<f64>> {
    let (problem, vars) = model.build();
    let options = SolveOptions {
        presolve,
        ..SolveOptions::default()
    };
    infeasible_or_answer_within_contract(
        model,
        problem.solve_with(options),
        &vars,
        solve_call(presolve),
    )
}

/// [`infeasible_or_within_contract`] on the `result` of `call`, such as an
/// edit, that `model` describes, `vars` being its vars as built.
pub(crate) fn infeasible_or_answer_within_contract(
    model: &Model,
    result: Result<SolveOutcome, Error>,
    vars: &[Variable],
    call: &str,
) -> Option<Vec<f64>> {
    match result {
        Err(Error::Infeasible) => None,
        Err(err) => panic!("{call} returned {err:?}:\n{}", model.source()),
        Ok(outcome) => Some(assert_answer_within_contract(model, outcome, vars, call, None).0),
    }
}
