//! A property test that `Problem::solve` returns a solution for every model
//! whose origin is feasible and whose bounds are all finite.
//!
//! The models are small and degenerate in the way issue #44's is: one to
//! three real variables bounded by small integers either side of zero, an
//! optional integer variable fixed at zero, and up to two rows whose
//! coefficients range from 1e-8 to 2e4. The origin satisfies every bound and
//! every row, and every bound is finite, so any `Err` from `solve` is a wrong
//! answer.

#[cfg(test)]
mod tests_feasibility {
    use crate::*;

    use hegel::generators as gs;

    /// Half a million cases, sized for the search rather than for anything in
    /// microlp's contract: the numerical states this property reaches are
    /// thin, and shallower runs mostly miss them. The ceiling is memory: hegel
    /// keeps about 7.7 KB per case until the run ends (the solves themselves
    /// keep nothing), so a run peaks near 3.9 GB, and 4.7 GB in CI's
    /// `RUST_LOG=trace` step, which also holds the captured logs. A million
    /// cases needed 7.7 GB, and CI's macOS runner has 7 GB in all.
    fn settings() -> hegel::Settings {
        hegel::Settings::new().test_cases(500_000)
    }

    /// Indices are positions in `Model::reals`, so a row never mentions the
    /// integer variable, which is also true of the rows in issue #44.
    type Row = (Vec<(usize, f64)>, ComparisonOp, f64);

    /// The drawn model as data rather than a `Problem`, so a counterexample can
    /// be printed as the code that rebuilds it.
    #[derive(Debug)]
    struct Model {
        direction: OptimizationDirection,
        reals: Vec<(f64, (f64, f64))>,
        integer: Option<(f64, (i32, i32))>,
        rows: Vec<Row>,
    }

    /// Prints the Rust that rebuilds the model, so a counterexample pastes into
    /// a plain `#[test]`.
    impl hegel::PrettyPrintable for Model {
        fn pretty_print(&self, printer: &mut hegel::PrettyPrinter) {
            let mut lines = vec![format!(
                "let mut problem = Problem::new(OptimizationDirection::{:?});",
                self.direction
            )];
            for (i, (obj, (lo, hi))) in self.reals.iter().enumerate() {
                lines.push(format!(
                    "let x{i} = problem.add_var({obj:?}, ({lo:?}, {hi:?}));"
                ));
            }
            if let Some((obj, (lo, hi))) = self.integer {
                lines.push(format!(
                    "problem.add_integer_var({obj:?}, ({lo:?}, {hi:?}));"
                ));
            }
            for (terms, op, rhs) in &self.rows {
                let terms: Vec<String> = terms
                    .iter()
                    .map(|(i, c)| format!("(x{i}, {c:?})"))
                    .collect();
                lines.push(format!(
                    "problem.add_constraint([{}], ComparisonOp::{op:?}, {rhs:?});",
                    terms.join(", ")
                ));
            }
            lines.push("problem".to_string());

            printer.begin_group(4, "{");
            for line in &lines {
                printer.hard_break();
                printer.text(line);
            }
            printer.shift_indent(-4);
            printer.hard_break();
            printer.end_group("}");
        }
    }

    fn build(model: &Model) -> Problem {
        let mut problem = Problem::new(model.direction);
        let vars: Vec<Variable> = model
            .reals
            .iter()
            .map(|&(obj, bounds)| problem.add_var(obj, bounds))
            .collect();
        if let Some((obj, bounds)) = model.integer {
            problem.add_integer_var(obj, bounds);
        }
        for (terms, op, rhs) in &model.rows {
            let expr: Vec<(Variable, f64)> = terms.iter().map(|&(i, c)| (vars[i], c)).collect();
            problem.add_constraint(expr, *op, *rhs);
        }
        problem
    }

    /// A coefficient whose decimal exponent lies in `lo..=hi`. Issue #44's
    /// model pairs an exact `-1` with a full-mantissa `1.19e-8`, and
    /// full-mantissa draws alone do not produce such pairs, so a quarter of
    /// the draws are exact small values.
    fn coeff(tc: &hegel::TestCase, lo: i32, hi: i32) -> f64 {
        if tc.draw(gs::weighted_booleans(0.25)) {
            return tc.draw(gs::sampled_from(vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5]));
        }
        let exp = tc.draw(gs::integers::<i32>().min_value(lo).max_value(hi));
        let mant = tc.draw(gs::floats::<f64>().min_value(1.0).max_value(2.0));
        let sign = if tc.draw(gs::booleans()) { 1.0 } else { -1.0 };
        sign * mant * 10f64.powi(exp)
    }

    #[hegel::composite]
    fn origin_feasible_models(tc: &hegel::TestCase) -> Model {
        let direction = if tc.draw(gs::booleans()) {
            OptimizationDirection::Minimize
        } else {
            OptimizationDirection::Maximize
        };

        // One to three variables, each bounded by small integers either side
        // of zero, so a variable fixed at zero by its own bounds is common,
        // and so is a model whose only feasible point is the origin, as in
        // #44.
        let nvars = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let mut reals = Vec::with_capacity(nvars);
        for _ in 0..nvars {
            let lo = -(tc.draw(gs::integers::<i32>().min_value(0).max_value(2)) as f64);
            let hi = tc.draw(gs::integers::<i32>().min_value(0).max_value(2)) as f64;
            let obj = coeff(tc, -1, 1);
            reals.push((obj, (lo, hi)));
        }

        // Fixed at zero and absent from every row, the integer variable is
        // there to put the solve on the MIP path rather than the pure-LP one,
        // which is where #44 failed. Its objective coefficient is drawn but
        // cannot matter.
        let integer = if tc.draw(gs::booleans()) {
            Some((coeff(tc, -1, 1), (0, 0)))
        } else {
            None
        };

        // One or two rows are drawn. A row that ends up with no terms is
        // dropped, so a model can have fewer, or none.
        let nrows = tc.draw(gs::integers::<usize>().min_value(1).max_value(2));
        let mut rows = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            // This row's exponent window for `coeff`. At `base == -8` it runs
            // to 1, so one row can hold a coefficient near 1e-8 beside one
            // near 10, and at `base == 3` it is two decades wide.
            let base = tc.draw(gs::integers::<i32>().min_value(-8).max_value(3));
            let top = (base + 9).min(4);
            let mut terms = Vec::with_capacity(nvars);
            for i in 0..nvars {
                // Each variable enters with probability 0.75, so rows are
                // often shorter than the variable list.
                if tc.draw(gs::weighted_booleans(0.75)) {
                    terms.push((i, coeff(tc, base, top)));
                }
            }
            if terms.is_empty() {
                continue;
            }
            let (op, rhs) = match tc.draw(gs::integers::<u8>().min_value(0).max_value(2)) {
                0 => (ComparisonOp::Eq, 0.0),
                1 => (
                    ComparisonOp::Le,
                    tc.draw(gs::floats::<f64>().min_value(0.0).max_value(10.0)),
                ),
                _ => (
                    ComparisonOp::Ge,
                    -tc.draw(gs::floats::<f64>().min_value(0.0).max_value(10.0)),
                ),
            };
            rows.push((terms, op, rhs));
        }

        Model {
            direction,
            reals,
            integer,
            rows,
        }
    }

    /// `Problem::solve` documents `Infeasible` for when no assignment satisfies
    /// all bounds and constraints, `Unbounded` for when the objective has no
    /// finite optimum, and `InternalError` for when the solve cannot continue.
    /// Every bound here brackets zero, every `Eq` row has rhs 0, and each
    /// inequality's rhs is on the side the origin satisfies, so the origin
    /// satisfies every bound and every row. Every bound is finite, so no
    /// objective improves without limit. Nothing in the model is invalid, so
    /// there is nothing for the solve to stop on. None of the three errors is
    /// a correct answer, which is why the property needs no tolerance and no
    /// reference solver.
    #[hegel::test(settings())]
    fn an_origin_feasible_bounded_model_solves(tc: hegel::TestCase) {
        let model = tc.draw(origin_feasible_models());
        let outcome = match build(&model).solve() {
            Ok(outcome) => outcome,
            Err(err) => panic!("solve() returned {err:?} on a feasible, bounded model"),
        };
        if let Err(err) = outcome.into_solution() {
            panic!("into_solution() returned {err:?} with no time or node limit set");
        }
    }

    /// The first counterexample the property above found. `x0` and `x2` are
    /// fixed at zero by their bounds, so the first row reads `-0.001 x1 = 0`
    /// and the only feasible point is the origin. The solve reports
    /// `InternalError`; a correct answer is a solution whose rows hold within
    /// the documented feasibility tolerance (which here allows `|x1|` up to
    /// `1e-4`).
    #[test]
    fn property_counterexample_with_a_guard_tolerance_below_eps() {
        use crate::tests::oracle::{assert_solves_within_contract, Model, Row, Var};
        let real = |obj, lo, hi| Var {
            obj,
            lo,
            hi,
            integer: false,
        };
        assert_solves_within_contract(&Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -0.0, 0.0),
                real(-1.0, -1.0, 0.0),
                real(-1.0, -0.0, 0.0),
                Var {
                    obj: 1.0,
                    lo: 0.0,
                    hi: 0.0,
                    integer: true,
                },
            ],
            rows: vec![
                Row {
                    terms: vec![(0, -10000.0), (1, -0.001)],
                    op: ComparisonOp::Eq,
                    rhs: 0.0,
                },
                Row {
                    terms: vec![(1, -10000.0)],
                    op: ComparisonOp::Le,
                    rhs: 2.0,
                },
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0],
        });
    }

    /// The second counterexample. Only the origin is feasible; the row
    /// `-20000 x0 + 0.0001 x2 = 0` pairs coefficients eight orders of
    /// magnitude apart, and the solve answers `Infeasible`.
    #[test]
    fn property_counterexample_with_a_genuine_sub_eps_pivot() {
        use crate::tests::oracle::{assert_solves_within_contract, Model, Row, Var};
        let real = |obj, lo, hi| Var {
            obj,
            lo,
            hi,
            integer: false,
        };
        assert_solves_within_contract(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.0, -1.0, 0.0),
                real(-1.0, -0.0, 0.0),
                real(1.0, -2.0, 0.0),
            ],
            rows: vec![
                Row {
                    terms: vec![(0, -1.0), (1, -100.0)],
                    op: ComparisonOp::Eq,
                    rhs: 0.0,
                },
                Row {
                    terms: vec![(0, -20000.0), (2, 0.0001)],
                    op: ComparisonOp::Eq,
                    rhs: 0.0,
                },
            ],
            xstar: vec![0.0, 0.0, 0.0],
        });
    }

    /// The counterexample CI's derandomized run of the property finds (hegel
    /// seeds a CI run from the test's name). The primal ratio test left x2's
    /// row out of its step bound because x2's entry, 1e-7 of the column's
    /// largest, fell an ulp under the relative pivot floor; the step walked
    /// the FIXED x2 to 1.25e-8, twenty-five times its tolerance, the phase
    /// exit rejected the point, the dual phase pivoted back on that very
    /// entry, and after eight rounds of this the solve gave up with
    /// `InternalError`. An entry too small to pivot on still bounds the step.
    #[test]
    fn property_counterexample_with_a_blocking_entry_below_the_pivot_floor() {
        use crate::tests::oracle::{assert_solves_within_contract, Model, Row, Var};
        let real = |obj, lo, hi| Var {
            obj,
            lo,
            hi,
            integer: false,
        };
        assert_solves_within_contract(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.0, -0.0, 1.0),
                real(-1.0, -1.0, 1.0),
                real(-1.0, -0.0, 0.0),
            ],
            rows: vec![
                Row {
                    terms: vec![(0, 10.0), (1, 1.2499999999999999e-6), (2, -100.0)],
                    op: ComparisonOp::Eq,
                    rhs: 0.0,
                },
                Row {
                    terms: vec![(0, -1.0), (1, -10000.0)],
                    op: ComparisonOp::Le,
                    rhs: 0.0,
                },
            ],
            xstar: vec![0.0, 0.0, 0.0],
        });
    }

    /// The counterexample CI's run of the property found on the first version
    /// of the primal simplex's bound shifting, which took a step backwards
    /// where the leaving var sat past its bound, as HiGHS does. The dual phase
    /// leaves `x1` basic `7.5e-15` past its bound, the round-off of a step
    /// of `128.0000128`; the primal step that set it onto its bound moved the
    /// entering `x0` by that over the element `5.12e-5`, to `1.5e-10` past
    /// its own bound, beyond its tolerance of `1e-10` (what `-100 x0 - x1 <= 0`
    /// can absorb); the dual phase moved `x0` back out, and the phases
    /// alternated until they gave up with `InternalError`. A step that would
    /// run backwards is zero, and `x1` leaves where it sits, within its
    /// tolerance of its bound.
    #[test]
    fn property_counterexample_where_a_step_backwards_amplified_round_off() {
        use crate::tests::oracle::{assert_solves_within_contract, Model, Row, Var};
        let real = |obj, lo, hi| Var {
            obj,
            lo,
            hi,
            integer: false,
        };
        assert_solves_within_contract(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![real(-1.0, -0.0, 1.0), real(1.0, -1.0, 0.0)],
            rows: vec![
                Row {
                    terms: vec![(0, -100.0), (1, -1.0)],
                    op: ComparisonOp::Le,
                    rhs: 0.0,
                },
                Row {
                    terms: vec![(0, -0.001), (1, 10000.0)],
                    op: ComparisonOp::Ge,
                    rhs: -0.0,
                },
            ],
            xstar: vec![0.0, 0.0],
        });
    }

    /// The third counterexample, and the one that showed a continuous var's
    /// bound tolerance had to be capped by its rows just as an integer var's
    /// already was (see [`crate::solver::structural_tol`]).
    ///
    /// The engine used to return `x0 = -5.0000000000000004e-8`. That is inside
    /// `x0`'s bound tolerance — its lower bound is `-0.0` and a continuous var
    /// was allowed the full `1e-7` there — but row 1 is `-2 x0 <= 0`, so the
    /// slack reached the row multiplied by two, as `1.0000000000000001e-7`
    /// against a row tolerance of exactly `1e-7`. Over by one ulp, and the
    /// acceptance guard rejected the engine's own answer as an `InternalError`.
    /// Now `x0`'s tolerance is capped at the share of the row's budget its
    /// coefficient can carry, so the point is never produced in the first place.
    #[test]
    fn property_counterexample_with_bound_slack_one_ulp_over_the_row_budget() {
        use crate::tests::oracle::{assert_solves_within_contract, Model, Row, Var};
        let real = |obj, lo, hi| Var {
            obj,
            lo,
            hi,
            integer: false,
        };
        assert_solves_within_contract(&Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![real(1.0, -0.0, 1.0), real(-1.0, -1.0, 1.0)],
            rows: vec![
                Row {
                    terms: vec![(0, -200.0), (1, 1e-5)],
                    op: ComparisonOp::Eq,
                    rhs: 0.0,
                },
                Row {
                    terms: vec![(0, -2.0)],
                    op: ComparisonOp::Le,
                    rhs: 0.0,
                },
            ],
            xstar: vec![0.0, 0.0],
        });
    }
}
