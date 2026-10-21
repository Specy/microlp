//! A property test that `Problem::solve` returns a solution for every model
//! whose origin is feasible and whose bounds are all finite: any `Err` is a
//! wrong answer. The models are small and degenerate like issue #44's:
//! one to three real variables bounded by small integers either side of zero,
//! an optional integer variable fixed at zero, and up to two rows with
//! coefficients from 1e-8 to 2e4.

#[cfg(test)]
mod tests_feasibility {
    use crate::*;

    use hegel::generators as gs;

    /// Keep the default small enough for CI, including captured traces.
    /// Set HEGEL_TEST_CASES for a deeper run without editing the source.
    fn settings() -> hegel::Settings {
        hegel::Settings::new().test_cases(50_000)
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

    /// A coefficient whose decimal exponent lies in `lo..=hi`. A quarter of
    /// the draws are exact small values, so that rows pair exact and
    /// full-mantissa data as issue #44's does (`-1` beside `1.19e-8`).
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

        // Bounds of small integers either side of zero, so that variables
        // fixed at zero and models feasible only at the origin (as in #44)
        // are common.
        let nvars = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
        let mut reals = Vec::with_capacity(nvars);
        for _ in 0..nvars {
            let lo = -(tc.draw(gs::integers::<i32>().min_value(0).max_value(2)) as f64);
            let hi = tc.draw(gs::integers::<i32>().min_value(0).max_value(2)) as f64;
            let obj = coeff(tc, -1, 1);
            reals.push((obj, (lo, hi)));
        }

        // Fixed at zero and in no row, the integer variable puts the solve on
        // the MIP path, where #44 failed.
        let integer = if tc.draw(gs::booleans()) {
            Some((coeff(tc, -1, 1), (0, 0)))
        } else {
            None
        };

        // A drawn row that ends up with no terms is dropped.
        let nrows = tc.draw(gs::integers::<usize>().min_value(1).max_value(2));
        let mut rows = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            // This row's exponent window: at `base == -8` it runs to 1 (1e-8
            // beside 10 in one row), and at `base == 3` it is two decades wide.
            let base = tc.draw(gs::integers::<i32>().min_value(-8).max_value(3));
            let top = (base + 9).min(4);
            let mut terms = Vec::with_capacity(nvars);
            for i in 0..nvars {
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

    /// The origin satisfies every bound and row (bounds bracket zero, `Eq`
    /// rows have rhs 0, inequalities hold at zero), every bound is finite and
    /// nothing is invalid, so neither `Infeasible`, `Unbounded` nor
    /// `InternalError` is a correct answer: the property needs no tolerance
    /// and no known optimum.
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

    /// A counterexample of the property above: `x0` and `x2` are fixed at
    /// zero, so the first row reads `-0.001 x1 = 0` and only the origin is
    /// feasible. The answer is a solution whose rows hold within the
    /// feasibility tolerance (here `|x1|` up to `1e-4`), not `InternalError`.
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

    /// A counterexample feasible only at the origin, whose row
    /// `-20000 x0 + 0.0001 x2 = 0` pairs coefficients eight orders of
    /// magnitude apart; the solve answered `Infeasible`.
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

    /// An entry small beside its column still bounds the primal step: x2's
    /// entry, 1e-7 of the column's largest, fell an ulp under a pivot floor
    /// relative to that largest, and leaving its row out of the step bound
    /// walked the fixed x2 to 25 times its tolerance, after which the phases
    /// alternated until `InternalError`.
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

    /// A primal step that would run backwards is zero, and the leaving var
    /// leaves where it sits. Here the dual phase leaves `x1` basic `7.5e-15`
    /// past its bound; a step setting it onto its bound moves `x0` by that
    /// over the element `5.12e-5`, to `1.5e-10` past its own bound, beyond
    /// its tolerance of `1e-10` (what `-100 x0 - x1 <= 0` can absorb), and
    /// the phases alternated until `InternalError`.
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

    /// A continuous var's bound slack reaches its rows through their
    /// coefficients: `x0` may sit up to its tolerance below its bound of
    /// `-0.0`, where row 1, `-2 x0 <= 0`, sees that doubled. The engine
    /// verifies each row at the values it reports, so the answer must meet
    /// the row as well as the bound.
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
