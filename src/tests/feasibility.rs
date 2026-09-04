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

    /// A million cases, sized for the search rather than for anything in
    /// microlp's contract: the numerical states this property reaches are
    /// thin, and shallower runs mostly miss them.
    fn settings() -> hegel::Settings {
        hegel::Settings::new().test_cases(1_000_000)
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
}
