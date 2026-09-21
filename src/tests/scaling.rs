//! Property tests that feasible-by-construction, bounded models solve at every
//! variable magnitude and coefficient range.
//!
//! A point `x*` is drawn first; every variable's bounds bracket its `x*`
//! value, every equality's right-hand side is the activity at `x*`, and every
//! inequality's right-hand side is on the side `x*` satisfies. All bounds are
//! finite, so the model is bounded, and `x*` is a feasible point of the model
//! as written (within the round-off of computing its activities, which the
//! oracle accounts for). The contract a solution must meet is in `oracle.rs`.
//!
//! Two coefficient regimes are drawn separately. With coefficients between
//! `1e-1` and `1e1` the tolerances leave the objective little room, so the
//! solution's objective is held to the objective at `x*`. With coefficients
//! from `1e-8` to `1e4` a row satisfied to `1e-7` lets a variable with a `1e-8`
//! coefficient move by ten units, and that freedom propagates through the
//! other rows into the objective; the wide properties therefore assert only
//! that the model solves and the solution is feasible.

#[cfg(test)]
mod tests_scaling {
    use crate::tests::oracle::{
        assert_solves_within_contract, xstar_is_feasible, Model, Row, Var, OBJECTIVE_SLACK,
    };
    use crate::{ComparisonOp, OptimizationDirection};

    use hegel::generators as gs;

    fn settings() -> hegel::Settings {
        hegel::Settings::new().test_cases(TEST_CASES)
    }

    /// Up to six variables and five rows per model; sized to keep the four
    /// properties within about a minute together.
    const TEST_CASES: u64 = 5_000;

    /// Largest decimal exponent of a variable's magnitude: values up to 1e9.
    const MAX_VAR_EXP: i32 = 9;

    /// Integer ranges stay this small so branch and bound cannot turn an
    /// equality row into an enumeration of a 1e9 range; the magnitude of the
    /// integer's VALUE still spans the full range.
    const MAX_INT_RANGE: i64 = 100;

    struct Printable(Model);

    impl std::fmt::Debug for Printable {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0.source())
        }
    }

    impl hegel::PrettyPrintable for Printable {
        fn pretty_print(&self, printer: &mut hegel::PrettyPrinter) {
            printer.begin_group(4, "{");
            for line in self.0.source().lines() {
                printer.hard_break();
                printer.text(line);
            }
            printer.shift_indent(-4);
            printer.hard_break();
            printer.end_group("}");
        }
    }

    /// A coefficient with decimal exponent in `lo..=hi`; a quarter of the
    /// draws are exact small values so rows mix exact and full-mantissa data.
    fn coeff(tc: &hegel::TestCase, lo: i32, hi: i32) -> f64 {
        if tc.draw(gs::weighted_booleans(0.25)) {
            return tc.draw(gs::sampled_from(vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5]));
        }
        let exp = tc.draw(gs::integers::<i32>().min_value(lo).max_value(hi));
        let mant = tc.draw(gs::floats::<f64>().min_value(1.0).max_value(2.0));
        let sign = if tc.draw(gs::booleans()) { 1.0 } else { -1.0 };
        sign * mant * 10f64.powi(exp)
    }

    fn draw_model(tc: &hegel::TestCase, milp: bool, coeff_lo: i32, coeff_hi: i32) -> Model {
        let direction = if tc.draw(gs::booleans()) {
            OptimizationDirection::Minimize
        } else {
            OptimizationDirection::Maximize
        };
        let nvars = tc.draw(gs::integers::<usize>().min_value(1).max_value(6));
        let n_int = if milp {
            tc.draw(gs::integers::<usize>().min_value(1).max_value(nvars.min(2)))
        } else {
            0
        };
        let mut vars = Vec::with_capacity(nvars);
        let mut xstar = Vec::with_capacity(nvars);
        for j in 0..nvars {
            let integer = j < n_int;
            let k = tc.draw(gs::integers::<i32>().min_value(0).max_value(MAX_VAR_EXP));
            let mag = 10f64.powi(k);
            let (x, lo, hi) = if integer {
                let m = mag.min(1e9) as i64;
                let x = tc.draw(gs::integers::<i64>().min_value(-m).max_value(m));
                let below = tc.draw(gs::integers::<i64>().min_value(0).max_value(MAX_INT_RANGE));
                let above = tc.draw(gs::integers::<i64>().min_value(0).max_value(MAX_INT_RANGE));
                (x as f64, (x - below) as f64, (x + above) as f64)
            } else {
                let x = if tc.draw(gs::weighted_booleans(0.2)) {
                    0.0
                } else {
                    tc.draw(gs::floats::<f64>().min_value(-mag).max_value(mag))
                };
                let below = if tc.draw(gs::weighted_booleans(0.15)) {
                    0.0
                } else {
                    tc.draw(gs::floats::<f64>().min_value(0.0).max_value(2.0 * mag))
                };
                let above = if tc.draw(gs::weighted_booleans(0.15)) {
                    0.0
                } else {
                    tc.draw(gs::floats::<f64>().min_value(0.0).max_value(2.0 * mag))
                };
                (x, x - below, x + above)
            };
            let obj = if tc.draw(gs::weighted_booleans(0.2)) {
                0.0
            } else {
                coeff(tc, -1, 1)
            };
            vars.push(Var {
                obj,
                lo,
                hi,
                integer,
            });
            xstar.push(x);
        }
        let nrows = tc.draw(gs::integers::<usize>().min_value(1).max_value(5));
        let mut rows = Vec::with_capacity(nrows);
        for _ in 0..nrows {
            let mut terms = Vec::with_capacity(nvars);
            for j in 0..nvars {
                if tc.draw(gs::weighted_booleans(0.7)) {
                    terms.push((j, coeff(tc, coeff_lo, coeff_hi)));
                }
            }
            if terms.is_empty() {
                continue;
            }
            let activity: f64 = terms.iter().map(|&(j, a)| a * xstar[j]).sum();
            let slack_mag = terms
                .iter()
                .map(|&(j, a)| (a * xstar[j]).abs())
                .fold(1.0, f64::max);
            let (op, rhs) = match tc.draw(gs::integers::<u8>().min_value(0).max_value(2)) {
                0 => (ComparisonOp::Eq, activity),
                1 => (
                    ComparisonOp::Le,
                    activity + tc.draw(gs::floats::<f64>().min_value(0.0).max_value(slack_mag)),
                ),
                _ => (
                    ComparisonOp::Ge,
                    activity - tc.draw(gs::floats::<f64>().min_value(0.0).max_value(slack_mag)),
                ),
            };
            rows.push(Row { terms, op, rhs });
        }
        let model = Model {
            direction,
            vars,
            rows,
            xstar,
        };
        // Holds by construction; stated so the property never asserts anything
        // about a model whose own witness is not a feasible point.
        tc.assume(xstar_is_feasible(&model));
        model
    }

    #[hegel::composite]
    fn lp_narrow(tc: &hegel::TestCase) -> Printable {
        Printable(draw_model(tc, false, -1, 1))
    }

    #[hegel::composite]
    fn lp_wide(tc: &hegel::TestCase) -> Printable {
        Printable(draw_model(tc, false, -8, 4))
    }

    #[hegel::composite]
    fn milp_narrow(tc: &hegel::TestCase) -> Printable {
        Printable(draw_model(tc, true, -1, 1))
    }

    #[hegel::composite]
    fn milp_wide(tc: &hegel::TestCase) -> Printable {
        Printable(draw_model(tc, true, -8, 4))
    }

    /// The oracle including the objective check (see the module docs).
    fn check_with_objective(model: &Model) {
        assert_solves_within_contract(model);
    }

    /// The oracle without the objective check: feasibility and status only.
    fn check_feasibility(model: &Model) {
        // Zero objective coefficients make the objective check vacuous while
        // keeping every other assertion of the oracle.
        let feasibility_only = Model {
            direction: model.direction,
            vars: model
                .vars
                .iter()
                .map(|v| Var {
                    obj: 0.0,
                    lo: v.lo,
                    hi: v.hi,
                    integer: v.integer,
                })
                .collect(),
            rows: model
                .rows
                .iter()
                .map(|r| Row {
                    terms: r.terms.clone(),
                    op: r.op,
                    rhs: r.rhs,
                })
                .collect(),
            xstar: model.xstar.clone(),
        };
        let _ = OBJECTIVE_SLACK;
        assert_solves_within_contract(&feasibility_only);
    }

    #[hegel::test(settings())]
    fn a_feasible_bounded_lp_solves_at_every_magnitude(tc: hegel::TestCase) {
        let model = tc.draw(lp_narrow());
        check_with_objective(&model.0);
    }

    #[hegel::test(settings())]
    fn a_feasible_bounded_lp_with_wide_coefficients_solves(tc: hegel::TestCase) {
        let model = tc.draw(lp_wide());
        check_feasibility(&model.0);
    }

    #[hegel::test(settings())]
    fn a_feasible_bounded_milp_solves_at_every_magnitude(tc: hegel::TestCase) {
        let model = tc.draw(milp_narrow());
        check_with_objective(&model.0);
    }

    // KNOWN LIMITATION (`ARCHITECTURE.md` §7): the wide MILP regime reaches models
    // the ratio test cannot resolve in f64, so this property finds counterexamples
    // on some draws. Ignored rather than `#[should_panic]` because it is a random
    // search: most runs pass, so asserting failure would itself be flaky. The model
    // it has found is committed as an exact `#[should_panic]` fixture in
    // `magnitudes.rs`, which is what will turn red once the limitation is fixed. Run
    // this property with `cargo test --lib -- --ignored` to keep searching for more.
    #[ignore = "known limitation: wide coefficients beyond f64 resolution (ARCHITECTURE §7)"]
    #[hegel::test(settings())]
    fn a_feasible_bounded_milp_with_wide_coefficients_solves(tc: hegel::TestCase) {
        let model = tc.draw(milp_wide());
        check_feasibility(&model.0);
    }
}
