//! Regressions found by fuzzing feasible-by-construction models across variable
//! magnitudes and coefficient ranges (see `scaling.rs` for the properties).
//! Each model is written around a point `x*` that satisfies every bound and
//! every row within the documented tolerance, and every bound is finite, so
//! `solve()` must return an optimal solution whose rows hold within that
//! tolerance and whose objective is no worse than at `x*` — the contract in
//! `oracle.rs`.
//!
//! A model the engine gets wrong when it is found, and that cannot be fixed
//! at once, is pinned here as a KNOWN LIMITATION with `#[should_panic]`: the
//! test records today's answer, so that fixing it shows up as a FAILING test
//! rather than passing unnoticed; the attribute is then deleted (the body
//! already asserts the right answer) and it is an ordinary regression test.
//! See §7 of `ARCHITECTURE.md`. Five fixtures were pinned this way and all
//! five were fixed: a presolve forcing rule, a skipped verification, a dual
//! ratio test that took "no pivot" for a proof of infeasibility (the
//! `wide_chain` fixtures), the row tolerance a var left behind at a huge
//! bound, and a fixed var's huge coefficient in a row added after the fix.
//! Twelve are pinned today: a forcing row whose fixing another row can tell
//! apart, two models of the phases alternating over a row repaired through
//! round-off, a vertex off its bound by round-off with nothing to move, a
//! MILP node whose feasible LP no var can repair, three models whose only
//! room is a coefficient of round-off size (an LP, its MILP and the LP with
//! that var unbounded), and four whose only room is such a coefficient of
//! a basic var (an LP and its MILP, with presolve on and off), all reported
//! infeasible before and failing loudly now.

#[cfg(test)]
mod regression_tests {
    use crate::tests::oracle::{
        assert_solves_feasibly_within_contract, assert_solves_within_contract,
        assert_solves_within_contract_in_time, assert_solves_within_contract_without_presolve,
        xstar_is_feasible, Model, Row, Var,
    };
    use crate::{ComparisonOp, Error, OptimizationDirection, SolveOptions};
    use std::time::Duration;

    fn real(obj: f64, lo: f64, hi: f64) -> Var {
        Var {
            obj,
            lo,
            hi,
            integer: false,
        }
    }

    fn int(obj: f64, lo: i32, hi: i32) -> Var {
        Var {
            obj,
            lo: lo as f64,
            hi: hi as f64,
            integer: true,
        }
    }

    fn row(terms: &[(usize, f64)], op: ComparisonOp, rhs: f64) -> Row {
        Row {
            terms: terms.to_vec(),
            op,
            rhs,
        }
    }

    fn check(model: Model) {
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        assert_solves_within_contract(&model);
    }

    /// One equality row whose coefficients span eleven orders of magnitude,
    /// with variable magnitudes up to 1e6. The row's tolerance is 1.3e-7. This
    /// holds today, but only by the pivot path taken: an implementation whose
    /// eta file stored the pivot row's factor as `1 - 1/pivot` reached a
    /// pivot of 7e6 here and reported the basic variable off by 5.7e-7 from
    /// the value the row implies — nine significant digits lost between the
    /// final basis and the reported values. The row must hold regardless of
    /// the path.
    #[test]
    fn single_row_wide_coefficients_holds_the_row() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-15.766694919221791, 2669.2033023769486, 11763.45265877632),
                real(0.0, -123.49646414104626, 189.15555501034143),
                real(-12.51869912438034, -144.0156304000614, 10.620835402144422),
                real(0.0, -31.34381442426171, 140.54940269381058),
                real(0.0, 618.2440473648861, 1644.2175235544687),
                real(14.699533180465284, -921086.7967931526, 983458.444117713),
            ],
            rows: vec![row(
                &[
                    (0, -1018.0990298234458),
                    (1, -0.00014722953519981963),
                    (2, -1.8628244434578861e-6),
                    (3, -1.501733303052947),
                    (5, -1.44172548099894e-8),
                ],
                ComparisonOp::Eq,
                -6552620.390149235,
            )],
            xstar: vec![
                6436.1077974294385,
                5.725090281967169,
                -87.55742628304048,
                16.846098373365365,
                761.7035526387249,
                -921086.7967931526,
            ],
        });
    }

    /// Four rows pinning one variable of magnitude 7e6. Two of the equalities
    /// are consistent with each other only to 1.3e-8 — the rounding of their
    /// right-hand sides at 1e8 activity — which is well inside the documented
    /// `1e-7`. The solve answers `Infeasible`.
    #[test]
    fn single_variable_pinned_by_two_equalities_at_1e7_is_feasible() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![real(
                -18.295116000451756,
                -14564880.1234482,
                -4940629.846466744,
            )],
            rows: vec![
                row(&[(0, -0.5)], ComparisonOp::Eq, 3399667.575541858),
                row(
                    &[(0, -17.49327568376694)],
                    ComparisonOp::Le,
                    145670785.26591063,
                ),
                row(
                    &[(0, 19.82973507916289)],
                    ComparisonOp::Eq,
                    -134829014.76043007,
                ),
                row(
                    &[(0, 13.580497351889516)],
                    ComparisonOp::Le,
                    -87984107.17269267,
                ),
            ],
            xstar: vec![-6799335.151083716],
        });
    }

    /// Three variables of magnitudes 5e4 to 7e8 under three equalities; the
    /// rounding of the right-hand sides is within the round-off allowance of
    /// their 1e10 activities. The solve answers `Infeasible`.
    #[test]
    fn three_equalities_at_1e9_are_feasible() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.8872900839886009, 46934.64307087529, 140401.84408567988),
                real(-17.81390080025668, 30269.277351460834, 197079.86692447605),
                real(0.5, 650595141.9081298, 1335458414.5630662),
            ],
            rows: vec![
                row(
                    &[
                        (0, 0.12316841104007537),
                        (1, 14.628915169575318),
                        (2, -13.31943468692973),
                    ],
                    ComparisonOp::Eq,
                    -9346799886.160616,
                ),
                row(&[(0, 0.5), (1, 2.0)], ComparisonOp::Eq, 176927.23199501407),
                row(
                    &[(0, -14.644350000625979), (2, -0.1347672648327481)],
                    ComparisonOp::Eq,
                    -95270523.7972426,
                ),
            ],
            xstar: vec![46934.64307087529, 76729.9552297882, 701826193.2846973],
        });
    }

    /// An integer variable in `[-7, -6]` pinned to `-7` by two equality rows
    /// whose activities are 2e8: their rounding puts the relaxation's value
    /// 2e-8 outside the bound, far inside the documented tolerance, yet the
    /// relaxation is declared infeasible and so is the model. `x* = (-7,
    /// -1.885e8)` is an integer feasible point.
    #[test]
    fn integer_pinned_by_large_equalities_is_feasible() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(19.488668853496847, -7, -6),
                real(0.5, -893967295.0513096, 130268442.0443216),
            ],
            rows: vec![
                row(
                    &[(1, 0.19033426749547788)],
                    ComparisonOp::Eq,
                    -35881881.9678811,
                ),
                row(
                    &[(0, -1.0), (1, 1.0)],
                    ComparisonOp::Ge,
                    -223010226.34335265,
                ),
                row(
                    &[(0, -1.4161865571957652), (1, -1.0778243028014487)],
                    ComparisonOp::Eq,
                    203191820.42716506,
                ),
            ],
            xstar: vec![-7.0, -188520346.02089512],
        });
    }

    /// Four rows with coefficients from 1e-8 to 1e4 and one variable of
    /// magnitude 8e4. The reported point violates two equality rows by 2.2e-4
    /// and 1.2e-5 against a tolerance of 1e-7.
    #[test]
    fn wide_coefficient_equalities_hold_at_1e5() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(0.0, -0.8534498277609872, 0.04803964132608707),
                real(
                    -0.16639927475938765,
                    -0.5786442966162033,
                    1.6500214250097849,
                ),
                real(-2.0, -83851.13851318957, 205985.52958139687),
                real(0.0, -0.02814904971068355, 2.5504093316020713),
                real(0.0, -1.0952387207881884, 144.4896145562784),
            ],
            rows: vec![
                row(
                    &[
                        (0, 1.185408886329322e-7),
                        (1, 0.00014002289989194163),
                        (2, 0.00012702680583457482),
                        (3, -1.950893810901408e-8),
                        (4, 17.242061704327856),
                    ],
                    ComparisonOp::Le,
                    1423.9270077845931,
                ),
                row(
                    &[
                        (0, -153.7041419414193),
                        (2, -1.4785086707969716e-8),
                        (3, -11893.61251059977),
                    ],
                    ComparisonOp::Eq,
                    -9826.563184636505,
                ),
                row(
                    &[
                        (0, 1.0),
                        (2, -0.01772881416886798),
                        (3, 186.08278806432853),
                        (4, 1.049484505132519e-8),
                    ],
                    ComparisonOp::Eq,
                    -1236.273937441155,
                ),
                row(
                    &[
                        (0, -1.153430886237291),
                        (1, 1.6363195424383565e-7),
                        (2, -0.15600208137925772),
                        (3, -1.545007710388937e-5),
                        (4, 1.2189813495394121e-5),
                    ],
                    ComparisonOp::Eq,
                    -12239.49967436864,
                ),
            ],
            xstar: vec![
                -0.7366192261919466,
                0.1617924631140446,
                78462.73597148646,
                0.8357245068980554,
                67.38520252154089,
            ],
        });
    }

    /// Maximize `x` under `y + 1e-11 x <= 1` with `x >= 0` unbounded above:
    /// the tiny coefficient is a genuine one, so `x` is bounded by `1e11` and
    /// the model has an optimum. Treating every coefficient below `1e-10` as
    /// zero answers `Unbounded`.
    #[test]
    fn a_tiny_genuine_coefficient_bounds_the_objective() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let y = problem.add_var(0.0, (0.0, 1.0));
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        problem.add_constraint([(y, 1.0), (x, 1e-11)], ComparisonOp::Le, 1.0);
        let solution = problem
            .solve()
            .unwrap_or_else(|err| panic!("solve() returned {err:?} on a bounded model"))
            .into_solution()
            .unwrap_or_else(|i| panic!("into_solution() returned {i:?} with no limits set"));
        let objective = solution.objective();
        assert!(
            (objective - 1e11).abs() <= 1e-6 * 1e11,
            "objective {objective:e}, expected 1e11 (x is bounded by the 1e-11 coefficient)"
        );
    }

    /// `x0` is pinned to `2^28` by an equality whose other coefficient is
    /// `1e-7`, so one row's terms span fifteen orders of magnitude, and
    /// `x* = [268435456, 0]` satisfies both rows exactly.
    ///
    /// Once pinned as a known limitation, this was a presolve bug: the row's
    /// round-off is `5e-6` while `x1`'s whole term range is `2.5e-8`, so the
    /// minimum activity equals the bound to round-off wherever `x1` sits. The
    /// forcing rule read that as "every var at its extreme" and fixed `x1` at
    /// `0.25`, which made row 1 (`-x1 = 0`) inconsistent; the engine solves
    /// the model exactly with presolve off. Forcing now requires every term
    /// to be resolvable above the row's round-off (`Work::terms_resolvable`).
    ///
    /// Found by the wide-coefficient LP property in `scaling.rs`; committed
    /// here because that property's counterexamples otherwise live only in
    /// the gitignored `.hegel/` corpus.
    #[test]
    fn wide_coefficients_pinning_a_variable_at_2e8() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![real(0.0, 268435455.0, 268435456.0), real(0.0, 0.0, 0.25)],
            rows: vec![
                row(&[(0, -1.0), (1, -1e-7)], ComparisonOp::Eq, -268435456.0),
                row(&[(1, -1.0)], ComparisonOp::Eq, -0.0),
            ],
            xstar: vec![268435456.0, 0.0],
        });
    }

    /// The wide-coefficient chain of the fixtures below: five variables (one
    /// integer, two fixed) under three rows whose coefficients run from
    /// `1e-8` to `1e4`. Rows 0 and 2 tie `x3` to `x2`, `x3 ≈ 1e-13 (x2 -
    /// shift)`, and `x3` lies in `[0, 1]`, so the rows hold only at `x2 =
    /// shift` with `x3 = x4 = 0`; `x2`'s bounds are `[lo, hi]`.
    fn wide_chain(lo: f64, hi: f64, shift: f64) -> Model {
        Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(0.0, 0, 0),
                real(0.0, 0.0, 0.0),
                real(0.0, lo, hi),
                real(0.0, 0.0, 1.0),
                real(0.0, -1.0, 1.0),
            ],
            rows: vec![
                row(&[(3, 1000.0), (4, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(2, -1.0), (4, -1e-5)], ComparisonOp::Ge, -shift),
                row(
                    &[(2, 1e-6), (3, -1e-8), (4, 10000.0)],
                    ComparisonOp::Eq,
                    1e-6 * shift,
                ),
            ],
            xstar: vec![0.0, 0.0, shift, 0.0, 0.0],
        }
    }

    /// Once pinned as a known limitation. `x* = 0` satisfies all three rows
    /// exactly, yet the solve answered `Infeasible` with presolve on or off.
    /// The start puts `x2` at its lower bound `-1000001`, and the only
    /// tableau entry that can carry it back is `1.5e-15` against a row
    /// maximum of `1.95`, below the round-off of computing the row: the dual
    /// ratio test rightly refused it as a pivot, found no other, and took
    /// "no pivot" for a proof of infeasibility. Moving `x2` needs no pivot:
    /// its flip to `0` carries the leaving var of row 0 exactly the
    /// `1.19e-11` (scaled) it is short. The dual ratio test flips boxed vars
    /// now, including through such entries when the flip lands the leaving
    /// var (checked through the column), and declares infeasibility only
    /// when no flip can reach the bound. The integer var plays no part.
    /// Found by the wide-coefficient MILP property in `scaling.rs`.
    #[test]
    fn wide_coefficients_reached_by_a_bound_flip() {
        check(wide_chain(-1000001.0, 0.0, 0.0));
        assert_solves_within_contract_without_presolve(&wide_chain(-1000001.0, 0.0, 0.0));
    }

    /// [`wide_coefficients_reached_by_a_bound_flip`] shifted so that `x2`'s
    /// feasible value is the bound farther from zero, where a start at the
    /// bound nearer zero (HiGHS's rule, which does solve the unshifted model)
    /// begins: the flip does not depend on where the solve starts.
    #[test]
    fn wide_coefficients_reached_by_a_flip_to_the_bound_far_from_zero() {
        check(wide_chain(0.0, 1000001.0, 1000001.0));
        assert_solves_within_contract_without_presolve(&wide_chain(0.0, 1000001.0, 1000001.0));
    }

    /// The chain with `x2`'s feasible value inside its range: the flip would
    /// carry the leaving var of row 0 past its bound (an equality's), so
    /// `x2` moves only as far as the row needs and stays non-basic between
    /// its bounds.
    #[test]
    fn wide_coefficients_reached_by_a_partial_move() {
        check(wide_chain(-2000002.0, 1000001.0, 0.0));
        assert_solves_within_contract_without_presolve(&wide_chain(-2000002.0, 1000001.0, 0.0));
    }

    /// The chain with `x2` confined to `[-2e7, -1e7]`: `x3 ≈ 1e-13 x2` is
    /// then at most `-1e-6`, beyond every tolerance, and even `x2`'s flip to
    /// `-1e7` leaves row 0 short. With no var to enter and no flip that can
    /// reach the bound, the row proves the model infeasible, and the answer
    /// must stay `Infeasible`: moving vars through entries below round-off
    /// must not turn a proof into a guess.
    #[test]
    fn wide_coefficients_beyond_every_flip_are_infeasible() {
        for presolve in [true, false] {
            let (problem, _) = wide_chain(-2e7, -1e7, 0.0).build();
            let mut options = SolveOptions::default();
            options.presolve = presolve;
            match problem.solve_with(options) {
                Err(Error::Infeasible) => {}
                other => panic!("presolve {presolve}: expected Infeasible, got {other:?}"),
            }
        }
    }

    /// Variable magnitudes are around `8e6` and row 1's terms reach `1.3e8`.
    /// The exactly integral optimum the engine finds must be reported, not
    /// rejected by its own feasibility guard as an `InternalError`.
    ///
    /// Once pinned as a known limitation, this was a bookkeeping bug in the
    /// verified termination: the sixth dual pivot filled the eta file, the
    /// periodic refactorization recomputed the basic values, and the
    /// recompute marked them verified. `verify_primal` then skipped its
    /// residual check although the recomputed values missed row 2 by four
    /// times its tolerance, and the guard caught what verification should
    /// have refined. A recompute no longer counts as a verification. Found
    /// by the narrow-coefficient MILP property in `scaling.rs`.
    #[test]
    fn exactly_integral_optimum_at_1e8_is_verified_before_it_is_reported() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, 0, 0),
                real(-1.0, 0.0, 0.0),
                real(1.0, 0.0, 68282.0),
                real(1.0, 7962362.0, 8516698.0),
                real(-1.0, 0.0, 0.0),
                real(0.0, -1.0, 0.0),
            ],
            rows: vec![
                row(&[(3, 1.0), (5, -1.0)], ComparisonOp::Ge, 7962362.0),
                row(
                    &[(2, -0.1), (3, -16.25), (5, 1.0)],
                    ComparisonOp::Ge,
                    -129648946.5,
                ),
                row(
                    &[(2, -0.5), (3, 0.15000000000000002), (5, -20.0)],
                    ComparisonOp::Eq,
                    1194354.3000000003,
                ),
                row(
                    &[(2, 1.0), (3, -1.0), (5, -1.0)],
                    ComparisonOp::Le,
                    -7962362.0,
                ),
            ],
            xstar: vec![0.0, 0.0, 0.0, 7962362.0, 0.0, 0.0],
        });
    }

    /// Variable magnitudes around `8.5e6`, found by the narrow-coefficient LP
    /// property. Row 2's equality slack sat past its bound within its
    /// tolerance when the primal simplex took a step that moved it further
    /// out; the Harris bound granted it its whole tolerance again, measured
    /// from the bound rather than from where it was, so the step carried it
    /// to 1.3 times its tolerance. The phase exit rejected the point, the dual
    /// phase undid the step, and after eight rounds of this the solve gave up
    /// with `InternalError`.
    #[test]
    fn harris_allowance_is_measured_from_where_a_basic_var_sits() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -43.0, 0.0),
                real(0.0, 0.0, 1.0),
                real(-1.0, 0.0, 1.0),
                real(0.0, -1.0, 0.0),
                real(1.0, -8470086.0, 125747738.0),
            ],
            rows: vec![
                row(&[(0, 0.1), (4, -1.0)], ComparisonOp::Eq, 8469986.0),
                row(
                    &[(1, 0.1), (2, -0.1), (3, -2.0), (4, 10.0)],
                    ComparisonOp::Eq,
                    -84699860.0,
                ),
                row(
                    &[(1, 0.1), (2, 1.0), (4, 1.0)],
                    ComparisonOp::Eq,
                    -8469986.0,
                ),
                row(&[(2, 20.0), (3, 20.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0, -8469986.0],
        });
    }

    /// Found by the wide-coefficient MILP property: an integer var with a
    /// coefficient of 1e-8 in an equality may leave the row's solution by up
    /// to the row's tolerance over that coefficient, ten units here, and
    /// presolve's integer rounding, holding integer points to the full
    /// contract, kept the points exactly at it. The guard then evaluated
    /// the row there a few ulps over the contract, and an exactly integral
    /// candidate was rejected (`InternalError`). Integer rounding leaves the
    /// guard the round-off of its own evaluation. (Tolerance leaves these
    /// models real freedom in the objective, so only feasibility is
    /// checked.)
    #[test]
    fn integer_var_with_a_tiny_coefficient_keeps_the_guards_margin() {
        let check_feasible = |model: Model| {
            assert!(xstar_is_feasible(&model), "fixture error");
            assert_solves_feasibly_within_contract(&model);
        };
        check_feasible(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![int(1.0, 1, 11)],
            rows: vec![row(&[(0, -1e-8)], ComparisonOp::Eq, -1e-8)],
            xstar: vec![1.0],
        });
        check_feasible(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![int(0.0, -1, 0), int(-1.0, 1, 1), real(0.0, 0.0, 0.0)],
            rows: vec![row(&[(0, -1e-7), (1, 1.0)], ComparisonOp::Eq, 1.0)],
            xstar: vec![0.0, 1.0, 0.0],
        });
    }

    /// Found by the wide-coefficient properties once they kept their
    /// objective: models whose only feasible region is reached through a
    /// genuine pivot of 4e-15 to 2e-14 of its column (1e-8 beside 1e4 in a
    /// row). Both computations of such a pivot agree exactly, which is what
    /// tells it from a round-off entry of the same size (those come out
    /// percents apart); refusing every pivot that small, as a threshold on
    /// size alone did, made all three `Infeasible`.
    #[test]
    fn genuine_pivots_near_the_round_off_floor_are_taken() {
        let check_feasible = |model: Model| {
            assert!(xstar_is_feasible(&model), "fixture error");
            assert_solves_feasibly_within_contract(&model);
        };
        check_feasible(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -49411.0, 0.0),
                real(-1.0, 0.0, 1.0),
                real(-1.0, 0.0, 0.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(1, -1.0), (2, -10000.0)], ComparisonOp::Eq, -0.0),
                row(&[(0, 10000.0), (3, -1.0)], ComparisonOp::Le, 0.0),
                row(&[(1, 10000.0), (3, -1.0)], ComparisonOp::Eq, 0.0),
                row(&[(0, 1.0), (3, 16250.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0],
        });
        check_feasible(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, 0.0, 1.0),
                real(0.0, 0.0, 0.0),
                real(0.0, -6.0, 0.0),
                real(-1.0, 0.0, 7.0),
            ],
            rows: vec![
                row(&[(2, 1.0), (3, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(0, -0.1), (3, -10000.0)], ComparisonOp::Le, 0.0),
                row(&[(2, 1000.0), (3, -1e-8)], ComparisonOp::Le, 0.0),
                row(&[(0, -10000.0), (3, -1e-8)], ComparisonOp::Eq, -0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0],
        });
        check_feasible(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, 0, 0),
                real(0.0, -5001.0, 0.0),
                real(-1.0, -6.0, 0.0),
                real(0.0, -1.0, 0.0),
                real(0.0, 0.0, 0.0),
                real(0.0, 0.0, 0.0),
            ],
            rows: vec![
                row(&[(2, -1e-8), (3, -100.0)], ComparisonOp::Eq, -0.0),
                row(
                    &[(1, -1.0), (2, 1000.0), (3, -0.001)],
                    ComparisonOp::Eq,
                    0.0,
                ),
                row(&[(1, -10.0), (3, -0.001)], ComparisonOp::Ge, -0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        });
    }

    /// Found by the wide-coefficient MILP property. The primal step that
    /// lowers x3 is bounded both by `-2 x3 <= -1` and, through the Harris
    /// relaxation, by the equality `x2 + 1e-7 x3 = 1.0000001` at its
    /// tolerance: the ratio test took the step, the equality's slack ended
    /// exactly at its tolerance, the phase exit's recomputation landed it an
    /// ulp past, the dual phase undid the step, the primal phase redid it,
    /// and after eight rounds the solve gave up (`InternalError`). The
    /// Harris allowance was then cut by the round-off of recomputing a var;
    /// now each round after the first cuts it a hundredfold, as SoPlex does
    /// when its algorithms loop (`LOOP_HARRIS_REDUCTION`): in the third round
    /// the slack, still at its tolerance, blocks the step (a zero one), and
    /// the solve ends at the vertex `x3 = 1`.
    #[test]
    fn harris_step_leaves_the_round_off_of_the_recomputation() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, 0, 0),
                real(-1.0, -1.0, 0.0),
                real(0.0, 0.0, 1.0),
                real(-1.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(1, -100.0), (3, 100.0)], ComparisonOp::Eq, 100.0),
                row(&[(2, 1.0), (3, 1e-7)], ComparisonOp::Eq, 1.0000001),
                row(&[(3, -2.0)], ComparisonOp::Le, -1.0),
            ],
            xstar: vec![0.0, 0.0, 1.0, 1.0],
        });
    }

    /// Found by CI's run of the wide-coefficient LP property on the first
    /// version of the primal simplex's bound shifting, which set a leaving
    /// var onto its bound, taking the step backwards where it sat past it, as
    /// HiGHS does. The fixed slack of `0.1 x1 + x2 = 0` sat basic `3e-17` off
    /// its value, the round-off of the updates; the primal step that set it
    /// onto it moved the entering slack of `10 x0 - 1e-8 x2 <= 0` by that over
    /// an element of `7.8e-8`, `3.9e-10` past its bound (`1.3e-7` in user
    /// units, beyond the contract); and the dual simplex's repair pivoted on
    /// `1e-7` into a basis the factorization rejected (`Singular matrix`,
    /// with presolve on). A step that would run backwards is zero, and a
    /// fixed var leaving within its tolerance of its value stays where it
    /// sits, through the cleanup too.
    #[test]
    fn a_fixed_slack_carrying_round_off_leaves_where_it_sits() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -1.0, 0.0),
                real(1.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(1, 0.1), (2, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(0, -1.0)], ComparisonOp::Ge, -0.0),
                row(&[(0, 10.0), (2, -1e-8)], ComparisonOp::Le, 0.0),
                row(&[(0, -1e-7), (1, -10000.0)], ComparisonOp::Le, 0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0],
        });
    }

    /// Found by the deep run of the wide-coefficient MILP property on this
    /// step's tree, and failing the same at 127db49 (the dual simplex's cost
    /// shifting) and after: `8 phase rounds` where 1dcd664 solves it. In a
    /// node LP whose branch left the integer `x0` at a bound, each dual phase
    /// after the first started with `x0`'s reduced cost dual infeasible (the
    /// shift of the zero dual step before it given back) and corrected it by
    /// flipping `x0` to its other bound, a move the phase then could not
    /// repair (no entering column for `x0`'s row) and undid through a zero
    /// step, and the verification found the same row violated, round after
    /// round. A phase that cleans up after another forces phase 2, as HiGHS's
    /// cleanup calls do, and shifts `x0`'s cost instead of flipping it.
    #[test]
    fn a_cleanup_dual_phase_shifts_a_boxed_var_rather_than_flipping_it() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, -2, 0),
                real(0.0, 0.0, 0.0),
                real(0.0, -8388643.0, -8388642.0),
                real(-1.0, -1.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(0, 1.0), (2, 1.0), (3, -2.0)],
                    ComparisonOp::Eq,
                    -8388643.0,
                ),
                row(
                    &[(0, 1.5000000000000002e-8), (2, 20.0)],
                    ComparisonOp::Eq,
                    -167772860.0,
                ),
                row(
                    &[(0, 0.0011298828125), (2, 13046.875)],
                    ComparisonOp::Eq,
                    -109445576640.625,
                ),
            ],
            xstar: vec![0.0, 0.0, -8388643.0, 0.0],
        });
    }

    /// Found by the wide-coefficient MILP property (`milp_wide_2` of the
    /// solver review's stress run, C2-17/C2-18): a solve that never
    /// returned. With the integer vars fixed at 0 and -115, the four
    /// equalities have zero right-hand sides and pin every continuous var at
    /// zero, a primal degenerate vertex that the pivots reach only within
    /// the rows' tolerances. The dual simplex went round a four-pivot cycle:
    /// two regular pivots on one row handed a tolerance-level violation to
    /// the other equality, and two pivots of its fixed slacks, absorbing the
    /// violation, handed it back, each a dual step backwards, so the
    /// objective rose and fell and the bases repeated forever. With cost
    /// shifting those pivots take a zero dual step, but the shift of a fixed
    /// slack is its whole reduced cost, given back when it leaves, and the
    /// vertices still come back; the vertex hash sees a pivot lead to a
    /// visited vertex right after the one before it did, makes that basis
    /// change taboo, and the phase ends at the optimum (HiGHS: 0 at `x*`).
    /// Without the hash the stall bound ends it in an error. The limit turns
    /// a regression into a failure instead of a hang.
    #[test]
    fn a_dual_simplex_that_cycled_reaches_the_optimum() {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(1.0, 0, 0),
                int(0.0, -115, -115),
                real(1.0, 0.0, 1101.2422443124906),
                real(-1.0, 0.0, 1.072334405752396),
                real(-2.0, -1615394529.5388072, 0.0),
                real(-1.0, 0.0, 1.0),
            ],
            rows: vec![
                row(
                    &[(1, -20.0), (2, -2.0), (4, 0.001), (5, 0.1)],
                    ComparisonOp::Eq,
                    2300.0,
                ),
                row(
                    &[(0, -1.0), (1, -1.0), (3, -1.0), (4, -1.0), (5, 1.0)],
                    ComparisonOp::Ge,
                    115.0,
                ),
                row(
                    &[
                        (1, -10000.0),
                        (2, 0.0015177021581456369),
                        (3, -1e-08),
                        (4, 1.9180368898411304e-08),
                    ],
                    ComparisonOp::Eq,
                    1150000.0,
                ),
                row(
                    &[
                        (0, -1.0),
                        (1, -1.0),
                        (2, -1.0),
                        (3, -0.001),
                        (4, 1.0),
                        (5, -10000.0),
                    ],
                    ComparisonOp::Eq,
                    115.0,
                ),
                row(
                    &[
                        (0, -1.0),
                        (2, -0.00019180368898411305),
                        (4, -0.001),
                        (5, 1.0),
                    ],
                    ComparisonOp::Eq,
                    0.0,
                ),
            ],
            xstar: vec![0.0, -115.0, 0.0, 0.0, 0.0, 0.0],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract_in_time(&model, Duration::from_secs(10));
    }

    /// A huge cost must not make a feasible model infeasible (found by the
    /// review of the dual simplex's cost perturbation). The model minimizes
    /// `4 x0 - 1e300 x1 - 5 x2 + 5 x3 - x4 - 2 x5` over four rows, the first
    /// two of which force `x1 = 2 x4 - 2 = 0`. The perturbation moved a cost
    /// by about `1e-6 |c|^(5/4)`, which for `1e300` is infinite: the reduced
    /// costs became infinite or not a number, the dual ratio test dropped
    /// every such candidate, and a row left with none declared the model
    /// infeasible, as an LP and as a MILP, with presolve on and off (for any
    /// cost above about `2e251`). A cost of `1e20` or more now takes no part
    /// in the perturbation, as none ever reaches HiGHS's (its
    /// `infinite_cost`), and a reduced cost that is not a number fails the
    /// ratio test loudly. With such a cost the multipliers are of its size
    /// and the small costs are round-off beside them, so the objective is
    /// only as good as that allows (4 is reported, 3 is optimal), which the
    /// contract's objective slack, relative to the objective's extent, allows
    /// too.
    #[test]
    fn a_huge_cost_leaves_a_feasible_model_feasible() {
        for integer in [false, true] {
            let binary = |obj: f64| {
                if integer {
                    int(obj, 0, 1)
                } else {
                    real(obj, 0.0, 1.0)
                }
            };
            let model = Model {
                direction: OptimizationDirection::Minimize,
                vars: vec![
                    binary(4.0),
                    real(-1e300, 0.0, 8.0),
                    real(-5.0, -2.0, -1.0),
                    real(5.0, -3.0, 4.0),
                    binary(-1.0),
                    real(-2.0, -2.0, 6.0),
                ],
                rows: vec![
                    row(
                        &[(0, 3.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, -1.0), (5, 2.0)],
                        ComparisonOp::Eq,
                        -1.0,
                    ),
                    row(
                        &[(0, 3.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 2.0)],
                        ComparisonOp::Eq,
                        1.0,
                    ),
                    row(
                        &[(0, 2.0), (1, -3.0), (2, 1.0), (3, -2.0), (4, -2.0)],
                        ComparisonOp::Le,
                        -3.0,
                    ),
                    row(
                        &[
                            (0, -1.0),
                            (1, -3.0),
                            (2, -3.0),
                            (3, -1.0),
                            (4, 2.0),
                            (5, 1.0),
                        ],
                        ComparisonOp::Ge,
                        -6.0,
                    ),
                ],
                xstar: vec![0.0, 0.0, -2.0, -0.5, 1.0, 1.25],
            };
            assert!(
                xstar_is_feasible(&model),
                "fixture error: x* must satisfy the model"
            );
            assert_solves_within_contract(&model);
            assert_solves_within_contract_without_presolve(&model);
        }
    }

    /// The differential fuzz's model `wide_lp-558` reaches its optimum
    /// (HiGHS: 231413.21367), within the fuzz's `1e-6` relative, with
    /// presolve on and off. Two decisions of the dual simplex rest on it,
    /// and without either one the solve ends at 231413.844, the objective at
    /// `x*` (`x1` at its lower bound instead of its upper), which the
    /// contract's objective slack does not tell from the optimum: the
    /// cleanup of a dual phase only restores the costs, so that
    /// `run_phases` verifies the values before it recomputes the reduced
    /// costs and runs the primal phase (a cleanup that recomputed them
    /// itself sent the primal phase onto unverified values and stale slack
    /// tolerances); and an absorbing pivot takes the absorber whose shift
    /// does the least harm when it is given back (here a fixed slack), not
    /// the one with the largest entry.
    #[test]
    fn wide_lp_558_reaches_the_optimum() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(4.0, -362732.08683384664, 1590851.6121015674),
                real(-2.0, 0.44352166456169106, 0.6864194512947741),
                real(5.0, -11.7608931155399, 0.9296651554717794),
                real(-4.0, -57865.9890336753, 58309.301095335744),
                real(1.0, -0.17335417252699625, 2.4772179264835907),
            ],
            rows: vec![
                row(
                    &[(0, 0.6196016712816187), (3, 4.505057732180742e-05)],
                    ComparisonOp::Le,
                    -2.6068962132644495,
                ),
                row(
                    &[(0, 41.31211423321696), (4, 1.623352003798267e-06)],
                    ComparisonOp::Eq,
                    1.1672677869244323e-06,
                ),
                row(
                    &[
                        (0, -0.0625575647400732),
                        (2, 2.1988892362384683e-06),
                        (3, -8.776267812547736),
                        (4, -67.52468996017767),
                    ],
                    ComparisonOp::Eq,
                    507798.86349160096,
                ),
                row(
                    &[
                        (1, 33.06827012942814),
                        (2, 276.9654820263267),
                        (4, -0.8992788059382684),
                    ],
                    ComparisonOp::Eq,
                    -2752.514608600898,
                ),
            ],
            xstar: vec![
                0.0,
                0.44352166456169106,
                -9.988733823688296,
                -57865.9890336753,
                0.7190478615810352,
            ],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
        let optimum = 231413.21367073804;
        for presolve in [true, false] {
            let (problem, _) = model.build();
            let objective = problem
                .solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                })
                .unwrap()
                .into_solution()
                .unwrap()
                .objective();
            assert!(
                objective - optimum <= 1e-6 * (1.0 + optimum),
                "presolve {presolve}: objective {objective} above the optimum {optimum}"
            );
        }
    }

    /// The differential fuzz's model `wide_lp-1897` solves within the
    /// contract, with presolve on and off (HiGHS reports -4019.66 at a point
    /// outside this contract; here the answer is the objective at `x*`). It
    /// rests on how a rebuild within a dual phase corrects a dual
    /// infeasibility: by shifting the cost, a boxed var's too. A step
    /// flipped a boxed var to its upper bound and the next left it dual
    /// infeasible there (its entry was below the pivot floor, so it did not
    /// bound the step); moving it back to its lower bound at the rebuild,
    /// as HiGHS's correction does, returned the phase to a vertex it had
    /// left, round a loop that the cycling detection ended in an
    /// `InternalError`.
    #[test]
    fn wide_lp_1897_solves_within_the_contract() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-3.0, 0.6883921955253283, 15.517025166440686),
                real(-4.0, 873.8664825794901, 928.7738519944153),
                real(0.0, -71.66441409674097, 36.95343826988335),
                real(6.0, 0.0, 12.289325360619888),
                real(7.8, -65.21510387756965, 72.78911622394193),
                real(0.0, 0.0, 666.5395941775796),
                real(5.85, -1.283497406554833, 0.35507766150091924),
                real(0.0, 264.3902621802938, 387.9852685320315),
            ],
            rows: vec![
                row(
                    &[
                        (0, 1.0),
                        (1, -0.5),
                        (3, -5.823957729416976e-05),
                        (5, -6.704199631700646),
                    ],
                    ComparisonOp::Eq,
                    -432.3062304655302,
                ),
                row(
                    &[
                        (1, -2.0),
                        (2, -568261.67098366),
                        (4, 38995.1516762666),
                        (5, -0.4317501721527071),
                        (6, -7.176881805461856e-06),
                    ],
                    ComparisonOp::Ge,
                    6556439.159718655,
                ),
                row(
                    &[
                        (0, 167575.07817296375),
                        (2, -7.191313132252542e-06),
                        (3, 0.0005844836777845698),
                        (4, -1.4402440281244824e-05),
                        (5, -5030.603934043473),
                        (6, -1.0),
                    ],
                    ComparisonOp::Eq,
                    775371.8153046283,
                ),
                row(
                    &[
                        (0, -28625.642053462398),
                        (1, 2.0),
                        (2, 3624.980672864286),
                        (6, 9556873.256277684),
                        (7, -0.00021227956360568158),
                    ],
                    ComparisonOp::Le,
                    -1239848.1423610675,
                ),
                row(
                    &[
                        (2, 0.8260768589663329),
                        (4, 0.0003244702493151561),
                        (5, -0.5),
                        (6, -7.935149186793402),
                        (7, 0.8606787959107634),
                    ],
                    ComparisonOp::Le,
                    367.2292388216485,
                ),
                row(
                    &[(0, -1423.3949571913254), (4, 1.0), (6, -839.8403651079586)],
                    ComparisonOp::Eq,
                    -6416.079244439468,
                ),
            ],
            xstar: vec![
                4.627010824214878,
                873.8664825794901,
                -6.54586514467897,
                0.0,
                72.78911622394193,
                0.0,
                -0.11573093820184721,
                264.3902621802938,
            ],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
    }

    /// A fixed var's huge coefficient must not silence the rest of its row
    /// (C2-06). `min y` s.t. `1e17 f + y >= 1`, `f` fixed at zero: neither
    /// scaling touches the row (a fixed column is left out of both), so the
    /// dual ratio test saw the row `[1e17, 1]`. Its round-off scale was
    /// taken over every entry, the fixed column's included, which made
    /// `y`'s exact entry of one "noise", and the row a proof of
    /// infeasibility, with presolve on (the row is kept as written) and off.
    /// The scale now leaves out the columns the scalings left out.
    #[test]
    fn a_fixed_vars_huge_coefficient_does_not_silence_its_row() {
        for presolve in [true, false] {
            let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
            let f = problem.add_var(0.0, (0.0, 0.0));
            let y = problem.add_var(1.0, (0.0, f64::INFINITY));
            problem.add_constraint([(f, 1e17), (y, 1.0)], ComparisonOp::Ge, 1.0);
            let solution = problem
                .solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                })
                .unwrap_or_else(|e| panic!("presolve {presolve}: {e:?}"))
                .into_solution()
                .unwrap();
            assert_eq!(
                (solution[f], solution[y], solution.objective()),
                (0.0, 1.0, 1.0),
                "presolve {presolve}"
            );
        }
    }

    /// The model above through the LP edit path (found by the second review
    /// of R0, and pinned as a known limitation until the dual simplex needed
    /// a proof to report infeasibility). `min y` s.t. `f + y >= 0`,
    /// `f ∈ [0, 10]`, then `fix_var(f, 0)` and
    /// `add_constraint(1e17 f + y >= 1)`: the added row leaves the now fixed
    /// `f` out of its equilibration, so the dual ratio test meets `[1e17, 1]`
    /// again, and its round-off scale leaves out only the columns fixed when
    /// the solve began, so `f`'s entry sets a floor of 100 and `y`'s exact
    /// entry of one is "noise": no candidate. The ratio test took that for a
    /// proof of infeasibility, and the edit came back `Infeasible` (the
    /// answer is `y = 1`), with presolve on and off, as at 1dcd664. The row's
    /// certificate does not hold now (`y` has no upper bound), the row is
    /// made taboo, and with no other row left the dual simplex takes SoPlex's
    /// last resort, a pivot on an entry its regular test refuses but above
    /// SoPlex's `lowstab`: `y`'s one. (The absolute pivot tolerances of R3
    /// would make `y` a regular candidate.)
    #[test]
    fn a_fixed_vars_huge_coefficient_in_an_added_row_does_not_silence_it() {
        for presolve in [true, false] {
            let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
            let f = problem.add_var(0.0, (0.0, 10.0));
            let y = problem.add_var(1.0, (0.0, f64::INFINITY));
            problem.add_constraint([(f, 1.0), (y, 1.0)], ComparisonOp::Ge, 0.0);
            let solution = problem
                .solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                })
                .unwrap()
                .into_solution()
                .unwrap()
                .fix_var(f, 0.0)
                .unwrap()
                .into_solution()
                .unwrap()
                .add_constraint([(f, 1e17), (y, 1.0)], ComparisonOp::Ge, 1.0)
                .unwrap_or_else(|e| panic!("presolve {presolve}: the edit returned {e:?}"))
                .into_solution()
                .unwrap();
            assert_eq!(
                (solution[f], solution[y]),
                (0.0, 1.0),
                "presolve {presolve}"
            );
        }
    }

    /// A row whose coefficients span thirty orders of magnitude keeps its
    /// small genuine entry. `min -g + y` s.t. `big g + y >= 2 big`,
    /// `g ∈ [0, 1]`, `y >= 0`: the optimum is `g = 1`, `y = big`. With the
    /// column factors bounded to `2^±20`, as HiGHS bounds them (C4-14 of the
    /// solver review), neither column could be scaled far enough and the
    /// equilibrated row was `[1, 1.1e-18]` at `big = 1e30`: `y`'s entry lay
    /// below the dual ratio test's round-off floor, `y` has no upper bound to
    /// flip to, and the row came back as a proof of infeasibility (as an LP
    /// with presolve on and off, and as a MILP with presolve off). HiGHS
    /// never meets the row, since it refuses matrix entries of 1e15 or more;
    /// this engine takes them, so its column factors are not bounded.
    #[test]
    fn a_row_spanning_thirty_orders_keeps_its_small_genuine_entry() {
        for big in [3e27, 1e28, 1e30] {
            for integer in [false, true] {
                for presolve in [true, false] {
                    let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
                    let g = if integer {
                        problem.add_integer_var(-1.0, (0, 1))
                    } else {
                        problem.add_var(-1.0, (0.0, 1.0))
                    };
                    let y = problem.add_var(1.0, (0.0, f64::INFINITY));
                    problem.add_constraint([(g, big), (y, 1.0)], ComparisonOp::Ge, 2.0 * big);
                    let label = format!("big {big:e}, integer {integer}, presolve {presolve}");
                    let solution = problem
                        .solve_with(SolveOptions {
                            presolve,
                            ..SolveOptions::default()
                        })
                        .unwrap_or_else(|e| panic!("{label}: {e:?}"))
                        .into_solution()
                        .unwrap();
                    assert_eq!(solution[g], 1.0, "{label}");
                    assert!(
                        (solution[y] - big).abs() <= 1e-12 * big,
                        "{label}: y = {:e}",
                        solution[y]
                    );
                }
            }
        }
    }

    /// Issue #3's model with `z ≤ B` for `B` of `3e14` to `4e15`, real
    /// bounds below the bound infinity of 2^52, pinned as a known limitation
    /// until the primal simplex shifted its bounds (ARCHITECTURE §7): `z`
    /// starts at `B`, where the rows' tolerances are taken (about 2.5 in the
    /// scaled row at 1e15), one dual pivot brings it down to 1.375, and
    /// `3x + 2y + z ≤ 20`, violated by 1.375, passes as feasible within that
    /// tolerance; the primal simplex then let the row's slack leave where it
    /// sat, off its bound, and the boundary reported `InternalError("LP
    /// solution violates constraint row 0 by 1.375")` (from 1dcd664 on). The
    /// slack still leaves where it sits (a step that would run backwards is
    /// zero), but the cleanup of the primal phase moves it onto its bound:
    /// the bound it was left past is a perturbed one, or else its tolerance
    /// at the current values is far below the one it was judged by
    /// (`a_slack_left_off_its_bound_beyond_its_current_tolerance_moves_onto_it`
    /// in `src/solver/tests.rs`), and the dual simplex repairs the row.
    #[test]
    fn a_row_left_with_the_tolerance_of_a_huge_bound_is_repaired() {
        for bound in [1e14, 3e14, 1e15, 4e15] {
            for presolve in [true, false] {
                let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_var(50.0, (2.0, f64::INFINITY));
                let y = problem.add_var(40.0, (0.0, 7.0));
                let z = problem.add_var(45.0, (0.0, bound));
                problem.add_constraint([(x, 3.0), (y, 2.0), (z, 1.0)], ComparisonOp::Le, 20.0);
                problem.add_constraint([(x, 2.0), (y, 1.0), (z, 3.0)], ComparisonOp::Le, 15.0);
                let label = format!("z <= {bound:e}, presolve {presolve}");
                let solution = problem
                    .solve_with(SolveOptions {
                        presolve,
                        ..SolveOptions::default()
                    })
                    .unwrap_or_else(|e| panic!("{label}: {e:?}"))
                    .into_solution()
                    .unwrap();
                assert!(
                    (solution.objective() - 420.0).abs() < 1e-6,
                    "{label}: objective {}",
                    solution.objective()
                );
            }
        }
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the phases alternate over a row
    /// only entries below the round-off floor can repair. `lp_wide_3` of the
    /// solver review's stress run: the only feasible point is the origin, a
    /// primal degenerate vertex. The dual phase ends there with `x0`'s
    /// reduced cost dual infeasible; the primal step that lowers `x0` is
    /// blocked, in exact arithmetic, by `-1e4 x2 + 1e-3 x3 >= 0`, whose
    /// entry, a genuine `3e-15`, is below the round-off floor of the column
    /// (largest entry 4), so the step of 32 passes it and ends the row
    /// `1e-7` (twice its tolerance) short: its bound is shifted, and the
    /// cleanup exposes the violation. The dual phase can repair the row only
    /// by moving `x5` across its range through another such entry, which
    /// gives the primal phase its step back; after eight rounds the solve
    /// fails with `InternalError`. The same at 1dcd664, where the move was the
    /// dual ratio test's flip-only stage (C2-10 of the solver review); now
    /// the row's certificate proves nothing (the model is feasible), SoPlex's
    /// instable leave finds no entry to pivot on, and the move is the dual
    /// simplex's round-off repair, its last resort. What resolves it is a
    /// pivot on the `3e-15` (a degenerate one, onto the optimal basis): the
    /// absolute pivot tolerances with a refined row of step R3.
    #[test]
    #[should_panic(expected = "8 phase rounds")]
    fn phases_alternating_over_a_row_repaired_through_round_off_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, -2.0, 0.0),
                real(0.0, 0.0, 0.0),
                real(0.0, 0.0, 1.0),
                real(1.0, -1.0, 0.0),
                real(1.0, 0.0, 0.0),
                real(-1.0, -1.0, 1.0),
            ],
            rows: vec![
                row(&[(0, 1.0), (3, -0.01), (5, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(2, -10000.0), (3, 0.001)], ComparisonOp::Ge, 0.0),
                row(
                    &[(0, -1.0), (2, -1e-8), (3, 10000.0)],
                    ComparisonOp::Eq,
                    0.0,
                ),
            ],
            xstar: vec![0.0; 6],
        });
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the same class, found by the deep
    /// run of the wide-coefficient MILP property on the primal simplex's
    /// bound shifting, which exposed it. `min x0 - x1` s.t.
    /// `-10 x0 - 1e-8 x1 - 10 x3 = 0`, `-1e-7 x0 + 1e4 x1 = 0`: the second row
    /// makes `x1 = 1e-11 x0`, so `x1 >= 0` needs `x0 >= 0`, and the optimum
    /// is 0. The primal step that lowers `x0` to -1 passes the row of `x1`
    /// through an entry below the round-off floor of its column; the point it
    /// reaches meets the engine's tolerances only off every vertex (the
    /// equality at its tolerance and `x1` at its own, which its row's reserve
    /// caps at `5e-12`), so the cleanup leaves `x1` beyond its tolerance, the
    /// dual phase moves `x3` across its range through another entry below the
    /// floor to repair it, and the primal step comes back, round after round.
    /// Up to the degenerate clamp this step removed (C3-21), the solve
    /// returned `x0 = -1` with the equality violated by exactly `1e-7`, the
    /// whole contract and twice the engine's share: the clamp had set the
    /// equality's slack to within its tolerance without moving anything else,
    /// and the residual that left in the row passed the verification. The
    /// same remedies apply (R3), or the full contract of R4, under which the
    /// vertex that sets `x1` to `-1e-11` verifies.
    #[test]
    #[should_panic(expected = "8 phase rounds")]
    fn an_equality_met_only_off_every_vertex_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(1.0, -1, 1),
                real(-1.0, 0.0, 1.0),
                real(1.0, 0.0, 0.0),
                real(0.0, -1.0, 1.0),
            ],
            rows: vec![
                row(
                    &[(0, -10.0), (1, -1e-8), (3, -10.0)],
                    ComparisonOp::Eq,
                    -0.0,
                ),
                row(&[(0, -1e-7), (1, 10000.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0],
        });
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): a vertex that misses a bound by the
    /// round-off of solving for it, by more than the tolerance the engine
    /// holds the var to, with nothing left to move. `lp_wide_1` of the
    /// property stress run of R1a: three equalities through
    /// `x* = (0, -230094, 0)`; after two dual pivots every non-basic var is
    /// an equality's fixed slack, and the vertex puts `x2` `3.3e-10` (scaled)
    /// past its upper bound of zero, three times the tolerance its rows'
    /// reserve caps it at. Up to the infeasibility proof, the dual ratio
    /// test took "no var can enter" for a proof and the solve came back
    /// `Infeasible` (with presolve on and off, as at 1dcd664). The row's
    /// certificate, summed from the data, puts the gap at `-6.7e-14`: it
    /// proves nothing (the vertex is off by round-off, not by the data), and
    /// the solve fails loudly. The full contract of R4 accepts the vertex.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_vertex_off_its_bound_by_round_off_with_nothing_to_move_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, 0.0, 0.0),
                real(-1.0, -493315.0, -230094.0),
                real(0.1, -1681001.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(1, -1.1275759759608115), (2, 0.1765625)],
                    ComparisonOp::Eq,
                    259448.46661272697,
                ),
                row(&[(1, -1.0), (2, 184.375)], ComparisonOp::Eq, 230094.0),
                row(&[(1, 1.0), (2, -0.01)], ComparisonOp::Eq, -230094.0),
            ],
            xstar: vec![0.0, -230094.0, 0.0],
        });
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): a MILP whose node LP is feasible
    /// but that no var can repair. `b3` of the property stress run of R1b:
    /// `-0.1 x2 + x5 = 0`, `1e4 x4 - 1e-6 x5 <= 10000010003` and
    /// `-1e4 x2 + 1e-8 x4 = 0.00500001`, with `x4 ∈ [500001, 1000002]`, pin
    /// `x5` at zero (`x* = 0` but for `x4 = 500001`). At the root node the
    /// vertex puts `x5` `3.05e-12` (scaled, `5e-8` in the user's units) past
    /// its upper bound of zero, beyond the tolerance its rows' reserve caps
    /// it at, and the only var that could move it is the slack of the
    /// second row, unbounded, with an entry of `3.2e-16`, below every pivot
    /// floor. Up to the infeasibility proof, the dual ratio test counted that
    /// entry as zero and took the row for a proof, and the branch and bound
    /// pruned the node: the MILP came back `Infeasible` (presolve on and
    /// off). The certificate's gap is `-1.6e-22`, no proof, so the node is
    /// not pruned: its LP fails loudly, and after the retry from the slack
    /// basis so does the MILP. What would repair it is a pivot on the tiny
    /// entry (the absolute pivot tolerances with a refined row of R3) or the
    /// full contract of R4, under which the vertex verifies.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_node_lp_no_var_can_repair_is_not_pruned_and_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(0.0, 0, 0),
                real(-1.0, 0.0, 0.0),
                real(0.0, 0.0, 1.0),
                real(-1.0, 0.0, 0.0),
                real(-1.0, 500001.0, 1000002.0),
                real(1.0, -1.0, 0.0),
            ],
            rows: vec![
                row(&[(2, -0.1), (5, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(4, 10000.0), (5, -1e-6)], ComparisonOp::Le, 10000010003.0),
                row(&[(2, -10000.0), (4, 1e-8)], ComparisonOp::Eq, 0.00500001),
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0, 500001.0, 0.0],
        });
    }

    /// A MILP node whose LP misses a bound by less than the user's tolerance
    /// but more than the engine's is proven infeasible and pruned, as HiGHS
    /// prunes it. `max x0` s.t. `-1e-4 x0 - 1e4 x2 = 0`, `-0.1 x2 = 0` over
    /// `x0 ∈ {-1, 0, 1}`, `x2 ∈ [-1, 0]`: the node `x0 <= -1` puts `x2` at
    /// `1e-8`, past its bound of zero, beyond the tolerance its first row's
    /// reserve caps it at but within the user's `1e-7`, and nothing can move.
    /// Its certificate's gap is exactly that `1e-8`. A proof that asked the
    /// gap to exceed the user's tolerance too left the node neither proven
    /// nor solvable, and the MILP a loud error (found by the property stress
    /// on such a proof); the proof asks what the simplex's own test asks, and
    /// the optimum is `x0 = 1` (where `x2 = -1e-8`), as HiGHS finds.
    #[test]
    fn a_node_infeasible_by_less_than_the_contract_is_pruned() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![int(1.0, -1, 1), real(0.0, 0.0, 0.0), real(0.0, -1.0, 0.0)],
            rows: vec![
                row(&[(0, -0.0001), (2, -10000.0)], ComparisonOp::Eq, -0.0),
                row(&[(2, -0.1)], ComparisonOp::Eq, -0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0],
        });
    }

    /// A MILP node whose certificate carries a basic var's round-off on a
    /// huge bound is proven infeasible all the same. `x3 ∈ [0, 64212616]` is
    /// basic at the node `x1 >= 1`, which violates the second row by
    /// thousands; the certificate's coefficient of `x3` was `1.45e-17`, the
    /// round-off of terms of magnitude `0.19`, and times `x3`'s scaled bound
    /// of `4.2e12` it alone closed the certificate's gap of `6.1e-5`: the
    /// proof failed, nothing could move, and the MILP failed loudly (found by
    /// the property stress). The refinement of the row of the basis inverse
    /// the certificate starts from, added for the next model and kept to
    /// twice the precision for [`multipliers_no_f64_holds_do_not_void_a_nodes_proof`],
    /// leaves `x3` no coefficient at all after one step, and after the second
    /// the noise of its solve, `7.7e-34`, some `3e-21` at that bound, and the
    /// node is proven. (For a while
    /// a coefficient of round-off size counted as zero whatever its var's
    /// bound, which kept this proof too but also proved a feasible model
    /// infeasible:
    /// `a_round_off_coefficient_on_a_far_bound_is_no_proof_and_is_a_known_limitation`.)
    #[test]
    fn a_round_off_coefficient_on_a_huge_bound_does_not_void_a_nodes_proof() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(1.0, 0, 0),
                int(1.0, 0, 2),
                real(0.0, 0.0, 0.0),
                real(0.0, 0.0, 64212616.0),
                real(1.0, -1.0, 0.0),
            ],
            rows: vec![
                row(&[(3, -10.0), (4, -1.7421875e-7)], ComparisonOp::Eq, -0.0),
                row(&[(1, -1.0), (4, -10000.0)], ComparisonOp::Le, 1.0),
                row(&[(1, -0.0001), (3, 10000.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 5],
        });
    }

    /// A MILP node proven infeasible through an ill-conditioned basis. At the
    /// node `x1 <= -10` (of `x1 ∈ [-11, 0]`), whose rows allow `x1 >= -9.99`
    /// only, the basis mixes coefficients from `1e-5` to `1e4`, and the row
    /// of its inverse leaves the basic `x3 ∈ [-2848554, 0]` a certificate
    /// coefficient of `8.4e-16`: the residual of the transposed solve,
    /// `1.8e-14` of the terms it is summed from, above any fixed round-off
    /// fraction, and times `x3`'s scaled bound of `3.6e8` exactly the gap of
    /// `3.05e-7`. The proof failed, nothing could move, and the MILP failed
    /// loudly (found by the property stress); one refinement of that row, as
    /// HiGHS refines it, leaves `x3` the coefficient `1.7e-18` rounded to
    /// `f64` and `2.5e-31` kept to twice the precision, a second none, and
    /// the node proven.
    #[test]
    fn an_ill_conditioned_basis_does_not_void_a_nodes_proof() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(1.0, 0, 0),
                int(1.0, -11, 0),
                real(-1.0, 0.0, 0.0),
                real(-1.0, -2848554.0, 0.0),
                real(1.0, -1.0, 0.0),
                real(-1.0, 0.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(1, -0.0001), (3, -10000.0), (4, -1.0)],
                    ComparisonOp::Le,
                    1.0,
                ),
                row(&[(1, 1e-5), (3, -1.0), (4, 100.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 6],
        });
    }

    /// A MILP node whose certificate no `f64` multipliers combine exactly is
    /// proven infeasible all the same. `max -x1` s.t.
    /// `-2e-8 x1 - 12.5 x2 = 0` and `-2e-7 x0 - 1e-6 x1 + 162.5 x2 = 0` over
    /// `x0 ∈ {0, 1}`, `x1 ∈ {-2, -1, 0}` and `x2 ∈ [-45451714, 0]`: the rows
    /// pin `x1` at zero, and the node `x1 <= -1` misses them by `6.5e-7`
    /// (scaled). Its certificate combines the rows with multipliers no `f64`
    /// holds, and with the row of the basis inverse refined but rounded to
    /// `f64`, as HiGHS rounds it, the basic `x2`'s coefficient was
    /// `8.7e-17`, the representation error of the row beside terms of
    /// magnitude `3.1`: counted at `x2`'s scaled bound of `-2.9e9`, it left a
    /// gap of `4.0e-7`, short of the allowance `4e-7` by a hair, and the
    /// MILP failed loudly (found by the property stress on a version that
    /// counted such coefficients at finite bounds, as the proof must, but
    /// rounded the refined row; two more nodes failed the same way, on
    /// bounds of `6.9e6` and `2.4e8`). The refinement now keeps its
    /// correction to twice the precision, and `x2`'s coefficient is zero.
    #[test]
    fn multipliers_no_f64_holds_do_not_void_a_nodes_proof() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(0.0, 0, 1),
                int(-1.0, -2, 0),
                real(0.0, -45451714.0, 0.0),
            ],
            rows: vec![
                row(&[(1, -2e-8), (2, -12.5)], ComparisonOp::Eq, -0.0),
                row(&[(0, -2e-7), (1, -1e-6), (2, 162.5)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0, 0.0, 0.0],
        });
    }

    /// The model the review of the infeasibility proof built:
    /// `min -x - y - t - z` s.t. `x + y + z + q >= 3` and
    /// `t - (1 + 2^-50) z - q >= 1e-6` over `x, y, t ∈ [0, 1]` (integer if
    /// asked), `z ∈ [z_min, 1]`, `q ∈ [-1e10, 1e10]` (`q` free, the review's
    /// form, behaves the same). Together the rows ask
    /// `x + y + t - 2^-50 z >= 3 + 1e-6`: only `z`'s coefficient `-2^-50`,
    /// which the rows' own data leave, makes room for them, at
    /// `z <= -2^50 · 1e-6 ≈ -1.13e9`, where the optimum is. The model is
    /// feasible in exact arithmetic at `x*`.
    fn room_through_a_round_off_coefficient(integer: bool, z_min: f64) -> Model {
        let unit = |obj| {
            if integer {
                int(obj, 0, 1)
            } else {
                real(obj, 0.0, 1.0)
            }
        };
        Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                unit(-1.0),
                unit(-1.0),
                unit(-1.0),
                real(-1.0, z_min, 1.0),
                real(0.0, -1e10, 1e10),
            ],
            rows: vec![
                row(
                    &[(0, 1.0), (1, 1.0), (3, 1.0), (4, 1.0)],
                    ComparisonOp::Ge,
                    3.0,
                ),
                row(
                    &[(2, 1.0), (3, -(1.0 + 2f64.powi(-50))), (4, -1.0)],
                    ComparisonOp::Ge,
                    1e-6,
                ),
            ],
            xstar: vec![1.0, 1.0, 1.0, -2e9, 2000000001.0000005],
        }
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): a feasible LP whose only room is a
    /// coefficient of round-off size on a far bound.
    /// [`room_through_a_round_off_coefficient`] with `z >= -2e9`: the dual
    /// simplex ends with a row that only `z` can repair, through an entry of
    /// `8.9e-16` beside ones, below every pivot floor. The row's certificate
    /// gives `z` the coefficient `-2^-50`, round-off beside its terms (ones)
    /// but the data's own. Counted as zero whatever its bound, as HiGHS
    /// counts proof entries below `1e-14`, it proved the model infeasible
    /// (a gap of `1e-6`), and the solve returned `Infeasible`, with presolve
    /// on and off (as HiGHS does). Counted as it is, times `z`'s bound, it
    /// leaves the certificate `7.8e-7` of room: no proof. Nothing repairs the
    /// row either: the round-off repair does not move a var by a coefficient
    /// below the round-off of its terms (moved by it, `z` goes to `-2e9`, and
    /// the primal step, whose column's round-off floor hides the same entry,
    /// takes it back, round after round), so the solve fails loudly. Only a
    /// pivot on the entry, below every reference's pivot tolerance, solves
    /// it.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_round_off_coefficient_on_a_far_bound_is_no_proof_and_is_a_known_limitation() {
        check(room_through_a_round_off_coefficient(false, -2e9));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the model of the test above with
    /// `x, y, t` integer. Its root node's LP is that LP, and the branch and
    /// bound pruned the node on the false proof: the MILP came back
    /// `Infeasible` (presolve on and off). The node's LP fails loudly now,
    /// and after the retry from the slack basis so does the MILP.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_node_lp_whose_room_is_round_off_is_not_pruned_and_is_a_known_limitation() {
        check(room_through_a_round_off_coefficient(true, -2e9));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the model of
    /// [`a_round_off_coefficient_on_a_far_bound_is_no_proof_and_is_a_known_limitation`]
    /// with `z` unbounded below. A round-off coefficient on an infinite bound
    /// once counted as zero for every var, HiGHS's concession, and the
    /// certificate proved the feasible model infeasible, with presolve on and
    /// off (HiGHS answers the same). A coefficient on an infinite bound counts
    /// as zero now only where the error of the multipliers could make it
    /// zero; this row of the basis inverse is exact, so `z`'s `-2^-50` makes
    /// the certificate's range unbounded, and nothing is proven. Nothing
    /// repairs the row either (the round-off repair moves no var toward an
    /// infinite bound), and the solve fails loudly.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_round_off_coefficient_on_an_infinite_bound_is_no_proof_and_is_a_known_limitation() {
        check(room_through_a_round_off_coefficient(
            false,
            f64::NEG_INFINITY,
        ));
    }

    /// The model the review of the basic vars' concession built:
    /// `min -L` s.t. `L + j + w = 1.000001`, `-(1 + 2^-50) j - w = 0` and
    /// `j + v <= 0` over `L, v ∈ [0, 1]` (`L` integer if asked) and `j, w`
    /// free. The first two rows ask `L - 2^-50 j = 1.000001`: only `j`'s
    /// coefficient `-2^-50` makes room for them, at `j ≈ -1.13e9`, where the
    /// optimum `L = 1` is. The model is feasible in exact arithmetic at `x*`.
    fn room_through_a_basic_vars_coefficient(integer: bool) -> Model {
        Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                if integer {
                    int(-1.0, 0, 1)
                } else {
                    real(-1.0, 0.0, 1.0)
                },
                real(0.0, f64::NEG_INFINITY, f64::INFINITY),
                real(0.0, f64::NEG_INFINITY, f64::INFINITY),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, 1.0), (1, 1.0), (2, 1.0)], ComparisonOp::Eq, 1.000001),
                row(
                    &[(1, -(1.0 + 2f64.powi(-50))), (2, -1.0)],
                    ComparisonOp::Eq,
                    0.0,
                ),
                row(&[(1, 1.0), (3, 1.0)], ComparisonOp::Le, 0.0),
            ],
            xstar: vec![1.0, -1125899906.75, 1125899906.750001, 0.0],
        }
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): a feasible LP whose only room is
    /// a basic var's coefficient of round-off size in what a certificate
    /// leaves. [`room_through_a_basic_vars_coefficient`]: the dual simplex
    /// ends at the basis `{j, w}` and the second row's slack, `1e-6` above
    /// its bound of zero, and only the third row's slack, unbounded, could
    /// repair that row, through an entry of `2^-50` beside ones, below every
    /// pivot floor. The row of the basis inverse, `(1, 1, 2^-50)`, needs the
    /// infinite side of that non-basic slack, so the certificate leaves the
    /// third row out, and the free basic `j` keeps the coefficient `-2^-50`,
    /// `4.4e-16` of its terms. The concession for a basic var's coefficient
    /// on an infinite bound stopped at the `f64` round-off floor of its terms
    /// and took it for a residual, and the certificate proved the model
    /// infeasible (a gap of `1e-6`), with presolve on and off (HiGHS's LP
    /// answers the same). A coefficient counts as zero now only where the
    /// error of the multipliers could make it zero, and this row of the basis
    /// inverse is exact: counted, the coefficient makes the range unbounded,
    /// and nothing is proven. Nothing repairs the row either, and the solve
    /// fails loudly. Only a pivot on the entry, below every reference's pivot
    /// tolerance, solves it.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_basic_vars_coefficient_left_by_a_row_left_out_is_no_proof_and_is_a_known_limitation() {
        check(room_through_a_basic_vars_coefficient(false));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the test above without presolve,
    /// on the model as written, which ends at the same basis; it too was
    /// proven infeasible, and it too fails loudly now.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_basic_vars_coefficient_left_by_a_row_left_out_without_presolve_is_a_known_limitation() {
        let model = room_through_a_basic_vars_coefficient(false);
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        assert_solves_within_contract_without_presolve(&model);
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the model of the tests above with
    /// `L` integer. Its root node's LP is that LP, and the branch and bound
    /// pruned the node on the false proof: the MILP came back `Infeasible`
    /// (presolve on and off), where its optimum is `L = 1` (HiGHS finds it).
    /// The node's LP fails loudly now, and after the retry from the slack
    /// basis so does the MILP.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_node_lp_whose_room_is_a_basic_vars_coefficient_is_not_pruned_and_is_a_known_limitation() {
        check(room_through_a_basic_vars_coefficient(true));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the test above without presolve;
    /// its root node was pruned the same way, and it fails loudly now too.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_node_lp_whose_room_is_a_basic_vars_coefficient_without_presolve_is_a_known_limitation() {
        let model = room_through_a_basic_vars_coefficient(true);
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        assert_solves_within_contract_without_presolve(&model);
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): presolve's forcing row
    /// `x1 + 100 x4 = 8388700` fixes `x1` at its lower bound `-2^-30`, where
    /// the row allows it within the row's tolerance, and
    /// `-1000 x1 + x4 / 1000 = 83.887`, which pins `x1` at zero, then misses
    /// by 9.3e-7, beyond the contract: a feasible model (`x1 = 0`,
    /// `x4 = 83887`) comes back `Infeasible` with presolve on. The instance
    /// the properties found (a coefficient of 100, the pinning row a
    /// singleton) is fixed by processing singleton rows first and judging a
    /// row whose vars are all fixed at the full contract; this one needs a
    /// forcing test that accounts for the var's other rows.
    #[test]
    #[should_panic(expected = "returned Infeasible")]
    fn forcing_row_fixing_a_var_another_row_pins_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -(2.0f64.powi(-30)), 1.0),
                real(1.0, 83887.0, 83888.0),
            ],
            rows: vec![
                row(&[(0, -1.0), (1, -100.0)], ComparisonOp::Eq, -8388700.0),
                row(&[(0, -1000.0), (1, 1e-3)], ComparisonOp::Eq, 83.887),
            ],
            xstar: vec![0.0, 83887.0],
        });
    }
}
