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
//! See §7 of `ARCHITECTURE.md`. Three fixtures were pinned this way and all
//! three were fixed: a presolve forcing rule, a skipped verification, and a
//! dual ratio test that took "no pivot" for a proof of infeasibility (the
//! `wide_chain` fixtures). None is pinned today.

#[cfg(test)]
mod regression_tests {
    use crate::tests::oracle::{
        assert_solves_feasibly_within_contract, assert_solves_within_contract,
        assert_solves_within_contract_without_presolve, xstar_is_feasible, Model, Row, Var,
    };
    use crate::{ComparisonOp, Error, OptimizationDirection, SolveOptions};

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
    /// ulp past, the dual phase undid the step, and after eight rounds the
    /// solve gave up (`InternalError`). The Harris allowance leaves the
    /// round-off of recomputing a var out of its tolerance.
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
}
