//! Regression tests on feasible-by-construction models across variable
//! magnitudes and coefficient ranges (the properties are in `scaling.rs`).
//! Each model is written around a point `x*` that satisfies every bound and
//! every row within the documented tolerance, and its objective is bounded, so
//! `solve()` must return an optimal solution whose rows hold within that
//! tolerance and whose objective is no worse than at `x*` — the contract in
//! `oracle.rs`.
//!
//! A model the engine gets wrong and that cannot be fixed at once is pinned
//! here as a KNOWN LIMITATION with `#[should_panic]`, which records today's
//! answer: a fix then shows up as a FAILING test, and deleting the attribute
//! (the body already asserts the right answer) makes it an ordinary
//! regression test. See §7 of `ARCHITECTURE.md`.

#[cfg(test)]
mod regression_tests {
    use crate::tests::oracle::{
        assert_solves_feasibly_within_contract, assert_solves_within_contract,
        assert_solves_within_contract_in_time, assert_solves_within_contract_without_presolve,
        xstar_is_feasible, Model, Row, Var,
    };
    use crate::{ComparisonOp, Error, OptimizationDirection, SolveOptions, TerminationReason};
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
    /// with variable magnitudes up to 1e6, must hold within its tolerance of
    /// 1.3e-7 whatever the pivot path: an eta file storing the pivot row's
    /// factor as `1 - 1/pivot` would reach a pivot of 7e6 here and report the
    /// basic variable 5.7e-7 off the row.
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

    /// Four rows pin one variable of magnitude 7e6, and two of the equalities
    /// agree only to 1.3e-8, the rounding of their right-hand sides at 1e8
    /// activity, well inside the documented `1e-7`: feasible, not `Infeasible`.
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

    /// Three variables of magnitudes 5e4 to 7e8 under three equalities whose
    /// right-hand sides are rounded within the round-off allowance of their
    /// 1e10 activities: feasible, not `Infeasible`.
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
    /// of activity 2e8, whose rounding puts the relaxation's value 2e-8
    /// outside the bound, far inside the documented tolerance: neither the
    /// relaxation nor the model is infeasible (`x* = (-7, -1.885e8)`).
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
    /// magnitude 8e4: the reported point must hold every row within `1e-7`,
    /// not miss two equalities by 2.2e-4 and 1.2e-5.
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
    /// the model has an optimum, not `Unbounded` as it would be with every
    /// coefficient below `1e-10` taken for zero.
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

    /// `y = 1e5 x` with `x` in `[0, 5e-8]`, a whole range narrower than the
    /// user's tolerance: maximizing `y` needs `x` at its upper bound, which
    /// the solve must reach although every value of `x` lies within the
    /// user's tolerance of both bounds. Taking `x` for fixed at 0 would answer
    /// 0, `5e-3` below the exact optimum (ARCHITECTURE §3).
    #[test]
    fn a_continuous_var_narrower_than_its_tolerance_reaches_the_exact_optimum() {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![real(0.0, 0.0, 5e-8), real(1.0, 0.0, 1.0)],
            rows: vec![row(&[(1, 1.0), (0, -1e5)], ComparisonOp::Eq, 0.0)],
            xstar: vec![5e-8, 5e-3],
        };
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// The MILP form: `y + 1e4 x - z >= 4e-4` with `x` in `[0, 5e-8]`, `y` in
    /// `[0, 1]` and `z` integer in `[0, 1]`. The minimum of `y` is 0, at `x`'s
    /// upper bound (`1e4 · 5e-8 = 5e-4`), `z = 0`.
    #[test]
    fn a_continuous_var_narrower_than_its_tolerance_reaches_the_exact_optimum_of_a_milp() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![real(0.0, 0.0, 5e-8), real(1.0, 0.0, 1.0), int(0.0, 0, 1)],
            rows: vec![row(
                &[(1, 1.0), (0, 1e4), (2, -1.0)],
                ComparisonOp::Ge,
                4e-4,
            )],
            xstar: vec![5e-8, 0.0, 0.0],
        };
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// `max 1e6 x + y` s.t. `y - x <= 0` with `x` in `[0, 5e-8]` and `y` in
    /// `[0, 1]`: `x`'s whole range lies within its tolerance, and the exact
    /// optimum, `x = y = 5e-8`, needs `x` at its upper bound, a gain of `5e-2`
    /// that comes through the objective, which no row shows. A non-basic var's
    /// state is the bound it sits at, not a tolerance test, so `x` is not
    /// taken for fixed (ARCHITECTURE §3).
    #[test]
    fn a_continuous_var_narrower_than_its_tolerance_reaches_its_objective_gain() {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![real(1e6, 0.0, 5e-8), real(1.0, 0.0, 1.0)],
            rows: vec![row(&[(1, 1.0), (0, -1.0)], ComparisonOp::Le, 0.0)],
            xstar: vec![5e-8, 5e-8],
        };
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// `x0` is pinned to `2^28` by an equality whose other coefficient is
    /// `1e-7`, so the row's terms span fifteen orders of magnitude, and
    /// `x* = [268435456, 0]` satisfies both rows exactly. The row's round-off,
    /// `5e-6`, exceeds `x1`'s whole term range, `2.5e-8`, so its minimum
    /// activity meets the bound wherever `x1` sits: read as "every var at its
    /// extreme", that would fix `x1` at `0.25` against `-x1 = 0`. Presolve
    /// fixes no continuous var, and fixes an integer var by a forcing row only
    /// where every term of the row is resolvable above its round-off
    /// (`Work::forcing_fixings`).
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
    /// integer, two fixed) under three rows whose coefficients run from `1e-8`
    /// to `1e4`. Rows 0 and 2 tie `x3` to `x2`, `x3 ≈ 1e-13 (x2 - shift)`, and
    /// `x3` lies in `[0, 1]`, so the rows hold only at `x2 = shift` with
    /// `x3 = x4 = 0`; `x2`'s bounds are `[lo, hi]`.
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

    /// `x* = 0` satisfies all three rows exactly, and the solve must find a
    /// solution with presolve on and off. The start puts `x2` at its lower
    /// bound `-1000001`, and the only tableau entry that can carry it back is
    /// `1.5e-15` against a row maximum of `1.95`, below the round-off of
    /// computing the row: no pivot, but no proof of infeasibility either.
    /// `x2`'s flip to `0` carries the leaving var of row 0 exactly the
    /// `1.19e-11` (scaled) it is short. The integer var plays no part.
    #[test]
    fn wide_coefficients_reached_by_a_bound_flip() {
        check(wide_chain(-1000001.0, 0.0, 0.0));
        assert_solves_within_contract_without_presolve(&wide_chain(-1000001.0, 0.0, 0.0));
    }

    /// [`wide_coefficients_reached_by_a_bound_flip`] shifted so that `x2`'s
    /// feasible value is the bound farther from zero: a start at the bound
    /// nearer zero must flip too.
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

    /// The chain with `x2` confined to `[-2e7, -1e7]`: `x3 ≈ 1e-13 x2` is then
    /// at most `-1e-6`, beyond every tolerance, and even `x2`'s flip to `-1e7`
    /// leaves row 0 short. The row proves the model infeasible, and moving
    /// vars through entries below round-off must not turn that proof into a
    /// guess.
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
    /// The exactly integral optimum must be reported, not rejected by the
    /// engine's own feasibility guard as an `InternalError`. The recompute of
    /// the periodic refactorization can leave the basic values missing row 2
    /// by four times its tolerance, so it is no verification: `verify_primal`
    /// must still refine them.
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

    /// Variable magnitudes around `8.5e6`. Row 2's equality slack sits past
    /// its bound within its tolerance when a primal step moves it further
    /// out: the Harris limit leaves it only the rest of its tolerance, counted
    /// from where it sits. Granted its whole tolerance again, it ends at 1.3
    /// times its tolerance, and the phases undo and redo the step until the
    /// solve fails with `InternalError`.
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

    /// An integer var with a coefficient of 1e-8 in an equality may leave the
    /// row's solution by up to the row's tolerance over that coefficient, ten
    /// units here. Presolve's integer rounding must leave the guard the
    /// round-off of its own evaluation: a point exactly at the contract
    /// evaluates a few ulps over it, and an exactly integral candidate is
    /// rejected (`InternalError`). The tolerance leaves the objective real
    /// freedom, so only feasibility is checked.
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

    /// Models whose only feasible region is reached through a genuine pivot
    /// of 4e-15 to 2e-14 of its column (1e-8 beside 1e4 in a row). Both
    /// computations of such a pivot agree exactly, which tells it from a
    /// round-off entry of the same size (whose two computations differ by
    /// percents); a threshold on size alone refuses it and answers
    /// `Infeasible`.
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

    /// LPs whose solve pivots on an entry some `1e-15` of its column (genuine,
    /// or round-off that both computations share) into a basis the
    /// factorization finds singular. The basis is repaired, a slack in place
    /// of the dependent column, and the solve goes on to an answer within the
    /// contract, with presolve on and off.
    #[test]
    fn lps_through_a_basis_the_factorization_finds_singular_solve() {
        let check_on_and_off = |model: Model| {
            assert!(xstar_is_feasible(&model), "fixture error");
            assert_solves_within_contract(&model);
            assert_solves_within_contract_without_presolve(&model);
        };
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, 0.0, 0.0),
                real(-1.0, 0.0, 1.0),
                real(0.0, -1.0, 50.0),
                real(1.0, -1.0, 0.0),
            ],
            rows: vec![
                row(&[(2, -1.0), (3, 10000.0)], ComparisonOp::Eq, -50.0),
                row(&[(1, 1.0), (3, -1e-5)], ComparisonOp::Eq, 0.0),
                row(&[(1, -1e-8), (2, -20000.0)], ComparisonOp::Le, 0.0),
            ],
            xstar: vec![0.0, 0.0, 50.0, 0.0],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(0.0, 0.0, 0.0),
                real(1.0, 0.0, 0.0),
                real(0.0, 0.0, 262144.0),
                real(-1.0, 0.0, 0.0),
                real(1.0, 0.0, 1.0),
                real(-1.0, -262146.0, 0.0),
            ],
            rows: vec![
                row(&[(2, 1000.0), (5, -0.01)], ComparisonOp::Eq, 0.0),
                row(&[(2, 1.0), (4, -1e-6), (5, -0.0001)], ComparisonOp::Ge, 0.0),
                row(&[(2, 1.0), (4, -1.0), (5, 1.0)], ComparisonOp::Ge, -1.0),
            ],
            xstar: vec![0.0; 6],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -1.0, 0.0),
                real(-1.0, -20.0, 0.0),
                real(1.0, 0.0, 0.0),
                real(0.0, -1.0, 0.0),
            ],
            rows: vec![
                row(&[(0, -1e-6), (3, -10000.0)], ComparisonOp::Eq, -0.0),
                row(&[(1, -1e-7)], ComparisonOp::Le, 1.0),
                row(&[(0, -20.0), (1, 1.0), (3, -1e-8)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 4],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(1.0, 0.0, 1.0),
                real(1.0, 0.0, 8388608.0),
                real(-1.0, 0.0, 1.0),
                real(0.0, 0.0, 0.0),
                real(0.0, 0.0, 0.0),
                real(-1.0, -1.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(0, -0.5), (1, 100.0), (2, 1.0), (5, 10000.0)],
                    ComparisonOp::Eq,
                    0.0,
                ),
                row(&[(1, 1.0), (5, -1e-5)], ComparisonOp::Eq, 0.0),
                row(&[(2, -2.0)], ComparisonOp::Le, 0.0),
                row(
                    &[(0, 1000.0), (1, -1e-8), (2, -1750.0)],
                    ComparisonOp::Eq,
                    0.0,
                ),
            ],
            xstar: vec![0.0; 6],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(0.0, -2.0, 0.0),
                real(0.0, 0.0, 1.0),
                real(1.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(1, 1.0), (2, -0.1)], ComparisonOp::Eq, 0.0),
                row(&[(0, -1.0), (3, -1.0)], ComparisonOp::Eq, -0.0),
                row(&[(0, 10.0), (1, -1e-8)], ComparisonOp::Le, 0.0),
                row(&[(0, -1e-7), (2, -10000.0)], ComparisonOp::Eq, -0.0),
            ],
            xstar: vec![0.0; 4],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(0.0, 0.0, 1.0),
                real(1.0, 5001.0, 10002.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, -1e-6), (1, 1.0)], ComparisonOp::Le, 10002.0),
                row(&[(0, 10000.0), (2, 0.1)], ComparisonOp::Eq, 0.0),
                row(&[(1, -100.0)], ComparisonOp::Ge, -1000101.0),
                row(
                    &[(1, -1e-6), (2, 10000.0)],
                    ComparisonOp::Eq,
                    -0.005000999999999999,
                ),
            ],
            xstar: vec![0.0, 5001.0, 0.0],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(1.0, 0.0, 0.0),
                real(-1.0, -2.0, 0.0),
                real(0.0, -1.0, 0.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(1, -100.0), (2, 1.0), (3, -100.0)], ComparisonOp::Eq, 0.0),
                row(&[(1, -1e-7), (3, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(2, -10000.0), (3, -1e-8)], ComparisonOp::Ge, -0.0),
            ],
            xstar: vec![0.0; 4],
        });
    }

    /// MILPs of the class above, whose root or node LP meets the singular
    /// basis.
    #[test]
    fn milps_through_a_basis_the_factorization_finds_singular_solve() {
        let check_on_and_off = |model: Model| {
            assert!(xstar_is_feasible(&model), "fixture error");
            assert_solves_within_contract(&model);
            assert_solves_within_contract_without_presolve(&model);
        };
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, -1, 0),
                int(0.0, 0, 0),
                real(0.0, -1.0, 0.0),
                real(1.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(2, -1.0), (3, -1.0)], ComparisonOp::Eq, -0.0),
                row(&[(3, -10.0), (4, -1e-8)], ComparisonOp::Le, 0.0),
                row(
                    &[(0, -0.001), (2, -1e-8), (4, 10000.0)],
                    ComparisonOp::Eq,
                    0.0,
                ),
            ],
            xstar: vec![0.0; 5],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(-1.0, -3, 0),
                real(-1.0, 0.0, 4.2147712653317626e-8),
                real(-1.0, 0.0, 301.0),
            ],
            rows: vec![
                row(&[(0, -1e-8), (2, -1.0)], ComparisonOp::Le, 0.0),
                row(&[(0, -2e-8), (1, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(0, -1.0), (1, -1e-8), (2, -0.01)], ComparisonOp::Eq, -0.0),
                row(&[(2, 100.0)], ComparisonOp::Ge, 0.0),
            ],
            xstar: vec![0.0; 3],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(0.0, 0, 0),
                int(-1.0, 0, 2),
                real(-1.0, 0.0, 1.0),
                real(-1.0, 0.0, 53688.0),
            ],
            rows: vec![
                row(&[(1, 1.0), (3, -0.001)], ComparisonOp::Ge, 0.0),
                row(&[(2, -1e-7), (3, 20000.0)], ComparisonOp::Eq, 0.0),
                row(&[(1, -1e-7), (2, -1.0), (3, 1.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 4],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(0.0, -1, 0),
                real(0.0, 0.0, 0.0),
                real(-1.0, -1.0, 0.0),
                real(0.0, 0.0, 0.0),
                real(-1.0, -1.0, 0.0),
                real(0.0, 0.0, 0.0),
            ],
            rows: vec![
                row(&[(0, -1.0), (2, 10.0)], ComparisonOp::Eq, 0.0),
                row(&[(2, 1e-5), (4, 10000.0)], ComparisonOp::Eq, 0.0),
                row(&[(0, 1000.0), (4, -1e-8)], ComparisonOp::Le, 0.0),
            ],
            xstar: vec![0.0; 6],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, -27, 0),
                int(-1.0, -16, 0),
                real(-1.0, -1.0, 0.0),
                real(0.0, 0.0, 0.0),
                real(-1.0, 0.0, 1.0),
            ],
            rows: vec![
                row(
                    &[(1, -1e-5), (2, -10000.0), (4, -1.0)],
                    ComparisonOp::Eq,
                    -0.0,
                ),
                row(
                    &[(0, -100.0), (1, 165.625), (2, -1e-6)],
                    ComparisonOp::Eq,
                    0.0,
                ),
                row(&[(0, 1.0), (4, -10.0)], ComparisonOp::Le, 0.0),
            ],
            xstar: vec![0.0; 5],
        });
        check_on_and_off(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, 0, 0),
                real(1.0, -2.0, 9.0),
                real(-1.0, 4194304.0, 4194304.0),
                real(-1.0, -1.0, 0.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(1, -1.0), (4, 1.0)], ComparisonOp::Eq, 2.0),
                row(
                    &[(1, -1.125e-7), (2, -2.0)],
                    ComparisonOp::Eq,
                    -8388607.999999775,
                ),
                row(
                    &[(2, -1.0), (3, -0.0001), (4, -1.0)],
                    ComparisonOp::Le,
                    -4194304.0,
                ),
                row(
                    &[(1, -1e-5), (2, 1.0), (4, 1.0)],
                    ComparisonOp::Eq,
                    4194304.00002,
                ),
            ],
            xstar: vec![0.0, -2.0, 4194304.0, 0.0, 0.0],
        });
    }

    /// `min -x + y` s.t. `x + y <= 1`, `x + (1 + 2^-48) y >= 1`, `x >= 0`, `y`
    /// free: the optimum is `-1` at `(1, 0)`, and its only optimal basis,
    /// `{x, y}`, is one the factorization finds singular. The primal step that
    /// completes it pivots on an entry `2^-48` of its column, and the repair at
    /// the step's refactorization takes `y` out. The step moved `y`'s row,
    /// which drops the record of the change back, so the phase takes `y` in
    /// again, on the eta file, and ends at the optimum instead of on the
    /// change's taboo.
    #[test]
    fn an_optimal_basis_the_factorization_finds_singular_is_reached_on_the_eta_file() {
        for presolve in [true, false] {
            let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(-1.0, (0.0, f64::INFINITY));
            let y = problem.add_var(1.0, (f64::NEG_INFINITY, f64::INFINITY));
            problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 1.0);
            let slanted = 1.0 + 2f64.powi(-48);
            problem.add_constraint([(x, 1.0), (y, slanted)], ComparisonOp::Ge, 1.0);
            let solution = match problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            }) {
                Ok(outcome) => outcome.into_solution().unwrap(),
                Err(err) => panic!("presolve {presolve}: {err:?}"),
            };
            let (objective, x, y) = (solution.objective(), solution[x], solution[y]);
            assert!(
                (objective + 1.0).abs() < 1e-9 && (x - 1.0).abs() < 1e-9 && y.abs() < 1e-9,
                "presolve {presolve}: objective {objective} at ({x}, {y})"
            );
        }
    }

    /// A model whose exact optimum, `x*` at `x2 = 5000`, is reached only
    /// through a basis the factorization finds singular (`x2` is blocked by an
    /// entry some `1e-14` of its column) solves: the rows admit
    /// `x2 = 9999.995` with `x5` `5e-8` below its bound, within its tolerance,
    /// and the simplex ends at that vertex.
    #[test]
    fn a_model_whose_exact_optimum_needs_a_singular_basis_solves() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(0.0, 0.0, 0.0),
                real(0.0, 0.0, 0.0),
                real(1.0, 5000.0, 10000.0),
                real(-1.0, -1.0, 1.0),
                real(1.0, 0.0, 0.0),
                real(1.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(3, -0.01), (5, 1.0)], ComparisonOp::Eq, 0.0),
                row(
                    &[(2, 1e-5), (3, 10000.0), (5, -1.0)],
                    ComparisonOp::Eq,
                    0.05,
                ),
                row(&[(2, 10000.0), (5, -1e-7)], ComparisonOp::Le, 99999950.0),
            ],
            xstar: vec![0.0, 0.0, 5000.0, 0.0, 0.0, 0.0],
        });
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): a model whose optimal basis the
    /// factorization finds singular fails loudly, where the origin, its only
    /// exactly feasible point, is optimal. The last primal step pivots on an
    /// entry some `1e-14` of its column into that basis (a 1-norm condition of
    /// about `4.5e14` in the scaled data); the repair takes the entering var
    /// back out, and the change back stays taboo, since the step, a zero one,
    /// moved no row. The evaluation margin keeps it pinned: held to its whole
    /// contract, `x0 - 1e-7 x4 = 0` admits `x4 = 1`, of objective `-1e9`,
    /// which the solve then returns; the engine's tolerance, short of the
    /// contract by the round-off of evaluating the row, excludes that point.
    /// The model lies outside the recommended range (a row dynamism of
    /// `1e11`).
    #[test]
    #[should_panic(expected = "leads to a singular basis")]
    fn an_optimal_basis_the_factorization_finds_singular_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.0, -1.0, 0.0),
                real(-1.0, 0.0, 0.0),
                real(-1.0, 0.0, 1.0),
                real(-1.0, 0.0, 1000001171.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, -1.0), (2, -1.0)], ComparisonOp::Eq, -0.0),
                row(&[(2, -1e-8), (3, -1000.0)], ComparisonOp::Le, 0.0),
                row(&[(0, 1.0), (4, -1e-7)], ComparisonOp::Eq, 0.0),
                row(&[(3, -1e-6), (4, 1000.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 5],
        });
    }

    /// The primal step that lowers x3 is bounded by `-2 x3 <= -1` and, through
    /// the Harris relaxation, by the equality `x2 + 1e-7 x3 = 1.0000001` at its
    /// tolerance: the step leaves the equality's slack exactly at its
    /// tolerance, the phase exit's recomputation finds it an ulp past, and the
    /// phases undo and redo the step. Each round after the first cuts the
    /// Harris allowance a hundredfold (`LOOP_HARRIS_REDUCTION`): in the third
    /// the slack blocks the step (a zero one), and the solve ends at the
    /// vertex `x3 = 1`.
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

    /// The fixed slack of `0.1 x1 + x2 = 0` sits basic `3e-17` off its value,
    /// the round-off of the updates. A primal step setting it onto its value
    /// would move the entering slack of `10 x0 - 1e-8 x2 <= 0` by that over an
    /// element of `7.8e-8`, `3.9e-10` past its bound (`1.3e-7` in user units,
    /// beyond the contract), and the repair would pivot into a singular basis.
    /// So a step that would run backwards is zero, and a fixed var leaving
    /// within its tolerance of its value stays where it sits, through the
    /// cleanup too.
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

    /// In a node LP whose branch left the integer `x0` at a bound, each dual
    /// phase after the first starts with `x0`'s reduced cost dual infeasible
    /// (the shift of the zero dual step before it given back). Flipping `x0`
    /// to its other bound makes a move the phase cannot repair and undoes
    /// through a zero step, round after round until `8 phase rounds`; a phase
    /// that cleans up after another shifts `x0`'s cost instead.
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

    /// With the integer vars fixed at 0 and -115, the four equalities pin every
    /// continuous var at zero, a primal degenerate vertex the pivots reach
    /// only within the rows' tolerances. Two regular pivots can hand a
    /// tolerance-level violation to the other equality and two absorbing
    /// pivots of its fixed slacks hand it back, a cycle cost shifting does not
    /// break: the vertex hash makes the repeated basis change taboo, and the
    /// phase ends at the optimum, 0 at `x*`. The time limit turns a regression
    /// into a failure instead of a hang.
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): the node `x0 = 2` holds only with
    /// two terms past their bounds at once, each within its tolerance: `x1`
    /// past its upper bound and the second equality's slack past zero, a
    /// point no single absorption reaches. The two equalities' slacks absorb
    /// the violation in turn, each carrying another basic var out of its
    /// tolerance for a regular pivot to move the violation back, until every
    /// violated row is taboo: the solve fails loudly ("dual simplex cycles",
    /// at the node and from the slack basis), presolve on and off. The rows
    /// lie within the recommended range.
    #[test]
    #[should_panic(expected = "dual simplex cycles")]
    fn a_node_feasible_only_through_two_tolerances_at_once_is_a_known_limitation() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(-1.0, 0, 2),
                real(-1.0, 354647636.0, 354647637.0),
                real(-1.0, 0.0, 0.0),
                real(-1.0, -1.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(0, -1e-7), (1, -0.001), (3, -1.75e-7)],
                    ComparisonOp::Eq,
                    -354647.637,
                ),
                row(
                    &[(0, 1.5e-6), (1, -1.0), (3, -1e-6)],
                    ComparisonOp::Eq,
                    -354647637.0,
                ),
            ],
            xstar: vec![0.0, 354647637.0, 0.0, 0.0],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract_in_time(&model, Duration::from_secs(10));
    }

    /// A huge cost must not make a feasible model infeasible. The model
    /// minimizes `4 x0 - 1e300 x1 - 5 x2 + 5 x3 - x4 - 2 x5` over four rows,
    /// the first two of which force `x1 = 2 x4 - 2 = 0`. Perturbed by about
    /// `1e-6 |c|^(5/4)`, a cost above about `2e251` becomes infinite and the
    /// reduced costs not numbers, so a cost of `1e20` or more takes no part in
    /// the perturbation. The small costs are round-off beside the huge one, so
    /// the objective is only that good (4 is reported, 3 is optimal), which
    /// the contract's objective slack allows.
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

    /// This model reaches its optimum, 231413.21367, within `1e-6` relative,
    /// with presolve on and off, not 231413.844, the objective at `x*` (`x1`
    /// at its lower bound instead of its upper), which the contract's
    /// objective slack cannot tell from it. Two rules of the dual simplex rest
    /// on it: the cleanup of a dual phase only restores the costs, so that
    /// `run_phases` verifies the values before it recomputes the reduced
    /// costs and runs the primal phase; and an absorbing pivot takes the
    /// absorber whose shift does the least harm when given back (here a fixed
    /// slack), not the one with the largest entry.
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

    /// This model solves within the contract, with presolve on and off; the
    /// answer is the objective at `x*`. It rests on a rebuild within a dual
    /// phase correcting a dual infeasibility by shifting the cost, a boxed
    /// var's too: a step flips a boxed var to its upper bound and the next
    /// leaves it dual infeasible there (its entry, below the pivot floor, does
    /// not bound the step), and moving it back to its lower bound would return
    /// the phase to a vertex it left, round a loop the cycling detection ends
    /// in an `InternalError`.
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

    /// A fixed var's huge coefficient must not silence the rest of its row.
    /// `min y` s.t. `1e17 f + y >= 1`, `f` fixed at zero: neither scaling
    /// touches the row (a fixed column is left out of both, and presolve keeps
    /// the row as written), so the dual ratio test sees `[1e17, 1]`. Its
    /// round-off scale leaves out the columns the scalings left out; over
    /// `f`'s entry, it would make `y`'s exact one "noise" and the row a false
    /// proof of infeasibility.
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

    /// The model above reached through the LP edit path, from `f + y >= 0`
    /// with `f ∈ [0, 10]`: `fix_var(f, 0)`, then
    /// `add_constraint(1e17 f + y >= 1)`. The added row leaves the now fixed
    /// `f` out of its equilibration, and the round-off scale leaves out only
    /// the columns fixed when the solve began, so `f`'s entry sets a floor of
    /// 100 and `y`'s exact one is "noise": no candidate, but no proof of
    /// infeasibility either (`y` has no upper bound). With the row taboo and
    /// no other left, the instable leave pivots on `y`'s one, above
    /// `INSTABLE_PIVOT_TOL`, and the answer is `y = 1`.
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
    /// `g ∈ [0, 1]`, `y >= 0`: the optimum is `g = 1`, `y = big`. The engine
    /// takes matrix entries this large, so its column factors are not
    /// bounded: bounded to `2^±20`, they would leave the row `[1, 1.1e-18]` at
    /// `big = 1e30`, `y`'s entry below the round-off floor with no bound to
    /// flip to, and the row a false proof of infeasibility.
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

    /// Issue #3's model with `z ≤ B` for `B` up to `4e15`, below the bound
    /// infinity of 2^52. `z` starts at `B`, where the rows' tolerances are
    /// taken (about 2.5 in the scaled row at 1e15), so `3x + 2y + z ≤ 20`
    /// passes the dual phase violated by 1.375; the primal phase lets its
    /// slack leave off its bound, and its cleanup must move the slack onto the
    /// bound for the dual simplex to repair the row
    /// (`a_slack_left_off_its_bound_beyond_its_current_tolerance_moves_onto_it`
    /// in `src/solver/tests.rs`).
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

    /// A row only entries below the round-off floor can repair: the origin is
    /// the only exact point, and a primal step that lowers `x0` passes
    /// `-1e4 x2 + 1e-3 x3 >= 0` through a tableau entry of `3e-15`, below its
    /// column's floor. Within `x2`'s bound tolerance (`x2 = -1e-11`) the rows
    /// admit `x3 = -1e-4`, a vertex where the phases agree.
    #[test]
    fn a_row_only_entries_below_the_floor_repair_solves() {
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

    /// The second row makes `x1 = 1e-11 x0`, so the exact rows need `x0 >= 0`
    /// for `x1 >= 0`. Within `x1`'s bound tolerance, `x1 = -1e-11` meets both
    /// equalities with `x0 = -1` at a vertex, where the solve ends.
    #[test]
    fn an_equality_met_within_a_bound_tolerance_at_a_vertex_solves() {
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): the phases alternate over a row
    /// only entries below the round-off floor can repair, and the solve fails
    /// with `8 phase rounds` where the origin is feasible. The better point,
    /// `x3 = -1`, `x4 = 1000`, needs the equality `1e4 x2 - 1e-7 x3 = 0` at
    /// exactly its whole contract, which the engine's tolerance stops short
    /// of by the round-off of evaluating the row. The primal step toward it
    /// passes that equality through an entry below its column's round-off
    /// floor and leaves it beyond its tolerance; the dual phase takes `x3`
    /// back, and the step comes again. The fix is a degenerate pivot on the
    /// tiny entry, which needs absolute pivot tolerances.
    #[test]
    #[should_panic(expected = "8 phase rounds")]
    fn phases_alternating_over_an_equality_a_long_step_passes_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, 0.0, 0.0),
                real(0.0, 0.0, 0.0),
                real(-1.0, 0.0, 1.0),
                real(1.0, -1.0, 0.0),
                real(1.0, 0.0, 256167.0),
            ],
            rows: vec![
                row(&[(3, -1.0), (4, -0.001)], ComparisonOp::Eq, -0.0),
                row(&[(2, -1e-5), (4, -10000.0)], ComparisonOp::Le, 0.0),
                row(&[(2, 10000.0), (3, -1e-7)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 5],
        });
    }

    /// A MILP whose `x4 ∈ [-5.96e-8, 0]` is narrower than its bound
    /// tolerance. Counted at both bounds, fixed, by a state taken from that
    /// tolerance, `x4` never moved, and at the node `x1 ∈ [2, 3]`, which the
    /// rows make infeasible, the phases alternated round after round. `x*` is
    /// optimal.
    #[test]
    fn a_var_whose_range_is_narrower_than_its_tolerance_is_not_fixed() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, -2, 0),
                int(1.0, 0, 4),
                real(-1.0, 0.0, 0.0),
                real(-1.0, 32768.0, 32769.0),
                real(-1.0, -5.960464477539063e-8, 0.0),
                real(-1.0, 0.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(1, -0.001), (3, 1000.0), (4, -1e-7)],
                    ComparisonOp::Eq,
                    32768000.0,
                ),
                row(&[(3, -1.0)], ComparisonOp::Le, -32768.0),
                row(
                    &[(0, 2.0), (1, 1.0), (3, 1.0), (4, -1.0)],
                    ComparisonOp::Eq,
                    32768.0,
                ),
            ],
            xstar: vec![0.0, 0.0, 0.0, 32768.0, 0.0, 0.0],
        });
    }

    /// A vertex that misses a bound by the round-off of solving for it: every
    /// non-basic var is an equality's fixed slack, and the rows of magnitude
    /// `2.6e5` put `x2` `4.7e-10` past its upper bound, within its tolerance.
    /// `x* = (0, -230094, 0)`.
    #[test]
    fn a_vertex_off_its_bound_by_round_off_is_within_its_vars_tolerance() {
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

    /// A MILP node LP whose vertex no pivot can move: the rows pin `x5` at
    /// zero, the root vertex puts it `5e-8` past its upper bound, within its
    /// tolerance, and only the second row's unbounded slack could pivot,
    /// through an entry of `3.2e-16`, below every pivot floor.
    /// `x* = 0` but for `x4 = 500001`.
    #[test]
    fn a_node_lp_vertex_no_pivot_can_move_is_within_its_vars_tolerance() {
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

    /// A vertex whose basic vars miss their bounds by the round-off of rows
    /// of magnitude `7.7e7` (`x0` and `x3` some `2.5e-10` below zero, `x2`
    /// `2.5e-11`), within their tolerances. The optimum is `x*`'s, `77137`.
    #[test]
    fn a_basic_var_off_its_bound_by_its_rows_round_off_is_within_its_tolerance() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.0, 0.0, 1.0),
                real(1.0, -1.0, 0.0),
                real(1.0, 0.0, 1.0),
                real(-1.0, 0.0, 2684355.000000002),
                real(-1.0, -77137.0, -64653.999999999156),
            ],
            rows: vec![
                row(
                    &[(0, -1e-7), (1, -1.0), (2, -1.0), (3, 1.0), (4, 1000.0)],
                    ComparisonOp::Eq,
                    -77137000.0,
                ),
                row(&[(1, 100.0), (4, 1.0)], ComparisonOp::Eq, -77137.0),
                row(&[(0, 1.0), (1, -0.0001), (2, -10.0)], ComparisonOp::Eq, 0.0),
                row(&[(0, 1.0), (1, -1.0), (3, -1.0)], ComparisonOp::Eq, 0.0),
                row(
                    &[(0, -0.01), (2, -1.0), (3, 1.0), (4, -1.0)],
                    ComparisonOp::Eq,
                    77137.0,
                ),
            ],
            xstar: vec![0.0, 0.0, 0.0, 0.0, -77137.0],
        };
        assert!(xstar_is_feasible(&model), "fixture error: rows at x*");
        assert_solves_within_contract_in_time(&model, Duration::from_secs(10));
    }

    /// A vertex off its bound whose only pivot is an entry below round-off:
    /// the rows put `x1` `5e-8` past its upper bound, within its tolerance,
    /// and the second row's slack, unbounded, reaches it through `5e-16`
    /// only. The vertex's objective is `x*`'s.
    #[test]
    fn a_vertex_off_its_bound_behind_a_round_off_entry_is_within_its_tolerance() {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, 0.0, 0.0),
                real(0.0, -1.0, 0.0),
                real(0.0, 0.0, 1.0),
                real(0.0, -1.0, 50000001.0),
            ],
            rows: vec![
                row(&[(1, 1.0), (2, -0.001)], ComparisonOp::Eq, 0.0),
                row(&[(1, -1e-8), (3, 1.0)], ComparisonOp::Ge, 0.0),
                row(&[(2, 10000.0), (3, 1e-8)], ComparisonOp::Eq, 0.50000001),
            ],
            xstar: vec![0.0, 0.0, 0.0, 50000001.0],
        };
        assert!(xstar_is_feasible(&model), "fixture error: rows at x*");
        assert_solves_within_contract_in_time(&model, Duration::from_secs(10));
    }

    /// A vertex computed to the round-off of rows of magnitude `4.9e6`, next
    /// to `x0`'s upper bound of 0, lands within the contract: `x0` and `x1`,
    /// whose room in `-200 x0 - 10 x1 <= 6e-8` is what can absorb that
    /// round-off, each keep their whole bound tolerance.
    #[test]
    fn a_vertex_computed_through_large_rows_next_to_a_bound_solves() {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(1.0, -1.0, 0.0),
                real(0.0, -1.0, 0.0),
                real(1.0, -27964045.0, -20294453.0),
            ],
            rows: vec![
                row(
                    &[(0, -200.0), (1, -10.0)],
                    ComparisonOp::Le,
                    5.960464477539063e-8,
                ),
                row(
                    &[(0, -1.0), (1, 0.001), (2, -0.17500000000000002)],
                    ComparisonOp::Eq,
                    4893707.875000001,
                ),
                row(&[(1, -0.0001), (2, 1.0)], ComparisonOp::Ge, -27964045.0),
            ],
            xstar: vec![0.0, 0.0, -27964045.0],
        };
        assert!(xstar_is_feasible(&model), "fixture error: rows at x*");
        assert_solves_feasibly_within_contract(&model);
    }

    /// The rows give `x2 = -1e-8 x0` and pin `x2` at zero, so the exact
    /// optimum is `x0 = 0`; `x0 = 1` needs the second row at `1e-9`, within
    /// its tolerance. The root LP ends at `x0 = 7e-9`, integral within
    /// `int_tol`, and the search closes there: its objective is the root's
    /// bound, and a point better only within the tolerances is not sought
    /// (ARCHITECTURE §7).
    #[test]
    fn a_root_lp_point_integral_within_int_tol_closes_the_search() {
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

    /// A MILP whose integer `x1` has one feasible value, 1, where the rows hold
    /// only within their tolerances: `x*` misses the widest row (magnitude
    /// `2e10`) by `1.2e-6`, within half its contract, and the others by less.
    fn integer_value_met_within_the_rows_tolerances() -> Model {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(1.0, 0, 0),
                int(-1.0, 0, 2),
                real(-1.0, -1048577.625000061, -1048575.0000000596),
                real(1.0, 0.0, 0.0),
                real(-1.0, -1.0, 1.0),
                real(0.0, 0.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(1, -2e-8), (2, -10.0), (4, -0.00175)],
                    ComparisonOp::Eq,
                    10485760.000000576,
                ),
                row(&[(1, -1e-6), (4, 1.0)], ComparisonOp::Eq, -1e-6),
                row(
                    &[(1, -0.01), (2, -10000.0), (4, 0.5)],
                    ComparisonOp::Eq,
                    10485759999.990595,
                ),
                row(&[(2, 1.0)], ComparisonOp::Eq, -1048576.0000000596),
            ],
            xstar: vec![0.0, 1.0, -1048576.0000000596, 0.0, 0.0, 0.0],
        };
        // A point within the contract exists, so `Infeasible` is wrong.
        assert!(xstar_is_feasible(&model), "fixture error: rows at x*");
        for (v, &x) in model.vars.iter().zip(&model.xstar) {
            assert!(
                x >= v.lo && x <= v.hi && (!v.integer || x == x.round()),
                "fixture error: bounds at x*"
            );
        }
        model
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the node `x1 = 1` is pruned on a
    /// proof that takes the rows as exact. In the node's basis `x1` is basic,
    /// and the three equalities that hold it, combined, keep it at least
    /// `7e-5` from 1, beyond its tolerance: the proof grants the leaving var
    /// its tolerance and the rows none, so the MILP is reported `Infeasible`
    /// where `x*` is optimal.
    #[test]
    #[should_panic(expected = "returned Infeasible")]
    fn a_node_feasible_only_within_its_rows_tolerances_is_pruned_and_is_a_known_limitation() {
        check(integer_value_met_within_the_rows_tolerances());
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the test above without presolve,
    /// which prunes the node the same way.
    #[test]
    #[should_panic(expected = "returned Infeasible")]
    fn a_node_feasible_only_within_its_rows_tolerances_without_presolve_is_a_known_limitation() {
        assert_solves_within_contract_without_presolve(
            &integer_value_met_within_the_rows_tolerances(),
        );
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): `2 x0 - x1 = 0`, `x1 - x2 = 0`
    /// and `x2 = 2 + 1.8e-7` over a binary `x0`, with `-878 x0 >= -878`:
    /// `x0 = 1` needs `x1 = x2 = 2 + 9e-8`, the first and third rows each
    /// `9e-8` off, within their tolerances together, and `x0 = 0` meets
    /// nothing. Presolve rewrites the last row and drops it; the root's LP
    /// point has `x0 = 1 + 9e-8`, within its bound's tolerance, which the row
    /// as written rejects, and the branch fixes `x0` at 1. There the engine
    /// holds `x0` exactly, no single row can take its offset, and the proof,
    /// which grants the leaving `x0` its tolerance, none, and the rows none,
    /// prunes the node: `Infeasible` where `x*` is optimal. Without presolve
    /// the root is pruned the same way, on an equality's slack.
    #[test]
    #[should_panic(expected = "returned Infeasible")]
    fn a_node_fixing_a_binary_feasible_only_within_two_rows_tolerances_is_a_known_limitation() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(0.0, 0, 1),
                real(-1.0, 0.0, 100.0),
                real(0.0, 0.0, 100.0),
            ],
            rows: vec![
                row(&[(0, 2.0), (1, -1.0)], ComparisonOp::Eq, 0.0),
                row(&[(1, 1.0), (2, -1.0)], ComparisonOp::Eq, 0.0),
                row(&[(2, 1.0)], ComparisonOp::Eq, 2.00000018),
                row(&[(0, -878.0)], ComparisonOp::Ge, -878.0),
            ],
            xstar: vec![1.0, 2.00000009, 2.00000009],
        });
    }

    /// `x0 + 1e-5 x2 = 262144.00001`, with `x0` fixed at `2^18`, holds exactly
    /// only at `x2 = 1 + 1.8e-6`, past `x2 <= 1`, since its right-hand side is
    /// not representable, and `x*` meets it at `x2 = 1`, `1.8e-11` off, within
    /// its tolerance. The dual phase hands that miss to the equality's slack
    /// (absorption), a later pivot takes the slack out of the basis, and the
    /// slack absorbs it again. Refusing the second absorption would send the
    /// row to the proof, which grants the slack none of its tolerance: the
    /// model would be reported `Infeasible`, presolve on and off.
    #[test]
    fn an_equality_whose_slack_must_absorb_twice_in_a_phase_solves() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(0.0, 262144, 262144),
                real(1.0, -1.0, 0.0),
                real(1.0, 0.0, 1.0),
                real(-1.0, 0.0, 1.0),
                real(-1.0, 0.0, 0.0),
            ],
            rows: vec![
                row(&[(3, -1.0)], ComparisonOp::Ge, -1.0),
                row(&[(1, -1.0), (3, 0.01)], ComparisonOp::Eq, 0.0),
                row(&[(0, 1.0), (2, 1e-5)], ComparisonOp::Eq, 262144.00001),
                row(
                    &[(1, -1e-5), (2, -2000.0), (3, 1.0)],
                    ComparisonOp::Eq,
                    -2000.0,
                ),
            ],
            xstar: vec![262144.0, 0.0, 1.0, 0.0, 0.0],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
    }

    /// The same in an LP within the recommended range (row dynamism `3.5e5`):
    /// `-67 x0 + 0.012 x4 = -70254591.988`, with `x0` fixed at `2^20`, holds
    /// exactly only at `x4 = 1 - 4.6e-7`, since its right-hand side is not
    /// representable, while `-2.58 x4 <= -2.58` needs `x4 >= 1`. `x*` meets
    /// the equality at `x4 = 1`, `5.5e-9` off, within its tolerance. The
    /// equality's slack absorbs a move, a regular pivot takes it out of the
    /// basis, and it absorbs again on the way to the optimum.
    #[test]
    fn an_lp_whose_equality_slack_must_absorb_twice_in_a_phase_solves() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, 1048576.0, 1048576.0),
                real(0.0, 0.0, 1.0),
                real(1.0, 0.0, 1.0),
                real(-2.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, -67.0), (4, 0.012)], ComparisonOp::Eq, -70254591.988),
                row(&[(4, -2.58)], ComparisonOp::Le, -2.58),
                row(
                    &[(2, -0.00182), (3, 631.0), (4, 493.0)],
                    ComparisonOp::Le,
                    494.0,
                ),
                row(&[(1, 6.71), (2, 2.74)], ComparisonOp::Ge, 6.71),
                row(
                    &[(1, 275.0), (3, 0.294), (4, -0.00504)],
                    ComparisonOp::Eq,
                    274.99496,
                ),
            ],
            xstar: vec![1048576.0, 1.0, 0.0, 0.0, 1.0],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
    }

    /// A MILP node whose certificate carries a basic var's round-off on a huge
    /// bound is proven infeasible all the same. `x3 ∈ [0, 64212616]` is basic
    /// at the node `x1 >= 1`, which violates the second row by thousands.
    /// Unrefined, the certificate gives `x3` the coefficient `1.45e-17`, the
    /// round-off of terms of magnitude `0.19`, which times `x3`'s scaled bound
    /// of `4.2e12` closes the gap of `6.1e-5`; the refined row leaves it
    /// `7.7e-34`, some `3e-21` at that bound, and the node is proven.
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
    /// only, the basis mixes coefficients from `1e-5` to `1e4`. The unrefined
    /// row of its inverse leaves the basic `x3 ∈ [-2848554, 0]` a certificate
    /// coefficient of `8.4e-16`, `1.8e-14` of its terms, which times `x3`'s
    /// scaled bound of `3.6e8` is exactly the gap of `3.05e-7`; refined twice,
    /// the row leaves `x3` none, and the node is proven.
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
    /// proven infeasible all the same. The rows pin `x1` at zero, and the node
    /// `x1 <= -1` misses them by `6.5e-7` (scaled). With the refined row of
    /// the basis inverse rounded to `f64`, the basic `x2` keeps the row's
    /// representation error, `8.7e-17` beside terms of `3.1`, which at `x2`'s
    /// scaled bound of `-2.9e9` leaves the gap a hair short of the allowance
    /// `4e-7`; kept to twice the precision, the refined row leaves `x2` no
    /// coefficient.
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

    /// `min -x - y - t - z` s.t. `x + y + z + q >= 3` and
    /// `t - (1 + 2^-50) z - q >= 1e-6` over `x, y, t ∈ [0, 1]` (integer if
    /// asked), `z ∈ [z_min, 1]`, `q ∈ [-1e10, 1e10]` (`q` free behaves the
    /// same). Together the rows ask `x + y + t - 2^-50 z >= 3 + 1e-6`: only
    /// `z`'s coefficient `-2^-50`, the rows' own data, makes room for them, at
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
    /// coefficient of round-off size on a far bound fails loudly instead of
    /// reaching its optimum. The dual simplex ends with a row only `z` can
    /// repair, through an entry of `8.9e-16` beside ones, below every pivot
    /// floor. The certificate's `-2^-50` for `z`, counted at `z >= -2e9`,
    /// leaves `7.8e-7` of room (counted as zero, it would prove the model
    /// infeasible), and the certificate repair does not move a var by a
    /// coefficient below the round-off of its terms. Only a pivot on that
    /// entry solves it.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_round_off_coefficient_on_a_far_bound_is_no_proof_and_is_a_known_limitation() {
        check(room_through_a_round_off_coefficient(false, -2e9));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the model of the test above with
    /// `x, y, t` integer, whose root node's LP is that LP: the node must not
    /// be pruned on a false proof, and the MILP fails loudly (after the retry
    /// from the slack basis) where it has an optimum.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_node_lp_whose_room_is_round_off_is_not_pruned_and_is_a_known_limitation() {
        check(room_through_a_round_off_coefficient(true, -2e9));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the model of
    /// [`a_round_off_coefficient_on_a_far_bound_is_no_proof_and_is_a_known_limitation`]
    /// with `z` unbounded below. A coefficient on an infinite bound counts as
    /// zero only where the multipliers' error could make it zero; this row of
    /// the basis inverse is exact, so `z`'s `-2^-50` makes the certificate's
    /// range unbounded and proves nothing, and the certificate repair moves no
    /// var toward an infinite bound: the solve fails loudly where the model
    /// has an optimum.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_round_off_coefficient_on_an_infinite_bound_is_no_proof_and_is_a_known_limitation() {
        check(room_through_a_round_off_coefficient(
            false,
            f64::NEG_INFINITY,
        ));
    }

    /// `min -L` s.t. `L + j + w = 1.000001`, `-(1 + 2^-50) j - w = 0` and
    /// `j + v <= 0` over `L, v ∈ [0, 1]` (`L` integer if asked) and `j, w` free.
    /// The first two rows ask `L - 2^-50 j = 1.000001`: only `j`'s coefficient
    /// `-2^-50` makes room for them, at `j ≈ -1.13e9`, where the optimum
    /// `L = 1` is. The model is feasible in exact arithmetic at `x*`.
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): a feasible LP whose only room is a
    /// basic var's coefficient of round-off size in what a certificate leaves
    /// fails loudly instead of reaching `L = 1`. The dual simplex ends at the
    /// basis `{j, w}` and the second row's slack, `1e-6` above its bound, which
    /// only the third row's unbounded slack could repair, through an entry of
    /// `2^-50` beside ones. The certificate leaves that row out (its multiplier
    /// needs the slack's infinite side), so the free basic `j` keeps its exact
    /// `-2^-50`, which makes the range unbounded: no proof. Only a pivot on
    /// that entry solves it.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_basic_vars_coefficient_left_by_a_row_left_out_is_no_proof_and_is_a_known_limitation() {
        check(room_through_a_basic_vars_coefficient(false));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the test above without presolve, on
    /// the model as written, which ends at the same basis and fails loudly too.
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
    /// `L` integer, whose root node's LP is that LP: the node must not be
    /// pruned on a false proof, and the MILP fails loudly (after the retry
    /// from the slack basis) where its optimum is `L = 1`.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_node_lp_whose_room_is_a_basic_vars_coefficient_is_not_pruned_and_is_a_known_limitation() {
        check(room_through_a_basic_vars_coefficient(true));
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the test above without presolve,
    /// which fails loudly the same way.
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

    /// The forcing row `x0 + 100 x1 = 8388700` leaves `x0` a room of its
    /// budget, `8.4e-8`, off its lower bound `-2^-30`, which
    /// `-1000 x0 + x1 / 1000 = 83.887` amplifies past its own: fixed at the
    /// bound, `x0` makes that row miss by `9.3e-7`. The solve must reach the
    /// optimum `x0 = 0`, `x1 = 83887`.
    #[test]
    fn forcing_row_fixing_a_var_another_row_pins() {
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

    /// The rows of the test above added to a solved MILP one edit at a time:
    /// each edit's re-solve presolves the edited model again from the MILP's
    /// base and must reach the optimum `x0 = 0`, `x1 = 83887`.
    #[test]
    fn forcing_rows_added_by_edits_are_presolved_again_and_solved() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let x0 = problem.add_var(-1.0, (-(2.0f64.powi(-30)), 1.0));
        let x1 = problem.add_integer_var(1.0, (83887, 83888));
        let solution = |outcome: Result<crate::SolveOutcome, Error>| {
            outcome
                .expect("the solve must succeed")
                .into_solution()
                .expect("the solve must return a solution")
        };
        let sol = solution(problem.solve());
        let sol =
            solution(sol.add_constraint([(x0, -1.0), (x1, -100.0)], ComparisonOp::Eq, -8388700.0));
        let sol =
            solution(sol.add_constraint([(x0, -1000.0), (x1, 1e-3)], ComparisonOp::Eq, 83.887));
        assert_eq!(sol.var_value(x1), 83887.0);
        assert!(
            sol.var_value(x0).abs() <= 1e-9,
            "x0 = {}",
            sol.var_value(x0)
        );
    }

    /// MILPs whose forcing row's round-off at its magnitude leaves a var room
    /// the corner does not show (`1e-7 x4` at `1e9`, `2e-8 x3` at `5e8`, a
    /// unit of `1.375e-4 x1` at `1.4e10`, `x2`'s `7.45e-9` lost at `6.7e7`):
    /// fixed at the corner, the var contradicts the other row. Each is
    /// feasible within the contract and must solve.
    #[test]
    fn milps_whose_forcing_rows_leave_round_off_room_solve() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, 0, 0),
                int(0.0, -1, 0),
                real(1.0, -3.5762786865234375e-7, -3.5762786865234375e-7),
                real(0.0, 562122509.5000763, 562122509.5000763),
                real(0.0, -1.0, 214.0),
                real(1.0, 0.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(3, -1.0), (4, -1.0)],
                    ComparisonOp::Eq,
                    -562122509.5000763,
                ),
                row(
                    &[(1, -1.0), (2, -1.0), (3, -1.91015625), (4, 1e-7)],
                    ComparisonOp::Eq,
                    -1073741824.7872548,
                ),
            ],
            xstar: vec![
                0.0,
                0.0,
                -3.5762786865234375e-7,
                562122509.5000763,
                0.0,
                0.0,
            ],
        });
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(1.0, 536871, 536871),
                real(-1.0, 0.0, 0.0),
                real(-1.0, -1.0, 0.0),
                real(-1.0, -532.0, 5.0),
                real(0.0, 0.0, 0.0),
            ],
            rows: vec![
                row(&[(2, -1.0), (3, -1.0)], ComparisonOp::Eq, -2.0),
                row(
                    &[(0, -1000.0), (2, -0.001), (3, 2e-8)],
                    ComparisonOp::Eq,
                    -536870999.9989998,
                ),
            ],
            xstar: vec![536871.0, 0.0, -1.0, 3.0, 0.0],
        });
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, -2, 3),
                int(-1.0, -1, 2),
                real(-1.0, 681878.0, 681879.0),
            ],
            rows: vec![
                row(&[(0, -1.0), (1, 1.5)], ComparisonOp::Eq, 0.0),
                row(
                    &[(1, 0.0001375), (2, 20000.0)],
                    ComparisonOp::Eq,
                    13637560000.0,
                ),
            ],
            xstar: vec![0.0, 0.0, 681878.0],
        });
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(1.0, 0, 0),
                real(1.0, 3355444.0, 3355445.0),
                real(1.0, -7.450580596923828e-9, 1.0),
            ],
            rows: vec![
                row(&[(1, -20.0), (2, -1.0)], ComparisonOp::Eq, -67108880.0),
                row(&[(1, 1.0), (2, -20.0)], ComparisonOp::Eq, 3355444.0),
            ],
            xstar: vec![0.0, 3355444.0, 0.0],
        });
    }

    /// `x1 - x2 - 0.1 x4 = -16384.1` at the fixed `x1` and `x4` puts `x2` at
    /// `-1.5e-12`, the round-off of solving it. Made the bound
    /// `x3 >= 1.5e-7`, `-x2 - 1e-5 x3 <= 0` would take `x3`'s room there (its
    /// budget over `1e-5`) from `-x2 + x3 = 0`, which then contradicts it
    /// beyond the contract; `x2 = x3 = 0` meets every row as written.
    #[test]
    fn a_fixings_round_off_is_not_amplified_into_a_bound() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(1.0, 0.0, 0.0),
                real(-1.0, -16384.0, -16384.0),
                real(1.0, -1.0, 0.0),
                real(-1.0, -1.0, 1.0),
                real(-1.0, 1.0, 1.0),
            ],
            rows: vec![
                row(&[(2, -1.0), (3, -1e-5)], ComparisonOp::Le, 0.0),
                row(&[(2, -1.0), (3, 1.0)], ComparisonOp::Eq, 0.0),
                row(
                    &[(1, 1.0), (2, -1.0), (4, -0.1)],
                    ComparisonOp::Eq,
                    -16384.1,
                ),
            ],
            xstar: vec![0.0, -16384.0, 0.0, 0.0, 1.0],
        });
    }

    /// `x1 - x3 = -524288` is forcing at `x3`'s bound of `2^-34`, half an ulp
    /// its corner sum loses, and leaves `x3` a room of its budget, which
    /// `-1000 x3 + x4 = 0` amplifies past its own: fixed at the bound, `x3`
    /// needs `x4 = 5.8e-8`, beyond `x4 <= 0` by more than the engine holds
    /// `x4` to. `x3 = x4 = 0` meets every row exactly.
    #[test]
    fn a_forcing_row_leaves_a_var_another_row_resolves() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, 0.0, 0.0),
                real(-1.0, -524288.0, -524287.0),
                real(1.0, 0.0, 0.0),
                real(-1.0, -1.0, 2.0f64.powi(-34)),
                real(-1.0, -1.0, 0.0),
                real(0.0, 0.0, 0.0),
            ],
            rows: vec![
                row(&[(1, 1.0), (3, -1.0)], ComparisonOp::Eq, -524288.0),
                row(&[(3, -1000.0), (4, 1.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0, -524288.0, 0.0, 0.0, 0.0, 0.0],
        });
    }

    /// Two forcing rows `x_i + y_i <= 0` whose rooms add up in a third row
    /// (vars `x1, y1, w1, x2, y2, w2, z, u, z2, u2`; two equalities with `w_i`
    /// put `x_i` at `4e-8`, two with `u` or `u2` put `z` or `z2` at 0). `x*`
    /// has `x1 = x2 = 4e-8`, which the forcing rows accept within their
    /// budget; fixed at 0, each var would be within every row's budget, and
    /// both land in `0.9 x1 + 0.9 x2 + z + z2 = 7.2e-8`, which would then miss
    /// by `7.2e-8`, beyond its budget. With `milp`, `u` and `u2` are integer.
    fn two_rooms_in_one_kept_row(milp: bool) -> Model {
        let (e, c) = (4e-8, 0.9);
        let u = || {
            if milp {
                int(0.0, 0, 2)
            } else {
                real(0.0, 0.0, 2.0)
            }
        };
        Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
                real(0.0, 0.0, 2.0),
                real(0.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
                real(0.0, 0.0, 2.0),
                real(0.0, -1.0, 1.0),
                u(),
                real(0.0, -1.0, 1.0),
                u(),
            ],
            rows: vec![
                row(&[(0, 1.0), (1, 1.0)], ComparisonOp::Le, 0.0),
                row(&[(0, c), (2, 1.0)], ComparisonOp::Eq, 1.0 + c * e),
                row(&[(0, c), (2, -1.0)], ComparisonOp::Eq, -1.0 + c * e),
                row(&[(3, 1.0), (4, 1.0)], ComparisonOp::Le, 0.0),
                row(&[(3, c), (5, 1.0)], ComparisonOp::Eq, 1.0 + c * e),
                row(&[(3, c), (5, -1.0)], ComparisonOp::Eq, -1.0 + c * e),
                row(
                    &[(0, c), (3, c), (6, 1.0), (8, 1.0)],
                    ComparisonOp::Eq,
                    2.0 * c * e,
                ),
                row(&[(6, 1.0), (7, 1.0)], ComparisonOp::Eq, 1.0),
                row(&[(6, 1.0), (7, -1.0)], ComparisonOp::Eq, -1.0),
                row(&[(8, 1.0), (9, 1.0)], ComparisonOp::Eq, 1.0),
                row(&[(8, 1.0), (9, -1.0)], ComparisonOp::Eq, -1.0),
            ],
            xstar: vec![e, 0.0, 1.0, e, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    }

    /// `model`, with and without presolve, stopped by a zero time limit before
    /// its verdict and resumed: an optimal solution at `x*`'s objective.
    fn solves_resumed_at_the_objective_of_xstar(model: &Model) {
        let (problem, _) = model.build();
        for presolve in [true, false] {
            let interrupted = problem
                .solve_with(SolveOptions {
                    presolve,
                    time_limit: Some(Duration::ZERO),
                    ..SolveOptions::default()
                })
                .unwrap();
            assert!(
                interrupted.solution().is_none(),
                "fixture: the solve stops before its verdict (presolve {presolve})"
            );
            let resumed = interrupted
                .resume_with(crate::ResumeOptions::default())
                .unwrap_or_else(|err| panic!("the resume returned {err:?} (presolve {presolve})"))
                .into_solution()
                .unwrap();
            assert_eq!(resumed.status(), crate::SolutionStatus::Optimal);
            assert_eq!(resumed.objective(), model.objective_at(&model.xstar));
        }
    }

    /// Presolve fixes neither forcing row's continuous vars, so the model
    /// above solves, with and without presolve, also when a zero time limit
    /// stops the solve before its verdict and a resume finishes it.
    #[test]
    fn rooms_of_forcing_rows_adding_up_in_a_kept_row_are_kept() {
        let model = two_rooms_in_one_kept_row(false);
        solves_resumed_at_the_objective_of_xstar(&model);
        assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
    }

    /// The MILP of the model above solves, with and without presolve, also
    /// when a zero time limit stops the search before its root and a resume
    /// finishes it.
    #[test]
    fn rooms_of_forcing_rows_adding_up_in_a_kept_row_of_a_milp_are_kept() {
        let model = two_rooms_in_one_kept_row(true);
        solves_resumed_at_the_objective_of_xstar(&model);
        assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
    }

    /// The MILP of the model above with a free var of cost -1 in no row: its
    /// relaxation is unbounded, and the search that classifies it finds an
    /// integer point, so the MILP is unbounded, with and without presolve.
    #[test]
    fn rooms_of_forcing_rows_adding_up_in_a_kept_row_of_an_unbounded_milp_are_kept() {
        let mut model = two_rooms_in_one_kept_row(true);
        model
            .vars
            .push(real(-1.0, f64::NEG_INFINITY, f64::INFINITY));
        let (problem, _) = model.build();
        for presolve in [true, false] {
            let options = SolveOptions {
                presolve,
                ..SolveOptions::default()
            };
            assert_eq!(
                problem.solve_with(options).map(|_| ()),
                Err(Error::Unbounded),
                "presolve {presolve}"
            );
        }
    }

    /// The model above without its kept row, which an edit then adds: as a
    /// MILP, the edit's search solves the model as written, with and without
    /// presolve.
    #[test]
    fn rooms_of_forcing_rows_adding_up_in_a_row_an_edit_adds_to_a_milp_are_kept() {
        for presolve in [true, false] {
            let edited = add_the_kept_row_by_an_edit(true, presolve)
                .unwrap_or_else(|err| panic!("the edit returned {err:?} (presolve {presolve})"))
                .into_solution()
                .unwrap();
            assert_eq!(edited.status(), crate::SolutionStatus::Optimal);
            assert_eq!(edited.objective(), 0.0);
        }
    }

    /// The LP of the test above: the edit works on the engine of the
    /// presolved model, which keeps both forcing rows' vars their rooms.
    #[test]
    fn rooms_of_forcing_rows_adding_up_in_a_row_an_edit_adds_to_an_lp_are_kept() {
        for presolve in [true, false] {
            let edited = add_the_kept_row_by_an_edit(false, presolve)
                .unwrap_or_else(|err| panic!("the edit returned {err:?} (presolve {presolve})"))
                .into_solution()
                .unwrap();
            assert_eq!(edited.status(), crate::SolutionStatus::Optimal);
            assert_eq!(edited.objective(), 0.0);
        }
    }

    /// Solve [`two_rooms_in_one_kept_row`] without the row the two rooms add
    /// up in, then add that row with `Solution::add_constraint`.
    fn add_the_kept_row_by_an_edit(
        milp: bool,
        presolve: bool,
    ) -> Result<crate::SolveOutcome, Error> {
        let mut model = two_rooms_in_one_kept_row(milp);
        let kept = model.rows.remove(6);
        let (problem, vars) = model.build();
        let solution = problem
            .solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            })
            .expect("fixture: the model without the row solves")
            .into_solution()
            .unwrap();
        let expr: Vec<_> = kept.terms.iter().map(|&(j, a)| (vars[j], a)).collect();
        solution.add_constraint(expr, kept.op, kept.rhs)
    }

    /// The MILP above with an integer `k` of cost -1 whose `1.44e-7 k` the kept
    /// row carries, which would need `k = 0.5` with `x1` and `x2` at 0, and
    /// `3 a + 5 b <= 10` over integers in `[0, 2]` of cost -1, which the
    /// search branches on.
    fn two_rooms_behind_a_search() -> Model {
        let (e, c) = (4e-8, 0.9);
        let mut model = two_rooms_in_one_kept_row(true);
        model
            .vars
            .extend([int(-1.0, 0, 3), int(-1.0, 0, 2), int(-1.0, 0, 2)]);
        model.rows[6] = row(
            &[(0, c), (3, c), (6, 1.0), (8, 1.0), (10, 4.0 * c * e)],
            ComparisonOp::Eq,
            2.0 * c * e,
        );
        model
            .rows
            .push(row(&[(11, 3.0), (12, 5.0)], ComparisonOp::Le, 10.0));
        model.xstar.extend([0.0, 2.0, 0.0]);
        model
    }

    /// [`two_rooms_behind_a_search`] is solved to optimality, with and without
    /// presolve.
    #[test]
    fn rooms_of_forcing_rows_adding_up_behind_a_search_are_kept() {
        let model = two_rooms_behind_a_search();
        let (problem, _) = model.build();
        for presolve in [true, false] {
            let options = SolveOptions {
                presolve,
                ..SolveOptions::default()
            };
            let reason = problem.solve_with(options).unwrap().termination_reason();
            assert_eq!(
                reason,
                TerminationReason::ProvenOptimal,
                "presolve {presolve}"
            );
        }
        assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
    }

    /// [`two_rooms_behind_a_search`] with `x*` as a warm-start hint, whose
    /// `x1 = 4e-8` only a model keeping `x1`'s room admits: under a budget of
    /// no node, the root adopts the hint, with and without presolve, its
    /// values as computed within their bound tolerance of the hint's.
    #[test]
    fn a_warm_start_hint_needing_the_rooms_of_forcing_rows_is_adopted() {
        let model = two_rooms_behind_a_search();
        let (problem, vars) = model.build();
        let hint: Vec<_> = vars
            .iter()
            .copied()
            .zip(model.xstar.iter().copied())
            .collect();
        for presolve in [true, false] {
            let hinted = problem
                .solve_with(SolveOptions {
                    presolve,
                    warm_start: Some(hint.clone()),
                    node_limit: Some(0),
                    ..SolveOptions::default()
                })
                .unwrap();
            let solution = hinted.into_solution().unwrap_or_else(|i| {
                panic!("the root must adopt the hint (presolve {presolve}): {i:?}")
            });
            let costs: f64 = model.vars.iter().map(|v| v.obj.abs()).sum();
            assert!(
                (solution.objective() - model.objective_at(&model.xstar)).abs() <= 1e-7 * costs,
                "presolve {presolve}: objective {}",
                solution.objective()
            );
        }
    }

    /// [`two_rooms_in_one_kept_row`] with `b` in `[0, 1]` of cost -1 whose
    /// `7.2e-8 b` the kept row carries (right-hand side 0), an integer with
    /// `milp`: `x*` has `b = 1`, which the model as written admits, and which
    /// fixing `x1` and `x2` at 0 would exclude, the row then missing by
    /// `7.2e-8 b`. The optimum is `x*`'s, with and without presolve.
    fn an_optimum_needing_the_rooms_of_forcing_rows(milp: bool) {
        let (e, c) = (4e-8, 0.9);
        let mut model = two_rooms_in_one_kept_row(milp);
        model.vars.push(if milp {
            int(-1.0, 0, 1)
        } else {
            real(-1.0, 0.0, 1.0)
        });
        model.rows[6] = row(
            &[(0, c), (3, c), (6, 1.0), (8, 1.0), (10, -2.0 * c * e)],
            ComparisonOp::Eq,
            0.0,
        );
        model.xstar.push(1.0);
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
    }

    #[test]
    fn an_lp_optimum_needing_the_rooms_of_forcing_rows_is_found() {
        an_optimum_needing_the_rooms_of_forcing_rows(false);
    }

    #[test]
    fn a_milp_optimum_needing_the_rooms_of_forcing_rows_is_found() {
        an_optimum_needing_the_rooms_of_forcing_rows(true);
    }

    /// The model of the tests above, without presolve, solves.
    #[test]
    fn rooms_of_forcing_rows_adding_up_in_a_kept_row_without_presolve() {
        let model = two_rooms_in_one_kept_row(false);
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        assert_solves_within_contract_without_presolve(&model);
    }

    /// The big-M row `100 x + k1 w1 + k2 w2 + m y <= m + 500 + k1 + k2` over
    /// `x ∈ [0, 10]`, a binary `y` and `w1, w2` fixed at 1 (`ks = [k1, k2]`),
    /// and `1000 x >= 5000 + 10 e`, which needs the row `e` beyond its bound at
    /// `y = 1`, `e` being `share` of the row's contract there. With `optimum`,
    /// the second row is `1000 x >= (5000 + 10 e) z` over a binary `z` of cost
    /// -1. `p - q + pin y >= pin` and `q - p + pin y >= pin` over
    /// `p, q ∈ [0, pq]` exclude `y = 0` together; with `pin > pq`, each alone
    /// does. Coefficient tightening rewrites the big-M row into about
    /// `100 x + k1 w1 + k2 w2 + 500 y <= 1000 + k1 + k2`. `x*` has `y` at
    /// `y_star`.
    fn big_m_row_met_beyond_its_bound(
        m: f64,
        ks: [f64; 2],
        pin: f64,
        pq: f64,
        share: f64,
        optimum: bool,
        y_star: f64,
    ) -> Model {
        let [k1, k2] = ks;
        let bound = m + 500.0 + (k1 + k2);
        let magnitude = bound.abs() + 500.0 + m + k1.abs() + k2.abs();
        let e = share * 1e-7f64.max(1e-14 * magnitude);
        let rhs = 5000.0 + 10.0 * e;
        let mut model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, 0.0, 10.0),
                int(0.0, 0, 1),
                real(0.0, 0.0, pq),
                real(0.0, 0.0, pq),
                real(0.0, 1.0, 1.0),
                real(0.0, 1.0, 1.0),
            ],
            rows: vec![
                row(
                    &[(0, 100.0), (4, k1), (5, k2), (1, m)],
                    ComparisonOp::Le,
                    bound,
                ),
                row(&[(0, 1000.0)], ComparisonOp::Ge, rhs),
                row(&[(2, 1.0), (3, -1.0), (1, pin)], ComparisonOp::Ge, pin),
                row(&[(2, -1.0), (3, 1.0), (1, pin)], ComparisonOp::Ge, pin),
            ],
            xstar: vec![5.0 + e / 100.0, y_star, 0.0, 0.0, 1.0, 1.0],
        };
        if optimum {
            model.vars.push(int(-1.0, 0, 1));
            model.rows[1] = row(&[(0, 1000.0), (6, -rhs)], ComparisonOp::Ge, 0.0);
            model.xstar.push(1.0);
        }
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        model
    }

    /// [`big_m_row_met_beyond_its_bound`] at `M = 2.5e6` with fixed terms of
    /// `1e8`, which make the big-M row's contract at `y = 1` its round-off,
    /// `1.05e-6`, and `e` at `0.49` of it: within the engine's budget of the
    /// row as written, so the rewrite must keep the point. Its binding branch
    /// keeps the contract at the fixed terms' magnitude: at the rewritten
    /// terms' alone both contracts are the absolute `1e-7`. `pq = 1e5` leaves
    /// `y` unfixed, and `pin = 1e4` holds it within `5e-12` of 1 in the LP.
    #[test]
    fn a_point_within_the_budget_of_a_big_m_row_with_fixed_terms_is_kept() {
        for optimum in [false, true] {
            let model =
                big_m_row_met_beyond_its_bound(2.5e6, [5e7, -5e7], 1e4, 1e5, 0.49, optimum, 1.0);
            assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        }
    }

    /// [`big_m_row_met_beyond_its_bound`] at `M = 1.8e6` with fixed terms of
    /// `4.7e5` and `-4.7e6`, where the big-M row's contract at `y = 1` is the
    /// absolute `1e-7`, and `e` at `0.484` of it: within the engine's budget.
    /// The rewrite keeps nothing for the contract, so the rewritten row at
    /// `y = 1` must be no tighter than the row as written, by a single bit,
    /// which a bound carried in `f64` through the fixed terms' fold, the
    /// rewrite and the unfold is not: an ulp of the fixed term `4.7e6`.
    #[test]
    fn a_point_within_the_budget_of_a_big_m_row_by_less_than_an_ulp_is_kept() {
        for optimum in [false, true] {
            let model = big_m_row_met_beyond_its_bound(
                1805051.3787215224,
                [469042.9001025199, -4706346.9724488305],
                1e4,
                1e5,
                0.484,
                optimum,
                1.0,
            );
            assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        }
    }

    /// Two facilities with big-M rows `x_a <= 1e8 y_a`, `x_b <= 1e8 y_b` and a
    /// demand of 10: opening `b` (20 + 9 · 10 = 110) beats opening `a`
    /// (100 + 2 · 10 = 120). With `int_tol = 0` the root's `y_a = 1e-7` is
    /// fractional, and the branch `y_a = 0` fixes it. The engine holds a
    /// fixed binary exactly at its value, so that branch moves the demand to
    /// `b`: had it held `y_a` within its bound tolerance, `1e-7`, the big-M
    /// row would turn that into all the demand, and a fixed var, which cannot
    /// be branched on, counted fractional would drop the branch, `b`'s
    /// optimum with it (the answer was 120). The point is integral, within
    /// the contract, and not worse than 110.
    #[test]
    fn a_branch_that_fixes_a_big_m_binary_is_kept() {
        let (m, demand) = (1e8, 10.0);
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(100.0, 0, 1),
                int(20.0, 0, 1),
                real(2.0, 0.0, m),
                real(9.0, 0.0, m),
            ],
            rows: vec![
                row(&[(2, 1.0), (3, 1.0)], ComparisonOp::Ge, demand),
                row(&[(2, 1.0), (0, -m)], ComparisonOp::Le, 0.0),
                row(&[(3, 1.0), (1, -m)], ComparisonOp::Le, 0.0),
            ],
            xstar: vec![0.0, 1.0, 0.0, demand],
        };
        let (problem, vars) = model.build();
        for presolve in [true, false] {
            let solution = problem
                .solve_with(SolveOptions {
                    presolve,
                    int_tol: 0.0,
                    ..SolveOptions::default()
                })
                .unwrap()
                .into_solution()
                .unwrap();
            let x: Vec<f64> = vars.iter().map(|&v| solution.var_value(v)).collect();
            assert!(
                solution.objective() <= 110.0 + 1e-6,
                "presolve {presolve}: objective {} at {x:?}",
                solution.objective()
            );
            for (i, r) in model.rows.iter().enumerate() {
                let violation = crate::tests::oracle::row_violation(r, &x);
                assert!(
                    violation <= crate::tests::oracle::row_tolerance(r, &x),
                    "presolve {presolve}: row {i} violated by {violation:e} at {x:?}"
                );
            }
        }
    }

    /// The big-M row `100 x + Σ m_i y_i + k w <= Σ m_i + 500 + k` over
    /// `x ∈ [0, 10]`, binaries `y_i` and, with `k > 0`, an integer
    /// `w ∈ [1, 2]` of cost 1, and `1000 x >= 5000 + 10 e`, which needs the
    /// row `e` beyond its bound at `y = w = 1`, `e` being `share` of the row's
    /// contract there. With `optimum`, the second row is
    /// `1000 x >= (5000 + 10 e) z` over a binary `z` of cost -1.
    /// `p_i - q_i + 1e4 y_i >= 1e4` and `q_i - p_i + 1e4 y_i >= 1e4` over
    /// `p_i, q_i ∈ [0, 1e5]` exclude `y_i = 0` together. Coefficient
    /// tightening rewrites the row once per binary, and once more after dual
    /// fixing fixes `w` at 1.
    fn big_m_row_rewritten_more_than_once(ms: &[f64], k: f64, share: f64, optimum: bool) -> Model {
        let total: f64 = ms.iter().sum();
        let e = share * 1e-7f64.max(1e-14 * (2.0 * total + 2.0 * k + 1000.0));
        let rhs = 5000.0 + 10.0 * e;
        let mut vars = vec![real(0.0, 0.0, 10.0)];
        let mut xstar = vec![5.0 + e / 100.0];
        let mut terms = vec![(0, 100.0)];
        for &m in ms {
            terms.push((vars.len(), m));
            vars.push(int(0.0, 0, 1));
            xstar.push(1.0);
        }
        if k > 0.0 {
            terms.push((vars.len(), k));
            vars.push(int(1.0, 1, 2));
            xstar.push(1.0);
        }
        let mut rows = vec![
            row(&terms, ComparisonOp::Le, total + 500.0 + k),
            row(&[(0, 1000.0)], ComparisonOp::Ge, rhs),
        ];
        for y in 1..=ms.len() {
            let (p, q) = (vars.len(), vars.len() + 1);
            vars.extend([real(0.0, 0.0, 1e5), real(0.0, 0.0, 1e5)]);
            xstar.extend([0.0, 0.0]);
            rows.push(row(&[(p, 1.0), (q, -1.0), (y, 1e4)], ComparisonOp::Ge, 1e4));
            rows.push(row(&[(p, -1.0), (q, 1.0), (y, 1e4)], ComparisonOp::Ge, 1e4));
        }
        if optimum {
            rows[1] = row(&[(0, 1000.0), (vars.len(), -rhs)], ComparisonOp::Ge, 0.0);
            vars.push(int(-1.0, 0, 1));
            xstar.push(1.0);
        }
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars,
            rows,
            xstar,
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        model
    }

    /// [`big_m_row_rewritten_more_than_once`] with one binary at
    /// `m = 3.75e6` and `k = 2.5e6`, `e` at `0.45` of the row's contract at
    /// `y = w = 1`, `1.25e-7`: within the engine's budget. The rewrite comes
    /// before dual fixing fixes `w`, whose term the binding branch's contract
    /// counts once it is fixed; at the rewrite's magnitudes alone both
    /// contracts are the absolute `1e-7`.
    #[test]
    fn a_point_within_the_budget_of_a_big_m_row_whose_var_is_fixed_after_its_rewrite_is_kept() {
        for optimum in [false, true] {
            let model = big_m_row_rewritten_more_than_once(&[3.75e6], 2.5e6, 0.45, optimum);
            assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        }
    }

    /// [`big_m_row_rewritten_more_than_once`] with two binaries at
    /// `m = 3.5e6`, `e` at `0.45` of the row's contract at `y1 = y2 = 1`,
    /// `1.4e-7`: within the engine's budget. Each binary's rewrite alone
    /// takes out of that contract less than the two together.
    #[test]
    fn a_point_within_the_budget_of_a_big_m_row_of_two_rewritten_binaries_is_kept() {
        for optimum in [false, true] {
            let model = big_m_row_rewritten_more_than_once(&[3.5e6, 3.5e6], 0.0, 0.45, optimum);
            assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        }
    }

    /// [`big_m_row_met_beyond_its_bound`] at `M = 5e7` with fixed terms of
    /// `1e8` and `e` at `0.8` of the big-M row's contract. `pin = 100 < pq`,
    /// so only the two pin rows together keep `y` at one, and presolve
    /// rewrites `y`'s coefficient to about `500`.
    fn big_m_row_presolve_rewrites(optimum: bool) -> Model {
        big_m_row_met_beyond_its_bound(5e7, [5e7, -5e7], 100.0, 1000.0, 0.8, optimum, 1.0)
    }

    /// [`big_m_row_met_beyond_its_bound`] at `M = 5e7` without fixed terms and
    /// `e` at `0.8` of the big-M row's contract. `pin = 1e4 > pq`, so each pin
    /// row alone keeps `y` at one, and presolve rounds `y`'s lower bound to 1.
    fn big_m_row_presolve_fixes(optimum: bool) -> Model {
        big_m_row_met_beyond_its_bound(5e7, [0.0, -0.0], 1e4, 1000.0, 0.8, optimum, 1.0)
    }

    /// [`big_m_row_presolve_rewrites`]: with `y` at one the point needs the
    /// big-M row at `0.8` of its contract, within the engine's box, the whole
    /// contract less the round-off of evaluating the row. The rewrite keeps
    /// the binding branch's contract, so presolve on and off find the point.
    #[test]
    fn a_point_at_most_of_a_big_m_rows_contract_presolve_rewrites_is_kept() {
        let model = big_m_row_presolve_rewrites(false);
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// The optimum form of
    /// [`a_point_at_most_of_a_big_m_rows_contract_presolve_rewrites_is_kept`]:
    /// presolve on and off find `z = 1`, of objective -1.
    #[test]
    fn an_optimum_at_most_of_a_big_m_rows_contract_presolve_rewrites_is_found() {
        let model = big_m_row_presolve_rewrites(true);
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// [`big_m_row_presolve_fixes`]: with `y` at one the point needs the big-M
    /// row at `0.8` of its contract, within the engine's box. Presolve fixes
    /// `y` at 1, which keeps the point, and presolve on and off find it.
    #[test]
    fn a_point_at_most_of_a_big_m_rows_contract_presolve_fixes_is_kept() {
        let model = big_m_row_presolve_fixes(false);
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// The optimum form of
    /// [`a_point_at_most_of_a_big_m_rows_contract_presolve_fixes_is_kept`]:
    /// presolve on and off find `z = 1`, of objective -1.
    #[test]
    fn an_optimum_at_most_of_a_big_m_rows_contract_presolve_fixes_is_found() {
        let model = big_m_row_presolve_fixes(true);
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): [`big_m_row_met_beyond_its_bound`]
    /// with `e` at `1.03` of the big-M row's contract, `3%` beyond it with `y`
    /// at one. The point lies within the contract only with `y` off its
    /// integer within `int_tol`: `x*` has `y = 1 - 3e-14`, whose offset times
    /// `M = 5e7` makes the room. Without presolve the search reaches it
    /// through the LP (`y = 1 - 2e-14`); presolve rewrites `y`'s coefficient
    /// to about `500`, which takes that room away, and answers `Infeasible`.
    #[test]
    #[should_panic(expected = "solve() returned Infeasible")]
    fn a_point_needing_the_int_tol_of_a_binary_presolve_rewrites_is_a_known_limitation() {
        let model = big_m_row_met_beyond_its_bound(
            5e7,
            [5e7, -5e7],
            100.0,
            1000.0,
            1.03,
            false,
            1.0 - 3e-14,
        );
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7): the model of
    /// [`a_point_needing_the_int_tol_of_a_binary_presolve_rewrites_is_a_known_limitation`]
    /// without fixed terms and with each pin row alone keeping `y` at one:
    /// presolve fixes `y` at 1, which takes the room away, and answers
    /// `Infeasible`.
    #[test]
    #[should_panic(expected = "solve() returned Infeasible")]
    fn a_point_needing_the_int_tol_of_a_binary_presolve_fixes_is_a_known_limitation() {
        let model =
            big_m_row_met_beyond_its_bound(5e7, [0.0, -0.0], 1e4, 1000.0, 1.03, false, 1.0 - 3e-14);
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// A facility model whose capacity row `x2 + ... + x6 <= 2.918e9 x0`
    /// presolve rewrites to a coefficient of about `25`. The root's LP point
    /// has `x0` basic at `-7.8e-16`, within round-off of 0, which the row as
    /// written turns into `2.3e-6` of violation, beyond its contract: the
    /// guard rejects the point. The branches at `int_tol = 0` fix `x1` at 1
    /// and then `x0` at 0, which the engine holds there exactly, where the row
    /// holds. Without presolve the engine holds the row as written. Its
    /// coefficient, `2.918e9` beside ones, lies far outside the recommended
    /// range.
    #[test]
    fn a_binary_round_off_under_a_big_m_row_presolve_rewrites_solves() {
        let flow = |obj: f64, hi: f64| real(obj, 0.0, hi);
        let unbounded = f64::INFINITY;
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(17.46, 0, 1),
                int(54.35, 0, 1),
                flow(2.43, 4.39),
                flow(15.54, unbounded),
                flow(15.74, unbounded),
                flow(16.73, unbounded),
                flow(6.37, 5.09),
                flow(13.59, unbounded),
                flow(5.17, 4.3),
                flow(3.66, 2.2),
                flow(13.0, unbounded),
                flow(9.46, 5.09),
            ],
            rows: vec![
                row(&[(2, 1.0), (7, 1.0)], ComparisonOp::Eq, 4.39),
                row(&[(3, 1.0), (8, 1.0)], ComparisonOp::Eq, 4.3),
                row(&[(4, 1.0), (9, 1.0)], ComparisonOp::Eq, 2.2),
                row(&[(5, 1.0), (10, 1.0)], ComparisonOp::Ge, 9.09),
                row(&[(6, 1.0), (11, 1.0)], ComparisonOp::Ge, 5.09),
                row(
                    &[
                        (2, 1.0),
                        (3, 1.0),
                        (4, 1.0),
                        (5, 1.0),
                        (6, 1.0),
                        (0, -2918000000.0),
                    ],
                    ComparisonOp::Le,
                    0.0,
                ),
                row(&[(2, 1.0), (0, -4.39)], ComparisonOp::Le, 0.0),
                row(&[(3, 1.0), (0, -4.3)], ComparisonOp::Le, 0.0),
                row(&[(4, 1.0), (0, -2.2)], ComparisonOp::Le, 0.0),
                row(&[(5, 1.0), (0, -9.09)], ComparisonOp::Le, 0.0),
                row(&[(6, 1.0), (0, -5.09)], ComparisonOp::Le, 0.0),
                row(
                    &[
                        (7, 1.0),
                        (8, 1.0),
                        (9, 1.0),
                        (10, 1.0),
                        (11, 1.0),
                        (1, -25.07),
                    ],
                    ComparisonOp::Le,
                    0.0,
                ),
                row(&[(0, 1.0), (1, 1.0)], ComparisonOp::Le, 1.0),
            ],
            xstar: vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.39, 4.3, 2.2, 9.09, 5.09,
            ],
        };
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// The class above within the recommended range (row dynamism at most
    /// `72.5`, column `571`). The round-off of the data of
    /// `1.09 x0 + 0.952 x2 = 1090004.222`, with `x0` fixed at `1000003`, puts
    /// `x2` at `1 + 5.1e-11`, which `7.5 x1 - 544 x2 = -536.5` turns into the
    /// binary `x1 = 1 + 3.7e-9`, within its bound's tolerance. Presolve
    /// rewrites `-878 x1 >= -878` to a coefficient of `8.8e-5` and drops it,
    /// and the row as written turns the offset into `3.2e-6` of violation: the
    /// guard rejects the point, and the branch that fixes `x1` at 1 holds it
    /// there exactly, an equality's slack taking the offset within its
    /// tolerance. Without presolve the engine holds the row as written.
    #[test]
    fn a_binary_round_off_under_a_row_presolve_rewrites_solves() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(0.0, 1000003, 1000003),
                int(0.0, 0, 1),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, 1.09), (2, 0.952)], ComparisonOp::Eq, 1090004.222),
                row(&[(1, 7.5), (2, -544.0)], ComparisonOp::Eq, -536.5),
                row(&[(1, -878.0)], ComparisonOp::Ge, -878.0),
            ],
            xstar: vec![1000003.0, 1.0, 1.0],
        };
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// `-x0 + x3 = 0` and `-2 x0 - 10 x3 <= 7.642e-7` over a binary `x0` and
    /// `x3 ∈ [-1, 100]`, whose every point has the objective `-x0 + x3 = 0`
    /// (row and column dynamism at most `10`). Presolve rewrites the second
    /// row's coefficient on `x0`, which holds as written only at an integral
    /// `x0`, and the vertex of the rewritten rows puts `x0` at `-7.6e-8`,
    /// past its bound within its tolerance, where the row as written misses
    /// by `1.5e-7`: the guard rejects the point, and the branch that fixes
    /// `x0` at 0 holds it there exactly, where the row holds.
    #[test]
    fn a_binary_the_lp_puts_past_its_bound_under_a_row_presolve_rewrites_solves() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![int(-1.0, 0, 1), real(1.0, -1.0, 100.0)],
            rows: vec![
                row(&[(0, -1.0), (1, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(0, -2.0), (1, -10.0)], ComparisonOp::Le, 7.642e-7),
            ],
            xstar: vec![0.0, 0.0],
        };
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }
}
