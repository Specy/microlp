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

    /// `x0` is pinned to `2^28` by an equality whose other coefficient is
    /// `1e-7`, so the row's terms span fifteen orders of magnitude, and
    /// `x* = [268435456, 0]` satisfies both rows exactly. The row's round-off,
    /// `5e-6`, exceeds `x1`'s whole term range, `2.5e-8`, so its minimum
    /// activity meets the bound wherever `x1` sits: presolve's forcing rule
    /// must not read that as "every var at its extreme" and fix `x1` at `0.25`
    /// against `-x1 = 0`. Forcing requires every term to be resolvable above
    /// the row's round-off (`Work::terms_resolvable`).
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): a model whose optimum is reached
    /// only through a basis the factorization finds singular fails loudly,
    /// where `x*`, at `x2 = 5000`, is optimal. `x2`, the last var that
    /// improves the objective, is blocked only by an entry some `1e-14` of its
    /// column: the pivot on it leaves a singular basis, the repair takes `x2`
    /// out, and the phase comes back to that basis by another route until only
    /// changes to it are left.
    #[test]
    #[should_panic(expected = "leads to a singular basis")]
    fn an_optimum_reached_only_through_a_singular_basis_is_a_known_limitation() {
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
    /// feasible point, is optimal. The last primal step pivots on an entry
    /// some `1e-14` of its column into that basis (a 1-norm condition of about
    /// `4.5e14` in the scaled data); the repair takes the entering var back
    /// out, and the change back stays taboo, since the step, a zero one, moved
    /// no row.
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): the phases alternate over a row
    /// only entries below the round-off floor can repair, and the solve fails
    /// with `8 phase rounds` where the answer is the origin, the only feasible
    /// point. The primal step that lowers `x0` passes `-1e4 x2 + 1e-3 x3 >= 0`,
    /// whose genuine `3e-15` is below its column's round-off floor (largest
    /// entry 4), and leaves the row `1e-7` short; the dual phase's
    /// certificate repair moves `x5` across its range through another such
    /// entry, which gives the step back. The fix is a degenerate pivot on the `3e-15`,
    /// which needs absolute pivot tolerances and a refined row.
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): the same class. The second row
    /// makes `x1 = 1e-11 x0`, so `x1 >= 0` needs `x0 >= 0` and the optimum is
    /// 0; the solve fails with `8 phase rounds`. The primal step that lowers
    /// `x0` to -1 passes `x1`'s row through an entry below its column's
    /// round-off floor, to a point that meets the engine's tolerances only off
    /// every vertex (the equality at its tolerance, `x1` at its own, capped at
    /// `5e-12` by its row's reserve); the dual phase moves `x3` back through
    /// another such entry. The fix is the one above, or the engine's
    /// tolerances raised to the whole contract.
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

    /// A vertex that misses a bound by the round-off of solving for it, beyond
    /// the tolerance the engine holds the var to, is repaired within the
    /// tolerance of a fixed slack. After two dual pivots every
    /// non-basic var is an equality's fixed slack, and `x2` sits `3.3e-10`
    /// (scaled) past its upper bound, three times its tolerance; the row's
    /// certificate proves nothing (a gap of `-6.7e-14`), and only a slack
    /// moved within its tolerance lands `x2`. `x* = (0, -230094, 0)`.
    #[test]
    fn a_vertex_off_its_bound_by_round_off_is_repaired_within_the_slacks_tolerances() {
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

    /// A MILP node LP that no pivot can repair is repaired within the
    /// tolerance of a fixed slack, neither pruned nor failed. The rows pin
    /// `x5` at zero; the root vertex puts it `3.05e-12` (scaled, `5e-8` in
    /// user units) past its upper bound, beyond the tolerance its rows'
    /// reserve caps it at, and only the second row's unbounded slack could
    /// pivot, through an entry of `3.2e-16`, below every pivot floor. The
    /// certificate proves nothing, and the first row's fixed slack, moved
    /// within its tolerance, lands `x5`. `x* = 0` but for `x4 = 500001`.
    #[test]
    fn a_node_lp_no_pivot_can_repair_is_repaired_within_a_slacks_tolerance() {
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

    /// A vertex whose basic var misses its bound by the round-off of rows of
    /// magnitude `7.7e7` is repaired within the tolerances of two fixed
    /// slacks. After five dual pivots `x2` is computed `2^-24` (scaled) below
    /// its lower bound, beyond its tolerance, where the vertex is exactly
    /// feasible (the certificate's gap is `-3e-33`); every var of its row is
    /// fixed or at its bound in the direction that helps, and no one slack
    /// can absorb the move. The slacks of the last two rows, each moved
    /// within its tolerance, land `x2`. The optimum is `x*`'s, `77137`.
    #[test]
    fn a_basic_var_off_its_bound_by_its_rows_round_off_is_repaired() {
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

    /// A vertex off its bound by round-off whose only pivot is an entry below
    /// round-off is repaired within the tolerance of a fixed slack. `x1` is
    /// computed `3.05e-12` (scaled) past its upper bound, `6e-20` more than
    /// the first row's slack could absorb alone; the second row's slack,
    /// unbounded, reaches the row through `5e-16` only. The first row's slack
    /// moved by its tolerance lands `x1` within its own.
    #[test]
    fn a_vertex_off_its_bound_behind_a_round_off_entry_is_repaired() {
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): a vertex whose computed value
    /// misses a bound by more than the other terms of its row's certificate
    /// can absorb, though its exact value does not, fails loudly where `x*`
    /// is feasible. The dual simplex ends at a vertex that computes `x0`
    /// `3.7e-9` (scaled) past its upper bound of 0, the round-off of rows of
    /// magnitude `4.9e6`, where the vertex is inside it (the certificate's
    /// gap is `-1.2e-9`). `x1` and the first row's slack are at the bounds
    /// that block them, and moved within their tolerances they carry `x0` by
    /// `2e-9`, which with its own tolerance of `1e-9` leaves it `7e-10`
    /// short. Values computed to the certificate's precision would accept
    /// the vertex.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_vertex_computed_past_its_bound_beyond_its_terms_tolerances_is_a_known_limitation() {
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

    /// A MILP node whose LP misses a bound by less than the user's tolerance
    /// but more than the engine's is proven infeasible and pruned. The node
    /// `x0 <= -1` puts `x2` at `1e-8`, past its bound of zero, beyond the
    /// tolerance its first row's reserve caps it at but within the user's
    /// `1e-7`, and nothing can move; the certificate's gap is exactly that
    /// `1e-8`. A proof that asks more than the simplex's own test leaves the
    /// node neither proven nor solvable. The optimum is `x0 = 1` (where
    /// `x2 = -1e-8`).
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

    /// Presolve's forcing row `x0 + 100 x1 = 8388700` fixes `x0` at its lower
    /// bound `-2^-30`, where the row allows it within its tolerance, and
    /// `-1000 x0 + x1 / 1000 = 83.887`, which pins `x0` at zero, then misses
    /// by `9.3e-7`: presolve finds the model infeasible. The solve must go on
    /// with the model as written and reach the optimum `x0 = 0`, `x1 = 83887`.
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
    /// the second edit's re-solve presolves the edited model, finds it
    /// infeasible, and must go on with it as written.
    #[test]
    fn forcing_rows_added_by_edits_are_solved_as_written() {
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

    /// MILPs that presolve finds infeasible, each feasible within the
    /// contract: a forcing row's round-off at its magnitude leaves a var room
    /// that the forcing rule does not count (`1e-7 x4` at `1e9`, `2e-8 x3` at
    /// `5e8`, a unit of `1.375e-4 x1` at `1.4e10`, `x2`'s `7.45e-9` lost at
    /// `6.7e7`), so the rule fixes the var at a bound where another row
    /// contradicts it. The solve must go on with the model as written.
    #[test]
    fn milps_presolve_finds_infeasible_are_solved_as_written() {
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

    /// KNOWN LIMITATION (ARCHITECTURE §7): presolve fixes `x2` at
    /// `-1.5e-12`, the round-off of solving `x1 - x2 - 0.1 x4 = -16384.1` at
    /// the fixed `x1` and `x4`, and `-x2 - 1e-5 x3 <= 0` then becomes the
    /// bound `x3 >= 1.5e-7`, which `-x2 + x3 = 0` contradicts beyond the
    /// contract. Presolve's own checks pass, and the engine proves the
    /// reduced model infeasible, where `x2 = x3 = 0` meets every row as
    /// written. The fix is a presolve that counts a fixing's round-off where
    /// a coefficient amplifies it.
    #[test]
    #[should_panic(expected = "returned Infeasible")]
    fn a_fixings_round_off_amplified_into_a_bound_is_a_known_limitation() {
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
}
