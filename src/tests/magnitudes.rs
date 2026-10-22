//! Regression tests on feasible-by-construction models across variable
//! magnitudes and coefficient ranges (the properties are in `scaling.rs`).
//! Each model is written around a point `x*` that satisfies every bound and
//! every row within the documented tolerance, and its objective is bounded.
//! Where `x*` has its integer vars at integers, `solve()` must return an
//! optimal solution whose rows hold within that tolerance and whose objective
//! is no worse than at `x*` — the contract in `oracle.rs`. A model with no
//! integral point with its fixed vars at their values is written around a
//! point that relies on the tolerances, and may be answered `Infeasible`
//! instead.
//!
//! A model the engine gets wrong and that cannot be fixed at once is pinned
//! here as a KNOWN LIMITATION with `#[should_panic]`, which records today's
//! answer: a fix then shows up as a FAILING test, and deleting the attribute
//! (the body already asserts the right answer) makes it an ordinary regression
//! test. See §7.9 of `ARCHITECTURE.md`.

#[cfg(test)]
mod regression_tests {
    use crate::tests::oracle::{
        assert_solves_feasibly_within_contract, assert_solves_within_contract,
        assert_solves_within_contract_in_time, assert_solves_within_contract_without_presolve,
        infeasible_or_answer_within_contract, infeasible_or_within_contract, row_tolerance,
        row_violation, rows_hold, within_recommended_range, xstar_is_feasible, Model, Row, Var,
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

    /// `model` with `count` more rows `z + z' <= 1.5` over vars of their own,
    /// in `[0, 1]` at zero cost, which the optimum does not involve: the
    /// factors of the larger basis take the eta file longer to outgrow, and a
    /// turnover of the basis takes more pivots, so the refactorizations and
    /// their drift checks fall elsewhere on the path.
    fn with_idle_rows(mut model: Model, count: usize) -> Model {
        for _ in 0..count {
            let z = model.vars.len();
            model.vars.push(real(0.0, 0.0, 1.0));
            model.vars.push(real(0.0, 0.0, 1.0));
            model
                .rows
                .push(row(&[(z, 1.0), (z + 1, 1.0)], ComparisonOp::Le, 1.5));
            model.xstar.extend([0.0, 0.0]);
        }
        model
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
    /// must still refine them, with residuals summed in compensated
    /// arithmetic, and through a fresh factorization if that is not enough.
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
    /// units here. The engine holds the row short of its contract by the
    /// round-off of its own evaluation, which the guard's compensated sum
    /// leaves it, so the candidate passes, and presolve's integer rounding
    /// must keep that margin: an exactly integral candidate the guard rejects
    /// is an `InternalError`. The tolerance leaves the objective real freedom,
    /// so only feasibility is checked.
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

    /// A model within the recommended range (dynamism at most `2e5`) whose
    /// only repair is an entry below the pivot tolerance: `x2`, `x3` and `x4`
    /// tie `x1` to the third row's slack through the ratios `1e4`, `2000` and
    /// `100`, which leave `x1`'s entry there at `1e-9` of the row's largest. A
    /// primal flip of `x1` through that entry, which bounds no step, leaves the
    /// row `1.25e-5` (scaled) past its bound, and only `x1` can repair it: a
    /// pivot on an entry genuine on the refined row, the dual simplex's last
    /// resort. Without it, the certificate repair flipped `x1` back, the flip
    /// came again, and the solve failed after `8 phase rounds`.
    #[test]
    fn a_row_a_chain_of_ratios_leaves_a_tiny_genuine_repair_solves() {
        check(Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, 0.0, 0.0),
                real(0.0, -99888.0, 0.0),
                real(-1.0, -1.0, 0.0),
                real(0.0, -10.0, 0.0),
                real(1.0, -1.0, 0.0),
            ],
            rows: vec![
                row(&[(2, 1.0), (3, -1.0), (4, 2000.0)], ComparisonOp::Eq, 0.0),
                row(&[(1, -1.0), (3, 10000.0), (4, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(2, -100.0), (4, -0.01)], ComparisonOp::Eq, -0.0),
            ],
            xstar: vec![0.0; 5],
        });
    }

    /// A model outside the recommended range (constraint 0 spans `1e8`) on
    /// which the phases alternated between two vertices: on multipliers from
    /// one solve, each primal phase ended where the reduced costs showed the
    /// other's entering var dual infeasible by `2.3e-10` to `3e-10`, the
    /// round-off `-x0 + 1e-8 x4 = 0` amplifies into the multipliers, until
    /// `8 phase rounds`. The verification refines the multipliers once, and
    /// the solve ends, with presolve on and off.
    #[test]
    fn phases_settle_on_the_reduced_costs_of_refined_multipliers() {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, 0.0, 1.0),
                real(0.0, -1.0, 0.0),
                real(0.0, 0.0, 2.0),
                real(-1.0, 0.0, 0.0),
                real(-1.0, -11.0, 0.0),
            ],
            rows: vec![
                row(&[(0, -1.0), (4, 1e-8)], ComparisonOp::Eq, 0.0),
                row(
                    &[(1, -0.0015), (2, -10.0), (4, -1.0)],
                    ComparisonOp::Eq,
                    -0.0,
                ),
                row(&[(0, 1.0), (1, -162.5), (2, 0.0125)], ComparisonOp::Le, 1.0),
            ],
            xstar: vec![0.0; 5],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
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

    /// `min x + 2y` s.t. `x + (1 + 2^-48) y = -2` and `x + y = -2`, `x` in
    /// `[-10, 10]`, `y` free, within the recommended range: exactly, only
    /// `(-2, 0)` meets both rows, and within their tolerances the segment
    /// `y = -2 - x` does, to `(10, -12)`. The primal phase starts at
    /// `x = -10`, with `y` and the second row's slack basic, and `x` enters:
    /// the slack's entry is `3.6e-15`, genuine (the columns differ by
    /// `2^-48`) but below the pivot tolerance, so it does not bound the step,
    /// and `x` moves to its upper bound, the slack `7e-14` off its value.
    /// Taken for the pivot, the entry stopped the step at `(-2, 0)`, in a
    /// basis the factorization finds singular; the repair took `x` out, the
    /// change back stayed taboo, and the solve failed loudly.
    #[test]
    fn a_primal_step_passes_a_genuine_entry_below_the_pivot_tolerance() {
        for presolve in [true, false] {
            let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(1.0, (-10.0, 10.0));
            let y = problem.add_var(2.0, (f64::NEG_INFINITY, f64::INFINITY));
            let slanted = 1.0 + 2f64.powi(-48);
            problem.add_constraint([(x, 1.0), (y, slanted)], ComparisonOp::Eq, -2.0);
            problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Eq, -2.0);
            let solution = match problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            }) {
                Ok(outcome) => outcome.into_solution().unwrap(),
                Err(err) => panic!("presolve {presolve}: {err:?}"),
            };
            let (objective, x, y) = (solution.objective(), solution[x], solution[y]);
            assert!(
                (-10.0 - 1e-9..=10.0 + 1e-9).contains(&x)
                    && (x + slanted * y + 2.0).abs() < 1e-9
                    && (x + y + 2.0).abs() < 1e-9
                    && (objective - (x + 2.0 * y)).abs() < 1e-9
                    && objective <= -2.0 + 1e-9,
                "presolve {presolve}: objective {objective} at ({x}, {y})"
            );
        }
    }

    /// `max x` s.t. `0.3 x + 0.7 y >= 1` and `0.9000000003 x + 2.1 y <= 3`,
    /// `x >= 0`, `y` free, within the recommended range: in decimals the rows
    /// meet only at `x = 0`, and with their `f64` data they hold only within
    /// their tolerances, near it, where the answer is. The step that raises
    /// `x` meets the second row's slack through an entry of `1.5e-10`, below
    /// the pivot tolerance and toward an infinite bound, that the nearly
    /// parallel rows leave and the plain column computes inexactly. The
    /// refined column has it beyond the error the refinement measured, so it
    /// bounds the step. Taken for round-off because the plain column did not
    /// agree with the refined one to seven digits, it left the step unbounded,
    /// and the solve returned `Unbounded`.
    #[test]
    fn a_ray_an_inexactly_computed_genuine_entry_blocks_is_bounded() {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(1.0, 0.0, f64::INFINITY),
                real(0.0, f64::NEG_INFINITY, f64::INFINITY),
            ],
            rows: vec![
                row(&[(0, 0.3), (1, 0.7)], ComparisonOp::Ge, 1.0),
                row(&[(0, 0.9000000003), (1, 2.1)], ComparisonOp::Le, 3.0),
            ],
            xstar: vec![0.0, 1.0 / 0.7],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
    }

    /// `min 1.7 x + 2.5 y` s.t. `2.08 x + 0.57 y <= 2.16` and
    /// `-8.735999999901162 x - 2.394000000027085 y <= -9.072000000000001`,
    /// `x`, `y` free, within the recommended range: the second row is `-4.2`
    /// times the first but for the last digits, and the two meet at the
    /// optimum, `5.6195517638` at `(0.51923, 1.89475)`, a vertex of a narrow
    /// wedge. After a dual pivot brings `y` in, the primal step that moves
    /// `x` meets the first row's slack through an entry of `2.4e-11`, below
    /// the pivot tolerance, which the plain column computes inexactly: it
    /// bounds the step, beyond the error the refinement measured. Taken for
    /// round-off, it left the step unbounded, and the solve returned
    /// `Unbounded`.
    #[test]
    fn a_bounded_lp_of_nearly_parallel_rows_is_not_taken_for_unbounded() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(1.7, f64::NEG_INFINITY, f64::INFINITY),
                real(2.5, f64::NEG_INFINITY, f64::INFINITY),
            ],
            rows: vec![
                row(&[(0, 2.08), (1, 0.57)], ComparisonOp::Le, 2.16),
                row(
                    &[(0, -8.735999999901162), (1, -2.394000000027085)],
                    ComparisonOp::Le,
                    -9.072000000000001,
                ),
            ],
            xstar: vec![0.5192284317218896, 1.8947453719622278],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
    }

    /// The exact ray relation is y = -z and x = 2^-30 - 2^-52 z.
    /// A genuine blocking entry below the residual's a priori round-off
    /// allowance must still bound the step.
    #[test]
    fn a_tiny_exact_blocker_within_the_residual_allowance_still_bounds_the_step() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let x = problem.add_var(0.0, (0.0, f64::INFINITY));
        let y = problem.add_var(0.0, (f64::NEG_INFINITY, f64::INFINITY));
        let z = problem.add_var(1.0, (0.0, f64::INFINITY));
        let a = 1.0 + 2f64.powi(-52);
        let e = 2f64.powi(-45);
        problem.add_constraint(
            [(x, 1.0), (y, 1.0), (z, a)],
            ComparisonOp::Eq,
            2f64.powi(-30),
        );
        problem.add_constraint(
            [(x, 1.0), (y, 1.0 + e), (z, a + e)],
            ComparisonOp::Eq,
            2f64.powi(-30),
        );
        for presolve in [true, false] {
            let options = crate::SolveOptions {
                presolve,
                ..Default::default()
            };
            let outcome = problem.solve_with(options).unwrap();
            let sol = outcome.solution().unwrap();
            assert_eq!(sol.objective(), 2f64.powi(22));
            assert_eq!(sol[z], 2f64.powi(22));
        }
    }

    /// `max -4 x0 - 3 x1 + x2` s.t. the redundant equalities of
    /// [`a_ray_past_a_redundant_equality_does_not_pivot_on_round_off`] and
    /// `x0 - (5 + 2^-30) x1 <= -18 - 2^-30`, all vars free, which on their
    /// line `(-8, 2, 0) + t (5, 1, 6)` holds `t >= -1`: its entry in the
    /// column of the step toward the ray, far below the pivot tolerance, is a
    /// genuine blocker, and the optimum is `43`, presolve on and off. These
    /// solves meet no entry of unresolved sign beside it;
    /// `solver::tests::a_resolved_blocker_beside_an_unresolved_entry_is_the_pivot`
    /// plants one.
    #[test]
    fn a_tiny_blocker_bounds_a_step_along_redundant_equalities() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let x0 = problem.add_var(-4.0, free);
        let x1 = problem.add_var(-3.0, free);
        let x2 = problem.add_var(1.0, free);
        problem.add_constraint(
            [(x0, 18.0), (x1, -18.0), (x2, -12.0)],
            ComparisonOp::Eq,
            -180.0,
        );
        problem.add_constraint([(x0, -1.0), (x1, 5.0)], ComparisonOp::Eq, 18.0);
        problem.add_constraint([(x0, -5.0), (x1, 1.0), (x2, 4.0)], ComparisonOp::Eq, 42.0);
        let delta = 2f64.powi(-30);
        problem.add_constraint(
            [(x0, 1.0), (x1, -(5.0 + delta))],
            ComparisonOp::Le,
            -18.0 - delta,
        );
        for presolve in [true, false] {
            let options = crate::SolveOptions {
                presolve,
                ..Default::default()
            };
            let outcome = problem.solve_with(options).unwrap();
            let sol = outcome.solution().unwrap();
            assert!(
                (sol.objective() - 43.0).abs() < 1e-5,
                "{presolve}: {}",
                sol.objective()
            );
        }
    }

    /// `max -4 x0 - 3 x1 + x2` s.t. `18 x0 - 18 x1 - 12 x2 = -180`,
    /// `-x0 + 5 x1 = 18` and `-5 x0 + x1 + 4 x2 = 42`, all vars free, within
    /// the recommended range: the first row is `-3` times the sum of the
    /// others, and the rows leave the line `(-8, 2, 0) + t (5, 1, 6)`, along
    /// which the objective is `26 - 17 t`, unbounded. At `(-8, 2, 0)`, with
    /// `x0`, `x1` and the third row's slack basic, `x2` enters toward its
    /// infinite bound through no entry above the pivot tolerance, and its
    /// column is exactly zero in the slack's row. The refined column put that
    /// zero at `-8.2e-33`, beyond its second correction of `2.1e-33` but
    /// within the round-off of the refinement's residual: it bounded the
    /// step, the pivot on it left the basis singular, and with every
    /// non-basic var a fixed slack the solve returned `Optimal` at `26`,
    /// presolve on and off. Unresolved, the entry does not pivot, and over
    /// `2^52` units of `x2` it moves the slack, at zero, by far less than its
    /// tolerance: a ray.
    #[test]
    fn a_ray_past_a_redundant_equality_does_not_pivot_on_round_off() {
        assert_unbounded(
            &redundant_equality_ray(1.0),
            "ray past a redundant equality",
        );
    }

    /// The model of
    /// [`a_ray_past_a_redundant_equality_does_not_pivot_on_round_off`] with a
    /// unit of `x2` `unit` times as large, a power of two: `x2`'s cost and
    /// coefficients are multiplied by `unit`.
    fn redundant_equality_ray(unit: f64) -> crate::Problem {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let x0 = problem.add_var(-4.0, free);
        let x1 = problem.add_var(-3.0, free);
        let x2 = problem.add_var(unit, free);
        problem.add_constraint(
            [(x0, 18.0), (x1, -18.0), (x2, -12.0 * unit)],
            ComparisonOp::Eq,
            -180.0,
        );
        problem.add_constraint([(x0, -1.0), (x1, 5.0)], ComparisonOp::Eq, 18.0);
        problem.add_constraint(
            [(x0, -5.0), (x1, 1.0), (x2, 4.0 * unit)],
            ComparisonOp::Eq,
            42.0,
        );
        problem
    }

    /// [`redundant_equality_ray`] with a unit of `x2` `2^40` times as large,
    /// outside the recommended range: the column scaling takes the factor
    /// out, so the slack's unresolved entry is the same, but `2^52` units of
    /// `x2` are `2^40` times as many in the engine's, over which that entry,
    /// at its magnitude plus its error, could take the slack past its bound
    /// by more than its tolerance. The ray cannot be told from a genuine
    /// blocker there, and the solve fails loudly, presolve on and off.
    #[test]
    fn a_ray_past_an_unresolved_entry_beyond_its_tolerance_fails_loudly() {
        let problem = redundant_equality_ray(2f64.powi(40));
        for (got, presolve) in both_ways(&problem).into_iter().zip([true, false]) {
            match got {
                Err(Error::InternalError(message))
                    if message.contains("unboundedness cannot be determined") => {}
                other => panic!("presolve {presolve}: {other:?}"),
            }
        }
    }

    /// The MILP counterpart: `max -x0 + 3 x2 + 3 x3 + 3 x4`, `x1 >= 1`, `x2`
    /// integer in `[-6, -1]`, the others free, s.t. `4 x0 - x1 <= 9`,
    /// `2 x0 + x1 >= 11` and four equalities, the second `-3` times the first
    /// less the third and the fourth, within the recommended range: the point
    /// `(3, 6, -5, 0, 4)` and a ray of the relaxation, along which `x2` stays
    /// put, make it unbounded. The root's primal step pivoted on the
    /// redundant row's slack through an entry the refinement left within the
    /// round-off of its residual, `3.1e-33` against a second correction of
    /// zero, and the solve returned `Optimal` at `9.3e16`, presolve on and
    /// off. Unresolved, the entry does not pivot, and is negligible up to
    /// the magnitude taken for infinite: a ray.
    #[test]
    fn a_milp_with_a_ray_past_a_redundant_equality_does_not_pivot_on_round_off() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let x0 = problem.add_var(-1.0, free);
        let x1 = problem.add_var(0.0, (1.0, f64::INFINITY));
        let x2 = problem.add_integer_var(3.0, (-6, -1));
        let x3 = problem.add_var(3.0, free);
        let x4 = problem.add_var(3.0, free);
        problem.add_constraint([(x0, 4.0), (x1, -1.0)], ComparisonOp::Le, 9.0);
        problem.add_constraint([(x0, -5.0), (x1, 4.0), (x4, -2.0)], ComparisonOp::Eq, 1.0);
        problem.add_constraint(
            [(x0, 15.0), (x1, -16.0), (x2, 5.0), (x3, -3.0), (x4, 6.0)],
            ComparisonOp::Eq,
            -52.0,
        );
        problem.add_constraint([(x1, 1.0), (x3, 3.0), (x4, 2.0)], ComparisonOp::Eq, 14.0);
        problem.add_constraint([(x1, 3.0), (x2, -5.0), (x4, -2.0)], ComparisonOp::Eq, 35.0);
        problem.add_constraint([(x0, 2.0), (x1, 1.0)], ComparisonOp::Ge, 11.0);
        assert_unbounded(&problem, "MILP ray past a redundant equality");
    }

    /// `min -1.7 x - 1.5 y` s.t. `8.36 x + 1.7 y >= 1.18` and
    /// `45.144 x + 9.179999999998072 y = 5.372`, `x, y >= 0`, within the
    /// recommended range, is infeasible: the second row is `5.4` times the
    /// first but for `y`'s last digits and asks it for `0.995`. The first
    /// row's only repair is `y`, through an entry of `1.785e-13` that its
    /// certificate leaves on `y`'s infinite bound, so it proves nothing: the
    /// first last resort pivots on it, genuine on the refined row, and the row
    /// of `x`, `1.7e12` past its bound there, proves the model infeasible,
    /// with presolve on and off. Computed as `1.7852e-13` and refined to
    /// `1.7849e-13`, the entry did not agree to seven digits, was taken for
    /// round-off, and the solve failed loudly ("no var can repair row").
    #[test]
    fn an_infeasible_lp_of_nearly_parallel_rows_is_proven_infeasible() {
        for presolve in [true, false] {
            let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(-1.7, (0.0, f64::INFINITY));
            let y = problem.add_var(-1.5, (0.0, f64::INFINITY));
            problem.add_constraint([(x, 8.36), (y, 1.7)], ComparisonOp::Ge, 1.18);
            problem.add_constraint(
                [(x, 45.144), (y, 9.179999999998072)],
                ComparisonOp::Eq,
                5.372,
            );
            match problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            }) {
                Err(Error::Infeasible) => {}
                other => panic!("presolve {presolve}: {other:?}"),
            }
        }
    }

    /// `min -2.1 x - 1.7 y - 2.5 w` s.t. `-1.46 x - 3.23 y + 0.4 w = 1.02` and
    /// `9.636 x + 21.317997527064456 y - 2.64 w = -5.732`, `x <= 0`,
    /// `y >= 0`, `w` an integer in `[0, 3]`, within the recommended range, is
    /// infeasible: the second row is `-6.6` times the first but for `y`'s
    /// digits and its right-hand side, which leaves `y` a value below its
    /// bound. At the root LP, `y`'s row proves nothing (its certificate leaves
    /// `x` an entry of `3.7e-10` on `x`'s infinite bound), while `w`'s row
    /// proves infeasibility. The genuine entries of a row are its last resort,
    /// tried only when every other row's regular pivot and proof failed:
    /// taken at once, `x`'s entry entered with a step of `8.9e15`, into a
    /// basis the factorization finds singular, and the solve failed loudly.
    #[test]
    fn a_milp_of_nearly_parallel_rows_is_proven_infeasible_by_another_row() {
        for presolve in [true, false] {
            let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(-2.1, (f64::NEG_INFINITY, 0.0));
            let y = problem.add_var(-1.7, (0.0, f64::INFINITY));
            let w = problem.add_integer_var(-2.5, (0, 3));
            problem.add_constraint([(x, -1.46), (y, -3.23), (w, 0.4)], ComparisonOp::Eq, 1.02);
            problem.add_constraint(
                [(x, 9.636), (y, 21.317997527064456), (w, -2.64)],
                ComparisonOp::Eq,
                -5.731999999999999,
            );
            match problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            }) {
                Err(Error::Infeasible) => {}
                other => panic!("presolve {presolve}: {other:?}"),
            }
        }
    }

    /// Answers of a solve, presolve on and off.
    fn both_ways(problem: &crate::Problem) -> [Result<crate::SolveOutcome, Error>; 2] {
        [true, false].map(|presolve| {
            problem.solve_with(SolveOptions {
                presolve,
                ..SolveOptions::default()
            })
        })
    }

    /// Assert that `problem` is answered `Infeasible`, presolve on and off,
    /// reporting both answers where one is not.
    fn assert_infeasible(problem: &crate::Problem, what: &str) {
        let [on, off] = both_ways(problem);
        if !matches!(on, Err(Error::Infeasible)) || !matches!(off, Err(Error::Infeasible)) {
            panic!("{what}: presolve on {on:?}, off {off:?}");
        }
    }

    /// [`assert_solves_within_contract`] presolve on and off, both checked
    /// before a failure is reported, with the other setting's.
    fn assert_within_contract_both_ways(model: &Model) {
        let failures: Vec<String> = [true, false]
            .into_iter()
            .filter_map(|presolve| {
                let check = || {
                    if presolve {
                        assert_solves_within_contract(model)
                    } else {
                        assert_solves_within_contract_without_presolve(model)
                    }
                };
                let payload =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(check)).err()?;
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                Some(format!("presolve {presolve}: {message}"))
            })
            .collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Assert that `problem` is answered with a solution or `Unbounded`,
    /// presolve on and off, and return the solutions' objectives (`None` for
    /// `Unbounded`). A solution is the one the solve checked against the
    /// model as written.
    fn assert_answered(problem: &crate::Problem, what: &str) -> [Option<f64>; 2] {
        let [on, off] = both_ways(problem).map(|got| match got {
            Ok(outcome) if outcome.solution().is_some() => {
                Ok(Some(outcome.solution().unwrap().objective()))
            }
            Err(Error::Unbounded) => Ok(None),
            other => Err(format!("{other:?}")),
        });
        match (on, off) {
            (Ok(on), Ok(off)) => [on, off],
            (on, off) => panic!("{what}: presolve on {on:?}, off {off:?}"),
        }
    }

    /// Assert that `problem` is answered with a solution, presolve on and
    /// off, and return its objectives.
    fn assert_solved(problem: &crate::Problem, what: &str) -> [f64; 2] {
        assert_answered(problem, what).map(|got| got.unwrap_or_else(|| panic!("{what}: Unbounded")))
    }

    /// Assert that `problem` is answered `Unbounded` or with a solution whose
    /// objective lies within `1e-9` of `optimum`, relatively, presolve on and
    /// off: the solve need not reach the optimum of nearly parallel rows,
    /// whose near-null direction, within the contract, is also a ray.
    fn assert_optimal_or_unbounded(problem: &crate::Problem, optimum: f64, what: &str) {
        for got in assert_answered(problem, what).into_iter().flatten() {
            assert!(
                (got - optimum).abs() <= 1e-9 * optimum.abs().max(1.0),
                "{what}: {got} vs {optimum}"
            );
        }
    }

    /// Assert that `problem` is answered `Unbounded`, presolve on and off.
    fn assert_unbounded(problem: &crate::Problem, what: &str) {
        for got in assert_answered(problem, what) {
            assert_eq!(got, None, "{what}: a solution");
        }
    }

    /// `min x - y` s.t. `x + (1 + 2^-k) y <= -2` and `x + y >= -1`,
    /// `x >= 0`, `y` in `[y_min, 10]`, within the recommended range: the rows
    /// need `2^-k y <= -1`, so the model is infeasible for `y_min > -2^k`
    /// and its optimum is `2^(k+1) - 1` at `x = 2^k - 1`, `y = -2^k`, for
    /// `y_min = -2^(k+2)`.
    fn slanted_rows(k: i32, y_min: f64) -> crate::Problem {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_var(1.0, (0.0, f64::INFINITY));
        let y = problem.add_var(-1.0, (y_min, 10.0));
        let slanted = 1.0 + 2f64.powi(-k);
        problem.add_constraint([(x, 1.0), (y, slanted)], ComparisonOp::Le, -2.0);
        problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, -1.0);
        problem
    }

    /// [`slanted_rows`] at `k = 48` and `y` in `[-10, 10]`, within the
    /// recommended range, is infeasible: the rows need `2^-48 y <= -1`. The
    /// second row's proof needs `x`'s genuine entry, and the basis that
    /// enters it, with `y` basic, is one the factorization finds singular.
    /// A repair that drops `x` leaves its coefficient on its infinite bound
    /// in the first row's certificate, which then proves nothing, and the
    /// solve fails loudly ("no var can repair row"); the repair drops `y`,
    /// both of whose bounds are finite, and the model is proven infeasible.
    #[test]
    fn an_infeasible_lp_through_a_singular_basis_is_proven_infeasible() {
        assert_infeasible(&slanted_rows(48, -10.0), "2^-48");
    }

    /// [`slanted_rows`] for every slant `2^-k`, `k` from `20` to `52`, is
    /// proven infeasible with `y` in `[-10, 10]`, and solved at its optimum
    /// with `y` down to `-2^(k+2)`: from `k = 47`, where the factorization
    /// finds the basis of both rows singular, the singular basis must not
    /// cost the proof, nor the repair the feasible twins their optimum.
    #[test]
    fn rows_slanted_by_a_power_of_two_are_proven_infeasible_or_solved() {
        for k in 20..=52 {
            assert_infeasible(&slanted_rows(k, -10.0), &format!("2^-{k}"));
            let want = 2f64.powi(k + 1) - 1.0;
            for got in assert_solved(&slanted_rows(k, -(2f64.powi(k + 2))), &format!("2^-{k}")) {
                assert!(
                    (got - want).abs() <= 1e-12 * want,
                    "2^-{k}: {got} vs {want}"
                );
            }
        }
    }

    /// `min -0.3 x0 - 2.2 x1 - 0.6 x2` s.t. `0.31 x0 + 0.81 x1 + 6.17 x2 =
    /// 0.42`, `-0.77 x0 - 3.35 x1 - 0.36 x2 >= 2.4` and `-7.798 x0 - 32.418 x1
    /// - (20.516 + delta) x2 = 19.424`, `x0` in `[-x0_max, x0_max]`, `x1`
    /// free, `x2 >= 0`, within the recommended range: the third row is
    /// `9` times the second less `2.8` times the first but for `x2`'s
    /// coefficient (`delta` off) and its right-hand side, so on the line the
    /// two equalities leave, the second row misses by `0.111` wherever `x0`
    /// lies within `[-10, 10]`.
    fn nearly_dependent_rows(delta: f64, x0_max: f64) -> crate::Problem {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(-0.3, (-x0_max, x0_max));
        let x1 = problem.add_var(-2.2, (f64::NEG_INFINITY, f64::INFINITY));
        let x2 = problem.add_var(-0.6, (0.0, f64::INFINITY));
        problem.add_constraint([(x0, 0.31), (x1, 0.81), (x2, 6.17)], ComparisonOp::Eq, 0.42);
        problem.add_constraint(
            [(x0, -0.77), (x1, -3.35), (x2, -0.36)],
            ComparisonOp::Ge,
            2.4,
        );
        problem.add_constraint(
            [(x0, -7.798), (x1, -32.418), (x2, -20.516 - delta)],
            ComparisonOp::Eq,
            19.424,
        );
        problem
    }

    /// [`nearly_dependent_rows`], `x2`'s coefficient `1.8e-9` off, is
    /// infeasible. The second row's proof needs `x2`'s genuine entry, and the
    /// basis that enters it is so nearly singular that two refinement steps
    /// of `x0`'s row of its inverse do not converge: they leave the free
    /// basic `x1` a residual beyond the round-off of its sum, which voided
    /// the proof, and the solve failed loudly ("no var can repair row").
    /// Refined until it converges, the row proves the model infeasible.
    #[test]
    fn an_infeasible_lp_of_nearly_dependent_rows_is_proven_infeasible() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(-0.3, (-10.0, 10.0));
        let x1 = problem.add_var(-2.2, (f64::NEG_INFINITY, f64::INFINITY));
        let x2 = problem.add_var(-0.6, (0.0, f64::INFINITY));
        problem.add_constraint([(x0, 0.31), (x1, 0.81), (x2, 6.17)], ComparisonOp::Eq, 0.42);
        problem.add_constraint(
            [(x0, -0.77), (x1, -3.35), (x2, -0.36)],
            ComparisonOp::Ge,
            2.4,
        );
        problem.add_constraint(
            [(x0, -7.798), (x1, -32.418), (x2, -20.516000001846947)],
            ComparisonOp::Eq,
            19.424,
        );
        assert_infeasible(&problem, "1.8e-9");
    }

    /// [`nearly_dependent_rows`] for every `delta` from `1.8e-9` down to
    /// `1.8e-17` is proven infeasible with `x0` in `[-10, 10]`, and, for
    /// `delta` down to `1.8e-12`, solved at its exact optimum with `x0` in
    /// `[-1e12, 1e12]` scaled by `1.8e-9 / delta`, where the rows meet.
    #[test]
    fn nearly_dependent_rows_off_by_a_delta_are_proven_infeasible_or_solved() {
        let optima = [
            -213072495199.00894,
            -2130724952027.2402,
            -21307249520309.55,
            -213072495203132.7,
        ];
        let deltas = [
            1.846947e-9,
            1.846947e-10,
            1.846947e-11,
            1.846947e-12,
            1.846947e-13,
            1.846947e-14,
            1.846947e-15,
            1.846947e-16,
            1.846947e-17,
        ];
        for (j, delta) in deltas.into_iter().enumerate() {
            assert_infeasible(&nearly_dependent_rows(delta, 10.0), &format!("{delta:e}"));
            if let Some(&want) = optima.get(j) {
                let twin = nearly_dependent_rows(delta, 1e12 * 10f64.powi(j as i32));
                for got in assert_solved(&twin, &format!("{delta:e}")) {
                    assert!(
                        (got - want).abs() <= 1e-9 * want.abs(),
                        "{delta:e}: {got} vs {want}"
                    );
                }
            }
        }
    }

    /// Three rows within the recommended range, the second and the third
    /// `-3.333` and `-13` times the first but for `x0`'s coefficient in the
    /// first, meet only far out: every point has a var beyond `3.7e13`.
    /// Counted as zero within the error bound of a refinement that converged
    /// only after its second step, the non-basic `x4`'s coefficient proved
    /// the model infeasible.
    #[test]
    fn a_long_refinement_concedes_no_coefficient_of_a_feasible_lp_far_out() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(1.1, (f64::NEG_INFINITY, 5.0));
        let x1 = problem.add_var(-1.9, (f64::NEG_INFINITY, 4.0));
        problem.add_var(2.5, (0.0, f64::INFINITY));
        problem.add_var(-2.7, (-3.0, 11.0));
        let x4 = problem.add_var(2.7, (0.0, f64::INFINITY));
        problem.add_constraint(
            [(x0, 2.2400999999995452), (x1, 1.185), (x4, 0.3567)],
            ComparisonOp::Eq,
            -2.369,
        );
        problem.add_constraint(
            [(x0, -7.467), (x1, -3.95), (x4, -1.189)],
            ComparisonOp::Le,
            7.9,
        );
        problem.add_constraint(
            [
                (x0, -29.12129999999999),
                (x1, -15.405000000000001),
                (x4, -4.6371),
            ],
            ComparisonOp::Eq,
            30.815,
        );
        assert_answered(&problem, "points beyond 3.7e13");
    }

    /// Two equalities within the recommended range, the second `4.3` times
    /// the first but for `x0`'s coefficient and the right-hand side, and an
    /// inequality nearly `1.1` times the first, meet only far out: every
    /// point has a var beyond `2.6e15`. Counted as zero within the error
    /// bound of a refinement that converged only after its second step, the
    /// non-basic `x4`'s coefficient proved the model infeasible.
    #[test]
    fn a_long_refinement_concedes_no_non_basic_coefficient() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(-0.8, (0.0, 13.0));
        let x1 = problem.add_var(0.4, (-6.0, 1.0));
        problem.add_var(0.2, (0.0, f64::INFINITY));
        let x3 = problem.add_var(-2.2, (f64::NEG_INFINITY, f64::INFINITY));
        let x4 = problem.add_var(0.7, (-2.0, f64::INFINITY));
        problem.add_constraint(
            [(x0, 0.82), (x1, -2.08), (x3, 1.743), (x4, 3.017)],
            ComparisonOp::Eq,
            23.009,
        );
        problem.add_constraint(
            [
                (x0, 3.5260000000004545),
                (x1, -8.943999999999999),
                (x3, 7.4949),
                (x4, 12.973099999999999),
            ],
            ComparisonOp::Eq,
            97.93870000000409,
        );
        problem.add_constraint(
            [
                (x0, 0.902),
                (x1, -2.2880000000000003),
                (x3, 1.917300000000002),
                (x4, 3.3187),
            ],
            ComparisonOp::Le,
            25.508900000000008,
        );
        assert_answered(&problem, "points beyond 2.6e15");
    }

    /// Three rows within the recommended range, nearly multiples of one
    /// another, are unbounded, every point with a var beyond `3.4e12`.
    /// Counted as zero within the multipliers' error bound of a refinement
    /// that converged in two steps, two non-basic coefficients proved the
    /// model infeasible; a non-basic coefficient counts as zero only within
    /// the round-off of its sum, and the model is unbounded.
    #[test]
    fn an_unbounded_lp_of_nearly_dependent_rows_is_not_proven_infeasible() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let x0 = problem.add_var(2.2, (-8.0, -6.0));
        let x1 = problem.add_var(-0.9, (f64::NEG_INFINITY, 1.0));
        let x2 = problem.add_var(2.0, (-5.0, f64::INFINITY));
        let x3 = problem.add_var(0.5, (-4.0, f64::INFINITY));
        problem.add_constraint(
            [(x0, -9.282), (x1, -7.88), (x2, 8.253), (x3, -8.7)],
            ComparisonOp::Le,
            113.994,
        );
        problem.add_constraint(
            [
                (x0, -32.486999999999995),
                (x1, -27.58),
                (x2, 28.8855),
                (x3, -30.449999999999996),
            ],
            ComparisonOp::Ge,
            398.62999999999994,
        );
        problem.add_constraint(
            [
                (x0, 50.122800238418584),
                (x1, 42.552),
                (x2, -44.5662),
                (x3, 46.98),
            ],
            ComparisonOp::Eq,
            -615.0226016689301,
        );
        assert_unbounded(&problem, "points beyond 3.4e12");
    }

    /// An inequality and an equality within the recommended range, nearly
    /// parallel, meet only far out, every point with a var beyond `2.8e11`,
    /// the optimum `-2.186155248e11`. Counted as zero within the
    /// multipliers' error bound of a refinement that converged in two steps,
    /// a non-basic coefficient proved the model infeasible.
    #[test]
    fn a_feasible_lp_of_nearly_parallel_rows_is_not_proven_infeasible() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let x0 = problem.add_var(-2.6, (0.0, f64::INFINITY));
        let x1 = problem.add_var(-2.4, (-7.0, 2.0));
        let x2 = problem.add_var(2.1, (-3.0, f64::INFINITY));
        problem.add_constraint(
            [(x0, 6.734), (x1, 3.02), (x2, -7.759)],
            ComparisonOp::Ge,
            18.625,
        );
        problem.add_constraint(
            [
                (x0, -40.403999999999996),
                (x1, -18.11999988079071),
                (x2, 46.554),
            ],
            ComparisonOp::Eq,
            -111.74900071525573,
        );
        assert_optimal_or_unbounded(&problem, -218615524799.95874, "points beyond 2.8e11");
    }

    /// Two equalities and two inequalities within the recommended range, the
    /// fourth nearly a combination of the others, meet only far out, every
    /// point with a var beyond `1.2e13`, the optimum `4.413385618588e14`.
    /// Counted as zero within the multipliers' error bound of a refinement
    /// that converged in two steps, non-basic coefficients proved the model
    /// infeasible.
    #[test]
    fn a_feasible_lp_met_far_out_is_not_proven_infeasible() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let x0 = problem.add_var(0.3, (-5.0, f64::INFINITY));
        let x1 = problem.add_var(-0.4, (0.0, 5.0));
        let x2 = problem.add_var(2.9, (-8.0, 11.0));
        let x3 = problem.add_var(0.9, (-5.0, f64::INFINITY));
        let x4 = problem.add_var(1.7, (f64::NEG_INFINITY, f64::INFINITY));
        problem.add_constraint(
            [
                (x0, 14.300000000000004),
                (x1, -4.73),
                (x2, -3.06),
                (x3, 138.6932),
                (x4, -44.8324),
            ],
            ComparisonOp::Le,
            -213.9716,
        );
        problem.add_constraint(
            [(x0, 5.944), (x1, 3.17), (x3, -8.324)],
            ComparisonOp::Eq,
            19.817999999999998,
        );
        problem.add_constraint(
            [
                (x0, -9.97),
                (x1, -3.5),
                (x2, 0.45),
                (x3, -9.379),
                (x4, 6.593),
            ],
            ComparisonOp::Le,
            5.421999999999999,
        );
        problem.add_constraint(
            [
                (x0, -101.86860000000001),
                (x1, -42.153),
                (x2, 3.0150000030150004),
                (x3, -13.727699999999999),
                (x4, 44.1731),
            ],
            ComparisonOp::Eq,
            -81.939799978895,
        );
        assert_optimal_or_unbounded(&problem, 441338561858807.9, "points beyond 1.2e13");
    }

    /// Three equalities within the recommended range, nearly multiples of
    /// one another (`x3`'s coefficient in the second `1e-11` off), are
    /// unbounded, every point with a var beyond `2.3e13`. The factorization
    /// finds a basis of their columns singular, and the repair that drops a
    /// fixed slack, both of whose bounds are finite, leads to a proof that
    /// counted two non-basic coefficients as zero within the multipliers'
    /// error bound and proved the model infeasible; a non-basic coefficient
    /// counts as zero only within the round-off of its sum.
    #[test]
    fn a_repaired_basis_leads_to_no_proof_through_a_non_basic_coefficient() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        problem.add_var(0.7, (f64::NEG_INFINITY, 0.0));
        let x1 = problem.add_var(1.2, (-3.0, f64::INFINITY));
        let x2 = problem.add_var(1.7, (-2.0, f64::INFINITY));
        let x3 = problem.add_var(2.2, (f64::NEG_INFINITY, 1.0));
        problem.add_constraint(
            [(x1, -4.428), (x2, 9.02), (x3, 3.1119)],
            ComparisonOp::Eq,
            21.1469,
        );
        problem.add_constraint(
            [
                (x1, 7.776000000000001),
                (x2, -15.840000000000002),
                (x3, -5.464800054648),
            ],
            ComparisonOp::Eq,
            -37.139800054648006,
        );
        problem.add_constraint(
            [(x1, -1.08), (x2, 2.2), (x3, 0.759)],
            ComparisonOp::Eq,
            5.159000000000001,
        );
        assert_unbounded(&problem, "points beyond 2.3e13");
    }

    /// `min 1.9 x0 - 1.5 x1` s.t. two equalities within the recommended
    /// range, the second `-4.5` times the first but for the last digits,
    /// `x0` free, `x1 >= -3`, meet at one point, the optimum `3.8939321` at
    /// `(4.2566, 2.7958)`. The factorization finds their basis singular and
    /// drops the free `x0`. A repair that drops `x1` instead, whose other
    /// bound is infinite, put it at `-3`, and the solve failed loudly ("every
    /// basis change left leads to a singular basis"): a repair drops another
    /// var than the factorization's only where each it drops has both bounds
    /// finite.
    #[test]
    fn a_basis_repair_keeps_its_drop_where_the_other_has_an_infinite_bound() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(1.9, (f64::NEG_INFINITY, f64::INFINITY));
        let x1 = problem.add_var(-1.5, (-3.0, f64::INFINITY));
        problem.add_constraint([(x0, -0.08), (x1, -2.059)], ComparisonOp::Eq, -6.097);
        problem.add_constraint(
            [(x0, 0.3599999999999998), (x1, 9.265500000000001)],
            ComparisonOp::Eq,
            27.436500000000002,
        );
        assert_optimal_or_unbounded(&problem, 3.893932106797555, "one point");
    }

    /// Three equalities within the recommended range, the first two `-5.3`
    /// and `-8.6` times the third but for their last digits, are infeasible:
    /// the smallest relaxation of the rows that admits a point is `0.24`.
    /// The factorization finds a basis of their columns singular; ordered by
    /// the vars' bounds it finds no dependency, and a repair that kept that
    /// basis led to values near `1.8e13`, where the rows' round-off floor
    /// accepted them: `Optimal`. Where the second factorization finds
    /// another number of dependencies, the first one's repair stands, and
    /// the model is proven infeasible.
    #[test]
    fn a_basis_repair_stands_where_the_ranked_factorization_finds_no_dependency() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(0.6, (f64::NEG_INFINITY, 5.0));
        problem.add_var(0.4, (-7.0, 10.0));
        let x2 = problem.add_var(1.1, (-4.0, f64::INFINITY));
        problem.add_var(1.0, (-2.0, f64::INFINITY));
        let x4 = problem.add_var(1.8, (f64::NEG_INFINITY, f64::INFINITY));
        problem.add_constraint(
            [(x0, -39.56450000000005), (x2, -51.7439), (x4, 51.5372)],
            ComparisonOp::Eq,
            -299.27680000000026,
        );
        problem.add_constraint(
            [(x0, -64.19900000000091), (x2, -83.9618), (x4, 83.6264)],
            ComparisonOp::Eq,
            -486.24160000000455,
        );
        problem.add_constraint(
            [(x0, 7.465), (x2, 9.763), (x4, -9.724)],
            ComparisonOp::Eq,
            56.656,
        );
        assert_infeasible(&problem, "relaxation 0.24");
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): `min 0.4 x0 + 0.4 x1` s.t. two
    /// equalities and an inequality within the recommended range, each
    /// nearly a multiple of the others, `x0 >= 0`, `x1` free, are infeasible:
    /// the smallest relaxation of the rows that admits a point is `9.8e-5`.
    /// The factorization finds the basis of `x0` and `x1` singular and drops
    /// `x1`, whose coefficient then voids the proof on its infinite bounds;
    /// `x0`, the only other var of the dependency, has an infinite bound
    /// too, so no repair counts both at a bound, and the solve fails loudly
    /// where it should answer `Infeasible`.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn an_infeasible_lp_whose_dependency_has_no_boxed_var_is_a_known_limitation() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(0.4, (0.0, f64::INFINITY));
        let x1 = problem.add_var(0.4, (f64::NEG_INFINITY, f64::INFINITY));
        problem.add_constraint(
            [(x0, 70.9800000000001), (x1, -35.945)],
            ComparisonOp::Eq,
            175.17600000000053,
        );
        problem.add_constraint([(x0, -7.8), (x1, 3.95)], ComparisonOp::Eq, -19.25);
        problem.add_constraint(
            [(x0, 17.16000000372529), (x1, -8.690000000000001)],
            ComparisonOp::Le,
            42.549000018626444,
        );
        assert_infeasible(&problem, "relaxation 9.8e-5");
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): `max -2.1 x0 + 1.9 x1` s.t. two
    /// equalities within the recommended range, the first `8.6` times the
    /// second but for the last digits, `x0 >= -4`, `x1` free, meet at one
    /// point, the optimum `10.1246858` at `(-1.6009, 3.5594)`, where
    /// `Unbounded`, through the rows' near-null direction, is as admissible
    /// within the contract. The factorization finds the basis of `x0` and
    /// `x1` singular, and neither has both bounds finite: the repair drops
    /// `x0`, the primal simplex is left only changes that lead back to that
    /// basis, and the solve fails loudly where it should answer.
    #[test]
    #[should_panic(expected = "every basis change left leads to a singular basis")]
    fn a_feasible_lp_whose_dependency_has_no_boxed_var_is_a_known_limitation() {
        let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
        let x0 = problem.add_var(-2.1, (-4.0, f64::INFINITY));
        let x1 = problem.add_var(1.9, (f64::NEG_INFINITY, f64::INFINITY));
        problem.add_constraint(
            [(x0, 68.7054), (x1, -49.02000000000001)],
            ComparisonOp::Eq,
            -284.47080000000005,
        );
        problem.add_constraint([(x0, 7.989), (x1, -5.7)], ComparisonOp::Eq, -33.078);
        assert_optimal_or_unbounded(&problem, 10.124685778557733, "one point");
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): three rows within the
    /// recommended range, each nearly a multiple of the others, over
    /// `x1 >= 0`, `x3 >= -1` and `x4 <= 4`, are infeasible: the smallest
    /// relaxation of the rows that admits a point is `6e-4`. The proof needs
    /// the non-basic `x1`'s coefficient, on its infinite bound, within the
    /// multipliers' error but beyond the round-off of its sum, to count as
    /// zero, and the solve fails loudly where it should answer `Infeasible`.
    #[test]
    #[should_panic(expected = "no var can repair row")]
    fn a_proof_through_a_non_basic_coefficient_within_its_error_is_a_known_limitation() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        problem.add_var(-0.8, (f64::NEG_INFINITY, f64::INFINITY));
        let x1 = problem.add_var(0.9, (0.0, f64::INFINITY));
        problem.add_var(-2.3, (f64::NEG_INFINITY, f64::INFINITY));
        let x3 = problem.add_var(-2.2, (-1.0, f64::INFINITY));
        let x4 = problem.add_var(2.5, (f64::NEG_INFINITY, 4.0));
        problem.add_constraint(
            [(x1, -48.68500000000005), (x3, 56.0742), (x4, -34.0522)],
            ComparisonOp::Ge,
            85.42599999999977,
        );
        problem.add_constraint(
            [
                (x1, -35.309999999999995),
                (x3, 40.6692),
                (x4, -24.697200246972),
            ],
            ComparisonOp::Eq,
            62.101,
        );
        problem.add_constraint(
            [(x1, 5.35), (x3, -6.162), (x4, 3.742)],
            ComparisonOp::Le,
            -9.41,
        );
        assert_infeasible(&problem, "relaxation 6e-4");
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): `min 0.8 x0 + 2.6 x1 + 0.1 x2`
    /// s.t. three nearly parallel rows within the recommended range,
    /// `x0 >= -1`, `x1 >= -3` and `x2` in `[-4, 15]`, is infeasible beyond
    /// the absolute tolerance: the smallest relaxation of the rows that
    /// admits a point is `1.9e-4`, and only the round-off floor of the rows
    /// near `(5.8e8, 1.05e9, 15)` admits one, which is answered `Infeasible`
    /// (ARCHITECTURE §7.1). No row's certificate proves it; the last resort of the dual phase moves
    /// `x0` by some `4.7e9`, and the next pivot's element is `4.2e-7` through
    /// the row and zero through the column of a fresh factorization at the
    /// highest threshold, so the solve fails loudly where it should answer
    /// `Infeasible`, presolve on and off.
    #[test]
    #[should_panic(expected = "0e0 through the column of a fresh factorization")]
    fn an_infeasible_lp_whose_pivot_row_and_column_disagree_is_a_known_limitation() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x0 = problem.add_var(0.8, (-1.0, f64::INFINITY));
        let x1 = problem.add_var(2.6, (-3.0, f64::INFINITY));
        let x2 = problem.add_var(0.1, (-4.0, 15.0));
        problem.add_constraint(
            [
                (x0, -43.071600000000004),
                (x1, 23.760000000000005),
                (x2, -9.592000000959203),
            ],
            ComparisonOp::Le,
            -163.13540000383682,
        );
        problem.add_constraint(
            [(x0, -61.6707), (x1, 34.020000003402004), (x2, -13.734)],
            ComparisonOp::Ge,
            -232.67879999319598,
        );
        problem.add_constraint(
            [(x0, -9.789), (x1, 5.4), (x2, -2.18)],
            ComparisonOp::Eq,
            -37.076,
        );
        assert_infeasible(&problem, "relaxation 1.9e-4");
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9):
    /// `min 4e6 x0 + 3e6 x4 - 5e6 x5 - 8e6 x6` over seven vars in
    /// `[-1000, 1000]` s.t. two rows within the recommended range, nearly
    /// parallel (each coefficient of the first within `2e-10` of the
    /// second's, relatively), `>= -3656` and `<= -3656`: the optimum is
    /// `-1.1108571e10`, at `x*`. Each primal optimum leaves a slack `2.5e-6`
    /// past its bound once the model's bounds replace the working ones, and
    /// each vertex the dual phase restores leaves reduced costs dual
    /// infeasible, so the solve fails loudly where it should answer, presolve
    /// on and off.
    #[test]
    #[should_panic(expected = "8 phase rounds")]
    fn an_lp_whose_phases_undo_each_other_is_a_known_limitation() {
        let var = |obj| real(obj, -1000.0, 1000.0);
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                var(4e6),
                var(0.0),
                var(0.0),
                var(0.0),
                var(3e6),
                var(-5e6),
                var(-8e6),
            ],
            rows: vec![
                row(
                    &[
                        (0, 7.0000000007),
                        (1, -8.9999999991),
                        (2, 3.0000000006),
                        (3, 1.0000000002),
                        (4, 5.0000000005),
                        (5, -7.0000000014),
                        (6, -4.0000000004),
                    ],
                    ComparisonOp::Ge,
                    -3656.0,
                ),
                row(
                    &[
                        (0, 7.0),
                        (1, -9.0),
                        (2, 3.0),
                        (3, 1.0),
                        (4, 5.0),
                        (5, -7.0),
                        (6, -4.0),
                    ],
                    ComparisonOp::Le,
                    -3656.0,
                ),
            ],
            xstar: vec![
                -1000.0,
                -321.77777777777777,
                1000.0,
                1000.0,
                -1000.0,
                -778.2857142857143,
                1000.0,
            ],
        };
        assert!(xstar_is_feasible(&model), "fixture: x* holds the rows");
        assert_within_contract_both_ways(&model);
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): ten equalities over eight vars
    /// within the recommended range, at zero cost, exactly dependent but for
    /// the fourth's right-hand side, `1e-6` off (`-55.999999` for `-56`): no
    /// point meets them exactly, but `x*` meets each within `8.9e-8`, inside
    /// the tolerance. No row's certificate proves anything, the rows agreeing
    /// within the room the engine can give them, and the dual phase runs out
    /// of basis changes that do not return to visited vertices, so the solve
    /// fails loudly where it should answer, presolve on and off.
    #[test]
    #[should_panic(expected = "dual simplex cycles")]
    fn an_lp_of_equalities_agreeing_only_within_their_tolerances_is_a_known_limitation() {
        let free = |obj| real(obj, f64::NEG_INFINITY, f64::INFINITY);
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                free(0.0),
                free(0.0),
                free(0.0),
                free(0.0),
                real(0.0, -1.0, f64::INFINITY),
                real(0.0, -7.0, -6.0),
                real(0.0, f64::NEG_INFINITY, -2.0),
                real(0.0, f64::NEG_INFINITY, 6.0),
            ],
            rows: vec![
                row(
                    &[
                        (0, 4.0),
                        (1, -1.0),
                        (3, 5.0),
                        (5, 3.0),
                        (6, -1.0),
                        (7, -5.0),
                    ],
                    ComparisonOp::Eq,
                    -25.0,
                ),
                row(
                    &[
                        (0, -5.0),
                        (1, -3.0),
                        (2, -2.0),
                        (3, -5.0),
                        (5, 3.0),
                        (7, -3.0),
                    ],
                    ComparisonOp::Eq,
                    -41.0,
                ),
                row(
                    &[(0, -4.0), (1, -4.0), (4, -3.0), (5, -5.0), (7, -1.0)],
                    ComparisonOp::Eq,
                    9.0,
                ),
                row(
                    &[
                        (0, 3.0),
                        (1, -7.0),
                        (2, -13.0),
                        (3, -10.0),
                        (4, 12.0),
                        (5, 17.0),
                        (6, -8.0),
                        (7, -12.0),
                    ],
                    ComparisonOp::Eq,
                    -55.999999,
                ),
                row(&[(3, -2.0), (6, -2.0)], ComparisonOp::Eq, 4.0),
                row(
                    &[
                        (0, 21.0),
                        (1, -11.0),
                        (2, -1.0),
                        (3, 5.0),
                        (4, -12.0),
                        (5, 1.0),
                        (6, 14.0),
                        (7, -15.0),
                    ],
                    ComparisonOp::Eq,
                    -19.0,
                ),
                row(
                    &[(0, 4.0), (1, -3.0), (2, -2.0), (5, 2.0), (6, -2.0)],
                    ComparisonOp::Eq,
                    20.0,
                ),
                row(
                    &[
                        (0, -17.0),
                        (1, -13.0),
                        (2, -1.0),
                        (3, -9.0),
                        (4, -20.0),
                        (5, -15.0),
                        (6, 3.0),
                        (7, -14.0),
                    ],
                    ComparisonOp::Eq,
                    -48.0,
                ),
                row(
                    &[(2, -4.0), (3, -1.0), (4, -4.0), (7, -3.0)],
                    ComparisonOp::Eq,
                    9.0,
                ),
                row(
                    &[(0, 1.0), (2, -4.0), (3, -3.0), (5, -4.0)],
                    ComparisonOp::Eq,
                    52.0,
                ),
            ],
            xstar: vec![
                4.000000012774403,
                -1.2943612927196516e-8,
                -5.9999999893756035,
                -1.9168361472627195e-8,
                3.739222182419721e-8,
                -6.000000015171468,
                -2.0000000250655248,
                4.999999971856553,
            ],
        };
        assert!(xstar_is_feasible(&model), "fixture: x* holds the rows");
        assert_within_contract_both_ways(&model);
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): with `e = 2^-52` and `p`, `t`,
    /// `r` the exact dyadic values of the literals below, these rows are
    /// feasible in exact arithmetic. Set `x = 1`, `d = t - p(1-e)`,
    /// `j = d/e^2`, `k = p - (1+e)j`, and `z = r-j`. The first and third
    /// rows then hold by substitution; the second uses
    /// `(1-e)(1+e) = 1-e^2` and becomes `t + p(1-e) + d = 2t`.
    /// The free vars have no bounds to violate, and doubling the literals
    /// and forming `1 +/- e` are exact in f64. The coefficient concessions
    /// can discard the genuine residual of these nearly dependent rows and
    /// report this feasible zero-objective model `Infeasible`.
    #[test]
    #[should_panic(expected = "Err(Infeasible)")]
    fn a_feasible_lp_with_a_tiny_cofactor_is_a_known_limitation() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let x = problem.add_var(0.0, (0.0, 1.0));
        let j = problem.add_var(0.0, (f64::NEG_INFINITY, f64::INFINITY));
        let k = problem.add_var(0.0, (f64::NEG_INFINITY, f64::INFINITY));
        let z = problem.add_var(0.0, (f64::NEG_INFINITY, f64::INFINITY));
        let e = f64::EPSILON;
        let (p, t, r) = (0.338, 0.262, 1.5819999999999999);
        problem.add_constraint([(x, p), (j, 1.0 + e), (k, 1.0)], ComparisonOp::Eq, 2.0 * p);
        problem.add_constraint([(x, t), (j, 1.0), (k, 1.0 - e)], ComparisonOp::Eq, 2.0 * t);
        problem.add_constraint([(x, r), (j, 1.0), (z, 1.0)], ComparisonOp::Eq, 2.0 * r);
        assert_eq!(assert_solved(&problem, "feasible cofactor model"), [0.0; 2]);
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): `min -x9` over ten free vars
    /// s.t. nine equalities and two inequalities within the recommended
    /// range, nearly dependent (some coefficients `2^-26` off a multiple of
    /// another row's), is unbounded. The primal ratio test's column is
    /// blocked only by an entry within the round-off floor of its refinement,
    /// which does not converge on that basis: no error is measured to bound
    /// the entry by, so no ray is taken, and the solve fails loudly where it
    /// should answer `Unbounded`, presolve on and off.
    #[test]
    #[should_panic(expected = "unboundedness cannot be determined")]
    fn a_ray_past_an_unresolved_entry_of_an_unconverged_refinement_is_a_known_limitation() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let x: Vec<_> = (0..10)
            .map(|j| problem.add_var(if j == 9 { -1.0 } else { 0.0 }, free))
            .collect();
        problem.add_constraint(
            [
                (x[0], -6.0),
                (x[1], -8.0),
                (x[2], 12.0),
                (x[3], -10.0),
                (x[4], -1.0),
                (x[5], -2.0),
                (x[6], 16.0),
                (x[7], -2.0),
                (x[8], 10.0),
                (x[9], -1.0),
            ],
            ComparisonOp::Eq,
            268.0,
        );
        problem.add_constraint(
            [(x[2], -1.0), (x[3], -3.999999985098839), (x[5], -4.0)],
            ComparisonOp::Eq,
            76.00000001490116,
        );
        problem.add_constraint([(x[1], -3.0), (x[9], 3.0)], ComparisonOp::Eq, 32.0);
        problem.add_constraint(
            [
                (x[0], -3.0),
                (x[1], 11.0),
                (x[2], -1.0),
                (x[3], 8.000000014901161),
                (x[4], 6.0),
                (x[5], -11.999999985098839),
                (x[6], -12.0),
                (x[7], -8.0),
                (x[8], -4.0),
                (x[9], -14.0),
            ],
            ComparisonOp::Eq,
            99.99999995529652,
        );
        problem.add_constraint(
            [
                (x[0], -3.0),
                (x[1], -4.0),
                (x[3], 2.0),
                (x[7], 5.0),
                (x[8], 2.0),
            ],
            ComparisonOp::Eq,
            -24.0,
        );
        problem.add_constraint(
            [
                (x[1], 6.0),
                (x[2], 1.0),
                (x[3], 6.0),
                (x[4], 4.0),
                (x[6], -4.0),
                (x[7], -3.0),
                (x[8], -6.0),
                (x[9], -5.0),
            ],
            ComparisonOp::Eq,
            -36.0,
        );
        problem.add_constraint(
            [(x[3], -4.0), (x[5], -5.0), (x[6], 5.0), (x[7], -2.0)],
            ComparisonOp::Le,
            9.0,
        );
        problem.add_constraint(
            [
                (x[1], 15.0),
                (x[2], 8.0),
                (x[3], 25.999999970197678),
                (x[4], 5.0),
                (x[5], 17.999999970197678),
                (x[6], -4.0),
                (x[7], -8.0),
                (x[8], -6.0),
                (x[9], -14.0),
            ],
            ComparisonOp::Eq,
            -415.99999991059303,
        );
        problem.add_constraint(
            [(x[2], 3.0), (x[5], 4.0), (x[6], 4.0)],
            ComparisonOp::Eq,
            -32.0,
        );
        problem.add_constraint(
            [(x[2], 3.0), (x[5], 3.999999985098839), (x[6], 4.0)],
            ComparisonOp::Eq,
            -31.999999940395355,
        );
        problem.add_constraint(
            [(x[0], -5.0), (x[6], 4.0), (x[7], -4.0), (x[8], -4.0)],
            ComparisonOp::Ge,
            -44.0,
        );
        assert_unbounded(&problem, "nearly dependent equalities");
    }

    /// KNOWN LIMITATION (ARCHITECTURE §7.9): `min 4 x0 - x1 + 2^38 x2` s.t.
    /// `-2 x1 - 2^-73 x2 + w = -2^-109` and three equalities over `x0`, `x1`
    /// and `x2`, the third five times the first plus the second, `w >= 0`
    /// and the other vars free, outside the recommended range, is bounded:
    /// the equalities hold `x1` at zero, so `w >= 0` holds `x2` at `2^-36` or
    /// more, and the optimum is `24`. The column of `x2` meets that blocker
    /// only through entries its refinement leaves within their measured
    /// correction, which count as zero where no entry is unresolved, and the
    /// solve answers `Unbounded`, presolve on and off.
    #[test]
    #[should_panic(expected = "Unbounded")]
    fn a_ray_past_an_entry_within_its_measured_correction_is_a_known_limitation() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let x0 = problem.add_var(4.0, free);
        let x1 = problem.add_var(-1.0, free);
        let x2 = problem.add_var(2f64.powi(38), free);
        let w = problem.add_var(0.0, (0.0, f64::INFINITY));
        problem.add_constraint(
            [(x1, -2.0), (x2, -(2f64.powi(-73))), (w, 1.0)],
            ComparisonOp::Eq,
            -(2f64.powi(-109)),
        );
        problem.add_constraint(
            [(x0, -3.0), (x1, 1.0), (x2, -(2f64.powi(37)))],
            ComparisonOp::Eq,
            -17.0,
        );
        problem.add_constraint(
            [(x0, -3.0), (x1, 3.0), (x2, -(2f64.powi(37)))],
            ComparisonOp::Eq,
            -17.0,
        );
        problem.add_constraint(
            [(x0, -18.0), (x1, 8.0), (x2, -3.0 * 2f64.powi(38))],
            ComparisonOp::Eq,
            -102.0,
        );
        for objective in assert_solved(&problem, "a blocker within its measured correction") {
            assert!((objective - 24.0).abs() <= 1e-6, "{objective}");
        }
    }

    /// `min -3 x1 + 3 x3` s.t. six equalities of rank four within the
    /// recommended range, `x1` in `[1, 7]`, `x5 <= 1` and the other vars free,
    /// is unbounded. The primal phase meets the ray past an unresolved entry
    /// with the last row's slack `4.4e-6` off its value, its working bound
    /// shifted past it, beyond its tolerance: measured from there, the slack
    /// would have no room left. The ray starts from the point the phase started
    /// on, which the verification accepted, where the slack has its tolerance's
    /// room: `Unbounded`, presolve on and off.
    #[test]
    fn an_unresolved_ray_starts_from_the_verified_point_the_phase_left() {
        let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        let x0 = problem.add_var(0.0, free);
        let x1 = problem.add_var(-3.0, (1.0, 7.0));
        let x2 = problem.add_var(0.0, free);
        let x3 = problem.add_var(3.0, free);
        let x4 = problem.add_var(0.0, free);
        let x5 = problem.add_var(0.0, (f64::NEG_INFINITY, 1.0));
        let x6 = problem.add_var(0.0, free);
        problem.add_constraint(
            [(x0, -4.0), (x1, 2.0), (x2, 3.0), (x4, -4.0)],
            ComparisonOp::Eq,
            -29.0,
        );
        problem.add_constraint(
            [
                (x0, -4.0),
                (x1, -4.0),
                (x2, 5.0),
                (x3, -8.0),
                (x4, 6.0),
                (x5, 6.0),
                (x6, -6.0),
            ],
            ComparisonOp::Eq,
            -105.0,
        );
        problem.add_constraint(
            [(x1, -6.0), (x3, 2.0), (x5, 6.0), (x6, 2.0)],
            ComparisonOp::Eq,
            -20.0,
        );
        problem.add_constraint(
            [
                (x0, -8.0),
                (x1, 4.0),
                (x2, 4.0),
                (x3, 10.0),
                (x4, -18.0),
                (x6, 8.0),
            ],
            ComparisonOp::Eq,
            -2.0,
        );
        problem.add_constraint(
            [
                (x0, -7.0),
                (x1, -10.0),
                (x2, 3.0),
                (x3, 8.0),
                (x4, -4.0),
                (x5, 12.0),
                (x6, 3.0),
            ],
            ComparisonOp::Eq,
            -77.0,
        );
        problem.add_constraint(
            [
                (x0, -11.0),
                (x1, -14.0),
                (x2, 7.0),
                (x3, 5.0),
                (x4, -3.0),
                (x5, 18.0),
                (x6, 1.0),
            ],
            ComparisonOp::Eq,
            -154.0,
        );
        assert_unbounded(&problem, "a ray from the verified start");
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

    /// The origin, the only exactly feasible point, is optimal. The first
    /// dual phase ends at `x4 = 1`, `x3 = 1e9`, where `x0 - 1e-7 x4 = 0`
    /// misses its tolerance, and the cleanup dual phase takes `x4` back to
    /// `0`. Unrefined, `x3` kept the round-off of its computation, which that
    /// step left past its bound beyond its tolerance; the repair of its row
    /// led to a primal step on an entry some `1e-14` of its column into a
    /// basis the factorization finds singular (a 1-norm condition of about
    /// `4.5e14` in the scaled data), whose change back stayed taboo: a loud
    /// failure. Refined, the step leaves `x3` at `0`. The model lies outside
    /// the recommended range (a row dynamism of `1e11`).
    #[test]
    fn an_optimal_basis_the_factorization_finds_singular_is_avoided() {
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

    /// The origin, the only exactly feasible point, is optimal. The primal
    /// phase took the slack of `-100 x2 - 1e-7 x3 <= 0` into the basis on a
    /// zero step and an element of `-5e-15`; the factorization found the
    /// basis singular, the repair took `x2` out, and the change back stayed
    /// taboo, the step having moved no row: no basis change was left, a loud
    /// failure. Below the pivot tolerance, that element is no pivot, and the
    /// solve reaches the origin. The rows lie outside the recommended range
    /// (row dynamism `1e9`).
    #[test]
    fn a_primal_step_on_an_element_below_the_pivot_tolerance_is_not_taken() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(0.0, 0, 0),
                real(1.0, -1.0, 0.0),
                real(1.0, 0.0, 100000011.0),
                real(0.0, -1.0, 0.0),
                real(-1.0, 0.0, 0.0),
            ],
            rows: vec![
                row(&[(1, 0.0001), (3, 1.0)], ComparisonOp::Eq, 0.0),
                row(&[(2, -100.0), (3, -1e-7)], ComparisonOp::Le, 0.0),
                row(&[(1, 1.0), (2, 1e-8)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 5],
        });
    }

    /// `-13.1 x0 + 0.0112 x1 = -3434086.4` with `x0` fixed at `262144` puts
    /// `x1` exactly at its bound `0`, but the plain sum that first computes
    /// the row cancels `3.4e6` against itself and leaves an ulp of it: `x1`
    /// sits `1.2e-8` past its bound, within its tolerance, and the plain
    /// residual cannot see that. The equality
    /// `357 x1 - 2.21 x2 + 0.00417 x3 + 9.98 x4 = -19.96` amplifies it into
    /// `x3 = -1e-3`, an objective `1e-3` worse than the optimum, `0` at `x*`.
    /// The verification refines the values with residuals summed in
    /// compensated arithmetic, which see the ulp: the optimum, as an LP and
    /// as a MILP.
    #[test]
    fn round_off_putting_a_var_past_its_bound_is_refined_away() {
        for milp in [false, true] {
            let var = |obj: f64, lo: i32, hi: i32| {
                if milp {
                    int(obj, lo, hi)
                } else {
                    real(obj, lo as f64, hi as f64)
                }
            };
            let model = || Model {
                direction: OptimizationDirection::Minimize,
                vars: vec![
                    var(0.0, 262144, 262144),
                    real(1.0, -2.0, 0.0),
                    real(-2.0, 0.0, 1.0),
                    real(-1.0, -2.0, 0.0),
                    var(0.5, -2, 0),
                    real(1.0, 0.0, 1.0),
                ],
                rows: vec![
                    row(&[(0, -13.1), (1, 0.0112)], ComparisonOp::Eq, -3434086.4),
                    row(
                        &[(2, -2.07), (4, 0.00218), (5, 0.00173)],
                        ComparisonOp::Eq,
                        -0.0026300000000000004,
                    ),
                    row(&[(5, -96.9)], ComparisonOp::Ge, -97.9),
                    row(
                        &[(1, 357.0), (2, -2.21), (3, 0.00417), (4, 9.98)],
                        ComparisonOp::Eq,
                        -19.96,
                    ),
                ],
                xstar: vec![262144.0, 0.0, 0.0, 0.0, -2.0, 1.0],
            };
            check(model());
            assert_exact_optimum(&model(), 0.0);
        }
    }

    /// Solves `model` with presolve on and off, and asserts each objective
    /// within `1e-12` of `optimum`, the exact optimum of the model as written.
    fn assert_exact_optimum(model: &Model, optimum: f64) {
        let (problem, _) = model.build();
        for presolve in [true, false] {
            let options = SolveOptions {
                presolve,
                ..SolveOptions::default()
            };
            let solution = problem
                .solve_with(options)
                .unwrap()
                .into_solution()
                .unwrap();
            let objective = solution.objective();
            assert!(
                (objective - optimum).abs() <= 1e-12,
                "presolve {presolve}: objective {objective:e}, optimum {optimum:e}:\n{}",
                model.source()
            );
        }
    }

    /// `-432 x0 - 0.00552 x3 = -432001296` with `x0` fixed at `1000003` puts
    /// `x3` exactly at `0`, inside its bounds `[-2, 0]`, but the dual step
    /// that takes `x3` into the basis reads the row's slack from a plain sum
    /// that cancels `4.3e8` against itself, and leaves `x3` at `-5e-6`: no
    /// var lies past its bound, and the plain residual, `2.8e-8`, is below
    /// half an ulp of the row's terms. The equality `0.015 x1 + 150 x3 = 0`
    /// amplifies it into `x1 = 0.05`, an objective `0.025` worse than the
    /// optimum, `-2` at `x*`. Values the rows hold are refined too: the
    /// optimum, as an LP and as a MILP.
    #[test]
    fn round_off_inside_a_vars_bounds_is_refined_away() {
        for milp in [false, true] {
            let model = || Model {
                direction: OptimizationDirection::Minimize,
                vars: vec![
                    if milp {
                        int(0.0, 1000003, 1000003)
                    } else {
                        real(0.0, 1000003.0, 1000003.0)
                    },
                    real(0.5, 0.0, 1.0),
                    real(-2.0, -2.0, 0.0),
                    real(0.0, -2.0, 0.0),
                    real(1.0, -2.0, 0.0),
                ],
                rows: vec![
                    row(
                        &[(0, -432.0), (3, -0.00552)],
                        ComparisonOp::Eq,
                        -432001296.0,
                    ),
                    row(
                        &[(1, 0.0435), (2, 13.0), (3, 0.00164), (4, -234.0)],
                        ComparisonOp::Ge,
                        466.0,
                    ),
                    row(&[(1, 0.015), (3, 150.0)], ComparisonOp::Eq, 0.0),
                ],
                xstar: vec![1000003.0, 0.0, 0.0, 0.0, -2.0],
            };
            check(model());
            assert_exact_optimum(&model(), -2.0);
        }
    }

    /// `522 x0 + 0.0019 x1 = 522001566` with `x0` fixed at `1000003` puts
    /// `x1` exactly at `0`. The dual phase ends with that row's slack basic,
    /// absorbing `x1 = -2.4e-8` (the row moves by `4.7e-11`, within its
    /// tolerance), and the primal phase's last pivot takes `x2` into the
    /// basis on a zero step and the slack out onto its value. The values
    /// stay, `x2 = 1`, while the new basis amplifies the row's residual by
    /// `2e10`: its vertex has `x2 = 0`, the optimum `0` at `x*`. Unrefined,
    /// the answer is the values' point, within the contract and `1.0002`
    /// worse; refined, the vertex's: as an LP and as a MILP.
    #[test]
    fn a_residual_the_rows_accept_is_refined_out_of_the_values() {
        for milp in [false, true] {
            let var = |obj: f64, lo: i32, hi: i32| {
                if milp {
                    int(obj, lo, hi)
                } else {
                    real(obj, lo as f64, hi as f64)
                }
            };
            let model = || Model {
                direction: OptimizationDirection::Maximize,
                vars: vec![
                    var(0.0, 1000003, 1000003),
                    real(-2.0, -2.0, 0.0),
                    real(-1.0, 0.0, 1.0),
                    var(0.0, 0, 1),
                    real(0.0, 0.0, 1.0),
                    real(-1.0, 0.0, 1.0),
                ],
                rows: vec![
                    row(&[(0, 522.0), (1, 0.0019)], ComparisonOp::Eq, 522001566.0),
                    row(
                        &[(1, 539.0), (2, 0.0143), (3, -0.00233), (4, -6.86)],
                        ComparisonOp::Eq,
                        -0.00233,
                    ),
                    row(
                        &[(1, 0.00567), (3, -0.00524), (4, 0.00175), (5, 0.111)],
                        ComparisonOp::Ge,
                        -1.00524,
                    ),
                    row(
                        &[
                            (1, 0.00858),
                            (2, -0.0355),
                            (3, 114.0),
                            (4, -0.402),
                            (5, 160.0),
                        ],
                        ComparisonOp::Eq,
                        114.0,
                    ),
                    row(
                        &[(1, -69.6), (3, -2.78), (5, -0.0075)],
                        ComparisonOp::Eq,
                        -2.78,
                    ),
                ],
                xstar: vec![1000003.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            };
            check(model());
            assert_exact_optimum(&model(), 0.0);
        }
    }

    /// `-10 x0 + 0.01 x1 = 20971520.000000007`, with `x0` held at
    /// `-2097152.000000001` by the first row, puts `x1` at `-1.86e-7`, past
    /// its bound beyond its tolerance; but the row's tolerance (`4.2e-7`, the
    /// round-off floor of its magnitude) also admits `x1 = 0`, which is `x*`.
    /// In plain `f64`, `10 x0` rounds onto the right-hand side, so the vertex
    /// with both vars basic computes `x1 = 0`, by a step or by a solve
    /// through the basis (the cancellation lies inside the solve, beyond a
    /// compensated `b - N x_N`). Refined, `x1` lies past its bound, but the
    /// phases do not leave the vertex for good: a primal zero step or a
    /// rebuild's solve computes `x1 = 0` again, until the rounds run out. A
    /// refinement that turns accepted values into rejected ones keeps the
    /// computed values instead. The rows lie inside the recommended range
    /// (row dynamism `1e3`). In the second model, a dual step through a row
    /// that cancels `2.4e9` against itself (`x0` fixed) computes
    /// `x1 = 0.125000428`, within its bound, where exactly it is
    /// `0.125000683`, `1.8e-7` past it.
    #[test]
    fn a_vertex_whose_exact_values_miss_a_bound_keeps_its_computed_ones() {
        let rounded_onto_the_rhs = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -2097152.000000001, -2097151.000000001),
                real(1.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, 1.0)], ComparisonOp::Eq, -2097152.000000001),
                row(
                    &[(0, -10.0), (1, 0.01)],
                    ComparisonOp::Eq,
                    20971520.000000007,
                ),
            ],
            xstar: vec![-2097152.000000001, 0.0],
        };
        let cancelling = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.0, 121736438.0, 121736438.0),
                real(1.0, -0.8749995719061721, 0.12500050027266646),
            ],
            rows: vec![
                row(
                    &[(0, -19.630917228768432), (1, 1.0)],
                    ComparisonOp::Eq,
                    -2389797937.9780993,
                ),
                row(&[(1, 1.0)], ComparisonOp::Eq, 0.125000428093828),
            ],
            xstar: vec![121736438.0, 0.125000428093828],
        };
        for model in [rounded_onto_the_rhs, cancelling] {
            assert_solves_within_contract_without_presolve(&model);
            check(model);
        }
    }

    /// An ill-conditioned vertex whose computed values the rows and bounds
    /// accept, but whose exact values miss a bound, goes on from its refined
    /// values where they gain the objective beyond the tolerances.
    /// `53.3 x0 + 0.0016 x2 = 3576909.7016` with `x0` fixed at `67109` puts
    /// `x2` at `1 - 7.9e-9`, which the plain sum computes as `1 + 4.7e-8`,
    /// within its tolerance; the active
    /// `-0.0194 x1 - 824 x2 - 0.0011 x3 >= -824.0011` amplifies the difference
    /// `7.5e5` times, so the vertex with `x3` basic in it has `x3 = 0.964` as
    /// computed and `1.0059`, past its bound, exactly. Kept as computed, the
    /// answer would be that vertex's, `-1.4288`, within the contract; the
    /// refined values gain `0.083` on it, and lead to the optimum,
    /// `-1.5000000039580001`, presolve on and off, with idle rows that move
    /// the refactorizations, and with no upper bound on `x2`. The rows lie
    /// inside the recommended range (row dynamism `7.5e5`).
    #[test]
    fn a_vertex_whose_exact_values_miss_a_bound_goes_on_from_its_refined_values() {
        let model = |x2_max: f64| Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, 67109.0, 67109.0),
                real(-2.0, 0.0, 1.0),
                real(0.5, 0.0, x2_max),
                real(-2.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, 53.3), (2, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(
                    &[(1, -0.0194), (2, -824.0), (3, -0.0011)],
                    ComparisonOp::Ge,
                    -824.0011,
                ),
                row(&[(1, 0.0155)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![67109.0, 0.0, 1.0, 1.0],
        };
        for x2_max in [1.0, f64::INFINITY] {
            for idle_rows in [0, 1, 2, 4, 8] {
                let model = with_idle_rows(model(x2_max), idle_rows);
                assert_exact_optimum(&model, -1.5000000039580001);
                assert_solves_within_contract_without_presolve(&model);
                check(model);
            }
        }
    }

    /// Solves `model` with presolve on and off, and asserts each answer within
    /// the contract and its objective no worse than at `x*` beyond `1e-6` of
    /// it.
    fn assert_no_worse_than_xstar(model: &Model) {
        assert_solves_within_contract_without_presolve(model);
        assert_solves_within_contract(model);
        let at_star = model.objective_at(&model.xstar);
        let (problem, _) = model.build();
        for presolve in [true, false] {
            let options = SolveOptions {
                presolve,
                ..SolveOptions::default()
            };
            let objective = problem
                .solve_with(options)
                .unwrap()
                .into_solution()
                .unwrap()
                .objective();
            let excess = match model.direction {
                OptimizationDirection::Minimize => objective - at_star,
                OptimizationDirection::Maximize => at_star - objective,
            };
            assert!(
                excess <= 1e-6 * (1.0 + at_star.abs()),
                "presolve {presolve}: objective {objective:e}, at x* {at_star:e}:\n{}",
                model.source()
            );
        }
    }

    /// Blocks of the model of
    /// `a_vertex_whose_exact_values_miss_a_bound_goes_on_from_its_refined_values`
    /// coupled through `x0`, whose data are infeasible: the model is feasible
    /// only within the contract. At the answer, `x*`, within `5e-7` of the
    /// optimum of the model relaxed by a tenth of the contract,
    /// `-6.8952036`, the refined values put the first row's slack past its
    /// bound and gain `5.8e-9` on the computed ones: `x6`, at a cost of `0.1`,
    /// is `5.5e-8` from its refined value, within its tolerance, and the
    /// computed values stay. Gone on from the refined values, the solve would
    /// end at `-5.9000`, and return to `x*`.
    #[test]
    fn computed_values_within_their_tolerances_of_the_vertex_stay() {
        assert_no_worse_than_xstar(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-3.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-2.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-2.0, 0.0, 1.0),
                real(0.1, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-2.0, 0.0, 1.0),
                real(1.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-1.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(
                    &[(5, -0.0194), (6, -412.0), (0, -0.0022)],
                    ComparisonOp::Le,
                    -412.0022,
                ),
                row(&[(4, 53.3), (6, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(&[(1, 53.3), (3, 0.0016)], ComparisonOp::Le, 3576909.7016),
                row(
                    &[(8, -0.0194), (9, -824.0), (0, -0.0022)],
                    ComparisonOp::Ge,
                    -824.0022,
                ),
                row(
                    &[(10, 53.3), (12, 0.0016), (0, -0.5)],
                    ComparisonOp::Le,
                    3576909.2065999997,
                ),
                row(
                    &[(2, -0.0194), (3, -412.0), (0, -0.0022)],
                    ComparisonOp::Le,
                    -412.0022,
                ),
                row(
                    &[(7, 53.3), (9, 0.0016), (0, -1.0)],
                    ComparisonOp::Eq,
                    3576908.7016,
                ),
                row(
                    &[(11, -0.0194), (12, -824.0), (0, -0.0011)],
                    ComparisonOp::Ge,
                    -824.0011,
                ),
                row(&[(11, 0.0155)], ComparisonOp::Eq, 0.0),
                row(&[(5, 0.0155)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![
                0.9983999999240041,
                67109.0,
                1.0,
                0.9999529211650491,
                67109.0,
                0.0,
                1.0000000474974513,
                67109.0,
                1.0,
                0.0,
                67109.0,
                0.0,
                0.0,
            ],
        });
    }

    /// Coupled blocks as in
    /// `computed_values_within_their_tolerances_of_the_vertex_stay`, data
    /// infeasible: at the answer, `x*`, the refined values miss a bound and
    /// lose `0.011` on the computed ones, which stay. Gone on from the
    /// refined values, the solve would end at `-1.59200`, and return to `x*`,
    /// at `-1.59876`.
    #[test]
    fn computed_values_better_than_the_vertex_stay() {
        assert_no_worse_than_xstar(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-3.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(0.0, 0.0, 1.0),
                real(-0.1, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(0.0, 0.0, 1.0),
                real(0.5, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-1.0, 0.0, 1.0),
                real(1.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-1.0, 0.0, 1.0),
                real(-0.1, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(1, 53.3), (3, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(&[(8, 0.0155)], ComparisonOp::Eq, 0.0),
                row(
                    &[(8, -0.0194), (9, -412.0), (0, -0.0011)],
                    ComparisonOp::Ge,
                    -412.0011,
                ),
                row(
                    &[(2, -0.0194), (3, -1648.0), (0, -0.00055)],
                    ComparisonOp::Ge,
                    -1648.00055,
                ),
                row(
                    &[(7, 53.3), (9, 0.0016), (0, 1.0)],
                    ComparisonOp::Eq,
                    3576910.6656,
                ),
                row(
                    &[(11, -0.0194), (12, -1648.0), (0, -0.0022)],
                    ComparisonOp::Ge,
                    -1648.0022,
                ),
                row(
                    &[(4, 53.3), (6, 0.0016), (0, 1.0)],
                    ComparisonOp::Ge,
                    3576910.6656,
                ),
                row(&[(2, 0.0155)], ComparisonOp::Eq, 0.0),
                row(
                    &[(5, -0.0194), (6, -412.0), (0, -0.0022)],
                    ComparisonOp::Ge,
                    -412.0022,
                ),
                row(
                    &[(10, 53.3), (12, 0.0016), (0, -0.01)],
                    ComparisonOp::Eq,
                    3576909.69196,
                ),
            ],
            xstar: vec![
                0.9640000002287328,
                67109.0,
                0.0,
                1.0,
                67109.0,
                0.0,
                1.0,
                67109.0,
                0.0,
                1.0,
                67109.0,
                0.006761683774244132,
                0.999999968460761,
            ],
        });
    }

    /// Coupled blocks as in
    /// `computed_values_within_their_tolerances_of_the_vertex_stay`, data
    /// infeasible: the dual phase ends at a vertex whose reduced costs the
    /// dual check rejects, and whose refined values miss a bound and gain
    /// `0.021` on the computed ones. The refined values are weighed only at a
    /// vertex the dual check accepts: the phases go on from the computed
    /// values, and the answer is `x*`, at `-0.98999`. Gone on from the refined
    /// values at that vertex, the excursion would end at `0.0089`, and the
    /// solve would return to the vertex, at `-0.98221`, which is no answer.
    #[test]
    fn a_vertex_the_phases_leave_keeps_its_computed_values() {
        assert_no_worse_than_xstar(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-2.0, 0.0, 1.0),
                real(1.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-1.0, 0.0, 1.0),
                real(-0.1, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-1.0, 0.0, 1.0),
                real(0.1, 0.0, 1.0),
            ],
            rows: vec![
                row(
                    &[(8, -0.0194), (9, -824.0), (0, -0.0022)],
                    ComparisonOp::Le,
                    -824.0022,
                ),
                row(&[(1, 53.3), (3, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(&[(8, 0.0155)], ComparisonOp::Eq, 0.0),
                row(&[(2, 0.0155)], ComparisonOp::Eq, 0.0),
                row(&[(7, 53.3), (9, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(
                    &[(4, 53.3), (6, 0.0016), (0, -1.0)],
                    ComparisonOp::Ge,
                    3576908.7115999996,
                ),
                row(
                    &[(5, -0.0194), (6, -412.0), (0, -0.0022)],
                    ComparisonOp::Ge,
                    -412.0022,
                ),
                row(
                    &[(2, -0.0194), (3, -412.0), (0, -0.00055)],
                    ComparisonOp::Ge,
                    -412.00055,
                ),
            ],
            xstar: vec![
                0.9899999245419873,
                67109.0,
                0.0,
                1.0000000133496152,
                67109.0,
                1.0,
                0.9999529660198204,
                67109.0,
                0.0,
                1.0000000474974513,
            ],
        });
    }

    /// Coupled blocks as in
    /// `computed_values_within_their_tolerances_of_the_vertex_stay`, whose
    /// exact optimum is `-2.0999999992084`: at the vertex the dual check
    /// accepts first, the computed values put `x0` at `0.964`, the refined
    /// ones past its upper bound, and the refined values gain `0.083` on the
    /// computed ones: they go on, to `x*`, at `-2.1`, where kept as computed
    /// the answer would be `-2.0288`.
    #[test]
    fn coupled_refined_values_that_gain_beyond_the_tolerances_go_on() {
        assert_no_worse_than_xstar(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-2.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(0.0, 0.0, 1.0),
                real(-0.1, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-1.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(5, 0.0155)], ComparisonOp::Eq, 0.0),
                row(
                    &[(5, -0.0194), (6, -824.0), (0, -0.0011)],
                    ComparisonOp::Ge,
                    -824.0011,
                ),
                row(&[(1, 53.3), (3, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(
                    &[(2, -0.0194), (3, -824.0), (0, -0.0011)],
                    ComparisonOp::Ge,
                    -824.0011,
                ),
                row(&[(4, 53.3), (6, 0.0016)], ComparisonOp::Eq, 3576909.7016),
            ],
            xstar: vec![1.0, 67109.0, 0.0, 1.0, 67109.0, 0.0, 0.9999999920839997],
        });
    }

    /// Coupled blocks as in
    /// `computed_values_within_their_tolerances_of_the_vertex_stay`, data
    /// infeasible: at the answer, `x*`, at `-0.8644`, the refined values gain
    /// `0.042` on the computed ones and go on; the excursion ends at `0.110`,
    /// worse, and the solve returns to `x*`.
    #[test]
    fn an_excursion_on_refined_values_that_ends_worse_returns_to_the_answer() {
        assert_no_worse_than_xstar(&Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(-1.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-2.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-1.0, 0.0, 1.0),
                real(0.0, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(0.0, 0.0, 1.0),
                real(0.1, 0.0, 1.0),
                real(0.0, 67109.0, 67109.0),
                real(-2.0, 0.0, 1.0),
                real(1.0, 0.0, 1.0),
            ],
            rows: vec![
                row(
                    &[(5, -0.0194), (6, -824.0), (0, -0.0011)],
                    ComparisonOp::Le,
                    -824.0011,
                ),
                row(&[(1, 53.3), (3, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(
                    &[(10, 53.3), (12, 0.0016), (0, -0.5)],
                    ComparisonOp::Ge,
                    3576909.2065999997,
                ),
                row(&[(7, 53.3), (9, 0.0016)], ComparisonOp::Eq, 3576909.7016),
                row(&[(2, 0.0155)], ComparisonOp::Eq, 0.0),
                row(&[(8, 0.0155)], ComparisonOp::Eq, 0.0),
                row(
                    &[(2, -0.0194), (3, -824.0), (0, -0.00055)],
                    ComparisonOp::Ge,
                    -824.00055,
                ),
                row(
                    &[(8, -0.0194), (9, -824.0), (0, -0.0011)],
                    ComparisonOp::Ge,
                    -824.0011,
                ),
                row(&[(5, 0.0155)], ComparisonOp::Eq, 0.0),
                row(
                    &[(11, -0.0194), (12, -412.0), (0, -0.00055)],
                    ComparisonOp::Ge,
                    -412.00055,
                ),
                row(&[(11, 0.0155)], ComparisonOp::Eq, 0.0),
                row(&[(4, 53.3), (6, 0.0016)], ComparisonOp::Eq, 3576909.7016),
            ],
            xstar: vec![
                0.9644200910051128,
                67109.0,
                0.0,
                1.0000000237487257,
                67109.0,
                0.0,
                1.0000000474974513,
                67109.0,
                0.0,
                1.0000000474974513,
                67109.0,
                0.0,
                0.0,
            ],
        });
    }

    /// `1.89 x0 + 0.00683 x2 = 188999979.21` with `x0` fixed at `99999989`
    /// cancels to the last bit, and pins `x2` within `5.5e-4` of `2.7e-6`
    /// through the row's round-off floor `3.8e-6`, which the answer uses.
    /// Edited at the answer's values (a var fixed at its value, or a row
    /// bounding it there), the model solves, presolve on and off. A recompute
    /// of the basic values summed `b - N x_N` in `f64`, which kept `1.2e-8` of
    /// the first row's round-off; through the basis that put `x2` `1.7e-6`
    /// off, and the fourth row's slack `3.2e-3` off its exact value where
    /// the row allows `1e-7`: the dual simplex repaired a violation of the
    /// wrong sign, and no var could ("no var can repair row"). In the
    /// recommended range (dynamism at most `5.5e4`).
    #[test]
    fn a_cancelling_row_leaves_no_round_off_in_the_recomputed_values() {
        let model = |fixed: Option<(usize, f64)>, bounded: Option<(usize, f64)>| {
            let mut vars = vec![
                real(-1.0, 99999989.0, 99999989.0),
                real(-2.0, 0.0, 1.0),
                real(-1.0, -2.0, 0.0),
                real(0.0, -2.0, 0.0),
                real(1.0, -2.0, 0.0),
                real(0.5, -2.0, 0.0),
            ];
            let mut rows = vec![
                row(&[(0, 1.89), (2, 0.00683)], ComparisonOp::Eq, 188999979.21),
                row(&[(2, 1.44)], ComparisonOp::Le, 0.0),
                row(
                    &[
                        (1, -0.0214),
                        (2, 5.69),
                        (3, -0.796),
                        (4, -69.2),
                        (5, -0.00562),
                    ],
                    ComparisonOp::Eq,
                    1.5706,
                ),
                row(
                    &[(2, 318.0), (3, 218.0), (4, 0.00581)],
                    ComparisonOp::Eq,
                    -436.0,
                ),
                row(
                    &[(1, -29.8), (4, -0.0309), (5, 14.2)],
                    ComparisonOp::Eq,
                    -29.8,
                ),
            ];
            if let Some((var, value)) = fixed {
                (vars[var].lo, vars[var].hi) = (value, value);
            }
            if let Some((var, value)) = bounded {
                rows.push(row(&[(var, 1.0)], ComparisonOp::Le, value));
            }
            Model {
                direction: OptimizationDirection::Maximize,
                vars,
                rows,
                xstar: vec![
                    99999989.0,
                    0.8964647784891895,
                    -0.0005016330258672402,
                    -1.999268260081533,
                    0.0,
                    -0.21727814091707387,
                ],
            }
        };
        let (x1, x2, x5) = (
            0.8964647784891895,
            -0.0005016330258672402,
            -0.21727814091707387,
        );
        for (fixed, bounded) in [
            (Some((1, x1)), None),
            (Some((5, x5)), None),
            (None, Some((1, x1))),
            (None, Some((2, x2))),
            (None, Some((5, x5))),
        ] {
            let model = model(fixed, bounded);
            assert!(xstar_is_feasible(&model), "fixture: the answer's point");
            assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        }
    }

    /// `2^53 x0 + 0.75 (x1 + ... + x300) - 2^53 x301 >= 225` with `x0` and
    /// `x301` fixed at `1` forces every other var to its upper bound `1`. A
    /// plain sum of the row in its own order drops each `0.75` against `2^53`
    /// and finds the optimum `225` short, beyond its contract of `180`: a
    /// sum of 302 terms errs by more than the round-off floor of its
    /// magnitude. The engine sums such a row in compensated arithmetic, as the
    /// boundary sums every row, and the solve reaches the optimum.
    #[test]
    fn a_row_whose_plain_sum_drops_its_small_terms_is_held_exactly() {
        let n = 300;
        let model = || {
            let mut vars = vec![real(0.0, 1.0, 1.0)];
            vars.extend((0..n).map(|_| real(1.0, 0.0, 1.0)));
            vars.push(real(0.0, 1.0, 1.0));
            let mut terms = vec![(0, 2f64.powi(53))];
            terms.extend((1..=n).map(|k| (k, 0.75)));
            terms.push((n + 1, -(2f64.powi(53))));
            Model {
                direction: OptimizationDirection::Minimize,
                vars,
                rows: vec![row(&terms, ComparisonOp::Ge, 225.0)],
                xstar: vec![1.0; n + 2],
            }
        };
        check(model());
        assert_solves_within_contract_without_presolve(&model());
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

    /// A vertex the dual simplex left on drifted values is no cycle when it
    /// returns. The root LP starts `x2` at its lower bound, where the second
    /// row's terms reach `1.4e11`; its first step flips `x2` to 0, and the
    /// values the steps carry on keep the round-off of those terms: two
    /// pivots later they put `x4` `9.5e-6` past its upper bound, beyond its
    /// tolerance `4.2e-6`, where exactly it lies within, and the phase leaves
    /// that vertex. The drift check after the next pivot recomputes the
    /// values; the way back to the vertex followed a revisit, which made the
    /// change back taboo for good ("dual simplex cycles"), presolve on and
    /// off. The model is solved again with eight rows the optimum does not
    /// involve, which put the refactorizations and the drift checks elsewhere
    /// on the path: there, values recomputed at every refactorization without
    /// a restart of the cycling detection made the return a cycle. Within the
    /// recommended range (row dynamism `6.7e5`).
    #[test]
    fn a_vertex_left_on_drifted_values_can_be_returned_to() {
        let model = || Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                int(-1.0, 0, 1),
                real(1.0, -651624630.0, -651623443.0),
                real(0.0, -136593742.0, 0.0),
                real(-1.0, -1.0, 0.0),
                real(-1.0, -422117195.0000001, -422117194.0000001),
            ],
            rows: vec![
                row(
                    &[(0, 0.01), (1, -1.0), (4, -1.0)],
                    ComparisonOp::Eq,
                    1073741824.0,
                ),
                row(
                    &[(1, 0.0015), (2, 1000.0), (3, 0.1), (4, 2.0)],
                    ComparisonOp::Eq,
                    -845211824.9450003,
                ),
                row(&[(3, 1.0), (4, -0.01)], ComparisonOp::Ge, 4221171.940000001),
            ],
            xstar: vec![0.0, -651624630.0, 0.0, 0.0, -422117194.0000001],
        };
        for idle_rows in [0, 8] {
            let model = with_idle_rows(model(), idle_rows);
            assert!(
                xstar_is_feasible(&model),
                "fixture error: x* must satisfy the model"
            );
            assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        }
    }

    /// A refactorization does not recompute basic values the rows accept: a
    /// recompute can move a var on its bound past it by a round-off amount
    /// the rows allow but its own tolerance does not.
    /// `-1.9979395087316894 x0 - 0.1 x2 = 268435456.33020216`, with `x0` fixed
    /// at `-134356148` by the first row, holds at `x2 = 0` within its
    /// tolerance, but a solve from the rows puts `x2` at `-1.6e-7`, past its
    /// tolerance `1e-7` at its lower bound: the round-off of terms at `2.7e8`,
    /// through the coefficient `0.1`. The primal step that makes `x2` basic
    /// leaves it exactly on that bound, and the basis of two rows is
    /// refactorized right after it. Recomputed there, `x2` went past its
    /// bound, the dual simplex took it back out, and the phases alternated
    /// over the vertex until the rounds ran out ("8 phase rounds"), presolve
    /// on and off. Within the recommended range (row dynamism `20`).
    #[test]
    fn a_refactorization_keeps_the_values_the_rows_accept() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, -134356148.0, -134356147.0),
                real(1.0, 0.0, 0.0),
                real(-1.0, 0.0, 1.0),
            ],
            rows: vec![
                row(&[(0, 1.0)], ComparisonOp::Eq, -134356148.0),
                row(
                    &[(0, -1.9979395087316894), (2, -0.1)],
                    ComparisonOp::Eq,
                    268435456.33020216,
                ),
            ],
            xstar: vec![-134356148.0, 0.0, 0.0],
        };
        assert!(
            within_recommended_range(&model),
            "fixture error: the model must lie within the recommended range"
        );
        assert_solves_within_contract_without_presolve(&model);
        check(model);
    }

    /// The node `x0 = 2` holds only with two terms past their bounds at once,
    /// each within its tolerance: `x1` past its upper bound and the second
    /// equality's slack past zero, a point no single absorption reaches. One
    /// equality's slack absorbs the violation; the other, which crossed its
    /// bound once in the solve, does not absorb it back, so the slacks cannot
    /// hand it back and forth until every row is taboo. The proof grants both
    /// slacks their room and proves nothing, and the widening moves the other
    /// slack's bound: the node solves, presolve on and off. The rows lie
    /// within the recommended range.
    #[test]
    fn a_node_feasible_only_through_two_tolerances_at_once_is_solved() {
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

    /// A cost tiny beside its column's coefficients still decides the optimum:
    /// every reduced cost is held to its tolerance in the user's units as in
    /// the engine's, so a column factor of `2^-30`, which takes a cost of
    /// `0.1` below `1e-10`, does not hide it. Each model is `x <= 1` written
    /// with a large coefficient (every row and column within the recommended
    /// range), maximizing `cost · x`, with `x` nonnegative or free.
    #[test]
    fn a_cost_tiny_beside_its_columns_coefficients_decides_the_optimum() {
        for (cost, coeff, lo) in [
            (0.1, 1e9, 0.0),
            (1.0, 2e10, 0.0),
            (0.1, 1e9, f64::NEG_INFINITY),
        ] {
            for presolve in [true, false] {
                let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_var(cost, (lo, f64::INFINITY));
                problem.add_constraint([(x, coeff)], ComparisonOp::Le, coeff);
                let options = SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                };
                let solution = problem
                    .solve_with(options)
                    .unwrap()
                    .into_solution()
                    .unwrap();
                assert!(
                    (solution[x] - 1.0).abs() <= 1e-9
                        && (solution.objective() - cost).abs() <= 1e-9,
                    "max {cost} x s.t. {coeff:e} x <= {coeff:e}, x >= {lo} (presolve {presolve}): \
                     x = {}, objective {}",
                    solution[x],
                    solution.objective()
                );
            }
        }
    }

    /// A slack's reduced cost is held per unit of its row's activity divided by
    /// the row's largest coefficient as written, whatever factor the row is
    /// written at. `max x` s.t. `2e10 x >= 2e10`, `x >= 0`, is unbounded, its
    /// ray through the row's surplus improving the objective by `5e-11` per
    /// unit of the surplus as written, one per unit of `x`; with `x <= 3` the
    /// edge through the surplus ends at `x = 3`. Presolve on and off.
    #[test]
    fn an_improving_ray_through_a_row_written_large_is_found() {
        for (max, expected) in [(f64::INFINITY, None), (3.0, Some(3.0))] {
            for presolve in [true, false] {
                let mut problem = crate::Problem::new(OptimizationDirection::Maximize);
                let x = problem.add_var(1.0, (0.0, max));
                problem.add_constraint([(x, 2e10)], ComparisonOp::Ge, 2e10);
                let options = SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                };
                let result = problem.solve_with(options);
                let label = format!("x <= {max}, presolve {presolve}");
                match expected {
                    None => assert!(
                        matches!(result, Err(Error::Unbounded)),
                        "{label}: the unbounded LP is answered {result:?}"
                    ),
                    Some(value) => {
                        let solution = result.unwrap().into_solution().unwrap();
                        assert!(
                            (solution[x] - value).abs() <= 1e-9,
                            "{label}: x = {}",
                            solution[x]
                        );
                    }
                }
            }
        }
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

    /// A model of the differential fuzz outside the recommended range
    /// (constraint 1 spans `7.9e10`, variable 6 `1.3e12`) solves within the
    /// contract, with presolve on and off; the answer is the objective at `x*`.
    /// Its only way to the optimum pivots on entries some `2.2e-10` of their
    /// scaled rows and columns, below every pivot tolerance: each primal phase
    /// leaves a row violated that such an entry alone repairs, and the dual
    /// simplex's first last resort takes it, genuine on the refined row, three
    /// phase rounds in all. Earlier, a step that flipped a boxed var through
    /// such an entry left it dual infeasible, and a rebuild that moved it back,
    /// rather than shifting its cost (`rebuild_dual`), returned the dual phase
    /// to a vertex it had left, round a loop the cycling detection ended.
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
    /// the row as written), so the dual ratio test sees `[1e17, 1]`. The
    /// growth test measures a pivot against the entries of the vars whose
    /// bounds do not fix them, so `y`'s exact one is a regular pivot. Taken
    /// for round-off, it would leave the row no var to repair it and a false
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
    /// `f` out of its equilibration, and so does the growth test, which looks
    /// at the bounds in force, not at those the solve began with: `y`'s exact
    /// one is a regular pivot, and the answer is `y = 1`.
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
    /// `big = 1e30`, `y`'s entry far below the pivot tolerance with no bound to
    /// flip to.
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

    /// A row only entries far below the pivot tolerance can repair: the origin is
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

    /// The primal step toward the better point, `x3 = -1`, `x4 = 1000`,
    /// passes the equality `1e4 x2 - 1e-7 x3 = 0` through an entry at or below
    /// the pivot tolerance and leaves it beyond its tolerance
    /// (`a_step_past_a_row_the_ratio_test_ignores_shifts_its_bound`), and the
    /// phases can alternate over it. The first dual phase ends at that point;
    /// refined, its values miss the equality, and the cleanup dual phase takes
    /// `x3` back to the origin with no round-off left for a slack to absorb,
    /// which would bring the step back: the solve ends at the exact optimum,
    /// `0`.
    #[test]
    fn an_equality_a_long_step_passes_below_its_floor_settles() {
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

    /// The phases do not alternate over a basic var that only an entry below
    /// the pivot tolerance can repair. `-1e-8 x0 - 1e4 x1 = 0` puts `x1` at
    /// `-1e-12 x0`, which `x1 <= 0` holds within its tolerance, `1e-7`, as far
    /// as `x0 = -1e5`. The primal step flips `x2` to its lower bound, which
    /// carries `x0` to `-100010.01` (`100 x0 - 1e-5 x1 - x2 = 0`) and `x1` to
    /// `1.0001e-7`, past its tolerance, through such an entry. In the dual
    /// phase only `x2` can repair `x1`'s row, through that entry: the proof
    /// proves nothing, and the certificate repair flipped `x2` back, for the
    /// step to come again, until the solve failed with `8 phase rounds`. The
    /// entry is genuine on the refined row, and the dual simplex's first last
    /// resort pivots on it. The rows lie outside the recommended range (row
    /// dynamism `1e12`).
    #[test]
    fn phases_over_a_var_a_bound_flip_passes_settle() {
        check(Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![
                real(-1.0, -100011.0, 0.0),
                real(-1.0, -1.0, 0.0),
                real(-1.0, -10001001.0, 0.0),
            ],
            rows: vec![
                row(&[(0, 100.0), (1, -1e-5), (2, -1.0)], ComparisonOp::Eq, 0.0),
                row(&[(1, -1.0), (2, -1000.0)], ComparisonOp::Ge, 0.0),
                row(&[(0, -1e-8), (1, -10000.0)], ComparisonOp::Eq, 0.0),
            ],
            xstar: vec![0.0; 3],
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
    /// through an entry of `3.2e-16`, far below the pivot tolerance.
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
    /// (ARCHITECTURE §7.1).
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

    /// The node `x1 = 1` holds only within its rows' tolerances: in the node's
    /// basis `x1` is basic, and the three equalities that hold it, taken as
    /// exact, keep it at least `7e-5` from 1. The proof grants every term of
    /// their certificate its room, so it proves nothing, the widening moves
    /// the bounds of the slacks that close the gap, and the search reaches
    /// `x*`'s optimum.
    #[test]
    fn a_node_feasible_only_within_its_rows_tolerances_is_solved() {
        check(integer_value_met_within_the_rows_tolerances());
    }

    /// The test above without presolve.
    #[test]
    fn a_node_feasible_only_within_its_rows_tolerances_is_solved_without_presolve() {
        assert_solves_within_contract_without_presolve(
            &integer_value_met_within_the_rows_tolerances(),
        );
    }

    /// An edit holds a var to its model bounds after a solve widened one.
    /// `x + y >= 2 + 2.5e-7` over `x, y ∈ [0, 1]` needs room past both upper
    /// bounds: the widening moves `y`'s to `1 + 1e-7`, and the answer puts `y`
    /// there. Fixed at `0.5`, either var leaves the other at most `1 + 1e-7`,
    /// so each edit is `Infeasible`; a fix that kept the widening of `y`'s
    /// bound would solve at `y = 1 + 1e-7`.
    #[test]
    fn a_var_fixed_after_its_bound_was_widened_keeps_its_model_bounds() {
        for presolve in [true, false] {
            let mut problem = crate::Problem::new(OptimizationDirection::Minimize);
            let x = problem.add_var(1.0, (0.0, 1.0));
            let y = problem.add_var(0.5, (0.0, 1.0));
            problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Ge, 2.0 + 2.5e-7);
            let options = SolveOptions {
                presolve,
                ..SolveOptions::default()
            };
            let solution = problem
                .solve_with(options)
                .unwrap()
                .into_solution()
                .unwrap();
            assert!(
                solution[y] > 1.0,
                "presolve {presolve}: fixture: y = {} lies on its widened bound",
                solution[y]
            );
            for var in [x, y] {
                assert_eq!(
                    solution.clone().fix_var(var, 0.5).map(|_| ()),
                    Err(Error::Infeasible),
                    "presolve {presolve}: fixing {var:?} at 0.5"
                );
            }
        }
    }

    /// `2 x0 - x1 = 0`, `x1 - x2 = 0` and `x2 = 2 + 1.8e-7` over a binary
    /// `x0`, with `-878 x0 >= -878`: `x0 = 1` needs `x1 = x2 = 2 + 9e-8`, the
    /// first and third rows each `9e-8` off, within their tolerances together,
    /// and `x0 = 0` meets nothing. Presolve rewrites the last row and drops
    /// it; the root's LP point has `x0 = 1 + 9e-8`, within its bound's
    /// tolerance, which the row as written rejects, and the branch fixes `x0`
    /// at 1, where the engine holds it exactly. No single row can take its
    /// offset, so no vertex of the node's bounds meets the rows; the proof
    /// grants both equalities their room and proves nothing, and the widening
    /// moves both slacks' bounds, where the node's LP finds the point. Without
    /// presolve the root needs the same, on its equalities.
    #[test]
    fn a_node_fixing_a_binary_feasible_only_within_two_rows_tolerances_is_solved() {
        assert_solves_within_contract_in_time(
            &Model {
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
            },
            Duration::from_secs(60),
        );
    }

    /// `-1125 x0 - 1.125e-7 x1 = -5876999.999999887` with `x0` fixed at `5224`
    /// holds `x1 = -1` and, `1.02e-11` inside the engine's room of the row (its
    /// contract less the round-off margin), `x1 = -2`. The LP's vertex,
    /// `x1 = -1.0017`, branches, and the node `x1 = -2` fixes both vars. The
    /// row's plain sum, whose round-off there is `3.8e-10`, put that point
    /// `3.7e-10` beyond the room: no var could move, the proof, summed
    /// exactly, proved nothing, and the solve failed loudly. Such a rejection
    /// is summed again, compensated, which finds the point within the room.
    /// The row lies outside the recommended range (row dynamism `1e10`).
    #[test]
    fn a_node_point_within_the_round_off_of_its_rows_room_is_solved() {
        let model = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![int(0.0, 5224, 5224), int(1.0, -2, -1)],
            rows: vec![row(
                &[(0, -1125.0), (1, -1.125e-7)],
                ComparisonOp::Eq,
                -5876999.999999887,
            )],
            xstar: vec![5224.0, -2.0],
        };
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
    }

    /// Nodes like the one above within the recommended range (row dynamism
    /// `2e4` to `1e5`): `a0 x0 + a1 x1 = b` with `x0` fixed at an integer
    /// between `4e8` and `2e9`, and `x1`'s better integer inside the row's
    /// room by `8e-9` to `2.2e-7`, the plain sum putting it beyond by `6e-9`
    /// to `8.5e-8`. Each solve failed loudly, presolve on and off.
    #[test]
    fn in_range_node_points_within_the_round_off_of_their_rows_room_are_solved() {
        let cases: [(i32, f64, f64, i32, f64); 6] = [
            (1711692785, 1.281, -5.415e-05, -5, 2192678457.585229),
            (1993684016, -1.489, 7.377e-05, 3, -2968595499.8237224),
            (469151876, -1.227, -1.451e-05, -5, -575649351.8519385),
            (1904183514, 1.179, -5.23e-05, -4, 2245032363.0061665),
            (436447346, -0.9354, 9.084e-06, 3, -408252847.448365),
            (1489342251, -1.49, -4.678e-05, -1, -2219119953.9899955),
        ];
        for (k, a0, a1, x1, b) in cases {
            let model = Model {
                direction: OptimizationDirection::Minimize,
                vars: vec![int(0.0, k, k), int(1.0, x1, x1 + 1)],
                rows: vec![row(&[(0, a0), (1, a1)], ComparisonOp::Eq, b)],
                xstar: vec![k as f64, x1 as f64],
            };
            assert!(
                within_recommended_range(&model),
                "fixture error: the model must lie within the recommended range"
            );
            assert_solves_within_contract(&model);
            assert_solves_within_contract_without_presolve(&model);
        }
    }

    /// The row above with `a1` an ulp or two steeper, over `x0` and `x1` fixed
    /// at `5224` and `-2` as continuous vars, without presolve: the model's
    /// one point has its exact slack `1.3e-23` to `6.6e-23` beyond the row's
    /// room. The compensated row check finds it beyond, and the proof, summed
    /// exactly, found no gap beyond the room: the solve failed loudly. The
    /// proof withholds from the room of the slack the round-off within which
    /// the two verdicts can disagree, and answers `Infeasible`, as for any
    /// point needing more than the room (ARCHITECTURE §7.1).
    #[test]
    fn a_point_round_off_beyond_its_rows_room_is_proven_infeasible() {
        for a1 in [
            -1.1250510347794818e-7,
            -1.1250510347794819e-7,
            -1.125051034779482e-7,
        ] {
            let (problem, _) = Model {
                direction: OptimizationDirection::Minimize,
                vars: vec![real(0.0, 5224.0, 5224.0), real(1.0, -2.0, -2.0)],
                rows: vec![row(
                    &[(0, -1125.0), (1, a1)],
                    ComparisonOp::Eq,
                    -5876999.999999887,
                )],
                xstar: vec![5224.0, -2.0],
            }
            .build();
            let options = SolveOptions {
                presolve: false,
                ..SolveOptions::default()
            };
            assert_eq!(
                problem.solve_with(options).map(|_| ()),
                Err(Error::Infeasible),
                "a1 = {a1:e}"
            );
        }
    }

    /// `x0 + x1 = 348952339.66499335` over `x0 ∈ [0, 1e9]` and `x1 ∈ [0, 1]`
    /// at cost one, solved, then edited: `x0` fixed at `348952339`, then `x1`
    /// at `0.665`. That point meets the row `1.5e-8` inside the engine's room,
    /// which the plain sum puts it `6.6e-9` beyond: no var could move, the
    /// proof proved nothing, and the second edit failed loudly, presolve on
    /// and off (an edit does not presolve). Within the recommended range (row
    /// dynamism one).
    #[test]
    fn an_lp_edit_fixing_a_point_within_the_round_off_of_its_rows_room_solves() {
        let model = |x0: (f64, f64), x1: (f64, f64)| Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![real(0.0, x0.0, x0.1), real(1.0, x1.0, x1.1)],
            rows: vec![row(
                &[(0, 1.0), (1, 1.0)],
                ComparisonOp::Eq,
                348952339.66499335,
            )],
            xstar: vec![348952339.0, 0.665],
        };
        let base = model((0.0, 1e9), (0.0, 1.0));
        assert!(
            within_recommended_range(&base) && xstar_is_feasible(&base),
            "fixture error: x* must satisfy the model, which lies within the recommended range"
        );
        let (problem, vars) = base.build();
        let fixed = model((348952339.0, 348952339.0), (0.665, 0.665));
        for presolve in [true, false] {
            let solution = problem
                .solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                })
                .unwrap()
                .into_solution()
                .unwrap()
                .fix_var(vars[0], 348952339.0)
                .unwrap()
                .into_solution()
                .unwrap();
            assert!(
                infeasible_or_answer_within_contract(
                    &fixed,
                    solution.fix_var(vars[1], 0.665),
                    &vars,
                    "fix_var(x1, 0.665)",
                )
                .is_some(),
                "presolve {presolve}: fix_var(x1, 0.665) returned Infeasible where x* lies \
                 within the engine's room"
            );
        }
    }

    /// `x0 + x1 + x2 >= 348952339.66500664` with `x0` and `x1` fixed at
    /// `348952339` and `0.665` and `x2 ∈ [0, 1e-5]` at cost `-1`, solved, then
    /// `x2` fixed at `4.594137084480981e-8`. The row's plain sum accepts that
    /// point, `3.5e-8` inside the row's room, its exact slack `2e-21` beyond:
    /// a plain acceptance stands, the evaluation margin covering it, and the
    /// edit answers the point, within the contract. Summed again,
    /// compensated, the row would be found violated, and the proof would
    /// answer `Infeasible`.
    #[test]
    fn an_lp_edit_fixing_a_point_its_rows_plain_sum_accepts_solves() {
        let model = |x2: (f64, f64)| Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                real(0.0, 348952339.0, 348952339.0),
                real(0.0, 0.665, 0.665),
                real(-1.0, x2.0, x2.1),
            ],
            rows: vec![row(
                &[(0, 1.0), (1, 1.0), (2, 1.0)],
                ComparisonOp::Ge,
                348952339.66500664,
            )],
            xstar: vec![348952339.0, 0.665, 1e-5],
        };
        let base = model((0.0, 1e-5));
        assert!(
            within_recommended_range(&base) && xstar_is_feasible(&base),
            "fixture error: x* must satisfy the model, which lies within the recommended range"
        );
        let (problem, vars) = base.build();
        let w = 4.594137084480981e-8;
        let fixed = model((w, w));
        for presolve in [true, false] {
            let solution = problem
                .solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                })
                .unwrap()
                .into_solution()
                .unwrap();
            assert!(
                infeasible_or_answer_within_contract(
                    &fixed,
                    solution.fix_var(vars[2], w),
                    &vars,
                    "fix_var(x2, w)",
                )
                .is_some(),
                "presolve {presolve}: fix_var(x2, w) returned Infeasible where the row's plain \
                 sum accepts the point"
            );
        }
    }

    /// `x0 + 1e-5 x2 = 262144.00001`, with `x0` fixed at `2^18`, holds exactly
    /// only at `x2 = 1 + 1.8e-6`, past `x2 <= 1`, since its right-hand side is
    /// not representable, and `x*` meets it at `x2 = 1`, `1.8e-11` off, within
    /// its tolerance. The dual phase hands that miss to the equality's slack
    /// (absorption), and a later pivot takes the slack out of the basis. When
    /// the miss comes back, the slack, which crossed its bound once in the
    /// solve, does not absorb it again: the proof grants it its room and
    /// proves nothing, and the widening moves its bound.
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
    /// equality's slack absorbs a move and a regular pivot takes it out of the
    /// basis; the move it needs again goes to the proof, which grants the
    /// slack its room, and the widening moves its bound.
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

    /// A MILP node within its terms' tolerances is solved, not pruned. The
    /// rows allow `x1 >= -9.99` at `x4 = 0`, and the node `x1 <= -10` holds
    /// with `x4` `1e-9` past its bound, within its tolerance: the optimum is
    /// `x1 = -10`, at `-9.999900099`, presolve on and off.
    #[test]
    fn a_node_within_its_terms_tolerances_is_solved() {
        let model = Model {
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
            xstar: vec![0.0, -10.0, 0.0, -9.99e-5, 1e-9, 0.0],
        };
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model"
        );
        assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        let (problem, vars) = model.build();
        for presolve in [true, false] {
            let options = SolveOptions {
                presolve,
                ..SolveOptions::default()
            };
            let int_tol = options.int_tol;
            let solution = problem
                .solve_with(options)
                .unwrap()
                .into_solution()
                .unwrap();
            assert!(
                (solution[vars[1]] + 10.0).abs() <= int_tol,
                "presolve {presolve}: x1 = {}",
                solution[vars[1]]
            );
            assert!(
                (solution.objective() + 9.999900099).abs() < 1e-9,
                "presolve {presolve}: objective {}",
                solution.objective()
            );
        }
    }

    /// A MILP node proven infeasible through an ill-conditioned basis. The
    /// rows allow `x0 >= -0.898` (at `x2 = 0`, its best), so the node
    /// `x0 <= -1` misses them by `0.10` of `x0`, beyond the `0.022` its
    /// terms' tolerances make room for. The unrefined row of its basis
    /// inverse leaves the basic `x1 ∈ [-1e9, 0]` a certificate coefficient of
    /// `4.8e-14` of its terms, which at `x1`'s scaled bound of `-3.2e10`
    /// takes `0.165` off the gap, and the proof fails; refined, the row
    /// leaves `x1` none, the node is proven, and the optimum is `0`. The rows
    /// lie inside the recommended range (row dynamism `6.6e5` and `2.2e5`).
    /// The second model's rows lie outside it (`2.1e7` and `1.8e7`): they
    /// allow `x0 >= -0.982` (at `x2 = 0`; `x2 = -1` needs `x0 >= 1.8e7`), the
    /// node misses them by `0.018` of `x0` against `0.010` of room, and the
    /// unrefined row leaves `x1 ∈ [-9439526, 0]` `2e-14` of its terms, which
    /// at its scaled bound of `-1.2e9` takes `0.021` off the gap.
    #[test]
    fn an_ill_conditioned_basis_does_not_void_a_nodes_proof() {
        let in_range = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![int(1.0, -5, 0), real(-2.0, -1e9, 0.0), real(1.0, -1.0, 0.0)],
            rows: vec![
                row(
                    &[(0, 0.00041), (1, -270.0), (2, 0.14)],
                    ComparisonOp::Le,
                    0.14,
                ),
                row(
                    &[(0, -0.0011), (1, 1.9), (2, -240.0)],
                    ComparisonOp::Eq,
                    0.0,
                ),
            ],
            xstar: vec![0.0; 3],
        };
        let out_of_range = Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(1.0, -12, 0),
                real(-2.0, -9439526.0, 0.0),
                int(2.0, -1, 0),
            ],
            rows: vec![
                row(
                    &[(0, 0.000785), (1, -16300.0), (2, -0.232)],
                    ComparisonOp::Le,
                    0.288,
                ),
                row(
                    &[(0, -9.94e-6), (1, 0.551), (2, -181.0)],
                    ComparisonOp::Eq,
                    0.0,
                ),
            ],
            xstar: vec![0.0; 3],
        };
        for model in [in_range, out_of_range] {
            assert_solves_within_contract_in_time(&model, Duration::from_secs(60));
        }
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

    /// A feasible LP whose only room is a coefficient of round-off size on a
    /// far bound reaches its optimum. The dual simplex ends with a row only
    /// `z` can repair, through an entry of `8.9e-16` beside ones, far below
    /// the pivot tolerance. The certificate's `-2^-50` for `z`, counted at
    /// `z >= -2e9`, leaves `7.8e-7` of room (counted as zero, it would prove
    /// the model infeasible): no proof. On the refined row, summed in
    /// compensated arithmetic, the entry is genuine, and the first last
    /// resort's pivot on it solves the model.
    #[test]
    fn a_room_through_a_round_off_coefficient_on_a_far_bound_is_reached() {
        check(room_through_a_round_off_coefficient(false, -2e9));
    }

    /// The model of the test above with `x, y, t` integer, whose root node's
    /// LP is that LP: the node is not pruned on a false proof, and the MILP
    /// reaches its optimum.
    #[test]
    fn a_node_lp_whose_room_is_round_off_is_solved() {
        check(room_through_a_round_off_coefficient(true, -2e9));
    }

    /// The model of
    /// [`a_room_through_a_round_off_coefficient_on_a_far_bound_is_reached`]
    /// with `z` unbounded below. A coefficient on an infinite bound counts as
    /// zero only where the multipliers' error could make it zero; this row of
    /// the basis inverse is exact, so `z`'s `-2^-50` makes the certificate's
    /// range unbounded and proves nothing, and the pivot on the entry, genuine
    /// on the refined row, reaches the optimum.
    #[test]
    fn a_room_through_a_round_off_coefficient_on_an_infinite_bound_is_reached() {
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

    /// A feasible LP whose only room is a basic var's coefficient of round-off
    /// size in what a certificate leaves reaches `L = 1`. The dual simplex
    /// ends at the basis `{j, w}` and the second row's slack, `1e-6` above its
    /// bound, which only the third row's unbounded slack can repair, through
    /// an entry of `2^-50` beside ones. The certificate leaves that row out
    /// (its multiplier needs the slack's infinite side), so the free basic `j`
    /// keeps its exact `-2^-50`, which makes the range unbounded: no proof.
    /// The entry is genuine on the refined row, and the first last resort's
    /// pivot on it solves the model.
    #[test]
    fn a_room_through_a_basic_vars_round_off_coefficient_is_reached() {
        check(room_through_a_basic_vars_coefficient(false));
    }

    /// The test above without presolve, on the model as written.
    #[test]
    fn a_room_through_a_basic_vars_round_off_coefficient_is_reached_without_presolve() {
        let model = room_through_a_basic_vars_coefficient(false);
        assert!(
            xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model it was built around"
        );
        assert_solves_within_contract_without_presolve(&model);
    }

    /// The model of the tests above with `L` integer, whose root node's LP is
    /// that LP: the node is not pruned on a false proof, and the MILP reaches
    /// its optimum, `L = 1`.
    #[test]
    fn a_node_lp_whose_room_is_a_basic_vars_coefficient_is_solved() {
        check(room_through_a_basic_vars_coefficient(true));
    }

    /// The test above without presolve.
    #[test]
    fn a_node_lp_whose_room_is_a_basic_vars_coefficient_is_solved_without_presolve() {
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
            rows_hold(&model, &model.xstar),
            "fixture error: x* must hold the rows of the model it was built around"
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
    /// the contract, and not worse than 110. At the default `int_tol` the
    /// root's `y_a = 1e-7` is integral, and its point, of cost `20.00001`,
    /// better than every integral point, is the answer (ARCHITECTURE §7.1).
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
        // The example of ARCHITECTURE §7.1, which the contract allows but does
        // not require: a failure here means updating that example.
        assert_solves_within_contract(&model);
        assert_solves_within_contract_without_presolve(&model);
        for presolve in [true, false] {
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
                objective < 110.0,
                "presolve {presolve}: objective {objective}"
            );
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

    /// [`big_m_row_met_beyond_its_bound`] with `e` at `1.03` of the big-M
    /// row's contract, `3%` beyond it with `y` at one. With `w1, w2` at their
    /// values, the model has points only with `y` off its integer within
    /// `int_tol`: `x*` has `y = 1 - 3e-14`, whose offset times `M = 5e7` makes
    /// the room. With `y = 1`, the second row holds within its tolerance only
    /// from `x = (rhs - 1e-7) / 1000`, where the big-M row misses beyond its
    /// own, and the pin rows exclude `y = 0`.
    fn big_m_row_needing_int_tol(ks: [f64; 2], pin: f64) -> Model {
        let model = big_m_row_met_beyond_its_bound(5e7, ks, pin, 1000.0, 1.03, false, 1.0 - 3e-14);
        let x_lo = (model.rows[1].rhs - 1e-7) / 1000.0;
        let at_one = [x_lo, 1.0, 0.0, 0.0, 1.0, 1.0];
        assert!(
            row_violation(&model.rows[0], &at_one) > row_tolerance(&model.rows[0], &at_one),
            "fixture error: with y = 1 the model must have no point"
        );
        model
    }

    /// [`big_m_row_needing_int_tol`] with fixed terms of `1e8`, only the two
    /// pin rows together keeping `y` at one. Presolve rewrites `y`'s
    /// coefficient to about `500`, which holds as written only at an integral
    /// `y`. No point has `y` integral and `w1, w2` at their values, so
    /// `Infeasible` and a point relying on `y`'s offset, meeting the rows as
    /// written, are both right, presolve on and off.
    #[test]
    fn a_model_feasible_only_off_its_integers_presolve_rewrites_is_answered_either_way() {
        let model = big_m_row_needing_int_tol([5e7, -5e7], 100.0);
        for presolve in [true, false] {
            infeasible_or_within_contract(&model, presolve);
        }
    }

    /// [`big_m_row_needing_int_tol`] without fixed terms, each pin row alone
    /// keeping `y` at one, so the model has no integral point: presolve fixes
    /// `y` at 1, where the engine holds it exactly, and the solve answers
    /// `Infeasible`. Without presolve the search may land on a point relying
    /// on `y`'s offset: either answer is right.
    #[test]
    fn a_model_feasible_only_off_its_integers_is_infeasible_where_presolve_fixes_its_binary() {
        let model = big_m_row_needing_int_tol([0.0, -0.0], 1e4);
        assert_eq!(infeasible_or_within_contract(&model, true), None);
        infeasible_or_within_contract(&model, false);
    }

    /// `max y` s.t. `x >= x_min`, `x <= coeff y`, `y <= 0.5`, `y` binary:
    /// the rows hold together only with `y` off its integer, from
    /// `y = x_min / coeff`, which is within `int_tol` of 0. The model has no
    /// integral point: `y = 0` leaves `x` at most `1e-7` and at least
    /// `x_min - 1e-7`, and `y = 1` misses `y <= 0.5`.
    fn binary_off_its_integer(x_min: f64, coeff: f64) -> Model {
        let model = Model {
            direction: OptimizationDirection::Maximize,
            vars: vec![real(0.0, 0.0, f64::INFINITY), int(1.0, 0, 1)],
            rows: vec![
                row(&[(0, 1.0)], ComparisonOp::Ge, x_min),
                row(&[(0, 1.0), (1, -coeff)], ComparisonOp::Le, 0.0),
                row(&[(1, 1.0)], ComparisonOp::Le, 0.5),
            ],
            xstar: vec![x_min, x_min / coeff],
        };
        assert!(
            rows_hold(&model, &model.xstar) && x_min > 2e-7,
            "fixture error: x* must hold the rows, and no point have y integral"
        );
        model
    }

    /// [`binary_off_its_integer`] with `x >= 5` and `x <= 1e7 y`, from
    /// `y = 5e-7`. Without presolve the root's LP puts `y` at `0.5`, and the
    /// branch fixes it at 0 and at 1, where the engine holds it exactly: both
    /// children are infeasible. Presolve fixes `y` at 0 from `y <= 0.5`, and
    /// the root is proven infeasible without a branch. The solve answers
    /// `Infeasible`, presolve on and off.
    #[test]
    fn a_model_feasible_only_off_its_integers_is_infeasible() {
        let model = binary_off_its_integer(5.0, 1e7);
        for presolve in [true, false] {
            assert_eq!(
                infeasible_or_within_contract(&model, presolve),
                None,
                "presolve {presolve}"
            );
        }
    }

    /// The previous test's model, in range, with `x >= 9e-7` and `x <= y`,
    /// from `y = 9e-7`: `Infeasible` too, presolve on and off.
    #[test]
    fn a_model_feasible_only_off_its_integers_in_range_is_infeasible() {
        let model = binary_off_its_integer(9e-7, 1.0);
        assert!(
            within_recommended_range(&model),
            "fixture error: the model must lie within the recommended range"
        );
        for presolve in [true, false] {
            assert_eq!(
                infeasible_or_within_contract(&model, presolve),
                None,
                "presolve {presolve}"
            );
        }
    }

    /// The models of the two previous tests minimizing `y`: the root's LP
    /// puts `y` at `x_min / coeff`, within `int_tol` of 0, a point the search
    /// may return and presolve may take away. Either answer is right,
    /// presolve on and off: which one comes can depend on the objective and
    /// on presolve.
    #[test]
    fn a_model_feasible_only_off_its_integers_minimizing_its_binary_is_answered_either_way() {
        for (x_min, coeff) in [(5.0, 1e7), (9e-7, 1.0)] {
            let mut model = binary_off_its_integer(x_min, coeff);
            model.direction = OptimizationDirection::Minimize;
            for presolve in [true, false] {
                infeasible_or_within_contract(&model, presolve);
            }
        }
    }

    /// The models of [`a_model_feasible_only_off_its_integers_is_infeasible`]
    /// and the test after it, maximizing a `z >= x` of their own instead of
    /// `y`: the relaxation is unbounded, and without presolve the search for
    /// a point that classifies it answers `Unbounded` where it lands on a
    /// point relying on `y`'s offset, `Infeasible` where it does not.
    /// Presolve fixes `y` at 0 from `y <= 0.5`, and the root is proven
    /// infeasible without a branch. Both answers are right, presolve on and
    /// off.
    #[test]
    fn an_unbounded_relaxation_feasible_only_off_its_integers_is_unbounded_or_infeasible() {
        for (x_min, coeff) in [(5.0, 1e7), (9e-7, 1.0)] {
            let mut model = binary_off_its_integer(x_min, coeff);
            model.vars[1].obj = 0.0;
            model.vars.push(real(1.0, 0.0, f64::INFINITY));
            model
                .rows
                .push(row(&[(2, 1.0), (0, -1.0)], ComparisonOp::Ge, 0.0));
            model.xstar.push(x_min);
            let (problem, _) = model.build();
            for presolve in [true, false] {
                let result = problem.solve_with(SolveOptions {
                    presolve,
                    ..SolveOptions::default()
                });
                assert!(
                    matches!(result, Err(Error::Unbounded | Error::Infeasible)),
                    "presolve {presolve}: {result:?}:\n{}",
                    model.source()
                );
            }
        }
    }

    /// `-24.4 x1 - 144 x2 + 0.287 x3 = -24.4` and three more rows, in range,
    /// over `x1 ∈ [x1_lo, x1_hi]`, the binary `x2` and `x0` fixed at
    /// `1048576`. `x*`, at `x1 = 1`, is integral.
    fn binary_a_row_relies_on(x1_lo: f64, x1_hi: f64) -> Model {
        Model {
            direction: OptimizationDirection::Minimize,
            vars: vec![
                int(0.0, 1048576, 1048576),
                real(1.0, x1_lo, x1_hi),
                int(-1.0, 0, 1),
                real(1.0, -2.0, 0.0),
            ],
            rows: vec![
                row(
                    &[(0, 54.9), (1, -0.509)],
                    ComparisonOp::Eq,
                    57566821.890999995,
                ),
                row(&[(1, 2.52)], ComparisonOp::Le, 2.52),
                row(
                    &[(1, -24.4), (2, -144.0), (3, 0.287)],
                    ComparisonOp::Eq,
                    -24.4,
                ),
                row(&[(1, -241.0), (3, 70.4)], ComparisonOp::Ge, -241.0),
            ],
            xstar: vec![1048576.0, 1.0, 0.0, 0.0],
        }
    }

    /// [`binary_a_row_relies_on`] through an edit: the solve without presolve
    /// reports `x1 = 1 + 6.6e-9` and the binary `x2` at `-1.1e-9`, each past
    /// its bound within its tolerance, the third row relying on `x2`'s
    /// offset. Fixed at that value, `x1` leaves the third row short by
    /// `1.6e-7`, which only `x2` off its integer makes up: `x2 = 0` needs
    /// `x3 >= 2.1e-7`, past its bound beyond its tolerance, and `x2 = 1` far
    /// more. With `x1` at its value the edited model has no integral point, so
    /// `Infeasible` and a point the contract accepts, `x2` off its integer or
    /// `x1` off its value, are both right. Fixed at its bound, where `x*` has
    /// it, `x1` leaves an integral point, and the edit solves.
    #[test]
    fn a_fix_at_a_value_only_points_off_their_integers_meet_is_answered_either_way() {
        let model = binary_a_row_relies_on(0.0, 1.0);
        assert!(
            within_recommended_range(&model) && xstar_is_feasible(&model),
            "fixture error: x* must satisfy the model, which lies within the recommended range"
        );
        let (problem, vars) = model.build();
        let options = SolveOptions {
            presolve: false,
            ..SolveOptions::default()
        };
        let solution = problem
            .solve_with(options)
            .unwrap()
            .into_solution()
            .unwrap();
        let value = solution[vars[1]];
        let x3_min = (24.4 * (value - 1.0) - 1e-7) / 0.287;
        assert!(
            x3_min > 1e-7,
            "fixture error: x1 = {value:?} must leave x2 = 0 needing x3 >= {x3_min:e}, past \
             its bound beyond its tolerance"
        );
        infeasible_or_answer_within_contract(
            &binary_a_row_relies_on(value, value),
            solution.clone().fix_var(vars[1], value),
            &vars,
            &format!("fix_var(x1, {value:?})"),
        );
        assert!(
            infeasible_or_answer_within_contract(
                &binary_a_row_relies_on(1.0, 1.0),
                solution.fix_var(vars[1], 1.0),
                &vars,
                "fix_var(x1, 1.0)",
            )
            .is_some(),
            "fix_var(x1, 1.0) returned Infeasible where x* is an integral point"
        );
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
