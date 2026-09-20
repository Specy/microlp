//! Regressions found by fuzzing feasible-by-construction models across variable
//! magnitudes and coefficient ranges (see `scaling.rs` for the properties).
//! Each model is written around a point `x*` that satisfies every bound and
//! every row within the documented tolerance, and every bound is finite, so
//! `solve()` must return an optimal solution whose rows hold within that
//! tolerance and whose objective is no worse than at `x*` — the contract in
//! `oracle.rs`. All but the first fail today.

#[cfg(test)]
mod regression_tests {
    use crate::tests::oracle::{assert_solves_within_contract, xstar_is_feasible, Model, Row, Var};
    use crate::{ComparisonOp, OptimizationDirection};

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
}
