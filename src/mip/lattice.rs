//! The values the objective takes at integral points. Where every var with a
//! nonzero cost is an integer var or fixed, the objective of a point whose
//! integer vars are integral and whose fixed vars are at their values is
//! `offset + k * step` for an integer `k`: `offset` sums the fixed vars'
//! terms, and `step` is the greatest common divisor of the other costs. A
//! bound on the objective of a node's exactly integral points (those that
//! also meet its rows and bounds exactly) then rounds up to the least such
//! value at or above it.

use crate::solver::{bound_tolerance, UNIT_ROUNDOFF};
use crate::VarDomain;

/// The lattice `offset + k * step` (internal minimize space) of the
/// objective's values at the integral points of a search's model, and the
/// margin by which a computed bound can lie above the least of them in a
/// node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ObjectiveLattice {
    offset: f64,
    step: f64,
    /// How far the computed bound of a node, its LP objective less its dual
    /// slack (`Solver::dual_slack`), can lie above the same bound at the
    /// exact vertex: each cost var's cost times its tolerance, within which
    /// the contract cannot tell a computed value from the vertex's, plus the
    /// round-off of the plain sums of the objective and of `offset` and of
    /// rounding a bound onto the lattice, `(2n + 8) 2^-53` of the most the
    /// `n` cost terms add up to within their bounds and tolerances.
    margin: f64,
}

impl ObjectiveLattice {
    /// The lattice of the objective `obj_coeffs` over the bounds the search
    /// starts from, or `None` where a var with a nonzero cost is continuous
    /// and not fixed, no integer var that is not fixed has one, the step is
    /// not a normal `f64` number or lies within the margin, or the offset or
    /// the margin is not finite.
    pub(crate) fn of(
        obj_coeffs: &[f64],
        var_mins: &[f64],
        var_maxs: &[f64],
        domains: &[VarDomain],
        feasibility: f64,
    ) -> Option<Self> {
        let mut offset = 0.0;
        // The step as its odd part and its power of two.
        let mut step: Option<(u64, i32)> = None;
        let (mut room, mut magnitude, mut terms) = (0.0, 0.0, 0u32);
        for (v, &cost) in obj_coeffs.iter().enumerate() {
            if cost == 0.0 {
                continue;
            }
            let (lo, hi) = (var_mins[v], var_maxs[v]);
            let integer = matches!(domains[v], VarDomain::Integer | VarDomain::Boolean);
            // The engine holds an integer var its bounds fix exactly at its
            // value (`structural_tol`).
            let tol = if integer && lo == hi {
                0.0
            } else {
                bound_tolerance(feasibility, lo).max(bound_tolerance(feasibility, hi))
            };
            if lo == hi {
                offset += cost * lo;
            } else if integer {
                let (odd, exponent) = odd_and_exponent(cost);
                step = Some(match step {
                    None => (odd, exponent),
                    Some((g, e)) => (gcd(g, odd), e.min(exponent)),
                });
            } else {
                return None;
            }
            terms += 1;
            room += cost.abs() * tol;
            magnitude += cost.abs() * (lo.abs().max(hi.abs()) + tol);
        }
        let (odd, exponent) = step?;
        let margin = room + (2.0 * f64::from(terms) + 8.0) * UNIT_ROUNDOFF * magnitude;
        if !(-1022..=1023).contains(&exponent) || !margin.is_finite() || !offset.is_finite() {
            return None;
        }
        // Exact: at most the least cost, and a normal number.
        let step = odd as f64 * pow2(exponent);
        // A step within the margin rounds no bound up past itself, and the
        // count of steps in a bound could then exceed `f64`.
        if step <= margin {
            return None;
        }
        Some(Self {
            offset,
            step,
            margin,
        })
    }

    /// The least lattice value at or above `bound` less the margin, where
    /// `bound` bounds the objective of a node's exactly integral points
    /// (`Node::lattice_bound`): a lower bound on it too.
    pub(crate) fn integral_bound(&self, bound: f64) -> f64 {
        let steps = ((bound - self.margin - self.offset) / self.step).ceil();
        self.offset + steps * self.step
    }
}

/// The odd integer and the exponent of a finite, nonzero `x`:
/// `|x| = odd * 2^exponent`. The greatest common divisor of such numbers is
/// that of their odd parts times the least of their powers of two.
fn odd_and_exponent(x: f64) -> (u64, i32) {
    let bits = x.abs().to_bits();
    let biased = (bits >> 52) as i32;
    let fraction = bits & ((1 << 52) - 1);
    let (mantissa, exponent) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1 << 52), biased - 1075)
    };
    let zeros = mantissa.trailing_zeros();
    (mantissa >> zeros, exponent + zeros as i32)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `2^exponent` for an exponent of a normal `f64`.
fn pow2(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VarDomain::{Boolean, Integer, Real};

    const F: f64 = 1e-7;

    fn lattice(
        costs: &[f64],
        bounds: &[(f64, f64)],
        domains: &[VarDomain],
    ) -> Option<ObjectiveLattice> {
        let (mins, maxs): (Vec<f64>, Vec<f64>) = bounds.iter().copied().unzip();
        ObjectiveLattice::of(costs, &mins, &maxs, domains, F)
    }

    #[test]
    fn a_floats_odd_part_and_power_of_two_give_it_back() {
        for x in [
            1.0,
            0.5,
            6.0,
            -12.0,
            0.1,
            3e300,
            1e-300,
            f64::MIN_POSITIVE,
            5e-324,
        ] {
            let (odd, exponent) = odd_and_exponent(x);
            assert_eq!(odd % 2, 1, "{x}");
            let half = exponent / 2;
            assert_eq!(
                odd as f64 * pow2(half) * pow2(exponent - half),
                x.abs(),
                "{x}"
            );
        }
        assert_eq!(odd_and_exponent(6.0), (3, 1));
        assert_eq!(odd_and_exponent(0.75), (3, -2));
    }

    #[test]
    fn the_step_is_the_greatest_common_divisor_of_the_integer_vars_costs() {
        let step = |costs: &[f64]| {
            let n = costs.len();
            lattice(costs, &vec![(0.0, 10.0); n], &vec![Integer; n]).map(|l| l.step)
        };
        assert_eq!(step(&[6.0, 10.0, 15.0]), Some(1.0));
        assert_eq!(step(&[4.0, -6.0]), Some(2.0));
        assert_eq!(step(&[-5.0]), Some(5.0));
        assert_eq!(step(&[0.5, 1.5]), Some(0.5));
        assert_eq!(step(&[0.75, 3.0]), Some(0.75));
        // `0.2` is twice `0.1` in `f64`; `0.3` is not three times it, and
        // their step, `2^-55`, lies within the margin.
        assert_eq!(step(&[0.1, 0.2]), Some(0.1));
        assert_eq!(step(&[0.1, 0.3]), None);
        assert_eq!(step(&[1e20, 3e20]), Some(1e20));
    }

    #[test]
    fn a_fixed_var_with_a_cost_offsets_the_lattice_and_a_continuous_one_breaks_it() {
        let fixed = lattice(
            &[1.0, 0.7, 0.5, 2.0],
            &[(0.0, 9.0), (0.3, 0.3), (2.0, 2.0), (1.0, 1.0)],
            &[Integer, Real, Integer, Boolean],
        )
        .unwrap();
        assert_eq!(fixed.step, 1.0);
        assert_eq!(fixed.offset, 0.7 * 0.3 + 0.5 * 2.0 + 2.0);

        assert_eq!(
            lattice(&[1.0, 0.7], &[(0.0, 9.0), (0.0, 1.0)], &[Integer, Real]),
            None,
            "a continuous var with a cost"
        );
        assert!(
            lattice(&[1.0, 0.0], &[(0.0, 9.0), (0.0, 1.0)], &[Integer, Real]).is_some(),
            "a continuous var without a cost"
        );
        assert_eq!(
            lattice(&[0.7, 0.0], &[(0.3, 0.3), (0.0, 9.0)], &[Real, Integer]),
            None,
            "no integer var with a cost"
        );
    }

    /// A cost of `2^-1000` beside `1`: a bound of `2e7` would be more steps
    /// than `f64` holds, and rounding it gives infinity.
    #[test]
    fn a_step_within_the_margin_gives_no_lattice() {
        let l = lattice(
            &[1.0, 2f64.powi(-1000)],
            &[(0.0, 1e8), (0.0, 10.0)],
            &[Integer, Integer],
        );
        assert_eq!(l, None);
    }

    #[test]
    fn a_bound_rounds_up_to_the_lattice_less_its_margin() {
        let l = lattice(
            &[2.0, 2.0, 0.25],
            &[(0.0, 9.0), (-3.0, 3.0), (1.0, 1.0)],
            &[Integer, Integer, Real],
        )
        .unwrap();
        assert_eq!((l.offset, l.step), (0.25, 2.0));
        assert!(l.margin > 0.0 && l.margin < 1e-5, "margin {}", l.margin);
        assert_eq!(l.integral_bound(-6.75), -5.75);
        assert_eq!(l.integral_bound(-5.75), -5.75);
        assert_eq!(l.integral_bound(-5.75 + 0.5 * l.margin), -5.75);
        assert_eq!(l.integral_bound(-5.75 + 2.0 * l.margin), -3.75);
        assert_eq!(l.integral_bound(-4.0), -3.75);
    }
}
