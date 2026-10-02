//! The recommended coefficient range of a model: within every row, and within
//! every column, the largest nonzero coefficient magnitude is at most
//! [`RECOMMENDED_DYNAMISM`] times the smallest. Nothing outside it is refused
//! or changed, and every solution is checked against the model as written all
//! the same. The range is where the solver commits to an answer: a loud error
//! on a model inside it is a bug or a known limitation (ARCHITECTURE §7.9),
//! while outside it the spread of the data can leave the simplex no pivot it
//! can trust. So a loud error names the row and the column of the widest
//! spread when they lie outside the range ([`CoefficientRange::annotate`]).

use crate::{ComparisonOp, CsVec, Error};

/// The recommended largest ratio between two nonzero coefficient magnitudes
/// of one row, or of one column: six orders of magnitude.
pub(crate) const RECOMMENDED_DYNAMISM: f64 = 1e6;

/// The spread of the coefficients of a model as written, row by row and
/// column by column.
#[derive(Clone, Debug, Default)]
pub(crate) struct CoefficientRange {
    /// How many rows have been measured, the next one's index.
    rows: usize,
    /// The row of the widest spread so far, and that spread.
    widest_row: Option<(usize, f64)>,
    /// Per column, the smallest and the largest nonzero coefficient
    /// magnitude (infinite and zero for a column with none).
    col_min: Vec<f64>,
    col_max: Vec<f64>,
}

impl CoefficientRange {
    /// The range of `rows`, numbered from zero.
    pub(crate) fn of(rows: &[(CsVec, ComparisonOp, f64)]) -> Self {
        let mut range = Self::default();
        for (coeffs, _, _) in rows {
            range.add_row(coeffs);
        }
        range
    }

    /// Measure one more row, numbered after those measured so far. A
    /// coefficient that is zero or not finite has no magnitude to compare.
    pub(crate) fn add_row(&mut self, coeffs: &CsVec) {
        let (mut min, mut max) = (f64::INFINITY, 0.0f64);
        for (var, &coeff) in coeffs.iter() {
            let magnitude = coeff.abs();
            if magnitude == 0.0 || !magnitude.is_finite() {
                continue;
            }
            min = min.min(magnitude);
            max = max.max(magnitude);
            if var >= self.col_min.len() {
                self.col_min.resize(var + 1, f64::INFINITY);
                self.col_max.resize(var + 1, 0.0);
            }
            self.col_min[var] = self.col_min[var].min(magnitude);
            self.col_max[var] = self.col_max[var].max(magnitude);
        }
        if max > 0.0 {
            let spread = max / min;
            if self.widest_row.is_none_or(|(_, widest)| spread > widest) {
                self.widest_row = Some((self.rows, spread));
            }
        }
        self.rows += 1;
    }

    /// The column of the widest spread, and that spread.
    fn widest_col(&self) -> Option<(usize, f64)> {
        let mut widest: Option<(usize, f64)> = None;
        for (col, (&min, &max)) in self.col_min.iter().zip(&self.col_max).enumerate() {
            if max > 0.0 {
                let spread = max / min;
                if widest.is_none_or(|(_, w)| spread > w) {
                    widest = Some((col, spread));
                }
            }
        }
        widest
    }

    /// Whether every row and every column lies within the range.
    #[cfg(test)]
    pub(crate) fn is_within(&self) -> bool {
        self.note().is_none()
    }

    /// What a loud error says of the range: the row and the column of the
    /// widest spread that lie outside it, if any, as the constraint (in the
    /// order added) and the variable (its index) they are.
    fn note(&self) -> Option<String> {
        let outside = [
            ("constraint", self.widest_row),
            ("variable", self.widest_col()),
        ]
        .into_iter()
        .filter_map(|(kind, widest)| {
            widest
                .filter(|&(_, spread)| spread > RECOMMENDED_DYNAMISM)
                .map(|(index, spread)| format!("{kind} {index} spans {spread:.1e}"))
        })
        .collect::<Vec<_>>();
        (!outside.is_empty()).then(|| {
            format!(
                "outside the recommended coefficient range, at most {RECOMMENDED_DYNAMISM:.0e} \
                 between the largest and the smallest coefficient magnitude of a constraint or a \
                 variable: {}",
                outside.join(", ")
            )
        })
    }

    /// `error`, an `InternalError` naming the row and the column outside the
    /// range when the model has them (see the module doc).
    pub(crate) fn annotate(&self, error: Error) -> Error {
        match error {
            Error::InternalError(message) => Error::InternalError(match self.note() {
                Some(note) => format!("{message} ({note})"),
                None => message,
            }),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(terms: &[(usize, f64)]) -> (CsVec, ComparisonOp, f64) {
        let (vars, coeffs) = terms.iter().copied().unzip();
        (
            CsVec::new_from_unsorted(4, vars, coeffs).unwrap(),
            ComparisonOp::Le,
            1.0,
        )
    }

    fn internal(message: &str) -> Error {
        Error::InternalError(message.to_string())
    }

    /// A spread of exactly six orders, in a row and in a column, is within
    /// the range, so the error is left as it is.
    #[test]
    fn an_error_of_a_model_within_the_range_names_nothing() {
        let range = CoefficientRange::of(&[row(&[(0, 1e-3), (1, 1e3)]), row(&[(0, 1e3)])]);
        assert_eq!(range.annotate(internal("stuck")), internal("stuck"));
    }

    /// The row and the column of the widest spread are named, each where it
    /// lies outside the range: row 0 spans `1e12` (`1e-8` to `1e4`), column
    /// 0 `1e8` (`1e-8` to `1`), column 1 only `1e4`.
    #[test]
    fn an_error_names_the_row_and_the_column_outside_the_range() {
        let range =
            CoefficientRange::of(&[row(&[(0, 1e-8), (1, 1e4)]), row(&[(0, -1.0), (1, 1.0)])]);
        let Error::InternalError(message) = range.annotate(internal("stuck")) else {
            panic!("an InternalError stays one");
        };
        assert!(message.starts_with("stuck ("), "{message}");
        assert!(
            message.ends_with("constraint 0 spans 1.0e12, variable 0 spans 1.0e8)"),
            "{message}"
        );
    }

    /// A column alone can lie outside the range, each of its rows within it.
    #[test]
    fn an_error_names_a_column_outside_the_range_alone() {
        let range = CoefficientRange::of(&[row(&[(2, 1e-4)]), row(&[(2, 1e4), (3, 1e4)])]);
        let Error::InternalError(message) = range.annotate(internal("stuck")) else {
            panic!("an InternalError stays one");
        };
        assert!(message.ends_with(": variable 2 spans 1.0e8)"), "{message}");
    }

    /// Only a loud failure of the solve names the range: an answer is no
    /// error, and an `Infeasible` is one the solve proved.
    #[test]
    fn only_an_internal_error_names_the_range() {
        let range = CoefficientRange::of(&[row(&[(0, 1e-8), (1, 1e4)])]);
        assert_eq!(range.annotate(Error::Infeasible), Error::Infeasible);
    }

    /// A row an edit adds is numbered after the problem's rows; zero and
    /// non-finite coefficients have no magnitude to compare.
    #[test]
    fn an_added_row_is_numbered_after_the_problems_rows() {
        let mut range = CoefficientRange::of(&[row(&[(0, 1.0), (1, 0.0)]), row(&[(1, 2.0)])]);
        assert_eq!(range.annotate(internal("stuck")), internal("stuck"));
        range.add_row(&row(&[(2, 1e-9), (3, 1.0), (0, f64::NAN)]).0);
        let Error::InternalError(message) = range.annotate(internal("stuck")) else {
            panic!("an InternalError stays one");
        };
        assert!(
            message.ends_with(": constraint 2 spans 1.0e9)"),
            "{message}"
        );
    }
}
