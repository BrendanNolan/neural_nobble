use crate::common::*;

pub fn quadratic(expected: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    0.5 * (expected - actual).mapv(|x| x.powi(2)).sum()
}
