use crate::common::*;

pub fn quadratic(expected: &Array1<f64>, actual: &Array1<f64>) -> f64 {
    assert!(expected.len() == actual.len());
    (1 / expected.len()) as f64 * (expected - actual).mapv(|x| x.powi(2)).sum()
}

pub fn mean_squared_error_derivative(
    partial_position: usize,
    input: &Array1<f64>,
    target: &Array1<f64>,
) -> f64 {
    todo!();
}
