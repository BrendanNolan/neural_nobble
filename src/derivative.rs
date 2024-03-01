use crate::{activation_functions::*, common::*, cost_functions::*};

pub fn derivative(f: fn(f64) -> f64, at: f64) -> Option<f64> {
    match f {
        sigmoid => Some(sigmoid_derivative(at)),
        _ => None,
    }
}

pub fn partial_derivative(
    f: fn(&Array1<f64>, &Array1<f64>) -> f64,
    partial_position: usize,
    input: &Array1<f64>,
    target: &Array1<f64>,
) -> Option<f64> {
    match f {
        mean_squared_error => Some(mean_squared_error_derivative(
            partial_position,
            input,
            target,
        )),
        _ => None,
    }
}
