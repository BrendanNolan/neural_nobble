use crate::{activation_functions::*, common::*, cost_functions::*};

pub fn derivative(f: fn(f64) -> f64, at: f64) -> Option<f64> {
    match f {
        sigmoid => Some(sigmoid_derivative(at)),
        _ => None,
    }
}

fn sigmoid_derivative(x: f64) -> f64 {
    let sig = sigmoid(x);
    sig * (1.0 - sig)
}
