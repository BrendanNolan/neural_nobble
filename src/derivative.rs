use crate::{activation_functions::*, cost_functions::*};

pub fn derivative(f: fn(f64)->f64) -> Option<fn(f64)->f64> {
    match f {
        sigmoid => Some(sigmoid_derivative),
        _ => None,
    }
}
