use crate::{activation_functions::*, common::*, cost_functions::*};

pub trait ActivationFunction: Copy + Clone {
    fn apply(&self, weighted_inputs: &Array2<f64>) -> Array2<f64>;
    fn derivative(&self, weighted_inputs: &Array2<f64>) -> Array2<f64>;
}
