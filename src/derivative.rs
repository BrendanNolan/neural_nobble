use crate::{activation_functions::*, common::*, cost_functions::*};

pub trait ActivationFunction: Copy + Clone {
    fn apply(&self, input: f64) -> f64;
    fn derivative(&self, at: f64) -> f64;
}
