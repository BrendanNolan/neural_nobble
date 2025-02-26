use std::result;

use crate::common::*;
use crate::derivative::ActivationFunction;

#[derive(Debug, Clone, Copy, Default)]
pub struct IdFunc {}

impl ActivationFunction for IdFunc {
    fn apply(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        weighted_inputs.clone()
    }
    fn derivative(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        Array2::<f64>::from_elem(weighted_inputs.dim(), 1.0)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SigmoidFunc {}

impl ActivationFunction for SigmoidFunc {
    fn apply(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        let elems: Vec<_> = weighted_inputs
            .iter()
            .map(|x| 1.0 / (1.0 + (-*x).exp()))
            .collect();
        Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
    }
    fn derivative(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        let elems: Vec<_> = weighted_inputs.iter().map(|x| *x * (1.0 - *x)).collect();
        Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReluFunc {}

impl ActivationFunction for ReluFunc {
    fn apply(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        let elems: Vec<_> = weighted_inputs.iter().map(|x| x.max(0.0)).collect();
        Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
    }
    fn derivative(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        let elems: Vec<_> = weighted_inputs
            .iter()
            .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
            .collect();
        Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
    }
}
