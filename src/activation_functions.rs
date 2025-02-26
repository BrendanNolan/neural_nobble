pub use ndarray_rand::{
    rand_distr::{Distribution, Normal},
    RandomExt,
};
use std::result;

use crate::common::*;

#[derive(Copy, Clone, Debug)]
pub enum ActivationFunction {
    IdFunc,
    SigmoidFunc,
    ReluFunc,
}

impl ActivationFunction {
    pub fn apply(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        match *self {
            ActivationFunction::IdFunc => weighted_inputs.clone(),
            ActivationFunction::SigmoidFunc => {
                let elems: Vec<_> = weighted_inputs
                    .iter()
                    .map(|x| 1.0 / (1.0 + (-*x).exp()))
                    .collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
            ActivationFunction::ReluFunc => {
                let elems: Vec<_> = weighted_inputs.iter().map(|x| x.max(0.0)).collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
        }
    }

    pub fn derivative(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        match *self {
            ActivationFunction::IdFunc => Array2::<f64>::from_elem(weighted_inputs.dim(), 1.0),
            ActivationFunction::SigmoidFunc => {
                let elems: Vec<_> = weighted_inputs.iter().map(|x| *x * (1.0 - *x)).collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
            ActivationFunction::ReluFunc => {
                let elems: Vec<_> = weighted_inputs
                    .iter()
                    .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
        }
    }

    pub fn suggested_distribution(&self, prev_layer_neuron_count: usize) -> Normal<f64> {
        match *self {
            ActivationFunction::IdFunc => Normal::new(0.0, 1.0).unwrap(),
            ActivationFunction::SigmoidFunc => {
                Normal::new(0.0, (1.0 / prev_layer_neuron_count as f64)).unwrap()
            }
            ActivationFunction::ReluFunc => {
                Normal::new(0.0, (2.0 / prev_layer_neuron_count as f64)).unwrap()
            }
        }
    }
}
