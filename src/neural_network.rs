use self::network_feed_forward::*;
use crate::array_utils::*;
use autodiff::*;
use ndarray::{array, Array1, Array2};
use ndarray_rand::{
    rand_distr::{Distribution, Normal},
    RandomExt,
};

pub mod builder;
mod network_feed_forward;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
    activation_function: fn(F1) -> F1,
}

impl NeuralNetwork {
    pub fn neuron_count(&self, layer: usize) -> Option<usize> {
        if layer < self.weight_matrices.len() {
            Some(row_count(&self.weight_matrices[layer]))
        } else {
            None
        }
    }
}
