use self::feed_forward::*;
use crate::common::*;

pub mod builder;
mod feed_forward;
mod mini_batch;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    pub weight_matrices: Vec<Array2<f64>>,
    pub bias_vectors: Vec<Array1<f64>>,
    pub activation_function: fn(F1) -> F1,
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
