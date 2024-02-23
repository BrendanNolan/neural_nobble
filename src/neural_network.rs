use crate::{common::*, feed_forward::*};

pub mod builder;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
    pub activation_function: fn(F1) -> F1,
}

impl NeuralNetwork {
    pub fn weights(&self) -> &[Array2<f64>] {
        &self.weight_matrices
    }

    pub fn biases(&self) -> &[Array1<f64>] {
        &self.bias_vectors
    }

    pub fn layer_count(&self) -> usize {
        self.weight_matrices.len()
    }

    pub fn neuron_count(&self, layer: usize) -> Option<usize> {
        if layer < self.weight_matrices.len() {
            Some(row_count(&self.weight_matrices[layer]))
        } else {
            None
        }
    }
}
