use crate::{common::*, feed_forward::*};
use std::num::NonZeroUsize;

pub mod builder;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
    pub activation_function: fn(f64) -> f64,
    pub cost_function: fn(&Array1<f64>, &Array1<f64>) -> f64,
}

impl NeuralNetwork {
    pub fn weights(&self, layer: NonZeroUsize) -> &Array2<f64> {
        &self.weight_matrices[layer.get()]
    }

    pub fn biases(&self, layer: NonZeroUsize) -> &Array1<f64> {
        &self.bias_vectors[layer.get()]
    }

    pub fn layer_count(&self) -> usize {
        self.weight_matrices.len()
    }

    pub fn neuron_count(&self, layer: usize) -> usize {
        if layer == 0 {
            column_count(&self.weight_matrices[1])
        } else {
            row_count(&self.weight_matrices[layer])
        }
    }
}
