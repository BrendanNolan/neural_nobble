use crate::{activation_functions::ActivationFunction, common::*, feed_forward::*};

pub mod builder;

#[derive(Debug)]
pub struct NeuralNetwork {
    weight_matrices: Vec<Array2<f32>>,
    bias_vectors: Vec<Array1<f32>>,
    activation_functions: Vec<ActivationFunction>,
}

impl NeuralNetwork {
    pub fn weight_matrices_mut(&mut self) -> &mut [Array2<f32>] {
        &mut self.weight_matrices[1..]
    }

    pub fn bias_vectors_mut(&mut self) -> &mut [Array1<f32>] {
        &mut self.bias_vectors[1..]
    }

    pub fn weights(&self, layer: NonZeroUsize) -> &Array2<f32> {
        &self.weight_matrices[layer.get()]
    }

    pub fn biases(&self, layer: NonZeroUsize) -> &Array1<f32> {
        &self.bias_vectors[layer.get()]
    }

    pub fn activation_function(&self, layer: NonZeroUsize) -> ActivationFunction {
        self.activation_functions[layer.get()]
    }

    pub fn last_activation(&self) -> ActivationFunction {
        *self.activation_functions.last().unwrap()
    }

    pub fn layer_count(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.weight_matrices.len()).unwrap()
    }

    pub fn neuron_count(&self, layer: usize) -> usize {
        if layer == 0 {
            column_count(&self.weight_matrices[1])
        } else {
            row_count(&self.weight_matrices[layer])
        }
    }
}
