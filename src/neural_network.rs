use crate::array_utils::*;
use autodiff::*;
use ndarray::{array, Array1, Array2};

pub struct NeuralNetwork {
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
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

pub struct NeuralNetworkBuilder {
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
}

impl NeuralNetworkBuilder {
    pub fn new() -> Self {
        NeuralNetworkBuilder {
            weight_matrices: Vec::new(),
            bias_vectors: Vec::new(),
        }
    }

    pub fn add_layer(
        mut self,
        weight_matrix: Array2<f64>,
        bias_vector: Array1<f64>,
    ) -> Option<Self> {
        if !self.new_layer_valid(&weight_matrix, &bias_vector) {
            return None;
        }
        self.weight_matrices.push(weight_matrix);
        self.bias_vectors.push(bias_vector);
        Some(self)
    }

    fn layers_added(&self) -> usize {
        self.weight_matrices.len()
    }

    fn new_layer_valid(&self, weight_matrix: &Array2<f64>, bias_vector: &Array1<f64>) -> bool {
        if row_count(weight_matrix) != bias_vector.len() {
            return false;
        }
        if self.layers_added() == 0 {
            return true;
        }
        let previous_layer_neuron_count = row_count(&self.weight_matrices[self.layers_added() - 1]);
        column_count(weight_matrix) == previous_layer_neuron_count
    }

    fn build(self) -> NeuralNetwork {
        NeuralNetwork {
            weight_matrices: self.weight_matrices,
            bias_vectors: self.bias_vectors,
        }
    }
}
