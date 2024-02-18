use crate::array_utils::*;
use autodiff::*;
use ndarray::{array, Array1, Array2};
use ndarray_rand::{
    rand_distr::{Distribution, Normal},
    RandomExt,
};

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

pub struct NeuralNetworkBuilder {
    input_size: usize,
    activation_function: fn(F1) -> F1,
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
}

impl NeuralNetworkBuilder {
    pub fn new(input_size: usize, activation_function: fn(F1) -> F1) -> Self {
        NeuralNetworkBuilder {
            input_size,
            activation_function,
            weight_matrices: Vec::new(),
            bias_vectors: Vec::new(),
        }
    }

    fn previous_layer_neuron_count(&self) -> usize {
        if self.layers_added() == 0 {
            self.input_size
        } else {
            self.neuron_count(self.layers_added() - 1).unwrap()
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

    pub fn add_layer_random(mut self, neuron_count: usize) -> Self {
        let weight_matrix = Array2::random(
            (neuron_count, self.previous_layer_neuron_count()),
            Normal::new(0.0, 1.0).unwrap(),
        );
        let bias_vector = Array1::random(neuron_count, Normal::new(0.0, 1.0).unwrap());
        self.add_layer(weight_matrix, bias_vector).unwrap()
    }

    fn layers_added(&self) -> usize {
        self.weight_matrices.len()
    }

    pub fn neuron_count(&self, layer: usize) -> Option<usize> {
        if layer < self.weight_matrices.len() {
            Some(row_count(&self.weight_matrices[layer]))
        } else {
            None
        }
    }

    fn new_layer_valid(&self, weight_matrix: &Array2<f64>, bias_vector: &Array1<f64>) -> bool {
        if row_count(weight_matrix) != bias_vector.len() {
            return false;
        }
        column_count(weight_matrix) == self.previous_layer_neuron_count()
    }

    fn build(self) -> NeuralNetwork {
        NeuralNetwork {
            weight_matrices: self.weight_matrices,
            bias_vectors: self.bias_vectors,
            activation_function: self.activation_function,
        }
    }
}
