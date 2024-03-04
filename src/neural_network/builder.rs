use crate::common::*;
use crate::neural_network::NeuralNetwork;

pub struct NeuralNetworkBuilder {
    input_size: usize,
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
}

impl NeuralNetworkBuilder {
    pub fn new(input_size: usize) -> Self {
        NeuralNetworkBuilder {
            input_size,
            // Sacrificial zeroth layer to make indexing easier
            weight_matrices: vec![Array2::zeros((0, 0))],
            // Sacrificial zeroth layer to make indexing easier
            bias_vectors: vec![Array1::zeros(0)],
        }
    }

    fn last_layer_neuron_count(&self) -> usize {
        let last_layer = self.weight_matrices.len() == 1;
        if last_layer {
            self.input_size
        } else {
            row_count(self.weight_matrices.last().unwrap())
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
            (neuron_count, self.last_layer_neuron_count()),
            Normal::new(0.0, 1.0).unwrap(),
        );
        let bias_vector = Array1::random(neuron_count, Normal::new(0.0, 1.0).unwrap());
        self.add_layer(weight_matrix, bias_vector).unwrap()
    }

    fn new_layer_valid(&self, weight_matrix: &Array2<f64>, bias_vector: &Array1<f64>) -> bool {
        if row_count(weight_matrix) != bias_vector.len() {
            return false;
        }
        column_count(weight_matrix) == self.last_layer_neuron_count()
    }

    pub fn build(self) -> NeuralNetwork {
        NeuralNetwork {
            weight_matrices: self.weight_matrices,
            bias_vectors: self.bias_vectors,
        }
    }
}
