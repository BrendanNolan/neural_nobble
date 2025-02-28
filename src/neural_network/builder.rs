use crate::activation_functions;
use crate::activation_functions::*;
use crate::common::*;
use crate::distribution;
use crate::neural_network::NeuralNetwork;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub struct NeuralNetworkBuilder {
    input_size: usize,
    weight_matrices: Vec<Array2<f64>>,
    bias_vectors: Vec<Array1<f64>>,
    activation_functions: Vec<ActivationFunction>,
    random_number_generator: StdRng,
}

impl NeuralNetworkBuilder {
    pub fn new(input_size: usize) -> Self {
        NeuralNetworkBuilder {
            input_size,
            // sacrificial zeroth layer to make indexing easier
            weight_matrices: vec![Array2::zeros((0, 0))],
            // sacrificial zeroth layer to make indexing easier
            bias_vectors: vec![Array1::zeros(0)],
            // sacrificial zeroth layer to make indexing easier
            activation_functions: vec![ActivationFunction::Id],
            random_number_generator: StdRng::seed_from_u64(147),
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
        activation_function: ActivationFunction,
    ) -> Option<Self> {
        if !self.new_layer_valid(&weight_matrix, &bias_vector) {
            return None;
        }
        self.weight_matrices.push(weight_matrix);
        self.bias_vectors.push(bias_vector);
        self.activation_functions.push(activation_function);
        Some(self)
    }

    pub fn add_layer_random(
        mut self,
        neuron_count: usize,
        activation_function: ActivationFunction,
    ) -> Option<Self> {
        let weight_matrix = match activation_function
            .suggested_distribution(self.last_layer_neuron_count(), neuron_count)
        {
            distribution::Distribution::Normal {
                mean,
                standard_deviation,
            } => Array2::random_using(
                (neuron_count, self.last_layer_neuron_count()),
                Normal::new(mean, standard_deviation).unwrap(),
                &mut self.random_number_generator,
            ),
            distribution::Distribution::Uniform {
                lower_bound,
                upper_bound,
            } => Array2::random_using(
                (neuron_count, self.last_layer_neuron_count()),
                Uniform::new(lower_bound, upper_bound),
                &mut self.random_number_generator,
            ),
        };
        let bias_vector = Array1::zeros(neuron_count);
        self.add_layer(weight_matrix, bias_vector, activation_function)
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
            activation_functions: self.activation_functions,
        }
    }
}
