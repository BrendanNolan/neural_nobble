use std::ops::AddAssign;

use crate::{common::*, mini_batch::MiniBatch, neural_network::NeuralNetwork};

#[derive(Debug, Default)]
pub struct FeedForwardResult {
    pub activations: Vec<Array2<f64>>,
    pub weighted_inputs: Vec<Array2<f64>>,
}

pub enum FeedForwardError {
    InappropriateMiniBatchInputSize,
    InappropriateMiniBatchTargetSize,
}

pub fn feed_forward(network: &NeuralNetwork, mini_batch: &MiniBatch) -> FeedForwardResult {
    let mut activations: Vec<Array2<f64>> = Vec::with_capacity(network.layer_count());
    let mut weighted_inputs: Vec<Array2<f64>> = Vec::with_capacity(network.layer_count());
    weighted_inputs.push(Array2::zeros((0, 0))); // Sacrificial empty matrix to make indexing easier
    for layer in 1..network.layer_count() {
        let prev_activations = &activations[layer - 1];
        let mut weighted_input = network.weights()[layer].dot(prev_activations);
        for column in 0..weighted_input.ncols() {
            weighted_input
                .column_mut(column)
                .add_assign(&network.biases()[layer]);
        }
        weighted_inputs.push(weighted_input.clone());
        activations = weighted_inputs.clone();
        for value in activations.iter_mut().flat_map(|matrix| matrix.iter_mut()) {
            *value = (network.activation_function)((*value).into()).into();
        }
        *activations.first_mut().unwrap() = mini_batch.inputs.clone();
    }
    FeedForwardResult {
        activations,
        weighted_inputs,
    }
}
