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
    activations.push(mini_batch.inputs.clone());
    // for layer in 1..network.layer_count() {
    //     let prev_activations = &activations[layer - 1];
    //     let weighted_input = network.weight_matrices[layer].dot(&activation)
    //         + &network.bias_vectors[layer].insert_axis(Axis(1));
    //     let activation = (network.activation_function)(weighted_input);
    //     activations.push(activation);
    //     weighted_inputs.push(weighted_input);
    // }
    FeedForwardResult {
        activations,
        weighted_inputs,
    }
}
