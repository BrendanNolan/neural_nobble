use crate::{
    activation_functions::*, common::*, cost_functions::CostFunction, derivative::*,
    feed_forward::FeedForwardResult, mini_batch::MiniBatch, neural_network::NeuralNetwork,
};

fn propagate_error_back(
    known_layer: usize,
    known_error: &Array2<f64>,
    network: &NeuralNetwork,
) -> Array2<f64> {
    todo!()
}

fn compute_error_at_last_layer(
    feedforward_result: &FeedForwardResult,
    network: &NeuralNetwork,
    activation_function: fn(f64) -> f64,
    cost_function: &impl CostFunction,
    mini_batch: &MiniBatch,
) -> Array2<f64> {
    let activation_derivatives = feedforward_result
        .activations
        .last()
        .unwrap()
        .map(|x| derivative(activation_function, *x).unwrap());
    let cost_gradients = mini_batch.inputs.columns().into_iter().enumerate().map()
}
