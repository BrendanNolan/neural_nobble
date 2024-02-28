use crate::{
    common::*, feed_forward::FeedForwardResult, mini_batch::MiniBatch,
    neural_network::NeuralNetwork,
};

fn propagate_error_back(
    known_layer: usize,
    known_error: &Array2<f64>,
    network: &NeuralNetwork,
) -> Array2<f64> {
    todo!()
}

fn propagate_error_back_from_last_layer(
    feedforward_result: &FeedForwardResult,
    network: &NeuralNetwork,
) -> Array2<f64> {
    todo!()
}
