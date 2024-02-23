use super::{mini_batch::MiniBatch, NeuralNetwork};
use crate::common::*;

#[derive(Debug, Default)]
pub struct FeedForwardResult {
    pub activations: Vec<Array2<f64>>,
    pub weighted_inputs: Vec<Array2<f64>>,
}

pub enum FeedForwardError {
    InappropriateMiniBatchInputSize,
    InappropriateMiniBatchTargetSize,
}

pub fn feed_forward(neural_network: &NeuralNetwork, mini_batch: &MiniBatch) -> FeedForwardResult {
    let mut activations = Vec::with_capacity(neural_network.weight_matrices.len() + 1);
    let mut weighted_inputs = Vec::with_capacity(neural_network.weight_matrices.len());
}
