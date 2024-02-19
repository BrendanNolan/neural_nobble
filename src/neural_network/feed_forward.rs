use super::NeuralNetwork;
use crate::common::*;

#[derive(Debug, Default)]
struct MiniBatch {
    pub inputs: Array2<f64>,
    pub targets: Array2<f64>,
}

impl MiniBatch {
    pub fn new(inputs: Array2<f64>, targets: Array2<f64>) -> MiniBatch {
        MiniBatch { inputs, targets }
    }
}

#[derive(Debug, Default)]
struct FeedForwardResult {
    pub activations: Vec<Array2<f64>>,
    pub weighted_inputs: Vec<Array2<f64>>,
}
