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

#[cfg(test)]
mod tests {
    use crate::activation_functions::*;
    use crate::feed_forward::*;
    use crate::neural_network;
    use crate::neural_network::*;
    use ndarray::{arr1, arr2, Array1};

    #[test]
    fn test_feed_forward() {
        let network = neural_network::builder::NeuralNetworkBuilder::new(2, identity)
            .add_layer(
                arr2(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
                arr1(&[1.0, 2.0, 3.0]),
            )
            .unwrap()
            .add_layer(
                arr2(&[[6.0, 4.0, 2.0], [5.0, 3.0, 1.0]]),
                arr1(&[10.0, 20.0]),
            )
            .unwrap()
            .build();
        let mini_batch = MiniBatch {
            inputs: arr2(&[[2.0, 2.0], [3.0, 3.0]]),
            targets: arr2(&[[1.0], [0.0]]),
        };
        let result = feed_forward(&network, &mini_batch);
        assert_eq!(result.weighted_inputs.len(), 3);
        assert_eq!(result.activations.len(), 3);
        assert_eq!(result.weighted_inputs[0], arr2(&[[0.0, 0.0], [0.0, 0.0]]));
        assert_eq!(result.activations[0], mini_batch.inputs);
        assert_eq!(
            result.weighted_inputs[1],
            arr2(&[[15.0, 15.0], [21.0, 21.0], [27.0, 27.0],])
        );
        assert_eq!(result.activations[1], result.weighted_inputs[1],);
        assert_eq!(
            result.weighted_inputs[2],
            arr2(&[[238.0, 238.0], [185.0, 185.0]])
        );
        assert_eq!(result.activations[2], result.weighted_inputs[2],);
    }
}
