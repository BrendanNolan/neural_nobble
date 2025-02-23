use crate::{
    common::*, derivative::DifferentiableFunction, mini_batch::MiniBatch,
    neural_network::NeuralNetwork,
};
use std::ops::AddAssign;

#[derive(Debug, Default)]
pub struct FeedForwardResult {
    pub activations: Vec<Array2<f64>>,
    pub weighted_inputs: Vec<Array2<f64>>,
}

impl FeedForwardResult {
    pub fn number_of_training_examples(&self) -> usize {
        column_count(self.activations.first().unwrap())
    }
}

pub enum FeedForwardError {
    InappropriateMiniBatchInputSize,
    InappropriateMiniBatchTargetSize,
}

pub fn feed_forward(
    network: &NeuralNetwork,
    activation_function: impl DifferentiableFunction,
    mini_batch: &MiniBatch,
) -> FeedForwardResult {
    let mut activations: Vec<Array2<f64>> = Vec::with_capacity(network.layer_count().get());
    activations.push(mini_batch.inputs.clone());
    let mut weighted_inputs: Vec<Array2<f64>> = Vec::with_capacity(network.layer_count().get());
    weighted_inputs.push(Array2::zeros((0, 0))); // sacrificial empty matrix to make indexing easier
    for layer in 1..network.layer_count().get() {
        let prev_activations = &activations[layer - 1];
        let mut weighted_input = network
            .weights(NonZeroUsize::new(layer).unwrap())
            .dot(prev_activations);
        for column in 0..weighted_input.ncols() {
            weighted_input
                .column_mut(column)
                .add_assign(network.biases(NonZeroUsize::new(layer).unwrap()));
        }
        weighted_inputs.push(weighted_input.clone());
        activations.push(weighted_inputs.last().unwrap().clone());
        activations
            .last_mut()
            .unwrap()
            .map_inplace(|x| *x = activation_function.apply(*x));
    }
    FeedForwardResult {
        activations,
        weighted_inputs,
    }
}

pub fn print_details(feed_forward: &FeedForwardResult, targets: &Array2<f64>, count: usize) {
    let prediction_matrix = feed_forward.activations.last().unwrap();
    for col in 0..count {
        let column_a: Vec<String> = prediction_matrix
            .column(col)
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect();
        let column_b: Vec<String> = targets
            .column(col)
            .iter()
            .map(|x| format!("{:.3}", x))
            .collect();
        println!("\nPred   {:?}\nActual: {:?}\n", column_a, column_b);
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
        let network = neural_network::builder::NeuralNetworkBuilder::new(2)
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
            targets: arr2(&[[1.0, 1.0], [0.0, 0.0]]),
        };
        let result = feed_forward(&network, IdFunc::default(), &mini_batch);
        assert_eq!(result.weighted_inputs.len(), 3);
        assert_eq!(result.activations.len(), 3);
        assert_eq!(result.weighted_inputs[0], Array2::zeros((0, 0)));
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
