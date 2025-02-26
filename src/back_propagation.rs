use crate::{
    activation_functions::*, common::*, cost_functions::CostFunction,
    feed_forward::FeedForwardResult, mini_batch::MiniBatch, neural_network::NeuralNetwork,
};

// takes the average of the gradients over all training examples
pub fn compute_gradient_of_cost_wrt_weights(
    network: &NeuralNetwork,
    feedforward_result: &FeedForwardResult,
    layer: NonZeroUsize,
    errors_by_layer: &[Array2<f64>],
) -> Array2<f64> {
    let layer = layer.get();
    let gradient = errors_by_layer[layer].dot(&feedforward_result.activations[layer - 1].t());
    // effectively averaging over the gradients of the batch members
    (1.0 / feedforward_result.number_of_training_examples() as f64) * gradient
}

// takes the average of the gradients over all training examples
pub fn compute_gradient_of_cost_wrt_biases(
    layer: NonZeroUsize,
    errors_by_layer: &[Array2<f64>],
) -> Array1<f64> {
    errors_by_layer[layer.get()]
        .mean_axis(Axis(1))
        .unwrap()
        .to_owned()
}

pub fn compute_errors_by_layer(
    network: &NeuralNetwork,
    mini_batch: &MiniBatch,
    feedforward_result: &FeedForwardResult,
    cost_function: &impl CostFunction,
) -> Vec<Array2<f64>> {
    let mut errors = vec![];
    errors.push(compute_error_at_last_layer(
        feedforward_result,
        network.activation_function(NonZeroUsize::new(network.layer_count().get() - 1).unwrap()),
        cost_function,
        mini_batch,
    ));
    let last_layer_from_which_to_propagare_back = 2; // No such thing as error in 0th layer
    for layer in (last_layer_from_which_to_propagare_back..network.layer_count().get()).rev() {
        errors.push(propagate_error_back(
            network,
            feedforward_result,
            NonZeroUsize::new(layer).unwrap(),
            errors.last().unwrap(),
        ));
    }
    // Dummy error for 0th layer, to make returned array indexable by layer
    errors.push(Array2::<f64>::zeros((0, 0)));
    errors.reverse();
    errors
}

fn propagate_error_back(
    network: &NeuralNetwork,
    feedforward_result: &FeedForwardResult,
    known_layer: NonZeroUsize,
    known_error: &Array2<f64>,
) -> Array2<f64> {
    let needed_layer = known_layer.get() - 1;
    let needed_layer_nonzero = NonZeroUsize::new(known_layer.get() - 1).unwrap();
    let activation_derivatives_at_weighted_inputs = network
        .activation_function(needed_layer_nonzero)
        .derivative(&feedforward_result.activations[needed_layer]);
    network.weights(known_layer).t().dot(known_error) * activation_derivatives_at_weighted_inputs
}

fn compute_error_at_last_layer(
    feedforward_result: &FeedForwardResult,
    activation_function: ActivationFunction,
    cost_function: &impl CostFunction,
    mini_batch: &MiniBatch,
) -> Array2<f64> {
    let activation_derivatives_at_weighted_inputs =
        activation_function.derivative(feedforward_result.weighted_inputs.last().unwrap());
    let mut cost_gradients = cost_function.partial_derivative(
        feedforward_result.activations.last().unwrap(),
        &mini_batch.targets,
    );
    cost_gradients * activation_derivatives_at_weighted_inputs
}

#[cfg(test)]
mod tests {
    use crate::activation_functions::*;
    use crate::common::*;
    use crate::cost_functions::HalfSSECostFunction;
    use crate::feed_forward::*;
    use crate::mini_batch::*;
    use crate::neural_network;
    use crate::neural_network::*;
    use ndarray::{arr1, arr2, Array1, Array2};

    use super::compute_errors_by_layer;
    use super::compute_gradient_of_cost_wrt_biases;
    use super::compute_gradient_of_cost_wrt_weights;

    #[test]
    fn test_back_prop() {
        let network = neural_network::builder::NeuralNetworkBuilder::new(2)
            .add_layer(
                arr2(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
                arr1(&[1.0, 2.0, 3.0]),
                ActivationFunction::Id,
            )
            .unwrap()
            .add_layer(
                arr2(&[[6.0, 4.0, 2.0], [5.0, 3.0, 1.0]]),
                arr1(&[10.0, 20.0]),
                ActivationFunction::Id,
            )
            .unwrap()
            .build();
        let mini_batch = MiniBatch {
            inputs: arr2(&[[2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]),
            targets: arr2(&[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]),
        };
        let feedforward_result = feed_forward(&network, &mini_batch);
        let cost_function = HalfSSECostFunction;
        let errors_by_layer =
            compute_errors_by_layer(&network, &mini_batch, &feedforward_result, &cost_function);
        assert_eq!(
            errors_by_layer[2],
            arr2(&[[237.0, 237.0, 237.0, 237.0], [183.0, 183.0, 183.0, 183.0]])
        );
        for layer in 1..=2 {
            let layer = NonZeroUsize::new(layer).unwrap();
            let cost_gradient_with_respect_to_weights = compute_gradient_of_cost_wrt_weights(
                &network,
                &feedforward_result,
                layer,
                &errors_by_layer,
            );
            let cost_gradient_with_respect_to_biases =
                compute_gradient_of_cost_wrt_biases(layer, &errors_by_layer);
        }
    }
}
