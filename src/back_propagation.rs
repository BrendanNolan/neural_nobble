use crate::{
    activation_functions::*, common::*, cost_functions::CostFunction, derivative::*,
    feed_forward::FeedForwardResult, mini_batch::MiniBatch, neural_network::NeuralNetwork,
};

pub fn compute_gradient_of_cost_wrt_weights(
    network: &NeuralNetwork,
    mini_batch: &MiniBatch,
    feedforward_result: &FeedForwardResult,
    layer: NonZeroUsize,
    errors_by_layer: &[Array2<f64>],
) -> Array2<f64> {
    let row_count = row_count(network.weights(layer));
    let layer = layer.get();
    let gradient = &feedforward_result.activations[layer - 1].dot(&errors_by_layer[layer].t());
    // Effectively averaging over the gradients of the batch members
    (1.0 / row_count as f64) * gradient
}

pub fn compute_gradient_of_cost_wrt_biases(
    layer: NonZeroUsize,
    errors_by_layer: &[Array2<f64>],
) -> Array1<f64> {
    errors_by_layer[layer.get()]
        .mean_axis(Axis(1))
        .unwrap()
        .to_owned()
}

fn compute_errors_by_layer(
    network: &NeuralNetwork,
    mini_batch: &MiniBatch,
    feedforward_result: &FeedForwardResult,
    activation_function: fn(f64) -> f64,
    cost_function: &impl CostFunction,
) -> Vec<Array2<f64>> {
    let mut errors = vec![];
    for layer in (1..=network.final_layer().get()).rev() {
        if layer == network.final_layer().get() {
            errors.push(compute_error_at_last_layer(
                feedforward_result,
                activation_function,
                cost_function,
                mini_batch,
            ));
            continue;
        }
        errors.push(propagate_error_back(
            network,
            feedforward_result,
            activation_function,
            NonZeroUsize::new(layer).unwrap(),
            errors.last().unwrap(),
        ));
    }
    errors.reverse();
    errors
}

fn propagate_error_back(
    network: &NeuralNetwork,
    feedforward_result: &FeedForwardResult,
    activation_function: fn(f64) -> f64,
    known_layer: NonZeroUsize,
    known_error: &Array2<f64>,
) -> Array2<f64> {
    let activation_derivatives_at_weighted_inputs =
        compute_activation_derivatives_at_weighted_inputs(
            &feedforward_result.activations[known_layer.get() - 1],
            activation_function,
        );
    network.weights(known_layer).t().dot(known_error) * activation_derivatives_at_weighted_inputs
}

fn compute_error_at_last_layer(
    feedforward_result: &FeedForwardResult,
    activation_function: fn(f64) -> f64,
    cost_function: &impl CostFunction,
    mini_batch: &MiniBatch,
) -> Array2<f64> {
    let activation_derivatives_at_weighted_inputs =
        compute_activation_derivatives_at_weighted_inputs(
            feedforward_result.weighted_inputs.last().unwrap(),
            activation_function,
        );
    let mut cost_gradients = feedforward_result.activations.last().unwrap().clone();
    cost_gradients
        .columns_mut()
        .into_iter()
        .enumerate()
        .for_each(|(i, mut x)| {
            x[i] =
                cost_function.partial_derivative(i, x.view(), mini_batch.targets.column(i).view())
        });
    cost_gradients * activation_derivatives_at_weighted_inputs
}

fn compute_activation_derivatives_at_weighted_inputs(
    weighted_inputs: &Array2<f64>,
    activation_function: fn(f64) -> f64,
) -> Array2<f64> {
    weighted_inputs.map(|x| derivative(activation_function, *x).unwrap())
}

#[cfg(test)]
mod tests {
    use crate::activation_functions::*;
    use crate::common::*;
    use crate::cost_functions::SSECostFunction;
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
        let feedforward_result = feed_forward(&network, identity, &mini_batch);
        let cost_function = SSECostFunction;
        let errors_by_layer = compute_errors_by_layer(
            &network,
            &mini_batch,
            &feedforward_result,
            sigmoid,
            &cost_function,
        );
        for layer in 1..=2 {
            let layer = NonZeroUsize::new(layer).unwrap();
            let cost_gradient_with_respect_to_weights = compute_gradient_of_cost_wrt_weights(
                &network,
                &mini_batch,
                &feedforward_result,
                layer,
                &errors_by_layer,
            );
            let cost_gradient_with_respect_to_biases =
                compute_gradient_of_cost_wrt_biases(layer, &errors_by_layer);
        }
    }
}
