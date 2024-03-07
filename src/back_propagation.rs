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
    let (rows, columns) = network.weights(layer).dim();
    let mut gradient = Array2::zeros((rows, columns));
    let layer = layer.get();
    for (j, k) in (0..rows).zip(0..columns) {
        gradient[(j, k)] = feedforward_result.activations[layer - 1]
            .row(k)
            .dot(&errors_by_layer[layer].column(j));
    }
    // Effectively averaging over the gradients of the batch members
    (1.0 / rows as f64) * gradient
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
    }
    todo!("Finish");
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
