use rand::Rng;

use crate::{
    activation_functions, back_propagation, common::*, cost_functions::CostFunction,
    derivative::DifferentiableFunction, feed_forward::feed_forward, mini_batch::MiniBatch,
    neural_network::NeuralNetwork,
};
use std::collections::hash_set::HashSet;

pub fn train(
    network: &mut NeuralNetwork,
    activation_function: impl DifferentiableFunction,
    inputs: &Array2<f64>,
    targets: &Array2<f64>,
    cost_function: &impl CostFunction,
    batch_size: usize,
) {
    let mini_batch = create_minibatch(inputs, targets, batch_size);
    let mut finished = false;
    let feed_forward_result = feed_forward(network, activation_function, &mini_batch);
    let errors_by_layer = back_propagation::compute_errors_by_layer(
        network,
        &mini_batch,
        &feed_forward_result,
        activation_function,
        cost_function,
    );
    while !finished {
        let weight_gradients = (1..network.layer_count().get())
            .rev()
            .map(|layer| {
                back_propagation::compute_gradient_of_cost_wrt_weights(
                    network,
                    &feed_forward_result,
                    NonZeroUsize::new(layer).unwrap(),
                    &errors_by_layer,
                )
            })
            .collect::<Vec<_>>();
        let bias_gradients = (1..network.layer_count().get())
            .rev()
            .map(|layer| {
                back_propagation::compute_gradient_of_cost_wrt_biases(
                    NonZeroUsize::new(layer).unwrap(),
                    &errors_by_layer,
                )
            })
            .collect::<Vec<_>>();
        finished = true;
    }
}

fn create_minibatch(inputs: &Array2<f64>, targets: &Array2<f64>, size: usize) -> MiniBatch {
    let mut random_number_generator = rand::rng();
    let mut indices = HashSet::new();
    while indices.len() < size {
        indices.insert(random_number_generator.random_range(0..inputs.len()));
    }
    let mut batch_inputs = Array2::<f64>::zeros((row_count(inputs), size));
    let mut batch_targets = Array2::<f64>::zeros((row_count(inputs), size));
    for (batch_index, global_index) in indices.iter().enumerate() {
        batch_inputs
            .slice_mut(s![.., batch_index])
            .assign(&inputs.slice(s![.., *global_index]));
        batch_targets
            .slice_mut(s![.., batch_index])
            .assign(&targets.slice(s![.., *global_index]));
    }
    MiniBatch::create(batch_inputs, batch_targets)
}
