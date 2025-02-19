use rand::Rng;

use crate::{
    activation_functions::{self, SigmoidFunc},
    back_propagation,
    common::*,
    cost_functions::{self, CostFunction},
    derivative::DifferentiableFunction,
    feed_forward::feed_forward,
    gradient_descent::{descend, gradient_magnitude},
    mini_batch::MiniBatch,
    neural_network::NeuralNetwork,
};
use std::collections::hash_set::HashSet;

pub struct TrainingOptions<C: CostFunction> {
    cost_function: C,
    batch_size: usize,
    learning_rate: f64,
    gradient_magnitude_stopping_criterion: f64,
    cost_difference_stopping_criterion: f64,
}

pub fn train<C: CostFunction, D: DifferentiableFunction>(
    network: &mut NeuralNetwork,
    activation_function: D,
    inputs: &Array2<f64>,
    targets: &Array2<f64>,
    training_options: &TrainingOptions<C>,
) {
    let mut previous_cost: Option<f64> = None;
    loop {
        let mini_batch = create_minibatch(inputs, targets, training_options.batch_size);
        let feed_forward_result = feed_forward(network, activation_function, &mini_batch);
        let errors_by_layer = back_propagation::compute_errors_by_layer(
            network,
            &mini_batch,
            &feed_forward_result,
            activation_function,
            &training_options.cost_function,
        );
        let cost = training_options.cost_function.cost(
            feed_forward_result.activations.last().unwrap(),
            &mini_batch.targets,
        );
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
            .rev()
            .collect::<Vec<_>>();
        let bias_gradients = (1..network.layer_count().get())
            .rev()
            .map(|layer| {
                back_propagation::compute_gradient_of_cost_wrt_biases(
                    NonZeroUsize::new(layer).unwrap(),
                    &errors_by_layer,
                )
            })
            .rev()
            .collect::<Vec<_>>();
        let pre_descent_gradient_magnitude = gradient_magnitude(&weight_gradients, &bias_gradients);
        if let Some(prev_cost) = previous_cost {
            let cost_reduction = prev_cost - cost;
            if cost_reduction > 0.0
                && cost_reduction < training_options.cost_difference_stopping_criterion
                && pre_descent_gradient_magnitude
                    < training_options.gradient_magnitude_stopping_criterion
            {
                break;
            }
        }
        previous_cost = Some(cost);
        descend(
            &weight_gradients,
            &bias_gradients,
            network,
            training_options.learning_rate,
        );
    }
}

fn create_minibatch(inputs: &Array2<f64>, targets: &Array2<f64>, size: usize) -> MiniBatch {
    let mut random_number_generator = rand::rng();
    let mut indices = HashSet::new();
    while indices.len() < size {
        indices.insert(random_number_generator.random_range(0..column_count(inputs)));
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

#[cfg(test)]
use crate::neural_network::builder;

#[test]
fn test_training() {
    let mut network = builder::NeuralNetworkBuilder::new(2)
        .add_layer(
            arr2(&[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]),
            arr1(&[0.5, 0.5, 0.5]),
        )
        .unwrap()
        .add_layer(
            arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            arr1(&[0.5, 0.5, 0.5]),
        )
        .unwrap()
        .add_layer(
            arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            arr1(&[0.5, 0.5, 0.5]),
        )
        .unwrap()
        .add_layer(
            arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            arr1(&[0.5, 0.5]),
        )
        .unwrap()
        .build();
    let inputs = arr2(&[[2.0, 2.0], [3.0, 3.0]]);
    let targets = arr2(&[[1.0, 1.0], [0.0, 0.0]]);
    let training_options = TrainingOptions {
        cost_function: cost_functions::HalfSSECostFunction,
        batch_size: 2,
        learning_rate: 0.001,
        gradient_magnitude_stopping_criterion: 0.000001,
        cost_difference_stopping_criterion: 0.000001,
    };
    train(
        &mut network,
        SigmoidFunc::default(),
        &inputs,
        &targets,
        &training_options,
    );
}
