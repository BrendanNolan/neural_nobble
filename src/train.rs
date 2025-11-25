use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    activation_functions::*,
    back_propagation,
    common::*,
    cost_functions::{self, CostFunction},
    feed_forward::{feed_forward, print_details},
    gradient_descent::{descend, gradient_magnitude},
    mini_batch::MiniBatch,
    neural_network::NeuralNetwork,
};
use std::collections::hash_set::HashSet;

pub struct TrainingOptions<C: CostFunction> {
    pub cost_function: C,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub gradient_magnitude_stopping_criterion: f64,
    pub cost_difference_stopping_criterion: f64,
    pub epoch_limit: usize,
}

pub fn train<C: CostFunction>(
    network: &mut NeuralNetwork,
    inputs: &Array2<f64>,
    targets: &Array2<f64>,
    training_options: &TrainingOptions<C>,
) {
    let mut previous_cost: Option<f64> = None;
    #[cfg(feature = "neural_nobble_log")]
    {
        println!("Training begins __________");
    }
    let mut epoch_counter = 0;
    let mut descent_counter = 0;
    let mut random_number_generator = StdRng::seed_from_u64(155);
    // TODO: Can we reuse the vectors/matrices etc. from one loop iteration to the next? This would
    // cut down a lot on allocations, which are showing up in profiling.
    loop {
        #[cfg(feature = "neural_nobble_log")]
        {
            println!(
                "Epoch: {epoch_counter}. Descent: {descent_counter}. Weight and bias sum: {}",
                network.weight_and_bias_sum()
            );
        }
        let mini_batch = create_minibatch(
            inputs,
            targets,
            training_options.batch_size,
            &mut random_number_generator,
        );
        let feed_forward_result = feed_forward(network, &mini_batch);
        let errors_by_layer = back_propagation::compute_errors_by_layer(
            network,
            &mini_batch,
            &feed_forward_result,
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
        #[cfg(feature = "neural_nobble_log")]
        {
            print!("Cost: {cost}. Gradient magnitude: {pre_descent_gradient_magnitude} ");
        }
        if let Some(prev_cost) = previous_cost {
            let cost_reduction = prev_cost - cost;
            #[cfg(feature = "neural_nobble_log")]
            {
                println!(" Cost reduction: {cost_reduction}");
            }
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
        descent_counter += 1;
        epoch_counter = descent_counter / training_options.batch_size;
        if epoch_counter >= training_options.epoch_limit {
            break;
        }
    }
    #[cfg(feature = "neural_nobble_log")]
    {
        println!("__________ training ends.")
    }
}

fn create_minibatch(
    inputs: &Array2<f64>,
    targets: &Array2<f64>,
    size: usize,
    rng: &mut StdRng,
) -> MiniBatch {
    let mut indices = HashSet::new();
    while indices.len() < size {
        indices.insert(rng.random_range(0..column_count(inputs)));
    }
    let mut batch_inputs = Array2::<f64>::zeros((row_count(inputs), size));
    let mut batch_targets = Array2::<f64>::zeros((row_count(targets), size));
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
            ActivationFunction::Sigmoid,
        )
        .unwrap()
        .add_layer(
            arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            arr1(&[0.5, 0.5, 0.5]),
            ActivationFunction::Sigmoid,
        )
        .unwrap()
        .add_layer(
            arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            arr1(&[0.5, 0.5, 0.5]),
            ActivationFunction::Sigmoid,
        )
        .unwrap()
        .add_layer(
            arr2(&[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            arr1(&[0.5, 0.5]),
            ActivationFunction::Sigmoid,
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
        epoch_limit: 1000,
    };
    let activation = ActivationFunction::Sigmoid;
    train(&mut network, &inputs, &targets, &training_options);
    let feed_forward = feed_forward(
        &network,
        &MiniBatch {
            inputs: inputs.clone(),
            targets: targets.clone(),
        },
    );
    print_details(&feed_forward, &targets, 2);
}
