# Overview

A from-scratch neural-network implementation, in Rust. Only dependencies are a matrix-multiplication
library and a random-number generator.

# General Usage

The following is a brief example of the functionality provided:

```Rust
let mut network = builder::NeuralNetworkBuilder::new(image_size)
    .add_layer_random(32, ActivationFunction::Relu)
    .unwrap()
    .add_layer_random(32, ActivationFunction::Relu)
    .unwrap()
    .add_layer_random(10, ActivationFunction::SoftMax)
    .unwrap()
    .build();

let training_options = TrainingOptions {
    cost_function: CrossEntropyCost,
    batch_size: 64,
    learning_rate: 0.02,
    gradient_magnitude_stopping_criterion: 0.0001,
    cost_difference_stopping_criterion: 0.0001,
    epoch_limit: 100,
};

train(
    &mut network,
    &train_data,
    &train_labels_one_hot_encoded,
    &training_options,
);

let whole_test_data = MiniBatch {
    inputs: test_data.clone(),
    targets: test_labels_one_hot_encoded.clone(),
};
let feed_forward_result = feed_forward(&network, &whole_test_data);
let prediction_matrix = feed_forward_result.activations.last().unwrap();
let cost = training_options.cost_function.cost(
    feed_forward_result.activations.last().unwrap(),
    &whole_test_data.targets,
);
```
