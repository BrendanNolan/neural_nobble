use crate::{common::*, logging, neural_network::NeuralNetwork};

pub fn descend(
    weight_gradients: &[Array2<f32>],
    bias_gradients: &[Array1<f32>],
    network: &mut NeuralNetwork,
    learning_rate: f32,
) {
    let gradient_magnitude = gradient_magnitude(weight_gradients, bias_gradients);
    let adjustment_factor = -(learning_rate * gradient_magnitude);
    logging::log(&format!("Adjustment factor: {adjustment_factor}"));
    for (weight_gradient, weight) in weight_gradients
        .iter()
        .zip(network.weight_matrices_mut().iter_mut())
    {
        for row in 0..row_count(weight_gradient) {
            for col in 0..column_count(weight_gradient) {
                weight[(row, col)] += adjustment_factor * weight_gradient[(row, col)];
            }
        }
    }
    for (bias_gradient, bias) in bias_gradients
        .iter()
        .zip(network.bias_vectors_mut().iter_mut())
    {
        for index in 0..bias_gradient.len() {
            bias[index] += adjustment_factor * bias_gradient[index];
        }
    }
}

pub fn gradient_magnitude(weight_gradients: &[Array2<f32>], bias_gradients: &[Array1<f32>]) -> f32 {
    (weight_gradients.iter().map(sum_of_squares).sum::<f32>()
        + bias_gradients.iter().map(sum_of_squares).sum::<f32>())
    .sqrt()
}

fn assert_dimensions(weight_gradients: &[&Array2<f32>], bias_gradients: &[&Array1<f32>]) {
    assert!(weight_gradients.len() == bias_gradients.len());
    for (w, b) in weight_gradients.iter().zip(bias_gradients.iter()) {
        assert!(row_count(w) == b.len());
    }
}
