use crate::common::*;

// TODO: needs to report details about the magnitude of the new
// gradient and the difference in the cost between the pre-
// and post- descended weights and biases
pub fn descend(
    weight_gradients: &[&Array2<f64>],
    bias_gradients: &[&Array1<f64>],
    weights: &mut [&mut Array2<f64>],
    biases: &mut [&mut Array1<f64>],
    learning_rate: f64,
) {
    let gradient_dim = weight_gradients.iter().map(|m| m.len()).sum::<usize>()
        + bias_gradients.iter().map(|m| m.len()).sum::<usize>();
    let gradient_magnitude = (1.0 / gradient_dim as f64)
        * (weight_gradients
            .iter()
            .map(|m| sum_of_squares(m))
            .sum::<f64>()
            + bias_gradients
                .iter()
                .map(|m| sum_of_squares(m))
                .sum::<f64>())
        .sqrt();
    let adjustment_factor = -(learning_rate * gradient_magnitude);
    for (weight_gradient, weight) in weight_gradients.iter().zip(weights.iter_mut()) {
        for row in 0..row_count(weight_gradient) {
            for col in 0..column_count(weight_gradient) {
                weight[(row, col)] = adjustment_factor * weight_gradient[(row, col)];
            }
        }
    }
    for (bias_gradient, bias) in bias_gradients.iter().zip(biases.iter_mut()) {
        for index in 0..bias_gradient.len() {
            bias[index] = adjustment_factor * bias_gradient[index];
        }
    }
}

fn assert_dimensions(weight_gradients: &[&Array2<f64>], bias_gradients: &[&Array1<f64>]) {
    assert!(weight_gradients.len() == bias_gradients.len());
    for (w, b) in weight_gradients.iter().zip(bias_gradients.iter()) {
        assert!(row_count(w) == b.len());
    }
}
