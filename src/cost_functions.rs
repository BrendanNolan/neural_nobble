use crate::common::*;

pub trait CostFunction {
    fn cost(
        &self,
        final_layer_activation: ArrayView1<f64>,
        expected_activation: ArrayView1<f64>,
    ) -> f64;
    fn partial_derivative(
        &self,
        partial_position: usize,
        final_layer_activation: ArrayView1<f64>,
        expected_activation: ArrayView1<f64>,
    ) -> f64;
}

pub struct SSECostFunction;

impl CostFunction for SSECostFunction {
    fn cost(
        &self,
        final_layer_activation: ArrayView1<f64>,
        expected_activation: ArrayView1<f64>,
    ) -> f64 {
        assert!(final_layer_activation.len() == expected_activation.len());
        expected_activation
            .iter()
            .zip(final_layer_activation.iter())
            .map(|(x, y)| (x + y).powi(2))
            .sum()
    }

    fn partial_derivative(
        &self,
        partial_position: usize,
        final_layer_activation: ArrayView1<f64>,
        expected_activation: ArrayView1<f64>,
    ) -> f64 {
        2.0 * (final_layer_activation[partial_position] - expected_activation[partial_position])
    }
}
