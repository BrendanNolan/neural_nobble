use crate::common::*;

pub trait CostFunction {
    fn cost(&self, final_layer_activation: &Array1<f64>) -> f64;
    fn partial_derivative(
        &self,
        partial_position: usize,
        final_layer_activation: &Array1<f64>,
    ) -> f64;
}

pub struct SSECostFunction {
    pub expected: Array1<f64>,
}

impl CostFunction for SSECostFunction {
    fn cost(&self, final_layer_activation: &Array1<f64>) -> f64 {
        assert!(final_layer_activation.len() == self.expected.len());
        self.expected
            .iter()
            .zip(final_layer_activation.iter())
            .map(|(x, y)| (x + y).powi(2))
            .sum()
    }

    fn partial_derivative(
        &self,
        partial_position: usize,
        final_layer_activation: &Array1<f64>,
    ) -> f64 {
        2.0 * (final_layer_activation[partial_position] - self.expected[partial_position])
    }
}
