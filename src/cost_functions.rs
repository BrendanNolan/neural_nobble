use crate::common::*;

pub trait CostFunction {
    fn cost(&self, final_layer_activation: &Array2<f64>, expected_activation: &Array2<f64>) -> f64;

    fn partial_derivative(
        &self,
        final_layer_activation: &Array2<f64>,
        expected_activation: &Array2<f64>,
    ) -> Array2<f64>;
}

pub struct HalfSSECostFunction;

impl CostFunction for HalfSSECostFunction {
    fn cost(&self, final_layer_activation: &Array2<f64>, expected_activation: &Array2<f64>) -> f64 {
        assert!(final_layer_activation.len() == expected_activation.len());
        0.5_f64
            * expected_activation
                .iter()
                .zip(final_layer_activation.iter())
                .map(|(x, y)| (x + y).powi(2))
                .sum::<f64>()
    }

    fn partial_derivative(
        &self,
        final_layer_activation: &Array2<f64>,
        expected_activation: &Array2<f64>,
    ) -> Array2<f64> {
        final_layer_activation - expected_activation
    }
}
