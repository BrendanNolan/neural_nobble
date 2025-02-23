use crate::common::*;

pub trait CostFunction: Copy + Clone {
    fn cost(&self, final_layer_activation: &Array2<f64>, expected_activation: &Array2<f64>) -> f64;

    fn partial_derivative(
        &self,
        final_layer_activation: &Array2<f64>,
        expected_activation: &Array2<f64>,
    ) -> Array2<f64>;
}

#[derive(Copy, Clone)]
pub struct SSECostFunction;

impl CostFunction for SSECostFunction {
    fn cost(&self, final_layer_activation: &Array2<f64>, expected_activation: &Array2<f64>) -> f64 {
        let number_of_examples = column_count(final_layer_activation);
        assert!(final_layer_activation.dim() == expected_activation.dim());
        (1.0 / number_of_examples as f64)
            * expected_activation
                .iter()
                .zip(final_layer_activation.iter())
                .map(|(x, y)| (x - y).powi(2))
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

#[cfg(test)]
use super::*;

#[test]
fn test_half_sse_cost() {
    let final_layer_activation = Array::from_shape_vec(
        (10, 2),
        vec![
            0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1,
            0.2, 0.1, 0.2,
        ],
    )
    .unwrap();
    let expected_activation = Array::from_shape_vec(
        (10, 2),
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ],
    )
    .unwrap();
    let cost_func = SSECostFunction;
    let cost = cost_func.cost(&final_layer_activation, &expected_activation);
    println!("Cost: {cost}");
}
