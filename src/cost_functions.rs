use crate::common::*;

// TODO: Have this take a feedforward result; the FeedForwardResult type contains
// final layer activation and final layer weighted input - SSE uses the former,
// CrossEntropy will use the latter when I implement it correctly.
pub trait CostFunction: Copy + Clone {
    fn cost(&self, final_layer_activation: &Array2<f64>, expected_activation: &Array2<f64>) -> f64;

    fn partial_derivative(
        &self,
        final_layer_activation: &Array2<f64>,
        expected_activation: &Array2<f64>,
    ) -> Array2<f64>;
}

#[derive(Copy, Clone)]
pub struct HalfSSECostFunction;

impl CostFunction for HalfSSECostFunction {
    fn cost(&self, final_layer_activation: &Array2<f64>, expected_activation: &Array2<f64>) -> f64 {
        let number_of_examples = column_count(final_layer_activation);
        assert!(final_layer_activation.dim() == expected_activation.dim());
        (1.0 / number_of_examples as f64)
            * 0.5_f64
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

#[derive(Default, Copy, Clone)]
pub struct CrossEntropyCost;

impl CostFunction for CrossEntropyCost {
    fn cost(&self, final_layer_activation: &Array2<f64>, expected_activation: &Array2<f64>) -> f64 {
        let number_of_examples = column_count(final_layer_activation);
        assert!(final_layer_activation.dim() == expected_activation.dim());
        -(1.0 / number_of_examples as f64)
            * final_layer_activation
                .iter()
                .zip(expected_activation.iter())
                .map(|(a, y)| y * a.ln() + (1.0 - y) * (1.0 - a).ln())
                .sum::<f64>()
    }

    fn partial_derivative(
        &self,
        final_layer_activation: &Array2<f64>,
        expected_activation: &Array2<f64>,
    ) -> Array2<f64> {
        let mut result = final_layer_activation.clone();
        result
            .iter_mut()
            .zip(expected_activation.iter())
            .for_each(|(a, y)| *a = y * (1.0 / *a) + (1.0 - y) * -1.0 / (1.0 - *a));
        result
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
    let cost_func = HalfSSECostFunction;
    let cost = cost_func.cost(&final_layer_activation, &expected_activation);
    println!("Cost: {cost}");
}
