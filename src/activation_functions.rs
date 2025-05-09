use crate::{common::*, distribution};

#[derive(Copy, Clone, Debug)]
pub enum ActivationFunction {
    Id,
    Sigmoid,
    Relu,
    SoftMax,
}

impl ActivationFunction {
    pub fn apply(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        match *self {
            ActivationFunction::Id => weighted_inputs.clone(),
            ActivationFunction::Sigmoid => {
                let elems: Vec<_> = weighted_inputs
                    .iter()
                    .map(|x| 1.0 / (1.0 + (-*x).exp()))
                    .collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
            ActivationFunction::Relu => {
                let elems: Vec<_> = weighted_inputs.iter().map(|x| x.max(0.0)).collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
            ActivationFunction::SoftMax => {
                let col_exp_sums: Vec<f64> = weighted_inputs
                    .columns()
                    .into_iter()
                    .map(|col| col.iter().map(|x| x.exp()).sum())
                    .collect();
                Array2::<f64>::from_shape_fn(weighted_inputs.dim(), |(i, j)| {
                    weighted_inputs[(i, j)].exp() * (1_f64 / col_exp_sums[j])
                })
            }
        }
    }

    pub fn derivative(&self, weighted_inputs: &Array2<f64>) -> Array2<f64> {
        match *self {
            ActivationFunction::Id => Array2::<f64>::from_elem(weighted_inputs.dim(), 1.0),
            ActivationFunction::Sigmoid => {
                let elems: Vec<_> = weighted_inputs.iter().map(|x| *x * (1.0 - *x)).collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
            ActivationFunction::Relu => {
                let elems: Vec<_> = weighted_inputs
                    .iter()
                    .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
                    .collect();
                Array2::<f64>::from_shape_vec(weighted_inputs.dim(), elems).unwrap()
            }
            ActivationFunction::SoftMax => {
                let softmax = self.apply(weighted_inputs);
                let ones = Array2::<f64>::from_elem(weighted_inputs.dim(), 1_f64);
                &softmax * (ones - &softmax)
            }
        }
    }

    pub fn suggested_distribution(
        &self,
        prev_layer_neuron_count: usize,
        neuron_count: usize,
    ) -> distribution::Distribution {
        match *self {
            ActivationFunction::Id => distribution::Distribution::Normal {
                mean: 0.0,
                standard_deviation: 1.0,
            },
            ActivationFunction::Sigmoid => {
                xavier_glorot_normal(1.0, prev_layer_neuron_count, neuron_count)
            }
            ActivationFunction::Relu => {
                xavier_glorot_normal(2.0, prev_layer_neuron_count, neuron_count)
            }
            ActivationFunction::SoftMax => {
                xavier_glorot_uniform(prev_layer_neuron_count, neuron_count)
            }
        }
    }
}

fn xavier_glorot_normal(
    denominator: f64,
    prev_layer_neuron_count: usize,
    neuron_count: usize,
) -> distribution::Distribution {
    distribution::Distribution::Normal {
        mean: 0.0,
        standard_deviation: (denominator / prev_layer_neuron_count as f64).sqrt(),
    }
}

fn xavier_glorot_uniform(
    prev_layer_neuron_count: usize,
    neuron_count: usize,
) -> distribution::Distribution {
    let neurons_in_and_out = prev_layer_neuron_count + neuron_count;
    let range_arond_zero = (6_f64 / neurons_in_and_out as f64).sqrt();
    distribution::Distribution::Uniform {
        lower_bound: -range_arond_zero,
        upper_bound: range_arond_zero,
    }
}
