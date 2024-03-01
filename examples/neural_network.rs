use ndarray::*;
use neural_nobble::{
    activation_functions::sigmoid, cost_functions::quadratic,
    neural_network::builder::NeuralNetworkBuilder,
};

fn main() {
    let network = NeuralNetworkBuilder::new(2, sigmoid, quadratic)
        .add_layer(array![[0.1, 0.2], [0.3, 0.4]], array![0.5, 0.6])
        .unwrap()
        .add_layer(array![[0.7, 0.8]], array![0.9])
        .unwrap()
        .build();
    println!("{:?}", network);
}
