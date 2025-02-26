use ndarray::*;
use neural_nobble::{
    activation_functions::ActivationFunction, neural_network::builder::NeuralNetworkBuilder,
};

fn main() {
    let network = NeuralNetworkBuilder::new(2)
        .add_layer(
            array![[0.1, 0.2], [0.3, 0.4]],
            array![0.5, 0.6],
            ActivationFunction::IdFunc,
        )
        .unwrap()
        .add_layer(array![[0.7, 0.8]], array![0.9], ActivationFunction::IdFunc)
        .unwrap()
        .build();
    println!("{:?}", network);
}
