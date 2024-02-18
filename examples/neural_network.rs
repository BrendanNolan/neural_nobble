use autodiff::*;
use ndarray::*;
use neural_nolan::neural_network::builder::NeuralNetworkBuilder;

fn sigmoid(x: F1) -> F1 {
    F1::var(1.0) / (F1::var(1.0) + (-x).exp())
}

fn main() {
    let network = NeuralNetworkBuilder::new(2, sigmoid)
        .add_layer(array![[0.1, 0.2], [0.3, 0.4]], array![0.5, 0.6])
        .unwrap()
        .add_layer(array![[0.7, 0.8]], array![0.9])
        .unwrap()
        .build();
    println!("{:?}", network);
}
