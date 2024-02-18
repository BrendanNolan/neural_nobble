use autodiff::*;

fn sigmoid(x: F1) -> F1 {
    F1::var(1.0) / (F1::var(1.0) + (-x).exp())
}

fn main() {
    let x = F1::var(2.0);
    let sig_x = sigmoid(x);
    println!("Sigmoid value: {}", sig_x.value());
    println!("Derivative of the sigmoid at x=2.0: {}", sig_x.deriv());
}
