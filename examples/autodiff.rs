use autodiff::*;
use ndarray::*;

fn sigmoid(x: F1) -> F1 {
    F1::var(1.0) / (F1::var(1.0) + (-x).exp())
}

fn run_sigmoid_example() {
    let x = F1::var(2.0);
    let sig_x = sigmoid(x);
    println!("Sigmoid value: {}", sig_x.value());
    println!("Derivative of the sigmoid at x=2.0: {}", sig_x.deriv());
}

fn sum_of_squares(x: &Array1<F1>, y: &Array1<F1>) -> F1 {
    x.iter()
        .zip(y.iter())
        .map(|(x, y)| (*x - *y).powi(2))
        .fold(F1::var(0.0), |acc, x| acc + x)
}

fn run_sum_of_squares_example() {
    let x = array![F1::var(1.0), F1::var(2.0), F1::var(3.0)];
    let y = array![F1::var(4.0), F1::var(5.0), F1::var(6.0)];
    let result = sum_of_squares(&x, &y);
    println!("Sum of squares: {}", result.value());
    println!("Gradient of the sum of squares: {}", grad(&result, &x));
}

fn main() {
    run_sigmoid_example();
    run_sum_of_squares_example();
}
