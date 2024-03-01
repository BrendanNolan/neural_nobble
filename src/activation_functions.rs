pub fn identity(x: F1) -> F1 {
    x
}

pub fn sigmoid(x: F1) -> F1 {
    F1::var(1.0) / (F1::var(1.0) + (-x).x.exp())
}

pub fn sigmoid_derivative(x: F1) -> F1 {
    let sig = sigmoid(x);
    sig * (F1::var(1.0) - sig)
}
