use autodiff::F1;

pub fn identity(x: F1) -> F1 {
    x
}

pub fn sigmoid(x: F1) -> F1 {
    F1::var(1.0) / (F1::var(1.0) + (-x).x.exp())
}
