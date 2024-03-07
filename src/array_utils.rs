use ndarray::{Array, Array2, Dimension};

pub fn row_count(matrix: &Array2<f64>) -> usize {
    matrix.dim().0
}

pub fn column_count(matrix: &Array2<f64>) -> usize {
    matrix.dim().1
}

pub fn sum_of_squares<D: ndarray::Dimension>(matrix: &Array<f64, D>) -> f64 {
    matrix.iter().map(|f| f * f).sum()
}
