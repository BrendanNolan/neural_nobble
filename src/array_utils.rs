use ndarray::Array2;

pub fn row_count(matrix: &Array2<f64>) -> usize {
    matrix.dim().0
}

pub fn column_count(matrix: &Array2<f64>) -> usize {
    matrix.dim().1
}
