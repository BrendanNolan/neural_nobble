use ndarray::{array, Array1, Array2};

fn main() {
    let vector: Array1<f32> = array![1.0, 2.0, 3.0];
    println!("Vector:\n{}", vector);

    let matrix: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    println!("Matrix:\n{}", matrix);

    let matrix_vector_product = matrix.dot(&vector);
    println!("Matrix * Vector:\n{}", matrix_vector_product);

    let matrix_matrix_product = matrix.dot(&matrix);
    println!("Matrix * Matrix:\n{}", matrix_matrix_product);
}
