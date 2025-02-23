use crate::common::*;

pub fn one_hot_encode(row_matrix: &Array1<u8>, limit: u8) -> Array2<u8> {
    let col_count = row_matrix.dim();
    let mut one_hot = Array2::<u8>::zeros((limit as usize, col_count));
    for col in 0..col_count {
        let number = row_matrix[col];
        one_hot[(number as usize, col)] = 1;
    }
    one_hot
}

#[cfg(test)]
use super::*;

#[test]
fn test_one_hot_encode() {
    let row_matrix = array![2, 0, 1, 3];
    let limit = 4;
    let expected = array![[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]];
    let result = one_hot_encode(&row_matrix, limit);
    assert_eq!(result, expected);
}
