use crate::lin_alg::Dim;
use std::ops::{Index, IndexMut};

pub type HostVector = Vec<f64>;

pub struct HostMatrix {
    raw: Vec<f64>,
    dim: Dim,
}

impl HostMatrix {
    fn zeroes(dim: Dim) -> Self {
        let raw = vec![0_f64; dim.rows * dim.cols];
        Self { raw, dim }
    }
}

impl Index<(usize, usize)> for HostMatrix {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.raw[i * self.dim.cols + j]
    }
}

impl IndexMut<(usize, usize)> for HostMatrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.raw[i * self.dim.cols + j]
    }
}
