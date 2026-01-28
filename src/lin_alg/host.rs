use crate::lin_alg::Dim;
use std::ops::{Index, IndexMut};

pub type HostVector = Vec<f32>;

#[derive(Clone, Debug)]
pub struct HostMatrix {
    pub data: Vec<f32>,
    pub dim: Dim,
}

impl HostMatrix {
    pub fn zeroes(dim: Dim) -> Self {
        let data = vec![0_f32; dim.rows * dim.columns];
        Self { data, dim }
    }

    pub fn almost_equal(&self, other: &HostMatrix) -> bool {
        const TOLERANCE: f32 = 0.000001;
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(x, y)| (x - y).abs() < TOLERANCE)
    }
}

impl Index<(usize, usize)> for HostMatrix {
    type Output = f32;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[i * self.dim.columns + j]
    }
}

impl IndexMut<(usize, usize)> for HostMatrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.data[i * self.dim.columns + j]
    }
}
