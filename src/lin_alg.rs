use std::ops::{Index, IndexMut};

use crate::ffi::{allocate_on_device, copy_from_device, copy_to_device};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim {
    pub rows: usize,
    pub columns: usize,
}

impl Dim {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self { rows, columns }
    }

    pub fn size(&self) -> usize {
        self.rows * self.columns
    }
}

pub fn floats_almost_equal(a: f32, b: f32) -> bool {
    const TOLERANCE: f32 = 0.000001_f32;
    (a - b).abs() < TOLERANCE
}

pub type HostVector = Vec<f32>;

pub fn host_vectors_almost_equal(a: &HostVector, b: &HostVector) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(x, y)| floats_almost_equal(*x, *y))
}

#[derive(Clone, Debug)]
pub struct HostMatrix {
    pub data: HostVector,
    pub dim: Dim,
}

impl HostMatrix {
    pub fn zeroes(dim: Dim) -> Self {
        let data = vec![0_f32; dim.rows * dim.columns];
        Self { data, dim }
    }
}

pub fn host_matrices_almost_equal(a: &HostMatrix, b: &HostMatrix) -> bool {
    a.dim == b.dim && host_vectors_almost_equal(&a.data, &b.data)
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

impl From<&DeviceVector> for HostVector {
    fn from(value: &DeviceVector) -> Self {
        let mut host: Self = vec![0 as f32; value.len];
        copy_from_device(value.data, value.len, &mut host);
        host
    }
}

impl From<&DeviceMatrix> for HostMatrix {
    fn from(value: &DeviceMatrix) -> Self {
        let mut host_matrix = Self {
            data: vec![0.0; value.dim.size()],
            dim: value.dim,
        };
        copy_from_device(value.data, value.dim.size(), &mut host_matrix.data);
        host_matrix
    }
}

pub struct DeviceVector {
    pub data: *mut f32,
    pub len: usize,
}

impl DeviceVector {}

pub struct DeviceMatrix {
    pub data: *mut f32,
    pub dim: Dim,
}

impl DeviceMatrix {
    pub fn dot_to_new(&self, other: &DeviceMatrix) -> DeviceMatrix {
        todo!();
    }

    pub fn dot_to_existing(&self, other: &DeviceMatrix, result: &mut DeviceMatrix) {
        todo!();
    }
}

impl From<&HostVector> for DeviceVector {
    fn from(value: &HostVector) -> Self {
        let device_data = allocate_on_device(value.len());
        copy_to_device(value, device_data);
        Self {
            data: device_data,
            len: value.len(),
        }
    }
}

impl From<&HostMatrix> for DeviceMatrix {
    fn from(value: &HostMatrix) -> Self {
        let device_data = allocate_on_device(value.dim.size());
        copy_to_device(&value.data, device_data);
        Self {
            data: device_data,
            dim: value.dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_round_trip_matrices() {
        let host_matrix = HostMatrix::zeroes(Dim::new(8, 8));
        let device_matrix = DeviceMatrix::from(&host_matrix);
        let host_matrix_round = HostMatrix::from(&device_matrix);
        assert!(host_matrices_almost_equal(&host_matrix, &host_matrix_round));
    }

    #[test]
    fn test_round_trip_vectors() {
        let host_vector: HostVector = vec![0 as f32; 8];
        let device_vector = DeviceVector::from(&host_vector);
        let host_vector_round = HostVector::from(&device_vector);
        assert!(host_vectors_almost_equal(&host_vector, &host_vector_round));
    }
}
