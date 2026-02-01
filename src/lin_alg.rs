pub mod device;
pub mod host;

use crate::ffi::{allocate_on_device, copy_from_device, copy_to_device};
pub use device::{DeviceMatrix, DeviceVector};
use host::{HostMatrix, HostVector};

impl From<DeviceVector> for HostVector {
    fn from(value: DeviceVector) -> Self {
        let mut host: Self = vec![0 as f32; value.len];
        copy_from_device(value.data, value.len, &mut host);
        host
    }
}

impl From<HostVector> for DeviceVector {
    fn from(value: HostVector) -> Self {
        let device_data = allocate_on_device(value.len());
        copy_to_device(&value, device_data);
        Self {
            data: device_data,
            len: value.len(),
        }
    }
}

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
        assert!(host_matrix.almost_equal(&host_matrix_round));
    }

    #[test]
    fn test_round_trip_vectors() {
        let host_vector: HostVector = vec![0 as f32; 8];
        let device_vector = DeviceVector::from(&host_vector);
        let host_vector_round = HostVector::from(&device_vector);
        assert!(host_vector.almost_equal(&host_matrix_round));
    }
}
