pub mod device;
pub mod host;

use crate::ffi::{allocate_on_device, copy_to_device};
pub use device::{DeviceMatrix, DeviceVector};
use host::{HostMatrix, HostVector};

impl From<DeviceVector> for HostVector {
    fn from(value: DeviceVector) -> Self {
        todo!();
    }
}

impl From<HostVector> for DeviceVector {
    fn from(value: HostVector) -> Self {
        todo!();
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
        todo!()
    }
}

impl From<&HostMatrix> for DeviceMatrix {
    fn from(value: &HostMatrix) -> Self {
        let device_data = allocate_on_device(value.dim.size());
        copy_to_device(&value.data, device_data);
        Self {
            device: device_data,
            dim: value.dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_round_trip() {
        let host_matrix = HostMatrix::zeroes(Dim::new(8, 8));
        let device_matrix = DeviceMatrix::from(&host_matrix);
        let host_matrix_round = HostMatrix::from(&device_matrix);
        assert_eq!(host_matrix, host_matrix_round);
    }
}
