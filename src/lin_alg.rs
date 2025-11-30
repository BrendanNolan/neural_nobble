pub mod device;
pub mod host;

use device::{DeviceMatrix, DeviceVector};
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

pub struct Dim {
    pub rows: usize,
    pub cols: usize,
}

impl From<DeviceMatrix> for HostMatrix {
    fn from(value: DeviceMatrix) -> Self {
        todo!()
    }
}

impl From<HostMatrix> for DeviceMatrix {
    fn from(value: HostMatrix) -> Self {
        todo!()
    }
}
