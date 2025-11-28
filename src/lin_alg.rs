pub mod device;
pub mod host;

use device::DeviceMatrix;
use host::HostMatrix;

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
