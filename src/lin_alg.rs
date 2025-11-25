pub mod device;
pub mod host;

use device::DeviceMatrix;
use host::HostMatrix;

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
