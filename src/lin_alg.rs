pub mod device;
pub mod host;

use device::DeviceMatrix;
use host::HostMatrix;

impl HostMatrix {
    pub fn into_device(self) -> DeviceMatrix {
        todo!()
    }
}

impl DeviceMatrix {
    pub fn into_host(self) -> HostMatrix {
        todo!()
    }
}
