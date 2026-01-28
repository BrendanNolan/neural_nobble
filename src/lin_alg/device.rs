use crate::lin_alg::Dim;

pub struct DeviceVector {
    pub data: *mut f32,
    pub len: usize,
}

pub struct DeviceMatrix {
    pub data: *mut f32,
    pub dim: Dim,
}
