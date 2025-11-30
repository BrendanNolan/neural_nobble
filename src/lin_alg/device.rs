use crate::lin_alg::Dim;

pub struct DeviceVector {
    pub device: *mut f32,
    pub len: usize,
}

pub struct DeviceMatrix {
    pub device: *mut f32,
    pub dim: Dim,
}

impl DeviceMatrix {
    pub fn transpose(&mut self) {
        todo!(); // Call Out To C++
    }
}

pub fn multiply(a: &DeviceMatrix, b: &DeviceMatrix, result: &mut DeviceMatrix) {
    todo!(); // Call Out To C++
}
