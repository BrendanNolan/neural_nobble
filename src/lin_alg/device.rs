use crate::lin_alg::Dim;

pub struct DeviceVector {
    cuda: *mut f32,
    len: usize,
}

pub struct DeviceMatrix {
    cuda: *mut f32,
    dim: Dim,
}

impl DeviceMatrix {
    pub fn transpose(&mut self) {
        todo!(); // Call Out To C++
    }
}

pub fn multiply(a: &DeviceMatrix, b: &DeviceMatrix, result: &mut DeviceMatrix) {
    todo!(); // Call Out To C++
}
