use crate::lin_alg::Dim;

pub struct DeviceMatrix {
    cuda: *mut f64,
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
