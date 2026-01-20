use crate::lin_alg::{DeviceMatrix, DeviceVector};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Op {
    Identity = 0,
    Transpose = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct ConstMatrixDetails {
    data: *const f32,
    rows: u32,
    columns: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct MutableMatrixDetails {
    data: *mut f32,
    rows: u32,
    columns: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct GemmParams {
    a: ConstMatrixDetails,
    op_a: Op,
    alpha: f32,
    b: ConstMatrixDetails,
    op_b: Op,
    beta: f32,
    c: *mut f32,
}

mod inner {
    #[link(name = "cuda_lin_alg")]
    extern "C" {
        pub fn allocate_on_device(count: usize) -> *mut f32;
        pub fn copy_to_device(host_array: *const f32, count: usize, device_memory: *mut f32);
        pub fn copy_from_device(device_array: *const f32, count: usize, host_array: *mut f32);
        pub fn launch_tiled_multiply(
            params: super::GemmParams,
            grid: super::Dim3,
            block: super::Dim3,
            shared_mem_size: u32,
        );
    }
}

pub struct LaunchConfig {
    grid: Dim3,
    block: Dim3,
    shared_mem_size: u32,
}

pub fn allocate_on_device(count: usize) -> *mut f32 {
    unsafe { inner::allocate_on_device(count) }
}

pub fn copy_to_device(elements: &[f32], device_memory: *mut f32) {
    unsafe {
        inner::copy_to_device(elements.as_ptr(), elements.len(), device_memory);
    }
}

pub fn copy_from_device(device_array: *const f32, count: usize, host_array: *mut f32) {
    unsafe {
        inner::copy_from_device(device_array, count, host_array);
    }
}

pub struct DeviceGemmParams<'a> {
    a: &'a DeviceMatrix,
    op_a: Op,
    alpha: f32,
    b: &'a DeviceMatrix,
    op_b: Op,
    beta: f32,
    c: &'a mut DeviceMatrix,
    launch_config: LaunchConfig,
}

pub fn launch_gpu_gemm(params: DeviceGemmParams) {
    let LaunchConfig {
        grid,
        block,
        shared_mem_size,
    } = params.launch_config;
    let a = ConstMatrixDetails {
        data: params.a.device,
        rows: params.a.dim.rows as u32,
        columns: params.a.dim.columns as u32,
    };
    let b = ConstMatrixDetails {
        data: params.b.device,
        rows: params.b.dim.rows as u32,
        columns: params.b.dim.columns as u32,
    };
    let params = GemmParams {
        a,
        op_a: params.op_a,
        alpha: params.alpha,
        b,
        op_b: params.op_b,
        beta: params.beta,
        c: params.c.device,
    };
    unsafe {
        inner::launch_tiled_multiply(params, grid, block, shared_mem_size);
    }
}

#[derive(Default)]
pub struct DeviceMemoryPoolBuilder {
    slots: Vec<(*mut f32, usize)>,
}

impl DeviceMemoryPoolBuilder {
    pub fn create_slot(&mut self, count: usize) -> *mut f32 {
        let device_memory = allocate_on_device(count);
        self.slots.push((device_memory, count));
        device_memory
    }

    pub fn build(self) -> DeviceMemoryPool {
        DeviceMemoryPool {
            slots: self.slots,
            index: 0,
        }
    }
}

pub struct DeviceMemoryPool {
    slots: Vec<(*mut f32, usize)>,
    index: usize,
}

impl DeviceMemoryPool {
    pub fn next(&mut self) -> Option<(*mut f32, usize)> {
        if self.index >= self.slots.len() {
            return None;
        }
        let prev_index = self.index;
        self.index += 1;
        Some(self.slots[prev_index])
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_ffi_matrix_multiplication() {
        allocate_on_device(147);
    }
}
