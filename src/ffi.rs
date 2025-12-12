use crate::lin_alg::{DeviceMatrix, DeviceVector};

#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

mod inner {
    extern "C" {
        pub fn allocate_on_cuda(host_array: *const f32, count: usize) -> *mut f32;
        pub fn transfer_from_cuda(device_array: *const f32, count: usize) -> *mut f32;
        pub fn launch_tiled_multiply(
            A: *const f32,
            ai: u32,
            aj: u32,
            B: *const f32,
            bj: u32,
            C: *const f32,
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

pub fn allocate_on_cuda(elements: &[f32]) -> *mut f32 {
    unsafe { inner::allocate_on_cuda(elements.as_ptr(), elements.len()) }
}

pub fn transfer_from_cuda(device_array: *const f32, count: usize) -> Vec<f32> {
    unsafe { Vec::from_raw_parts(inner::transfer_from_cuda(device_array, count), count, count) }
}

pub fn launch_tiled_multiply(
    a: &DeviceMatrix,
    b: &DeviceMatrix,
    c: &mut DeviceMatrix,
    launch_config: LaunchConfig,
) {
    let LaunchConfig {
        grid,
        block,
        shared_mem_size,
    } = launch_config;
    unsafe {
        inner::launch_tiled_multiply(
            a.device,
            a.dim.rows as u32,
            a.dim.cols as u32,
            b.device,
            b.dim.cols as u32,
            c.device,
            grid,
            block,
            shared_mem_size,
        );
    }
}
