#[repr(C)]
struct Dim3 {
    x: u32,
    y: u32,
    z: u32,
}

mod inner {
    extern "C" {
        pub fn transfer_to_cuda(host_array: *const f32, count: usize) -> *mut f32;
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

pub fn transfer_to_cuda(elements: &[f32]) -> *mut f32 {
    unsafe { inner::transfer_to_cuda(elements.as_ptr(), elements.len()) }
}

pub fn transfer_from_cuda(device_array: *const f32, count: usize) -> Vec<f32> {
    unsafe { Vec::from_raw_parts(inner::transfer_from_cuda(device_array, count), count, count) }
}
