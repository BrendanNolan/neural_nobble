use crate::ffi;
use std::ptr;

pub struct DeviceBlockAllocator {
    original_spot: *mut f32,
    next_spot: *mut f32,
    block_size: usize,
    mode: Mode,
}

pub enum Mode {
    Building,
    Built,
}

impl DeviceBlockAllocator {
    pub fn new() -> Self {
        Self {
            original_spot: ptr::null_mut(),
            next_spot: ptr::null_mut(),
            block_size: 0,
            mode: Mode::Building,
        }
    }

    pub fn allocate(&mut self, count: usize) -> Option<*mut f32> {
        match self.mode {
            Mode::Building => {
                self.block_size += count;
                Some(ffi::allocate_on_device(count))
            }
            Mode::Built => {
                if (self.can_allocate(count)) {
                    return None;
                }
                let next_available = self.next_spot;
                self.next_spot = unsafe { self.next_spot.add(count) };
                Some(next_available)
            }
        }
    }

    pub fn finalise(&mut self) {
        self.mode = Mode::Built;
        self.next_spot = ffi::allocate_on_device(self.block_size);
        self.original_spot = self.next_spot;
    }

    fn already_allocated(&self) -> usize {
        match self.mode {
            Mode::Building => self.block_size,
            Mode::Built => unsafe { self.next_spot.offset_from_unsigned(self.original_spot) },
        }
    }

    fn can_allocate(&self, count: usize) -> bool {
        count + self.already_allocated() <= self.block_size
    }
}
