use crate::ffi;
use std::ptr;

pub struct DeviceArenaAllocator {
    original_spot: *mut f32,
    next_spot: *mut f32,
    arena_size: usize,
    mode: Mode,
}

pub enum Mode {
    Building,
    Built,
}

impl DeviceArenaAllocator {
    pub fn initialise() -> Self {
        Self {
            original_spot: ptr::null_mut(),
            next_spot: ptr::null_mut(),
            arena_size: 0,
            mode: Mode::Building,
        }
    }

    pub fn allocate(&mut self, count: usize) -> Option<*mut f32> {
        match self.mode {
            Mode::Building => {
                self.arena_size += count;
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
        self.finalise_padded(1.0);
    }

    pub fn finalise_padded(&mut self, padding_factor: f32) {
        self.mode = Mode::Built;
        let allocation_size = (self.arena_size as f32 * padding_factor) as usize;
        self.next_spot = ffi::allocate_on_device(allocation_size);
        self.original_spot = self.next_spot;
    }

    fn already_allocated(&self) -> usize {
        match self.mode {
            Mode::Building => self.arena_size,
            Mode::Built => unsafe { self.next_spot.offset_from_unsigned(self.original_spot) },
        }
    }

    fn can_allocate(&self, count: usize) -> bool {
        count + self.already_allocated() <= self.arena_size
    }
}
