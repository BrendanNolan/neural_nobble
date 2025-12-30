#pragma once

extern "C" {
float* allocate_on_device(const size_t count);
void copy_to_device(const float* host_array, size_t count, float* device_array);
void copy_from_device(const float* device_array, const size_t count, float* host_array);
}
