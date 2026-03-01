#pragma once

extern "C" {
float* allocate_on_device(const size_t count);
void copy_to_device(const float* host_array, size_t count, float* device_array);
void copy_from_device(const float* device_array, const size_t count, float* host_array);
}

namespace cuda_helpers {
template <typename T>
__device__ void swap(T& a, T& b) {
    const auto tmp = b;
    b = a;
    a = tmp;
}

__forceinline__ __device__ float max(const float a, const float b) {
    return a < b ? b : a;
}

__forceinline__ __device__ float min(const float a, const float b) {
    return a > b ? b : a;
}
}// namespace cuda_helpers
