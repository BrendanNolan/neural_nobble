#include "cuda_utils.h"

#include <cuda_runtime.h>

float* allocate_on_device(const size_t count) {
    float* device_array;
    cudaMalloc(&device_array, count * sizeof(float));
    return device_array;
}

void copy_to_device(const float* host_array, size_t count, float* device_array) {
    cudaMemcpy(device_array, host_array, count * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_from_device(const float* device_array, const size_t count, float* host_array) {
    cudaMemcpy(host_array, device_array, count * sizeof(float), cudaMemcpyDeviceToHost);
}
