#pragma once

#include <cuda_runtime.h>

extern "C" {
void launch_tiled_multiply(const float* A,
        const unsigned int ai,
        const unsigned int aj,
        const float* B,
        const unsigned int bj,
        float* C,
        const dim3 grid,
        const dim3 block,
        const unsigned int shared_mem_size);
}
