#pragma once

#include "utils.h"
#include <cuda_runtime.h>

extern "C" {
// C = alpha * op(A) * op(B) + beta * C
// where the ops are either identity or transpose depending on transpose_A, transpose_B
// e.g. C = A * B results from setting transpose_A and transpose_B to no_transpose,
// alpha to 1.0, and beta to 0.0
void launch_tiled_multiply(const float* A,
        const Op op_A,
        const float alpha,
        const unsigned int ai,
        const unsigned int aj,
        const float* B,
        const Op op_B,
        const unsigned int bi,
        const unsigned int bj,
        float* C,
        const float beta,
        const dim3 grid,
        const dim3 block,
        const unsigned int shared_mem_size);
}
