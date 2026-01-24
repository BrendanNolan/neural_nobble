#pragma once

#include "utils.h"
#include <cuda_runtime.h>

extern "C" {
struct ConstMatrixDetails {
    const float* data;
    unsigned int rows;
    unsigned int columns;
};
struct MutableMatrixDetails {
    float* data;
    unsigned int rows;
    unsigned int columns;
};

struct GemmParams {
    ConstMatrixDetails A;
    Op op_A;
    float alpha;
    ConstMatrixDetails B;
    Op op_B;
    float beta;
    float* C;
};

// Cuda's dim3 type has constructors and is, as such, not a POD type, making it not strictly safe to
// pass across the FFI boundary.
struct Dim3POD {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

// C = alpha * op(A) * op(B) + beta * C
// where the ops are either identity or transpose depending on transpose_A, transpose_B
// e.g. C = A * B results from setting transpose_A and transpose_B to no_transpose,
// alpha to 1.0, and beta to 0.0
void launch_tiled_multiply(GemmParams params,
        const Dim3POD grid,
        const Dim3POD block,
        const unsigned int shared_mem_size);
}
