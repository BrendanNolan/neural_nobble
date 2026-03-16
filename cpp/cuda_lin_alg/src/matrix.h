#pragma once

#include "cuda_utils.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <optional>

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

constexpr auto target_elements_per_thread = 4u;
constexpr auto square_root_of_target_elements_per_thread = 2u;

class GemmLaunchConfig {
 public:
    static std::optional<GemmLaunchConfig> create(const dim3& grid_dim, const dim3& block_dim);
    const dim3& grid_dim() const;
    const dim3& block_dim() const;
    unsigned int shared_mem_per_block() const;
 private:
    GemmLaunchConfig() = default;
    bool is_legal() const;
    dim3 grid_dim_;
    dim3 block_dim_;
};

// C = alpha * op(A) * op(B) + beta * C
// where the ops are either identity or transpose depending on transpose_A, transpose_B
// e.g. C = A * B results from setting transpose_A and transpose_B to no_transpose,
// alpha to 1.0, and beta to 0.0
void run_tiled_multiply(GemmParams params,
        const Dim3POD grid,
        const Dim3POD block,
        const unsigned int shared_mem_size);
}
