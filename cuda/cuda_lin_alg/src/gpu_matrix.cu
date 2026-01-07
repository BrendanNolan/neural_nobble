#include <cassert>
#include <utility>

#include "utils.h"
#include "gpu_matrix.h"

__global__ void tiled_multiply(const float* A,
        const Op op_A,
        const float alpha,
        unsigned int ai,
        unsigned int aj,
        const float* B,
        const Op op_B,
        unsigned int bi,
        unsigned int bj,
        float* C,
        const float beta) {
    assert(blockDim.x == blockDim.y);
    const auto T = blockDim.x;
    extern __shared__ float shared[];
    float* a_tile = shared;
    float* b_tile = a_tile + T * T;
    float* c_tile = b_tile + T * T;
    if (op_A == Op::transpose) {
        std::swap(ai, aj);
    }
    if (op_B == Op::transpose) {
        std::swap(bi, bj);
    }
    for (auto x = 0U; x < ai; x += gridDim.x * blockDim.x) {
        for (auto y = 0U; y < bj; y += gridDim.y * blockDim.y) {
            const auto g_i = x + blockIdx.x * blockDim.x + threadIdx.x;
            const auto g_j = y + blockIdx.y * blockDim.y + threadIdx.y;
            const auto l_c_cell = threadIdx.x * T + threadIdx.y;
            const auto c_global_index = g_i * bj + g_j;
            const auto c_global_index_valid = g_i < ai && g_j < bj;
            c_tile[l_c_cell] = c_global_index_valid ? beta * C[c_global_index] : 0U;
            for (auto k = 0U; k < aj; k += T) {
                const auto in_scope_for_a = (g_i < ai && k + threadIdx.y < aj);
                const auto in_scope_for_b = (k + threadIdx.x < aj && g_j < bj);
                a_tile[l_c_cell] = in_scope_for_a ? A[g_i * aj + (k + threadIdx.y)] : 0U;
                b_tile[l_c_cell] = in_scope_for_b ? B[(k + threadIdx.x) * bj + g_j] : 0U;
                __syncthreads();
                for (auto kk = 0U; kk < T; ++kk) {
                    c_tile[l_c_cell] +=
                            alpha * a_tile[threadIdx.x * T + kk] * b_tile[kk * T + threadIdx.y];
                }
                __syncthreads();
            }
            if (c_global_index_valid)
                C[c_global_index] = c_tile[l_c_cell];
        }
    }
}

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
        const unsigned int shared_mem_size) {
    tiled_multiply<<<grid, block, shared_mem_size>>>(
            A, op_A, alpha, ai, aj, B, op_B, bi, bj, C, beta);
}
