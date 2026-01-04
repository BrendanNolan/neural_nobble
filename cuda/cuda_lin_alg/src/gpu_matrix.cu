#include <cassert>

#include "gpu_matrix.h"

__global__ void tiled_multiply(const float* A,
        const Op op_A,
        const float alpha,
        const unsigned int ai,
        const unsigned int aj,
        const float* B,
        const Op op_B,
        const unsigned int bj,
        float* C,
        const float beta) {
    assert(blockDim.x == blockDim.y);
    const auto T = blockDim.x;
    extern __shared__ float shared[];
    float* a_tile = shared;
    float* b_tile = a_tile + T * T;
    float* c_tile = b_tile + T * T;
    for (auto x = 0U; x < ai; x += gridDim.x * blockDim.x) {
        for (auto y = 0U; y < bj; y += gridDim.y * blockDim.y) {
            const auto g_i = x + blockIdx.x * blockDim.x + threadIdx.x;
            const auto g_j = y + blockIdx.y * blockDim.y + threadIdx.y;
            const auto l_c_cell = threadIdx.x * T + threadIdx.y;
            const auto c_global_index = g_i * bj + g_j;
            c_tile[l_c_cell] = beta * C[c_global_index];
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
            if (g_i < ai && g_j < bj)
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
        const unsigned int bj,
        float* C,
        const float beta,
        const dim3 grid,
        const dim3 block,
        const unsigned int shared_mem_size) {
    tiled_multiply<<<grid, block, shared_mem_size>>>(A, op_A, alpha, ai, aj, B, op_B, bj, C, beta);
}
