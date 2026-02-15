#include <cassert>
#include <utility>

#include "cuda_utils.h"
#include "gpu_matrix.h"
#include "utils.h"

__device__ float element(const float* matrix,
        const Op op,
        unsigned int columns,
        unsigned int i,
        unsigned int j) {
    switch (op) {
    case Identity:
        return matrix[i * columns + j];
    case Transpose:
        return matrix[j * columns + i];
    default:
        assert(false && "Unhandled Op case");
        return 0.0f;
    }
}

__global__ void tiled_multiply(GemmParams params) {
    assert(blockDim.x == blockDim.y);
    const auto T = blockDim.x;
    extern __shared__ float shared[];
    float* a_tile = shared;
    float* b_tile = a_tile + T * T;
    float* c_tile = b_tile + T * T;
    auto a_at = [params](unsigned int i, unsigned int j) {
        return element(params.A.data, params.op_A, params.A.columns, i, j);
    };
    auto b_at = [params](unsigned int i, unsigned int j) {
        return element(params.B.data, params.op_B, params.B.columns, i, j);
    };
    if (params.op_A == Transpose) {
        cuda_helpers::swap(params.A.rows, params.A.columns);
    }
    if (params.op_B == Transpose) {
        cuda_helpers::swap(params.B.rows, params.B.columns);
    }
    const auto ai = params.A.rows;
    const auto aj = params.A.columns;
    const auto bi = params.B.rows;
    const auto bj = params.B.columns;
    for (auto x = 0U; x < ai; x += gridDim.x * blockDim.x) {
        for (auto y = 0U; y < bj; y += gridDim.y * blockDim.y) {
            const auto g_i = x + blockIdx.x * blockDim.x + threadIdx.x;
            const auto g_j = y + blockIdx.y * blockDim.y + threadIdx.y;
            const auto l_c_cell = threadIdx.x * T + threadIdx.y;
            const auto c_global_index = g_i * bj + g_j;
            const auto c_global_index_valid = g_i < ai && g_j < bj;
            c_tile[l_c_cell] = c_global_index_valid ? params.beta * params.C[c_global_index] : 0U;
            for (auto k = 0U; k < aj; k += T) {
                const auto in_scope_for_a = (g_i < ai && k + threadIdx.y < aj);
                const auto in_scope_for_b = (k + threadIdx.x < bi && g_j < bj);
                a_tile[l_c_cell] = in_scope_for_a ? a_at(g_i, k + threadIdx.y) : 0U;
                b_tile[l_c_cell] = in_scope_for_b ? b_at(k + threadIdx.x, g_j) : 0U;
                __syncthreads();
                for (auto kk = 0U; kk < T; ++kk) {
                    c_tile[l_c_cell] += params.alpha * a_tile[threadIdx.x * T + kk]
                            * b_tile[kk * T + threadIdx.y];
                }
                __syncthreads();
            }
            if (c_global_index_valid)
                params.C[c_global_index] = c_tile[l_c_cell];
        }
    }
}

namespace {
dim3 dim3pod_to_cuda_dim3(const Dim3POD& pod) {
    return dim3{pod.x, pod.y, pod.z};
}
}// namespace

void launch_tiled_multiply(GemmParams params,
        const Dim3POD grid,
        const Dim3POD block,
        const unsigned int shared_mem_size) {
    const auto cuda_grid = dim3pod_to_cuda_dim3(grid);
    const auto cuda_block = dim3pod_to_cuda_dim3(block);
    tiled_multiply<<<cuda_grid, cuda_block, shared_mem_size>>>(params);
    cudaDeviceSynchronize();
}

__global__ void sum_reduce(const float* input, unsigned int input_length, float* output) {
    extern __shared__ float shared[];
    const auto global_index = blockDim.x * blockIdx.x + threadIdx.x;
    shared[threadIdx.x] = (global_index < input_length) ? input[global_index] : 0.0f;
    __syncthreads();
    for (auto i = blockDim.x / 2U; i > 0U; i /= 2U) {
        if (threadIdx.x < i) {
            shared[threadIdx.x] += shared[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        output[blockIdx.x] = shared[0U];
    }
}

void launch_sum_reduction(float* input,
        const unsigned int length,
        float* result,
        const unsigned int block_x) {
    const auto grid_x = cover_divide(cover_divide(length, block_x), 2U);
    auto* big_output = allocate_on_device(grid_x);
    sum_reduce<<<grid_x, block_x, block_x>>>(input, length, big_output);
    cudaDeviceSynchronize();
    sum_reduce<<<1U, block_x, block_x>>>(big_output, block_x, result);
    cudaDeviceSynchronize();
    cudaFree(big_output);
}
