#include <cassert>
#include <iterator>
#include <utility>

#include "cuda_utils.h"
#include "matrix.h"
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
    assert(blockDim.y == blockDim.x);
    const auto T = blockDim.y;
    extern __shared__ float shared[];
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
    auto final_c_value = 0.0f;
    float* a_tile = shared;
    float* b_tile = a_tile + T * T;
    // Remember that cuda indexes grid/block rows with y and columns with x, like the x and y axes
    // of a graph, not the rows and columns of a matrix.
    for (auto row = 0u; row < ai; row += gridDim.y * blockDim.y) {
        for (auto column = 0u; column < bj; column += gridDim.x * blockDim.x) {
            const auto g_i = row + blockIdx.y * blockDim.y + threadIdx.y;
            const auto g_j = column + blockIdx.x * blockDim.x + threadIdx.x;
            const auto tile_slot = threadIdx.y * T + threadIdx.x;
            const auto c_global_index = g_i * bj + g_j;
            const auto c_global_index_valid = g_i < ai && g_j < bj;
            final_c_value = c_global_index_valid ? params.beta * params.C[c_global_index] : 0u;
            for (auto k = 0u; k < aj; k += T) {
                const auto in_scope_for_a = (g_i < ai && k + threadIdx.x < aj);
                const auto in_scope_for_b = (k + threadIdx.y < bi && g_j < bj);
                a_tile[tile_slot] = in_scope_for_a ? a_at(g_i, k + threadIdx.x) : 0u;
                b_tile[tile_slot] = in_scope_for_b ? b_at(k + threadIdx.y, g_j) : 0u;
                __syncthreads();
                for (auto kk = 0u; kk < T; ++kk) {
                    final_c_value += a_tile[threadIdx.y * T + kk] * b_tile[kk * T + threadIdx.x];
                }
                __syncthreads();
            }
            if (c_global_index_valid)
                params.C[c_global_index] = params.alpha * final_c_value;
        }
    }
}

namespace {
inline dim3 dim3pod_to_cuda_dim3(const Dim3POD& pod) {
    return dim3{pod.x, pod.y, pod.z};
}
}// namespace

void run_tiled_multiply(GemmParams params,
        const Dim3POD grid,
        const Dim3POD block,
        const unsigned int shared_mem_size) {
    const auto cuda_grid = dim3pod_to_cuda_dim3(grid);
    const auto cuda_block = dim3pod_to_cuda_dim3(block);
    tiled_multiply<<<cuda_grid, cuda_block, shared_mem_size>>>(params);
    cudaDeviceSynchronize();
}

std::optional<GemmLaunchConfig> GemmLaunchConfig::create(const dim3& grid_dim,
        const dim3& block_dim) {
    auto config = GemmLaunchConfig{};
    config.grid_dim_ = grid_dim;
    config.block_dim_ = block_dim;
    if (!config.is_legal()) {
        return std::nullopt;
    }
    return config;
}

const dim3& GemmLaunchConfig::grid_dim() const {
    return grid_dim_;
}

const dim3& GemmLaunchConfig::block_dim() const {
    return block_dim_;
}

unsigned int GemmLaunchConfig::shared_mem_per_block() const {
    assert(block_dim().x == block_dim().y);
    return block_dim().x * block_dim().y * 2u * sizeof(float);
}

bool GemmLaunchConfig::is_legal() const {
    auto properties = cudaDeviceProp{};
    cudaGetDeviceProperties(&properties, 0);
    if (block_dim_.x * block_dim_.y * block_dim_.z
            > static_cast<unsigned int>(properties.maxThreadsPerBlock)) {
        return false;
    }
    if (block_dim_.x > static_cast<unsigned int>(properties.maxThreadsDim[0])
            || block_dim_.y > static_cast<unsigned int>(properties.maxThreadsDim[1])
            || block_dim_.z > static_cast<unsigned int>(properties.maxThreadsDim[2])) {
        return false;
    }
    if (grid_dim_.x > static_cast<unsigned int>(properties.maxGridSize[0])
            || grid_dim_.y > static_cast<unsigned int>(properties.maxGridSize[1])
            || grid_dim_.z > static_cast<unsigned int>(properties.maxGridSize[2])) {
        return false;
    }
    if (shared_mem_per_block() > static_cast<unsigned int>(properties.sharedMemPerBlock)) {
        return false;
    }
    return true;
}
