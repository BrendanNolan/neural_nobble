#include <cassert>
#include <iterator>
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
    for (auto x = 0u; x < ai; x += gridDim.x * blockDim.x) {
        for (auto y = 0u; y < bj; y += gridDim.y * blockDim.y) {
            const auto g_i = x + blockIdx.x * blockDim.x + threadIdx.x;
            const auto g_j = y + blockIdx.y * blockDim.y + threadIdx.y;
            const auto l_c_cell = threadIdx.x * T + threadIdx.y;
            const auto c_global_index = g_i * bj + g_j;
            const auto c_global_index_valid = g_i < ai && g_j < bj;
            c_tile[l_c_cell] = c_global_index_valid ? params.beta * params.C[c_global_index] : 0u;
            for (auto k = 0u; k < aj; k += T) {
                const auto in_scope_for_a = (g_i < ai && k + threadIdx.y < aj);
                const auto in_scope_for_b = (k + threadIdx.x < bi && g_j < bj);
                a_tile[l_c_cell] = in_scope_for_a ? a_at(g_i, k + threadIdx.y) : 0u;
                b_tile[l_c_cell] = in_scope_for_b ? b_at(k + threadIdx.x, g_j) : 0u;
                __syncthreads();
                for (auto kk = 0u; kk < T; ++kk) {
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

namespace {
__device__ constexpr bool is_power_of_2_in_range(const unsigned int x,
        const unsigned int lower_bound_inclusive,
        const unsigned int upper_bound_exclusive) {
    for (auto power = lower_bound_inclusive; power < upper_bound_exclusive; ++power) {
        if (x == 1u << power) {
            return true;
        }
    }
    return false;
}

template <unsigned int BlockDimX, unsigned int BlockDimXLowerBound, typename Float>
__device__ __forceinline__ void run_reduction_step(Float shared, const unsigned int thread_id) {
    static_assert(std::is_same_v<Float, float*>
            || (std::is_same_v<Float, volatile float*>
                    && is_power_of_2_in_range(BlockDimXLowerBound, 1u, 7u)));
    static_assert(is_power_of_2_in_range(BlockDimX, 0u, 10u));
    if constexpr (BlockDimX >= BlockDimXLowerBound) {
        if (thread_id < BlockDimXLowerBound / 2u) {
            shared[thread_id] += shared[thread_id + BlockDimXLowerBound / 2u];
        }
    }
}

// Without the volatile keyword, since there is no __syncthreads, the compiler may optimise away
// shared-memory reads and writes by keeping values in registers
template <unsigned int BlockDimX>
__device__ __forceinline__ void warp_reduce(volatile float* shared, const unsigned int thread_id) {
    static_assert(is_power_of_2_in_range(BlockDimX, 0u, 10u));
    assert(thread_id < 32u);
    run_reduction_step<BlockDimX, 64u>(shared, thread_id);
    run_reduction_step<BlockDimX, 32u>(shared, thread_id);
    run_reduction_step<BlockDimX, 16u>(shared, thread_id);
    run_reduction_step<BlockDimX, 8u>(shared, thread_id);
    run_reduction_step<BlockDimX, 4u>(shared, thread_id);
    run_reduction_step<BlockDimX, 2u>(shared, thread_id);
}
}// namespace

template <unsigned int BlockDimX>
__global__ void sum_reduce(const float* input, unsigned int input_length, float* output) {
    // 2^9 == 512, the laregest allowed block dimension
    static_assert(is_power_of_2_in_range(BlockDimX, 0u, 10u));
    assert(BlockDimX == blockDim.x);
    extern __shared__ float shared[];
    const auto thread_id = threadIdx.x;
    shared[thread_id] = 0u;
    auto add_to_shared = [input_length, input, thread_id](const unsigned int global_index) {
        if (global_index < input_length)
            shared[thread_id] += input[global_index];
    };
    const auto threads_per_grid = gridDim.x * BlockDimX;
    const auto index_within_stride = (blockIdx.x * BlockDimX * 2u) + thread_id;
    for (auto stride_start = 0u; stride_start < input_length;
            stride_start += 2u * threads_per_grid) {
        const auto global_index = stride_start + index_within_stride;
        add_to_shared(global_index);
        add_to_shared(global_index + BlockDimX);
    }
    __syncthreads();
    run_reduction_step<BlockDimX, 512u>(shared, thread_id);
    __syncthreads();
    run_reduction_step<BlockDimX, 256u>(shared, thread_id);
    __syncthreads();
    run_reduction_step<BlockDimX, 128u>(shared, thread_id);
    __syncthreads();
    // At the previous reduction step, there were 64 active threads and there are now 64 active
    // elements in shared memory; at the next step, there will be only 32 active threads, so
    // there will be only one active warp and we will not need to call __syncthreads
    if (thread_id < 32u) {
        warp_reduce<BlockDimX>(shared, thread_id);
    }
    if (thread_id == 0u) {
        output[blockIdx.x] = shared[0u];
    }
}

namespace {
void launch_sum_reduce(float* input,
        unsigned int length,
        float* result,
        const unsigned int grid_dim_x,
        const unsigned int block_dim_x) {
    switch (block_dim_x) {
    case 512u:
        sum_reduce<512u><<<grid_dim_x, 512u, 512u * sizeof(float)>>>(input, length, result);
        return;
    case 256u:
        sum_reduce<256u><<<grid_dim_x, 256u, 256u * sizeof(float)>>>(input, length, result);
        return;
    case 128u:
        sum_reduce<128u><<<grid_dim_x, 128u, 128u * sizeof(float)>>>(input, length, result);
        return;
    case 64u:
        sum_reduce<64u><<<grid_dim_x, 64u, 64u * sizeof(float)>>>(input, length, result);
        return;
    case 32u:
        sum_reduce<32u><<<grid_dim_x, 32u, 32u * sizeof(float)>>>(input, length, result);
        return;
    case 16u:
        sum_reduce<16u><<<grid_dim_x, 16u, 16u * sizeof(float)>>>(input, length, result);
        return;
    case 8u:
        sum_reduce<8u><<<grid_dim_x, 8u, 8u * sizeof(float)>>>(input, length, result);
        return;
    case 4u:
        sum_reduce<4u><<<grid_dim_x, 4u, 4u * sizeof(float)>>>(input, length, result);
        return;
    case 2u:
        sum_reduce<2u><<<grid_dim_x, 2u, 2u * sizeof(float)>>>(input, length, result);
        return;
    case 1u:
        sum_reduce<1u><<<grid_dim_x, 1u, 1u * sizeof(float)>>>(input, length, result);
        return;
    }
    assert(false && "block_dim_x should be a power of 2u between 1u and 512u inclusively");
}
}// namespace

SumReduceLaunchConfig compute_sum_reduce_launch_config(unsigned int input_length) {
    auto fit_block_size = [](const unsigned int upper_bound) {
        if (upper_bound > 512u) {
            return 512u;
        }
        auto block_dim_x = 512u;
        while (block_dim_x >= upper_bound && block_dim_x >= 64u) {
            block_dim_x /= 2u;
        }
        return block_dim_x;
    };
    if (input_length <= 512u) {
        return SumReduceLaunchConfig{.grid_dim_x = 1u, .block_dim_x = fit_block_size(512u)};
    }
    const auto block_dim_x = 512u;
    auto props = cudaDeviceProp{};
    cudaGetDeviceProperties(&props, 0);
    const auto sm_count = static_cast<unsigned int>(props.multiProcessorCount);
    constexpr auto blocks_per_sm = 2u;
    const auto max_grid_x = sm_count * blocks_per_sm;
    // TODO: Make items_per_thread and grid_stride paramrs of this func and run_sum_reduce
    const auto items_per_thread = 2u;
    constexpr auto grid_stride = 4u;
    const auto work_per_block = 512u * items_per_thread * grid_stride;
    const auto blocks_needed = input_length / work_per_block;
    return SumReduceLaunchConfig{
            .grid_dim_x = std::min(max_grid_x, blocks_needed), .block_dim_x = block_dim_x};
}

void run_sum_reduce(float* input, unsigned int length, float* result) {
    auto launch_config = compute_sum_reduce_launch_config(length);
    const auto initial_grid_x = launch_config.grid_dim_x;
    auto* scratch_a = allocate_on_device(initial_grid_x);
    auto* scratch_b = allocate_on_device(initial_grid_x);
    auto* output = scratch_a;
    auto first_iteration = true;
    while (true) {
        launch_sum_reduce(
                input, length, output, launch_config.grid_dim_x, launch_config.block_dim_x);
        if (launch_config.grid_dim_x == 1u) {
            break;
        }
        length = launch_config.grid_dim_x;
        launch_config = compute_sum_reduce_launch_config(launch_config.grid_dim_x);
        if (first_iteration) {
            first_iteration = false;
            input = output;
            output = scratch_b;
        } else {
            std::swap(input, output);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result, output, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(scratch_a);
    cudaFree(scratch_b);
}
