#include "reduce.h"

#include <cassert>
#include <iterator>
#include <utility>

#include "cuda_utils.h"
#include "utils.h"

namespace {
__device__ __forceinline__ constexpr bool is_power_of_2_in_range(const unsigned int x,
        const unsigned int lower_bound_inclusive,
        const unsigned int upper_bound_exclusive) {
    for (auto power = lower_bound_inclusive; power < upper_bound_exclusive; ++power) {
        if (x == 1u << power) {
            return true;
        }
    }
    return false;
}

template <unsigned int BlockDimX, unsigned int BlockDimXLowerBound, typename FloatPointer>
__device__ __forceinline__ void run_reduction_step(FloatPointer shared,
        const unsigned int thread_id) {
    static_assert(std::is_same_v<FloatPointer, float*>
            || (std::is_same_v<FloatPointer, volatile float*>
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
    constexpr auto blocks_per_sm = 3u;
    const auto max_grid_x = sm_count * blocks_per_sm;
    // TODO: Make elements_per_thread and grid_stride paramrs of this func and run_sum_reduce
    const auto elements_per_thread = 2u;
    constexpr auto grid_stride = 4u;
    const auto work_per_block = 512u * elements_per_thread * grid_stride;
    const auto blocks_needed = ceiling(input_length, work_per_block);
    return SumReduceLaunchConfig{.grid_dim_x = std::max(1u, std::min(max_grid_x, blocks_needed)),
            .block_dim_x = block_dim_x};
}

void run_sum_reduce(float* input, unsigned int length, float* result) {
    if (length == 0u) {
        const auto zero = 0.0f;
        cudaMemcpy(result, &zero, sizeof(float), cudaMemcpyHostToDevice);
        return;
    }
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
