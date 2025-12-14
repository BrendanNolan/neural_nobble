#include "gpu_matrix.h"
#include "matrix.hpp"
#include "test_config.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace {

class LaunchConfig {
 public:
    static std::optional<LaunchConfig> create(const dim3& grid_dim, const dim3& block_dim) {
        auto config = LaunchConfig{};
        config.grid_dim_ = grid_dim;
        config.block_dim_ = block_dim;
        if (!config.is_legal()) {
            return std::nullopt;
        }
        return config;
    }
    const dim3& grid_dim() const {
        return grid_dim_;
    }
    const dim3& block_dim() const {
        return block_dim_;
    }
    unsigned int shared_mem_per_block() const {
        return tile_size() * 3U * sizeof(float);
    }
 private:
    LaunchConfig() = default;
    bool is_legal() const {
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
    unsigned int tile_size() const {
        assert(block_dim().x == block_dim().y);
        return block_dim().x * block_dim().x;
    }
    dim3 grid_dim_;
    dim3 block_dim_;
};

std::string to_string(const dim3& dim) {
    return "(" + std::to_string(dim.x) + "," + std::to_string(dim.y) + "," + std::to_string(dim.z)
            + ")";
}

std::string to_string(const LaunchConfig& config) {
    return "gridDim: " + to_string(config.grid_dim())
            + " blockDim: " + to_string(config.block_dim());
}

using Dim = lin_alg::Dimension;

struct CudaInput {
    const float* A = nullptr;
    const unsigned int ai = 0U;
    const unsigned int aj = 0U;
    const float* B = nullptr;
    const unsigned int bj = 0U;
    float* C = nullptr;
    LaunchConfig config;
};

std::chrono::milliseconds raw_cuda_multiply(const CudaInput& input) {
    const auto start = std::chrono::high_resolution_clock::now();
    launch_tiled_multiply(input.A,
            input.ai,
            input.aj,
            input.B,
            input.bj,
            input.C,
            input.config.grid_dim(),
            input.config.block_dim(),
            input.config.shared_mem_per_block());
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

CudaInput ExtractInput(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const std::optional<LaunchConfig>& optional_config) {
    const auto a_bytes = raw_size(a) * sizeof(float);
    float* A;
    cudaMalloc(&A, a_bytes);
    cudaMemcpy(A, a.raw(), a_bytes, cudaMemcpyHostToDevice);
    const auto b_bytes = raw_size(b) * sizeof(float);
    float* B;
    cudaMalloc(&B, b_bytes);
    cudaMemcpy(B, b.raw(), b_bytes, cudaMemcpyHostToDevice);
    const auto c_bytes = a.dim().i * b.dim().j * sizeof(float);
    float* C;
    cudaMalloc(&C, c_bytes);
    const auto default_block_edge_size = 4U;
    const auto default_launch_config = LaunchConfig::create(
            dim3{static_cast<unsigned int>(
                         (a.dim().i + default_block_edge_size - 1) / default_block_edge_size),
                    static_cast<unsigned int>(
                            (b.dim().j + default_block_edge_size - 1) / default_block_edge_size)},
            dim3{default_block_edge_size, default_block_edge_size})
                                               .value();
    return CudaInput{.A = A,
            .ai = a.dim().i,
            .aj = a.dim().j,
            .B = B,
            .bj = b.dim().j,
            .C = C,
            .config = optional_config.value_or(default_launch_config)};
}

struct MultiplyResult {
    lin_alg::Matrix result_matrix;
    std::chrono::milliseconds duration;
    LaunchConfig launch_config_used;
};
std::string to_string(const MultiplyResult& result) {
    return "duration: " + std::to_string(result.duration.count())
            + "ms, launch config :" + to_string(result.launch_config_used);
}

MultiplyResult cuda_tiled_multiply(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const std::optional<LaunchConfig>& optional_config = std::nullopt) {
    const auto input = ExtractInput(a, b, optional_config);
    const auto duration_ms = raw_cuda_multiply(input);
    const auto c_bytes = input.ai * input.bj * sizeof(float);
    float* h_C = static_cast<float*>(malloc(c_bytes));
    cudaMemcpy(h_C, input.C, c_bytes, cudaMemcpyDeviceToHost);
    return MultiplyResult{.result_matrix = lin_alg::Matrix::from_raw(
                                  h_C, lin_alg::Dimension{a.dim().i, b.dim().j}),
            .duration = duration_ms,
            .launch_config_used = input.config};
}

LaunchConfig get_test_config() {
    const auto config = LaunchConfig::create(
            dim3{TestConfig::instance().block_edge, TestConfig::instance().block_edge, 1U},
            dim3{TestConfig::instance().block_edge, TestConfig::instance().block_edge, 1U});
    if (!config) {
        throw std::invalid_argument{"Invalid Launch Configuration"};
    }
    return config.value();
}

std::vector<LaunchConfig> generate_launch_configs() {
    auto configs = std::vector<LaunchConfig>{get_test_config()};
    const auto sizes = std::vector<unsigned int>{256U, 128U, 64U, 32U, 16U, 8U, 1U};
    for (const auto grid_edge : sizes) {
        for (const auto block_edge : sizes) {
            if (grid_edge == 1U && block_edge == 1U) {
                continue;
            }
            const auto grid_dim = dim3(grid_edge, grid_edge);
            const auto block_dim = dim3(block_edge, block_edge);
            const auto config = LaunchConfig::create(grid_dim, block_dim);
            if (!config) {
                continue;
            }
            configs.push_back(config.value());
        }
    }
    return configs;
}

void correctness_test(const unsigned int rows_left,
        const unsigned int common,
        const unsigned int columns_right) {
    const auto a = lin_alg::Matrix::random(Dim{rows_left, common});
    const auto b = lin_alg::Matrix::random(Dim{common, columns_right});
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    const auto cuda_multiply_result = cuda_tiled_multiply(a, b);
    EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    for (const auto& config : generate_launch_configs()) {
        const auto cuda_multiply_result = cuda_tiled_multiply(a, b, config);
        EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    }
}

void speed_test(const unsigned int dim_of_square_matrix) {
    const auto a = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    const auto b = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});

    auto start = std::chrono::high_resolution_clock::now();
    const auto naive_multiply_result = lin_alg::naive_multiply(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    const auto naive_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Naive CPU execution time: " << naive_time << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto tiled_multiply_result = lin_alg::tiled_multiply(a, b, 4U);
    end = std::chrono::high_resolution_clock::now();
    const auto optimised_cpu_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Optimised CPU execution time: " << optimised_cpu_time << " ms" << std::endl;

    for (const auto& config : generate_launch_configs()) {
        const auto cuda_multiply_result = cuda_tiled_multiply(a, b, config);
        std::cout << "Optimised GPU execution " << to_string(cuda_multiply_result) << std::endl;
        EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
        EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    }
}

}// namespace

TEST(SpeedTest, SevenElements) {
    speed_test(7U);
}

TEST(SpeedTest, ThirtyThreeElements) {
    speed_test(33U);
}

TEST(SpeedTest, OneThousandElements) {
    speed_test(1U << 7U);
}

TEST(SpeedTest, OneMillionElements) {
    speed_test(1U << 10U);
}

TEST(CorrectnessTest, Small) {
    const auto rows_left = (1U << 5) + 1U;
    const auto common = (1U << 4) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test(rows_left, common, columns_right);
}

TEST(CorrectnessTest, Large) {
    const auto rows_left = (1U << 8) + 1U;
    const auto common = (1U << 7) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test(rows_left, common, columns_right);
}

TEST(PrintGpuStats, Basic) {
    auto prop = cudaDeviceProp{};
    cudaGetDeviceProperties(&prop, 0);
    const auto threads_per_block = prop.maxThreadsPerBlock;
    const auto threads_per_sm = prop.maxThreadsPerMultiProcessor;
    const auto total_sms = prop.multiProcessorCount;
    const auto max_concurrent_threads = threads_per_sm * total_sms;
    printf("GPU: %s\n", prop.name);
    printf("SM Count: %d\n", total_sms);
    printf("Max Threads Per SM: %d\n", threads_per_sm);
    printf("Max Threads Per Block: %d\n", threads_per_block);
    printf("Maximum Concurrent Threads: %d\n", max_concurrent_threads);
}
