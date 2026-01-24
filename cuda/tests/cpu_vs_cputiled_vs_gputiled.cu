#include "gpu_matrix.h"
#include "matrix.hpp"
#include "test_config.hpp"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include <stdlib.h>

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
    GemmParams params;
    LaunchConfig config;
};

namespace {
Dim3POD cuda_dim3_to_dim3pod(const dim3& dim) {
    return Dim3POD{.x = dim.x, .y = dim.y, .z = dim.z};
}
}// namespace

std::chrono::milliseconds raw_cuda_multiply(const CudaInput& input) {
    const auto start = std::chrono::high_resolution_clock::now();
    launch_tiled_multiply(input.params,
            cuda_dim3_to_dim3pod(input.config.grid_dim()),
            cuda_dim3_to_dim3pod(input.config.block_dim()),
            input.config.shared_mem_per_block());
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

CudaInput ExtractInput(const lin_alg::Matrix& a,
        const Op op_a,
        const float alpha,
        const lin_alg::Matrix& b,
        const Op op_b,
        const float beta,
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
    return CudaInput{
            .params = GemmParams{.A = ConstMatrixDetails{.data = A,
                                         .rows = a.dim().i,
                                         .columns = a.dim().j},
                    .op_A = op_a,
                    .alpha = alpha,
                    .B = ConstMatrixDetails{.data = B, .rows = b.dim().i, .columns = b.dim().j},
                    .op_B = op_b,
                    .beta = beta,
                    .C = C},
            .config = optional_config.value_or(default_launch_config)};
}

struct MultiplyResult {
    lin_alg::Matrix result_matrix;
    std::chrono::milliseconds duration;
    LaunchConfig launch_config_used;
};
std::string to_string(const MultiplyResult& result) {
    return "duration:    " + std::to_string(result.duration.count())
            + "ms,    launch config :" + to_string(result.launch_config_used);
}

MultiplyResult cuda_tiled_multiply(const lin_alg::Matrix& a,
        const Op op_a,
        const float alpha,
        const lin_alg::Matrix& b,
        const Op op_b,
        const float beta,
        const std::optional<LaunchConfig>& optional_config = std::nullopt) {
    const auto input = ExtractInput(a, op_a, alpha, b, op_b, beta, optional_config);
    const auto duration_ms = raw_cuda_multiply(input);
    const auto c_bytes = input.params.A.rows * input.params.B.columns * sizeof(float);
    float* h_C = static_cast<float*>(malloc(c_bytes));
    cudaMemcpy(h_C, input.params.C, c_bytes, cudaMemcpyDeviceToHost);
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

enum class LaunchConfigRangeHint { all, only_sensible };

std::vector<LaunchConfig> generate_launch_configs(const LaunchConfigRangeHint range_hint) {
    auto configs = std::vector<LaunchConfig>{get_test_config()};
    const auto sizes = std::vector<unsigned int>{256U, 128U, 64U, 32U, 16U, 8U, 4U, 2U, 1U};
    for (const auto grid_edge : sizes) {
        for (const auto block_edge : sizes) {
            if (range_hint == LaunchConfigRangeHint::only_sensible
                    && (grid_edge < 8U || block_edge < 8U)) {
                continue;
            }
            const auto grid_dim = dim3(grid_edge, grid_edge);
            const auto block_dim = dim3(block_edge, block_edge);
            if (const auto config = LaunchConfig::create(grid_dim, block_dim)) {
                configs.push_back(config.value());
            }
        }
    }
    return configs;
}

void check_multiplication_results(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const LaunchConfigRangeHint range_hint) {
    const auto naive_multiply_result = lin_alg::naive_multiply(a, Identity, 1.0, b, Identity);
    const auto cuda_multiply_result = cuda_tiled_multiply(a, Identity, 1.0, b, Identity, 1.0);
    EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    for (const auto& config : generate_launch_configs(range_hint)) {
        std::cout << "Checking for correctness with launch config: " << to_string(config)
                  << std::endl;
        const auto cuda_multiply_result =
                cuda_tiled_multiply(a, Identity, 1.0, b, Identity, 1.0, config);
        EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    }
}

void correctness_test(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const LaunchConfigRangeHint range_hint) {
    check_multiplication_results(a, b, range_hint);
}

void correctness_test_random(const unsigned int rows_left,
        const unsigned int common,
        const unsigned int columns_right,
        const LaunchConfigRangeHint range_hint) {
    const auto a = lin_alg::Matrix::random(Dim{rows_left, common});
    const auto b = lin_alg::Matrix::random(Dim{common, columns_right});
    check_multiplication_results(a, b, range_hint);
}

void speed_test(const unsigned int dim_of_square_matrix, const LaunchConfigRangeHint range_hint) {
    const auto a = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    const auto b = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});

    auto start = std::chrono::high_resolution_clock::now();
    const auto naive_multiply_result = lin_alg::naive_multiply(a, Identity, 1.0, b, Identity);
    auto end = std::chrono::high_resolution_clock::now();
    const auto naive_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Naive CPU execution time: " << naive_time << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    auto tiled_multiply_result = lin_alg::tiled_multiply<Identity, Identity>(a, 1.0, b, 8U);
    end = std::chrono::high_resolution_clock::now();
    const auto optimised_cpu_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Optimised CPU execution time: " << optimised_cpu_time << " ms" << std::endl;

    for (const auto& config : generate_launch_configs(range_hint)) {
        const auto cuda_multiply_result =
                cuda_tiled_multiply(a, Identity, 1.0, b, Identity, 1.0, config);
        std::cout << "Optimised GPU execution " << to_string(cuda_multiply_result) << std::endl;
        EXPECT_EQ(tiled_multiply_result, naive_multiply_result);
        EXPECT_EQ(cuda_multiply_result.result_matrix, naive_multiply_result);
    }
}

}// namespace

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

TEST(CorrectnessTest, Tiny) {
    auto* a_raw = static_cast<float*>(malloc(3 * 3 * sizeof(float)));
    auto* b_raw = static_cast<float*>(malloc(3 * 3 * sizeof(float)));
    a_raw[0] = 1.0f;
    a_raw[1] = 2.0f;
    a_raw[2] = 3.0f;
    a_raw[3] = 4.0f;
    a_raw[4] = 5.0f;
    a_raw[5] = 6.0f;
    a_raw[6] = 7.0f;
    a_raw[7] = 8.0f;
    a_raw[8] = 9.0f;
    b_raw[0] = 9.0f;
    b_raw[1] = 8.0f;
    b_raw[2] = 7.0f;
    b_raw[3] = 6.0f;
    b_raw[4] = 5.0f;
    b_raw[5] = 4.0f;
    b_raw[6] = 3.0f;
    b_raw[7] = 2.0f;
    b_raw[8] = 1.0f;
    const auto a = lin_alg::Matrix::from_raw(a_raw, lin_alg::Dimension{2U, 2U});
    const auto b = lin_alg::Matrix::from_raw(b_raw, lin_alg::Dimension{2U, 2U});
    correctness_test(a, b, LaunchConfigRangeHint::all);
}

TEST(RandomCorrectnessTest, Tiny) {
    const auto rows_left = (1U << 2) + 1U;
    const auto common = (1U << 2) + 3U;
    const auto columns_right = (1U << 2) + 1U;
    correctness_test_random(rows_left, common, columns_right, LaunchConfigRangeHint::all);
}

TEST(RandomCorrectnessTest, Small) {
    const auto rows_left = 11U;
    const auto common = 7U;
    const auto columns_right = 9U;
    correctness_test_random(rows_left, common, columns_right, LaunchConfigRangeHint::all);
}

TEST(RandomCorrectnessTest, Medium) {
    const auto rows_left = (1U << 5) + 1U;
    const auto common = (1U << 4) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test_random(rows_left, common, columns_right, LaunchConfigRangeHint::all);
}

TEST(RandomCorrectnessTest, Large) {
    const auto rows_left = (1U << 8) + 1U;
    const auto common = (1U << 7) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test_random(rows_left, common, columns_right, LaunchConfigRangeHint::only_sensible);
}

TEST(SpeedTest, SevenElements) {
    speed_test(7U, LaunchConfigRangeHint::all);
}

TEST(SpeedTest, ThirtyThreeElements) {
    speed_test(33U, LaunchConfigRangeHint::all);
}

TEST(SpeedTest, OneThousandElements) {
    speed_test(1U << 7U, LaunchConfigRangeHint::all);
}

TEST(SpeedTest, OneMillionElements) {
    speed_test(1U << 10U, LaunchConfigRangeHint::only_sensible);
}
