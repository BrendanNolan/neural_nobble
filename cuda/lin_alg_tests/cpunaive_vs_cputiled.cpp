#include "matrix.hpp"
#include "test_config.hpp"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <optional>
#include <ratio>
#include <stdexcept>
#include <vector>

#include <stdlib.h>

#include <gtest/gtest.h>

namespace {

using Dim = lin_alg::Dimension;

struct MultiplyResult {
    lin_alg::Matrix result_matrix;
    std::chrono::milliseconds duration;
};

enum class TilePolicy { exclude_small, all };

enum class Timing { time_calls, do_not_time_calls };

std::vector<unsigned int> get_tile_sizes(const TilePolicy tile_policy) {
    auto tile_sizes = std::vector<unsigned int>{};
    tile_sizes.emplace_back(TestConfig::instance().tile_edge);
    for (const auto i : std::vector{1U, 2U, 4U, 8U, 16U, 32U, 64U}) {
        if (tile_policy == TilePolicy::exclude_small && i < 8U) {
            continue;
        }
        if (std::find(tile_sizes.cbegin(), tile_sizes.cend(), i) != tile_sizes.cend()) {
            continue;
        }
        tile_sizes.emplace_back(i);
    }
    return tile_sizes;
}

std::string op_to_string(const Op op) {
    if (op == Identity) {
        return "Identity";
    }
    if (op == Transpose) {
        return "Transpose";
    }
    assert(false && "Unrecognised Op Value");
    return {};
}

template <Op op_a, Op op_b>
void run_test(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const float alpha,
        const unsigned int tile_size,
        const Timing timing) {
    if (!can_multiply(a, op_a, b, op_b)) {
        return;
    }
    const auto naive_start = std::chrono::high_resolution_clock::now();
    const auto naive_result = lin_alg::naive_multiply(a, op_a, alpha, b, op_b);
    const auto naive_end = std::chrono::high_resolution_clock::now();
    const auto tiled_start = std::chrono::high_resolution_clock::now();
    const auto tiled_result = lin_alg::tiled_multiply<op_a, op_b>(a, alpha, b, tile_size);
    const auto tiled_end = std::chrono::high_resolution_clock::now();
    if (timing == Timing::time_calls) {
        std::cout << "op_a" << op_to_string(op_a) << " op_b" << op_to_string(op_b) << " "
                  << display(a.dim()) << "x" << display(b.dim()) << " Naive:"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(naive_end - naive_start)
                             .count()
                  << "ms"
                  << " Tiled(tile size " << tile_size << "):"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tiled_end - tiled_start)
                             .count()
                  << "ms" << std::endl;
    }
    EXPECT_EQ(naive_result, tiled_result);
}

void test(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const TilePolicy tile_policy,
        const Timing timing) {
    const auto alpha = 2.3f;
    for (const auto tile_size : get_tile_sizes(tile_policy)) {
        run_test<Identity, Identity>(a, b, alpha, tile_size, timing);
        run_test<Identity, Transpose>(a, b, alpha, tile_size, timing);
        run_test<Transpose, Identity>(a, b, alpha, tile_size, timing);
        run_test<Transpose, Transpose>(a, b, alpha, tile_size, timing);
    }
}

void correctness_test(const lin_alg::Matrix& a, const lin_alg::Matrix& b) {
    test(a, b, TilePolicy::all, Timing::do_not_time_calls);
}

void speed_test(const lin_alg::Matrix& a, const lin_alg::Matrix& b) {
    test(a, b, TilePolicy::exclude_small, Timing::time_calls);
}

void correctness_test_random(const unsigned int rows_left,
        const unsigned int common,
        const unsigned int columns_right) {
    const auto a = lin_alg::Matrix::random(Dim{rows_left, common});
    const auto b = lin_alg::Matrix::random(Dim{common, columns_right});
    correctness_test(a, b);
}

void speed_test(const unsigned int dim_of_square_matrix) {
    const auto a = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    const auto b = lin_alg::Matrix::random(Dim{dim_of_square_matrix, dim_of_square_matrix});
    speed_test(a, b);
}

}// namespace

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
    const auto a = lin_alg::Matrix::from_raw(a_raw, lin_alg::Dimension{3U, 3U});
    const auto b = lin_alg::Matrix::from_raw(b_raw, lin_alg::Dimension{3U, 3U});
    correctness_test(a, b);
}

TEST(RandomCorrectnessTest, Tiny) {
    const auto rows_left = (1U << 2) + 1U;
    const auto common = (1U << 2) + 3U;
    const auto columns_right = (1U << 2) + 1U;
    correctness_test_random(rows_left, common, columns_right);
}

TEST(RandomCorrectnessTest, Small) {
    const auto rows_left = 11U;
    const auto common = 7U;
    const auto columns_right = 9U;
    correctness_test_random(rows_left, common, columns_right);
}

TEST(RandomCorrectnessTest, Medium) {
    const auto rows_left = (1U << 5) + 1U;
    const auto common = (1U << 4) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test_random(rows_left, common, columns_right);
}

TEST(RandomCorrectnessTest, Large) {
    const auto rows_left = (1U << 8) + 1U;
    const auto common = (1U << 7) + 3U;
    const auto columns_right = (1U << 6) + 1U;
    correctness_test_random(rows_left, common, columns_right);
}

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
