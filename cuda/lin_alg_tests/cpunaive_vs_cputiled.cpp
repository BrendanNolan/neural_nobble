#include "matrix.hpp"
#include "test_config.hpp"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <optional>
#include <ostream>
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
    for (const auto i : std::vector{1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
        if (tile_policy == TilePolicy::exclude_small && i < 8u) {
            continue;
        }
        if (std::find(tile_sizes.cbegin(), tile_sizes.cend(), i) != tile_sizes.cend()) {
            continue;
        }
        tile_sizes.emplace_back(i);
    }
    return tile_sizes;
}

namespace {
std::string op_to_string(const Op op) {
    if (op == Identity) {
        return "Identity";
    }
    if (op == Transpose) {
        return "Transpose";
    }
    throw std::invalid_argument{"Unrecognised Op"};
}
}// namespace

template <Op op_a, Op op_b>
void run_test(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const float alpha,
        const std::map<std::pair<Op, Op>, lin_alg::Matrix>& expected_answers,
        const unsigned int tile_size,
        const Timing timing) {
    if (!can_multiply(a, op_a, b, op_b)) {
        return;
    }
    const auto tiled_start = std::chrono::high_resolution_clock::now();
    const auto tiled_result = lin_alg::tiled_multiply(a, op_a, alpha, b, op_b, tile_size);
    const auto tiled_end = std::chrono::high_resolution_clock::now();
    if (timing == Timing::time_calls) {
        std::cout << "op_a: " << op_to_string(op_a) << " op_b: " << op_to_string(op_b) << " "
                  << display(a.dim()) << "x" << display(b.dim()) << " Tiled(tile size " << tile_size
                  << "):"
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tiled_end - tiled_start)
                             .count()
                  << "ms" << std::endl;
    }
    EXPECT_EQ(expected_answers.at(std::make_pair(op_a, op_b)), tiled_result)
            << "a: " << a << "op_a: " << op_to_string(op_a) << std::endl
            << "b: " << b << "op_b: " << op_to_string(op_b) << std::endl
            << "alpha: " << alpha << std::endl;
}

void test(const lin_alg::Matrix& a,
        const lin_alg::Matrix& b,
        const TilePolicy tile_policy,
        const Timing timing) {
    const auto alpha = 2.3f;
    auto expected_answers = std::map<std::pair<Op, Op>, lin_alg::Matrix>{};
    for (const auto op_a : {Identity, Transpose}) {
        for (const auto op_b : {Identity, Transpose}) {
            if (!can_multiply(a, op_a, b, op_b)) {
                continue;
            }
            const auto naive_start = std::chrono::high_resolution_clock::now();
            expected_answers.emplace(
                    std::make_pair(op_a, op_b), lin_alg::naive_multiply(a, op_a, alpha, b, op_b));
            const auto naive_end = std::chrono::high_resolution_clock::now();
            if (timing == Timing::time_calls) {
                std::cout << "op_a: " << op_to_string(op_a) << " op_b: " << op_to_string(op_b)
                          << " " << display(a.dim()) << "x" << display(b.dim()) << " Naive:"
                          << std::chrono::duration_cast<std::chrono::milliseconds>(
                                     naive_end - naive_start)
                                     .count()
                          << "ms" << std::endl;
            }
        }
    }
    for (const auto tile_size : get_tile_sizes(tile_policy)) {
        run_test<Identity, Identity>(a, b, alpha, expected_answers, tile_size, timing);
        run_test<Identity, Transpose>(a, b, alpha, expected_answers, tile_size, timing);
        run_test<Transpose, Identity>(a, b, alpha, expected_answers, tile_size, timing);
        run_test<Transpose, Transpose>(a, b, alpha, expected_answers, tile_size, timing);
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

TEST(CorrectnessTest, Minute) {
    auto a_raw = std::vector<float>(4u, 0.0f);
    auto b_raw = a_raw;
    a_raw[0] = 1.0f;
    a_raw[1] = 1.0f;
    a_raw[2] = 2.0f;
    a_raw[3] = 2.0f;
    b_raw[0] = 1.0f;
    b_raw[1] = 1.0f;
    b_raw[2] = 2.0f;
    b_raw[3] = 2.0f;
    const auto a = lin_alg::Matrix::from_raw(a_raw, lin_alg::Dimension{2u, 2u});
    const auto b = lin_alg::Matrix::from_raw(b_raw, lin_alg::Dimension{2u, 2u});
    correctness_test(a, b);
}

TEST(CorrectnessTest, Tiny) {
    auto a_raw = std::vector<float>(9u, 0.0f);
    auto b_raw = a_raw;
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
    const auto a = lin_alg::Matrix::from_raw(a_raw, lin_alg::Dimension{3u, 3u});
    const auto b = lin_alg::Matrix::from_raw(b_raw, lin_alg::Dimension{3u, 3u});
    correctness_test(a, b);
}

TEST(RandomCorrectnessTest, Tiny) {
    const auto rows_left = (1u << 2) + 1u;
    const auto common = (1u << 2) + 3u;
    const auto columns_right = (1u << 2) + 1u;
    correctness_test_random(rows_left, common, columns_right);
}

TEST(RandomCorrectnessTest, Small) {
    const auto rows_left = 11u;
    const auto common = 7u;
    const auto columns_right = 9u;
    correctness_test_random(rows_left, common, columns_right);
}

TEST(RandomCorrectnessTest, Medium) {
    const auto rows_left = (1u << 5) + 1u;
    const auto common = (1u << 4) + 3u;
    const auto columns_right = (1u << 6) + 1u;
    correctness_test_random(rows_left, common, columns_right);
}

TEST(RandomCorrectnessTest, Large) {
    const auto rows_left = (1u << 8) + 1u;
    const auto common = (1u << 7) + 3u;
    const auto columns_right = (1u << 6) + 1u;
    correctness_test_random(rows_left, common, columns_right);
}

TEST(SpeedTest, SevenElements) {
    speed_test(7u);
}

TEST(SpeedTest, ThirtyThreeElements) {
    speed_test(33u);
}

TEST(SpeedTest, OneThousandElements) {
    speed_test(1u << 7u);
}

TEST(SpeedTest, OneMillionElements) {
    speed_test(1u << 10u);
}
