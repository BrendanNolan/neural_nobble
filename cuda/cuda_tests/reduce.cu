#include "cuda_utils.h"
#include "float_utils.h"
#include "gpu_matrix.h"

#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdlib.h>

TEST(ReductionTest, Basic) {
    const auto total_length = 1U << 16U;
    const auto grid_length = 256U;
    const auto block_length = 256U;
    auto* input = allocate_on_device(total_length);
    const auto input_host = std::vector<float>(total_length, 1.0f);
    copy_to_device(input_host.data(), input_host.size(), input);
    auto* result_d = allocate_on_device(1U);
    launch_sum_reduction(input, total_length, result_d, grid_length, block_length);
    auto result = 0.0f;
    copy_from_device(result_d, 1U, &result);
    EXPECT_TRUE(almost_equal(result, total_length));
}
