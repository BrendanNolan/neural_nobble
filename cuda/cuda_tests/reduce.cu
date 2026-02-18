#include "cuda_utils.h"
#include "float_utils.h"
#include "gpu_matrix.h"

#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <stdlib.h>

TEST(ReductionTest, Basic) {
    const auto total_length = 256U * 256U;
    auto* input = allocate_on_device(total_length);
    const auto input_host = std::vector<float>(total_length, 1.0f);
    copy_to_device(input_host.data(), input_host.size(), input);
    const auto grid_x = 256U;
    const auto block_x = 256U;
    auto* result_d = allocate_on_device(1U);
    run_sum_reduce(input, total_length, result_d, grid_x, block_x);
    auto result = 0.0f;
    copy_from_device(result_d, 1U, &result);
    const auto expected = static_cast<float>(total_length);
    EXPECT_TRUE(almost_equal(result, expected)) << "Expected " << expected << " Got " << result;
}
