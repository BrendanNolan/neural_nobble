#pragma once

#include "cuda_utils.h"
#include "utils.h"

struct SumReduceLaunchConfig {
    unsigned int grid_dim_x;
    unsigned int block_dim_x;
};

SumReduceLaunchConfig compute_sum_reduce_launch_config(unsigned int input_length);

void run_sum_reduce(float* input, unsigned int length, float* result);
