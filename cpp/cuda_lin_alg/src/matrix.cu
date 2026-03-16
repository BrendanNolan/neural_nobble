#include <cassert>
#include <iterator>
#include <utility>

#include "cuda_utils.h"
#include "matrix.h"
#include "utils.h"

__device__ __forceinline__ bool access_legal(const float* data,
        const unsigned int rows,
        const int columns,
        const unsigned int i,
        const unsigned int j,
        const Op op) {
    return (i < (op == Transpose) ? columns : rows) && (j < (op == Transpose) ? rows : columns);
}

__device__ __forceinline__ bool access_legal(const ConstMatrixDetails& matrix,
        const unsigned int i,
        const unsigned int j,
        const Op op = Identity) {
    return access_legal(matrix.data, matrix.rows, matrix.columns, i, j, op);
}

__device__ __forceinline__ bool access_legal(const MutableMatrixDetails& matrix,
        const unsigned int i,
        const unsigned int j,
        const Op op = Identity) {
    return access_legal(matrix.data, matrix.rows, matrix.columns, i, j, op);
}

struct Index {
    unsigned int i = 0u;
    unsigned int j = 0u;
};

template <Op op>
class GConstMatrixDetails {
 public:
    __device__ __forceinline__ GConstMatrixDetails(const ConstMatrixDetails* inner)
        : inner_{inner} {
    }
    __device__ __forceinline__ const float& at(const unsigned int i, const unsigned int j) const {
        if constexpr (op == Transpose) {
            return inner_->data[j * inner_->columns + j];
        } else {
            return inner_->data[i * inner_->columns + j];
        }
    }
    __device__ __forceinline__ unsigned int rows() const {
        if constexpr (op == Transpose) {
            return inner_->columns;
        } else {
            return inner_->rows;
        }
    }
    __device__ __forceinline__ unsigned int columns() const {
        if constexpr (op == Transpose) {
            return inner_->rows;
        } else {
            return inner_->columns;
        }
    }
 private:
    const ConstMatrixDetails* inner_ = nullptr;
};

template <Op op>
class GMutableMatrixDetails {
 public:
    __device__ __forceinline__ GMutableMatrixDetails(MutableMatrixDetails* inner)
        : inner_{inner} {
    }
    __device__ __forceinline__ float& at(const unsigned int i, const unsigned int j) {
        if constexpr (op == Transpose) {
            return inner_->data[j * inner_->columns + j];
        } else {
            return inner_->data[i * inner_->columns + j];
        }
    }
    __device__ __forceinline__ unsigned int rows() const {
        if constexpr (op == Transpose) {
            return inner_->columns;
        } else {
            return inner_->rows;
        }
    }
    __device__ __forceinline__ unsigned int columns() const {
        if constexpr (op == Transpose) {
            return inner_->rows;
        } else {
            return inner_->columns;
        }
    }
 private:
    MutableMatrixDetails* inner_ = nullptr;
};

template <Op op_A, Op op_B>
struct GGemmParams {
    GConstMatrixDetails<op_A> A;
    float alpha;
    GConstMatrixDetails<op_B> B;
    float beta;
    float* C;
};

// Deals with GConstMatrixDetails and GMutableMatrixDetails, sometimes for the performance
// optimisation of knowing the op at compile time, sometimes for the convenience of having the .at()
// method, which is not available on the ConstMatrixDetails and MutableMatrixDetails types since
// they must be extern C.
template <Op op_A, Op op_B>
__global__ void tiled_multiply(GGemmParams<op_A, op_B> params) {
    assert(blockDim.y == ratio_block_y_to_block_x * blockDim.x);
    assert(square_root_of_ratio_block_y_to_block_x * square_root_of_ratio_block_y_to_block_x
            == ratio_block_y_to_block_x);
    extern __shared__ float shared[];
    auto c_inner = MutableMatrixDetails{
            .data = params.C, .rows = params.A.rows(), .columns = params.B.columns()};
    auto C = GMutableMatrixDetails<Identity>{&c_inner};
    const auto T = blockDim.y;
    auto final_c_value = 0.0f;
    auto a_tile_inner = MutableMatrixDetails{.data = shared, .rows = T, .columns = T};
    auto a_tile = GMutableMatrixDetails<Identity>{&a_tile_inner};
    auto b_tile_inner = MutableMatrixDetails{.data = a_tile.data + T * T, .rows = T, .columns = T};
    auto b_tile = GMutableMatrixDetails<Identity>{&b_tile_inner};
    // Remember that cuda indexes grid/block rows with y and columns with x, like the x and y axes
    // of a graph, not the rows and columns of a matrix.
    for (auto row = 0u; row < ai; row += gridDim.y * blockDim.y) {
        for (auto column = 0u; column < bj;
                column += ratio_block_y_to_block_x * gridDim.x * blockDim.x) {
            const auto g_i = row + blockIdx.y * blockDim.y + threadIdx.y;
            const auto g_j = column + blockIdx.x * blockDim.x + threadIdx.x;
            auto g_ij = [&](const unsigned int mini_i, const unsigned int mini_j) -> Index {
                return Index{.i = g_i + mini_i, .j = g_j + mini_j};
            };
            auto tile_slot = [&](const unsigned int mini_i, const unsigned int mini_j) -> Index {
                return Index{.i = threadIdx.y + mini_i, .j = threadIdx.x + mini_j};
            };
            float final_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            auto final_c_values = MutableMatrixDetails{.data = final_data,
                    .rows = square_root_of_ratio_block_y_to_block_x,
                    .columns = square_root_of_ratio_block_y_to_block_x};
            for (auto k = 0u; k < aj; k += T) {
                for (auto kk = 0u; kk < bj; kk += T) {
                    for (auto mini_i = 0u; mini_i < square_root_of_ratio_block_y_to_block_x;
                            ++mini_i) {
                        for (auto mini_j = 0u; mini_j < square_root_of_ratio_block_y_to_block_x;
                                ++mini_j) {
                            a_tile[tile_slot] = in_scope_for_a ? a_at(g_i, k + threadIdx.x) : 0u;
                            b_tile[tile_slot] = in_scope_for_b ? b_at(k + threadIdx.y, g_j) : 0u;
                        }
                        __syncthreads();
                        for (auto mini_i = 0u; mini_i < square_root_of_ratio_block_y_to_block_x;
                                ++mini_i) {
                            for (auto mini_j = 0u; mini_j < square_root_of_ratio_block_y_to_block_x;
                                    ++mini_j) {
                                __syncthreads();
                            }
                        }
                    }
                }
                // Write back to main memory
            }
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
    assert(block_dim.y == ratio_block_y_to_block_x * block_dim.x);
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
    assert(block_dim().y == ratio_block_y_to_block_x * block_dim().x);
    return block_dim().y * block_dim().y * 2u * sizeof(float);
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
