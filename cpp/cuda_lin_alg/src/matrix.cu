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
    __device__ __forceinline__ Index operator+(const Index& other) const {
        return Index{.i = i + other.i, .j = j + other.j};
    }
    __device__ __forceinline__ Index operator*(const unsigned int factor) const {
        return Index{.i = i * factor, .j = j * factor};
    }
};

enum class MatrixLayout { row_major, column_major };

template <Op op>
class GConstMatrixDetails {
 public:
    __host__ __device__ __forceinline__ GConstMatrixDetails(ConstMatrixDetails inner)
        : inner_{inner} {
    }
    __device__ __forceinline__ const float* data() const {
        return inner_.data;
    }
    __device__ __forceinline__ const float* operator()(const unsigned int i,
            const unsigned int j) const {
        if constexpr (op == Transpose) {
            return &inner_.data[j * inner_.columns + j];
        } else {
            return &inner_.data[i * inner_.columns + j];
        }
    }
    __device__ __forceinline__ bool access_legal(const unsigned int i, const unsigned int j) const {
        return ::access_legal(inner_, i, j, op);
    }
    __host__ __device__ __forceinline__ unsigned int rows() const {
        if constexpr (op == Transpose) {
            return inner_.columns;
        } else {
            return inner_.rows;
        }
    }
    __host__ __device__ __forceinline__ unsigned int columns() const {
        if constexpr (op == Transpose) {
            return inner_.rows;
        } else {
            return inner_.columns;
        }
    }
    __device__ __forceinline__ static constexpr MatrixLayout layout() {
        if constexpr (op == Transpose) {
            return MatrixLayout::column_major;
        } else {
            return MatrixLayout::row_major;
        }
    }
 private:
    ConstMatrixDetails inner_;
};

__device__ ConstMatrixDetails to_const(const MutableMatrixDetails& matrix) {
    return ConstMatrixDetails{.data = matrix.data, .rows = matrix.rows, .columns = matrix.columns};
}

template <Op op>
class GMutableMatrixDetails {
 public:
    __host__ __device__ __forceinline__ GMutableMatrixDetails(MutableMatrixDetails inner)
        : inner_{inner} {
    }
    __device__ GConstMatrixDetails<op> to_const() const {
        return GConstMatrixDetails<op>{::to_const(inner_)};
    }
    __device__ __forceinline__ float* data() {
        return inner_.data;
    }
    __device__ __forceinline__ float* operator()(const unsigned int i, const unsigned int j) {
        if constexpr (op == Transpose) {
            return &inner_.data[j * inner_.columns + j];
        } else {
            return &inner_.data[i * inner_.columns + j];
        }
    }
    __device__ __forceinline__ bool access_legal(const unsigned int i, const unsigned int j) const {
        return ::access_legal(inner_, i, j, op);
    }
    __host__ __device__ __forceinline__ unsigned int rows() const {
        if constexpr (op == Transpose) {
            return inner_.columns;
        } else {
            return inner_.rows;
        }
    }
    __host__ __device__ __forceinline__ unsigned int columns() const {
        if constexpr (op == Transpose) {
            return inner_.rows;
        } else {
            return inner_.columns;
        }
    }
    __device__ __forceinline__ static constexpr MatrixLayout layout() {
        if constexpr (op == Transpose) {
            return MatrixLayout::column_major;
        } else {
            return MatrixLayout::row_major;
        }
    }
 private:
    MutableMatrixDetails inner_;
};

template <Op op_A, Op op_B>
struct GGemmParams {
    GConstMatrixDetails<op_A> A;
    float alpha;
    GConstMatrixDetails<op_B> B;
    float beta;
    GMutableMatrixDetails<Identity> C;
};

template <Op op_A, Op op_B>
__device__ __forceinline__ void register_multiply(const GConstMatrixDetails<op_A>& A,
        const GConstMatrixDetails<op_B>& B,
        GMutableMatrixDetails<Identity>& C) {
}

template <Op op_source, Op op_target>
__device__ __forceinline__ void wide_load(const GConstMatrixDetails<op_source>& source,
        const Index source_index,
        GMutableMatrixDetails<op_target>& target,
        const Index target_index) {
    for (auto offset = 0u; offset < target_elements_per_thread; ++offset) {
        if (!target.access_legal(target_index.i + offset, target_index.j)) {
            return;
        }
        if (source.access_legal(source_index.i + offset, source_index.j)) {
            *target(target_index.i + offset, target_index.j) =
                    *source(source_index.i + offset, source_index.j);
        } else {
            *target(target_index.i + offset, target_index.j) = 0.0f;
        }
    }
}

// Deals with GConstMatrixDetails and GMutableMatrixDetails, sometimes for the performance
// optimisation of knowing the op at compile time, sometimes for the convenience of having the .at()
// method, which is not available on the ConstMatrixDetails and MutableMatrixDetails types since
// they must be extern C.
template <Op op_A, Op op_B>
__global__ void tiled_multiply(GGemmParams<op_A, op_B> params) {
    assert(blockDim.y == target_elements_per_thread * blockDim.x);
    assert(square_root_of_target_elements_per_thread * square_root_of_target_elements_per_thread
            == target_elements_per_thread);
    extern __shared__ float shared[];
    const auto T = blockDim.y;
    auto a_tile_inner = MutableMatrixDetails{.data = shared, .rows = T, .columns = T};
    auto a_tile = GMutableMatrixDetails<Identity>{a_tile_inner};
    auto b_tile_inner = MutableMatrixDetails{.data = shared + T * T, .rows = T, .columns = T};
    auto b_tile = GMutableMatrixDetails<Identity>{b_tile_inner};
    // Remember that cuda indexes grid/block rows with y and columns with x, like the x and y axes
    // of a graph, not the rows and columns of a matrix.
    float final_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    auto final_inner = MutableMatrixDetails{.data = &final_data[0], .rows = 2u, .columns = 2u};
    auto final = GMutableMatrixDetails<Identity>{final_inner};
    for (auto row = 0u; row < params.A.rows(); row += gridDim.y * blockDim.y) {
        for (auto column = 0u; column < params.B.columns();
                column += target_elements_per_thread * gridDim.x * blockDim.x) {
            for (auto k = 0u; k < params.A.columns(); k += T) {
                const auto tile_load_index =
                        Index{.i = threadIdx.y, .j = threadIdx.x * target_elements_per_thread};
                // const auto global_row = row + blockIdx.y * blockDim.y + tile_row;
                // const auto global_column = column + target_elements_per_thread * blockIdx.x *
                // blockDim.x + tile_column;
                const auto a_tile_corner = Index{.i = row + blockIdx.y * blockDim.y, .j = k * T};
                const auto b_tile_corner = Index{.i = k * T,
                        .j = column + blockIdx.x * blockDim.x * target_elements_per_thread};
                wide_load(params.A, a_tile_corner + tile_load_index, a_tile, tile_load_index);
                wide_load(params.B, b_tile_corner + tile_load_index, b_tile, tile_load_index);
                __syncthreads();
                register_multiply(a_tile.to_const(), b_tile.to_const(), final);
            }
        }
    }
    // Write back to main memory
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
    if (params.op_A == Identity && params.op_B == Identity) {
        const auto A = GConstMatrixDetails<Identity>{params.A};
        const auto B = GConstMatrixDetails<Identity>{params.B};
        auto C_inner =
                MutableMatrixDetails{.data = params.C, .rows = A.rows(), .columns = B.columns()};
        auto g_params = GGemmParams<Identity, Identity>{.A = A,
                .alpha = params.alpha,
                .B = B,
                .beta = params.beta,
                .C = GMutableMatrixDetails<Identity>{C_inner}};
        tiled_multiply<<<cuda_grid, cuda_block, shared_mem_size>>>(g_params);
    }
    cudaDeviceSynchronize();
}

std::optional<GemmLaunchConfig> GemmLaunchConfig::create(const dim3& grid_dim,
        const dim3& block_dim) {
    assert(block_dim.y == target_elements_per_thread * block_dim.x);
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
    assert(block_dim().y == target_elements_per_thread * block_dim().x);
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
