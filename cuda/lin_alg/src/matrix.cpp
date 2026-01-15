#include "matrix.hpp"

#include <algorithm>
#include <cstdlib>
#include <random>

namespace lin_alg {

std::string display(const Dimension& dim) {
    return "(" + std::to_string(dim.i) + ", " + std::to_string(dim.j) + ")";
}

bool Dimension::operator==(const Dimension& other) const {
    return this->i == other.i && this->j == other.j;
}

bool Dimension::operator!=(const Dimension& other) const {
    return !(*this == other);
}

Matrix::Matrix(float* impl, const Dimension& dim)
    : data_{impl}
    , dim_{dim} {
}

Matrix Matrix::from_raw(float* impl, const Dimension& dim) {
    return Matrix{impl, dim};
}

Matrix::~Matrix() {
    free(data_);
}

Matrix Matrix::zeroes(const Dimension& dim) {
    return Matrix{static_cast<float*>(calloc(dim.i * dim.j, sizeof(float))), dim};
}

Matrix Matrix::all_same(float entry, const Dimension& dim) {
    auto* entries = static_cast<float*>(malloc(dim.i * dim.j * sizeof(float)));
    for (auto index = 0U; index < dim.i * dim.j; ++index) {
        entries[index] = entry;
    }
    return Matrix{entries, dim};
}

float Matrix::operator()(const unsigned int i, const unsigned int j) const {
    return data_[i * dim_.j + j];
}

float& Matrix::operator()(const unsigned int i, const unsigned int j) {
    return data_[i * dim_.j + j];
}

Matrix Matrix::random(const Dimension& dim) {
    std::mt19937 gen(147);
    std::uniform_int_distribution<> dist(0, 100);
    auto matrix = Matrix::zeroes(dim);
    for (auto i = 0U; i < dim.i; ++i) {
        for (auto j = 0U; j < dim.j; ++j) {
            matrix(i, j) = dist(gen);
        }
    }
    return matrix;
}

Dimension Matrix::dim() const {
    return dim_;
}

bool Matrix::operator==(const Matrix& other) const {
    constexpr auto tolerance = 0.00000001f;
    if (this->dim() != other.dim())
        return false;
    for (auto row = 0U; row < this->dim().i; ++row) {
        for (auto column = 0U; column < this->dim().j; ++column) {
            if (std::abs((*this)(row, column) - other(row, column)) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

const float* Matrix::raw() const {
    return data_;
}

unsigned int Matrix::element_count() const {
    return dim().i * dim().j;
}

void Matrix::scale(const float scalar) {
    for (auto index = 0U; index < element_count(); ++index)
        data_[index] *= scalar;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    const auto* other_raw = other.raw();
    for (auto index = 0U; index < element_count(); ++index)
        data_[index] += other_raw[index];
    return *this;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << std::endl;
    if (matrix.dim().i == 0U || matrix.dim().j == 0U) {
        return os;
    }
    for (auto i = 0U; i < matrix.dim().i; ++i) {
        for (auto j = 0U; j < matrix.dim().j; ++j) {
            os << matrix(i, j) << ' ';
        }
        os << std::endl;
    }
    return os;
}

unsigned int raw_size(const Matrix& matrix) {
    return matrix.dim().i * matrix.dim().j;
}

bool admits_tile(const Matrix& matrix, unsigned int tile_size) {
    const auto dim = matrix.dim();
    return tile_size > 0U && tile_size <= dim.i && tile_size <= dim.j;
}

Matrix naive_multiply(const Matrix& a,
        const Op op_a,
        const float alpha,
        const Matrix& b,
        const Op op_b) {
    assert(a.dim().j == b.dim().i);
    auto c = Matrix::zeroes(Dimension{a.dim().i, b.dim().j});
    auto element =
            [](const Matrix& matrix, const Op op, const unsigned int i, const unsigned int j) {
                return op == Transpose ? matrix(j, i) : matrix(i, j);
            };
    for (auto i = 0U; i < a.dim().i; ++i) {
        for (auto j = 0U; j < b.dim().j; ++j) {
            for (auto k = 0U; k < a.dim().j; ++k) {
                c(i, j) += alpha * element(a, op_a, i, k) * element(b, op_b, k, j);
            }
        }
    }
    return c;
}

namespace {
enum class TransposeStrategy { neither, a_only, b_only, both };

TransposeStrategy get_transpose_strategy(const Op op_a, const Op op_b) {
    if (op_a == Transpose && op_b == Transpose) {
        return TransposeStrategy::both;
    } else if (op_a == Transpose && op_b == Identity) {
        return TransposeStrategy::a_only;
    } else if (op_a == Identity && op_b == Transpose) {
        return TransposeStrategy::b_only;
    } else {
        return TransposeStrategy::neither;
    }
}
}// namespace

Matrix tiled_multiply(const Matrix& a,
        Op op_a,
        const float alpha,
        const Matrix& b,
        Op op_b,
        const unsigned int tile_size) {
    auto M = a.dim().i;
    auto N = b.dim().j;
    auto K = a.dim().j;
    if (op_a == Transpose) {
        M = a.dim().j;
        K = a.dim().i;
    }
    if (op_b == Transpose) {
        N = b.dim().i;
    }
    const auto T = tile_size;
    auto C = Matrix::zeroes(Dimension{M, N});
    const auto transpose_strategy = get_transpose_strategy(op_a, op_b);
    for (auto i = 0U; i < M; i += T) {
        for (auto j = 0U; j < N; j += T) {
            // top left of current C block is at (i,j)
            for (auto k = 0U; k < K; k += T) {
                for (auto ii = i; ii < std::min(i + T, M); ++ii) {
                    for (auto kk = k; kk < std::min(k + T, K); ++kk) {
                        for (auto jj = j; jj < std::min(j + T, N); ++jj) {
                            switch (transpose_strategy) {
                            case TransposeStrategy::neither:
                                C(ii, jj) += alpha * a(ii, kk) * b(kk, jj);
                                break;
                            case TransposeStrategy::a_only:
                                C(ii, jj) += alpha * a(kk, ii) * b(kk, jj);
                                break;
                            case TransposeStrategy::b_only:
                                C(ii, jj) += alpha * a(ii, kk) * b(jj, kk);
                                break;
                            case TransposeStrategy::both:
                                C(ii, jj) += alpha * a(kk, ii) * b(jj, kk);
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    return C;
}

}// namespace lin_alg
