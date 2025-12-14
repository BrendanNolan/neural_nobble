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

Matrix naive_multiply(const Matrix& a, const Matrix& b) {
    assert(a.dim().j == b.dim().i);
    auto c = Matrix::zeroes(Dimension{a.dim().i, b.dim().j});
    for (auto i = 0U; i < a.dim().i; ++i) {
        for (auto j = 0U; j < b.dim().j; ++j) {
            for (auto k = 0U; k < a.dim().j; ++k) {
                c(i, j) += a(i, k) * b(k, j);
            }
        }
    }
    return c;
}

Matrix tiled_multiply(const Matrix& a, const Matrix& b, const unsigned int tile_size) {
    assert(a.dim().j == b.dim().i);
    assert(admits_tile(a, tile_size) && admits_tile(b, tile_size) && tile_size > 0U);
    const auto M = a.dim().i;
    const auto N = b.dim().j;
    const auto K = a.dim().j;
    const auto T = tile_size;
    auto C = Matrix::zeroes(Dimension{a.dim().i, b.dim().j});
    for (auto i = 0U; i < M; i += T) {
        for (auto j = 0U; j < N; j += T) {
            // top left of current C block is at (i,j)
            for (auto k = 0U; k < K; k += T) {
                for (auto ii = i; ii < std::min(i + T, M); ++ii) {
                    for (auto kk = k; kk < std::min(k + T, K); ++kk) {
                        for (auto jj = j; jj < std::min(j + T, N); ++jj) {
                            C(ii, jj) += a(ii, kk) * b(kk, jj);
                        }
                    }
                }
            }
        }
    }
    return C;
}

}// namespace lin_alg
