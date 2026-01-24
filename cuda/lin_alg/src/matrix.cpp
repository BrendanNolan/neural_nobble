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

}// namespace lin_alg
