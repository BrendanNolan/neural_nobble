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

void Matrix::transpose() {
    auto* new_data = static_cast<float*>(malloc(dim_.size() * sizeof(float)));
    dim_ = Dimension{.i = dim_.j, .j = dim_.i};
    for (auto i = 0U; i < dim_.i; ++i) {
        for (auto j = 0U; j < dim_.j; ++j) {
            new_data[i * dim_.j + j] = data_[j * dim_.i + i];
        }
    }
    auto* old_data = data_;
    data_ = new_data;
    free(old_data);
}

Matrix::~Matrix() {
    free(data_);
}

Matrix::Matrix(const Matrix& other) {
    dim_ = other.dim_;
    data_ = static_cast<float*>(malloc(dim_.size() * sizeof(float)));
    for (auto i = 0U; i < dim_.size(); ++i) {
        data_[i] = other.data_[i];
    }
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

namespace {
bool almost_equal(const float a, const float b) {
    constexpr auto TOLERANCE = 1e-4f;
    using namespace std;
    return abs(a - b) / max(abs(a), abs(b)) < TOLERANCE;
}
}// namespace

bool Matrix::operator==(const Matrix& other) const {
    if (this->dim() != other.dim())
        return false;
    for (auto row = 0U; row < this->dim().i; ++row) {
        for (auto column = 0U; column < this->dim().j; ++column) {
            if (!almost_equal((*this)(row, column), other(row, column))) {
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
    auto c = Matrix::zeroes(Dimension{.i = (op_a == Transpose ? a.dim().j : a.dim().i),
            .j = (op_b == Transpose ? b.dim().i : b.dim().j)});
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

bool can_multiply(const Matrix& a, const Op op_a, const Matrix& b, const Op op_b) {
    if (op_a == Identity && op_b == Identity) {
        return a.dim().j == b.dim().i;
    }
    if (op_a == Identity && op_b == Transpose) {
        return a.dim().j == b.dim().j;
    }
    if (op_a == Transpose && op_b == Identity) {
        return a.dim().i == b.dim().i;
    }
    if (op_a == Transpose && op_b == Transpose) {
        return a.dim().i == b.dim().j;
    }
    assert(false && "Missed a case");
    return false;
}

}// namespace lin_alg
