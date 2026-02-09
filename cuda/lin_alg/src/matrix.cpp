#include "matrix.hpp"

#include <algorithm>
#include <cstdlib>
#include <random>
#include <utility>

namespace lin_alg {

std::string display(const Dimension& dim) {
    return "(" + std::to_string(dim.rows) + ", " + std::to_string(dim.columns) + ")";
}

bool Dimension::operator==(const Dimension& other) const {
    return this->rows == other.rows && this->columns == other.columns;
}

bool Dimension::operator!=(const Dimension& other) const {
    return !(*this == other);
}

Matrix::Matrix(std::vector<float> data, const Dimension& dim)
    : data_{std::move(data)}
    , dim_{dim} {
}

Matrix Matrix::from_raw(std::vector<float> impl, const Dimension& dim) {
    return Matrix{impl, dim};
}

void Matrix::transpose() {
    for (auto i = 0U; i < dim_.rows; ++i) {
        for (auto j = 0U; j < dim_.columns; ++j) {
            const auto current_index = i * dim_.columns + j;
            const auto target_index = j * dim_.rows + i;
            if (current_index < target_index)
                std::swap(data_[current_index], data_[target_index]);
        }
    }
    dim_ = Dimension{.rows = dim_.columns, .columns = dim_.rows};
}

Matrix Matrix::zeroes(const Dimension& dim) {
    return Matrix{std::vector<float>(dim.size(), 0.0f), dim};
}

Matrix Matrix::all_same(float entry, const Dimension& dim) {
    auto entries = std::vector<float>(dim.size(), 0.0f);
    for (auto index = 0U; index < dim.rows * dim.columns; ++index) {
        entries[index] = entry;
    }
    return Matrix{entries, dim};
}

Matrix Matrix::random(const Dimension& dim) {
    std::mt19937 gen(147);
    std::uniform_int_distribution<> dist(0, 100);
    auto matrix = Matrix::zeroes(dim);
    for (auto i = 0U; i < dim.rows; ++i) {
        for (auto j = 0U; j < dim.columns; ++j) {
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
    constexpr auto ABSOLUTE_TOLERANCE = 1e-6f;
    constexpr auto RELATIVE_TOLERANCE = 1e-4f;
    using namespace std;
    const auto largest_absolute = max(abs(a), abs(b));
    return abs(a - b) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * largest_absolute);
}
}// namespace

bool Matrix::operator==(const Matrix& other) const {
    if (this->dim() != other.dim())
        return false;
    for (auto row = 0U; row < this->dim().rows; ++row) {
        for (auto column = 0U; column < this->dim().columns; ++column) {
            if (!almost_equal((*this)(row, column), other(row, column))) {
                return false;
            }
        }
    }
    return true;
}

const float* Matrix::raw() const {
    return data_.data();
}

unsigned int Matrix::element_count() const {
    return dim().rows * dim().columns;
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
    if (matrix.dim().rows == 0U || matrix.dim().columns == 0U) {
        return os;
    }
    for (auto i = 0U; i < matrix.dim().rows; ++i) {
        for (auto j = 0U; j < matrix.dim().columns; ++j) {
            os << matrix(i, j) << ' ';
        }
        os << std::endl;
    }
    return os;
}

unsigned int raw_size(const Matrix& matrix) {
    return matrix.dim().rows * matrix.dim().columns;
}

bool admits_tile(const Matrix& matrix, unsigned int tile_size) {
    const auto dim = matrix.dim();
    return tile_size > 0U && tile_size <= dim.rows && tile_size <= dim.columns;
}

Matrix naive_multiply(Matrix a, const Op op_a, const float alpha, Matrix b, const Op op_b) {
    if (op_a == Transpose) {
        a.transpose();
    }
    if (op_b == Transpose) {
        b.transpose();
    }
    auto c = Matrix::zeroes(Dimension{.rows = a.dim().rows, .columns = b.dim().columns});
    for (auto i = 0U; i < a.dim().rows; ++i) {
        for (auto j = 0U; j < b.dim().columns; ++j) {
            for (auto k = 0U; k < a.dim().columns; ++k) {
                c(i, j) += alpha * a(i, k) * b(k, j);
            }
        }
    }
    return c;
}

bool can_multiply(const Matrix& a, const Op op_a, const Matrix& b, const Op op_b) {
    if (op_a == Identity && op_b == Identity) {
        return a.dim().columns == b.dim().rows;
    }
    if (op_a == Identity && op_b == Transpose) {
        return a.dim().columns == b.dim().columns;
    }
    if (op_a == Transpose && op_b == Identity) {
        return a.dim().rows == b.dim().rows;
    }
    if (op_a == Transpose && op_b == Transpose) {
        return a.dim().rows == b.dim().columns;
    }
    assert(false && "Missed a case");
    return false;
}

}// namespace lin_alg
