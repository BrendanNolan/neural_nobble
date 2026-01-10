#pragma once

#include <cassert>
#include <ostream>
#include <string>

#include "utils.h"

namespace lin_alg {

struct Dimension {
    unsigned int i = 0U;
    unsigned int j = 0U;
    bool operator==(const Dimension& other) const;
    bool operator!=(const Dimension& other) const;
};
std::string display(const Dimension& dim);

class Matrix {
 public:
    static Matrix zeroes(const Dimension& dim);
    static Matrix all_same(float entry, const Dimension& dim);
    static Matrix random(const Dimension& dim);
    static Matrix from_raw(float* entries, const Dimension& dim);
    ~Matrix();
    Dimension dim() const;
    float operator()(unsigned int i, unsigned int j) const;
    float& operator()(unsigned int i, unsigned int j);
    bool operator==(const Matrix& other) const;
    const float* raw() const;
    void scale(float scalar);
    Matrix& operator+=(const Matrix& other);
    unsigned int element_count() const;
 private:
    Matrix(float* entries, const Dimension& dim);
 private:
    float* data_ = nullptr;
    Dimension dim_;
};

unsigned int raw_size(const Matrix& m);
std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
bool admits_tile(const Matrix& matrix, unsigned int tile_size);
Matrix naive_multiply(const Matrix& a,
        const Op op_a,
        const float alpha,
        const Matrix& b,
        const Op op_b);

template <Op op_a, Op op_b>
Matrix tiled_multiply(const Matrix& a, const float alpha, const Matrix& b) {
    assert(a.dim().j == b.dim().i);
    assert(admits_tile(a, tile_size) && admits_tile(b, tile_size) && tile_size > 0U);
    auto M = a.dim().i;
    auto N = b.dim().j;
    auto K = a.dim().j;
    if constexpr (op_a == Op::Transpose) {
        M = a.dim().j;
        K = a.dim().i;
    }
    if constexpr (op_b == Op::Transpose) {
        N = b.dim().i;
    }
    const auto T = tile_size;
    auto C = Matrix::zeroes(Dimension{M, N});
    for (auto i = 0U; i < M; i += T) {
        for (auto j = 0U; j < N; j += T) {
            // top left of current C block is at (i,j)
            for (auto k = 0U; k < K; k += T) {
                for (auto ii = i; ii < std::min(i + T, M); ++ii) {
                    for (auto kk = k; kk < std::min(k + T, K); ++kk) {
                        for (auto jj = j; jj < std::min(j + T, N); ++jj) {
                            auto a_term = 0.0f;
                            if constexpr (op_a == Op::Transpose) {
                                a_term == a(kk, ii);
                            } else {
                                a_term = a(ii, kk);
                            }
                            auto b_term = 0.0f;
                            if constexpr (op_b == Op::Transpose) {
                                b_term = b(jj, kk);
                            } else {
                                b_term = b(kk, jj);
                            }
                            C(ii, jj) += alpha * a_term * b_term;
                        }
                    }
                }
            }
        }
    }
    return C;
}

}// namespace lin_alg
