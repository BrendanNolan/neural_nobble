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

Matrix tiled_multiply(const Matrix& a,
        Op op_a,
        const float alpha,
        const Matrix& b,
        Op op_b,
        const unsigned int tile_size);

}// namespace lin_alg
