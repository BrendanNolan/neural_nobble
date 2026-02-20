#pragma once

#include <cassert>
#include <ostream>
#include <string>
#include <vector>

#include "utils.h"

namespace lin_alg {

struct Dimension {
    unsigned int rows = 0u;
    unsigned int columns = 0u;
    bool operator==(const Dimension& other) const;
    bool operator!=(const Dimension& other) const;
    unsigned int size() const {
        return rows * columns;
    }
};
std::string display(const Dimension& dim);

class Matrix {
 public:
    static Matrix zeroes(const Dimension& dim);
    static Matrix all_same(float entry, const Dimension& dim);
    static Matrix random(const Dimension& dim);
    static Matrix from_raw(std::vector<float> data, const Dimension& dim);
    void transpose();
    Dimension dim() const;
    float operator()(unsigned int i, unsigned int j) const {
        return data_[i * dim_.columns + j];
    }
    float& operator()(unsigned int i, unsigned int j) {
        return data_[i * dim_.columns + j];
    }
    bool operator==(const Matrix& other) const;
    const float* raw() const;
    void scale(float scalar);
    Matrix& operator+=(const Matrix& other);
    unsigned int element_count() const;
 private:
    Matrix(std::vector<float> data, const Dimension& dim);
 private:
    std::vector<float> data_;
    Dimension dim_;
};

bool can_multiply(const Matrix& a, const Op op_a, const Matrix& b, const Op op_b);

std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

bool admits_tile(const Matrix& matrix, unsigned int tile_size);

Matrix naive_multiply(Matrix a, const Op op_a, const float alpha, Matrix b, const Op op_b);

Matrix tiled_multiply(const Matrix& a,
        Op op_a,
        float alpha,
        const Matrix& b,
        Op op_b,
        unsigned int tile_size);

}// namespace lin_alg
