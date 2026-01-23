#include <iostream>

#include "matrix.hpp"

int main() {
    const auto a = lin_alg::Matrix::zeroes(lin_alg::Dimension{5U, 5U});
    const auto b = lin_alg::Matrix::zeroes(lin_alg::Dimension{5U, 5U});
    const auto c = lin_alg::tiled_multiply(a, Op::Identity, 1.0, b, Op::Identity, 1U);
    std::cout << display(c.dim()) << std::endl;
}
