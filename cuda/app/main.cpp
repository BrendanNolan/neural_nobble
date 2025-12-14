#include <iostream>

#include "matrix.hpp"

int main() {
    const auto a = lin_alg::Matrix::zeroes(lin_alg::Dimension{5U, 5U});
    const auto b = lin_alg::Matrix::zeroes(lin_alg::Dimension{5U, 5U});
    const auto c = tiled_multiply(a, b, 1U);
    std::cout << display(c.dim()) << std::endl;
}
