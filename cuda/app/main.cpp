#include <iostream>

#include "matrix.hpp"

int main() {
    const auto a = lin_alg::Matrix::zeroes(lin_alg::Dimension{5U, 5U});
    const auto b = lin_alg::Matrix::zeroes(lin_alg::Dimension{5U, 5U});
    const auto c = lin_alg::tiled_multiply<Identity, Identity>(a, 1.0, b, 1U);
    std::cout << display(c.dim()) << std::endl;
}
