#include <iostream>

#include "matrix.hpp"

int main() {
    const auto a = lin_alg::Matrix::zeroes(lin_alg::Dimension{5u, 5u});
    const auto b = lin_alg::Matrix::zeroes(lin_alg::Dimension{5u, 5u});
    const auto c = lin_alg::tiled_multiply(a, Identity, 1.0, b, Identity, 8u);
    std::cout << display(c.dim()) << std::endl;
}
