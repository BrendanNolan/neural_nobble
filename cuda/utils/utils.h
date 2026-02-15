#pragma once

#include <cstdint>

extern "C" {
typedef uint8_t Op;
enum { Identity, Transpose };
}

inline unsigned int cover_divide(const unsigned int numerator, const unsigned int denominator) {
    return (numerator + denominator - 1U) / denominator;
}
