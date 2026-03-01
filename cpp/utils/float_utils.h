#include <algorithm>

inline bool almost_equal(const float a, const float b) {
    constexpr auto ABSOLUTE_TOLERANCE = 1e-6f;
    constexpr auto RELATIVE_TOLERANCE = 1e-4f;
    using namespace std;
    const auto largest_absolute = max(abs(a), abs(b));
    return abs(a - b) <= max(ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE * largest_absolute);
}
