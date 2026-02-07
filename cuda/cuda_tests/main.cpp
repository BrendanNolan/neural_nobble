#include "test_config.hpp"
#include <cassert>
#include <gtest/gtest.h>
#include <string>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Now argc/argv contain only custom args
    for (int i = 1; i < argc; ++i) {
        const auto arg_name = std::string{argv[i]};
        if (arg_name == "--block-edge") {
            ++i;
            assert(i < argc);
            TestConfig::instance().block_edge = static_cast<unsigned int>(std::stoi(argv[i]));
            break;
        }
    }

    return RUN_ALL_TESTS();
}
