#include "test_config.hpp"

TestConfig& TestConfig::instance() {
    static auto the_instance = TestConfig{};
    return the_instance;
}
