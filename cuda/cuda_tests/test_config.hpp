#pragma once

class TestConfig {
 public:
    static TestConfig& instance();
    unsigned int block_edge = 1U;
};
