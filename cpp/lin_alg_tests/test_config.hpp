#pragma once

class TestConfig {
 public:
    static TestConfig& instance();
    unsigned int tile_edge = 1u;
};
