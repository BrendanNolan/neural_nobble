cmake_minimum_required(VERSION 3.20)
project(linear_algebra LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include Conan toolchain early
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)

# Ensure CMake can find Conan-generated package configs
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")

enable_testing()

add_subdirectory(lin_alg)
add_subdirectory(cuda_lin_alg)
add_subdirectory(app)
add_subdirectory(tests)

