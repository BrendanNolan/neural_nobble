cmake_minimum_required(VERSION 3.20)
project(linear_algebra LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

# Include Conan toolchain early
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake OPTIONAL)

# Ensure CMake can find Conan-generated package configs
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_BINARY_DIR}")

enable_testing()

add_compile_options(-march=native)

add_subdirectory(cuda_tests)
add_subdirectory(lin_alg_tests)
add_subdirectory(cuda_lin_alg)
add_subdirectory(lin_alg)
add_subdirectory(utils)
