find_package(GTest CONFIG REQUIRED)

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)

add_compile_options(-Wall -Wextra)
add_executable(lin_alg_tests ${TEST_SOURCES})

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON) # Ensure that the tests find libcuda_lin_alg.so at runtime
target_link_libraries(lin_alg_tests
    PRIVATE
        lin_alg
        cuda_lin_alg
        GTest::gtest
)

target_include_directories(lin_alg_tests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/lin_alg/src
)

include(GoogleTest)
gtest_discover_tests(lin_alg_tests)

