find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

file(GLOB LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
file(GLOB LIB_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
add_compile_options(-Wall -Wextra)
add_library(cuda_lin_alg SHARED ${LIB_SOURCES} ${LIB_HEADERS})
set_target_properties(cuda_lin_alg PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(cuda_lin_alg
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(cuda_lin_alg
    PRIVATE lin_alg
    PRIVATE CUDA::cudart)

