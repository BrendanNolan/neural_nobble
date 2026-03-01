file(GLOB LIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
file(GLOB LIB_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
add_compile_options(-Wall -Wextra -Werror)
add_library(lin_alg ${LIB_SOURCES} ${LIB_HEADERS})
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_target_properties(lin_alg PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
endif()
target_include_directories(lin_alg
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(lin_alg
    PUBLIC utils)
