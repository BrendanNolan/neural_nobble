add_compile_options(-Wall -Wextra -Werror)
add_executable(lin_alg_exec main.cpp)

target_link_libraries(lin_alg_exec
    PRIVATE
        lin_alg
)

