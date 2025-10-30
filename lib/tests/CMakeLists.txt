find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(tests STATIC
    test_time_scheme.cpp
    test_linear_algebra.cpp
)

target_include_directories(tests PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/include/tests
)

target_link_libraries(tests PUBLIC 
    nlohmann_json::nlohmann_json
    utils
    solver
)
