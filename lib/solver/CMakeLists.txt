find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(solver STATIC
    beam1d.cpp
    linearElastic2d.cpp
    nonlinear2d.cpp
    axisymmetric.cpp
    parabolic.cpp
)

target_include_directories(solver PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(solver PUBLIC nlohmann_json::nlohmann_json)
target_include_directories(solver PRIVATE ${CMAKE_SOURCE_DIR}/include)
