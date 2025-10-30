find_package(nlohmann_json 3.2.0 REQUIRED)

add_library(utils STATIC
    operations.cpp
    integration.cpp
    logging.cpp
    boundary.cpp
    time_scheme.cpp
    errors.cpp
    matrix_helper.cpp
)

target_include_directories(utils PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_include_directories(utils PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(utils PUBLIC nlohmann_json::nlohmann_json)
