find_package(nlohmann_json 3.2.0 REQUIRED)

# Create a library named 'mesh' that includes the specified source files
add_library(mesh STATIC
    beam.cpp
    datasource.cpp
    delaunay.cpp
    helpers.cpp
    voronoi.cpp
    voronoiMesh.cpp
    uniform.cpp
)

# Include the current source directory to allow header file includes
target_include_directories(mesh PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(mesh PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

target_include_directories(mesh PRIVATE ${CMAKE_SOURCE_DIR}/include)
