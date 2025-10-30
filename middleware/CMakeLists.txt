cmake_minimum_required(VERSION 3.9)
project(polivem_python)

# Find Python
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python: ${Python_VERSION}")

# Find packages
find_package(pybind11 REQUIRED)
message(STATUS "pybind11: ${pybind11_VERSION}")

# Find Eigen
find_package(Eigen3 REQUIRED)
if(NOT EIGEN3_FOUND)
    message(FATAL_ERROR "Eigen3 not found. Please install it.")
endif()

# Print Eigen include directories for debugging
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/middleware/include
    ${CMAKE_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
    /opt/homebrew/include/eigen3
)

# Create Python module
pybind11_add_module(polivem_py
    src/polivem_module.cpp
    src/beam_wrapper.cpp
    src/solver_wrapper.cpp
    src/enums_wrapper.cpp
    src/material_wrapper.cpp
)

# Link against the component libraries instead of the executable
target_link_libraries(polivem_py PRIVATE 
    utils
    solver
    material
    mesh
)

# Set the output directory for the Python module
set_target_properties(polivem_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")

# Add the minimal test module
pybind11_add_module(minimal_py
    src/minimal_module.cpp
)

# Set the output directory for the minimal test module
set_target_properties(minimal_py PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")

# Install the Python module
install(TARGETS polivem_py LIBRARY DESTINATION "${Python_SITELIB}")