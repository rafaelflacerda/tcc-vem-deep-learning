cmake_minimum_required(VERSION 3.9.1)
set(CMAKE_CXX_STANDARD 17)
project(polivem)

# Add new Accelerate flags for macOS
if(APPLE)
    add_definitions(-DUSE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64)
endif()

# Define the output paths
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}/lib)

# Add executable target with given name
add_executable(${PROJECT_NAME} main.cpp)

# Add proprietary lib
# add_library(mesh SHARED lib/mesh/datasource.cpp)
add_library(material SHARED lib/material/mat.cpp)

# Check if library exists on the computer before building executable
find_package(Eigen3)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(OpenCV REQUIRED)

# Include library headers
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

# Add subdirectories
add_subdirectory(lib/utils)
add_subdirectory(lib/solver)
add_subdirectory(lib/mesh)
add_subdirectory(middleware)
add_subdirectory(lib/tests)
add_subdirectory(lib/linear_algebra)

# Add a library into the linker's command line
target_link_libraries(${PROJECT_NAME} 
    PUBLIC 
        ${Eigen3_LIBRARIES} 
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
    PRIVATE 
        utils 
        solver 
        material
        mesh 
        tests
        linear_algebra
)

# Include directories
target_include_directories(solver PRIVATE utils)
target_include_directories(utils PRIVATE solver)
target_include_directories(mesh PRIVATE solver)
target_include_directories(tests PRIVATE utils)
target_include_directories(tests PRIVATE solver)
target_include_directories(tests PRIVATE mesh)
target_include_directories(tests PRIVATE linear_algebra)

# Include models directory
target_include_directories(${PROJECT_NAME} PRIVATE models)
target_include_directories(solver PRIVATE models)
target_include_directories(material PRIVATE models)
target_include_directories(mesh PRIVATE models)
target_include_directories(utils PRIVATE models)
target_include_directories(tests PRIVATE models)
target_include_directories(tests PRIVATE linear_algebra)

# Add the include directory to the include path
include_directories(${CMAKE_SOURCE_DIR}/include)

# Update target include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(solver PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(material PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(mesh PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(utils PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include/linear_algebra)

# Checking property of the target
message("[DEBUG] CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

get_target_property(inc_dir Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
message("[DEBUG] inc_dir: ${inc_dir}")
message("[DEBUG] EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")

message("[DEBUG] EIGEN3_INCLUDE_DIRS: ${nlohmann_json_INCLUDE_DIRS}")

# Simple beam example
add_executable(beam_simple examples/main_simple.cpp)
target_link_libraries(beam_simple 
    mesh 
    solver 
    material
    utils
    Eigen3::Eigen
)
target_include_directories(beam_simple PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Dataset generator for ML
add_executable(generate_dataset examples/generate_dataset.cpp)
target_link_libraries(generate_dataset 
    mesh 
    solver 
    material
    utils
    Eigen3::Eigen
)
target_include_directories(generate_dataset PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(run_generate_dataset examples/generate_dataset.cpp)
target_link_libraries(run_generate_dataset solver mesh material utils Eigen3::Eigen)

# === Executável de teste: run_single_case ===
add_executable(run_single_case examples/run_single_case.cpp)

# Vincular às bibliotecas internas do projeto
target_link_libraries(run_single_case
    mesh
    solver
    material
    utils
    linear_algebra
    Eigen3::Eigen
)

# === Executável de exemplo simples ===
add_executable(main_simple examples/main_simple.cpp)

target_link_libraries(main_simple
    mesh
    solver
    material
    utils
    linear_algebra
    Eigen3::Eigen
)

# === Novo gerador de dataset ===
add_executable(generate_dataset_NOVO examples/generate_dataset_NOVO.cpp)

target_link_libraries(generate_dataset_NOVO
    mesh
    solver
    material
    utils
    linear_algebra
    Eigen3::Eigen
)

# === Gerador de dataset analítico ===
add_executable(generate_dataset_ANALYTICAL examples/generate_dataset_ANALYTICAL.cpp)

target_link_libraries(generate_dataset_ANALYTICAL
    Eigen3::Eigen
)

add_executable(beam_main examples/beam_main.cpp)
target_link_libraries(beam_main
    mesh
    solver
    material
    utils
    linear_algebra
    Eigen3::Eigen
)