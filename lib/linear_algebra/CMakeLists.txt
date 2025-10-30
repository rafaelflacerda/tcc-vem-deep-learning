# CMakeLists.txt for lib/linear_algebra
cmake_minimum_required(VERSION 3.9.1)

# Create linear_algebra library
add_library(linear_algebra SHARED
    gauss_elimination.cpp
    sparse_gauss_elimination.cpp
    cholesky_decomposition.cpp
    sparse_cholesky_decomposition.cpp
    power_method.cpp
)

# Find required packages
find_package(Eigen3 REQUIRED)

# Link libraries based on platform
if(APPLE)
    # Use Apple Accelerate framework on macOS
    find_library(ACCELERATE_LIBRARY Accelerate)
    target_link_libraries(linear_algebra PUBLIC ${ACCELERATE_LIBRARY})
    target_compile_definitions(linear_algebra PRIVATE USE_ACCELERATE)
else()
    # Use OpenBLAS/LAPACK on other platforms
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENBLAS REQUIRED openblas)
    pkg_check_modules(LAPACK REQUIRED lapack)
    
    target_link_libraries(linear_algebra PUBLIC ${OPENBLAS_LIBRARIES} ${LAPACK_LIBRARIES})
    target_include_directories(linear_algebra PUBLIC ${OPENBLAS_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
    target_compile_options(linear_algebra PUBLIC ${OPENBLAS_CFLAGS_OTHER} ${LAPACK_CFLAGS_OTHER})
endif()

# Link with Eigen3
target_link_libraries(linear_algebra PUBLIC Eigen3::Eigen)

# Set include directories
target_include_directories(linear_algebra 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(linear_algebra PRIVATE
        -O3                    # Maximum optimization
        -march=native          # Target current architecture
        -ffast-math           # Fast floating-point math
        -DNDEBUG              # Disable assertions in release
    )
endif()

# ARM64-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(linear_algebra PRIVATE
            -mcpu=native       # Target specific ARM64 CPU
        )
    endif()
endif()

# Set properties
set_target_properties(linear_algebra PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Export for use by other targets
target_include_directories(linear_algebra INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)