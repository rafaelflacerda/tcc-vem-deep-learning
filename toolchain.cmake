# Force CMake to use the regular Clang compiler, not clang-cl
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
set(CMAKE_SYSTEM_NAME "Darwin") 