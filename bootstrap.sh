#!/bin/bash
set -e  # Exit on error

echo "=== Cleaning all CMake cache ==="
rm -rf build
rm -f CMakeCache.txt
find . -name "CMakeFiles" -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== Setting compiler paths ==="
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
echo "Using C compiler: $CC"
echo "Using C++ compiler: $CXX"

echo "=== Creating build directory ==="
mkdir -p build
cd build

echo "=== Running CMake with explicit compiler settings ==="
cmake \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  ..

echo "=== Bootstrap completed successfully ==="

# ./run.sh