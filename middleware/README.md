# PoliVEM Python Middleware

This directory contains the Python bindings for the PoliVEM library, allowing Python users to access the C++ functionality.

## Overview

The middleware uses [pybind11](https://github.com/pybind/pybind11) to create Python bindings for the C++ classes and functions in the PoliVEM library. The main components are:

- `polivem_module.cpp`: Main module definition
- `beam_wrapper.cpp`: Bindings for beam mesh functionality
- `solver_wrapper.cpp`: Bindings for solver functionality
- `enums_wrapper.cpp`: Bindings for enumerations

## Building the Module

There are two ways to build the Python module:

### 1. Using the Direct Build Approach (Recommended)

This approach builds the module directly and avoids Python caching issues:

```bash
# From the project root
./build_direct.sh
```

This script:

1. Removes any old module files
2. Runs the bootstrap.sh script to set up the build
3. Builds the module using make
4. Copies the built module to the correct location
5. Tests the import

### 2. Using pip Install (May Encounter Caching Issues)

```bash
# From the project root
cd middleware
pip install -e .
```

## Known Issues: Python Module Caching

### The Problem

When developing Python C extensions, you may encounter caching issues where Python continues to use an old version of your module even after rebuilding. This happens because:

1. Python caches compiled C extension modules (`.so` files on Unix/Mac, `.pyd` files on Windows)
2. When using `pip install -e .` (editable install), pip creates a `.pth` file that points to your source directory, but doesn't always properly rebuild or update the C extension modules
3. Python may find the module in multiple locations and prioritize the cached version

### Symptoms

You might see these symptoms:

- Changes to your C++ code don't appear in Python
- Python reports missing methods or classes that you've added
- Python finds old class/method names that you've changed

### Solutions

#### Solution 1: Direct Build (Recommended)

Use the direct build approach as described above. This bypasses pip's installation mechanism entirely.

```bash
./build_direct.sh
```

#### Solution 2: Clean Everything

If you need to use pip install, try cleaning everything first:

```bash
# Remove Python cache
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -exec rm -f {} + 2>/dev/null || true

# Uninstall package
pip uninstall -y polivem

# Remove build artifacts
rm -rf build/
rm -rf middleware/build/
rm -rf middleware/dist/
rm -rf middleware/*.egg-info/

# Remove compiled module
rm -f middleware/polivem/polivem_py.cpython-*.so

# Remove CMake cache
find . -name "CMakeCache.txt" -exec rm -f {} + 2>/dev/null || true
find . -name "cmake_install.cmake" -exec rm -f {} + 2>/dev/null || true
find . -name "Makefile" -exec rm -f {} + 2>/dev/null || true

# Rebuild
./bootstrap.sh
cd middleware
pip install -e . --no-cache-dir
```

#### Solution 3: Check Module Locations

If you're still having issues, check where Python is finding your module:

```bash
python -c "
import sys
import importlib.util
spec = importlib.util.find_spec('polivem')
print(f'polivem found at: {spec.origin}')
spec = importlib.util.find_spec('polivem.polivem_py')
print(f'polivem_py found at: {spec.origin}')
"
```

## Development Tips

1. **Always use the direct build approach during development** to avoid caching issues
2. **Check the console output** for any errors during the build process
3. **Use the `find_module.py` script** to debug module loading issues
4. **Keep your build environment clean** by regularly removing build artifacts

## Troubleshooting

If you encounter issues:

1. **Check the build output** for any compilation errors
2. **Verify include paths** in CMakeLists.txt are correct
3. **Check for multiple installations** of the package with `pip list | grep polivem`
4. **Try a fresh virtual environment** to rule out environment issues

## Project Structure

```
middleware/
├── CMakeLists.txt          # CMake configuration
├── include/
│   └── polivem_python.hpp  # Main header for Python bindings
├── polivem/
│   └── __init__.py         # Python package initialization
├── setup.py                # Python package setup
└── src/
    ├── beam_wrapper.cpp    # Bindings for beam functionality
    ├── enums_wrapper.cpp   # Bindings for enumerations
    ├── polivem_module.cpp  # Main module definition
    └── solver_wrapper.cpp  # Bindings for solver functionality
```
