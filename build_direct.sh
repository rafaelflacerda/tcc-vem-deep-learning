#!/bin/bash
# build_direct.sh - Build directly without pip

set -e  # Exit on error

echo "=== Removing old module ==="
rm -f middleware/polivem/polivem_py.cpython-*.so

echo "=== Running bootstrap ==="
./bootstrap.sh

echo "=== Building module directly ==="
cd build
make
cd ..

echo "=== Copying module to middleware/polivem ==="
mkdir -p middleware/polivem
cp build/python/polivem_py.cpython-*.so middleware/polivem/

echo "=== Testing direct import ==="
python -c "
import sys
sys.path.insert(0, '$(pwd)/middleware')
import polivem
from polivem import polivem_py
print('Available in solver module:', [attr for attr in dir(polivem_py.solver) if not attr.startswith('__')])
" 