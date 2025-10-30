#!/bin/bash
set -e  # Exit on error

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv_polivem" ]; then
    echo "=== Creating virtual environment ==="
    python3.13 -m venv venv_polivem
fi

echo "=== Activating virtual environment ==="
source venv_polivem/bin/activate

# Now we're using the Python from the virtual environment
echo "Using Python: $(python --version)"

echo "=== Cleaning Python cache ==="
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -exec rm -f {} + 2>/dev/null || true

echo "=== Running bootstrap.sh ==="
./bootstrap.sh

echo "=== Removing old module ==="
rm -f middleware/polivem/polivem_py.cpython-*.so

echo "=== Building module directly ==="
cd build
make
cd ..

echo "=== Copying module to middleware/polivem ==="
mkdir -p middleware/polivem
cp build/python/polivem_py.cpython-*.so middleware/polivem/

echo "=== Testing module ==="
python -c "
import sys
sys.path.insert(0, '$(pwd)/middleware')
import polivem
from polivem import polivem_py
print('Available in solver module:', [attr for attr in dir(polivem_py.solver) if not attr.startswith('__')])
"

# Run the test modules script
echo "=== Running test modules ==="
python test_modules.py

