import os
import shutil
import sys
from setuptools import setup, find_packages

# Project structure paths
middleware_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(middleware_dir)
build_dir = os.path.join(project_root, 'build')
module_path = os.path.join(build_dir, 'python')

print(f"Looking for module in: {module_path}")

# List all files in module path for debugging
if os.path.exists(module_path):
    all_files = os.listdir(module_path)
    print(f"All files in module path: {all_files}")
    module_files = [f for f in all_files if f.startswith('polivem_py')]
    print(f"Found module files: {module_files}")
else:
    module_files = []
    print("Module path does not exist!")

if not module_files:
    raise RuntimeError("Built module not found. Please build the module first with CMake.")

# Create package directory
package_dir = os.path.join(middleware_dir, 'polivem')
print(f"Creating package directory: {package_dir}")
os.makedirs(package_dir, exist_ok=True)

# Write a more robust __init__.py
init_file = os.path.join(package_dir, '__init__.py')
print(f"Writing to {init_file}")
with open(init_file, 'w') as f:
    f.write('# Package initialization\n')
    f.write('import sys\n')
    f.write('import os\n\n')
    f.write('# Get the absolute path to the directory containing this file\n')
    f.write('package_dir = os.path.dirname(os.path.abspath(__file__))\n\n')
    f.write('# Import the module directly using its full path\n')
    f.write('module_path = os.path.join(package_dir, "polivem_py.cpython-313-darwin.so")\n')
    f.write('if not os.path.exists(module_path):\n')
    f.write('    # Try to find the module with a different extension\n')
    f.write('    for file in os.listdir(package_dir):\n')
    f.write('        if file.startswith("polivem_py") and file.endswith(".so"):\n')
    f.write('            module_path = os.path.join(package_dir, file)\n')
    f.write('            break\n\n')
    f.write('if not os.path.exists(module_path):\n')
    f.write('    raise ImportError(f"Cannot find polivem_py module in {package_dir}")\n\n')
    f.write('# Add the directory to the Python path\n')
    f.write('sys.path.insert(0, package_dir)\n\n')
    f.write('# Now import the module\n')
    f.write('import polivem_py\n\n')
    f.write('# Make it available as polivem.polivem_py\n')
    f.write('sys.modules["polivem.polivem_py"] = polivem_py\n\n')
    f.write('# Expose submodules at the package level if they exist\n')
    f.write('if hasattr(polivem_py, "mesh"):\n')
    f.write('    mesh = polivem_py.mesh\n')
    f.write('if hasattr(polivem_py, "solver"):\n')
    f.write('    solver = polivem_py.solver\n')
    f.write('if hasattr(polivem_py, "enums"):\n')
    f.write('    enums = polivem_py.enums\n')

# Copy the built module with better error handling
print(f"Copying module files...")
for module_file in module_files:
    src_path = os.path.join(module_path, module_file)
    dest_path = os.path.join(package_dir, module_file)
    print(f"  Copying {src_path} to {dest_path}")
    try:
        # Make sure destination is writable if it exists
        if os.path.exists(dest_path):
            os.chmod(dest_path, 0o644)  # Make writable
            print(f"  Removing existing file: {dest_path}")
            os.remove(dest_path)
        shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
        print(f"  Copy successful")
    except Exception as e:
        print(f"  Error copying file: {e}")
        raise RuntimeError(f"Failed to copy module file: {e}")

print("Setting up package...")
setup(
    name='polivem',
    version='0.1.0',
    author='Paulo Akira',
    description='Python bindings for PoliVEM library',
    packages=['polivem'],
    package_data={
        'polivem': ['*.so', '*.dylib', '*.pyd'],
    },
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
    ],
)