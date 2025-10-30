# Package initialization
import sys
import os

# Get the absolute path to the directory containing this file
package_dir = os.path.dirname(os.path.abspath(__file__))

# Import the module directly using its full path
module_path = os.path.join(package_dir, "polivem_py.cpython-313-darwin.so")
if not os.path.exists(module_path):
    # Try to find the module with a different extension
    for file in os.listdir(package_dir):
        if file.startswith("polivem_py") and file.endswith(".so"):
            module_path = os.path.join(package_dir, file)
            break

if not os.path.exists(module_path):
    raise ImportError(f"Cannot find polivem_py module in {package_dir}")

# Add the directory to the Python path
sys.path.insert(0, package_dir)

# Now import the module
import polivem_py

# Make it available as polivem.polivem_py
sys.modules["polivem.polivem_py"] = polivem_py

# Expose submodules at the package level if they exist
if hasattr(polivem_py, "mesh"):
    mesh = polivem_py.mesh
if hasattr(polivem_py, "solver"):
    solver = polivem_py.solver
if hasattr(polivem_py, "enums"):
    enums = polivem_py.enums
if hasattr(polivem_py, "material"):
    material = polivem_py.material
