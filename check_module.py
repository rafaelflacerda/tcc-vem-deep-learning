import os
import sys

# Get the package directory
package_dir = os.path.join(os.getcwd(), "middleware", "polivem")
print(f"Package directory: {package_dir}")

# List all files in the package directory
if os.path.exists(package_dir):
    files = os.listdir(package_dir)
    print(f"Files in package directory: {files}")
    
    # Find the module file
    module_file = None
    for file in files:
        if file.startswith("polivem_py") and (file.endswith(".so") or file.endswith(".pyd") or file.endswith(".dylib")):
            module_file = file
            break
    
    if module_file:
        print(f"Found module file: {module_file}")
        
        # Try to load the module directly
        module_path = os.path.join(package_dir, module_file)
        print(f"Module path: {module_path}")
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("polivem_py", module_path)
            polivem_py = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(polivem_py)
            print(f"Successfully loaded module")
            print(f"Module attributes: {[attr for attr in dir(polivem_py) if not attr.startswith('__')]}")
        except Exception as e:
            print(f"Error loading module: {e}")
    else:
        print(f"Module file not found")
else:
    print(f"Package directory does not exist") 