#!/usr/bin/env python3
# find_module.py - Find where modules are being loaded from

import sys
import os
import importlib.util

def find_module(module_name):
    """Find the location of a module."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"Module {module_name} not found")
            return None
        
        print(f"Module {module_name} found at: {spec.origin}")
        return spec.origin
    except Exception as e:
        print(f"Error finding module {module_name}: {e}")
        return None

def check_sys_modules():
    """Check what modules are already loaded."""
    print("\nModules already in sys.modules:")
    for name, module in sys.modules.items():
        if name.startswith('polivem'):
            file_path = getattr(module, '__file__', 'No __file__ attribute')
            print(f"  {name}: {file_path}")

def check_sys_path():
    """Check the Python path."""
    print("\nPython path (sys.path):")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
        if os.path.exists(path) and os.path.isdir(path):
            # Check for polivem-related files
            for item in os.listdir(path):
                if 'polivem' in item:
                    full_path = os.path.join(path, item)
                    print(f"    Found: {full_path}")

def check_site_packages():
    """Check site-packages directories."""
    print("\nChecking site-packages directories:")
    for path in sys.path:
        if 'site-packages' in path and os.path.exists(path):
            print(f"  Site-packages directory: {path}")
            # Look for polivem-related files
            for item in os.listdir(path):
                if 'polivem' in item:
                    full_path = os.path.join(path, item)
                    print(f"    Found: {full_path}")
                    if os.path.isdir(full_path):
                        for subitem in os.listdir(full_path):
                            sub_path = os.path.join(full_path, subitem)
                            print(f"      {sub_path}")

def main():
    """Main function."""
    print("=== Python Environment ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check what's already loaded
    check_sys_modules()
    
    # Check Python path
    check_sys_path()
    
    # Check site-packages
    check_site_packages()
    
    # Try to find specific modules
    print("\nFinding specific modules:")
    find_module('polivem')
    find_module('polivem.polivem_py')
    
    # Try to import and check
    print("\nTrying to import modules:")
    try:
        import polivem
        print(f"Imported polivem from: {polivem.__file__}")
        
        try:
            from polivem import polivem_py
            print(f"Imported polivem_py from: {polivem_py.__file__}")
            
            # Check solver module
            if hasattr(polivem_py, 'solver'):
                print("Solver module exists")
                print(f"Solver attributes: {[attr for attr in dir(polivem_py.solver) if not attr.startswith('__')]}")
            else:
                print("Solver module does not exist")
        except ImportError as e:
            print(f"Error importing polivem_py: {e}")
    except ImportError as e:
        print(f"Error importing polivem: {e}")

if __name__ == "__main__":
    main() 