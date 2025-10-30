# test_beam.py
import sys
import os

# Add the directory containing the module to Python's path
# module_path = os.path.join(os.getcwd(), 'build/python')
# print(f"Looking for module in: {module_path}")
# print(f"Directory exists: {os.path.exists(module_path)}")
# if os.path.exists(module_path):
#     print(f"Contents: {os.listdir(module_path)}")

# sys.path.append(module_path)

try:
    # Import from the installed package
    from polivem import polivem_py
    print("Successfully imported polivem_py from polivem package")
except ImportError as e:
    print(f"Error importing from package: {e}")
    
    # Fallback to direct import with sys.path
    module_path = os.path.join(os.getcwd(), 'build/python')
    print(f"Trying direct import from: {module_path}")
    if os.path.exists(module_path):
        sys.path.append(module_path)
        try:
            import polivem_py
            print("Successfully imported polivem_py directly")
        except ImportError as e:
            print(f"Error importing directly: {e}")
            sys.exit(1)

try:
    # Create a beam object
    beam = polivem_py.mesh.Beam()
    print("Successfully created a Beam object")
    
    # Test the horizontal_bar_disc method
    beam.horizontal_bar_disc(1.0, 10)
    print("Successfully created a horizontal bar mesh")
    
    # Print some information about the mesh
    print(f"Number of nodes: {beam.nodes.shape[0]}")
    print(f"Number of elements: {beam.elements.shape[0]}")
    
    # Print the first few nodes
    print("First 3 nodes:")
    for i in range(min(3, beam.nodes.shape[0])):
        print(f"Node {i}: ({beam.nodes[i, 0]}, {beam.nodes[i, 1]})")
    
    # Print the first few elements
    print("First 3 elements:")
    for i in range(min(3, beam.elements.shape[0])):
        print(f"Element {i}: {beam.elements[i, :]}")
    
except Exception as e:
    print(f"Error during testing: {e}")