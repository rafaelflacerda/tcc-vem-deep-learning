# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve
# import matplotlib.tri as mtri

# # Import the VEM axisymmetric functions
# from axisymmetric import (
#     build_constitutive_matrix,
#     compute_element_area,
#     compute_stiffness_matrix,
#     compute_element_load_body_force,
#     compute_element_load_boundary_traction,
#     assemble_element_load_vector
# )

# def run_vem_patch_test():
#     """
#     Run a comprehensive patch test for axisymmetric VEM formulation.
    
#     This tests all four fundamental strain states:
#     1. Constant radial strain
#     2. Constant axial strain
#     3. Constant hoop strain
#     4. Constant shear strain
    
#     Verifies that the VEM implementation properly reproduces these strain states.
#     """
#     print("=== AXISYMMETRIC VEM PATCH TEST ===")
    
#     # Material properties (same as in the FEM implementation)
#     E = 1.0       # Young's modulus
#     nu = 0.3      # Poisson's ratio
    
#     # Create the constitutive matrix
#     C = build_constitutive_matrix(E, nu)
    
#     print("\nConstitutive matrix C:")
#     print(C)
    
#     # Domain setup
#     r_inner = 1.0
#     r_outer = 3.0
#     z_min = 0.0
#     z_max = 2.0
    
#     # Create a mesh of triangular elements
#     n_r = 4  # elements in r direction
#     n_z = 4  # elements in z direction
    
#     # Generate mesh nodes
#     nr_nodes = n_r + 1
#     nz_nodes = n_z + 1
#     grid_nodes = np.zeros((nr_nodes * nz_nodes, 2))  # (r, z) coordinates
    
#     node_idx = 0
#     for iz in range(nz_nodes):
#         for ir in range(nr_nodes):
#             r = r_inner + (r_outer - r_inner) * ir / (nr_nodes - 1)
#             z = z_min + (z_max - z_min) * iz / (nz_nodes - 1)
#             grid_nodes[node_idx] = [r, z]
#             node_idx += 1
    
#     # Create triangular elements (two triangles per grid cell)
#     n_cell_triangles = 2  # Number of triangles per cell
#     total_triangles = n_r * n_z * n_cell_triangles
#     triangles = np.zeros((total_triangles, 3), dtype=int)
    
#     triangle_idx = 0
#     for iz in range(n_z):
#         for ir in range(n_r):
#             # Indices of the four corners of the cell
#             n1 = iz * nr_nodes + ir
#             n2 = iz * nr_nodes + (ir + 1)
#             n3 = (iz + 1) * nr_nodes + (ir + 1)
#             n4 = (iz + 1) * nr_nodes + ir
            
#             # Create two triangles
#             triangles[triangle_idx] = [n1, n2, n4]  # Lower triangle
#             triangle_idx += 1
#             triangles[triangle_idx] = [n2, n3, n4]  # Upper triangle
#             triangle_idx += 1
    
#     # Copy grid nodes to be our final nodes
#     nodes = grid_nodes.copy()
#     elements = triangles.copy()
    
#     total_nodes = len(nodes)
#     total_elements = len(elements)
    
#     print(f"\nTotal nodes: {total_nodes}")
#     print(f"Total elements: {total_elements}")
#     print("First triangle nodes:", elements[0])
    
#     # Total degrees of freedom
#     ndof = 2 * total_nodes  # 2 DOFs per node (r, z displacements)
#     print(f"Total DOFs: {ndof}")
    
#     # Assemble global stiffness matrix
#     K_global = lil_matrix((ndof, ndof))
    
#     for e in range(total_elements):
#         # Get vertex indices for this element
#         node_indices = elements[e]
        
#         # Get node coordinates for this element
#         element_vertices = nodes[node_indices]
        
#         # Compute element stiffness matrix
#         K_elem, _, _ = compute_stiffness_matrix(element_vertices, E, nu)
        
#         # Map local DOFs to global DOFs
#         dof_indices = np.zeros(2 * len(node_indices), dtype=int)
#         for i in range(len(node_indices)):
#             dof_indices[2*i] = 2 * node_indices[i]     # r-displacement
#             dof_indices[2*i+1] = 2 * node_indices[i] + 1  # z-displacement
        
#         # Assemble element matrix into global matrix
#         for i in range(len(dof_indices)):
#             for j in range(len(dof_indices)):
#                 K_global[dof_indices[i], dof_indices[j]] += K_elem[i, j]
    
#     # Find boundary nodes
#     boundary_nodes = []
#     for i in range(total_nodes):
#         r, z = nodes[i]
#         if (abs(r - r_inner) < 1e-6 or abs(r - r_outer) < 1e-6 or 
#             abs(z - z_min) < 1e-6 or abs(z - z_max) < 1e-6):
#             boundary_nodes.append(i)
    
#     print(f"\nNumber of boundary nodes: {len(boundary_nodes)}")
    
#     # Convert to CSR format for efficient solving
#     K_global_csr = K_global.tocsr()
    
#     # Run the four patch tests
#     all_tests_passed = True
    
#     for strain_case in range(1, 5):
#         passed = run_patch_test_case(strain_case, nodes, elements, K_global_csr, boundary_nodes, ndof, total_nodes)
#         all_tests_passed = all_tests_passed and passed
    
#     # Summary
#     print("\n=== PATCH TEST SUMMARY ===")
#     print(f"Overall result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
    
#     return all_tests_passed

# def run_patch_test_case(strain_case, nodes, elements, K_global, boundary_nodes, ndof, total_nodes):
#     """
#     Run a specific patch test case.
    
#     Parameters:
#         strain_case (int): Which strain case to test:
#                           1 = Constant radial strain
#                           2 = Constant axial strain
#                           3 = Constant hoop strain
#                           4 = Constant shear strain
#         nodes (np.ndarray): Array of node coordinates
#         elements (np.ndarray): Array of element connectivity
#         K_global (scipy.sparse.csr_matrix): Global stiffness matrix
#         boundary_nodes (list): List of boundary node indices
#         ndof (int): Total number of degrees of freedom
#         total_nodes (int): Total number of nodes
        
#     Returns:
#         bool: True if the test passed, False otherwise
#     """
#     # Initialize force vector
#     F = np.zeros(ndof)
    
#     # Set strain magnitude
#     strain_magnitude = 0.01
    
#     # Define strain state based on test case
#     strain_r = 0.0
#     strain_z = 0.0
#     strain_theta = 0.0
#     strain_rz = 0.0
    
#     if strain_case == 1:
#         strain_r = strain_magnitude      # Radial strain test
#         test_name = "Radial Strain"
#     elif strain_case == 2:
#         strain_z = strain_magnitude      # Axial strain test
#         test_name = "Axial Strain"
#     elif strain_case == 3:
#         strain_theta = strain_magnitude  # Hoop strain test
#         test_name = "Hoop Strain"
#     elif strain_case == 4:
#         strain_rz = strain_magnitude     # Shear strain test
#         test_name = "Shear Strain"
    
#     print(f"\n\n--- Testing Case {strain_case}: Constant {test_name} ---")
    
#     # Exact displacement field based on strain state
#     u_exact = np.zeros(ndof)
#     for i in range(total_nodes):
#         r, z = nodes[i]
        
#         if strain_case == 1:  # Radial strain
#             u_exact[2*i] = strain_r * r       # Radial displacement (ur)
#             u_exact[2*i+1] = 0                # Axial displacement (uz)
        
#         elif strain_case == 2:  # Axial strain
#             u_exact[2*i] = 0                  # Radial displacement (ur)
#             u_exact[2*i+1] = strain_z * z     # Axial displacement (uz)
        
#         elif strain_case == 3:  # Hoop strain (same as radial for displacement)
#             u_exact[2*i] = strain_theta * r   # Radial displacement (ur)
#             u_exact[2*i+1] = 0                # Axial displacement (uz)
        
#         elif strain_case == 4:  # Shear strain
#             u_exact[2*i] = strain_rz * z/2    # Radial displacement (ur)
#             u_exact[2*i+1] = strain_rz * r/2  # Axial displacement (uz)
    
#     # Set up boundary conditions
#     free_dofs = np.ones(ndof, dtype=bool)
#     boundary_dofs = []
#     for i in boundary_nodes:
#         boundary_dofs.extend([2*i, 2*i+1])
        
#     free_dofs[boundary_dofs] = False
    
#     # Solve the system
#     F_mod = F - K_global.dot(u_exact)
#     F_reduced = F_mod[free_dofs]
#     K_reduced = K_global[free_dofs, :][:, free_dofs]
    
#     u_reduced = spsolve(K_reduced, F_reduced)
#     u = u_exact.copy()
#     u[free_dofs] = u_reduced
    
#     # Extract displacements
#     u_r = u[0::2]
#     u_z = u[1::2]
    
#     # Calculate strains at the centroid of each element
#     total_elements = len(elements)
#     strains = np.zeros((total_elements, 4))  # [εr, εz, εθ, γrz]
    
#     for e in range(total_elements):
#         node_indices = elements[e]
#         element_vertices = nodes[node_indices]
        
#         # Get element displacements
#         n_vertices = len(node_indices)
#         elem_disps = np.zeros(2 * n_vertices)
#         for i in range(n_vertices):
#             elem_disps[2*i] = u[2*node_indices[i]]       # ur
#             elem_disps[2*i+1] = u[2*node_indices[i]+1]   # uz
        
#         # Calculate strain at element centroid
#         strains[e] = calculate_strain_at_centroid(element_vertices, elem_disps)
    
#     # Compute average strains and errors
#     avg_strains = np.mean(strains, axis=0)
    
#     print(f"\nAverage strains across all elements:")
#     print(f"εr (Radial strain): {avg_strains[0]:.6f} (Expected: {strain_r:.6f})")
#     print(f"εz (Axial strain): {avg_strains[1]:.6f} (Expected: {strain_z:.6f})")
#     print(f"εθ (Hoop strain): {avg_strains[2]:.6f} (Expected: {strain_theta:.6f})")
#     print(f"γrz (Shear strain): {avg_strains[3]:.6f} (Expected: {strain_rz:.6f})")
    
#     # Calculate strain errors
#     strain_errors = np.abs([
#         avg_strains[0] - strain_r,
#         avg_strains[1] - strain_z, 
#         avg_strains[2] - strain_theta,
#         avg_strains[3] - strain_rz
#     ])
    
#     print("\nStrain errors:")
#     print(f"Error in εr: {strain_errors[0]:.6e}")
#     print(f"Error in εz: {strain_errors[1]:.6e}")
#     print(f"Error in εθ: {strain_errors[2]:.6e}")
#     print(f"Error in γrz: {strain_errors[3]:.6e}")
    
#     # Check if patch test passed
#     tolerance = 1e-3
#     passed = np.all(strain_errors < tolerance)
    
#     # print(f"\nPatch test for {test_name}: {'PASSED' if passed else 'FAILED'} with tolerance {tolerance}")
    
#     # Create plots
#     # plot_patch_test_results(strain_case, nodes, elements, u_r, u_z, strains, strain_r, strain_z, strain_theta, strain_rz)
    
#     return passed

# def calculate_strain_at_centroid(element_vertices, displacements):
#     """
#     Calculate the strain at the centroid of an element.
    
#     For VEM, we use the projection of the displacement field to compute the strain.
#     This is a simplified version that approximates the strain using shape function derivatives
#     evaluated at the element centroid.
    
#     Parameters:
#         element_vertices (np.ndarray): Vertex coordinates of the element
#         displacements (np.ndarray): Element displacement vector
        
#     Returns:
#         np.ndarray: Strain vector [εr, εz, εθ, γrz]
#     """
#     # Calculate centroid
#     centroid_r = np.mean(element_vertices[:, 0])
#     centroid_z = np.mean(element_vertices[:, 1])
    
#     n_vertices = len(element_vertices)
    
#     # Initialize strain
#     strain = np.zeros(4)
    
#     # For triangular elements, we can approximate the constant strain as:
#     if n_vertices == 3:
#         # Get vertex coordinates
#         r1, z1 = element_vertices[0]
#         r2, z2 = element_vertices[1]
#         r3, z3 = element_vertices[2]
        
#         # Compute area
#         area = compute_element_area(element_vertices)
        
#         # Shape function derivatives for triangular elements
#         # These are constant for linear triangles
#         dr12 = r1 - r2
#         dr13 = r1 - r3
#         dr23 = r2 - r3
#         dz12 = z1 - z2
#         dz13 = z1 - z3
#         dz23 = z2 - z3
        
#         # These are coefficients for the shape function derivatives
#         # derived from the area coordinates formula
#         a1 = (z2 - z3)
#         b1 = -(r2 - r3)
#         a2 = (z3 - z1)
#         b2 = -(r3 - r1)
#         a3 = (z1 - z2)
#         b3 = -(r1 - r2)
        
#         # Normalize by 2*area
#         denom = 2.0 * area
#         a1 /= denom
#         b1 /= denom
#         a2 /= denom
#         b2 /= denom
#         a3 /= denom
#         b3 /= denom
        
#         # Extract displacements
#         u_r1 = displacements[0]
#         u_z1 = displacements[1]
#         u_r2 = displacements[2]
#         u_z2 = displacements[3]
#         u_r3 = displacements[4]
#         u_z3 = displacements[5]
        
#         # Compute strains
#         # εr = ∂u_r/∂r
#         strain[0] = a1 * u_r1 + a2 * u_r2 + a3 * u_r3
        
#         # εz = ∂u_z/∂z
#         strain[1] = b1 * u_z1 + b2 * u_z2 + b3 * u_z3
        
#         # εθ = u_r/r (evaluated at centroid)
#         # Linear interpolation of u_r at centroid
#         u_r_centroid = 0
#         for i in range(n_vertices):
#             # Use simple barycentric coordinates for triangle
#             u_r_centroid += displacements[2*i] / n_vertices
        
#         strain[2] = u_r_centroid / centroid_r
        
#         # γrz = ∂u_r/∂z + ∂u_z/∂r
#         strain[3] = (b1 * u_r1 + b2 * u_r2 + b3 * u_r3) + (a1 * u_z1 + a2 * u_z2 + a3 * u_z3)
    
#     else:
#         # For other polygons, use a simple approximation
#         # This is less accurate but should work for patch tests
#         # with constant strains
        
#         # Compute average displacement derivatives using a finite difference approximation
#         du_r_dr = 0
#         du_r_dz = 0
#         du_z_dr = 0
#         du_z_dz = 0
        
#         # Count how many edges contribute to each derivative
#         count_r = 0
#         count_z = 0
        
#         # Loop over all edges
#         for i in range(n_vertices):
#             j = (i + 1) % n_vertices
            
#             # Get the coordinates and displacements at the endpoints
#             ri, zi = element_vertices[i]
#             rj, zj = element_vertices[j]
            
#             u_ri = displacements[2*i]
#             u_zi = displacements[2*i+1]
#             u_rj = displacements[2*j]
#             u_zj = displacements[2*j+1]
            
#             # Skip edges with zero length
#             edge_length = np.sqrt((rj-ri)**2 + (zj-zi)**2)
#             if edge_length < 1e-10:
#                 continue
            
#             # For vertical edges (constant r), compute z derivatives
#             if abs(rj - ri) < 1e-10:
#                 du_r_dz += (u_rj - u_ri) / (zj - zi)
#                 du_z_dz += (u_zj - u_zi) / (zj - zi)
#                 count_z += 1
            
#             # For horizontal edges (constant z), compute r derivatives
#             elif abs(zj - zi) < 1e-10:
#                 du_r_dr += (u_rj - u_ri) / (rj - ri)
#                 du_z_dr += (u_zj - u_zi) / (rj - ri)
#                 count_r += 1
            
#             # For general edges, decompose based on direction
#             else:
#                 # Compute unit edge vector
#                 dr = (rj - ri) / edge_length
#                 dz = (zj - zi) / edge_length
                
#                 # Projection of derivatives along the edge
#                 du_r_ds = (u_rj - u_ri) / edge_length
#                 du_z_ds = (u_zj - u_zi) / edge_length
                
#                 # Decompose based on direction cosines
#                 du_r_dr += du_r_ds * dr * abs(dr)
#                 du_r_dz += du_r_ds * dz * abs(dz)
#                 du_z_dr += du_z_ds * dr * abs(dr)
#                 du_z_dz += du_z_ds * dz * abs(dz)
                
#                 count_r += abs(dr)
#                 count_z += abs(dz)
        
#         # Average the derivatives
#         du_r_dr = du_r_dr / max(1, count_r)
#         du_r_dz = du_r_dz / max(1, count_z)
#         du_z_dr = du_z_dr / max(1, count_r)
#         du_z_dz = du_z_dz / max(1, count_z)
        
#         # Compute strains
#         strain[0] = du_r_dr                  # εr = ∂u_r/∂r
#         strain[1] = du_z_dz                  # εz = ∂u_z/∂z
        
#         # For εθ, compute average u_r at centroid
#         u_r_centroid = 0
#         for i in range(n_vertices):
#             u_r_centroid += displacements[2*i]
#         u_r_centroid /= n_vertices
        
#         strain[2] = u_r_centroid / centroid_r  # εθ = u_r/r
#         strain[3] = du_r_dz + du_z_dr          # γrz = ∂u_r/∂z + ∂u_z/∂r
    
#     return strain

# def plot_patch_test_results(strain_case, nodes, elements, u_r, u_z, strains, strain_r, strain_z, strain_theta, strain_rz):
#     """
#     Create plots showing the results of the patch test.
    
#     Parameters:
#         strain_case (int): Which strain case was tested
#         nodes (np.ndarray): Array of node coordinates
#         elements (np.ndarray): Array of element connectivity
#         u_r (np.ndarray): Radial displacements
#         u_z (np.ndarray): Axial displacements
#         strains (np.ndarray): Computed strains for each element
#         strain_r, strain_z, strain_theta, strain_rz (float): Expected strain values
#     """
#     # Create a triangulation for plotting
#     tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
#     # Prepare to get centroids for strain plotting
#     total_elements = len(elements)
#     centroids = np.zeros((total_elements, 2))
#     for e in range(total_elements):
#         centroids[e] = np.mean(nodes[elements[e]], axis=0)
    
#     # Set up the figure
#     plt.figure(figsize=(15, 12))
    
#     # Plot 1: Mesh
#     plt.subplot(2, 2, 1)
#     plt.triplot(tri, 'k-', lw=0.5)
#     plt.scatter(nodes[:, 0], nodes[:, 1], c='b', s=20)
#     plt.title('Triangular Mesh')
#     plt.xlabel('r')
#     plt.ylabel('z')
#     plt.axis('equal')
    
#     # Plot 2: Displacement
#     plt.subplot(2, 2, 2)
    
#     if strain_case in [1, 3]:  # Radial or hoop strain
#         plot_data = u_r
#         title = 'Radial displacement (u_r)'
#     elif strain_case == 2:  # Axial strain
#         plot_data = u_z
#         title = 'Axial displacement (u_z)'
#     else:  # Shear strain
#         # For shear, plot magnitude
#         plot_data = np.sqrt(u_r**2 + u_z**2)
#         title = 'Displacement magnitude'
    
#     plt.tripcolor(tri, plot_data, cmap='viridis')
#     plt.colorbar(label=title)
#     plt.title(title)
#     plt.xlabel('r')
#     plt.ylabel('z')
#     plt.axis('equal')
    
#     # Plot 3: The specific strain component being tested
#     plt.subplot(2, 2, 3)
    
#     if strain_case == 1:
#         strain_idx = 0  # Radial strain
#         strain_name = 'Radial strain (εr)'
#         expected = strain_r
#     elif strain_case == 2:
#         strain_idx = 1  # Axial strain
#         strain_name = 'Axial strain (εz)'
#         expected = strain_z
#     elif strain_case == 3:
#         strain_idx = 2  # Hoop strain
#         strain_name = 'Hoop strain (εθ)'
#         expected = strain_theta
#     else:
#         strain_idx = 3  # Shear strain
#         strain_name = 'Shear strain (γrz)'
#         expected = strain_rz
    
#     sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=strains[:, strain_idx], cmap='viridis', s=50)
#     plt.colorbar(sc, label=strain_name)
#     plt.title(f'{strain_name} (Target: {expected})')
#     plt.xlabel('r')
#     plt.ylabel('z')
#     plt.axis('equal')
    
#     # Plot 4: Strain error
#     plt.subplot(2, 2, 4)
#     strain_comp_errors = np.abs(strains[:, strain_idx] - expected)
#     sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=strain_comp_errors, cmap='jet', s=50)
#     plt.colorbar(sc, label=f'{strain_name} error')
#     plt.title(f'{strain_name} error')
#     plt.xlabel('r')
#     plt.ylabel('z')
#     plt.axis('equal')
    
#     plt.tight_layout()
#     plt.savefig(f'vem_patch_test_case_{strain_case}.png')
#     plt.close()

# if __name__ == "__main__":
#     run_vem_patch_test()

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

# Import the VEM axisymmetric functions
from axisymmetric import (
    build_constitutive_matrix,
    compute_element_area,
    compute_stiffness_matrix,
    compute_element_load_body_force,
    compute_element_load_boundary_traction,
    assemble_element_load_vector
)

STAB_TYPE = "boundary"

def run_vem_quad_patch_test():
    """
    Run a comprehensive patch test for axisymmetric VEM formulation with quadrilateral elements.
    
    This tests all four fundamental strain states:
    1. Constant radial strain
    2. Constant axial strain
    3. Constant hoop strain
    4. Constant shear strain
    
    Verifies that the VEM implementation properly reproduces these strain states.
    """
    print("=== AXISYMMETRIC VEM PATCH TEST WITH QUADRILATERAL ELEMENTS ===")
    
    # Material properties
    E = 1.0       # Young's modulus
    nu = 0.3      # Poisson's ratio
    
    # Create the constitutive matrix
    C = build_constitutive_matrix(E, nu)
    
    print("\nConstitutive matrix C:")
    print(C)
    
    # Domain setup
    r_inner = 1.0
    r_outer = 3.0
    z_min = 0.0
    z_max = 2.0
    
    # Create a mesh of quadrilateral elements
    n_r = 4  # elements in r direction
    n_z = 4  # elements in z direction
    
    # Generate mesh nodes
    nr_nodes = n_r + 1
    nz_nodes = n_z + 1
    grid_nodes = np.zeros((nr_nodes * nz_nodes, 2))  # (r, z) coordinates
    
    node_idx = 0
    for iz in range(nz_nodes):
        for ir in range(nr_nodes):
            r = r_inner + (r_outer - r_inner) * ir / (nr_nodes - 1)
            z = z_min + (z_max - z_min) * iz / (nz_nodes - 1)
            grid_nodes[node_idx] = [r, z]
            node_idx += 1
    
    # Create quadrilateral elements
    total_quads = n_r * n_z
    quads = np.zeros((total_quads, 4), dtype=int)
    
    quad_idx = 0
    for iz in range(n_z):
        for ir in range(n_r):
            # Indices of the four corners of the quad
            n1 = iz * nr_nodes + ir
            n2 = iz * nr_nodes + (ir + 1)
            n3 = (iz + 1) * nr_nodes + (ir + 1)
            n4 = (iz + 1) * nr_nodes + ir
            
            # Create quad (counter-clockwise ordering)
            quads[quad_idx] = [n1, n2, n3, n4]
            quad_idx += 1
    
    # Copy grid nodes to be our final nodes
    nodes = grid_nodes.copy()
    elements = quads.copy()
    
    total_nodes = len(nodes)
    total_elements = len(elements)
    
    print(f"\nTotal nodes: {total_nodes}")
    print(f"Total elements: {total_elements}")
    print("First quadrilateral nodes:", elements[0])
    
    # Total degrees of freedom
    ndof = 2 * total_nodes  # 2 DOFs per node (r, z displacements)
    print(f"Total DOFs: {ndof}")
    
    # Assemble global stiffness matrix
    K_global = lil_matrix((ndof, ndof))
    
    for e in range(total_elements):
        # Get vertex indices for this element
        node_indices = elements[e]
        
        # Get node coordinates for this element
        element_vertices = nodes[node_indices]
        
        # Compute element stiffness matrix
        K_elem, _, _ = compute_stiffness_matrix(element_vertices, E, nu, STAB_TYPE)
        
        # Map local DOFs to global DOFs
        dof_indices = np.zeros(2 * len(node_indices), dtype=int)
        for i in range(len(node_indices)):
            dof_indices[2*i] = 2 * node_indices[i]     # r-displacement
            dof_indices[2*i+1] = 2 * node_indices[i] + 1  # z-displacement
        
        # Assemble element matrix into global matrix
        for i in range(len(dof_indices)):
            for j in range(len(dof_indices)):
                K_global[dof_indices[i], dof_indices[j]] += K_elem[i, j]
    
    # Find boundary nodes
    boundary_nodes = []
    for i in range(total_nodes):
        r, z = nodes[i]
        if (abs(r - r_inner) < 1e-6 or abs(r - r_outer) < 1e-6 or 
            abs(z - z_min) < 1e-6 or abs(z - z_max) < 1e-6):
            boundary_nodes.append(i)
    
    print(f"\nNumber of boundary nodes: {len(boundary_nodes)}")
    
    # Convert to CSR format for efficient solving
    K_global_csr = K_global.tocsr()
    
    # Run the four patch tests
    all_tests_passed = True
    
    for strain_case in range(1, 5):
        passed = run_patch_test_case(strain_case, nodes, elements, K_global_csr, boundary_nodes, ndof, total_nodes)
        all_tests_passed = all_tests_passed and passed
    
    # Summary
    print("\n=== PATCH TEST SUMMARY ===")
    print(f"Overall result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
    
    return all_tests_passed

def run_patch_test_case(strain_case, nodes, elements, K_global, boundary_nodes, ndof, total_nodes):
    """
    Run a specific patch test case.

    Parameters:
        strain_case (int): Which strain case to test:
                          1 = Constant radial strain
                          2 = Constant axial strain
                          3 = Constant hoop strain
                          4 = Constant shear strain
        nodes (np.ndarray): Array of node coordinates
        elements (np.ndarray): Array of element connectivity
        K_global (scipy.sparse.csr_matrix): Global stiffness matrix
        boundary_nodes (list): List of boundary node indices
        ndof (int): Total number of degrees of freedom
        total_nodes (int): Total number of nodes

    Returns:
        bool: True if the test passed, False otherwise
    """
    # Initialize force vector
    F = np.zeros(ndof)
    
    # Set strain magnitude
    strain_magnitude = 0.01
    
    # Define strain state based on test case
    strain_r = 0.0
    strain_z = 0.0
    strain_theta = 0.0
    strain_rz = 0.0
    
    if strain_case == 1:
        strain_r = strain_magnitude      # Radial strain test
        test_name = "Radial Strain"
    elif strain_case == 2:
        strain_z = strain_magnitude      # Axial strain test
        test_name = "Axial Strain"
    elif strain_case == 3:
        strain_theta = strain_magnitude  # Hoop strain test
        test_name = "Hoop Strain"
    elif strain_case == 4:
        strain_rz = strain_magnitude     # Shear strain test
        test_name = "Shear Strain"
    
    print(f"\n\n--- Testing Case {strain_case}: Constant {test_name} ---")
    
    # Exact displacement field based on strain state
    u_exact = np.zeros(ndof)
    for i in range(total_nodes):
        r, z = nodes[i]
        
        if strain_case == 1:  # Radial strain
            u_exact[2*i] = strain_r * r       # Radial displacement (ur)
            u_exact[2*i+1] = 0                # Axial displacement (uz)
        
        elif strain_case == 2:  # Axial strain
            u_exact[2*i] = 0                  # Radial displacement (ur)
            u_exact[2*i+1] = strain_z * z     # Axial displacement (uz)
        
        elif strain_case == 3:  # Hoop strain (same as radial for displacement)
            u_exact[2*i] = strain_theta * r   # Radial displacement (ur)
            u_exact[2*i+1] = 0                # Axial displacement (uz)
        
        elif strain_case == 4:  # Shear strain
            u_exact[2*i] = strain_rz * z/2    # Radial displacement (ur)
            u_exact[2*i+1] = strain_rz * r/2  # Axial displacement (uz)
    
    # Set up boundary conditions
    free_dofs = np.ones(ndof, dtype=bool)
    boundary_dofs = []
    for i in boundary_nodes:
        boundary_dofs.extend([2*i, 2*i+1])
        
    free_dofs[boundary_dofs] = False
    
    # Solve the system
    F_mod = F - K_global.dot(u_exact)
    F_reduced = F_mod[free_dofs]
    K_reduced = K_global[free_dofs, :][:, free_dofs]
    
    u_reduced = spsolve(K_reduced, F_reduced)
    u = u_exact.copy()
    u[free_dofs] = u_reduced
    
    # Extract displacements
    u_r = u[0::2]
    u_z = u[1::2]
    
    # Calculate strains at the centroid of each element
    total_elements = len(elements)
    strains = np.zeros((total_elements, 4))  # [εr, εz, εθ, γrz]
    
    for e in range(total_elements):
        node_indices = elements[e]
        element_vertices = nodes[node_indices]
        
        # Get element displacements
        n_vertices = len(node_indices)
        elem_disps = np.zeros(2 * n_vertices)
        for i in range(n_vertices):
            elem_disps[2*i] = u[2*node_indices[i]]       # ur
            elem_disps[2*i+1] = u[2*node_indices[i]+1]   # uz
        
        # Calculate strain at element centroid
        strains[e] = calculate_strain_at_centroid(element_vertices, elem_disps)
    
    # Compute average strains and errors
    avg_strains = np.mean(strains, axis=0)
    
    print(f"\nAverage strains across all elements:")
    print(f"εr (Radial strain): {avg_strains[0]:.6f} (Expected: {strain_r:.6f})")
    print(f"εz (Axial strain): {avg_strains[1]:.6f} (Expected: {strain_z:.6f})")
    print(f"εθ (Hoop strain): {avg_strains[2]:.6f} (Expected: {strain_theta:.6f})")
    print(f"γrz (Shear strain): {avg_strains[3]:.6f} (Expected: {strain_rz:.6f})")
    
    # Calculate strain errors
    strain_errors = np.abs([
        avg_strains[0] - strain_r,
        avg_strains[1] - strain_z, 
        avg_strains[2] - strain_theta,
        avg_strains[3] - strain_rz
    ])
    
    print("\nStrain errors:")
    print(f"Error in εr: {strain_errors[0]:.6e}")
    print(f"Error in εz: {strain_errors[1]:.6e}")
    print(f"Error in εθ: {strain_errors[2]:.6e}")
    print(f"Error in γrz: {strain_errors[3]:.6e}")
    
    # Check if patch test passed
    tolerance = 1e-3
    passed = np.all(strain_errors < tolerance)
    
    print(f"\nPatch test for {test_name}: {'PASSED' if passed else 'FAILED'} with tolerance {tolerance}")
    
    # Create plots
    plot_patch_test_results(strain_case, nodes, elements, u_r, u_z, strains, strain_r, strain_z, strain_theta, strain_rz)
    
    return passed

def calculate_strain_at_centroid(element_vertices, displacements):
    """
    Calculate the strain at the centroid of an element.
    
    For VEM, we use the projection of the displacement field to compute the strain.
    This is a simplified version that approximates the strain using shape function derivatives
    evaluated at the element centroid.

    Parameters:
        element_vertices (np.ndarray): Vertex coordinates of the element
        displacements (np.ndarray): Element displacement vector

    Returns:
        np.ndarray: Strain vector [εr, εz, εθ, γrz]
    """
    # Calculate centroid
    centroid_r = np.mean(element_vertices[:, 0])
    centroid_z = np.mean(element_vertices[:, 1])
    
    n_vertices = len(element_vertices)
    
    # Initialize strain
    strain = np.zeros(4)
    
    # For quadrilateral elements, use the general polygon approach
    # Compute average displacement derivatives using a finite difference approximation
    du_r_dr = 0
    du_r_dz = 0
    du_z_dr = 0
    du_z_dz = 0
    
    # Count how many edges contribute to each derivative
    count_r = 0
    count_z = 0
    
    # Loop over all edges
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        
        # Get the coordinates and displacements at the endpoints
        ri, zi = element_vertices[i]
        rj, zj = element_vertices[j]
        
        u_ri = displacements[2*i]
        u_zi = displacements[2*i+1]
        u_rj = displacements[2*j]
        u_zj = displacements[2*j+1]
        
        # Skip edges with zero length
        edge_length = np.sqrt((rj-ri)**2 + (zj-zi)**2)
        if edge_length < 1e-10:
            continue
        
        # For vertical edges (constant r), compute z derivatives
        if abs(rj - ri) < 1e-10:
            du_r_dz += (u_rj - u_ri) / (zj - zi)
            du_z_dz += (u_zj - u_zi) / (zj - zi)
            count_z += 1
        
        # For horizontal edges (constant z), compute r derivatives
        elif abs(zj - zi) < 1e-10:
            du_r_dr += (u_rj - u_ri) / (rj - ri)
            du_z_dr += (u_zj - u_zi) / (rj - ri)
            count_r += 1
        
        # For general edges, decompose based on direction
        else:
            # Compute unit edge vector
            dr = (rj - ri) / edge_length
            dz = (zj - zi) / edge_length
            
            # Projection of derivatives along the edge
            du_r_ds = (u_rj - u_ri) / edge_length
            du_z_ds = (u_zj - u_zi) / edge_length
            
            # Decompose based on direction cosines
            du_r_dr += du_r_ds * dr * abs(dr)
            du_r_dz += du_r_ds * dz * abs(dz)
            du_z_dr += du_z_ds * dr * abs(dr)
            du_z_dz += du_z_ds * dz * abs(dz)
            
            count_r += abs(dr)
            count_z += abs(dz)
    
    # Average the derivatives
    du_r_dr = du_r_dr / max(1, count_r)
    du_r_dz = du_r_dz / max(1, count_z)
    du_z_dr = du_z_dr / max(1, count_r)
    du_z_dz = du_z_dz / max(1, count_z)
    
    # Compute strains
    strain[0] = du_r_dr                  # εr = ∂u_r/∂r
    strain[1] = du_z_dz                  # εz = ∂u_z/∂z
    
    # For εθ, compute average u_r at centroid
    u_r_centroid = 0
    for i in range(n_vertices):
        u_r_centroid += displacements[2*i]
    u_r_centroid /= n_vertices
    
    strain[2] = u_r_centroid / centroid_r  # εθ = u_r/r
    strain[3] = du_r_dz + du_z_dr          # γrz = ∂u_r/∂z + ∂u_z/∂r
    
    return strain

def plot_patch_test_results(strain_case, nodes, elements, u_r, u_z, strains, strain_r, strain_z, strain_theta, strain_rz):
    """
    Create plots showing the results of the patch test.
    
    Parameters:
        strain_case (int): Which strain case was tested
        nodes (np.ndarray): Array of node coordinates
        elements (np.ndarray): Array of element connectivity
        u_r (np.ndarray): Radial displacements
        u_z (np.ndarray): Axial displacements
        strains (np.ndarray): Computed strains for each element
        strain_r, strain_z, strain_theta, strain_rz (float): Expected strain values
    """
    # Set up the figure
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Mesh
    plt.subplot(2, 2, 1)
    # For quadrilaterals, we need to plot each element separately
    for e in range(len(elements)):
        quad = elements[e]
        # Add the first vertex again to close the loop
        vertices = np.vstack([nodes[quad], nodes[quad[0]]])
        plt.plot(vertices[:, 0], vertices[:, 1], 'k-', lw=0.5)
    
    plt.scatter(nodes[:, 0], nodes[:, 1], c='b', s=20)
    plt.title('Quadrilateral Mesh')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.axis('equal')
    
    # For displacement and strain plots, create a triangulation by splitting each quad into 2 triangles
    triangles = []
    for quad in elements:
        triangles.append([quad[0], quad[1], quad[3]])
        triangles.append([quad[1], quad[2], quad[3]])
    
    triangles = np.array(triangles)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], triangles)
    
    # Plot 2: Displacement
    plt.subplot(2, 2, 2)
    
    if strain_case in [1, 3]:  # Radial or hoop strain
        plot_data = u_r
        title = 'Radial displacement (u_r)'
    elif strain_case == 2:  # Axial strain
        plot_data = u_z
        title = 'Axial displacement (u_z)'
    else:  # Shear strain
        # For shear, plot magnitude
        plot_data = np.sqrt(u_r**2 + u_z**2)
        title = 'Displacement magnitude'
    
    plt.tripcolor(tri, plot_data, cmap='viridis')
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel('r')
    plt.ylabel('z')
    plt.axis('equal')
    
    # Prepare to get centroids for strain plotting
    total_elements = len(elements)
    centroids = np.zeros((total_elements, 2))
    for e in range(total_elements):
        centroids[e] = np.mean(nodes[elements[e]], axis=0)
    
    # Plot 3: The specific strain component being tested
    plt.subplot(2, 2, 3)
    
    if strain_case == 1:
        strain_idx = 0  # Radial strain
        strain_name = 'Radial strain (εr)'
        expected = strain_r
    elif strain_case == 2:
        strain_idx = 1  # Axial strain
        strain_name = 'Axial strain (εz)'
        expected = strain_z
    elif strain_case == 3:
        strain_idx = 2  # Hoop strain
        strain_name = 'Hoop strain (εθ)'
        expected = strain_theta
    else:
        strain_idx = 3  # Shear strain
        strain_name = 'Shear strain (γrz)'
        expected = strain_rz
    
    sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=strains[:, strain_idx], cmap='viridis', s=50)
    plt.colorbar(sc, label=strain_name)
    plt.title(f'{strain_name} (Target: {expected})')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.axis('equal')
    
    # Plot 4: Strain error
    plt.subplot(2, 2, 4)
    strain_comp_errors = np.abs(strains[:, strain_idx] - expected)
    sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=strain_comp_errors, cmap='jet', s=50)
    plt.colorbar(sc, label=f'{strain_name} error')
    plt.title(f'{strain_name} error')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'vem_quad_patch_test_case_{strain_case}.png')
    plt.close()

def test_load_vectors():
    """
    Test the computation of load vectors in the axisymmetric VEM formulation.
    
    This test verifies:
    1. Body force load vector computation
    2. Boundary traction load vector computation
    3. Combined load vector assembly
    """
    print("\n=== TESTING LOAD VECTOR COMPUTATION ===")
    
    # Material properties
    E = 1.0       # Young's modulus
    nu = 0.3      # Poisson's ratio
    
    # Create a simple mesh (2x2 elements)
    n_r = 2  # elements in r direction
    n_z = 2  # elements in z direction
    
    # Domain setup
    r_inner = 1.0
    r_outer = 2.0
    z_min = 0.0
    z_max = 1.0
    
    # Generate mesh nodes
    nr_nodes = n_r + 1
    nz_nodes = n_z + 1
    grid_nodes = np.zeros((nr_nodes * nz_nodes, 2))  # (r, z) coordinates
    
    node_idx = 0
    for iz in range(nz_nodes):
        for ir in range(nr_nodes):
            r = r_inner + (r_outer - r_inner) * ir / (nr_nodes - 1)
            z = z_min + (z_max - z_min) * iz / (nz_nodes - 1)
            grid_nodes[node_idx] = [r, z]
            node_idx += 1
    
    # Create quadrilateral elements
    total_quads = n_r * n_z
    quads = np.zeros((total_quads, 4), dtype=int)
    
    quad_idx = 0
    for iz in range(n_z):
        for ir in range(n_r):
            n1 = iz * nr_nodes + ir
            n2 = iz * nr_nodes + (ir + 1)
            n3 = (iz + 1) * nr_nodes + (ir + 1)
            n4 = (iz + 1) * nr_nodes + ir
            quads[quad_idx] = [n1, n2, n3, n4]
            quad_idx += 1
    
    nodes = grid_nodes.copy()
    elements = quads.copy()
    
    # Test 1: Body Force Load Vector
    print("\n1. Testing body force load vector")
    
    # Define a constant body force function
    def constant_body_force(r, z):
        return 1.0, -1.0  # Constant force in r and z directions
    
    body_forces = []
    for e in range(len(elements)):
        # Get element vertices
        element_vertices = nodes[elements[e]]
        
        # Compute element body force vector
        f_body = compute_element_load_body_force(element_vertices, constant_body_force)
        body_forces.append(f_body)
        
        # Verify the size of the load vector
        assert f_body.shape[0] == 8, f"Body force vector should have 8 components for quad element {e}"
        
        # Verify that the load vector is not zero
        assert not np.allclose(f_body, 0), f"Body force vector should not be zero for element {e}"
    
    print("✓ Body force load vector computation passed basic checks")
    
    # Test 2: Boundary Traction Load Vector
    print("\n2. Testing boundary traction load vector")
    
    # Define a constant traction function
    def constant_traction(r, z):
        return 2.0, 0.0  # Constant traction in r direction only
    
    traction_forces = []
    for e in range(len(elements)):
        element_vertices = nodes[elements[e]]
        
        # Check if any edge of this element is on the outer boundary (r = r_outer)
        edge_indices = []
    for i in range(4):
            j = (i + 1) % 4
            v1 = element_vertices[i]
            v2 = element_vertices[j]
            
            # If both vertices are on outer boundary
            if abs(v1[0] - r_outer) < 1e-6 and abs(v2[0] - r_outer) < 1e-6:
                edge_indices.append(i)
        
            if edge_indices:
                # Compute traction load vector for this edge
                f_trac = compute_element_load_boundary_traction(
                    element_vertices, edge_indices, constant_traction)
                traction_forces.append(f_trac)
                
                # Verify the size of the load vector
                assert f_trac.shape[0] == 8, f"Traction vector should have 8 components for quad element {e}"
                
                # Verify that the load vector is not zero
                assert not np.allclose(f_trac, 0), f"Traction vector should not be zero for boundary element {e}"
    
    print("✓ Boundary traction load vector computation passed basic checks")
    
    # Test 3: Combined Load Vector Assembly
    print("\n3. Testing load vector assembly")
    
    # Initialize global load vector
    ndof = 2 * len(nodes)
    F_global = np.zeros(ndof)
    
    # Assemble body forces and boundary tractions
    for e in range(len(elements)):
        element_vertices = nodes[elements[e]]
        node_indices = elements[e]
        
        # Compute both load vectors
        f_body = compute_element_load_body_force(element_vertices, constant_body_force)
        
        # Initialize traction vector
        f_trac = np.zeros_like(f_body)
        
        # Check each edge for boundary traction
        edge_indices = []
        for i in range(4):
            j = (i + 1) % 4
            v1 = element_vertices[i]
            v2 = element_vertices[j]
            
            if abs(v1[0] - r_outer) < 1e-6 and abs(v2[0] - r_outer) < 1e-6:
                edge_indices.append(i)
        
        if edge_indices:
            f_trac = compute_element_load_boundary_traction(
                element_vertices, edge_indices, constant_traction)
        
        # Combine load vectors
        f_total = assemble_element_load_vector(element_vertices, 
                                             constant_body_force,
                                             edge_indices if edge_indices else None,
                                             constant_traction if edge_indices else None)
        
        # Verify the size of the combined load vector
        assert f_total.shape[0] == 8, f"Combined load vector should have 8 components"
        
        # Assemble into global load vector
        for i in range(4):
            F_global[2*node_indices[i]] += f_total[2*i]      # r-component
            F_global[2*node_indices[i]+1] += f_total[2*i+1]  # z-component
    
    # Verify that the global load vector is not zero
    assert not np.allclose(F_global, 0), "Global load vector should not be zero"
    
    print("✓ Load vector assembly passed basic checks")
    
    # Verify equilibrium
    total_force_r = np.sum(F_global[0::2])
    total_force_z = np.sum(F_global[1::2])
    
    print(f"\nTotal force in r-direction: {total_force_r:.6e}")
    print(f"Total force in z-direction: {total_force_z:.6e}")
    
    # For this setup, we expect the total vertical force to be negative due to gravity
    assert total_force_z < 0, "Total vertical force should be negative due to gravity"
    
    print("\nLoad vector tests completed successfully!")
    return True

def run_comparative_stabilization_test():
    """
    Run a test comparing standard stabilization with boundary stabilization.
    
    This test will:
    1. Use a distorted mesh with some highly skewed elements
    2. Solve a problem where over-stiffening is problematic
    3. Compare the accuracy of both methods
    """
    print("=== COMPARATIVE STABILIZATION TEST: STANDARD VS BOUNDARY ===")
    
    # Material properties
    E = 1.0       # Young's modulus
    nu = 0.4      # Use a higher Poisson ratio for more challenging conditions
    
    # Test several mesh resolutions to see convergence patterns
    meshes = [4, 8, 16]  # Number of elements per side
    
    results = {
        "standard": {"errors": [], "condition_numbers": []},
        "boundary": {"errors": [], "condition_numbers": []}
    }
    
    for n_elem in meshes:
        print(f"\nTesting mesh with {n_elem}x{n_elem} elements")
        
        # Run the same test with both stabilization methods
        for stab_type in ["standard", "boundary"]:
            print(f"\n--- Using {stab_type} stabilization ---")
            
            # Create a distorted mesh (this is key to show differences)
            nodes, elements = create_distorted_mesh(n_elem)
            
            # Set up the problem
            K_global, u_exact, boundary_dofs = setup_bending_problem(nodes, elements, E, nu, stab_type)
            
            # Calculate condition number
            cond_num = calculate_condition_number(K_global)
            results[stab_type]["condition_numbers"].append(cond_num)
            print(f"Condition number: {cond_num:.2e}")
            
            # Solve the system and compute error
            u, error = solve_and_evaluate(K_global, u_exact, boundary_dofs)
            results[stab_type]["errors"].append(error)
            print(f"L2 error: {error:.6e}")
    
    # Plot results
    # plot_comparative_results(meshes, results)
    
    return results

def run_enhanced_comparative_test():
    """
    Run an enhanced test comparing standard vs boundary stabilization
    with specific scenarios designed to highlight differences.
    """
    print("=== ENHANCED COMPARATIVE STABILIZATION TEST ===")
    
    # Use high Poisson's ratio - near incompressible material
    E = 1.0
    nu = 0.499  # Very close to incompressible
    
    # Test several mesh resolutions
    meshes = [4, 8, 16]
    
    results = {
        "standard": {"errors": [], "condition_numbers": []},
        "boundary": {"errors": [], "condition_numbers": []}
    }
    
    for n_elem in meshes:
        print(f"\nTesting mesh with {n_elem}x{n_elem} elements")
        
        # Run with both stabilization methods
        for stab_type in ["standard", "boundary"]:
            print(f"\n--- Using {stab_type} stabilization ---")
            
            # Create a thin-walled cylinder mesh with high distortion
            nodes, elements = create_thin_cylinder_mesh(n_elem, distortion=0.4)
            
            # Set up a bending problem (thin structures are more sensitive to stabilization)
            K_global, u_exact, boundary_dofs = setup_bending_problem(
                nodes, elements, E, nu, stab_type, 
                h_scaling=3 if stab_type == "boundary" else -2
            )
            
            # Calculate condition number
            cond_num = estimate_condition_number(K_global)
            results[stab_type]["condition_numbers"].append(cond_num)
            print(f"Condition number: {cond_num:.2e}")
            
            # Solve and evaluate error
            u, error = solve_and_evaluate_error(K_global, u_exact, boundary_dofs)
            results[stab_type]["errors"].append(error)
            print(f"L2 error: {error:.6e}")
    
    # Plot comparison results
    plot_enhanced_results(meshes, results)
    
    return results

def create_distorted_mesh(n_elem):
    """
    Create a mesh with distorted elements to highlight stabilization differences.
    
    Parameters:
        n_elem (int): Number of elements per side
    
    Returns:
        tuple: (nodes, elements) arrays defining the mesh
    """
    # Domain setup
    r_inner = 1.0
    r_outer = 3.0
    z_min = 0.0
    z_max = 2.0
    
    # First create a regular mesh
    nr_nodes = n_elem + 1
    nz_nodes = n_elem + 1
    grid_nodes = np.zeros((nr_nodes * nz_nodes, 2))
    
    node_idx = 0
    for iz in range(nz_nodes):
        for ir in range(nr_nodes):
            r = r_inner + (r_outer - r_inner) * ir / (nr_nodes - 1)
            z = z_min + (z_max - z_min) * iz / (nz_nodes - 1)
            grid_nodes[node_idx] = [r, z]
            node_idx += 1
    
    # Create quadrilateral elements
    total_quads = n_elem * n_elem
    quads = np.zeros((total_quads, 4), dtype=int)
    
    quad_idx = 0
    for iz in range(n_elem):
        for ir in range(n_elem):
            n1 = iz * nr_nodes + ir
            n2 = iz * nr_nodes + (ir + 1)
            n3 = (iz + 1) * nr_nodes + (ir + 1)
            n4 = (iz + 1) * nr_nodes + ir
            quads[quad_idx] = [n1, n2, n3, n4]
            quad_idx += 1
    
    # Now distort the mesh - this is key to showing the difference
    # Between stabilization approaches
    np.random.seed(42)  # For reproducibility
    
    # Distort interior nodes (leave boundary nodes fixed)
    for i in range(len(grid_nodes)):
        r, z = grid_nodes[i]
        
        # Skip boundary nodes
        if (abs(r - r_inner) < 1e-6 or abs(r - r_outer) < 1e-6 or 
            abs(z - z_min) < 1e-6 or abs(z - z_max) < 1e-6):
            continue
        
        # Add distortion - more pronounced near inner radius
        # where elements are smaller
        distortion_factor = 0.2 * (r_outer - r) / (r_outer - r_inner)
        
        # Calculate element size for scaling distortion
        h = (r_outer - r_inner) / n_elem
        
        # Apply random distortion
        grid_nodes[i, 0] += distortion_factor * h * (np.random.random() - 0.5)
        grid_nodes[i, 1] += distortion_factor * h * (np.random.random() - 0.5)
    
    return grid_nodes, quads

def setup_bending_problem(nodes, elements, E, nu, stab_type):
    """
    Set up a bending problem that highlights differences in stabilization.
    
    Parameters:
        nodes (np.ndarray): Mesh nodes
        elements (np.ndarray): Element connectivity
        E (float): Young's modulus
        nu (float): Poisson's ratio
        stab_type (str): Type of stabilization ("standard" or "boundary")
    
    Returns:
        tuple: (K_global, u_exact, boundary_dofs)
    """
    # Total degrees of freedom
    total_nodes = len(nodes)
    ndof = 2 * total_nodes
    
    # Assemble global stiffness matrix
    K_global = lil_matrix((ndof, ndof))
    
    for e in range(len(elements)):
        # Get vertex indices for this element
        node_indices = elements[e]
        
        # Get node coordinates for this element
        element_vertices = nodes[node_indices]
        
        # Compute element stiffness matrix with specified stabilization
        K_elem, _, _ = compute_stiffness_matrix(element_vertices, E, nu, stab_type)
        
        # Map local DOFs to global DOFs
        dof_indices = np.zeros(2 * len(node_indices), dtype=int)
        for i in range(len(node_indices)):
            dof_indices[2*i] = 2 * node_indices[i]     # r-displacement
            dof_indices[2*i+1] = 2 * node_indices[i] + 1  # z-displacement
        
        # Assemble element matrix into global matrix
        for i in range(len(dof_indices)):
            for j in range(len(dof_indices)):
                K_global[dof_indices[i], dof_indices[j]] += K_elem[i, j]
    
    # Find boundary nodes
    boundary_nodes = []
    for i in range(total_nodes):
        r, z = nodes[i]
        if (abs(r - 1.0) < 1e-6 or abs(r - 3.0) < 1e-6 or 
            abs(z - 0.0) < 1e-6 or abs(z - 2.0) < 1e-6):
            boundary_nodes.append(i)
    
    # Create boundary DOFs list
    boundary_dofs = []
    for i in boundary_nodes:
        boundary_dofs.extend([2*i, 2*i+1])
    
    # Create an exact solution for a more complex deformation
    # This will be a cylindrical bending mode
    u_exact = np.zeros(ndof)
    
    # Parameter to control deformation complexity
    alpha = 0.1
    beta = 0.05
    
    for i in range(total_nodes):
        r, z = nodes[i]
        
        # Radial displacement (with quadratic variation in z)
        u_exact[2*i] = alpha * r * (z**2 - 2*z)
        
        # Axial displacement (with cubic variation in r)
        u_exact[2*i+1] = beta * (r**3 - 3*r*r + 2*r) * z
    
    # Convert to CSR format for efficient solving
    K_global_csr = K_global.tocsr()
    
    return K_global_csr, u_exact, boundary_dofs

def solve_and_evaluate(K_global, u_exact, boundary_dofs):
    """
    Solve the system and calculate the error.
    
    Parameters:
        K_global (scipy.sparse.csr_matrix): Global stiffness matrix
        u_exact (np.ndarray): Exact solution
        boundary_dofs (list): DOFs on the boundary
    
    Returns:
        tuple: (u, error) - Solution and L2 error
    """
    # Total degrees of freedom
    ndof = K_global.shape[0]
    
    # Initialize force vector
    F = np.zeros(ndof)
    
    # Set up boundary conditions
    free_dofs = np.ones(ndof, dtype=bool)
    free_dofs[boundary_dofs] = False
    
    # Solve the system
    F_mod = F - K_global.dot(u_exact)
    F_reduced = F_mod[free_dofs]
    K_reduced = K_global[free_dofs, :][:, free_dofs]
    
    u_reduced = spsolve(K_reduced, F_reduced)
    u = u_exact.copy()
    u[free_dofs] = u_reduced
    
    # Calculate L2 error
    error = np.sqrt(np.sum((u - u_exact)**2)) / np.sqrt(np.sum(u_exact**2))
    
    return u, error

def calculate_condition_number(K_global):
    """
    Calculate the condition number of the stiffness matrix.
    
    Parameters:
        K_global (scipy.sparse.csr_matrix): Global stiffness matrix
    
    Returns:
        float: Condition number
    """
    # For large matrices, we use an estimate
    # Convert to dense for smaller matrices
    if K_global.shape[0] <= 1000:
        # Convert to dense and use numpy
        K_dense = K_global.toarray()
        
        # Use SVD to compute condition number
        s = np.linalg.svd(K_dense, compute_uv=False)
        return s[0] / s[-1]
    else:
        # Use an iterative estimator for large matrices
        from scipy.sparse.linalg import lobpcg
        
        # Estimate largest and smallest eigenvalues
        n = K_global.shape[0]
        
        # Random initial vectors
        x_max = np.random.rand(n, 1)
        x_min = np.random.rand(n, 1)
        
        # Compute largest eigenvalue
        lambda_max, _ = lobpcg(K_global, x_max, largest=True, maxiter=100)
        
        # For smallest, we invert the matrix action
        def inv_matvec(x):
            return spsolve(K_global, x.ravel()).reshape(-1, 1)
        
        # Create a LinearOperator representing the inverse
        from scipy.sparse.linalg import LinearOperator
        K_inv = LinearOperator((n, n), matvec=inv_matvec)
        
        # Use power iteration to estimate smallest eigenvalue
        lambda_min_inv, _ = lobpcg(K_inv, x_min, largest=True, maxiter=100)
        lambda_min = 1.0 / lambda_min_inv[0]
        
        return lambda_max[0] / lambda_min
    
def plot_comparative_results(meshes, results):
    """
    Plot the comparative results of different stabilization methods.
    
    Parameters:
        meshes (list): List of mesh sizes
        results (dict): Dictionary of results for each method
    """
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Error convergence
    plt.subplot(2, 1, 1)
    h_values = [2.0/n for n in meshes]  # Element size
    
    plt.loglog(h_values, results["standard"]["errors"], 'o-', label='Standard Stabilization')
    plt.loglog(h_values, results["boundary"]["errors"], 's-', label='Boundary Stabilization')
    
    # Add reference lines for h^1 and h^2 convergence
    h_ref = np.array([h_values[0], h_values[-1]])
    err_ref1 = results["standard"]["errors"][0] * (h_ref / h_values[0])
    err_ref2 = results["standard"]["errors"][0] * (h_ref / h_values[0])**2
    
    plt.loglog(h_ref, err_ref1, 'k--', label='O(h)')
    plt.loglog(h_ref, err_ref2, 'k:', label='O(h²)')
    
    plt.grid(True)
    plt.xlabel('Element Size (h)')
    plt.ylabel('Relative L2 Error')
    plt.title('Error Convergence for Different Stabilization Methods')
    plt.legend()
    
    # Plot 2: Condition number
    plt.subplot(2, 1, 2)
    
    plt.semilogy(meshes, results["standard"]["condition_numbers"], 'o-', label='Standard Stabilization')
    plt.semilogy(meshes, results["boundary"]["condition_numbers"], 's-', label='Boundary Stabilization')
    
    plt.grid(True)
    plt.xlabel('Number of Elements per Side')
    plt.ylabel('Condition Number (log scale)')
    plt.title('Matrix Condition Number for Different Stabilization Methods')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('stabilization_comparison.png')
    plt.close()

def create_thin_cylinder_mesh(n_elem, distortion=0.3):
    """
    Create a thin-walled cylinder mesh with controlled distortion.
    
    Parameters:
        n_elem (int): Number of elements along each direction
        distortion (float): Level of element distortion (0-1)
    
    Returns:
        tuple: (nodes, elements)
    """
    # Thin cylinder parameters
    r_inner = 9.0
    r_outer = 10.0  # Only 1.0 thick wall (10% of radius)
    height = 5.0
    
    # Number of elements in each direction
    n_r = max(2, n_elem // 8)  # Fewer elements in radial direction
    n_z = n_elem
    n_theta = n_elem  # Elements around circumference
    
    # Generate nodes
    total_nodes = (n_r + 1) * (n_z + 1) * (n_theta + 1)
    nodes = np.zeros((total_nodes, 3))  # (r, theta, z) coordinates initially
    
    node_idx = 0
    for ir in range(n_r + 1):
        r = r_inner + (r_outer - r_inner) * ir / n_r
        for iz in range(n_z + 1):
            z = height * iz / n_z
            for itheta in range(n_theta + 1):
                theta = 2 * np.pi * itheta / n_theta
                
                # Store in cylindrical coordinates initially
                nodes[node_idx] = [r, theta, z]
                node_idx += 1
    
    # Apply distortion in cylindrical coordinates
    if distortion > 0:
        np.random.seed(42)
        for i in range(len(nodes)):
            r, theta, z = nodes[i]
            
            # Skip nodes at inner/outer surfaces
            if abs(r - r_inner) < 1e-6 or abs(r - r_outer) < 1e-6:
                continue
                
            # Distortion factor - higher in the middle of the cylinder
            z_factor = 1.0 - 4 * (z / height - 0.5)**2  # Max at z=height/2
            
            # Apply random distortion in r and z directions
            dr = distortion * (r_outer - r_inner) / n_r * z_factor * (np.random.random() - 0.5)
            dz = distortion * height / n_z * (np.random.random() - 0.5)
            
            # Update coordinates
            nodes[i, 0] += dr
            nodes[i, 2] += dz
    
    # Convert to Cartesian for 2D axisymmetric analysis
    # In axisymmetric, we only need the r-z slice at theta=0
    axisym_nodes = []
    node_map = {}  # Map from 3D node index to 2D node index
    
    for i in range(len(nodes)):
        r, theta, z = nodes[i]
        if abs(theta) < 1e-6 or abs(theta - 2*np.pi) < 1e-6:
            node_map[i] = len(axisym_nodes)
            axisym_nodes.append([r, z])
    
    # Create quad elements in r-z plane
    elements = []
    for ir in range(n_r):
        for iz in range(n_z):
            # Get 3D indices
            n1 = ir * (n_z + 1) * (n_theta + 1) + iz * (n_theta + 1) + 0
            n2 = (ir + 1) * (n_z + 1) * (n_theta + 1) + iz * (n_theta + 1) + 0
            n3 = (ir + 1) * (n_z + 1) * (n_theta + 1) + (iz + 1) * (n_theta + 1) + 0
            n4 = ir * (n_z + 1) * (n_theta + 1) + (iz + 1) * (n_theta + 1) + 0
            
            # Convert to 2D indices
            if n1 in node_map and n2 in node_map and n3 in node_map and n4 in node_map:
                elements.append([node_map[n1], node_map[n2], node_map[n3], node_map[n4]])
    
    return np.array(axisym_nodes), np.array(elements)



if __name__ == "__main__":
    # Run both the original patch tests and the new load vector tests
    patch_test_passed = run_vem_quad_patch_test()
    load_test_passed = test_load_vectors()
    comparative_test_passed = run_comparative_stabilization_test()
    
    print("\n=== FINAL TEST SUMMARY ===")
    print(f"Patch tests: {'PASSED' if patch_test_passed else 'FAILED'}")
    print(f"Load vector tests: {'PASSED' if load_test_passed else 'FAILED'}")
    