# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.sparse import lil_matrix
# from scipy.sparse.linalg import spsolve

# # Material properties
# E = 1.0       # Young's modulus
# nu = 0.3      # Poisson's ratio

# # Calculate constitutive matrix for axisymmetric elasticity
# factor = E / ((1 + nu) * (1 - 2*nu))
# C = factor * np.array([
#     [1-nu, nu, nu, 0],
#     [nu, 1-nu, nu, 0],
#     [nu, nu, 1-nu, 0],
#     [0, 0, 0, (1-2*nu)/2]
# ])

# print("Constitutive matrix C:")
# print(C)

# # Domain setup
# r_inner = 1.0
# r_outer = 3.0
# z_min = 0.0
# z_max = 2.0

# # Create a simple mesh (4x4 elements)
# n_r = 4  # elements in r direction
# n_z = 4  # elements in z direction

# # Generate mesh nodes
# nr_nodes = n_r + 1
# nz_nodes = n_z + 1
# total_nodes = nr_nodes * nz_nodes
# print(f"\nTotal nodes: {total_nodes}")

# # Node coordinates
# nodes = np.zeros((total_nodes, 2))  # (r, z) coordinates
# node_idx = 0
# for iz in range(nz_nodes):
#     for ir in range(nr_nodes):
#         r = r_inner + (r_outer - r_inner) * ir / (nr_nodes - 1)
#         z = z_min + (z_max - z_min) * iz / (nz_nodes - 1)
#         nodes[node_idx] = [r, z]
#         node_idx += 1

# # Create elements (connectivity)
# total_elements = n_r * n_z
# elements = np.zeros((total_elements, 4), dtype=int)
# elem_idx = 0
# for iz in range(n_z):
#     for ir in range(n_r):
#         n1 = iz * nr_nodes + ir
#         n2 = iz * nr_nodes + (ir + 1)
#         n3 = (iz + 1) * nr_nodes + (ir + 1)
#         n4 = (iz + 1) * nr_nodes + ir
#         elements[elem_idx] = [n1, n2, n3, n4]
#         elem_idx += 1

# print(f"Total elements: {total_elements}")
# print("First element nodes:", elements[0])

# # Total degrees of freedom
# ndof = 2 * total_nodes  # 2 DOFs per node (r, z displacements)
# print(f"Total DOFs: {ndof}")

# # Define Gauss quadrature points for numerical integration
# def get_gauss_points(n_points=2):
#     """Get Gauss quadrature points and weights for a square [-1,1]×[-1,1]"""
#     if n_points == 2:
#         # 2×2 Gauss points
#         points = np.array([
#             [-1/np.sqrt(3), -1/np.sqrt(3)],
#             [1/np.sqrt(3), -1/np.sqrt(3)],
#             [1/np.sqrt(3), 1/np.sqrt(3)],
#             [-1/np.sqrt(3), 1/np.sqrt(3)]
#         ])
#         weights = np.ones(4)
#         return points, weights
#     else:
#         raise ValueError("Only 2×2 quadrature implemented")

# # Shape functions for bilinear quadrilateral
# def shape_functions(xi, eta):
#     """Evaluate shape functions at (xi, eta)"""
#     N = np.zeros(4)
#     N[0] = 0.25 * (1 - xi) * (1 - eta)
#     N[1] = 0.25 * (1 + xi) * (1 - eta)
#     N[2] = 0.25 * (1 + xi) * (1 + eta)
#     N[3] = 0.25 * (1 - xi) * (1 + eta)
#     return N

# def shape_function_derivatives(xi, eta):
#     """Evaluate shape function derivatives w.r.t. (xi, eta)"""
#     dN_dxi = np.zeros(4)
#     dN_deta = np.zeros(4)
    
#     dN_dxi[0] = -0.25 * (1 - eta)
#     dN_dxi[1] = 0.25 * (1 - eta)
#     dN_dxi[2] = 0.25 * (1 + eta)
#     dN_dxi[3] = -0.25 * (1 + eta)
    
#     dN_deta[0] = -0.25 * (1 - xi)
#     dN_deta[1] = -0.25 * (1 + xi)
#     dN_deta[2] = 0.25 * (1 + xi)
#     dN_deta[3] = 0.25 * (1 - xi)
    
#     return dN_dxi, dN_deta

# # Improved element stiffness calculation with proper numerical integration
# def calculate_element_stiffness(node_coords):
#     """Calculate element stiffness matrix with proper numerical integration"""
#     # Gauss quadrature setup
#     gauss_points, gauss_weights = get_gauss_points(2)
    
#     # Initialize element matrices
#     K_elem = np.zeros((8, 8))
    
#     # Numerical integration loop
#     for i, (xi, eta) in enumerate(gauss_points):
#         # Shape function derivatives
#         dN_dxi, dN_deta = shape_function_derivatives(xi, eta)
        
#         # Jacobian matrix
#         J = np.zeros((2, 2))
#         J[0, 0] = np.sum(dN_dxi * node_coords[:, 0])  # dr/dxi
#         J[0, 1] = np.sum(dN_dxi * node_coords[:, 1])  # dz/dxi
#         J[1, 0] = np.sum(dN_deta * node_coords[:, 0])  # dr/deta
#         J[1, 1] = np.sum(dN_deta * node_coords[:, 1])  # dz/deta
        
#         # Determinant of Jacobian
#         detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        
#         # Inverse of Jacobian
#         Jinv = np.zeros((2, 2))
#         Jinv[0, 0] = J[1, 1] / detJ
#         Jinv[0, 1] = -J[0, 1] / detJ
#         Jinv[1, 0] = -J[1, 0] / detJ
#         Jinv[1, 1] = J[0, 0] / detJ
        
#         # Shape function derivatives w.r.t. r and z
#         dN_dr = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
#         dN_dz = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
        
#         # Compute r at the current Gauss point (for axisymmetric weight)
#         N = shape_functions(xi, eta)
#         r_gauss = np.sum(N * node_coords[:, 0])
        
#         # B matrix for axisymmetric elasticity (improved version)
#         B = np.zeros((4, 8))
        
#         # Loop over nodes to assemble B matrix
#         for n in range(4):
#             # Columns for radial displacement (ur)
#             B[0, 2*n] = dN_dr[n]               # εr = ∂ur/∂r
#             B[2, 2*n] = N[n] / r_gauss         # εθ = ur/r (improved formulation)
#             B[3, 2*n] = dN_dz[n]               # γrz = ∂ur/∂z + ∂uz/∂r (part 1)
            
#             # Columns for axial displacement (uz)
#             B[1, 2*n+1] = dN_dz[n]             # εz = ∂uz/∂z
#             B[3, 2*n+1] = dN_dr[n]             # γrz = ∂ur/∂z + ∂uz/∂r (part 2)
        
#         # Contribution to stiffness matrix: B^T * C * B * r * detJ * weight
#         K_gauss = np.matmul(np.matmul(B.T, C), B) * r_gauss * detJ * gauss_weights[i]
#         K_elem += K_gauss
    
#     # Add stabilization term - improved version with proper scaling
#     # Extract the part of stiffness related to constant strains
#     P0 = np.zeros((8, 4))
#     for n in range(4):
#         # Constant radial strain mode
#         P0[2*n, 0] = node_coords[n, 0]  # ur = r
#         # Constant axial strain mode
#         P0[2*n+1, 1] = node_coords[n, 1]  # uz = z
#         # Constant hoop strain mode
#         P0[2*n, 2] = node_coords[n, 0]  # ur = r
#         # Constant shear strain mode
#         P0[2*n, 3] = node_coords[n, 1] / 2  # ur = z/2
#         P0[2*n+1, 3] = node_coords[n, 0] / 2  # uz = r/2
    
#     # Orthogonalize P0
#     Q, R = np.linalg.qr(P0)
#     rank = np.sum(np.abs(np.diag(R)) > 1e-10)
#     Pc = Q[:, :rank]
    
#     # Projection matrix onto the orthogonal complement of Pc
#     I = np.eye(8)
#     Pi = I - np.matmul(Pc, Pc.T)
    
#     # Calculate trace for scaling
#     alpha = np.trace(K_elem) / np.trace(Pi)
    
#     # Stabilization term
#     K_stab = alpha * Pi
    
#     # Complete element stiffness matrix
#     K_elem_final = K_elem + K_stab
    
#     return K_elem_final, B

# # Assemble global stiffness matrix
# K_global = lil_matrix((ndof, ndof))

# for e in range(total_elements):
#     node_indices = elements[e]
#     node_coords = nodes[node_indices]
    
#     K_elem, _ = calculate_element_stiffness(node_coords)
    
#     # Map local DOFs to global DOFs
#     dof_indices = np.zeros(8, dtype=int)
#     for i in range(4):
#         dof_indices[2*i] = 2 * node_indices[i]     # r-displacement
#         dof_indices[2*i+1] = 2 * node_indices[i] + 1  # z-displacement
    
#     # Assemble element matrix into global matrix
#     for i in range(8):
#         for j in range(8):
#             K_global[dof_indices[i], dof_indices[j]] += K_elem[i, j]

# # Find boundary nodes
# boundary_nodes = []
# for i in range(total_nodes):
#     r, z = nodes[i]
#     if (abs(r - r_inner) < 1e-6 or abs(r - r_outer) < 1e-6 or 
#         abs(z - z_min) < 1e-6 or abs(z - z_max) < 1e-6):
#         boundary_nodes.append(i)

# print(f"\nNumber of boundary nodes: {len(boundary_nodes)}")

# def run_patch_test(strain_case):
#     """
#     Run patch test for the given strain state:
#     1 = Constant radial strain
#     2 = Constant axial strain
#     3 = Constant hoop strain
#     4 = Constant shear strain
#     """
#     # Force vector
#     F = np.zeros(ndof)
    
#     # Define strain state based on test case
#     strain_r = 0.0
#     strain_z = 0.0
#     strain_theta = 0.0
#     strain_rz = 0.0
    
#     if strain_case == 1:
#         strain_r = 0.01      # Radial strain test
#         test_name = "Radial Strain"
#     elif strain_case == 2:
#         strain_z = 0.01      # Axial strain test
#         test_name = "Axial Strain"
#     elif strain_case == 3:
#         strain_theta = 0.01  # Hoop strain test
#         test_name = "Hoop Strain"
#     elif strain_case == 4:
#         strain_rz = 0.01     # Shear strain test
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
#     K_csr = K_global.tocsr()
#     F_mod = F - K_csr.dot(u_exact)
#     F_reduced = F_mod[free_dofs]
#     K_reduced = K_csr[free_dofs, :][:, free_dofs]
    
#     u_reduced = spsolve(K_reduced, F_reduced)
#     u = u_exact.copy()
#     u[free_dofs] = u_reduced
    
#     # Extract displacements
#     u_r = u[0::2]
#     u_z = u[1::2]
    
#     # Calculate strains at Gauss points and project to element centroids
#     strains = np.zeros((total_elements, 4))  # [εr, εz, εθ, γrz]
    
#     # Define function to calculate strain at a point
#     def calculate_strain_at_point(elem_idx, xi, eta, u_elem):
#         node_indices = elements[elem_idx]
#         node_coords = nodes[node_indices]
        
#         # Shape function derivatives
#         dN_dxi, dN_deta = shape_function_derivatives(xi, eta)
        
#         # Jacobian matrix
#         J = np.zeros((2, 2))
#         J[0, 0] = np.sum(dN_dxi * node_coords[:, 0])  # dr/dxi
#         J[0, 1] = np.sum(dN_dxi * node_coords[:, 1])  # dz/dxi
#         J[1, 0] = np.sum(dN_deta * node_coords[:, 0])  # dr/deta
#         J[1, 1] = np.sum(dN_deta * node_coords[:, 1])  # dz/deta
        
#         # Determinant of Jacobian
#         detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        
#         # Inverse of Jacobian
#         Jinv = np.zeros((2, 2))
#         Jinv[0, 0] = J[1, 1] / detJ
#         Jinv[0, 1] = -J[0, 1] / detJ
#         Jinv[1, 0] = -J[1, 0] / detJ
#         Jinv[1, 1] = J[0, 0] / detJ
        
#         # Shape function derivatives w.r.t. r and z
#         dN_dr = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
#         dN_dz = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
        
#         # Compute r at the current point
#         N = shape_functions(xi, eta)
#         r_point = np.sum(N * node_coords[:, 0])
        
#         # Calculate displacements at this point
#         u_r_point = np.sum(N * u_elem[0::2])
#         u_z_point = np.sum(N * u_elem[1::2])
        
#         # Calculate spatial derivatives
#         du_r_dr = np.sum(dN_dr * u_elem[0::2])
#         du_r_dz = np.sum(dN_dz * u_elem[0::2])
#         du_z_dr = np.sum(dN_dr * u_elem[1::2])
#         du_z_dz = np.sum(dN_dz * u_elem[1::2])
        
#         # Calculate strains
#         strain = np.zeros(4)
#         strain[0] = du_r_dr                  # εr = ∂ur/∂r
#         strain[1] = du_z_dz                  # εz = ∂uz/∂z
#         strain[2] = u_r_point / r_point      # εθ = ur/r
#         strain[3] = du_r_dz + du_z_dr        # γrz = ∂ur/∂z + ∂uz/∂r
        
#         return strain
    
#     # Calculate strains at element centroids
#     for e in range(total_elements):
#         node_indices = elements[e]
        
#         # Get element displacements
#         elem_disps = np.zeros(8)
#         for i in range(4):
#             elem_disps[2*i] = u[2*node_indices[i]]       # ur
#             elem_disps[2*i+1] = u[2*node_indices[i]+1]   # uz
        
#         # Evaluate strains at element center (ξ=η=0)
#         strains[e] = calculate_strain_at_point(e, 0, 0, elem_disps)
    
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
#     tolerance = 1e-5
#     passed = np.all(strain_errors < tolerance)
    
#     print(f"\nPatch test for {test_name}: {'PASSED' if passed else 'FAILED'} with tolerance {tolerance}")
    
#     # Plotting
#     plt.figure(figsize=(15, 12))
    
#     # Plot mesh
#     plt.subplot(2, 2, 1)
#     for e in range(total_elements):
#         node_indices = elements[e]
#         r_coords = nodes[node_indices, 0]
#         z_coords = nodes[node_indices, 1]
#         # Close the loop for plotting
#         r_coords = np.append(r_coords, r_coords[0])
#         z_coords = np.append(z_coords, z_coords[0])
#         plt.plot(r_coords, z_coords, 'k-', linewidth=0.5)
#     plt.scatter(nodes[:, 0], nodes[:, 1], c='b', s=20)
#     plt.title('Mesh')
#     plt.xlabel('r')
#     plt.ylabel('z')
#     plt.axis('equal')
    
#     # Plot displacement
#     plt.subplot(2, 2, 2)
#     if strain_case in [1, 3]:  # Radial or hoop strain
#         plot_data = u_r
#         title = 'Radial displacement (ur)'
#     elif strain_case == 2:  # Axial strain
#         plot_data = u_z
#         title = 'Axial displacement (uz)'
#     else:  # Shear strain
#         # For shear, plot magnitude
#         plot_data = np.sqrt(u_r**2 + u_z**2)
#         title = 'Displacement magnitude'
        
#     sc = plt.scatter(nodes[:, 0], nodes[:, 1], c=plot_data, cmap='viridis', s=30)
#     plt.colorbar(sc, label=title)
#     plt.title(title)
#     plt.xlabel('r')
#     plt.ylabel('z')
#     plt.axis('equal')
    
#     # Plot the specific strain component being tested
#     plt.subplot(2, 2, 3)
#     # Element centroids for plotting strain
#     centroids = np.zeros((total_elements, 2))
#     for e in range(total_elements):
#         node_indices = elements[e]
#         centroids[e] = np.mean(nodes[node_indices], axis=0)
    
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
    
#     # Plot strain error
#     plt.subplot(2, 2, 4)
#     strain_comp_errors = np.abs(strains[:, strain_idx] - expected)
#     sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=strain_comp_errors, cmap='jet', s=50)
#     plt.colorbar(sc, label=f'{strain_name} error')
#     plt.title(f'{strain_name} error')
#     plt.xlabel('r')
#     plt.ylabel('z')
#     plt.axis('equal')
    
#     plt.tight_layout()
#     plt.savefig(f'patch_test_case_{strain_case}.png')
#     plt.show()
    
#     return passed

# # Run all four patch tests
# print("\n\n========== AXISYMMETRIC VEM PATCH TESTS ==========")

# all_tests_passed = True
# for test_case in range(1, 5):
#     test_result = run_patch_test(test_case)
#     all_tests_passed = all_tests_passed and test_result

# print("\n\n========== PATCH TEST SUMMARY ==========")
# print(f"Overall result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
# print("\nNotes on Axisymmetric VEM Implementation:")
# print("1. This implementation uses proper numerical integration with Gauss quadrature")
# print("2. The hoop strain εθ = ur/r is properly handled at integration points")
# print("3. The stabilization term uses VEM projection principles")
# print("4. The axisymmetric weight 'r' is included in the stiffness integration")
# print("5. Strain calculation is performed with shape function interpolation")

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

# Material properties
E = 1.0       # Young's modulus
nu = 0.3      # Poisson's ratio

# Calculate constitutive matrix for axisymmetric elasticity
factor = E / ((1 + nu) * (1 - 2*nu))
C = factor * np.array([
    [1-nu, nu, nu, 0],
    [nu, 1-nu, nu, 0],
    [nu, nu, 1-nu, 0],
    [0, 0, 0, (1-2*nu)/2]
])

print("Constitutive matrix C:")
print(C)

# Domain setup
r_inner = 1.0
r_outer = 3.0
z_min = 0.0
z_max = 2.0

# Create a triangular mesh
# First create a grid of points
n_r = 4  # Number of divisions in r direction
n_z = 4  # Number of divisions in z direction

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

# Create triangular elements (two triangles per grid cell)
n_cell_triangles = 2  # Number of triangles per cell
total_triangles = n_r * n_z * n_cell_triangles
triangles = np.zeros((total_triangles, 3), dtype=int)

triangle_idx = 0
for iz in range(n_z):
    for ir in range(n_r):
        # Indices of the four corners of the cell
        n1 = iz * nr_nodes + ir
        n2 = iz * nr_nodes + (ir + 1)
        n3 = (iz + 1) * nr_nodes + (ir + 1)
        n4 = (iz + 1) * nr_nodes + ir
        
        # Create two triangles
        triangles[triangle_idx] = [n1, n2, n4]  # Lower triangle
        triangle_idx += 1
        triangles[triangle_idx] = [n2, n3, n4]  # Upper triangle
        triangle_idx += 1

# Copy grid nodes to be our final nodes
nodes = grid_nodes.copy()
elements = triangles.copy()

total_nodes = len(nodes)
total_elements = len(elements)

print(f"\nTotal nodes: {total_nodes}")
print(f"Total elements: {total_elements}")
print("First triangle nodes:", elements[0])

# Total degrees of freedom
ndof = 2 * total_nodes  # 2 DOFs per node (r, z displacements)
print(f"Total DOFs: {ndof}")

# Define Gauss quadrature points for triangular elements
def get_triangle_gauss_points(n_points=3):
    """Get Gauss quadrature points and weights for a triangle in area coordinates"""
    if n_points == 1:
        # 1-point quadrature
        points = np.array([[1/3, 1/3, 1/3]])
        weights = np.array([1.0])
    elif n_points == 3:
        # 3-point quadrature
        points = np.array([
            [1/6, 1/6, 2/3],
            [1/6, 2/3, 1/6],
            [2/3, 1/6, 1/6]
        ])
        weights = np.array([1/3, 1/3, 1/3])
    else:
        raise ValueError("Only 1-point or 3-point quadrature implemented")
    
    # Convert to (L1, L2) coordinates (L3 = 1 - L1 - L2)
    L1L2_points = points[:, :2]
    
    return L1L2_points, weights

# Shape functions for linear triangular elements
def triangle_shape_functions(L1, L2):
    """Evaluate shape functions at area coordinates (L1, L2)
    L3 = 1 - L1 - L2 is calculated automatically"""
    L3 = 1 - L1 - L2
    N = np.array([L1, L2, L3])
    return N

def triangle_shape_function_derivatives():
    """
    Derivatives of shape functions with respect to area coordinates
    For a linear triangle, these are constants
    """
    # dN/dL1
    dN_dL1 = np.array([1, 0, -1])
    # dN/dL2
    dN_dL2 = np.array([0, 1, -1])
    
    return dN_dL1, dN_dL2

# Improved element stiffness calculation with proper numerical integration for triangles
def calculate_element_stiffness(node_coords):
    """Calculate element stiffness matrix with proper numerical integration for triangles"""
    # Gauss quadrature setup
    gauss_points, gauss_weights = get_triangle_gauss_points(3)
    
    # Initialize element matrices
    K_elem = np.zeros((6, 6))  # 3 nodes, 2 DOFs per node
    
    # Get area of the triangle
    r1, z1 = node_coords[0]
    r2, z2 = node_coords[1]
    r3, z3 = node_coords[2]
    
    # Compute area using cross product
    area = 0.5 * abs((r2-r1)*(z3-z1) - (r3-r1)*(z2-z1))
    
    # Compute derivatives of shape functions with respect to global coordinates
    # For a linear triangle, these are constants
    
    # Jacobian matrix components
    J11 = r1 * (z2 - z3) + r2 * (z3 - z1) + r3 * (z1 - z2)  # dr/dL1
    J12 = r1 * (z3 - z2) + r2 * (z1 - z3) + r3 * (z2 - z1)  # dr/dL2
    J21 = z1 * (r3 - r2) + z2 * (r1 - r3) + z3 * (r2 - r1)  # dz/dL1
    J22 = z1 * (r2 - r3) + z2 * (r3 - r1) + z3 * (r1 - r2)  # dz/dL2
    
    # Jacobian determinant (2 * area)
    detJ = 2 * area
    
    # Inverse of Jacobian (divided by determinant)
    dL1_dr = J22 / detJ
    dL1_dz = -J12 / detJ
    dL2_dr = -J21 / detJ
    dL2_dz = J11 / detJ
    
    # Shape function derivatives with respect to area coordinates
    dN_dL1, dN_dL2 = triangle_shape_function_derivatives()
    
    # Shape function derivatives with respect to global coordinates
    dN_dr = dN_dL1 * dL1_dr + dN_dL2 * dL2_dr
    dN_dz = dN_dL1 * dL1_dz + dN_dL2 * dL2_dz
    
    # Numerical integration loop
    for i, (L1, L2) in enumerate(gauss_points):
        # Shape functions at current Gauss point
        N = triangle_shape_functions(L1, L2)
        
        # Compute r at the current Gauss point (for axisymmetric weight)
        r_gauss = np.sum(N * node_coords[:, 0])
        
        # B matrix for axisymmetric elasticity
        B = np.zeros((4, 6))
        
        # Loop over nodes to assemble B matrix
        for n in range(3):
            # Columns for radial displacement (ur)
            B[0, 2*n] = dN_dr[n]               # εr = ∂ur/∂r
            B[2, 2*n] = N[n] / r_gauss         # εθ = ur/r
            B[3, 2*n] = dN_dz[n]               # γrz = ∂ur/∂z + ∂uz/∂r (part 1)
            
            # Columns for axial displacement (uz)
            B[1, 2*n+1] = dN_dz[n]             # εz = ∂uz/∂z
            B[3, 2*n+1] = dN_dr[n]             # γrz = ∂ur/∂z + ∂uz/∂r (part 2)
        
        # Contribution to stiffness matrix: B^T * C * B * r * detJ * weight
        K_gauss = np.matmul(np.matmul(B.T, C), B) * r_gauss * area * gauss_weights[i]
        K_elem += K_gauss
    
    # Add stabilization term - improved version for triangular elements
    # Extract the part of stiffness related to constant strains
    P0 = np.zeros((6, 4))
    for n in range(3):
        # Constant radial strain mode
        P0[2*n, 0] = node_coords[n, 0]  # ur = r
        # Constant axial strain mode
        P0[2*n+1, 1] = node_coords[n, 1]  # uz = z
        # Constant hoop strain mode
        P0[2*n, 2] = node_coords[n, 0]  # ur = r
        # Constant shear strain mode
        P0[2*n, 3] = node_coords[n, 1] / 2  # ur = z/2
        P0[2*n+1, 3] = node_coords[n, 0] / 2  # uz = r/2
    
    # Orthogonalize P0
    Q, R = np.linalg.qr(P0)
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    Pc = Q[:, :rank]
    
    # Projection matrix onto the orthogonal complement of Pc
    I = np.eye(6)
    Pi = I - np.matmul(Pc, Pc.T)
    
    # Calculate trace for scaling
    alpha = np.trace(K_elem) / np.trace(Pi)
    
    # Stabilization term
    K_stab = alpha * Pi
    
    # Complete element stiffness matrix
    K_elem_final = K_elem + K_stab
    
    return K_elem_final

# Assemble global stiffness matrix
K_global = lil_matrix((ndof, ndof))

for e in range(total_elements):
    node_indices = elements[e]
    node_coords = nodes[node_indices]
    
    K_elem = calculate_element_stiffness(node_coords)
    
    # Map local DOFs to global DOFs
    dof_indices = np.zeros(6, dtype=int)
    for i in range(3):
        dof_indices[2*i] = 2 * node_indices[i]     # r-displacement
        dof_indices[2*i+1] = 2 * node_indices[i] + 1  # z-displacement
    
    # Assemble element matrix into global matrix
    for i in range(6):
        for j in range(6):
            K_global[dof_indices[i], dof_indices[j]] += K_elem[i, j]

# Find boundary nodes
boundary_nodes = []
for i in range(total_nodes):
    r, z = nodes[i]
    if (abs(r - r_inner) < 1e-6 or abs(r - r_outer) < 1e-6 or 
        abs(z - z_min) < 1e-6 or abs(z - z_max) < 1e-6):
        boundary_nodes.append(i)

print(f"\nNumber of boundary nodes: {len(boundary_nodes)}")

# Define function to calculate strain at a point in a triangle
def calculate_strain_at_point(elem_idx, L1, L2, u_elem):
    """Calculate strain at a point in a triangle with area coordinates (L1, L2)"""
    node_indices = elements[elem_idx]
    node_coords = nodes[node_indices]
    
    # Shape functions
    N = triangle_shape_functions(L1, L2)
    
    # Shape function derivatives
    r1, z1 = node_coords[0]
    r2, z2 = node_coords[1]
    r3, z3 = node_coords[2]
    
    # Area of the triangle
    area = 0.5 * abs((r2-r1)*(z3-z1) - (r3-r1)*(z2-z1))
    
    # Jacobian matrix components
    J11 = r1 * (z2 - z3) + r2 * (z3 - z1) + r3 * (z1 - z2)
    J12 = r1 * (z3 - z2) + r2 * (z1 - z3) + r3 * (z2 - z1)
    J21 = z1 * (r3 - r2) + z2 * (r1 - r3) + z3 * (r2 - r1)
    J22 = z1 * (r2 - r3) + z2 * (r3 - r1) + z3 * (r1 - r2)
    
    # Jacobian determinant
    detJ = 2 * area
    
    # Inverse of Jacobian
    dL1_dr = J22 / detJ
    dL1_dz = -J12 / detJ
    dL2_dr = -J21 / detJ
    dL2_dz = J11 / detJ
    
    # Shape function derivatives with respect to area coordinates
    dN_dL1, dN_dL2 = triangle_shape_function_derivatives()
    
    # Shape function derivatives with respect to global coordinates
    dN_dr = dN_dL1 * dL1_dr + dN_dL2 * dL2_dr
    dN_dz = dN_dL1 * dL1_dz + dN_dL2 * dL2_dz
    
    # Compute r at the given point
    r_point = np.sum(N * node_coords[:, 0])
    
    # Calculate displacements at this point
    u_r_point = np.sum(N * u_elem[0::2])
    u_z_point = np.sum(N * u_elem[1::2])
    
    # Calculate spatial derivatives
    du_r_dr = np.sum(dN_dr * u_elem[0::2])
    du_r_dz = np.sum(dN_dz * u_elem[0::2])
    du_z_dr = np.sum(dN_dr * u_elem[1::2])
    du_z_dz = np.sum(dN_dz * u_elem[1::2])
    
    # Calculate strains
    strain = np.zeros(4)
    strain[0] = du_r_dr                  # εr = ∂ur/∂r
    strain[1] = du_z_dz                  # εz = ∂uz/∂z
    strain[2] = u_r_point / r_point      # εθ = ur/r
    strain[3] = du_r_dz + du_z_dr        # γrz = ∂ur/∂z + ∂uz/∂r
    
    return strain

def run_patch_test(strain_case):
    """
    Run patch test for the given strain state:
    1 = Constant radial strain
    2 = Constant axial strain
    3 = Constant hoop strain
    4 = Constant shear strain
    """
    # Force vector
    F = np.zeros(ndof)
    
    # Define strain state based on test case
    strain_r = 0.0
    strain_z = 0.0
    strain_theta = 0.0
    strain_rz = 0.0
    
    if strain_case == 1:
        strain_r = 0.01      # Radial strain test
        test_name = "Radial Strain"
    elif strain_case == 2:
        strain_z = 0.01      # Axial strain test
        test_name = "Axial Strain"
    elif strain_case == 3:
        strain_theta = 0.01  # Hoop strain test
        test_name = "Hoop Strain"
    elif strain_case == 4:
        strain_rz = 0.01     # Shear strain test
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
    K_csr = K_global.tocsr()
    F_mod = F - K_csr.dot(u_exact)
    F_reduced = F_mod[free_dofs]
    K_reduced = K_csr[free_dofs, :][:, free_dofs]
    
    u_reduced = spsolve(K_reduced, F_reduced)
    u = u_exact.copy()
    u[free_dofs] = u_reduced
    
    # Extract displacements
    u_r = u[0::2]
    u_z = u[1::2]
    
    # Calculate strains at the centroid of each triangle
    strains = np.zeros((total_elements, 4))  # [εr, εz, εθ, γrz]
    
    for e in range(total_elements):
        node_indices = elements[e]
        
        # Get element displacements
        elem_disps = np.zeros(6)
        for i in range(3):
            elem_disps[2*i] = u[2*node_indices[i]]       # ur
            elem_disps[2*i+1] = u[2*node_indices[i]+1]   # uz
        
        # Evaluate strains at triangle centroid (L1=L2=L3=1/3)
        strains[e] = calculate_strain_at_point(e, 1/3, 1/3, elem_disps)
    
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
    tolerance = 1e-5
    passed = np.all(strain_errors < tolerance)
    
    print(f"\nPatch test for {test_name}: {'PASSED' if passed else 'FAILED'} with tolerance {tolerance}")
    
    # Plotting
    plt.figure(figsize=(15, 12))
    
    # Plot mesh
    plt.subplot(2, 2, 1)
    for e in range(total_elements):
        node_indices = elements[e]
        r_coords = nodes[node_indices, 0]
        z_coords = nodes[node_indices, 1]
        # Close the loop for plotting
        r_coords = np.append(r_coords, r_coords[0])
        z_coords = np.append(z_coords, z_coords[0])
        plt.plot(r_coords, z_coords, 'k-', linewidth=0.5)
    plt.scatter(nodes[:, 0], nodes[:, 1], c='b', s=20)
    plt.title('Triangular Mesh')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.axis('equal')
    
    # Plot displacement
    plt.subplot(2, 2, 2)
    # Create triangulation for plotting
    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elements)
    
    if strain_case in [1, 3]:  # Radial or hoop strain
        plot_data = u_r
        title = 'Radial displacement (ur)'
    elif strain_case == 2:  # Axial strain
        plot_data = u_z
        title = 'Axial displacement (uz)'
    else:  # Shear strain
        # For shear, plot magnitude
        plot_data = np.sqrt(u_r**2 + u_z**2)
        title = 'Displacement magnitude'
        
    plt.tricontourf(triang, plot_data, cmap='viridis')
    plt.colorbar(label=title)
    plt.title(title)
    plt.xlabel('r')
    plt.ylabel('z')
    plt.axis('equal')
    
    # Plot the specific strain component being tested
    plt.subplot(2, 2, 3)
    # Element centroids for plotting strain
    centroids = np.zeros((total_elements, 2))
    for e in range(total_elements):
        node_indices = elements[e]
        centroids[e] = np.mean(nodes[node_indices], axis=0)
    
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
    
    # Plot strain error
    plt.subplot(2, 2, 4)
    strain_comp_errors = np.abs(strains[:, strain_idx] - expected)
    sc = plt.scatter(centroids[:, 0], centroids[:, 1], c=strain_comp_errors, cmap='jet', s=50)
    plt.colorbar(sc, label=f'{strain_name} error')
    plt.title(f'{strain_name} error')
    plt.xlabel('r')
    plt.ylabel('z')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'patch_test_case_{strain_case}_triangular.png')
    plt.show()
    
    return passed

# Run all four patch tests
print("\n\n========== AXISYMMETRIC VEM PATCH TESTS (TRIANGULAR ELEMENTS) ==========")

all_tests_passed = True
for test_case in range(1, 5):
    test_result = run_patch_test(test_case)
    all_tests_passed = all_tests_passed and test_result

print("\n\n========== PATCH TEST SUMMARY ==========")
print(f"Overall result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")
print("\nNotes on Axisymmetric VEM Implementation (Triangular Elements):")
print("1. This implementation uses linear triangular elements instead of quadrilaterals")
print("2. Uses area coordinates and appropriate shape functions for triangles")
print("3. Implements proper numerical integration with Gauss quadrature for triangles")
print("4. The hoop strain εθ = ur/r is properly handled at integration points")
print("5. The stabilization term is adapted for triangular elements")