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

def run_volumetric_locking_patch_test(nu_values=None, show_plots=True):
    """
    Run a patch test to demonstrate volumetric locking and its remedy.
    
    This test applies a uniform pressure to a cylindrical tube and measures
    the radial displacement. For high Poisson's ratio, the standard VEM
    formulation should exhibit locking (artificially stiff response),
    while the enhanced formulation should show the correct displacement.
    
    Parameters:
        nu_values (list, optional): List of Poisson's ratio values to test.
                                    Defaults to [0.3, 0.499].
        show_plots (bool, optional): Whether to display plots. Defaults to True.
        
    Returns:
        tuple: (displacement_standard, displacement_enhanced) dictionaries
              mapping Poisson's ratio to displacement values
    """
    import matplotlib.pyplot as plt
    
    if nu_values is None:
        nu_values = [0.3, 0.499]  # Compressible and nearly incompressible
    
    # Material and geometry properties
    E = 1000.0  # Young's modulus
    r_inner = 1.0
    r_outer = 2.0
    height = 1.0
    pressure = 10.0  # Applied internal pressure
    
    # Results storage
    displacement_standard = {}
    displacement_enhanced = {}
    
    # Create mesh - a simple 4-element mesh of the tube
    # We'll use 4 rectangular elements (which become 8 triangles)
    n_r = 2  # Elements in radial direction
    n_h = 2  # Elements in axial direction
    
    # Generate a simple structured grid
    vertices = []
    elements = []
    
    # Generate vertices
    for i_h in range(n_h + 1):
        z = i_h * height / n_h
        for i_r in range(n_r + 1):
            r = r_inner + i_r * (r_outer - r_inner) / n_r
            vertices.append([r, z])
    
    # Convert to numpy array
    vertices = np.array(vertices)
    n_vertices = len(vertices)
    
    # Generate quadrilateral elements
    for i_h in range(n_h):
        for i_r in range(n_r):
            # Indices of the four corners
            v1 = i_h * (n_r + 1) + i_r
            v2 = i_h * (n_r + 1) + (i_r + 1)
            v3 = (i_h + 1) * (n_r + 1) + (i_r + 1)
            v4 = (i_h + 1) * (n_r + 1) + i_r
            
            # Add quadrilateral element
            elements.append([v1, v2, v3, v4])
    
    # Convert to numpy array
    elements = np.array(elements)
    n_elements = len(elements)
    
    # Function to assemble the global stiffness matrix and solve
    def solve_problem(nu, use_enhanced_stabilization):
        # Initialize global stiffness matrix and force vector
        n_dofs = 2 * n_vertices
        K_global = np.zeros((n_dofs, n_dofs))
        F_global = np.zeros(n_dofs)
        
        # Assemble element contributions
        for elem_idx, elem_verts_idx in enumerate(elements):
            # Extract element vertices
            elem_verts = vertices[elem_verts_idx]
            
            # Compute element stiffness matrix
            #print("Stabilization type: ", use_enhanced_stabilization)
            K_elem, _, _ = compute_stiffness_matrix(
                elem_verts, E, nu, stab_type = use_enhanced_stabilization)
            
            # Apply pressure load on inner elements
            if np.any(np.isclose(elem_verts[:, 0], r_inner)):
                # Identify inner edge
                for i in range(len(elem_verts)):
                    j = (i + 1) % len(elem_verts)
                    if np.isclose(elem_verts[i, 0], r_inner) and np.isclose(elem_verts[j, 0], r_inner):
                        # Found inner edge
                        edge_verts = np.array([elem_verts[i], elem_verts[j]])
                        
                        # Apply traction (inward pressure)
                        def pressure_traction(r, z):
                            return -pressure, 0.0  # Negative for inward pressure
                        
                        # Compute load vector for this edge
                        f_elem = compute_element_load_boundary_traction(
                            elem_verts, [i], pressure_traction)
                        
                        # Assemble into global force vector
                        for k, vertex_idx in enumerate(elem_verts_idx):
                            F_global[2*vertex_idx] += f_elem[2*k]      # r component
                            F_global[2*vertex_idx+1] += f_elem[2*k+1]  # z component
            
            # Assemble into global stiffness matrix
            for i, vi in enumerate(elem_verts_idx):
                for j, vj in enumerate(elem_verts_idx):
                    K_global[2*vi, 2*vj] += K_elem[2*i, 2*j]          # r-r
                    K_global[2*vi, 2*vj+1] += K_elem[2*i, 2*j+1]      # r-z
                    K_global[2*vi+1, 2*vj] += K_elem[2*i+1, 2*j]      # z-r
                    K_global[2*vi+1, 2*vj+1] += K_elem[2*i+1, 2*j+1]  # z-z
        
        # Apply boundary conditions
        # Fix bottom in z direction
        for i, v in enumerate(vertices):
            if np.isclose(v[1], 0.0):  # Bottom nodes
                # Zero z displacement
                K_global[2*i+1, :] = 0
                K_global[:, 2*i+1] = 0
                K_global[2*i+1, 2*i+1] = 1.0
                F_global[2*i+1] = 0.0
        
        # Fix one corner in r direction to prevent rigid body rotation
        K_global[0, :] = 0
        K_global[:, 0] = 0
        K_global[0, 0] = 1.0
        F_global[0] = 0.0
        
        # Solve the system
        U = np.linalg.solve(K_global, F_global)
        
        # Extract radial displacements at inner radius
        inner_displacements = []
        for i, v in enumerate(vertices):
            if np.isclose(v[0], r_inner):
                inner_displacements.append(U[2*i])
        
        # Return average radial displacement at inner radius
        return np.mean(inner_displacements)
    
    # Run tests for different Poisson's ratios
    for nu in nu_values:
        print(f"\nRunning test with Poisson's ratio ν = {nu}")
        
        # Standard VEM
        stab_type = "standard"
        u_standard = solve_problem(nu, stab_type)
        displacement_standard[nu] = u_standard
        print(f"  Standard VEM - Avg. radial displacement: {u_standard:.8f}")
        
        # Enhanced VEM with divergence stabilization
        stab_type = "divergence"
        u_enhanced = solve_problem(nu, stab_type)
        displacement_enhanced[nu] = u_enhanced
        print(f"  Enhanced VEM - Avg. radial displacement: {u_enhanced:.8f}")
        
        # Compute the analytical solution for a thick-walled cylinder under pressure
        # u_r(r) = P r_i^2/(E(r_o^2-r_i^2)) * [(1-ν)r + (1+ν)r_o^2/r]
        # Evaluate at r = r_inner
        analytical = pressure * r_inner**2 / (E * (r_outer**2 - r_inner**2)) * \
                    ((1 - nu) * r_inner + (1 + nu) * r_outer**2 / r_inner)
        
        print(f"  Analytical solution: {analytical:.8f}")
        print(f"  Standard VEM error: {abs(u_standard - analytical)/analytical*100:.2f}%")
        print(f"  Enhanced VEM error: {abs(u_enhanced - analytical)/analytical*100:.2f}%")
    
    # Plot the results
    if show_plots and len(nu_values) > 1:
        plt.figure(figsize=(10, 6))
        
        # Convert data for plotting
        nu_list = list(displacement_standard.keys())
        disp_standard = list(displacement_standard.values())
        disp_enhanced = list(displacement_enhanced.values())
        
        # Compute analytical solutions
        analytical_values = []
        for nu in nu_list:
            analytical = pressure * r_inner**2 / (E * (r_outer**2 - r_inner**2)) * \
                        ((1 - nu) * r_inner + (1 + nu) * r_outer**2 / r_inner)
            analytical_values.append(analytical)
        
        # Plot
        plt.plot(nu_list, analytical_values, 'k-', linewidth=2, label='Analytical')
        plt.plot(nu_list, disp_standard, 'ro-', label='Standard VEM')
        plt.plot(nu_list, disp_enhanced, 'bs-', label='Enhanced VEM')
        
        plt.title('Radial Displacement vs. Poisson\'s Ratio')
        plt.xlabel('Poisson\'s Ratio (ν)')
        plt.ylabel('Average Radial Displacement')
        plt.grid(True)
        plt.legend()
        
        # Plot error on a second axis
        ax2 = plt.twinx()
        error_standard = [abs(d - a)/a*100 for d, a in zip(disp_standard, analytical_values)]
        error_enhanced = [abs(d - a)/a*100 for d, a in zip(disp_enhanced, analytical_values)]
        
        ax2.plot(nu_list, error_standard, 'r--', alpha=0.7)
        ax2.plot(nu_list, error_enhanced, 'b--', alpha=0.7)
        ax2.set_ylabel('Error (%)')
        
        plt.tight_layout()
        plt.show()
    
    return displacement_standard, displacement_enhanced

def test_volumetric_locking():
    """
    Simple test to demonstrate volumetric locking with a direct comparison between
    standard and enhanced VEM for various Poisson's ratios.
    
    This test applies a uniform expansion (volumetric strain) to a simple square element
    and compares the energy required between standard and enhanced VEM formulations.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Material properties
    E = 1000.0
    
    # Square element centered at origin
    square_vertices = np.array([
        [1.0, 0.0],  # Vertex 1
        [2.0, 0.0],  # Vertex 2
        [2.0, 1.0],  # Vertex 3
        [1.0, 1.0]   # Vertex 4
    ])
    
    # Range of Poisson's ratios to test
    nu_values = np.linspace(0.3, 0.499, 10)
    
    # Storage for energy values
    energy_standard = []
    energy_enhanced = []
    
    # Prescribed uniform expansion (volumetric strain)
    # Apply unit radial displacement at all nodes
    u_expansion = np.zeros(8)
    for i in range(4):
        u_expansion[2*i] = 0.1  # Radial displacement = 0.1 at all nodes
    
    print("Uniform Expansion Test")
    print("=====================")
    print("Applying uniform radial displacement of 0.1 to all nodes")
    print("Computing strain energy for various Poisson's ratios")
    
    for nu in nu_values:
        print(f"\nTesting Poisson's ratio ν = {nu:.4f}")
        
        # Compute stiffness matrices
        stab_type = "standard"
        K_standard, _, _ = compute_stiffness_matrix(square_vertices, E, nu, stab_type)
        stab_type = "divergence"
        K_enhanced, _, _ = compute_stiffness_matrix(square_vertices, E, nu, stab_type)
        
        # Calculate strain energy: U = 0.5 * u^T * K * u
        energy_std = 0.5 * np.dot(u_expansion, np.dot(K_standard, u_expansion))
        energy_enh = 0.5 * np.dot(u_expansion, np.dot(K_enhanced, u_expansion))
        
        energy_standard.append(energy_std)
        energy_enhanced.append(energy_enh)
        
        print(f"  Standard VEM - Strain Energy: {energy_std:.6f}")
        print(f"  Enhanced VEM - Strain Energy: {energy_enh:.6f}")
        print(f"  Ratio (Standard/Enhanced): {energy_std/energy_enh:.6f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(nu_values, energy_standard, 'ro-', label='Standard VEM')
    plt.plot(nu_values, energy_enhanced, 'bs-', label='Enhanced VEM')
    plt.title('Strain Energy vs. Poisson\'s Ratio for Uniform Expansion')
    plt.xlabel('Poisson\'s Ratio (ν)')
    plt.ylabel('Strain Energy (0.5 * u^T * K * u)')
    plt.grid(True)
    plt.legend()
    
    # Log scale might be needed to see differences
    plt.figure(figsize=(10, 6))
    plt.semilogy(nu_values, energy_standard, 'ro-', label='Standard VEM')
    plt.semilogy(nu_values, energy_enhanced, 'bs-', label='Enhanced VEM')
    plt.title('Strain Energy vs. Poisson\'s Ratio (Log Scale)')
    plt.xlabel('Poisson\'s Ratio (ν)')
    plt.ylabel('Strain Energy (log scale)')
    plt.grid(True)
    plt.legend()
    
    plt.show()



# Run the patch test
if __name__ == "__main__":
    # nu_values = [0.3, 0.4, 0.45, 0.49, 0.499]
    # displacement_standard, displacement_enhanced = run_volumetric_locking_patch_test(nu_values)
    test_volumetric_locking()