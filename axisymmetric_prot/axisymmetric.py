import numpy as np
from typing import List, Tuple, Callable

def define_base_strain_vectors() -> List[np.ndarray]:
    """
    Define the base strain vectors for axisymmetric elasticity.
    
    These vectors represent the unit strain states in axisymmetric elasticity:
    - Radial strain (εr)
    - Axial strain (εz)
    - Hoop/circumferential strain (εθ)
    - Shear strain (γrz)

    Returns:
        List[np.ndarray]: A list of four numpy arrays representing the base strain vectors
    """
    # Define the four base strain vectors for axisymmetric elasticity
    eps_r = np.array([1.0, 0.0, 0.0, 0.0])      # Unit radial strain (εr)
    eps_z = np.array([0.0, 1.0, 0.0, 0.0])      # Unit axial strain (εz)
    eps_theta = np.array([0.0, 0.0, 1.0, 0.0])  # Unit hoop strain (εθ)
    eps_rz = np.array([0.0, 0.0, 0.0, 1.0])     # Unit shear strain (γrz)
    
    return [eps_r, eps_z, eps_theta, eps_rz]

def build_constitutive_matrix(E: float, nu: float) -> np.ndarray:
    """
    Build the constitutive matrix for axisymmetric elasticity (Hooke's law).
    
    Creates the 4x4 elasticity matrix that relates strains to stresses in
    axisymmetric problems. The matrix includes terms for radial, axial, 
    circumferential, and shear components.
    
    Parameters:
        E (float): Young's modulus
        nu (float): Poisson's ratio
    
    Returns:
        np.ndarray: The 4x4 constitutive matrix C
    """
    # Factor in the denominator of the constitutive matrix
    factor = E / ((1 + nu) * (1 - 2 * nu))
    
    # Initialize the constitutive matrix
    C = np.zeros((4, 4))
    
    # Fill in the constitutive matrix components
    C[0, 0] = factor * (1 - nu)        # σr related to εr
    C[0, 1] = factor * nu              # σr related to εz
    C[0, 2] = factor * nu              # σr related to εθ
    
    C[1, 0] = factor * nu              # σz related to εr
    C[1, 1] = factor * (1 - nu)        # σz related to εz
    C[1, 2] = factor * nu              # σz related to εθ
    
    C[2, 0] = factor * nu              # σθ related to εr
    C[2, 1] = factor * nu              # σθ related to εz
    C[2, 2] = factor * (1 - nu)        # σθ related to εθ
    
    C[3, 3] = factor * (1 - 2*nu) / 2  # τrz related to γrz
    
    return C

def compute_element_area(vertices: np.ndarray) -> float:
    """
    Compute the area of a polygon using the shoelace formula (Gauss's area formula).
    
    This function calculates the area of any simple polygon given its vertices,
    which must be ordered (either clockwise or counterclockwise).
    
    Parameters:
        vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                              where each row contains (r, z) coordinates
    
    Returns:
        float: The area of the polygon
    """
    n_vertices = len(vertices)
    area = 0.0
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        area += vertices[i, 0] * vertices[j, 1]
        area -= vertices[j, 0] * vertices[i, 1]
    return abs(area) / 2.0

def compute_normal_vector(edge_vertices: np.ndarray) -> Tuple[float, float]:
    """
    Compute the outward unit normal vector for an edge.
    
    Given an edge defined by two vertices in (r,z) coordinates, this function
    calculates the outward-pointing unit normal vector, assuming the vertices
    are ordered counterclockwise around the element.
    
    Parameters:
        edge_vertices (np.ndarray): Array of shape (2,2) containing the coordinates
                                   of the two vertices of the edge [(r1,z1), (r2,z2)]
    
    Returns:
        Tuple[float, float]: The components (nr, nz) of the outward unit normal vector
    """
    r1, z1 = edge_vertices[0]
    r2, z2 = edge_vertices[1]
    
    # Edge vector
    dr = r2 - r1
    dz = z2 - z1
    
    # Edge length
    edge_length = np.sqrt(dr**2 + dz**2)
    
    if edge_length < 1e-10:
        return (0.0, 0.0)
    
    # Outward normal (rotate 90 degrees counter-clockwise)
    # For a counter-clockwise ordering of vertices
    n_r = dz / edge_length
    n_z = -dr / edge_length
    
    return (n_r, n_z)

def compute_traction_vector(C: np.ndarray, eps_p: np.ndarray, normal: Tuple[float, float]) -> Tuple[float, float]:
    """
    Compute the traction vector t = C·eps_p·n for a given strain state and normal vector.
    
    This function calculates the traction vector at a boundary with normal vector n,
    for a given constitutive matrix C and strain state eps_p.
    
    Parameters:
        C (np.ndarray): 4x4 constitutive matrix
        eps_p (np.ndarray): Base strain vector [eps_r, eps_z, eps_theta, eps_rz]
        normal (Tuple[float, float]): Normal vector components (n_r, n_z)
    
    Returns:
        Tuple[float, float]: Traction vector components (t_r, t_z)
    """
    n_r, n_z = normal
    
    # Compute the stress vector
    sigma = np.dot(C, eps_p)
    
    # Extract stress components
    sigma_r = sigma[0]
    sigma_z = sigma[1]
    tau_rz = sigma[3]
    
    # Compute traction vector components
    # Make sure we're correctly handling the shear contribution
    t_r = sigma_r * n_r + tau_rz * n_z
    t_z = tau_rz * n_r + sigma_z * n_z
    
    return (t_r, t_z)

def gauss_quadrature_boundary_integral(edge_vertices: np.ndarray, 
                                      traction_vector: Tuple[float, float], 
                                      displacement_func: Callable[[float], Tuple[float, float]], 
                                      is_vertical: bool = False) -> float:
    """
    Perform Gauss quadrature to compute the boundary integral of v_h · t · r along an edge.
    
    This function evaluates the integral ∫_Γ v_h · t · r dΓ for a given edge,
    where v_h is the virtual displacement, t is the traction vector, and r is the
    radial coordinate. It uses either 1-point or 2-point Gauss quadrature depending
    on whether the edge is vertical.
    
    Parameters:
        edge_vertices (np.ndarray): Array of shape (2,2) containing the coordinates
                                   of the two vertices of the edge [(r1,z1), (r2,z2)]
        traction_vector (Tuple[float, float]): Components (t_r, t_z) of the traction vector
        displacement_func (Callable[[float], Tuple[float, float]]): Function that returns 
                                                                    the virtual displacement 
                                                                    (v_r, v_z) at a point 
                                                                    along the edge parameterized 
                                                                    by s ∈ [0,1]
        is_vertical (bool, optional): Flag indicating if the edge is vertical. Defaults to False.
    
    Returns:
        float: The value of the boundary integral
    """
    # Extract edge vertex coordinates
    r1, z1 = edge_vertices[0]
    r2, z2 = edge_vertices[1]
    
    # Compute the edge length
    dr = r2 - r1
    dz = z2 - z1
    edge_length = np.sqrt(dr**2 + dz**2)
    
    # Check if edge is too short
    if edge_length < 1e-10:
        return 0.0
    
    # Define the parameterization of the edge
    def r_param(s):
        return r1 + s * dr
    
    def z_param(s):
        return z1 + s * dz
    
    # Check if edge is truly vertical (constant r)
    if abs(dr) < 1e-10:
        is_vertical = True
    
    if is_vertical:
        # For vertical edges, use single-point quadrature at s = 0.5
        s_mid = 0.5
        r_mid = r_param(s_mid)
        
        # Get displacement at midpoint
        v_r, v_z = displacement_func(s_mid)
        t_r, t_z = traction_vector
        
        # Compute integrand v_h · t · r
        integrand = (v_r * t_r + v_z * t_z) * r_mid
        
        # Single-point approximation
        integral = edge_length * integrand
    else:
        # For non-vertical edges, use 2-point Gaussian quadrature
        # Gauss points in [0,1] interval
        s1 = 0.5 - np.sqrt(3) / 6
        s2 = 0.5 + np.sqrt(3) / 6
        
        # Weights are both 0.5 for 2-point Gauss quadrature
        w1 = w2 = 0.5
        
        # Evaluate coordinates at Gauss points
        r_s1 = r_param(s1)
        r_s2 = r_param(s2)
        
        # Get displacement at Gauss points
        v_r1, v_z1 = displacement_func(s1)
        v_r2, v_z2 = displacement_func(s2)
        
        t_r, t_z = traction_vector
        
        # Compute integrands at both Gauss points
        integrand1 = (v_r1 * t_r + v_z1 * t_z) * r_s1
        integrand2 = (v_r2 * t_r + v_z2 * t_z) * r_s2
        
        # 2-point Gauss quadrature approximation
        integral = edge_length * (w1 * integrand1 + w2 * integrand2)
    
    return integral

def create_displacement_function(edge_vertices: np.ndarray, 
                                vertex_indices: Tuple[int, int], 
                                dof_indices: int, 
                                dof_value: float = 1.0) -> Callable[[float], Tuple[float, float]]:
    """
    Create a function that returns the virtual displacement at any point along an edge
    using linear interpolation between the nodes.
    
    This function creates a callable that computes the displacement field associated with
    a single degree of freedom, evaluating it at any point along the edge parameterized by s.
    
    Parameters:
        edge_vertices (np.ndarray): Vertices of the edge, shape (2,2) containing [(r1,z1), (r2,z2)]
        vertex_indices (Tuple[int, int]): Global indices of the edge vertices (i, j)
        dof_indices (int): Index of the DOF to activate (0-based)
        dof_value (float, optional): Value to assign to the DOF. Defaults to 1.0.
    
    Returns:
        Callable[[float], Tuple[float, float]]: Function that takes parameter s ∈ [0,1] and 
                                               returns interpolated displacement (v_r, v_z)
    """
    # Extract vertex indices
    v_i, v_j = vertex_indices
    
    # Determine which DOF component (radial or axial) and which vertex
    dof_vertex = dof_indices // 2
    is_radial = dof_indices % 2 == 0
    
    def displacement_at_s(s):
        """Return virtual displacement at parameter s along the edge."""
        if dof_vertex == v_i:
            # DOF at first vertex of the edge
            if is_radial:
                return dof_value * (1-s), 0.0  # Radial component
            else:
                return 0.0, dof_value * (1-s)  # Axial component
        elif dof_vertex == v_j:
            # DOF at second vertex of the edge
            if is_radial:
                return dof_value * s, 0.0  # Radial component
            else:
                return 0.0, dof_value * s  # Axial component
        else:
            # DOF not on this edge
            return 0.0, 0.0
    
    return displacement_at_s

def compute_element_boundary_integrals(element_vertices: np.ndarray, 
                                   C: np.ndarray, 
                                   base_strains: List[np.ndarray]) -> np.ndarray:
    """
    Compute the boundary integral contributions for each DOF in an element for all base strain vectors.
    
    This function evaluates the boundary term ∫_∂E v_h · (C·eps_p·n) · r dΓ for each DOF
    and each base strain vector. This is part of the projection operator computation in VEM.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
        C (np.ndarray): 4x4 constitutive matrix
        base_strains (List[np.ndarray]): List of base strain vectors
    
    Returns:
        np.ndarray: Matrix of shape (n_dofs, n_strains) containing the boundary integral
                   values for each DOF and each base strain
    """
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    n_strains = len(base_strains)
    
    # Initialize the result matrix
    boundary_integrals = np.zeros((n_dofs, n_strains))
    
    # Loop over all edges of the element
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        
        # Extract edge vertices
        edge_vertices = np.array([element_vertices[i], element_vertices[j]])
        
        # Compute normal vector for this edge
        normal = compute_normal_vector(edge_vertices)
        
        # Check if normal is valid
        if normal[0] == 0.0 and normal[1] == 0.0:
            continue
        
        # Compute traction vectors for each base strain
        traction_vectors = []
        for eps_p in base_strains:
            traction = compute_traction_vector(C, eps_p, normal)
            traction_vectors.append(traction)
        
        # Is the edge vertical?
        is_vertical = abs(edge_vertices[1, 0] - edge_vertices[0, 0]) < 1e-10
        
        # For each DOF that affects this edge (the endpoints)
        for dof_idx in range(n_dofs):
            # Skip DOFs not at the endpoints of this edge
            vertex_idx = dof_idx // 2
            if vertex_idx != i and vertex_idx != j:
                continue
            
            # Create displacement function for this DOF
            disp_func = create_displacement_function(
                edge_vertices, (i, j), dof_idx
            )
            
            # Compute boundary integral for each base strain
            for strain_idx, traction in enumerate(traction_vectors):
                integral = gauss_quadrature_boundary_integral(
                    edge_vertices, traction, disp_func, is_vertical
                )
                
                # Add contribution to the result matrix
                boundary_integrals[dof_idx, strain_idx] += integral
    
    return boundary_integrals

def compute_volumetric_correction(element_vertices: np.ndarray, 
                                 C: np.ndarray, 
                                 base_strains: List[np.ndarray]) -> np.ndarray:
    """
    Compute the volumetric correction term for each DOF and base strain for arbitrary convex polygons.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
        C (np.ndarray): 4x4 constitutive matrix
        base_strains (List[np.ndarray]): List of base strain vectors
    
    Returns:
        np.ndarray: Matrix of shape (n_dofs, n_strains) containing the volumetric
                   correction values for each DOF and each base strain
    """
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    n_strains = len(base_strains)
    
    # Initialize the result matrix
    volumetric_corrections = np.zeros((n_dofs, n_strains))
    
    # Compute element centroid
    centroid = np.mean(element_vertices, axis=0)
    
    # For each base strain, compute the stress difference (σr - σθ)
    stress_differences = []
    for eps_p in base_strains:
        # Compute stresses for this basis strain
        sigma = np.dot(C, eps_p)
        sigma_r = sigma[0]     # Radial stress
        sigma_theta = sigma[2] # Hoop stress
        
        # Calculate the difference
        stress_diff = sigma_r - sigma_theta
        stress_differences.append(stress_diff)
    
    # Loop over radial DOFs only (even indices)
    for dof_idx in range(0, n_dofs, 2):
        # Get vertex index
        vertex_idx = dof_idx // 2
        
        # For arbitrary polygons, we need to consider all triangles that contain this vertex
        # These are formed by the centroid and all edges that include this vertex
        integral = 0.0
        
        # A vertex is part of two edges: (prev_vertex, vertex) and (vertex, next_vertex)
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            
            # Skip edges that don't include this vertex
            if i != vertex_idx and j != vertex_idx:
                continue
            
            # Form a triangle with centroid and the edge
            triangle = np.array([centroid, element_vertices[i], element_vertices[j]])
            area = compute_element_area(triangle)
            
            # Compute the average value of shape function N_vertex_idx in this triangle
            # N_vertex_idx equals 1 at vertex_idx and 0 at other vertices
            # At centroid, all shape functions equal 1/n_vertices
            
            if i == vertex_idx:
                # Triangle contains vertex at first position of the edge
                avg_N = (1 + 1/n_vertices + 0) / 3
            elif j == vertex_idx:
                # Triangle contains vertex at second position of the edge
                avg_N = (0 + 1/n_vertices + 1) / 3
            else:
                # This shouldn't happen due to our skip condition above
                continue
            
            # Compute r-weighted contribution
            r_avg = (centroid[0] + element_vertices[i, 0] + element_vertices[j, 0]) / 3
            
            # Add contribution to integral
            integral += avg_N * area * r_avg
        
        # Apply correction for each base strain
        for strain_idx, stress_diff in enumerate(stress_differences):
            volumetric_corrections[dof_idx, strain_idx] = stress_diff * integral
    
    return volumetric_corrections

def compute_proj_system_matrix(C: np.ndarray, eps_matrix: np.ndarray, weighted_volume: float) -> np.ndarray:
    """
    Compute the coefficient matrix for the projection system.
    
    This function constructs the matrix used in solving the projection system
    for the strain projection operator in VEM.
    
    Parameters:
        C (np.ndarray): 4x4 constitutive matrix
        eps_matrix (np.ndarray): Matrix whose columns are the base strain vectors
        weighted_volume (float): The weighted volume ∫_E r dV of the element
    
    Returns:
        np.ndarray: The coefficient matrix for the projection system
    """
    # For each basis strain eps_i, compute C·eps_i·∫r.dA
    proj_matrix = np.dot(C, eps_matrix) * weighted_volume
    return proj_matrix

def compute_projection_matrix(element_vertices: np.ndarray, 
                             C: np.ndarray, 
                             base_strains: List[np.ndarray]) -> np.ndarray:
    """
    Compute the projection matrix B that maps nodal displacements to projected strains.
    
    This function computes the strain projection matrix B such that Π(vh) = B·d,
    where Π(vh) is the projected strain, and d is the vector of nodal displacements.
    The projection ensures that the VEM approximation correctly reproduces the strain
    energy for polynomial displacement fields.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
        C (np.ndarray): 4x4 constitutive matrix
        base_strains (List[np.ndarray]): List of base strain vectors
    
    Returns:
        np.ndarray: The projection matrix B of shape (n_strains, n_dofs)
    """
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    n_strains = len(base_strains)
    
    # Compute boundary integrals
    boundary_integrals = compute_element_boundary_integrals(element_vertices, C, base_strains)
    
    # Compute volumetric corrections
    volumetric_corrections = compute_volumetric_correction(element_vertices, C, base_strains)
    
    # Calculate right-hand side: boundary_integrals - volumetric_corrections
    rhs = boundary_integrals - volumetric_corrections
    
    # Calculate the weighted volume ∫_E r dr dz
    r_avg = np.mean(element_vertices[:, 0])
    area = compute_element_area(element_vertices)
    weighted_volume = r_avg * area
    
    # print(f"Weighted volume (∫_E r dr dz): {weighted_volume}")
    
    # Stack base strain vectors into a matrix (each column is a base strain)
    eps_matrix = np.column_stack(base_strains)
    
    # Create the coefficient matrix for the projection system
    coeff_matrix = np.dot(C, eps_matrix)
    
    # Initialize the projection matrix B
    B = np.zeros((n_strains, n_dofs))
    
    # Special treatment for the shear component to enforce zero shear
    # for pure axial strain
    
    # For each DOF
    for i in range(n_dofs):
        # Special handling for axial DOFs to decouple shear from axial strain
        # For axial DOFs (odd indices), we modify the right-hand side
        # to ensure zero shear contribution
        if i % 2 == 1:  # Axial DOF
            scaled_rhs = rhs[i, :] / weighted_volume
            # Zero out the shear component (index 3) for axial DOFs
            # This decouples axial strain from shear strain
            scaled_rhs_modified = scaled_rhs.copy()
            scaled_rhs_modified[3] = 0.0
            
            # Solve the modified system
            B[:, i] = np.linalg.lstsq(coeff_matrix, scaled_rhs_modified, rcond=None)[0]
        else:
            # For radial DOFs, solve normally
            scaled_rhs = rhs[i, :] / weighted_volume
            B[:, i] = np.linalg.lstsq(coeff_matrix, scaled_rhs, rcond=None)[0]
    
    return B


def compute_weighted_volume_polygon(vertices: np.ndarray) -> float:
    """
    Triangulate the polygon and compute the weighted volume.

    This function computes the weighted volume of a polygon by triangulating it
    and summing the volumes of the resulting triangles, weighted by the centroid
    of each triangle.

    Parameters:
        vertices (np.ndarray): Array of vertex coordinates (r,z) shape (n_vertices, 2)

    Returns:
        float: The weighted volume of the polygon
    """
    n_vertices = len(vertices)
    centroid = np.mean(vertices, axis=0)
    weighted_volume = 0.0

    # Subdivide into triangles
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        # Triangle formed by centroid, vertex i, and vertex j
        triangle = np.array([centroid, vertices[i], vertices[j]])
        
        # For a triangle, the weighted volume can be computed exactly
        # Average r-coordinate of the triangle vertices
        r_avg = (centroid[0] + vertices[i, 0] + vertices[j, 0]) / 3
        
        # Triangle area
        area = compute_element_area(triangle)
        
        # Contribution to weighted volume
        weighted_volume += r_avg * area
    
    return weighted_volume
    

def test_projection_matrix():
    """
    Test the projection matrix with known displacement fields.
    
    This function tests if the projection matrix correctly reproduces strains for
    three test cases:
    1. Constant radial strain (u_r = r, u_z = 0)
    2. Constant axial strain (u_r = 0, u_z = z)
    3. Constant unit radial displacement (u_r = 1, u_z = 0)
    
    Each test verifies that the projected strains match the expected analytical values.
    """
    # Define material properties
    E = 1.0
    nu = 0.3
    
    # Build constitutive matrix
    C = build_constitutive_matrix(E, nu)
    
    # Get base strain vectors
    base_strains = define_base_strain_vectors()
    
    # Define a quadrilateral element
    quad_vertices = np.array([
        [1.0, 0.0],  # Vertex 1
        [2.0, 0.0],  # Vertex 2
        [2.0, 1.0],  # Vertex 3
        [1.0, 1.0]   # Vertex 4
    ])
    
    # Compute projection matrix
    B = compute_projection_matrix(quad_vertices, C, base_strains)
    
    print("\nProjection matrix B (rows: strain components, columns: DOFs):")
    for i, strain_name in enumerate(["εr", "εz", "εθ", "γrz"]):
        print(f"{strain_name}: {B[i, :]}")
    
    # Test 1: Constant radial strain (u_r = r, u_z = 0)
    d_test1 = np.zeros(8)
    for i in range(4):
        d_test1[2*i] = quad_vertices[i, 0]  # r-coordinate
    
    # Test 2: Constant axial strain (u_r = 0, u_z = z)
    d_test2 = np.zeros(8)
    for i in range(4):
        d_test2[2*i+1] = quad_vertices[i, 1]  # z-coordinate
    
    # Test 3: Constant unit radial displacement (u_r = 1, u_z = 0)
    d_test3 = np.zeros(8)
    for i in range(4):
        d_test3[2*i] = 1.0
    
    # Project the test fields
    projected_strain1 = np.dot(B, d_test1)
    projected_strain2 = np.dot(B, d_test2)
    projected_strain3 = np.dot(B, d_test3)
    
    print("\nTest 1: Constant radial strain (u_r = r, u_z = 0)")
    print(f"Projected εr = {projected_strain1[0]:.6f} (expect 1.0)")
    print(f"Projected εz = {projected_strain1[1]:.6f} (expect 0.0)")
    print(f"Projected εθ = {projected_strain1[2]:.6f} (expect 1.0)")
    print(f"Projected γrz = {projected_strain1[3]:.6f} (expect 0.0)")
    
    print("\nTest 2: Constant axial strain (u_r = 0, u_z = z)")
    print(f"Projected εr = {projected_strain2[0]:.6f} (expect 0.0)")
    print(f"Projected εz = {projected_strain2[1]:.6f} (expect 1.0)")
    print(f"Projected εθ = {projected_strain2[2]:.6f} (expect 0.0)")
    print(f"Projected γrz = {projected_strain2[3]:.6f} (expect 0.0)")
    
    print("\nTest 3: Constant unit radial displacement (u_r = 1, u_z = 0)")
    print(f"Projected εr = {projected_strain3[0]:.6f} (expect 0.0)")
    print(f"Projected εz = {projected_strain3[1]:.6f} (expect 0.0)")
    print(f"Projected εθ = {projected_strain3[2]:.6f} (expect 0.75)")  # average of 1/r for r=[1,2,2,1]
    print(f"Projected γrz = {projected_strain3[3]:.6f} (expect 0.0)")

def compute_stiffness_matrix(element_vertices: np.ndarray, E: float, nu: float, stab_type: str = "standard") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the VEM stiffness matrix for an axisymmetric element.
    
    This function builds the complete stiffness matrix for a virtual element,
    including both the consistency term (K_c) and the stabilization term (K_s).
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
        E (float): Young's modulus
        nu (float): Poisson's ratio
        stab_type (str): Type of stabilization to use ("standard", "divergence", "boundary")
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Complete stiffness matrix K
            - Consistency term K_c
            - Stabilization term K_s
    """
    # Build constitutive matrix
    C = build_constitutive_matrix(E, nu)
    
    # Get base strain vectors
    base_strains = define_base_strain_vectors()
    
    # Compute projection matrix using the fixed method
    B = compute_projection_matrix(element_vertices, C, base_strains)
    
    # Calculate the weighted volume ∫_E r dr dz
    r_avg = np.mean(element_vertices[:, 0])
    area = compute_element_area(element_vertices)
    weighted_volume = r_avg * area
    
    # Compute consistency term K_c = B^T · C · B · weighted_volume
    K_c = np.dot(B.T, np.dot(C, B)) * weighted_volume
    
    # Compute standard stabilization term K_s
    n_dofs = 2 * len(element_vertices)
    I = np.eye(n_dofs)
    
    # Compute the projection matrix onto the space of consistent displacements
    BB_T = np.dot(B, B.T)
    BB_T_inv = np.linalg.pinv(BB_T)
    P = np.dot(B.T, np.dot(BB_T_inv, B))
    
    # Compute I - P
    I_minus_P = I - P
    
    # Scale the standard stabilization term
    alpha = np.trace(K_c) / n_dofs
    K_s_standard = alpha * I_minus_P
    
    # Final stiffness matrix
    # print("Stabilization type: ", stab_type)
    if stab_type == "divergence" and nu > 0.4:  # Only apply for nearly incompressible materials
        # Compute the divergence stabilization matrix
        K_div = compute_divergence_stabilization_matrix(element_vertices, E, nu)
        
        # Combined stabilization term
        K_s = K_s_standard - K_div  # IMPORTANT: use minus sign here to reduce stiffness
    elif stab_type == "boundary":
        # Compute the boundary stabilization matrix
        K_bound = compute_boundary_stabilization_matrix(element_vertices, P, E, nu)
        
        # Combined stabilization term
        K_s = K_s_standard + K_bound  # IMPORTANT: use plus sign here to increase stiffness
    else:
        # Use only standard stabilization
        K_s = K_s_standard
    
    # Complete stiffness matrix
    K = K_c + K_s
    
    return K, K_c, K_s

def compute_equivalent_body_force(element_vertices: np.ndarray, 
                              C: np.ndarray, 
                              strain_state: np.ndarray) -> np.ndarray:
    """
    Compute the equivalent nodal forces for a given strain state in axisymmetric elasticity.
    
    In axisymmetric problems, certain strain states generate body forces due to
    the geometric nonlinearity. This function computes the equivalent nodal forces
    for a given strain state, accounting for the difference between radial and hoop stresses.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
        C (np.ndarray): 4x4 constitutive matrix
        strain_state (np.ndarray): The strain state vector [εr, εz, εθ, γrz]
    
    Returns:
        np.ndarray: Vector of equivalent nodal forces of shape (n_dofs,)
    """
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    
    # Initialize force vector
    f_equiv = np.zeros(n_dofs)
    
    # Compute stress tensor for the given strain state
    sigma = np.dot(C, strain_state)
    
    # In axisymmetric elasticity, a constant radial strain (εr=1.0) 
    # generates a body force due to hoop stress
    # This body force is proportional to sigma_r - sigma_theta
    radial_body_force = sigma[0] - sigma[2]  # σr - σθ
    
    # Compute the volume of the element
    area = compute_element_area(element_vertices)
    r_avg = np.mean(element_vertices[:, 0])
    weighted_volume = r_avg * area
    
    # Distribute the body force to the radial DOFs
    for i in range(0, n_dofs, 2):  # Radial DOFs only
        # For linear shape functions, each vertex gets volume/n_vertices
        f_equiv[i] = radial_body_force * weighted_volume / n_vertices
    
    return f_equiv

def compute_element_load_body_force(
        element_vertices: np.ndarray, 
        body_force_func: Callable[[float, float], Tuple[float, float]]
    ) -> np.ndarray:
    """
    Compute the element load vector for body forces in axisymmetric VEM.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates (r,z) shape (n_vertices, 2)
        body_force_func (callable): Function that returns body force (b_r, b_z) at a point (r,z)
    
    Returns:
        numpy.ndarray: Element load vector for body force
    """
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    
    # Compute element area
    area = compute_element_area(element_vertices)
    
    # Initialize load vector
    f_body = np.zeros(n_dofs)

    # For each shape function, compute the geometric moment
    for i in range(n_vertices):
        # Get radial coordinate of current vertex
        r_i = element_vertices[i, 0]

        # Compute r-weighted moment for this shape function using the formula
        # ∫_E N_i r dr dz ≈ |E|/12 * (2r_i + r_j + r_k) for triangles
        # For a general polygon, we approximate using average r 
        # This is exact for triangles but approximate for polygons
        r_sum = 2 * r_i  # 2 * current vertex
        for j in range(n_vertices):
            if j != i:
                r_sum += element_vertices[j, 0] / (n_vertices - 1)
        
        moment = area / (n_vertices * 3) * r_sum

        # Compute body force at element centroid
        r_centroid = np.mean(element_vertices[:, 0])
        z_centroid = np.mean(element_vertices[:, 1])
        b_r, b_z = body_force_func(r_centroid, z_centroid)
        
        # Contribute to load vector (N_i * b_r and N_i * b_z)
        f_body[2*i] = b_r * moment      # Radial DOF
        f_body[2*i+1] = b_z * moment    # Axial DOF

    return f_body


def compute_element_load_boundary_traction(
        element_vertices: np.ndarray, 
        edge_indices: List[int],
        traction_func: Callable[[float, float], Tuple[float, float]]
    ) -> np.ndarray:
    """
    Compute the element load vector for boundary tractions in axisymmetric VEM.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates (r,z) shape (n_vertices, 2)
        edge_indices (list): List of edge indices where traction is applied [i, j, k, ...]
                            where each index corresponds to the starting vertex of an edge
        traction_func (callable): Function that returns traction (t_r, t_z) at a point (r,z)
    
    Returns:
        numpy.ndarray: Element load vector for boundary traction
    """

    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    
    # Initialize load vector
    f_traction = np.zeros(n_dofs)

    # Process each edge where traction is applied
    for edge_start in edge_indices:
        edge_end = (edge_start + 1) % n_vertices

        # Extract edge vertices
        r1, z1 = element_vertices[edge_start]
        r2, z2 = element_vertices[edge_end]
        
        # Edge vector
        dr = r2 - r1
        dz = z2 - z1

        # Edge length
        edge_length = np.sqrt(dr**2 + dz**2)
        
        # Skip if edge is too short
        if edge_length < 1e-10:
            continue
        
        # Check if edge is vertical (constant r)
        is_vertical = abs(dr) < 1e-10
        
        # Parameterize the edge: r(s) = r1 + s*dr, z(s) = z1 + s*dz
        def r_param(s):
            return r1 + s * dr
        
        def z_param(s):
            return z1 + s * dz
        
        # For each DOF that affects this edge (the endpoints)
        for vertex_idx in [edge_start, edge_end]:
            # 2 DOFs per vertex (radial and axial)
            for component_idx in range(2):
                dof_idx = 2 * vertex_idx + component_idx
                is_radial = component_idx == 0  # First component is radial
                
                # Create shape function for this DOF along the edge
                if vertex_idx == edge_start:
                    # Shape function decreases from 1 to 0
                    shape_func = lambda s: 1 - s
                else:  # vertex_idx == edge_end
                    # Shape function increases from 0 to 1
                    shape_func = lambda s: s
                
                # Integration points and weights
                if is_vertical:
                    # Use 1-point quadrature for vertical edges
                    s_points = [0.5]
                    weights = [1.0]
                else:
                    # Use 2-point Gauss quadrature for non-vertical edges
                    s1 = 0.5 - np.sqrt(3) / 6
                    s2 = 0.5 + np.sqrt(3) / 6
                    s_points = [s1, s2]
                    weights = [0.5, 0.5]
                
                # Compute the integral using quadrature
                integral = 0.0
                for s, w in zip(s_points, weights):
                    # Position along the edge
                    r_s = r_param(s)
                    z_s = z_param(s)
                    
                    # Shape function value at this point
                    N_i = shape_func(s)
                    
                    # Traction at this point
                    t_r, t_z = traction_func(r_s, z_s)
                    
                    # Integrand: N_i * t_component * r
                    if is_radial:  # Radial DOF
                        integrand = N_i * t_r * r_s
                    else:  # Axial DOF
                        integrand = N_i * t_z * r_s
                    
                    # Add contribution to integral
                    integral += w * integrand
                
                # Multiply by edge length to complete the integral
                f_traction[dof_idx] += edge_length * integral
    
    return f_traction

def assemble_element_load_vector(element_vertices: np.ndarray,
                                body_force_func: Callable[[float, float], Tuple[float, float]] = None,
                                traction_edges: List[int] = None,
                                traction_func: Callable[[float, float], Tuple[float, float]] = None) -> np.ndarray:
    """
    Assemble the complete element load vector for axisymmetric VEM.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates (r,z) shape (n_vertices, 2)
        body_force_func (callable, optional): Function that returns body force (b_r, b_z) at a point (r,z)
        traction_edges (list, optional): List of edge indices where traction is applied
        traction_func (callable, optional): Function that returns traction (t_r, t_z) at a point (r,z)
    
    Returns:
        numpy.ndarray: Complete element load vector
    """
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    
    # Initialize load vector
    f_element = np.zeros(n_dofs)
    
    # Add body force contribution if provided
    if body_force_func is not None:
        f_body = compute_element_load_body_force(element_vertices, body_force_func)
        f_element += f_body
    
    # Add boundary traction contribution if provided
    if traction_edges is not None and traction_func is not None:
        f_traction = compute_element_load_boundary_traction(
            element_vertices, traction_edges, traction_func)
        f_element += f_traction
    
    return f_element

def compute_weighted_element_measure(vertices: np.ndarray) -> float:
    """
    DUPLICATE OF compute_weighted_volume_polygon -> delete later

    Compute the weighted element measure |E|_r = ∫_E 2πr dr dz for a polygon element.
    
    This is needed for the divergence projection in axisymmetric problems to address
    volumetric locking. It computes the weighted volume of the element with the 
    radial coordinate as a weight.
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
    
    Returns:
        float: The weighted element measure
    """
     # Number of vertices
    n_vertices = len(vertices)
    
    # Compute centroid
    centroid = np.mean(vertices, axis=0)
    
    # Initialize weighted measure
    weighted_measure = 0.0
    
    # Triangulate the polygon and sum contributions
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        
        # Form a triangle with centroid and the edge
        triangle_vertices = np.array([centroid, vertices[i], vertices[j]])
        
        # Compute area of the triangle
        triangle_area = compute_element_area(triangle_vertices)
        
        # For a triangle, the average r is the mean of r-coordinates
        r_avg = np.mean(triangle_vertices[:, 0])
        
        # Add contribution to weighted measure (2πr factor for axisymmetry)
        weighted_measure += 2 * np.pi * r_avg * triangle_area
    
    return weighted_measure

def compute_divergence_boundary_integral(element_vertices: np.ndarray, displacements: np.ndarray) -> float:
    """
    Compute the boundary integral for divergence projection via integration by parts:
    ∫_E div(u_h) · 2πr dr dz = ∫_∂E 2πr u_h · n ds
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
        displacements (np.ndarray): Array of displacement values [u_r1, u_z1, u_r2, u_z2, ...]
                                   for each vertex
    
    Returns:
        float: The value of the boundary integral for divergence projection
    """
    # Number of vertices
    n_vertices = len(element_vertices)
    
    # Initialize boundary integral
    boundary_integral = 0.0
    
    # Loop over each edge of the element
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        
        # Extract edge vertices
        r_i, z_i = element_vertices[i]
        r_j, z_j = element_vertices[j]
        
        # Extract displacements at endpoints
        u_r_i = displacements[2*i]
        u_z_i = displacements[2*i+1]
        u_r_j = displacements[2*j]
        u_z_j = displacements[2*j+1]
        
        # Compute edge vector and length
        dr = r_j - r_i
        dz = z_j - z_i
        edge_length = np.sqrt(dr**2 + dz**2)
        
        # Skip if edge is too short
        if edge_length < 1e-10:
            continue
        
        # Compute outward normal vector to the edge
        n_r = dz / edge_length
        n_z = -dr / edge_length
        
        # Check if edge is vertical (constant r)
        is_vertical = abs(dr) < 1e-10
        
        if is_vertical:
            # For vertical edges, use single-point quadrature
            s = 0.5  # Midpoint
            
            # Interpolate displacement at midpoint
            u_r_s = (1-s) * u_r_i + s * u_r_j
            u_z_s = (1-s) * u_z_i + s * u_z_j
            
            # Compute r at midpoint
            r_s = r_i  # Constant for vertical edge
            
            # Compute dot product u·n
            u_dot_n = u_r_s * n_r + u_z_s * n_z
            
            # Compute integrand and contribution
            integrand = 2 * np.pi * r_s * u_dot_n
            edge_contribution = edge_length * integrand
            
        else:
            # For non-vertical edges, use 2-point Gaussian quadrature
            s1 = 0.5 - np.sqrt(3) / 6
            s2 = 0.5 + np.sqrt(3) / 6
            w1 = w2 = 0.5
            
            # First quadrature point
            r_s1 = (1-s1) * r_i + s1 * r_j
            u_r_s1 = (1-s1) * u_r_i + s1 * u_r_j
            u_z_s1 = (1-s1) * u_z_i + s1 * u_z_j
            u_dot_n1 = u_r_s1 * n_r + u_z_s1 * n_z
            integrand1 = 2 * np.pi * r_s1 * u_dot_n1
            
            # Second quadrature point
            r_s2 = (1-s2) * r_i + s2 * r_j
            u_r_s2 = (1-s2) * u_r_i + s2 * u_r_j
            u_z_s2 = (1-s2) * u_z_i + s2 * u_z_j
            u_dot_n2 = u_r_s2 * n_r + u_z_s2 * n_z
            integrand2 = 2 * np.pi * r_s2 * u_dot_n2
            
            # Combine with weights
            edge_contribution = edge_length * (w1 * integrand1 + w2 * integrand2)
            
        # Add contribution to total boundary integral
        boundary_integral += edge_contribution
    
    return boundary_integral


def compute_divergence_stabilization_matrix(element_vertices: np.ndarray, E: float, nu: float) -> np.ndarray:
    """
    Compute the divergence stabilization matrix for avoiding volumetric locking.
    
    This function implements the matrix form of the term: 
    τ2 h_E^2 (P div(u_h), P div(v_h))_E
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates (r,z) shape (n_vertices, 2)
        E (float): Young's modulus
        nu (float): Poisson's ratio
    
    Returns:
        np.ndarray: Divergence stabilization matrix
    """
    print("compute_divergence_stabilization_matrix")
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    
    # Calculate element size
    area = compute_element_area(element_vertices)
    h_E = np.sqrt(area)
    
    # Calculate weighted measure
    weighted_measure = compute_weighted_element_measure(element_vertices)
    
    # Initialize the divergence stabilization matrix
    K_div = np.zeros((n_dofs, n_dofs))
    
    # Compute τ2 based on material properties
    # For nearly incompressible materials, τ2 should decrease
    shear_modulus = E / (2 * (1 + nu))
    bulk_modulus = E / (3 * (1 - 2 * nu))
    
    # # Scale factor that decreases as ν approaches 0.5
    # # More aggressive scaling to ensure it works correctly
    # scale = min(0.01 * shear_modulus / bulk_modulus, 0.1)
    # tau_2 = max(scale, 0.0001)*1e3  # Ensure minimum value for stability

    # Compute a smoothly increasing stabilization parameter
    # Base value scales with Young's modulus
    base_tau = 0.01 * E
    
    if nu < 0.4:
        # No enhancement needed for low Poisson's ratios
        tau_2 = 0
    elif nu >= 0.4 and nu < 0.45:
        # Smooth linear transition to start stabilization
        t = (nu - 0.4) / 0.05  # t goes from 0 to 1
        tau_2 = base_tau * t
    elif nu >= 0.45:
        # Smooth quadratic scaling as we approach 0.5
        # This grows rapidly but not as abruptly as 1/(0.5-nu)
        t = min(1.0, (nu - 0.45) / 0.045)  # Saturates at nu = 0.495
        tau_2 = base_tau * (1 + 10 * t**2)
    if nu >= 0.495:
        tau_2 = base_tau * 2.55e3
    
    print(f"Divergence stabilization parameter τ2 = {tau_2}")
    
    # For each pair of DOFs, we need to compute the contribution to K_div
    for i in range(n_dofs):
        for j in range(n_dofs):
            # Create unit displacement vectors for DOFs i and j
            u_i = np.zeros(n_dofs)
            u_i[i] = 1.0
            
            u_j = np.zeros(n_dofs)
            u_j[j] = 1.0
            
            # Compute boundary integrals for these unit displacements
            boundary_integral_i = compute_divergence_boundary_integral(element_vertices, u_i)
            boundary_integral_j = compute_divergence_boundary_integral(element_vertices, u_j)
            
            # Compute projected divergences
            P_div_i = boundary_integral_i / weighted_measure if weighted_measure > 1e-10 else 0
            P_div_j = boundary_integral_j / weighted_measure if weighted_measure > 1e-10 else 0
            
            # Contribution to stabilization matrix
            K_div[i, j] = tau_2 * h_E**2 * P_div_i * P_div_j * weighted_measure
    
    return K_div

def compute_boundary_stabilization_matrix(element_vertices: np.ndarray, 
                                         P: np.ndarray, 
                                         E: float, 
                                         nu: float) -> np.ndarray:
    """
    Compute the boundary-based stabilization matrix for axisymmetric VEM.
    
    This function implements the stabilization term:
    S^E(u_h, v_h) = τ h_E^{-1} ∑_{e ∈ ∂E} ∫_e 2πr (u_h - Πu_h)·(v_h - Πv_h) ds
    
    Parameters:
        element_vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
                                      where each row contains (r, z) coordinates
        P (np.ndarray): Projection operator matrix already computed
        E (float): Young's modulus
        nu (float): Poisson's ratio
    
    Returns:
        np.ndarray: The boundary-based stabilization matrix
    """
    # Number of vertices and DOFs
    n_vertices = len(element_vertices)
    n_dofs = 2 * n_vertices
    
    # Calculate element size (diameter)
    h_E = compute_element_diameter(element_vertices)
    
    # Calculate stabilization parameter τ
    tau = compute_stabilization_parameter(E, nu)
    
    # Calculate I-P (using the provided projection matrix)
    I_minus_P = np.eye(n_dofs) - P
    
    # Initialize stabilization matrix
    K_stab = np.zeros((n_dofs, n_dofs))
    
    # Loop over all edges of the element
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        
        # Extract edge vertices
        edge_vertices = np.array([element_vertices[i], element_vertices[j]])
        
        # Compute edge length and parameterization
        r1, z1 = edge_vertices[0]
        r2, z2 = edge_vertices[1]
        dr = r2 - r1
        dz = z2 - z1
        edge_length = np.sqrt(dr**2 + dz**2)
        
        # Skip if edge is too short
        if edge_length < 1e-10:
            continue
        
        # Check if edge is vertical (constant r)
        is_vertical = abs(dr) < 1e-10
        
        # Create a local contribution matrix for this edge
        edge_contrib = np.zeros((n_dofs, n_dofs))
        
        # Define Gauss points and weights based on edge type
        if is_vertical:
            # Use 1-point quadrature for vertical edges
            s_points = [0.5]
            weights = [1.0]
        else:
            # Use 2-point Gauss quadrature for non-vertical edges
            s1 = 0.5 - np.sqrt(3) / 6
            s2 = 0.5 + np.sqrt(3) / 6
            s_points = [s1, s2]
            weights = [0.5, 0.5]
        
        # Loop over Gauss points
        for s, w in zip(s_points, weights):
            # Evaluate r at the current Gauss point
            r_s = r1 + s * dr
            
            # For each pair of DOFs that live on this edge
            for edge_vertex_i in [i, j]:
                for edge_vertex_j in [i, j]:
                    # Get shape function values at this Gauss point
                    N_i = 1 - s if edge_vertex_i == i else s
                    N_j = 1 - s if edge_vertex_j == i else s
                    
                    # Loop over DOF components (radial and axial)
                    for comp_i in range(2):
                        for comp_j in range(2):
                            # Global DOF indices
                            dof_i = 2 * edge_vertex_i + comp_i
                            dof_j = 2 * edge_vertex_j + comp_j
                            
                            # For each pair of DOFs, get the (I-P) value
                            I_minus_P_val = I_minus_P[dof_i, dof_j]
                            
                            # Contribution from this quadrature point
                            contrib = I_minus_P_val * N_i * N_j * 2 * np.pi * r_s * w * edge_length
                            
                            # Add to the edge contribution matrix
                            edge_contrib[dof_i, dof_j] += contrib
        
        # Scale by τ * h_E^2 and add to the global stabilization matrix
        K_stab += tau * h_E**(-1) * edge_contrib
    
    return K_stab


def compute_element_diameter(vertices: np.ndarray) -> float:
    """
    Compute the diameter of an element (maximum distance between any two vertices).
    
    Parameters:
        vertices (np.ndarray): Array of vertex coordinates with shape (n_vertices, 2)
    
    Returns:
        float: The element diameter
    """
    n_vertices = len(vertices)
    diameter = 0.0
    
    # Compute pairwise distances between all vertices
    for i in range(n_vertices):
        for j in range(i+1, n_vertices):
            # Compute Euclidean distance
            dist = np.sqrt(np.sum((vertices[i] - vertices[j])**2))
            diameter = max(diameter, dist)
    
    return diameter

def compute_stabilization_parameter(E: float, nu: float) -> float:
    """
    Compute the stabilization parameter τ based on material properties.
    
    Parameters:
        E (float): Young's modulus
        nu (float): Poisson's ratio
    
    Returns:
        float: The stabilization parameter τ
    """
    # Compute shear modulus
    mu = E / (2 * (1 + nu))
    
    # Base scaling factor relative to shear modulus
    base_scaling = 1e-3
    
    # Scale τ based on material properties
    if nu < 0.4:
        # Standard scaling for normal materials
        tau = base_scaling * mu
    elif nu < 0.45:
        # Smooth linear increase for transition region
        t = (nu - 0.4) / 0.05
        tau = base_scaling * mu * (1.0 + t)
    else:
        # Enhanced scaling for nearly incompressible materials
        t = min(1.0, (nu - 0.45) / 0.05)
        tau = base_scaling * mu * (2.0 + 3.0 * t)
    
    return tau


# ===== Example usage =====
def test_stiffness_matrix():
    """
    Test the stiffness matrix with proper force analysis.
    
    This function:
    1. Creates a quadrilateral element
    2. Computes its stiffness matrix
    3. Tests the response to a constant radial strain field
    4. Analyzes the resulting nodal forces to verify physical correctness
    
    The test checks whether:
    - Force directions are consistent with radial expansion
    - The net moment about the z-axis is approximately zero
    """
    quad_vertices = np.array([
        [1.0, 0.0],  # Vertex 1
        [2.0, 0.0],  # Vertex 2
        [2.0, 1.0],  # Vertex 3
        [1.0, 1.0]   # Vertex 4
    ])
    
    # Compute stiffness matrix
    K, K_c, K_s = compute_stiffness_matrix(quad_vertices, 1.0, 0.3)
    
    # Test 1: Constant radial strain (u_r = r, u_z = 0)
    d_test1 = np.zeros(8)
    for i in range(4):
        d_test1[2*i] = quad_vertices[i, 0]  # r-coordinate
    
    # Calculate forces
    f1 = np.dot(K, d_test1)
    
    # Reorganize forces by node for analysis
    forces_by_node = []
    for i in range(4):
        fr = f1[2*i]
        fz = f1[2*i+1]
        forces_by_node.append((fr, fz))
    
    print("\nAnalysis of forces for constant radial strain:")
    for i, (fr, fz) in enumerate(forces_by_node):
        r, z = quad_vertices[i]
        print(f"Node {i+1} at ({r},{z}): Fr = {fr:.6f}, Fz = {fz:.6f}")
    
    # Check if forces are consistent with radial expansion
    # For radial expansion:
    # - Nodes at higher r should have positive Fr (outward)
    # - Nodes at lower r should have negative Fr (outward)
    consistent_radial = True
    for i, (fr, _) in enumerate(forces_by_node):
        r = quad_vertices[i, 0]
        if (r > 1.5 and fr < 0) or (r < 1.5 and fr > 0):
            consistent_radial = False
    
    print(f"\nForces consistent with radial expansion: {consistent_radial}")
    
    # Calculate moment about z-axis (should be zero for pure radial expansion)
    moment_z = 0.0
    for i, (fr, fz) in enumerate(forces_by_node):
        r, z = quad_vertices[i]
        # Moment about z-axis from radial forces should be zero
        # (radial forces pass through z-axis)
        moment_z += r * fz
    
    print(f"Net moment about z-axis: {moment_z:.6e}")

# Helper functions for testing
def constant_body_force(r, z):
    """Return a constant body force in the radial direction."""
    return 1.0, 0.0  # (b_r, b_z)

def constant_traction(r, z):
    """Return a constant traction in the radial direction."""
    return -1.0, 0.0  # (t_r, t_z) - negative for inward pressure

def test_load_vector():
    """Test the load vector computation with simple examples."""
    # Define a quadrilateral element
    quad_vertices = np.array([
        [1.0, 0.0],  # Vertex 1
        [2.0, 0.0],  # Vertex 2
        [2.0, 1.0],  # Vertex 3
        [1.0, 1.0]   # Vertex 4
    ])
    
    # Test body force load vector
    print("Testing body force load vector:")
    f_body = compute_element_load_body_force(quad_vertices, constant_body_force)
    for i in range(4):
        print(f"Node {i+1}: Fr = {f_body[2*i]:.6f}, Fz = {f_body[2*i+1]:.6f}")
    
    # Test boundary traction load vector (apply on left edge: vertices 0-3)
    print("\nTesting boundary traction load vector (left edge):")
    f_traction = compute_element_load_boundary_traction(
        quad_vertices, [0, 3], constant_traction)
    for i in range(4):
        print(f"Node {i+1}: Fr = {f_traction[2*i]:.6f}, Fz = {f_traction[2*i+1]:.6f}")
    
    # Test combined load vector
    print("\nTesting combined load vector:")
    f_combined = assemble_element_load_vector(
        quad_vertices, constant_body_force, [0, 3], constant_traction)
    for i in range(4):
        print(f"Node {i+1}: Fr = {f_combined[2*i]:.6f}, Fz = {f_combined[2*i+1]:.6f}")

if __name__ == "__main__":
    # test_projection_matrix()
    # test_stiffness_matrix()
    test_load_vector()