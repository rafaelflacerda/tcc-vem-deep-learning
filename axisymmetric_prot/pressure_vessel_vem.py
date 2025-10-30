import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union

class BoundaryConditionType(Enum):
    """Enumeration of boundary condition types."""
    DIRICHLET = 0
    NEUMANN = 1

class LoadType(Enum):
    """Enumeration of load types."""
    TRACTION = 0
    BODY_FORCE = 1
    POINT_LOAD = 2

class VEMElement(ABC):
    """Base abstract class for VEM elements."""
    
    def __init__(self, element_id: int, node_ids: List[int], material_id: int):
        self.element_id = element_id
        self.node_ids = node_ids
        self.material_id = material_id
        self.n_vertices = len(node_ids)
        self.n_dofs = 2 * self.n_vertices  # 2 DOFs per vertex (ur, uz)
    
    @abstractmethod
    def compute_stiffness_matrix(self, node_coords: np.ndarray, material: Dict) -> np.ndarray:
        """Compute the element stiffness matrix."""
        pass
    
    @abstractmethod
    def compute_load_vector(self, node_coords: np.ndarray, loads: List[Dict]) -> np.ndarray:
        """Compute the element load vector."""
        pass


class VEMQuadElement(VEMElement):
    """VEM quadrilateral element for axisymmetric analysis."""
    
    def __init__(self, element_id: int, node_ids: List[int], material_id: int):
        super().__init__(element_id, node_ids, material_id)
        if len(node_ids) != 4:
            raise ValueError("VEMQuadElement requires exactly 4 nodes.")
    
    def compute_stiffness_matrix(self, node_coords: np.ndarray, material: Dict) -> np.ndarray:
        """
        Compute the element stiffness matrix for axisymmetric problems.
        
        Parameters:
            node_coords (np.ndarray): Coordinates of all nodes (global numbering)
            material (Dict): Material properties including 'E' and 'nu'
        
        Returns:
            np.ndarray: Element stiffness matrix (8x8 for quad element)
        """
        # Extract element vertices using node_ids
        vertices = node_coords[self.node_ids]
        
        # Extract material properties
        E = material['E']
        nu = material['nu']
        
        # Build constitutive matrix
        factor = E / ((1 + nu) * (1 - 2 * nu))
        C = factor * np.array([
            [1 - nu, nu, nu, 0],
            [nu, 1 - nu, nu, 0],
            [nu, nu, 1 - nu, 0],
            [0, 0, 0, (1 - 2 * nu) / 2]
        ])
        
        # Compute element area
        r_coords = vertices[:, 0]
        z_coords = vertices[:, 1]
        area = 0.0
        for i in range(self.n_vertices):
            j = (i + 1) % self.n_vertices
            area += r_coords[i] * z_coords[j] - r_coords[j] * z_coords[i]
        area = abs(area) / 2
        
        # Compute centroid
        r_c = np.mean(r_coords)
        z_c = np.mean(z_coords)
        
        # Compute shape function derivatives
        dN_dr, dN_dz = self._compute_derivatives(r_coords, z_coords)
        
        # Initialize the B matrix for the VEM projection
        B = np.zeros((4, 8))
        
        # Construct the B matrix
        for i in range(4):
            idx_r = 2*i
            idx_z = 2*i + 1
            
            # εr = ∂u_r/∂r
            B[0, idx_r] = dN_dr[i]
            
            # εz = ∂u_z/∂z
            B[1, idx_z] = dN_dz[i]
            
            # εθ = u_r/r - For axisymmetric problems
            # Use a consistent approach with the centroid radius
            # For a bilinear quad, all shape functions contribute equally at the centroid
            B[2, idx_r] = 0.25 / r_c
            
            # γrz = ∂u_r/∂z + ∂u_z/∂r
            B[3, idx_r] = dN_dz[i]
            B[3, idx_z] = dN_dr[i]
        
        # For axisymmetric problems, we need to properly integrate the r-weighted terms
        # K_c = ∫ B^T * C * B * r dA ≈ B^T * C * B * r_c * A
        K_c = np.dot(np.dot(B.T, C), B) * r_c * area
        
        # Compute stabilization term
        D = B
        DT = D.T
        DDT = np.dot(D, DT)
        DDT_inv = np.linalg.pinv(DDT)
        P = np.dot(DT, np.dot(DDT_inv, D))
        I = np.eye(8)
        I_minus_Pi = I - P
        
        # Adjust stabilization parameter based on element properties
        # For axisymmetric problems, element aspect ratio matters
        r_min = np.min(r_coords)
        r_max = np.max(r_coords)
        thickness_ratio = (r_max - r_min) / r_min
        
        # Scale alpha based on element position and thickness
        if r_min < 1.5 * r_c:  # Elements near the axis need more careful stabilization
            alpha_scale = 1e3 * (1.0 + 5.0 * thickness_ratio)
        else:
            alpha_scale = 1e3
        
        alpha = np.trace(C) * area / (np.max(np.abs(K_c)) * alpha_scale)
        K_s = alpha * I_minus_Pi
        
        # Complete stiffness matrix
        K = K_c + K_s
        
        return K
    
    def compute_load_vector(self, node_coords: np.ndarray, loads: List[Dict]) -> np.ndarray:
        """
        Compute the element load vector for various types of loads.
        
        Parameters:
            node_coords (np.ndarray): Coordinates of all nodes (global numbering)
            loads (List[Dict]): List of load definitions
        
        Returns:
            np.ndarray: Element load vector
        """
        # Extract element vertices
        vertices = node_coords[self.node_ids]
        
        # Initialize load vector
        F = np.zeros(self.n_dofs)
        
        # Process each load
        for load in loads:
            load_type = load['type']
            
            if load_type == LoadType.TRACTION:
                # Check if this load applies to this element
                if self.element_id not in load['element_ids']:
                    continue
                
                # Get traction vector
                traction = np.array(load['value'])
                
                # Get the edges this traction applies to
                if 'edge_indices' in load:
                    # Edge indices are provided as local element edge indices
                    edge_indices = load['edge_indices']
                else:
                    # Apply to boundary edges that match the normal direction
                    normal = np.array(load['normal']) if 'normal' in load else None
                    edge_indices = self._find_boundary_edges(vertices, normal)
                
                # Compute traction load vector
                F_traction = self._compute_traction_load(vertices, edge_indices, traction)
                F += F_traction
                
            elif load_type == LoadType.BODY_FORCE:
                # Body force implementation would go here
                # Currently not implemented as per request
                pass
                
            elif load_type == LoadType.POINT_LOAD:
                # Point load implementation would go here
                # Not typically used in continuum VEM
                pass
        
        return F
    
    def _compute_derivatives(self, r_coords: np.ndarray, z_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the derivatives of shape functions with respect to r and z.
        
        Parameters:
            r_coords (np.ndarray): Radial coordinates of vertices
            z_coords (np.ndarray): Axial coordinates of vertices
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Derivatives with respect to r and z
        """
        # Shape function derivatives in natural coordinates
        dN_dxi = np.array([-0.25, 0.25, 0.25, -0.25])
        dN_deta = np.array([-0.25, -0.25, 0.25, 0.25])
        
        # Compute Jacobian at the centroid
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, r_coords)   # dr/dxi
        J[0, 1] = np.dot(dN_dxi, z_coords)   # dz/dxi
        J[1, 0] = np.dot(dN_deta, r_coords)  # dr/deta
        J[1, 1] = np.dot(dN_deta, z_coords)  # dz/deta
        
        # Compute inverse of Jacobian
        detJ = J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]
        Jinv = np.array([
            [J[1, 1]/detJ, -J[0, 1]/detJ],
            [-J[1, 0]/detJ, J[0, 0]/detJ]
        ])
        
        # Compute derivatives of shape functions w.r.t. r and z
        dN_dr = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
        dN_dz = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
        
        return dN_dr, dN_dz
    
    def _find_boundary_edges(self, vertices: np.ndarray, normal: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
        """
        Find boundary edges that match a given normal direction.
        
        Parameters:
            vertices (np.ndarray): Vertex coordinates
            normal (np.ndarray, optional): Normal direction to match
        
        Returns:
            List[Tuple[int, int]]: List of edge vertex indices
        """
        # For simplicity, return all edges if no normal is provided
        if normal is None:
            return [(i, (i+1) % self.n_vertices) for i in range(self.n_vertices)]
        
        # Find edges with matching normal
        edges = []
        for i in range(self.n_vertices):
            j = (i + 1) % self.n_vertices
            
            # Compute edge normal (2D)
            edge_vector = vertices[j] - vertices[i]
            edge_normal = np.array([-edge_vector[1], edge_vector[0]])
            edge_normal = edge_normal / np.linalg.norm(edge_normal)
            
            # Check if normal matches
            if np.dot(edge_normal, normal) > 0.9:  # Allow some tolerance
                edges.append((i, j))
        
        return edges
    
    def _compute_traction_load(self, vertices: np.ndarray, edge_indices: List[Tuple[int, int]], 
                              traction_vector: np.ndarray) -> np.ndarray:
        """
        Compute the local load vector due to traction forces.
        
        Parameters:
            vertices (np.ndarray): Vertex coordinates
            edge_indices (List[Tuple[int, int]]): List of edge vertex indices
            traction_vector (np.ndarray): Traction vector [tr, tz]
        
        Returns:
            np.ndarray: Local load vector
        """
        # Initialize the local load vector
        F_local = np.zeros(self.n_dofs)
        
        # Extract traction components
        tr, tz = traction_vector
        
        # Gauss quadrature points and weights for 2-point integration
        s_points = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
        s_weights = np.array([0.5, 0.5])
        
        # Process each edge with traction
        for i, j in edge_indices:
            # Get coordinates of edge vertices
            ri, zi = vertices[i]
            rj, zj = vertices[j]
            
            # Edge length
            L = np.sqrt((rj - ri)**2 + (zj - zi)**2)
            
            # Loop over Gauss points
            for s, w in zip(s_points, s_weights):
                # Parameterize the edge
                r_gauss = ri + s * (rj - ri)
                z_gauss = zi + s * (zj - zi)
                
                # Shape functions at Gauss point
                N_i = 1 - s
                N_j = s
                
                F_local[2*i] -= N_i * tr * r_gauss * L * w  # Note the negative sign
                F_local[2*i+1] -= N_i * tz * r_gauss * L * w
                F_local[2*j] -= N_j * tr * r_gauss * L * w
                F_local[2*j+1] -= N_j * tz * r_gauss * L * w
        
        return F_local


class VEMSolver:
    """Virtual Element Method solver for axisymmetric problems."""
    
    def __init__(self):
        self.nodes = []  # List of node coordinates
        self.elements = []  # List of elements
        self.materials = {}  # Dictionary of materials
        self.boundary_conditions = []  # List of boundary conditions
        self.loads = []  # List of loads
        self.n_nodes = 0
        self.n_elements = 0
        self.n_dofs = 0
    
    def add_nodes(self, coordinates: np.ndarray):
        """
        Add nodes to the model.
        
        Parameters:
            coordinates (np.ndarray): Array of node coordinates (r, z)
        """
        self.nodes = coordinates
        self.n_nodes = len(coordinates)
        self.n_dofs = 2 * self.n_nodes  # 2 DOFs per node (ur, uz)
    
    def add_element(self, element_type: str, node_ids: List[int], material_id: int):
        """
        Add an element to the model.
        
        Parameters:
            element_type (str): Type of element ('quad', 'tri', etc.)
            node_ids (List[int]): List of node IDs defining the element
            material_id (int): Material ID for the element
        """
        element_id = len(self.elements)
        
        if element_type.lower() == 'quad':
            element = VEMQuadElement(element_id, node_ids, material_id)
        else:
            raise ValueError(f"Unsupported element type: {element_type}")
        
        self.elements.append(element)
        self.n_elements = len(self.elements)
    
    def add_material(self, material_id: int, properties: Dict):
        """
        Add a material to the model.
        
        Parameters:
            material_id (int): Material ID
            properties (Dict): Material properties (E, nu, etc.)
        """
        self.materials[material_id] = properties
    
    def add_boundary_condition(self, bc_type: BoundaryConditionType, node_ids: List[int], 
                              dof_indices: List[int], values: List[float]):
        """
        Add a boundary condition to the model.
        
        Parameters:
            bc_type (BoundaryConditionType): Type of boundary condition
            node_ids (List[int]): List of node IDs where the BC is applied
            dof_indices (List[int]): List of DOF indices (0 for ur, 1 for uz)
            values (List[float]): Values to apply
        """
        self.boundary_conditions.append({
            'type': bc_type,
            'node_ids': node_ids,
            'dof_indices': dof_indices,
            'values': values
        })
    
    def add_traction_load(self, element_ids: List[int], value: List[float],
                        edge_indices: Optional[List[Tuple[int, int]]] = None, normal: Optional[List[float]] = None):
        """
        Add a traction load to the model.
        
        Parameters:
            element_ids (List[int]): List of element IDs where the load is applied
            edge_indices (List[Tuple[int, int]], optional): List of edge indices
            value (List[float]): Traction vector [tr, tz]
            normal (List[float], optional): Normal direction to identify edges
        """
        load = {
            'type': LoadType.TRACTION,
            'element_ids': element_ids,
            'value': value
        }
        
        if edge_indices is not None:
            load['edge_indices'] = edge_indices
        
        if normal is not None:
            load['normal'] = normal
        
        self.loads.append(load)
    
    def assemble_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assemble the global stiffness matrix and load vector.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Global stiffness matrix and load vector
        """
        # Initialize global stiffness matrix and load vector
        K_global = np.zeros((self.n_dofs, self.n_dofs))
        F_global = np.zeros(self.n_dofs)
        
        # Assemble element contributions
        for element in self.elements:
            # Get material properties
            material = self.materials[element.material_id]
            
            # Compute element stiffness matrix
            K_elem = element.compute_stiffness_matrix(self.nodes, material)
            
            # Compute element load vector
            F_elem = element.compute_load_vector(self.nodes, self.loads)
            
            # Map local DOFs to global DOFs
            dof_indices = []
            for node_id in element.node_ids:
                dof_indices.extend([2*node_id, 2*node_id+1])  # r and z DOFs
            
            # Assemble into global matrices
            for i in range(len(dof_indices)):
                F_global[dof_indices[i]] += F_elem[i]
                for j in range(len(dof_indices)):
                    K_global[dof_indices[i], dof_indices[j]] += K_elem[i, j]
        
        return K_global, F_global
    
    def apply_boundary_conditions(self, K_global: np.ndarray, F_global: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply boundary conditions to the global system.
        
        Parameters:
            K_global (np.ndarray): Global stiffness matrix
            F_global (np.ndarray): Global load vector
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Modified system and prescribed values
        """
        # Initialize arrays
        prescribed_dofs = []
        prescribed_values = []
        
        # Process boundary conditions
        for bc in self.boundary_conditions:
            if bc['type'] == BoundaryConditionType.DIRICHLET:
                for node_id in bc['node_ids']:
                    for dof_index in bc['dof_indices']:
                        global_dof = 2 * node_id + dof_index
                        prescribed_dofs.append(global_dof)
                        
                        # Get the value (use index 0 if only one value is provided)
                        if len(bc['values']) == 1:
                            prescribed_values.append(bc['values'][0])
                        else:
                            value_index = bc['dof_indices'].index(dof_index)
                            prescribed_values.append(bc['values'][value_index])
        
        # Convert to arrays
        prescribed_dofs = np.array(prescribed_dofs)
        prescribed_values = np.array(prescribed_values)
        
        # Identify free DOFs
        all_dofs = np.arange(self.n_dofs)
        free_dofs = np.setdiff1d(all_dofs, prescribed_dofs)
        
        # Modify the system to account for prescribed DOFs
        F_mod = F_global.copy()
        
        if len(prescribed_dofs) > 0:
            # Subtract the effect of prescribed values from the RHS
            for i, dof in enumerate(prescribed_dofs):
                F_mod[free_dofs] -= K_global[free_dofs, dof] * prescribed_values[i]
        
        # Prepare reduced system
        K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
        F_reduced = F_mod[free_dofs]
        
        return K_reduced, F_reduced, free_dofs
    
    def solve(self) -> np.ndarray:
        """
        Solve the VEM system.
        
        Returns:
            np.ndarray: Displacement vector
        """
        # Assemble the global system
        K_global, F_global = self.assemble_system()
        
        # Apply boundary conditions
        K_reduced, F_reduced, free_dofs = self.apply_boundary_conditions(K_global, F_global)
        
        # Solve the reduced system
        u_reduced = np.linalg.solve(K_reduced, F_reduced)
        
        # Initialize the full displacement vector
        u = np.zeros(self.n_dofs)
        
        # Fill in the free DOFs
        u[free_dofs] = u_reduced
        
        # Fill in the prescribed DOFs
        for bc in self.boundary_conditions:
            if bc['type'] == BoundaryConditionType.DIRICHLET:
                for node_id in bc['node_ids']:
                    for i, dof_index in enumerate(bc['dof_indices']):
                        global_dof = 2 * node_id + dof_index
                        if len(bc['values']) == 1:
                            u[global_dof] = bc['values'][0]
                        else:
                            u[global_dof] = bc['values'][i]
        
        return u
    
    def compute_strains(self, u: np.ndarray) -> List[np.ndarray]:
        """
        Compute strains in each element.
        
        Parameters:
            u (np.ndarray): Global displacement vector
        
        Returns:
            List[np.ndarray]: List of strain vectors [εr, εz, εθ, γrz] for each element
        """
        strains = []
        
        for element in self.elements:
            # Extract element displacements
            elem_disps = np.zeros(element.n_dofs)
            for i, node_id in enumerate(element.node_ids):
                elem_disps[2*i] = u[2*node_id]      # ur
                elem_disps[2*i+1] = u[2*node_id+1]  # uz
            
            # Get node coordinates
            vertices = self.nodes[element.node_ids]
            
            # For quad elements
            if isinstance(element, VEMQuadElement):
                # Compute derivatives
                dN_dr, dN_dz = element._compute_derivatives(vertices[:, 0], vertices[:, 1])
                
                # Compute centroid
                r_c = np.mean(vertices[:, 0])
                
                # Initialize strain vector [εr, εz, εθ, γrz]
                strain = np.zeros(4)
                
                # Compute strains from displacements
                for i in range(4):
                    # Radial displacement contribution
                    strain[0] += dN_dr[i] * elem_disps[2*i]         # εr += ∂N_i/∂r * u_ri
                    strain[3] += dN_dz[i] * elem_disps[2*i]         # γrz += ∂N_i/∂z * u_ri
                    
                    # Axial displacement contribution
                    strain[1] += dN_dz[i] * elem_disps[2*i+1]       # εz += ∂N_i/∂z * u_zi
                    strain[3] += dN_dr[i] * elem_disps[2*i+1]       # γrz += ∂N_i/∂r * u_zi
                    
                    # Hoop strain computation (circumferential)
                    strain[2] += 0.25 * elem_disps[2*i] / r_c        # εθ += N_i * u_ri / r
                
                strains.append(strain)
            
            # Could add other element types here
        
        return strains
    
    def compute_stresses(self, strains: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute stresses in each element.
        
        Parameters:
            strains (List[np.ndarray]): List of strain vectors for each element
        
        Returns:
            List[np.ndarray]: List of stress vectors [σr, σz, σθ, τrz] for each element
        """
        stresses = []
        
        for i, strain in enumerate(strains):
            # Get element and material
            element = self.elements[i]
            material = self.materials[element.material_id]
            
            # Extract material properties
            E = material['E']
            nu = material['nu']
            
            # Build constitutive matrix
            factor = E / ((1 + nu) * (1 - 2 * nu))
            C = factor * np.array([
                [1 - nu, nu, nu, 0],
                [nu, 1 - nu, nu, 0],
                [nu, nu, 1 - nu, 0],
                [0, 0, 0, (1 - 2 * nu) / 2]
            ])
            
            # Compute stress = C * strain
            stress = np.dot(C, strain)
            stresses.append(stress)
        
        return stresses
    
    def plot_mesh(self, ax=None, color='k', show_elements=True, show_nodes=True):
        """
        Plot the mesh of the model.
        
        Parameters:
            ax: Matplotlib axis
            color (str): Color for the mesh
            show_elements (bool): Whether to show element IDs
            show_nodes (bool): Whether to show node IDs
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot elements
        for i, element in enumerate(self.elements):
            vertices = self.nodes[element.node_ids]
            
            # Close the polygon for plotting
            r_coords = np.append(vertices[:, 0], vertices[0, 0])
            z_coords = np.append(vertices[:, 1], vertices[0, 1])
            
            ax.plot(r_coords, z_coords, color=color, linewidth=1)
            
            if show_elements:
                # Element ID at centroid
                r_c = np.mean(vertices[:, 0])
                z_c = np.mean(vertices[:, 1])
                ax.text(r_c, z_c, f"E{i}", fontsize=8, ha='center', va='center')
        
        # Plot nodes
        if show_nodes:
            ax.scatter(self.nodes[:, 0], self.nodes[:, 1], color='blue', s=30)
            for i, (r, z) in enumerate(self.nodes):
                ax.text(r, z, f"{i}", fontsize=8, color='blue', ha='right', va='bottom')
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('r-coordinate')
        ax.set_ylabel('z-coordinate')
        
        return ax
    
    def plot_loads(self, ax=None, scale=0.2, color='red'):
        """
        Plot the loads applied to the model.
        
        Parameters:
            ax: Matplotlib axis
            scale (float): Scaling factor for arrows
            color (str): Color for the load arrows
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            self.plot_mesh(ax)
        
        # Plot traction loads
        for load in self.loads:
            if load['type'] == LoadType.TRACTION:
                traction_value = np.array(load['value'])
                
                for element_id in load['element_ids']:
                    element = self.elements[element_id]
                    vertices = self.nodes[element.node_ids]
                    
                    # Determine which edges to plot
                    if 'edge_indices' in load:
                        edge_indices = load['edge_indices']
                    else:
                        normal = np.array(load['normal']) if 'normal' in load else None
                        edge_indices = element._find_boundary_edges(vertices, normal)
                    
                    # Plot traction arrows on edges
                    for i, j in edge_indices:
                        # Midpoint of the edge
                        r_mid = (vertices[i, 0] + vertices[j, 0]) / 2
                        z_mid = (vertices[i, 1] + vertices[j, 1]) / 2
                        
                        # Draw traction vector
                        ax.arrow(r_mid, z_mid, 
                                scale * traction_value[0], 
                                scale * traction_value[1],
                                head_width=0.05, 
                                head_length=0.1, 
                                fc=color, 
                                ec=color, 
                                width=0.01)
        
        ax.set_aspect('equal')
        return ax
    
    def plot_deformed_mesh(self, u, scale=1.0, ax=None):
        """
        Plot the deformed mesh.
        
        Parameters:
            u (np.ndarray): Displacement vector
            scale (float): Scaling factor for displacements
            ax: Matplotlib axis
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot original mesh
        self.plot_mesh(ax, color='lightgray')
        
        # Plot deformed mesh
        for element in self.elements:
            original_vertices = self.nodes[element.node_ids]
            deformed_vertices = np.zeros_like(original_vertices)
            
            for i, node_id in enumerate(element.node_ids):
                deformed_vertices[i, 0] = original_vertices[i, 0] + scale * u[2*node_id]      # r + ur
                deformed_vertices[i, 1] = original_vertices[i, 1] + scale * u[2*node_id+1]    # z + uz
            
            # Close the polygon for plotting
            r_coords = np.append(deformed_vertices[:, 0], deformed_vertices[0, 0])
            z_coords = np.append(deformed_vertices[:, 1], deformed_vertices[0, 1])
            
            ax.plot(r_coords, z_coords, color='red', linewidth=1)
        
        # Add displacement vectors
        for i in range(self.n_nodes):
            r, z = self.nodes[i]
            dr = scale * u[2*i]
            dz = scale * u[2*i+1]
            
            # Only draw significant displacements
            if np.sqrt(dr**2 + dz**2) > 1e-6:
                ax.arrow(r, z, dr, dz, 
                        head_width=0.05, 
                        head_length=0.1, 
                        fc='green', 
                        ec='green', 
                        width=0.01)
        
        ax.set_aspect('equal')
        ax.set_title(f'Deformed Mesh (scale={scale})')
        ax.grid(True)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='lightgray', lw=1, label='Original Mesh'),
            Line2D([0], [0], color='red', lw=1, label='Deformed Mesh'),
            Line2D([0], [0], color='green', lw=1, label='Displacement')
        ]
        ax.legend(handles=legend_elements)
        
        return ax
    
    def plot_results(self, u, result_type='displacement', component=0, ax=None):
        """
        Plot results on the mesh.
        
        Parameters:
            u (np.ndarray): Displacement vector
            result_type (str): Type of result ('displacement', 'strain', or 'stress')
            component (int): Component to plot (0=r, 1=z, 2=θ, 3=rz)
            ax: Matplotlib axis
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data for triangulation
        r_coords = self.nodes[:, 0]
        z_coords = self.nodes[:, 1]
        
        # Triangulate the mesh for plotting
        # This is an approximation for arbitrary polygons
        triangles = []
        for element in self.elements:
            if len(element.node_ids) == 4:  # Quad element
                triangles.append([element.node_ids[0], element.node_ids[1], element.node_ids[2]])
                triangles.append([element.node_ids[0], element.node_ids[2], element.node_ids[3]])
            elif len(element.node_ids) == 3:  # Triangle element
                triangles.append(element.node_ids)
            # Could handle more general polygons here
        
        # Create a triangulation
        triang = mtri.Triangulation(r_coords, z_coords, triangles)
        
        # Prepare data to plot
        if result_type.lower() == 'displacement':
            # Plot displacement component
            if component == 0:
                values = u[0::2]  # ur
                title = 'Radial Displacement (ur)'
            elif component == 1:
                values = u[1::2]  # uz
                title = 'Axial Displacement (uz)'
            elif component == 2:
                # Displacement magnitude
                values = np.sqrt(u[0::2]**2 + u[1::2]**2)
                title = 'Displacement Magnitude'
            else:
                raise ValueError("Invalid displacement component")
        
        elif result_type.lower() == 'strain' or result_type.lower() == 'stress':
            # Compute strains or stresses
            if result_type.lower() == 'strain':
                element_values = self.compute_strains(u)
                labels = ['Radial Strain (εr)', 'Axial Strain (εz)', 
                         'Hoop Strain (εθ)', 'Shear Strain (γrz)']
            else:  # stress
                strains = self.compute_strains(u)
                element_values = self.compute_stresses(strains)
                labels = ['Radial Stress (σr)', 'Axial Stress (σz)', 
                         'Hoop Stress (σθ)', 'Shear Stress (τrz)']
            
            if component > 3:
                raise ValueError("Invalid component index")
            
            # Project element values to nodes (simple averaging)
            values = np.zeros(self.n_nodes)
            count = np.zeros(self.n_nodes)
            
            for i, element in enumerate(self.elements):
                element_value = element_values[i][component]
                for node_id in element.node_ids:
                    values[node_id] += element_value
                    count[node_id] += 1
            
            # Average
            values = np.divide(values, count, where=count>0)
            title = labels[component]
        
        else:
            raise ValueError(f"Invalid result type: {result_type}")
        
        # Plot as filled contours
        contour = ax.tricontourf(triang, values, cmap='viridis')
        plt.colorbar(contour, ax=ax, label=title)
        
        # Add mesh outline
        self.plot_mesh(ax, color='k', show_elements=False, show_nodes=False)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        
        return ax


def example_pressure_vessel():
    """
    Example of a pressure vessel analysis using VEM with traction loads.
    """
    import matplotlib.pyplot as plt
    
    # Create a new VEM solver
    solver = VEMSolver()
    
    # Define material
    solver.add_material(1, {'E': 210e9, 'nu': 0.3})  # Steel
    
    # Create a simple cylinder mesh (r_inner=0.1m, r_outer=0.12m, height=0.2m)
    r_inner = 0.1
    r_outer = 0.12
    height = 0.2
    
    # Create nodes (2x2 elements in r and z directions)
    n_all = 22
    nr = n_all # nodes in r direction
    nz = n_all  # nodes in z direction
    
    nodes = np.zeros((nr * nz, 2))
    node_idx = 0
    
    for iz in range(nz):
        for ir in range(nr):
            r = r_inner + (r_outer - r_inner) * ir / (nr - 1)
            z = height * iz / (nz - 1)
            nodes[node_idx] = [r, z]
            node_idx += 1
    
    solver.add_nodes(nodes)
    
    # Create elements
    for iz in range(nz - 1):
        for ir in range(nr - 1):
            # Node indices for this element
            n1 = iz * nr + ir         # Bottom left
            n2 = iz * nr + (ir + 1)   # Bottom right
            n3 = (iz + 1) * nr + (ir + 1)  # Top right
            n4 = (iz + 1) * nr + ir   # Top left
            
            # Add quad element
            solver.add_element('quad', [n1, n2, n3, n4], 1)
    
    # Apply pressure load on inner surface (negative r direction)
    pressure = 10e6  # 10 MPa
    left_elements = [0, 2]  # Elements on the left side (inner radius)
    
    # Apply pressure on the inner surface (negative radial direction)
    solver.add_traction_load(
        element_ids=left_elements,
        edge_indices=[(0, 3)],  # Left edge (local nodes 0 and 3)
        value=[-pressure, 0.0]  # Negative pressure in radial direction
    )
    
    # Apply boundary conditions
    # Fix bottom in z direction
    bottom_nodes = [0, 1, 2]
    solver.add_boundary_condition(
        bc_type=BoundaryConditionType.DIRICHLET,
        node_ids=bottom_nodes,
        dof_indices=[1],  # z-direction
        values=[0.0]      # Fixed
    )
    
    # Fix a node in r direction to prevent rigid body motion
    solver.add_boundary_condition(
        bc_type=BoundaryConditionType.DIRICHLET,
        node_ids=[0],
        dof_indices=[0],  # r-direction
        values=[0.0]      # Fixed
    )
    
    # Solve the system
    u = solver.solve()
    
    # Plot results
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot mesh and loads
    ax = axs[0, 0]
    solver.plot_mesh(ax)
    solver.plot_loads(ax)
    ax.set_title('Mesh and Applied Loads')
    
    # Plot deformed mesh
    ax = axs[0, 1]
    scale = 1000  # Scale displacement for visibility
    solver.plot_deformed_mesh(u, scale=scale, ax=ax)
    ax.set_title(f'Deformed Shape (scale={scale})')
    
    # Plot radial displacement
    ax = axs[1, 0]
    solver.plot_results(u, result_type='displacement', component=0, ax=ax)
    
    # Plot hoop stress
    ax = axs[1, 1]
    solver.plot_results(u, result_type='stress', component=2, ax=ax)
    
    # plt.tight_layout()
    # plt.show()
    
    # Print some results
    strains = solver.compute_strains(u)
    stresses = solver.compute_stresses(strains)
    
    print("\nResults Summary:")
    print("----------------")
    print(f"Maximum radial displacement: {np.max(np.abs(u[0::2])):.6e} m")
    print(f"Maximum axial displacement: {np.max(np.abs(u[1::2])):.6e} m")
    
    # Get max hoop stress (component 2)
    max_hoop_stress = max([stress[2] for stress in stresses])
    print(f"Maximum hoop stress: {max_hoop_stress/1e6:.2f} MPa")
    
    # Compare with analytical solution for thin-walled cylinder
    analytical_hoop_stress = pressure * r_inner / (r_outer - r_inner)
    print(f"Analytical hoop stress: {analytical_hoop_stress/1e6:.2f} MPa")
    print(f"Error: {(max_hoop_stress - analytical_hoop_stress) / analytical_hoop_stress * 100:.2f}%")


if __name__ == "__main__":
    example_pressure_vessel()