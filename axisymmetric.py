import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Node:
    id: int
    r: float
    z: float

@dataclass
class Element:
    id: int
    nodes: List[Node]

class AxiSymmetricFEMSolver:
    def __init__(self, E: float, nu: float):
        self.E = E
        self.nu = nu
        self.nodes = []
        self.elements = []

    def add_node(self, id: int, r: float, z: float) -> None:
        self.nodes.append(Node(id, r, z))

    def add_element(self, id: int, node_ids: List[int]) -> None:
        nodes = [self.nodes[i-1] for i in node_ids]
        self.elements.append(Element(id, nodes))

    def compute_B_matrix(self, element: Element) -> tuple:
        """Compute B matrix according to equation (9.2.3)"""
        i, j, m = element.nodes

        # Calculate parameters
        alpha_i = i.r * m.z - m.r * i.z
        alpha_j = j.r * i.z - i.r * j.z
        alpha_m = m.r * j.z - j.r * m.z

        beta_i = j.z - m.z
        beta_j = m.z - i.z
        beta_m = i.z - j.z

        gamma_i = m.r - j.r
        gamma_j = i.r - m.r
        gamma_m = j.r - i.r

        # Calculate centroid and area
        r_bar = (i.r + j.r + m.r) / 3.0
        z_bar = (i.z + j.z + m.z) / 3.0
        A = abs(0.5 * (i.r * (j.z - m.z) + j.r * (m.z - i.z) + m.r * (i.z - j.z)))

        # Construct B matrix
        B = np.zeros((4, 6))
        B[0] = [beta_i/A, 0, beta_j/A, 0, beta_m/A, 0]
        B[1] = [0, gamma_i/A, 0, gamma_j/A, 0, gamma_m/A]
        B[2] = [(alpha_i/r_bar + beta_i)/A, 0, (alpha_j/r_bar + beta_j)/A, 0, (alpha_m/r_bar + beta_m)/A, 0]
        B[3] = [gamma_i/A, beta_i/A, gamma_j/A, beta_j/A, gamma_m/A, beta_m/A]

        return B, r_bar, A

    def compute_D_matrix(self) -> np.ndarray:
        """Compute D matrix according to equation (9.2.8)"""
        factor = self.E / ((1.0 + self.nu) * (1.0 - 2.0*self.nu))
        D = np.array([
            [1.0-self.nu, self.nu, self.nu, 0],
            [self.nu, 1.0-self.nu, self.nu, 0],
            [self.nu, self.nu, 1.0-self.nu, 0],
            [0, 0, 0, (1.0-2.0*self.nu)/2.0]
        ])
        return factor * D

    def compute_element_stiffness(self, element: Element) -> np.ndarray:
        """Compute element stiffness matrix according to equation (9.2.2)"""
        B, r_bar, A = self.compute_B_matrix(element)
        D = self.compute_D_matrix()
        BTD = B.T @ D
        return 2.0 * np.pi * r_bar * A * (BTD @ B)

    def assemble_global_stiffness(self) -> np.ndarray:
        """Assemble global stiffness matrix"""
        n_dof = 2 * len(self.nodes)
        K = np.zeros((n_dof, n_dof))

        for element in self.elements:
            k_e = self.compute_element_stiffness(element)
            dofs = []
            for node in element.nodes:
                dofs.extend([2*(node.id-1), 2*(node.id-1)+1])
            
            for i in range(6):
                for j in range(6):
                    K[dofs[i], dofs[j]] += k_e[i, j]

        return K

    def solve(self, forces: np.ndarray, fixed_dofs: List[int]) -> np.ndarray:
        """Solve the system"""
        fixed_dofs = [i-1 for i in fixed_dofs]
        K = self.assemble_global_stiffness()
        
        # Handle boundary conditions
        free_dofs = list(set(range(len(forces))) - set(fixed_dofs))
        K_free = K[np.ix_(free_dofs, free_dofs)]
        F_free = forces[free_dofs]
        
        # Solve system
        u_free = np.linalg.solve(K_free, F_free)
        
        # Reconstruct full solution
        u = np.zeros_like(forces)
        u[free_dofs] = u_free
        
        return u

def solve_cylinder_problem():
    # Initialize solver
    E = 30.0e6  # Young's modulus (psi)
    nu = 0.3    # Poisson's ratio
    solver = AxiSymmetricFEMSolver(E, nu)
    
    # Add nodes (from Figure 9-10)
    nodes = [
        (1, 0.5, 0.0),  # Node 1
        (2, 1.0, 0.0),  # Node 2
        (3, 1.0, 0.5),  # Node 3
        (4, 0.5, 0.5),  # Node 4
        (5, 0.75, 0.25) # Node 5 (center)
    ]
    
    for id, r, z in nodes:
        solver.add_node(id, r, z)
    
    # Add elements (clockwise ordering)
    elements = [
        (1, [1, 2, 5]),  # Element 1
        (2, [2, 3, 5]),  # Element 2
        (3, [3, 4, 5]),  # Element 3
        (4, [4, 1, 5])   # Element 4
    ]
    
    for id, node_ids in elements:
        solver.add_element(id, node_ids)

    # Applied forces (equation 9.2.16)
    # F1r = F4r = 2π(0.5)(0.5)/2 = 0.785 lb
    n_nodes = len(solver.nodes)
    forces = np.zeros(2 * n_nodes)
    forces[0] = 0.785  # F1r
    forces[6] = 0.785  # F4r

    # Fixed DOFs (w5 = 0 due to symmetry)
    fixed_dofs = [10]  # Node 5, vertical displacement
    
    # Solve the system
    displacements = solver.solve(forces, fixed_dofs)
    
    # Expected results from equation (9.2.17)
    expected = {
        'u1': 0.0322e-6, 'w1': 0.00115e-6,
        'u2': 0.0219e-6, 'w2': 0.00206e-6,
        'u3': 0.0219e-6, 'w3': -0.00206e-6,
        'u4': 0.0322e-6, 'w4': -0.00115e-6,
        'u5': 0.0244e-6, 'w5': 0.0
    }
    
    print("\nNodal Displacements (compared with expected values):")
    for i in range(n_nodes):
        node_num = i + 1
        u = displacements[2*i]
        w = displacements[2*i+1]
        u_exp = expected[f'u{node_num}']
        w_exp = expected[f'w{node_num}']
        
        print(f"\nNode {node_num}:")
        print(f"  u = {u:.6e} in  (Expected: {u_exp:.6e} in)")
        print(f"  w = {w:.6e} in  (Expected: {w_exp:.6e} in)")
        
        # Print relative error
        if abs(u_exp) > 1e-20:  # Avoid division by zero
            u_error = abs((u - u_exp)/u_exp) * 100
            print(f"  u relative error: {u_error:.2f}%")
        if abs(w_exp) > 1e-20:
            w_error = abs((w - w_exp)/w_exp) * 100
            print(f"  w relative error: {w_error:.2f}%")

if __name__ == "__main__":
    solve_cylinder_problem()