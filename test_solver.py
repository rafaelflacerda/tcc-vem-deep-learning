#!/usr/bin/env python3
# test_solver.py - Test script for the BeamSolver

import sys
import os
import numpy as np
import importlib

from middleware.utils.formatting import print_header, print_success, print_error, print_warning, print_info

# Force reload of the module
if 'polivem' in sys.modules:
    importlib.reload(sys.modules['polivem'])
if 'polivem.polivem_py' in sys.modules:
    importlib.reload(sys.modules['polivem.polivem_py'])

# Import the module
from polivem import polivem_py

# Print available attributes to debug
print("Available in solver module:", [attr for attr in dir(polivem_py.solver) if not attr.startswith('__')])

def test_beam_solver():
    try:
        print_header("TESTING BEAM SOLVER")
        # Test the solver module's test function
        test_result = polivem_py.solver.test_function()
        print(f"Test function result: {test_result}")
        
        # Create a beam mesh
        print("\n=== Creating Beam Mesh ===")
        beam_length = 0.45
        beam = polivem_py.mesh.Beam()
        beam.horizontal_bar_disc(beam_length, 20)  
        print(f"Created beam mesh with {beam.nodes.shape[0]} nodes and {beam.elements.shape[0]} elements")
        print(f"Nodes shape: {beam.nodes.shape}")
        print(f"Elements shape: {beam.elements.shape}")
        
        # Create a beam solver
        print("\n=== Creating BeamSolver ===")
        solver = polivem_py.solver.BeamSolver(
            nodes_coordinates=beam.nodes,
            elements_indices=beam.elements,
            model_order=5
        )
        print(f"Created beam solver: {solver}")
        
        # Set beam properties
        print("\n=== Setting Beam Properties ===")
        # Set moment of inertia (I)
        I = (0.02 * (0.003**3)) / 12
        solver.setInertiaMoment(I)
        print(f"Set inertia moment: {I}")
        
        # Set cross-sectional area
        area = 0.02 * 0.003
        solver.setArea(area)
        print(f"Set cross-sectional area: {area}")
        
        # Set support conditions
        supp = np.zeros((1, 4), dtype=int)
        supp[0, 0] = 0
        supp[0, 1] = 1
        supp[0, 2] = 1
        supp[0, 3] = 0
        solver.setSupp(supp)
        print(f"Set support conditions")

        # Test the buildGlobalK method
        print("\n=== Testing buildGlobalK method ===")
        K = solver.buildGlobalK(2.1e+11)
        print(f"Global stiffness matrix shape: {K.shape}")

        # Test the buildStaticCondensation method
        print("\n=== Testing buildStaticCondensation method ===")
        KII = solver.buildStaticCondensation(K, "KII")
        print(f"Static condensation matrix shape: {KII.shape}")
        KIM = solver.buildStaticCondensation(K, "KIM")
        print(f"Static condensation matrix shape: {KIM.shape}")
        KMI = solver.buildStaticCondensation(K, "KMI")
        print(f"Static condensation matrix shape: {KMI.shape}")
        KMM = solver.buildStaticCondensation(K, "KMM")
        print(f"Static condensation matrix shape: {KMM.shape}")

        # Set distributed load
        q = np.array([-1.0, -1.0])
        load_indices = beam.elements
        solver.setDistributedLoad(q, load_indices)
        print(f"Set distributed load: {q}")

        R = solver.buildGlobalDistributedLoad()
        RI = solver.buildStaticDistVector(R, "RI")
        print(f"Static condensation vector shape: {RI.shape}")
        RM = solver.buildStaticDistVector(R, "RM")
        print(f"Static condensation vector shape: {RM.shape}")

        # K_ = KII - KIM @ np.linalg.inv(KMM) @ KMI
        # print(f"K_ shape: {K_.shape}")
        # R_ = RI - KIM @ np.linalg.inv(KMM) @ RM
        # print(f"R_ shape: {R_.shape}")

        K_ = solver.condense_matrix(KII, KIM, KMI, KMM)
        R_ = solver.condense_vector(RI, RM, KIM, KMM)

        K_ = solver.applyDBCMatrix(K_)
        R_ = solver.applyDBCVec(R_)

        # Solve the system of equations
        print("\n=== Solving the System ===")
        uh = np.linalg.solve(K_, R_)
        print(f"Solution vector shape: {uh.shape}")

        if uh.shape[0] > 0:
            print(f"First few displacement values: {uh[:min(5, uh.shape[0])]}")
        
        # Reconstruct the full displacement vector (primary + moment DOFs)
        print("\n=== Reconstructing full displacement vector ===")
        # Calculate moment displacements: um = -inv(KMM) * (KMI * uh + RM)
        um = -np.linalg.inv(KMM) @ (KMI @ uh + RM)
        print(f"Moment DOFs vector shape: {um.shape}")

        # Combine primary and moment DOFs
        u_full = np.zeros(uh.shape[0] + um.shape[0])
        u_full[:uh.shape[0]] = uh
        u_full[uh.shape[0]:] = um
        print(f"Full displacement vector shape: {u_full.shape}")

        # Calculate strain using the full displacement vector
        print("\n=== Calculating Strain ===")
        strain = solver.calculateStrain(u_full, 2.1e+11, 10, 0.0015)
        print(f"Strain matrix shape: {strain.shape}")
        print(f"First few strain values: {strain[:min(5, strain.shape[0])]}")
        
        print("\n=== Calculating Stress ===")
        stress = solver.calculateStress(u_full, 2.1e+11, 10, 0.0015)
        print(f"Stress matrix shape: {stress.shape}")
        print(f"First few stress values: {stress[:min(5, stress.shape[0])]}")

        # Calculate stress at 1/4, 1/2, and 3/4 of the beam length
        for i in [0.25, 0.5, 0.75]:
            x_global = beam_length * i
            strain_at_point, stress_at_point = solver.getStrainStressAtPoint(u_full, 2.1e+11, x_global, 0.0015)
            print(f"Strain at {x_global} m: {strain_at_point}")
            print(f"Stress at {x_global} m: {stress_at_point}")

        print("\n=== Test Completed Successfully ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_linear_elastic_solver():
    print_header("TESTING LINEAR ELASTIC SOLVER")

    # Create a mesh with 9 nodes and 4 quadrilateral elements - same as C++ version
    nodes = np.array([
        [0.0, 0.0],  # Node 0
        [0.5, 0.0],  # Node 1
        [0.5, 0.5],  # Node 2
        [0.0, 0.5],  # Node 3
        [1.0, 0.0],  # Node 4
        [1.0, 0.5],  # Node 5
        [1.0, 1.0],  # Node 6
        [0.5, 1.0],  # Node 7
        [0.0, 1.0]   # Node 8
    ])
    
    elements = np.array([
        [0, 1, 2, 3],  # Element 0
        [1, 4, 5, 2],  # Element 1
        [2, 5, 6, 7],  # Element 2
        [3, 2, 7, 8]   # Element 3
    ])

    # Create the solver with order 1
    elastic_solver = polivem_py.solver.LinearElastic2DSolver(nodes, elements, 1)

    # Define support conditions - MATCH C++ VERSION
    supports = np.array([
        [0, 1, 1],  # Node 0: fixed in x and y
        [1, 0, 1],  # Node 1: fixed in y only
        [4, 0, 1],  # Node 4: fixed in y only
        [3, 1, 0],  # Node 3: fixed in x only
        [8, 1, 0]   # Node 8: fixed in x only
    ])
    elastic_solver.setSupp(supports)

    # Define load on the right edge
    loads = np.array([
        [4, 5],  # Edge from node 4 to node 5
        [5, 6]   # Edge from node 5 to node 6
    ])
    elastic_solver.setLoad(loads)

    # Create material
    mat = polivem_py.material.Material()
    mat.setElasticModule(7000)  # Young's modulus
    mat.setPoissonCoef(0.3)     # Poisson's ratio

    # Get elasticity matrix
    C = mat.build2DElasticity()
    print(f"Elasticity matrix C:\n{C}")
    print("C(0,0) =", C[0,0])
    print("C(0,1) =", C[0,1])
    print("C(1,0) =", C[1,0])
    print("C(1,1) =", C[1,1])
    print("C(2,2) =", C[2,2])

    # Build global stiffness matrix
    K = elastic_solver.buildGlobalK(C)
    print(f"Global stiffness matrix shape: {K.shape}")
    print(f"K contains NaN: {np.isnan(K).any()}")
    print(f"K contains Inf: {np.isinf(K).any()}")
    print(f"Min value in K: {np.min(K)}")
    print(f"Max value in K: {np.max(K)}")

    # Apply boundary conditions
    K_constrained = elastic_solver.applyDBC(K)
    print(f"K_constrained contains NaN: {np.isnan(K_constrained).any()}")
    print(f"K_constrained contains Inf: {np.isinf(K_constrained).any()}")
    print(f"Min value in K_constrained: {np.min(K_constrained)}")
    print(f"Max value in K_constrained: {np.max(K_constrained)}")

    # Apply loads
    qx = 2000.0  # Force in x direction
    qy = 0.0     # Force in y direction
    f = elastic_solver.applyNBC(qx, qy)
    print(f"Force vector shape: {f.shape}")
    print(f"Force vector:\n{f}")
    print(f"Min value in f: {np.min(f)}")
    print(f"Max value in f: {np.max(f)}")
    print(f"Sum of f: {np.sum(f)}")

    # Try different solvers
    try:
        # Standard solve
        u = np.linalg.solve(K_constrained, f)
        print(f"Standard solve - Max displacement: {np.max(np.abs(u))}")
        print(f"Displacement vector:\n{u}")
        
        # Try scaling the problem
        scale_factor = 1e-3  # Scale down by 1000
        K_scaled = K_constrained * scale_factor
        f_scaled = f * scale_factor
        u_scaled = np.linalg.solve(K_scaled, f_scaled)
        print(f"Scaled solve - Max displacement: {np.max(np.abs(u_scaled))}")
        
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}")
        
        # Try least squares solve
        u, residuals, rank, s = np.linalg.lstsq(K_constrained, f, rcond=None)
        print(f"Least squares solve - Max displacement: {np.max(np.abs(u))}")
        print(f"Displacement vector:\n{u}")

if __name__ == "__main__":
    test_linear_elastic_solver()