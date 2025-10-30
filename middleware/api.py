import sys
import logging
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from polivem import polivem_py
from domain.models import Beam1DMeshModel, Beam1DModel
from utils.responses import create_response, create_error_response
from utils.geometry import calculate_moment_of_inertia, calculate_area, calculate_extreme_fiber_distances

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.post("/beam1d/solve")
async def post_solve_1d_beam(mesh: dict):
    logger.info(f"Received mesh: {mesh}")
    logger.info(f"Generating beam mesh...")
    try:
        beam = polivem_py.mesh.Beam()
        beam.horizontal_bar_disc(mesh["length"], mesh["num_nodes"])
        mesh_model = Beam1DMeshModel(
            num_nodes=beam.nodes.shape[0],
            length=mesh["length"],
            nodes=beam.nodes,
            elements=beam.elements,
            model_order=mesh["model_order"],
            cross_section=mesh["cross_section"],
            cross_section_params=mesh["cross_section_params"],
            young_modulus=mesh["young_modulus"],
            post_processing_sample_points=mesh.get("post_processing_sample_points", 10)
        )
    except Exception as e:
        logger.error(f"Error generating beam mesh: {e}")
        return create_error_response(
            message=f"Error generating beam mesh: {e}",
            status_code=500
        )
    logger.info(f"Beam mesh generated successfully")
    logger.info("Solving beam...")
    try:
        solver = polivem_py.solver.BeamSolver(
            nodes_coordinates=beam.nodes,
            elements_indices=beam.elements,
            model_order=mesh["model_order"]
        )
        area = calculate_area(mesh_model.cross_section, mesh_model.cross_section_params)
        inertia_moment = calculate_moment_of_inertia(mesh_model.cross_section, mesh_model.cross_section_params)
        Ix = inertia_moment["Ix"]
        logger.info(f"Geometry :: Area: {area}")
        logger.info(f"Geometry :: Ix: {Ix}")
        solver.setArea(area)
        solver.setInertiaMoment(Ix)

        # Set support conditions
        supp = np.zeros((1, 4), dtype=int)
        supp[0, 0] = 0
        supp[0, 1] = 1
        supp[0, 2] = 1
        supp[0, 3] = 0
        solver.setSupp(supp)

        # Build global stiffness matrix
        E = mesh["young_modulus"]
        K = solver.buildGlobalK(E)
        KII = solver.buildStaticCondensation(K, "KII")
        KIM = solver.buildStaticCondensation(K, "KIM")
        KMI = solver.buildStaticCondensation(K, "KMI")
        KMM = solver.buildStaticCondensation(K, "KMM")

        # Set distributed load
        q = np.array([-1.0, -1.0])
        load_indices = beam.elements
        solver.setDistributedLoad(q, load_indices)

        # Build global distributed load
        R = solver.buildGlobalDistributedLoad()
        RI = solver.buildStaticDistVector(R, "RI")
        RM = solver.buildStaticDistVector(R, "RM")

        K_ = solver.condense_matrix(KII, KIM, KMI, KMM)
        R_ = solver.condense_vector(RI, RM, KIM, KMM)

        K_ = solver.applyDBCMatrix(K_)
        R_ = solver.applyDBCVec(R_)

        uh = np.linalg.solve(K_, R_)

        # Reconstruct the full displacement vector (primary + moment DOFs)
        um = -np.linalg.inv(KMM) @ (KMI @ uh + RM)

        # Combine primary and moment DOFs
        u_full = np.zeros(uh.shape[0] + um.shape[0])
        u_full[:uh.shape[0]] = uh
        u_full[uh.shape[0]:] = um

        # Calulate the distance from the neutral axis to the extreme fibers
        extreme_fiber_distances = calculate_extreme_fiber_distances(mesh_model.cross_section, mesh_model.cross_section_params)
        y_top = extreme_fiber_distances["y_top"]

        strain = solver.calculateStrain(
            u_full,
            E, 
            mesh_model.post_processing_sample_points, 
            y_top
        )

        # Calculate stress
        stress = solver.calculateStress(
            u_full, 
            E, 
            mesh_model.post_processing_sample_points, 
            y_top
        )

        # Consolidate the data
        beam_data = Beam1DModel(
            displacements=uh,
            geometry=mesh_model,
            strain=strain,
            stress=stress
        )
        
    except Exception as e:
        logger.error(f"Error solving beam: {e}")
        return create_error_response(
            message=f"Error solving beam: {e}",
            status_code=500
        )
    return create_response(
        data=beam_data,
        message="Mesh received",
        status_code=200
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)