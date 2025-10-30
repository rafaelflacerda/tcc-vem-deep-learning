/**
 * @file axisymmetric.hpp
 * @brief Defines the axisymmetric solver class
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#ifndef POLIVEM_AXISYMMETRIC_HPP
#define POLIVEM_AXISYMMETRIC_HPP

#include <iostream>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "utils/operations.hpp"
#include "material/mat.hpp"

const double VERTICAL_TOLERANCE = 1e-10;

namespace solver {
    class axisymmetric {
        public:
            /**
            * @brief Define the base strain vectors for axisymmetric elasticity
            * 
            * These vectors represent the unit strain states in axisymmetric elasticity:
            * - Radial strain (εr)
            * - Axial strain (εz)
            * - Hoop/circumferential strain (εθ)
            * - Shear strain (γrz)
            * 
            * @return std::vector<Eigen::Vector4d> A vector of four strain vectors
            */
            static std::vector<Eigen::Vector4d> define_base_strain_vectors();

            /**
             * @brief Compute the traction vector t = C·eps_p·n for a given strain state and normal vector
             * 
             * This function calculates the traction vector at a boundary with normal vector n,
             * for a given constitutive matrix C and strain state eps_p.
             * 
             * @param C 4x4 constitutive matrix
             * @param eps_p Base strain vector [eps_r, eps_z, eps_theta, eps_rz]
             * @param normal Normal vector components (n_r, n_z)
             * @return std::pair<double, double> Traction vector components (t_r, t_z)
             */
            static std::pair<double, double> compute_traction_vector(
                const Eigen::Matrix4d& C,
                const Eigen::Vector4d& eps_p,
                const std::pair<double, double>& normal
            );

            /** 
             * @brief Perform Gauss quadrature to compute the boundary integral of v_h · t · r along an edge.
             * 
             * This function evaluates the integral ∫_Γ v_h · t · r dΓ for a given edge,
             * where v_h is the virtual displacement, t is the traction vector, and r is the
             * radial coordinate. It uses either 1-point or 2-point Gauss quadrature depending
             * on whether the edge is vertical.
             * 
             * @param edge_vertices Array of shape (2,2) containing the coordinates
             * of the two vertices of the edge [(r1,z1), (r2,z2)]
             * @param traction_vector Components (t_r, t_z) of the traction vector
             * @param displacement_func Function that returns the virtual displacement
             * (v_r, v_z) at a point along the edge parameterized by s ∈ [0,1]
             * @param is_vertical Flag indicating if the edge is vertical. Defaults to False.
             * @return double The value of the boundary integral
            */
           static double gauss_quadrature_boundary_integral(
                const Eigen::Matrix2d& edge_vertices,
                const std::pair<double, double>& traction_vector,
                const std::function<std::pair<double, double>(double)>& displacement_func,
                bool is_vertical = false
           );

            /**
             * @brief Create a function that returns the virtual displacement at any point along an edge.
             * 
             * This function creates a callable that computes the displacement field associated with
             * a single degree of freedom, evaluating it at any point along the edge parameterized by s.
             * 
             * @param edge_vertices Matrix of shape (2,2) containing edge vertices [(r1,z1), (r2,z2)]
             * @param vertex_indices Pair of global indices of the edge vertices (i, j)
             * @param dof_index Index of the DOF to activate (0-based)
             * @param dof_value Value to assign to the DOF (defaults to 1.0)
             * @return std::function returning pair of displacement components (v_r, v_z) for parameter s ∈ [0,1]
             */
            static std::function<std::pair<double, double>(double)> create_displacement_function(
                const Eigen::Matrix2d& edge_vertices,
                const std::pair<int, int>& vertex_indices,
                int dof_index,
                double dof_value = 1.0
            );

            /**
             * @brief Compute the volumetric correction term for each DOF and base strain for arbitrary convex polygons.
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @param C 4x4 constitutive matrix
             * @param base_strains List of base strain vectors
             * @return Eigen::MatrixXd Matrix of shape (n_dofs, n_strains) containing the volumetric correction values for each DOF and each base strain
             */
            static Eigen::MatrixXd compute_volumetric_correction(
                const Eigen::MatrixXd& element_vertices,
                const Eigen::Matrix4d& C,
                const std::vector<Eigen::Vector4d>& base_strains
            );

            /**
             * @brief Compute the boundary integral contributions for each DOF in an element for all base strain vectors.
             * 
             * This function evaluates the boundary term ∫_∂E v_h · (C·eps_p·n) · r dΓ for each DOF
             * and each base strain vector. This is part of the projection operator computation in VEM.
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @param C 4x4 constitutive matrix
             * @param base_strains List of base strain vectors
             * @return Eigen::MatrixXd Matrix of shape (n_dofs, n_strains) containing the boundary integral values for each DOF and each base strain
             */
            static Eigen::MatrixXd compute_element_boundary_integrals(
                const Eigen::MatrixXd& element_vertices,
                const Eigen::Matrix4d& C,
                const std::vector<Eigen::Vector4d>& base_strains
            );

            /**
             * @brief Compute the coefficient matrix for the projection system.
             * 
             * This function constructs the matrix used in solving the projection system
             * for the strain projection operator in VEM.
             * 
             * @param C 4x4 constitutive matrix
             * @param eps_matrix Matrix whose columns are the base strain vectors
             * @param weighted_volume The weighted volume ∫_E r dV of the element
             * @return Eigen::MatrixXd Matrix of shape (n_dofs, n_strains) containing the volumetric correction values for each DOF and each base strain
             */
            static Eigen::MatrixXd compute_proj_system_matrix(
                const Eigen::Matrix4d& C,
                const Eigen::MatrixXd& eps_matrix,
                double weighted_volume
            );
            
            /**
             * @brief Compute the projection matrix B that maps nodal displacements to projected strains.
             * 
             * This function computes the strain projection matrix B such that Π(vh) = B·d,
             * where Π(vh) is the projected strain, and d is the vector of nodal displacements.
             * The projection ensures that the VEM approximation correctly reproduces the strain
             * energy for polynomial displacement fields.
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @param C 4x4 constitutive matrix
             * @param base_strains List of base strain vectors
             * @return Eigen::MatrixXd Matrix of shape (n_strains, n_dofs) containing the projection matrix B
             */
            static Eigen::MatrixXd compute_projection_matrix(
                const Eigen::MatrixXd& element_vertices,
                const Eigen::Matrix4d& C,
                const std::vector<Eigen::Vector4d>& base_strains
            );

            /**
             * @brief Triangulate the polygon and compute the weighted volume.
             * 
             * This function computes the weighted volume of a polygon by triangulating it
             * and summing the volumes of the resulting triangles, weighted by the centroid
             * of each triangle.
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @return double The weighted volume of the polygon
             */
            static double compute_weighted_volume_polygon(
                const Eigen::MatrixXd& element_vertices
            );

            /**
             * @brief Compute the VEM stiffness matrix for an axisymmetric element.
             * 
             * This function builds the complete stiffness matrix for a virtual element, 
             * including both the consistency term (K_c) and the stabilization term (K_s).
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @param E Young's modulus
             * @param nu Poisson's ratio
             * @param stab_type Type of stabilization to use ("standard", "divergence", "boundary")
             * @return Eigen::MatrixXd Matrix of shape (n_dofs, n_dofs) containing the complete stiffness matrix
             */
            static Eigen::MatrixXd compute_stiffness_matrix(
                const Eigen::MatrixXd& element_vertices,
                double E,
                double nu,
                const std::string& stab_type = "standard"
            );

            /**
             * @brief Compute the boundary integral for divergence projection via integration by parts.
             * 
             * This function computes the boundary integral for divergence projection via integration by parts:
             * ∫_E div(u_h) · 2πr dr dz = ∫_∂E 2πr u_h · n ds
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @param displacements Array of displacement values [u_r1, u_z1, u_r2, u_z2, ...]
             * @return double The value of the boundary integral for divergence projection
             */
            static double compute_divergence_boundary_integral(
                const Eigen::MatrixXd& element_vertices,
                const Eigen::VectorXd& displacements
            );



            // === LOADS ===
            /**
             * @brief Compute the equivalent nodal forces for a given strain state in axisymmetric elasticity.
             * 
             * In axisymmetric problems, certain strain states generate body forces due to 
             * the geometric nonlinearity. This function computes the equivalent nodal forces
             * for a given strain state, accounting for the difference between radial and hoop stresses.
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @param C 4x4 constitutive matrix
             * @param strain_state The strain state vector [εr, εz, εθ, γrz]
             * @return Eigen::VectorXd Vector of equivalent nodal forces of shape (n_dofs,)
             */
            static Eigen::VectorXd compute_equivalent_body_force(
                const Eigen::MatrixXd& element_vertices,
                const Eigen::Matrix4d& C,
                const Eigen::Vector4d& strain_state
            );

            /**
             * @brief Compute the element load vector for body forces in axisymmetric VEM.
             * 
             * TODO: Implement traingulation for general polygons
             * 
             * This function computes the equivalent nodal forces for a given traction on the boundary.
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @params body_force_vector Vector of body forces of shape (n_dofs,)
             * @return Eigen::VectorXd Vector of equivalent nodal forces of shape (n_dofs,)
             */
            static Eigen::VectorXd compute_element_load_body_force(
                const Eigen::MatrixXd& element_vertices,
                const std::function<std::pair<double, double>(double, double)>& body_force_func
            );


            /**
             * @brief Compute the element load vector for boundary tractions in axisymmetric VEM.
             * 
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates
             * @param edge_indices List of edge indices where traction is applied [i, j, k, ...]
             * @param traction_func Function that returns traction (t_r, t_z) at a point (r,z)
             * @return Eigen::VectorXd Vector of equivalent nodal forces of shape (n_dofs,)
             */
            static Eigen::VectorXd compute_element_load_boundary_traction(
                const Eigen::MatrixXd& element_vertices,
                const Eigen::VectorXi& edge_indices,
                const std::function<std::pair<double, double>(double, double)>& traction_func
            );


            /**
             * @brief Assemble the complete element load vector for axisymmetric VEM.
             * 
             * TODO: Implement this function
             * 
             * @param element_vertices Array of vertex coordinates with shape (n_vertices, 2) where each row contains (r, z) coordinates    
             * @param body_force_func Function that returns body force (b_r, b_z) at a point (r,z)
             * @param traction_edges List of edge indices where traction is applied [i, j, k, ...]
             * @param traction_func Function that returns traction (t_r, t_z) at a point (r,z)
             * @return Eigen::VectorXd Vector of equivalent nodal forces of shape (n_dofs,)
             */
            static Eigen::VectorXd assemble_element_load_vector(
                const Eigen::MatrixXd& element_vertices,
                const std::function<std::pair<double, double>(double, double)>& body_force_func,
                const Eigen::VectorXi& traction_edges,
                const std::function<std::pair<double, double>(double, double)>& traction_func
            );
        

            // === STABILIZATION TERMS ===
            static Eigen::MatrixXd compute_divergence_stabilization_matrix(
                const Eigen::MatrixXd& element_vertices,
                double E,
                double nu
            );

            static Eigen::MatrixXd compute_boundary_stabilization_matrix(
                const Eigen::MatrixXd& element_vertices,
                const Eigen::MatrixXd& P,
                double E,
                double nu
            );
            
    };
}

#endif