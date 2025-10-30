/**
 * @file parabolic.hpp
 * @brief Defines the parabolic solver class
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#ifndef POLIVEM_SOLVER_PARABOLIC_HPP
#define POLIVEM_SOLVER_PARABOLIC_HPP

#include <iostream>
#include <cmath>
#include <vector>
#include <map>
#include <set>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "utils/operations.hpp"
#include "models/templates.hpp"
#include "utils/integration.hpp"
namespace solver {
    class parabolic {
        public:
            // VEM Order (polynomial degree)
            int order;

            // Dimension of the polynomial space
            int N_k;

            // Displacement restrictions
            Eigen::MatrixXi supp;

            // Nodes (contains the coordinates)
            Eigen::MatrixXd nodes;

            // Elements (contains the indices)
            Eigen::MatrixXi elements;

            // Global system matrices
            Eigen::SparseMatrix<double> M_h; // Mass matrix
            Eigen::SparseMatrix<double> K_h; // Stiffness matrix

            // Source term function type
            using SourceFunction = std::function<double(const Eigen::Vector2d&, double)>;

            // Global load vector
            Eigen::VectorXd F_h;

            // Degrees of freedom
            int n_dofs;
            std::vector<std::vector<int>> element_dof_map;

            // Global solution vector
            Eigen::VectorXd U_h;

            // Constructor
            parabolic(
                int order, 
                Eigen::MatrixXd nodes_coordinates, 
                Eigen::MatrixXi elements_indices,
                bool use_lumped_mass_matrix,
                bool debug_mode = false
            ) : order(order), nodes(nodes_coordinates), elements(elements_indices), use_lumped_mass_matrix_(use_lumped_mass_matrix), debug_mode_(debug_mode) {

                if (order == 1) {
                    N_k = 3; // {1, x, y}
                }  else if (order == 2){
                    N_k = 6; // {1, x, y, x^2, xy, y^2}
                } else {
                    N_k = (order + 1) * (order + 2) / 2; // {1, x, y, x^2, xy, y^2, ..., x^order, y^order}
                }

                // Initialize the geometry
                nodes = nodes_coordinates;
                elements = elements_indices;

                // Initialize element DOF map vector
                element_dof_map.resize(elements.rows());

                // Setup DOF numbering for entire mesh
                setup_global_dofs();

                // Initialize global matrices
                initialize_global_matrices();

                if (debug_mode_) {
                    print_system_information();
                }
            }

            /**
             * @brief Sets up the global degree of freedom numbering for the entire mesh.
             * 
             * This function initializes the global DOF numbering for the entire mesh,
             * including linear and high-order DOFs.
             */
            void setup_global_dofs();

            // Get global matrices
            const Eigen::SparseMatrix<double>& get_global_mass_matrix() const { return M_h; }
            const Eigen::SparseMatrix<double>& get_global_stiffness_matrix() const { return K_h; }

            /**
             * @brief Get the number of degrees of freedom
             * @return Number of DOFs in the system
             */
            int get_n_dofs() const { return n_dofs; }

            /**
             * @brief Get the global load vector
             * @return Reference to the global load vector
             */
            const Eigen::VectorXd& get_load_vector() const { return F_h; }

            /**
             * @brief Assemble the complete VEM system (mass and stiffness matrices)
             * 
             * This method assembles all element contributions into the global system matrices.
             * It processes each element in the mesh and accumulates local matrices into 
             * the global sparse matrices M_h and K_h.
             */
            void assemble_system();

            // ============================================================================
            // SETUP DOFS
            // ============================================================================
            /**
             * @brief Sets up the linear degree of freedom numbering for the entire mesh.
            */
            void setup_linear_dofs();

            /**
             * @brief Sets up the high-order degree of freedom numbering for the entire mesh.
            */
            void setup_high_order_dofs();

            /** 
             * @brief Constructs the local-to-global degree of freedom mapping for a single element.
             * 
             * @param elem Index of the element to build the DOF map for
             * @param edge_dof_map Map of edge DOF indices
             * @param element_moments_dof_map Map of element moment DOF indices
            */
            void build_element_dof_map(
                int elem, 
                const std::map<std::pair<int, int>, int>& edge_dof_map, 
                const std::map<int, int>& element_moments_dof_map
            );

            /**
             * @brief Counts the number of local degrees of freedom for a given element.
             * 
             * @param n_vertices Number of vertices in the element
             * @return int Number of local degrees of freedom
            */
            int count_local_dofs(int n_vertices){
                if (order == 1) {
                    return n_vertices;
                } else {
                    // CRITICAL FIX: For k=2, we need 1 interior DOF (area functional)
                    // The formula (order-1)*(order-2)/2 gives 0 for k=2, but we need 1
                    int vertex_dofs = n_vertices;
                    int edge_dofs = n_vertices * (order - 1);
                    int interior_dofs;
                    
                    if (order == 2) {
                        interior_dofs = 1;  // One area functional: (1/|E|) ∫_E v dA
                    } else {
                        interior_dofs = (order - 1) * (order - 2) / 2;  // General formula for k≥3
                    }
                    
                    return vertex_dofs + edge_dofs + interior_dofs;
                }
            }

            // ============================================================================
            // MATRIX MANIPULATION
            // ============================================================================
            void initialize_global_matrices(){
                M_h.resize(n_dofs, n_dofs);
                K_h.resize(n_dofs, n_dofs);
                F_h.resize(n_dofs);
            };

            void assemble_element(int idx);

            void setup_element_geometry(int elem_idx, ElementData& element_data);

            void construct_monomial_basis(ElementData& element_data);

            void compute_polynomial_matrices(ElementData& element_data);

            void setup_quadrature_rule(
                ElementData& element_data, 
                std::vector<Eigen::Vector2d>& quad_points, 
                std::vector<double>& quad_weights
            );

            /**
             * @brief Compute the energy projection for a given element.
             * 
             * The energy projection solves: a_K(Π^∇φi, mj) = a_K(φi, mj) for all monomials mj
             * The LHS is given by K_poly(i,j) = a_K(mi, mj) = ∫_K ∇mi · ∇mj dx
             * 
             * For each DOF φi, we need to solve: K_poly * p_i = rhs
             * where rhs = [a_K(φi, m0), a_K(φi, m1), ..., a_K(φi, m_{N_k-1})]
             * 
             * @param element_data Element data structure containing element information
             * @return Eigen::VectorXd Energy projection for each DOF
             */
            void compute_energy_projection(ElementData& element_data);
            
            /**
             * @brief Compute the local stiffness matrix for a given element.
             * 
             * VEM formula: A_K = (P^∇)ᵀ · K_poly · P^∇ + α_K · S_K
             * 
             * @param element_data Element data structure containing element information
             * @return Eigen::MatrixXd Local stiffness matrix
            */
            void compute_local_stiffness_matrix(ElementData& element_data);

            /**
             * @brief Compute the local mass matrix for a given element.
             * 
             * VEM formula: M_K = (P^0)ᵀ · M_poly · P^0 + β_K · |K| · S_K
             * 
             * @param element_data Element data structure containing element information
            */
            void compute_local_mass_matrix(ElementData& element_data);

            /**
             * @brief Assemble the local matrices into the global system.
             * 
             * @param element_data Element data structure containing element information
            */
            void assemble_local_matrices(ElementData& element_data);

            /**
             * @brief Assemble the local matrices into the global system.
             * 
             * @param element_idx Index of the element
             * @param element_data Element data structure containing element information
            */
            void assemble_to_global_system(int element_idx,  ElementData& element_data);

            /**
             * @brief Compute the local load vector for a given element.
             * 
             * @param element_data Element data structure containing element information
             * @param f Source function
             * @param time Current time
             * @return Eigen::VectorXd Local load vector
            */
            void compute_local_load_vector(ElementData& element_data, const SourceFunction& f, double time);

            /**
             * @brief Assemble the local load vector into the global load vector.
             * 
             * @param element_idx Index of the element
             * @param element_data Element data structure containing element information
            */
            void assemble_local_to_global_load(int element_idx, ElementData& element_data);

            /**
             * @brief Assemble the load vector for a given element.
             * 
             * @param f Source function
             * @param time Current time
            */
            void assemble_load_vector(const SourceFunction& f, double time);

            // ============================================================================
            // LHS AND RHS METHODS
            // ============================================================================

            /**
             * @brief Compute the projection matrices for a given element.
             * 
             * @param element_data Element data structure containing element information
            */
            void compute_projection_matrices(ElementData& element_data);

            /**
             * @brief Compute the energy right-hand side for a given DOF.
             * 
             * For each monomial mj, compute a_K(φi, mj) using Green's formula:
             * a_K(φi, mj) = ∫_∂K φi (∇mj·n) ds - ∫_K φi Δmj dx
             * 
             * @param element_data Element data structure containing element information
             * @param dof_idx Index of the DOF
             * @return Energy right-hand side for the given DOF
            */
            Eigen::VectorXd compute_energy_rhs_for_dof(const ElementData& element_data, int dof_idx);

            /**
             * @brief Add boundary constraint (normalization constraiont) for k = 1.
             * 
             * For k=1, add constraint: ∫_∂K (φi - Π^∇φi) ds = 0,
             * where Π^∇φi = p0 + p1*x + p2*y (for k=1)
             * This ensures uniqueness of the constant part
             * 
             * @param element_data Element data structure containing element information
             * @param dof_idx Index of the DOF
             * @param A_constrained Matrix of the constrained system
             * @param b_constrained Right-hand side of the constrained system
             * @param rhs_i Right-hand side of the unconstrained system
            */
            void add_boundary_constraint(
                ElementData& element_data,
                int dof_idx,
                Eigen::MatrixXd& A_constrained,
                Eigen::VectorXd& b_constrained,
                const Eigen::VectorXd& rhs_i
            );

            /**
             * @brief Add interior constraint for k≥2.
             * 
             * For k>=2, add constraint: ∫_K (φi - Π^∇φi) dx = 0
             * 
             * @param element_data Element data structure containing element information
             * @param dof_idx Index of the DOF
             * @param A_constrained Matrix of the constrained system
             * @param b_constrained Right-hand side of the constrained system
             * @param rhs_i Right-hand side of the unconstrained system
            */
            void add_interior_constraint(
                ElementData& element_data,
                int dof_idx,
                Eigen::MatrixXd& A_constrained,
                Eigen::VectorXd& b_constrained,
                const Eigen::VectorXd& rhs_i
            );

            /**
             * @brief Compute the L2 projection for a given element.
             * 
             * Compute ∫_K v_h m dx = ∫_K Π^∇v_h m dx 
             * for all m ∈ P_{k-1} ∪ P_k
             * 
             * @param element_data Element data structure containing element information
            */
            void compute_l2_projection(ElementData& element_data);

            // ============================================================================
            // STABILIZATION COMPUTATION
            // ============================================================================
            /**
             * @brief Compute the stabilization parameter for a given element.
             * 
             * The trace is the sum of diagonal terms → roughly measures the magnitude of the matrix.
             * Scaling by N_k normalizes for the number of modes.
             * 
             * @param element_data Element data structure containing element information
             * @return double Stabilization parameter
            */
            double compute_stiffness_stabilization_parameter(const ElementData& element_data);

            /**
             * @brief Compute the mass stabilization parameter for a given element.
             * 
             * The mass matrix is sensitive to the size of the element.
             * Including the area ensures that mass stabilization is consistent across differently sized elements.
             * 
             * @param element_data Element data structure containing element information
             * @return double Stabilization parameter
            */
            double compute_mass_stabilization_parameter(const ElementData& element_data);

            /**
             * @brief Compute the stabilization matrix for a given element.
             * 
             * Stabilization matrix: S_K = I - P_proj where P_proj projects onto polynomial space
             * This penalizes the non-polynomial part of the virtual functions
             * 
             * @param element_data Element data structure containing element information
             * @return Eigen::MatrixXd Stabilization matrix
            */
            Eigen::MatrixXd compute_stabilization_matrix(const ElementData& element_data);


            // ============================================================================
            // MANIPULATION OF SOURCE TERMS
            // ============================================================================

            /**
             * @brief Compute the projected source term for a given element.
             * 
             * Compute L2 projection: find f_h in P_{projection_degree} such that
             * (f_h, m) = (f, m) for all monomials m in P_{projection_degree}
             * 
             * @param element_data Element data structure containing element information
             * @param f Source function
             * @param time Current time
             * @param fh_coeffs Coefficients of the projected source term
            */
            void compute_projected_source(
                ElementData& element_data,
                const SourceFunction& f,
                double time,
                Eigen::VectorXd& fh_coeffs
            );

            // ============================================================================
            // INITIAL CONDITIONS
            // ============================================================================
            
            /**
             * @brief Type alias for initial condition functions.
             * 
             * This type defines a function that takes a 2D point and returns a double value.
             * It is used to specify the initial condition for the problem.
             */
            using InitialFunctions = std::function<double(const Eigen::Vector2d&)>;

            /**
             * @brief Set the initial conditions for the problem using VEM interpolation.
             * 
             * This method computes the VEM interpolant I_h u0 by applying DOF functionals
             * χ_i(u0) to obtain the DOF vector. The VEM interpolant u_{h,0} ∈ W_h is then
             * uniquely determined by χ_i(u_{h,0}) = χ_i(u0) for all i = 1, ..., N_dof.
             * 
             * This ensures optimal convergence: ||u_{h,0} - u0||_1 ≤ Ch^k |u0|_{k+1}
             * 
             * @param u0 Initial condition function.
             */
            void set_initial_conditions(const InitialFunctions& u0);

            // ============================================================================
            // DEBUG METHODS
            // ============================================================================ 
            void print_system_information() {
                std::cout << "==== VEM Parabolic Solver Info ====" << std::endl;
                std::cout << "Polynomial order: " << order << std::endl;
                std::cout << "Number of DOFs: " << n_dofs << std::endl;
                std::cout << "Number of elements: " << elements.rows() << std::endl;
                std::cout << "===================================" << std::endl;
                std::cout << std::endl;
            }

            bool get_debug_mode() const { return debug_mode_; }
            
            // DEBUG: Get global DOF indices (only populated when debug_mode_ = true)
            const std::vector<std::pair<int, std::vector<int>>>& get_global_vertex_dof_indices() const { return global_vertex_dof_indices_ ; }
            const std::vector<std::pair<int, std::vector<int>>>& get_global_edge_dof_indices() const { return global_edge_dof_indices_ ; }
            const std::vector<std::pair<int, std::vector<int>>>& get_global_interior_dof_indices() const { return global_interior_dof_indices_ ; }

            // ============================================================================
            // GETTERS
            // ============================================================================ 

            // Get global DOF counts (only populated when debug_mode_ = true)
            int get_total_n_dof_vertex() const { return total_n_dof_vertex_; }
            int get_total_n_dof_edge() const { return total_n_dof_edge_; }
            int get_total_n_dof_interior() const { return total_n_dof_interior_; }

            /**
             * @brief Get the L² projection matrix for a specific element
             * @param element_idx Element index
             * @return P_0 matrix for the element
             */
            const Eigen::MatrixXd& get_element_P0_matrix(int element_idx) const;

            /**
             * @brief Get the energy projection matrix for a specific element
             * @param element_idx Element index
             * @return P_nabla matrix for the element
             */
            const Eigen::MatrixXd& get_element_P_nabla_matrix(int element_idx) const;

            /**
             * @brief Get cached element data for a specific element
             * @param element_idx Element index
             * @return ElementData for the element
             */
            const ElementData& get_element_data(int element_idx) const;

            /**
             * @brief Check if projection matrices are available
             * @return true if matrices are cached, false otherwise
             */
            bool has_projection_matrices() const;

            // Get edge DOF mapping
            const std::map<std::pair<int, int>, int>& get_edge_dof_map() const { return edge_dof_map_; }

        private:

            // ============================================================================
            // LUMPED MASS MATRIX
            // ============================================================================

            bool use_lumped_mass_matrix_;

            /**
             * @brief Compute the lumped mass matrix for a given element.
             * Lumping all mass matrix entries to the diagonal.
             * M_lumped[i,i] = sum_j M_local[i,j] for all j
             * 
             * @param element_data Element data structure containing element information
            */
            void compute_lumped_mass_matrix_naive(Eigen::MatrixXd& M_c);

            /**
             * @brief Compute the unscaled mass matrix for a given element.
             * This is used to compute the lumped mass matrix.
             * M_lumped = M_poly * D_E * D_E^T
             * 
             * @param element_data Element data structure containing element information
             * @return Eigen::MatrixXd Unscaled mass matrix
            */
            Eigen::MatrixXd compute_unscaled_mass_matrix(const ElementData& element_data);

            /**
             * @brief Compute the D matrix for a given element. This is used to build the lumped mass matrix.
             * It is used to compute the row sum vector s: s_E = D_E^T * w_E
             * 
             * For now, it is only implemented for order 1.
             * 
             * @param element_data Element data structure containing element information
             * @return Eigen::MatrixXd D matrix
            */
            Eigen::MatrixXd compute_D_matrix(const ElementData& element_data);

            /**
             * @brief Compute the c vector for a given element. This is used to build the lumped mass matrix.
             * It is used to compute the row sum vector s: s_E = D_E^T * w_E,
             * where M_poly_E * w_E = c_E
             * 
             * @param element_data Element data structure containing element information
             * @return Eigen::VectorXd c vector
            */
            Eigen::VectorXd compute_c_vector(const ElementData& element_data);

            /**
             * @brief Compute the lumped row sum vector for a given element.
             * 
             * s_E = D_E^T * w_E, where M_poly_E * w_E = c_E
             * 
             * @param element_data Element data structure containing element information
            */
            Eigen::VectorXd compute_lumped_row_sum_vector(ElementData& element_data);



            // ============================================================================
            // DEBUG VARIABLES
            // ============================================================================

            // Debug flag
            bool debug_mode_;

            // Debug Varaibles for DOF counting
            int total_n_dof_vertex_;
            int total_n_dof_edge_;
            int total_n_dof_interior_;

            // Debug DOF indices: element index and DOF indices
            std::vector<std::pair<int, std::vector<int>>> global_vertex_dof_indices_;
            std::vector<std::pair<int, std::vector<int>>> global_edge_dof_indices_;
            std::vector<std::pair<int, std::vector<int>>> global_interior_dof_indices_;

            // ============================================================================
            // PROJECTION MATRICES STORAGE
            // ============================================================================

            // Store projection matrices for each element (for error computation)
            std::vector<Eigen::MatrixXd> element_P_nabla_;  // Energy projection matrices
            std::vector<Eigen::MatrixXd> element_P_0_;      // L² projection matrices
            std::vector<ElementData> element_data_cache_;   // Cached element data  

            // ============================================================================
            // DOF MAPPING
            // ============================================================================
            std::map<std::pair<int, int>, int> edge_dof_map_;  // Store edge mapping
            
    };
}
#endif