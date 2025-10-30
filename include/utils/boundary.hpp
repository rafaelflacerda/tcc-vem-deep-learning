#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_BOUNDARY_HPP
#define POLIVEM_BOUNDARY_HPP

#include <functional>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "integration.hpp"

namespace utils {
    class boundary {
    public:
        enum class BCType{
            DIRICHLET,  ///< Essential boundary condition u = g
            NEUMANN,    ///< Natural boundary condition du/dn = g
            ROBIN       ///< Mixed boundary condition du/dn + alpha*u = g
        };

        enum class BoundaryRegion{
            LEFT,         ///< x = x_min boundary
            RIGHT,        ///< x = x_max boundary  
            BOTTOM,       ///< y = y_min boundary
            TOP,          ///< y = y_max boundary
            ENTIRE,       ///< Entire boundary
            CUSTOM        ///< User-defined region
        };

        struct BoundaryCondition {
            BCType type;
            std::function<double(const Eigen::Vector2d&, double)> value_function;
            std::set<int> affected_vertices;
            std::set<std::pair<int, int>> affected_edges;
            BoundaryRegion region;
            std::string name;

            // Robin BC parameters (αu + β∂u/∂n = g)
            double alpha = 0.0; ///< Coefficient for u term
            double beta = 1.0;  ///< Coefficient for du/dn term
        };

        /**
         * @brief Boundary segment information
         */
        struct BoundarySegment {
            std::pair<int, int> edge;
            Eigen::Vector2d start_point;
            Eigen::Vector2d end_point;
            Eigen::Vector2d outward_normal;
            double length;
            BoundaryRegion region;
        };

        // ============================================================================
        // CONSTRUCTORS AND DESTRUCTOR
        // ============================================================================
        boundary(); // Default constructor

        boundary(const Eigen::MatrixXd& nodes, const Eigen::MatrixXi& elements, int polynomial_order);

        ~boundary() = default; // Destructor

        // ============================================================================
        // INITIALIZATION AND SETUP
        // ============================================================================
        
        /**
         * @brief Initialize the boundary conditions from a mesh
         * @param nodes The node matrix from the mesh
         * @param elements The element matrix from the mesh
         * @param polynomial_order The polynomial order of the mesh
         */
        void initialize(
            const Eigen::MatrixXd& nodes,
            const Eigen::MatrixXi& elements,
            int polynomial_order
        );

        /**
         * @brief Set the DOF mapping for the boundary conditions
         * @param element_dof_mapping Local to global DOF mapping for each element
         * @param edge_to_dof_map Edge to global DOF map
         * @param total_dofs The total number of DOFs
         */
        void set_dof_mapping(
            const std::vector<std::vector<int>>& element_dof_mapping,
            const std::map<std::pair<int, int>, int>& edge_to_dof_map,
            int total_dofs
        );
        
        /**
         * @brief Automatically detect the boundary segments from the mesh
         * 
         * Detect edges that belong to only one element, which means they are boundary edges.
         * From the set of boundary edges, determine the boundary vertices.
         */
        void automatically_detect_boundary();

        /**
         * @brief Set the boundary segments manually
         * @param boundary_vertices The boundary vertices
         * @param boundary_edges The boundary edges
         */
        void set_boundary_manually(
            const std::set<int>& boundary_vertices,
            const std::set<std::pair<int, int>>& boundary_edges\
        );

    
        // ============================================================================
        // BOUNDARY CONDITION SPECIFICATION
        // ============================================================================

        /**
         * @brief Add Dirichlet boundary condition
         * @param value_function Function defining boundary values u(x,y,t)
         * @param region Boundary region to apply condition
         * @param name Optional name for the condition
         */
        void add_dirichlet(
            const std::function<double(const Eigen::Vector2d&, double)>& value_function,
            BoundaryRegion region = BoundaryRegion::ENTIRE,
            const std::string& name = ""
        );

        /**
         * @brief Add Neumann boundary condition
         * @param flux_function Function defining boundary flux ∂u/∂n(x,y,t)
         * @param region Boundary region to apply condition
         * @param name Optional name for the condition
         */
        void add_neumann(
            const std::function<double(const Eigen::Vector2d&, double)>& flux_function,
            BoundaryRegion region = BoundaryRegion::ENTIRE,
            const std::string& name = ""
        );

        /**
         * @brief Add Robin boundary condition
         * @param value_function Function defining boundary values u(x,y,t)
         * @param alpha Coefficient for u term
         * @param beta Coefficient for ∂u/∂n term 
         * @param region Boundary region to apply condition
         * @param name Optional name for the condition
         */
        void add_robin(
            const std::function<double(const Eigen::Vector2d&, double)>& value_function,
            double alpha,
            double beta,
            BoundaryRegion region = BoundaryRegion::ENTIRE,
            const std::string& name = ""
        );

        /**
         * @brief Add custom boundary condition on specific vertices/edges
         * @param type Boundary condition type
         * @param value_function Function defining boundary values
         * @param vertices Set of vertex indices
         * @param edges Set of edge pairs
         * @param name Optional name for the condition
         */
        void add_custom(
            BCType type,
            const std::function<double(const Eigen::Vector2d&, double)>& value_function,
            const std::set<int>& vertices,
            const std::set<std::pair<int, int>>& edges,
            BoundaryRegion region = BoundaryRegion::ENTIRE,
            const std::string& name = ""
        );

        // ============================================================================
        // CONVENIENCE FUNCTIONS
        // ============================================================================

        /**
         * @brief Set homogeneous Dirichlet boundary conditions
         */
        void set_homogeneous_dirichlet(){
            clear_conditions();
            add_dirichlet(
                [](const Eigen::Vector2d&, double){return 0.0;},
                BoundaryRegion::ENTIRE, 
                "Homogenous_Dirichlet"
            );
        }

        /**
         * @brief Set homogeneous Neumann boundary conditions
         */
        void set_homogeneous_neumann(){
            clear_conditions();
            add_neumann(
                [](const Eigen::Vector2d&, double){return 0.0;},
                BoundaryRegion::ENTIRE,
                "Homogenous_Neumann"
            );
        }

        /**
         * @brief Clear all boundary conditions
         */
        void clear_conditions(){
            boundary_conditions_.clear();
            dirichlet_dofs_.clear();
            neumann_dofs_.clear();
            robin_dofs_.clear();

            if (debug_mode_) {
                std::cout << "Cleared all boundary conditions" << std::endl;
            }
        }

        /**
         * @brief Set debug mode
         */
        void set_debug_mode(bool debug) {
            debug_mode_ = debug;
        }

        // ============================================================================
        // BOUNDARY CONDITION APPLICATION
        // ============================================================================

        /**
         * @brief Apply Dirichlet boundary conditions to the system matrices
         * 
         * @param K_h The stiffness matrix
         * @param M_h The mass matrix
         * @param F_h The load vector
         * @param time The current time
         * @param preserve_mass_diagonal If true, preserve the mass matrix diagonal during elimination
         */
        void apply_dirichlet_conditions(
            Eigen::SparseMatrix<double>& K_h,
            Eigen::SparseMatrix<double>& M_h,
            Eigen::VectorXd& F_h, 
            double time,
            bool preserve_mass_diagonal = false
        );

        /**
         * @brief Apply Neumann boundary conditions to the load vector
         * 
         * @param F_h The load vector
         * @param polynomial_order The polynomial order of the mesh
         * @param time The current time
         */
        void apply_neumann_conditions(
            Eigen::VectorXd& F_h,
            int polynomial_order,
            double time
        );

        /**
         * @brief Apply Robin boundary conditions to system matrices and load vector
         * 
         * TODO: Implement this later
         * 
         * @param K_h The stiffness matrix
         * @param F_h The load vector
         * @param polynomial_order The polynomial order of the mesh
         * @param time The current time
         */
        void apply_robin_conditions(
            Eigen::SparseMatrix<double>& K_h,
            Eigen::VectorXd& F_h,
            int polynomial_order,
            double time
        );

        /**
         * @brief Apply all boundary conditions to the system matrices and load vector
         * 
         * TODO: Implement this later
         * 
         * @param K_h The stiffness matrix
         * @param M_h The mass matrix
         * @param F_h The load vector
         * @param polynomial_order The polynomial order of the mesh
         * @param time The current time
         */
        void apply_all_conditions(
            Eigen::SparseMatrix<double>& K_h,
            Eigen::SparseMatrix<double>& M_h,
            Eigen::VectorXd& F_h,
            int polynomial_order,
            double time
        );

        // ============================================================================
        // QUERY AND ACCESS FUNCTIONS
        // ============================================================================

        /**
         * @brief Check if the vertex is a boundary vertex
         * 
         * @param vertex_idx Index of the vertex
         * @return true if the vertex is a boundary vertex, false otherwise
         */
        bool is_boundary_vertex(int vertex_idx) const {
            return boundary_vertices_.count(vertex_idx) > 0;
        }

        /**
         * @brief Check if the edge is a boundary edge
         * 
         * @param v1 Index of the first vertex
         * @param v2 Index of the second vertex
         * @return true if the edge is a boundary edge, false otherwise
         */
        bool is_boundary_edge(int v1, int v2) const {
            std::pair<int, int> edge = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);
            return boundary_edges_.count(edge) > 0;
        }

        /**
         * @brief Compute the geometric position in 2D space of any DOF (vertex, edge, moment)
         * 
         * @param global_dof_idx Index of the DOF
         * @return The geometric position of the DOF
         */
        Eigen::Vector2d get_dof_position(int global_dof_idx) const;

        /**
         * @brief Compute the geometric position in 2D space of a moment DOF
         * 
         * IMPORTANT: This is only a representation for visualization and debugging purposes.
         * Momento DOFs do not have a physical position in the domain.
         * 
         * @param global_dof_idx Index of the DOF
         * @param element_idx Index of the element
         * @return The geometric position of the DOF
         */
        Eigen::Vector2d get_moment_dof_position(
            int global_dof_idx,
            int element_idx
        ) const;

    private:
        // ============================================================================
        // PRIVATE MEMBER VARIABLES
        // ============================================================================

        // Mesh data
        Eigen::MatrixXd nodes_;
        Eigen::MatrixXi elements_;
        int polynomial_order_;
        int total_dofs_;

        // DOF mapping
        std::vector<std::vector<int>> element_dof_mapping_;
        std::map<std::pair<int, int>, int> edge_dof_map_;

        // Boundary features
        std::set<int> boundary_vertices_;
        std::set<std::pair<int, int>> boundary_edges_;
        std::vector<BoundarySegment> boundary_segments_;

        // Boundary conditions
        std::vector<BoundaryCondition> boundary_conditions_;
        std::set<int> dirichlet_dofs_;
        std::set<int> neumann_dofs_;
        std::set<int> robin_dofs_;

        // Configuration
        bool is_initialized_;
        bool debug_mode_;

        // ============================================================================
        // PRIVATE HELPER METHODS
        // ============================================================================

        /**
         * @brief Classify DOFs affected by boundary conditions
         */
        void classify_boundary_dofs();

        /**
         * @brief Get DOFs for a specific boundary region
         * @param region Boundary region
         * @return Set of DOF indices in the region
         */
        std::set<int> get_dofs_for_region(BoundaryRegion region) const;

        /**
         * @brief Get vertices for a specific boundary region
         * @param region Boundary region
         * @return Set of vertex indices in the region
         */
        std::set<int> get_vertices_for_region(BoundaryRegion region) const;
        
        /**
         * @brief Get edges for a specific boundary region
         * @param region Boundary region
         * @return Set of edge pairs in the region
         */
        std::set<std::pair<int,int>> get_edges_for_region(BoundaryRegion region) const;

        /**
         * @brief Determine which boundary region a vertex belongs to
         * @param vertex_idx The index of the vertex
         * @return The region of the vertex
         */
        BoundaryRegion classify_vertex_region(int vertex_idx) const;

        /**
         * @brief Determine which boundary region an edge belongs to
         * @param v1 The index of the first vertex
         * @param v2 The index of the second vertex
         * @return The region of the edge
         */
        BoundaryRegion classify_edge_region(int v1, int v2) const;

        /**
         * @brief Get global DOFs associated with an edge
         * @param v1 First vertex of edge
         * @param v2 Second vertex of edge
         * @return Set of global DOF indices
         */
        std::set<int> get_edge_dofs(int v1, int v2) const;

        /**
         * @brief Build boundary segments from detected boundary features
         */
        void build_boundary_segments();

        /**
         * @brief Compute the outward normal vector for an edge
         * @param v1 The index of the first vertex
         * @param v2 The index of the second vertex
         * @return The outward normal vector
         */
        Eigen::Vector2d compute_outward_normal_vector(int v1, int v2) const{
            Eigen::Vector2d edge_vector = nodes_.row(v2) - nodes_.row(v1);
            Eigen::Vector2d normal_vector(-edge_vector(1), edge_vector(0));
            normal_vector.normalize();
            return normal_vector;
        }
        
        /**
         * @brief Get the bounding box of the domain
         * @return A vector containing the minimum and maximum x and y coordinates
         */
        Eigen::Vector4d get_bounding_box() const {
            double x_min = nodes_.col(0).minCoeff();
            double x_max = nodes_.col(0).maxCoeff();
            double y_min = nodes_.col(1).minCoeff();
            double y_max = nodes_.col(1).maxCoeff();
            return Eigen::Vector4d(x_min, x_max, y_min, y_max);
        }

        // ============================================================================
        // PRIVATE BC MANIPULATION METHODS
        // ============================================================================

        /**
         * @brief Apply Dirichlet elimination to the system matrix regarding a single DOF
         * 
         * It modifies the global matrices and vector to enforce that a certain DOF has a 
         * prescribed value.
         * 
         * @param K_h The stiffness matrix
         * @param M_h The mass matrix
         * @param F_h The load vector
         * @param dof_idx The DOF index to apply the Dirichlet condition to
         * @param preserve_mass_diagonal If true, preserve the mass matrix diagonal during elimination
         * @param value The value of the Dirichlet condition
         */
        void apply_dirichlet_elimination(
            Eigen::SparseMatrix<double>& K_h,
            Eigen::SparseMatrix<double>& M_h,
            Eigen::VectorXd& F_h,
            int dof_idx, 
            double value,
            bool preserve_mass_diagonal
        );

        /**
         * @brief Compute the contribution of a Neumann edge to the load vector
         * 
         * @param edge The edge
         * @param flux_function The flux function
         * @param polynomial_order The polynomial order of the mesh
         * @param time The current time
         */
        std::map<int, double> compute_neumann_edge_contribution(
            const std::pair<int, int>& edge,
            const std::function<double(const Eigen::Vector2d&, double)>& flux_function,
            int polynomial_order, 
            double time
        );

        /**
         * @brief Evaluate the basis function for a boundary DOF
         * 
         * It computes the value of a boundary basis function (either vertex or edge DOF) 
         * at a parametric point.
         * We use a scaling factor to guarantee orthonormality of the basis functions.
         * 
         * VEM edge basis functions are scaled Legendre polynomials
         * For edge DOF j (j = 0, 1, ..., k-2), the basis function is:
         * φ_j(xi) = √((2j+3)/2) * L_{j+1}(xi)
         * where L_n(xi) is the n-th Legendre polynomial
         * 
         * @param global_idx Global index of the DOF
         * @param xi Parametric coordinate
         * @param edge Edge information
         * @param polynomial_order The polynomial order of the mesh
         * @return Value of the basis function
         */
        double evaluate_boundary_basis_function(
            int global_idx,
            double xi,
            const std::pair<int, int>& edge,
            int polynomial_order
        ) const;
        
    };
}

#endif