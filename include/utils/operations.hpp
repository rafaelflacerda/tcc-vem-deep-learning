#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_OPERATIONS_HPP
#define POLIVEM_OPERATIONS_HPP

#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "models/enums.hpp"
#include "models/templates.hpp"



namespace utils {
    class operations{
    public:
        // ============================================================================
        // GEOMETRIC OPERATIONS
        // ============================================================================
        // calculate a polygon area
        static double calcArea(Eigen::MatrixXd coords);

        // calculate a segment length
        static double calcLength(Eigen::MatrixXd coord);

        // calculate polygonal diameter
        static double calcPolygonalDiam(Eigen::MatrixXd coords, int num_vertices);

        // calculate angle for 1D structures
        static double calcAngle(Eigen::MatrixXd coord);

        // Calculate the perimeter of a polygon
        static double compute_perimeter(Eigen::MatrixXd coords);

        // calculate the centroid
        static Eigen::Vector2d calcCentroid(Eigen::MatrixXd coords);

        // calculate outward normal vectors
        static Eigen::Vector2d computerNormalVector(Eigen::MatrixXd coord);

        // Calculate triangle area
        static double compute_triangle_area(
            Eigen::Vector2d v0, 
            Eigen::Vector2d v1, 
            Eigen::Vector2d v2){
                double area = 0.5 * std::abs((v1 - v0).x() * (v2 - v0).y() - (v2 - v0).x() * (v1 - v0).y());
                return area;
            }

        // compute scaled coordinates 
        Eigen::Vector2d computeScaledCoord(Eigen::Vector2d node_coord, Eigen::Vector2d centroid, double h);

        // build an edge from two nodes
        Eigen::MatrixXd buildEdge(Eigen::MatrixXd startNode, Eigen::MatrixXd endNode);

        // get coordinates from a node list for the beam
        Eigen::MatrixXd getCooridanteBeam(Eigen::MatrixXi e, Eigen::MatrixXd nodes);

        // get coordinates from a node list for plane problems with 2 dof per node
        Eigen::MatrixXd getCoordinatesPlane(Eigen::MatrixXi e, Eigen::MatrixXd nodes);

        // get vector values
        Eigen::VectorXd getVectorValues(Eigen::MatrixXi e, Eigen::VectorXd u);

        // get the indices for the linear case of 2D plane linear elastic case
        Eigen::VectorXi getOrder1Indices(Eigen::MatrixXi nodeInd);

        // get the indices for 4th order beam
        Eigen::VectorXi getOrder2Indices(Eigen::MatrixXi nodeInd, int momentInd, BeamSolverType type = BeamSolverType::Beam);

        // get the indices for 45h order beam
        Eigen::VectorXi getOrder5Indices(Eigen::MatrixXi nodeInd, int momentInd, BeamSolverType type = BeamSolverType::Beam);

        // assemble global matrix
        Eigen::MatrixXd assembleMatrix(Eigen::MatrixXd K, Eigen::MatrixXd Kloc, Eigen::MatrixXi indices);

        // assemble global vector
        Eigen::VectorXd assembleVector(Eigen::VectorXd fb, Eigen::VectorXd floc, Eigen::VectorXi indices);

        // force matrix to be symmetric
        void forceSymmetry(Eigen::MatrixXd& K);

        // ============================================================================
        // MONOMIAL EVALUATION AND COMPUTATION METHODS
        // ============================================================================
        static double evaluate_monomial(
            int idx, 
            const Eigen::Vector2d& point, 
            const ElementData& element_data
        );

        static Eigen::Vector2d evaluate_monomial_gradient(
            int idx, 
            const Eigen::Vector2d& point, 
            const ElementData& element_data
        );

        /**
         * @brief Computes the Laplacian of a monomial
         * 
         * Compute Δm_j = ∂²m_j/∂x² + ∂²m_j/∂y²
         * 
         * @param element_data Element data structure containing element information
         * @param monomial_idx Index of the monomial
         * @return Laplacian of the monomial
         */
        static double compute_monomial_laplacian(
            const ElementData& element_data,
            int monomial_idx
        );

        /**
         * @brief Compute the moments of the DOFs
         * 
         * It calculates the integrals ∫_K φi m_j dx and assembles into a vector.
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param order Order of the polynomial
         * @param N_k Dimension of the polynomial space
         * @return Moments of the DOFs
         */
        static Eigen::VectorXd compute_dof_moments(
            const ElementData& element_data,
            int dof_idx,
            int order,
            int N_k
        );
        
        // ============================================================================
        // DOF CLASSIFICATION METHODS
        // ============================================================================

        // These functions determine the type of each DOF based on the DOF numbering scheme

        /**
         * @brief Checks if a DOF is a vertex DOF
         * 
         * Vertex DOFs are always the first n_vertices DOFs in the local numbering
         * 
         * @param dof_idx Index of the DOF
         * @param element_data Element data structure containing element information
         * @return true if the DOF is a vertex DOF, false otherwise
         */
        static bool is_vertex_dof(int dof_idx, const ElementData& element_data){
            return dof_idx < element_data.n_vertices;
        }

        /**
         * @brief Checks if a DOF is an edge DOF
         * 
         * Edge DOFs are the next n_vertices * (order - 1) DOFs in the local numbering
         * 
         * @param dof_idx Index of the DOF
         * @param element_data Element data structure containing element information
         * @return true if the DOF is an edge DOF, false otherwise
         */
        static bool is_edge_dof(int dof_idx, const ElementData& element_data, int order){
            int first_edge_dof = element_data.n_vertices;
            int num_edge_dofs = element_data.n_vertices * (order - 1);
            int last_edge_dof = first_edge_dof + num_edge_dofs - 1;

            return (dof_idx >= first_edge_dof && dof_idx <= last_edge_dof);
        }

        /**
         * @brief Checks if a DOF is a moment DOF
         * 
         * Interior DOFs come after vertex and edge DOFs.
         * For k = 1, no interior DOFs
         * For k = 2, one interior DOF per element
         * For k ≥ 3, (k-1)(k-2)/2 interior DOFs per element
         * 
         * @param dof_idx Index of the DOF
         * @param element_data Element data structure containing element information
         * @param order Order of the element
         * @return true if the DOF is a moment DOF, false otherwise
         */
        static bool is_moment_dof(int dof_idx, const ElementData& element_data, int order){
            if (order < 2) return false;

            int first_moment_dof = element_data.n_vertices + element_data.n_vertices * (order - 1);

            return (dof_idx >= first_moment_dof);
        }

        /**
         * @brief For vertex DOFs, return the local vertex index. Otherwise, return -1
         * 
         * @param dof_idx Index of the DOF
         * @param element_data Element data structure containing element information
         * @return Local vertex index if DOF is a vertex DOF, -1 otherwise
         */
        static int get_vertex_index(int dof_idx, const ElementData& element_data){
            if (!is_vertex_dof(dof_idx, element_data)) return -1;

            return dof_idx;
        }

        /**
         * @brief Extract edge information for an edge DOF
         * Sets: local_edge_dof (0 to k-2) and edge_on_element (0 to n_vertices-1)
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param local_edge_dof Local DOF index on the edge
         * @param edge_on_element Edge index on the element
         * @param order Order of the element
         * @return true if the DOF is an edge DOF, false otherwise
         */
        static bool get_edge_dof_info(
            const ElementData& element_data, 
            int dof_idx, 
            int& local_edge_dof, 
            int& edge_on_element,
            int order
        ){
            if (!is_edge_dof(dof_idx, element_data, order)) return false;

            // Get the local DOF index on the edge
            int relative_dof = dof_idx - element_data.n_vertices;

            edge_on_element = relative_dof / (order - 1);
            local_edge_dof = relative_dof % (order - 1);

            return true;
        }

        /**
         * @brief For moment DOFs, return the local index
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param order Order of the element
         * @return Local index if DOF is a moment DOF, -1 otherwise
         */ 
        static int get_interior_dof_index(
            const ElementData& element_data,
            int dof_idx,
            int order
        ){
            if (!is_moment_dof(dof_idx, element_data, order)) return -1;

            int first_moment_dof = element_data.n_vertices + element_data.n_vertices * (order - 1);

            return dof_idx - first_moment_dof;
        }
   
        static int get_total_vertex_dofs(const ElementData& element_data){
            return element_data.n_vertices;
        }

        static int get_total_edge_dofs(const ElementData& element_data, int order){
            if (order < 2) return 0;
            return element_data.n_vertices * (order - 1);   
        }

        static int get_total_moment_dofs(const ElementData& element_data, int order){
            if (order < 2) return 0;
            return (order - 1) * (order - 2) / 2;
        }
        
        static int get_edge_dofs_per_edge(int order){
            if (order < 2) return 0;
            return order - 1;
        }

        static bool is_vertex_on_edge(int vertex_idx, int edge_idx, int n_vertices){
            int edge_start = edge_idx;
            int edge_end = (edge_idx + 1) % n_vertices;

            return (vertex_idx == edge_start || vertex_idx == edge_end);
        }

        /**
         * @brief Check if the moment polynomial matches Δmj
         * This requires comparing polynomial degrees
         * 
         * @param moment_idx Index of the moment DOF
         * @param monomial_idx Index of the monomial
         * @param element_data Element data structure containing element information
         * @return true if the polynomial matches the Laplacian of the monomial, false otherwise    
         */
        static bool polynomial_matches_laplacian(
            int moment_idx,
            int monomial_idx,
            const ElementData& element_data
        );

        // ============================================================================
        // LEGENDRE POLYNOMIALS
        // ============================================================================
        
        /**
         * @brief Evaluate Legendre polynomial P_n(x) using recurrence relation
         * 
         * Legendre polynomials satisfy the recurrence relation:
         * P_0(x) = 1
         * P_1(x) = x
         * P_n(x) = ((2n-1)x P_{n-1}(x) - (n-1)P_{n-2}(x)) / n
         * 
         * @param n Degree of the Legendre polynomial
         * @param x Point at which to evaluate (should be in [-1, 1])
         * @return Value of P_n(x)
         */
        static double evaluate_legendre_polynomial(int n, double x);

        /**
         * @brief Evaluate the derivative of Legendre polynomial P'_n(x)
         * 
         * Uses the relation: P'_n(x) = n/(x²-1) * (x*P_n(x) - P_{n-1}(x))
         * Alternative for x = ±1: P'_n(±1) = ±n(n+1)/2
         * 
         * @param n Degree of the Legendre polynomial
         * @param x Point at which to evaluate
         * @return Value of P'_n(x)
         */
        static double evaluate_legendre_polynomial_derivative(int n, double x);

        // ============================================================================
        // MATHEMATICAL UTILITY FUNCTIONS
        // ============================================================================

        /**
         * @brief Compute binomial coefficient C(n,k) = n!/(k!(n-k)!)
         * 
         * Uses iterative approach to avoid overflow for large numbers.
         * Applies symmetry property C(n,k) = C(n,n-k) for efficiency.
         * 
         * @param n Upper parameter of binomial coefficient
         * @param k Lower parameter of binomial coefficient
         * @return Binomial coefficient C(n,k)
         */
        static double compute_binomial_coefficient(int n, int k);

        /**
         * @brief Compute factorial n! up to reasonable limits
         * 
         * For large n, this should be combined with binomial coefficient computation
         * to avoid overflow. Uses iterative computation for numerical stability.
         * 
         * @param n Integer for which to compute factorial
         * @return Factorial n!
         */
        static double compute_factorial(int n);

        /**
         * @brief Compute the coefficient (a! * b!) / total_degree!
         * 
         * Uses logarithms for numerical stability when dealing with large factorials.
         * For small values, uses direct computation. For larger values, applies
         * recursive properties to maintain numerical accuracy.
         * 
         * @param a First parameter of Beta coefficient
         * @param b Second parameter of Beta coefficient
         * @param total_degree Total degree of the polynomial
         * @return Beta coefficient (a!)*(b!)/(a+b+1)!
         */
        static double compute_beta_coefficient(int a, int b, int total_degree);

        // ============================================================================
        // TIME INTEGRATION UTILITIES
        // ============================================================================
        
        /**
         * @brief Compute recommended timestep for explicit time integration
         * 
         * For parabolic problems with explicit time schemes (RK3, RK4), the timestep
         * must satisfy a CFL-like condition for both stability AND accuracy:
         * 
         *   Δt ≤ C * h_e²
         * 
         * where:
         *   - h_e is the characteristic mesh size (diameter of elements)
         *   - C is a safety factor that controls the balance between stability and accuracy
         * 
         * Recommended values for C:
         *   - 0.05: Very conservative (guaranteed high accuracy)
         *   - 0.10: Moderate (good balance between accuracy and efficiency) [DEFAULT]
         *   - 0.15: Aggressive (faster but may degrade accuracy on fine meshes)
         * 
         * For convergence studies, use C = 0.10 to ensure temporal error doesn't
         * dominate spatial error, allowing measurement of optimal convergence rates.
         * 
         * @param h_e Characteristic mesh size (element diameter)
         * @param safety_factor CFL safety factor (default 0.10)
         * @return Recommended timestep Δt
         */
        static double compute_recommended_timestep(double h_e, double safety_factor = 0.10);

        // ============================================================================
        // DEBUG METHODS
        // ============================================================================
        static void print_dof_classification(const ElementData& element_data, int order);
     };
}
#endif 
