#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_INTEGRATION_HPP
#define POLIVEM_INTEGRATION_HPP

#include <cmath>
#include <iomanip>
#include <iostream>

#include <Eigen/Dense>

#include "models/templates.hpp"
#include "utils/operations.hpp"

namespace utils {
    class integration{
        public:

        // ============================================================================
        // GENERAL INTEGRATION METHODS
        // ============================================================================

        // Gauss-Legendre quadrature integration points (2D case)
        Eigen::MatrixXd gaussPoints;

        // Gauss-Legendre quadrature integration weights (2D case)
        Eigen::VectorXd gaussWeights;

        // parameterized coordinates
        Eigen::MatrixXd paramCoord;

        // set Gauss-Legendre quadrature points and weights
        void setGaussParams(int integration_order);

        // set parameterized coordinates
        void setParamCoords(Eigen::MatrixXd coords, double s, double t);

        /**
         * @brief Get Gauss-Legendre quadrature points and weights for 1D integration
         * 
         * @param required_order Order of the quadrature rule
         * @param points Vector of quadrature points
         * @param weights Vector of quadrature weights
         */
        static void get_gauss_quadrature_rule(
            int required_order, 
            std::vector<double>& points, 
            std::vector<double>& weights
        );


        // ============================================================================
        // DOF INTEGRATION METHODS
        // ============================================================================

        // These functions compute integrals of DOF functions over boundaries and element interiors

        /**
         * @brief Compute ∫_{∂K} φ_i ds
         * 
         * This is needed for boundary normalization constraint (k=1)
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @return Value of the integral
         */
        static double compute_dof_boundary_integral(
            const ElementData& element_data,
            int dof_idx,
            int order
        );

        /**
         * @brief Compute ∫_K φ_i dx
         * 
         * This is needed for interior DOF integration (k≥2)
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param order Order of the polynomial
         * @return Value of the integral
         */
        static double compute_dof_area_integral(
            const ElementData& element_data,
            int dof_idx,
            int order
        );

        /**
         * @brief Compute the load integral for a given DOF.
         * 
         * For vertex and edge DOFs:
         * Use the fact that ∫_K f_h φ_i dx = ∫_K f_h Π^0_k φ_i dx
         * 
         * For interior DOFs:
         * we can directly compute ∫_K f_h m_α dx
         * 
         * @param element_data Element data structure containing element information
         * @param fh_coeffs Coefficients of the projected source term
         * @param dof_idx Index of the DOF
         * @param N_k Dimension of the polynomial space
         * @param order Order of the polynomial
         * @return double Load integral
        */
        static double compute_load_integral_for_dof(
            const ElementData& element_data,
            const Eigen::VectorXd& fh_coeffs,
            int dof_idx,
            int N_k,
            int order
        );

        // ============================================================================
        // VERTEX DOF INTEGRATION
        // ============================================================================

        /**
         * @brief For vertex DOFs: ∫_{∂K} N_i ds where N_i is the vertex shape function
         * 
         * For linear elements, each vertex contributes to adjacent edges
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @return Value of the integral
         */
        static double compute_vertex_dof_boundary_integral(
            const ElementData& element_data,
            int dof_idx
        );
        
        /**
         * @brief For vertex DOFs: ∫_K N_i dx where N_i is the vertex shape function
         * For linear approximation, this is approximately area/n_vertice
         * 
         * This is typically not necessary. Maybe we are going to use it for the stabilization term.
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @return Value of the integral
         */
        static double compute_vertex_dof_area_integral(
            const ElementData& element_data,
            int idx,
            int order
        );
        
        /**
         * @brief Compute the contribution of a vertex to an edge
         * 
         * Compute ∫_edge N_vertex ds
         * For linear shape functions, this is edge_length/2
         * 
         * @param element_data Element data structure containing element information
         * @param vertex_idx Index of the vertex
         * @param edge_idx Index of the edge
         * @return Value of the contribution
         */
        static double compute_vertex_contribution_to_edge(
            const ElementData& element_data,
            int vertex_idx,
            int edge_idx
        );

        // ============================================================================
        // EDGE DOF INTEGRATION METHODS
        // ============================================================================

        /**
         * @brief Compute the boundary integral of the edge DOF
         * 
         * For edge DOFs: ∫_{∂K} φ_edge ds
         * Edge DOFs are typically orthogonal to constants, so integral should be 0.integration
         * This might change for a difference edge basis function.
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param order Order of the polynomial
         * @return Value of the integral
         */
        static double compute_edge_dof_boundary_integral(
            const ElementData& element_data,
            int dof_idx,
            int order
        ){
            return 0.0;
        }

        /**
         * @brief Compute the area integral of the edge DOF
         * 
         * For edge DOFs: ∫_K φ_edge dx
         * This requires integration over the entire element
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param order Order of the polynomial
         * @return Value of the integral
         */
        static double compute_edge_dof_area_integral(
            const ElementData& element_data,
            int dof_idx, 
            int order
        );

        /**
         * @brief Integrate edge basis function over entire element using triangulation
         * 
         * @param element_data Element data structure containing element information
         * @param edge_idx Index of the edge
         * @param local_dof_idx Index of the DOF
         * @param order Order of the polynomial
         * @return Value of the integral
         */
        static double integrate_edge_dof_over_element(
            const ElementData& element_data,
            int edge_idx,
            int local_dof_idx,
            int order
        );

        /**
         * @brief Evaluate edge basis function at arbitrary point in element
         * This extends the edge function into the element interior
         * 
         * @param element_data Element data structure containing element information
         * @param point Point to evaluate the basis function at
         * @param edge_idx Index of the edge
         * @param local_dof_idx Index of the DOF
         * @param order Order of the polynomial
         * @return Value of the basis function
         */
        static double evaluate_edge_basis_at_point(
            const ElementData& element_data,
            const Eigen::Vector2d& point,
            int edge_idx,
            int local_dof_idx,
            int order
        );


        // ============================================================================
        // MOMEMNT DOF INTEGRATION METHODS
        // ============================================================================

        /**
         * @brief Compute the area integral of the interior DOF
         * 
         * For interior DOFs: ∫_K φ_interior dx
         * These are typically area moments: ∫_K p(x,y) dx where p is a polynomial
         * 
         * Interior DOFs correspond to moments of polynomials of degree k-2
         * For k=2: 1 interior DOF corresponding to constant moment
         * For k=3: 2 interior DOFs corresponding to {1, x, y} moments
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param order Order of the polynomial
         * @return Value of the integral
         */
        static double compute_moment_area_integral(
            const ElementData& element_data,
            int dof_idx,
            int order
        );

        /**
         * @brief Get the monomial index for interior DOF.
         * 
         * Interior DOFs correspond to monomials of degree ≤ k-2
         * These are the FIRST monomials in our monomial basis (which goes up to degree k)
         * 
         * @param moment_idx Index of the moment DOF
         * @param order Order of the polynomial
         * @return Index of the monomial
         */
        static int get_monomial_index_for_interior_dof(int moment_idx, int order);


        // ============================================================================
        // MONOMIAL INTEGRATION METHODS
        // ============================================================================

        /**
         * @brief Compute ∫_{∂K} m_j ds where m_j is a scaled monomial
         * 
         * @param element_data Element data structure containing element information
         * @param monomial_idx Index of the monomial
         * @return Value of the integral
         */
        static double compute_monomial_boundary_integral(
            const ElementData& element_data,
            int monomial_idx
        );

        /**
         * @brief Compute ∫_K m_j dx where m_j is a scaled monomial
         * 
         * @param element_data Element data structure containing element information
         * @param monomial_idx Index of the monomial
         * @return Value of the integral
         */
        static double compute_monomial_area_integral(
            const ElementData& element_data,
            int monomial_idx
        );


        /**
         * @brief Compute ∫_{e_i} m_j ds where m_j is a scaled monomial
         * 
         * @param element_data Element data structure containing element information
         * @param edge_idx Index of the edge
         * @param monomial_idx Index of the monomial
         * @return Value of the integral
         */
        static double integrate_monomial_over_edge(
            const ElementData& element_data,
            int edge_idx,
            int monomial_idx
        );

        /**
         * @brief Use divergence theorem to convert area integral to boundary integral
         * ∫_K x^α1 y^α2 dx = ∫_{∂K} [x^{α1+1} y^α2 / (α1+1)] n_x ds
         * 
         * @param element_data Element data structure containing element information
         * @param alpha_1 Index of the first monomial
         * @param alpha_2 Index of the second monomial
         * @return Value of the integral
         */
        static double compute_divergence_x_integral(
            const ElementData& element_data,
            int alpha_1,
            int alpha_2
        );

        /**
         * @brief Compute the divergence y-integral
         * 
         * ∫_K x^α1 y^α2 dy = ∫_{∂K} [x^α1 y^{α2+1} / (α2+1)] n_y ds
         * 
         * @param element_data Element data structure containing element information
         * @param alpha_1 Index of the first monomial
         * @param alpha_2 Index of the second monomial
         * @return Value of the integral
         */
        static double compute_divergence_y_integral(
            const ElementData& element_data,
            int alpha_1,
            int alpha_2
        );

        /**
         * @brief Integrate the x-term in the divergence theorem
         * 
         *  Integrate x^α1 y^α2 n_x over edge
         * 
         * @param element_data Element data structure containing element information
         * @param alpha_1 Index of the first monomial
         * @param alpha_2 Index of the second monomial
         * @param edge_idx Index of the edge
         * @return Value of the integral
         */
        static double integrate_divergence_x_term(
            const ElementData& element_data,
            int alpha_1,
            int alpha_2,
            int edge_idx
        );

        /**
         * @brief Integrate the y-term in the divergence theorem
         * 
         *  Integrate x^α1 y^α2 n_y over edge
         * 
         * @param element_data Element data structure containing element information
         * @param alpha_1 Index of the first monomial
         * @param alpha_2 Index of the second monomial
         * @param edge_idx Index of the edge
         * @return Value of the integral
         */
        static double integrate_divergence_y_term(
            const ElementData& element_data,
            int alpha_1,
            int alpha_2,
            int edge_idx
        );

        /**
         * @brief Compute the area integral of the momement DOF monomial.
         * 
         * Only moment and vertex DOFs have a contribution to this integral.
         * 
         * Compute ∫_K φi Δmj dx.
         * Not every interior DOF will contribute to every ∫_K φi Δmj dx.
         * Only the DOF whose moment polynomial matches Δmj will have a non-zero contribution.
         * This is due to the biorthogonality property of the DOFs and basis functions.
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param monomial_idx Index of the monomial
         * @param order Order of the polynomial
         * @param N_k Dimension of the polynomial space
         * @return Value of the integral
         */
        static double compute_moment_dof_laplacian_monomial_area_integral(
            const ElementData& element_data,
            int dof_idx,
            int monomial_idx,
            int order,
            int N_k
        );

        /**
         * @brief Compute the boundary integral of a vertex DOF monomial for k=1.
         * 
         * Compute ∫_{∂K} φ_i ⋅ ∇m_j ⋅ n ds for vertex DOFs for k=1.
         * 
         * @param element_data Element data structure containing element information
         * @param vertex_idx Index of the vertex
         * @param monomial_idx Index of the monomial
         * @return Boundary integral of the vertex DOF monomial
         */
        static double compute_vertex_dof_monomial_boundary_integral_k1(
            const ElementData& element_data,
            int dof_idx, 
            int monomial_idx
        );

        /**
         * @brief Compute the boundary integral of a vertex DOF monomial.
         * 
         * Compute ∫_{∂K} φ_i ⋅ ∇m_j ⋅ n ds for vertex DOFs.
         * 
         * @param element_data Element data structure containing element information
         * @param vertex_idx Index of the vertex
         * @param monomial_idx Index of the monomial
         * @return Boundary integral of the vertex DOF monomial
         */
        static double compute_vertex_dof_monomial_boundary_integral(
            const ElementData& element_data,
            int vertex_idx, 
            int monomial_idx
        );

        /**
         * @brief Compute the boundary integral of an edge DOF monomial.
         * 
         * Compute ∫_{e_i} φ_edge ⋅ ∇m_j ⋅ n ds for edge DOFs.
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param monomial_idx Index of the monomial
         * @param order Order of the polynomial
         * @return Boundary integral of the edge DOF monomial
         */
        static double compute_edge_dof_monomial_boundary_integral(
            const ElementData& element_data,
            int dof_idx,
            int monomial_idx,
            int order
        );

        /**
         * @brief Compute the boundary integral of a DOF monomial.
         * 
         * Compute ∫_∂K φi (∇mj·n) ds
         * 
         * @param element_data Element data structure containing element information
         * @param dof_idx Index of the DOF
         * @param monomial_idx Index of the monomial
         * @param order Order of the polynomial
         * @return Boundary integral of the DOF monomial
         */
        static double compute_dof_monomial_boundary_integral(
            const ElementData& element_data,
            int dof_idx,
            int monomial_idx,
            int order
        );
        
        // ============================================================================
        // POLYGONAL MOMENT COMPUTATION
        // ============================================================================

        /**
         * @brief Compute exact polygonal moment I_pq = ∫_E (x-x_c)^p * (y-y_c)^q dx
         * 
         * This function implements the closed-form algebraic formula for computing
         * polygonal moments using Green's theorem and binomial expansion. The formula
         * converts the area integral to a boundary integral and evaluates it exactly.
         * 
         * Mathematical foundation:
         * I_pq = (1/(p+q+2)) * ∑_{r=0}^{N-1} [Δy_r * S1 - Δx_r * S2]
         * where S1 and S2 are double sums involving binomial coefficients and
         * Beta function coefficients.
         * 
         * @param element_data Element data structure containing vertices and geometry
         * @param p Power of x-coordinate (non-negative integer)
         * @param q Power of y-coordinate (non-negative integer)
         * @param centroid Centroid coordinates (if null, computed automatically)
         * @return Unscaled moment I_pq
         */
        static double compute_polygonal_moment_Ipq(
            const ElementData& element_data,
            int p,
            int q,
            const Eigen::Vector2d* centroid = nullptr
        );

        /**
         * @brief Compute scaled polygonal moment for VEM applications
         * 
         * This function computes the scaled moment I_pq = ∫_E ((x-x_c)/h_e)^p * ((y-y_c)/h_e)^q dx
         * which is what VEM actually needs for polynomial matrix computation.
         * 
         * The scaling factor h_e^(p+q) is applied to the unscaled moment to obtain
         * the dimensionless scaled moment used in VEM formulations.
         * 
         * @param element_data Element data structure containing vertices and geometry
         * @param p Power of scaled x-coordinate (non-negative integer)
         * @param q Power of scaled y-coordinate (non-negative integer)
         * @param centroid Centroid coordinates (if null, computed automatically)
         * @return Scaled moment for VEM applications
         */
        static double compute_scaled_polygonal_moment_Ipq(
            const ElementData& element_data,
            int p,
            int q,
            const Eigen::Vector2d* centroid = nullptr
        );

        /**
         * @brief Compute edge contribution to polygonal moment
         * 
         * This helper function computes the contribution of a single edge to the
         * total polygonal moment using the closed-form formula. It handles the
         * binomial expansion and Beta coefficient computation for one edge.
         * 
         * @param vertices_r Current edge vertex coordinates
         * @param vertices_next Next edge vertex coordinates
         * @param p Power of x-coordinate
         * @param q Power of y-coordinate
         * @param centroid Centroid coordinates
         * @return Edge contribution to the moment
         */
        static double compute_edge_moment_contribution(
            const Eigen::Vector2d& vertices_r,
            const Eigen::Vector2d& vertices_next,
            int p,
            int q,
            const Eigen::Vector2d& centroid
        );

        // ============================================================================
        // DEBUG METHODS
        // ============================================================================
        static void print_dof_classification(const ElementData& element_data, int order);
        
    };
}

#endif