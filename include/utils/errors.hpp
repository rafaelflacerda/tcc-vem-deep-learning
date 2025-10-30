#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_ERRORS_HPP
#define POLIVEM_ERRORS_HPP

#include <cmath>
#include <iomanip>
#include <iostream>

#include <Eigen/Dense>

#include "solver/parabolic.hpp"

namespace utils {
    class errors {
        
        public:
            errors(bool debug = false) : debug_(debug) {}
            ~errors() = default;

            // ============================================================================
            // ERROR COMPUTATION
            // ============================================================================

            /**
             * @brief Compute true element-wise H¹ seminorm error ( √∫|∇(u−u_h)|² )
             * 
             * @param vem_solver VEM solver
             * @param U_final Final solution
             * @param final_time Final time
             * @param exact_gradient Exact gradient function
             */
            static double compute_true_h1_error_with_stored_matrices(
                const solver::parabolic& vem_solver,
                const Eigen::VectorXd& U_final,
                double final_time,
                const std::function<Eigen::Vector2d(double, double, double)>& exact_gradient
            );

            /**
             * @brief Compute the error of the discrete VEM solution:
             * E²_{h,τ} = m_h(u_h - u_I, u_h - u_I) / m_h(u_h, u_h)
             * where u_I is the interpolated solution and u_h is the discrete VEM solution.
             * 
             * @param vem_solver VEM solver
             * @param U_final Final solution
             * @param final_time Final time
             * @param exact_solution Exact solution function
             */
            static double compute_l2_error_parabolic(
                const solver::parabolic& vem_solver,
                const Eigen::VectorXd& U_final,
                double final_time,
                const std::function<double(double, double, double)>& exact_solution
            );

            // ============================================================================
            // VEM INTERPOLANT
            // ============================================================================

            /**
             * @brief Compute VEM interpolant I_h u following exact DOF definitions from paper Section II.B
             * DOF functionals (D1)-(D3):
             * (D1) Vertex DOFs: χ_i(u) = u(V_i) 
             * (D2) Edge DOFs: Values at k-1 distinct points on each edge
             * (D3) Interior DOFs: Moments ∫_K u p_α dx for p_α ∈ P_{k-2}(K)
             * 
             * @param vem_solver VEM solver
             * @param time Time
             * @return VEM interpolant
             */
            static Eigen::VectorXd compute_vem_interpolant(
                const solver::parabolic& vem_solver,
                double time,
                const std::function<double(double, double, double)>& exact_solution
            );

            // ============================================================================
            // HELPER METHODS
            // ============================================================================
            /**
             * @brief Generate monomial powers for P_m(K): {(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...}
             * Following paper's monomial basis ordering
             * 
             * @param max_degree Maximum degree of monomial
             * @return Vector of monomial powers
             */
            static std::vector<std::pair<int, int>> generate_monomial_powers(int max_degree);

            /**
             * @brief Compute interior moment ∫_K u(t,x,y) * m_α(x,y) dx dy exactly
             * where m_α(x,y) = ((x-x_c)/h_K)^α_x * ((y-y_c)/h_K)^α_y is scaled monomial
             * 
             * @param element_data Element data
             * @param time Time
             * @param alpha_x x-degree of monomial
             * @param alpha_y y-degree of monomial
             * @param exact_solution Exact solution function
             * @return Interior moment
             */
            static double compute_interior_moment_exact(
                const ElementData& element_data, 
                double time, 
                int alpha_x, 
                int alpha_y,
                const std::function<double(double, double, double)>& exact_solution
            );

            /**
             * @brief Compute edge moment χ_j(u) = (1/L) ∫_e u(s) * m_j(t) ds
             * where m_j(t) = (2t - 1)^j is the scaled orthogonal polynomial
             * 
             * @param v1 Start vertex
             * @param v2 End vertex
             * @param edge_length Edge length
             * @param time Time
             * @param j Moment index
             * @param exact_solution Exact solution function
             * @return Edge moment
             */
            static double compute_edge_moment_orthogonal(
                const Eigen::Vector2d& v1, 
                const Eigen::Vector2d& v2, 
                double edge_length, 
                double time, 
                int j,
                const std::function<double(double, double, double)>& exact_solution
            );

            /**
             * @brief Generate Gauss-Legendre quadrature points and weights on [0,1]
             * This transforms standard Gauss-Legendre rule from [-1,1] to [0,1]:
             * - Point transformation: t = (ξ + 1)/2
             * - Weight scaling: w → w/2
             * 
             * TODO: Transport to integration.cpp
             * 
             * @param n Number of points
             * @param points Vector of quadrature points
             * @param weights Vector of quadrature weights
             */
            static void get_gauss_legendre_01(
                int n, 
                std::vector<double>& points, 
                std::vector<double>& weights
            );

        private:
            bool debug_ = false;

    };
}

#endif