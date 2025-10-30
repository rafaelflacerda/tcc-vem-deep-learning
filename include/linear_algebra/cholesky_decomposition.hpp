#ifndef POLIVEM_CHOLESKY_DECOMPOSITION_HPP
#define POLIVEM_CHOLESKY_DECOMPOSITION_HPP

#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <iostream>
#include <cstdlib>  // for setenv
#include <string>   // for std::to_string

#include "utils/scope_timer.hpp"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

namespace LinearAlgebra {
    class CholeskyDecomposition{
        public:
            // ============================================================================
            // CONSTRUCTORS AND DESTRUCTOR
            // ============================================================================
            
            /**
             * @brief Construct Cholesky solver for given matrix size
             * @param n Matrix dimension (must be square SPD)
             * @param block_size Block size for optimization (0 = auto-tune)
             */
            explicit CholeskyDecomposition(size_t n, size_t block_size = 0);
            
            /**
             * @brief Default destructor
             */
            ~CholeskyDecomposition() = default;
            
            // Disable copy operations (heavy objects)
            CholeskyDecomposition(const CholeskyDecomposition&) = delete;
            CholeskyDecomposition& operator=(const CholeskyDecomposition&) = delete;
            
            // Enable move operations
            CholeskyDecomposition(CholeskyDecomposition&&) = default;
            CholeskyDecomposition& operator=(CholeskyDecomposition&&) = default;

            // ============================================================================
            // MAIN SOLVER INTERFACE
            // ============================================================================

            /**
             * @brief Factorize SPD matrix A = L·L^T (in-place, lower triangle)
             * @param A SPD matrix (will be modified - lower triangle becomes L)
             * @param lda Leading dimension of A
             * @return 0 if successful, k>0 if not positive definite at position k
             */
            int factorize(double* A, size_t lda);
            
            /**
             * @brief Factorize SPD matrix A = L·L^T (Eigen interface)
             * @param A SPD matrix (will be modified - lower triangle becomes L)
             * @return 0 if successful, k>0 if not positive definite at position k
             */
            int factorize(Eigen::MatrixXd& A);
            
            /**
             * @brief Solve using pre-computed factorization
             * @param A Factorized matrix from factorize() (contains L in lower triangle)
             * @param x Solution vector(s) (input: RHS, output: solution)
             * @param b Right-hand side vector(s)
             * @param nrhs Number of right-hand sides (default 1)
             * @return true if successful
             */
            bool solve_factorized(const double* A, double* x, const double* b, size_t nrhs = 1) const;
            
            /**
             * @brief Solve system Ax = b using Cholesky (combined factorization + solve)
             * @param A SPD coefficient matrix (will be modified)
             * @param x Solution vector (input: initial guess, output: solution)
             * @param b Right-hand side vector
             * @return true if successful, false if not positive definite
             */
            bool solve(Eigen::MatrixXd& A, Eigen::VectorXd& x, const Eigen::VectorXd& b);
            
            /**
             * @brief Solve system AX = B for multiple right-hand sides
             * @param A SPD coefficient matrix (will be modified)
             * @param X Solution matrix (input: initial guess, output: solutions)
             * @param B Right-hand side matrix
             * @return true if successful, false if not positive definite
             */
            bool solve(Eigen::MatrixXd& A, Eigen::MatrixXd& X, const Eigen::MatrixXd& B);
            
            /**
             * @brief Combined solve (matrix remains unmodified)
             * @param A SPD coefficient matrix (const - will copy internally)
             * @param x Solution vector
             * @param b Right-hand side vector
             * @return true if successful, false if not positive definite
             */
            bool solve_system(const Eigen::MatrixXd& A, Eigen::VectorXd& x, const Eigen::VectorXd& b);

            // ============================================================================
            // PERFORMANCE AND DIAGNOSTICS
            // ============================================================================
            
            /**
             * @brief Get factorization performance in GFLOPS
             */
            double getFactorizationGFLOPs() const;
            
            /**
             * @brief Get factorization time in seconds
             */
            double getFactorizeTime() const { return factorize_time_; }
            
            /**
             * @brief Get solve time in seconds
             */
            double getSolveTime() const { return solve_time_; }
            
            /**
             * @brief Get matrix size
             */
            size_t size() const { return matrix_size_; }
            
            /**
             * @brief Get current block size
             */
            size_t getBlockSize() const { return block_size_; }

        private:
            // ============================================================================
            // PRIVATE MEMBER VARIABLES
            // ============================================================================
            
            size_t matrix_size_;           /// Matrix dimension
            size_t block_size_;            /// Block size for optimization
            bool use_accelerate_;          /// Whether to use Apple Accelerate

            // Storage
            std::vector<double> work_buffer_;   /// Temporary workspace
            
            // Performance tracking
            mutable double factorize_time_;     /// Factorization time (seconds)
            mutable double solve_time_;         /// Solve time (seconds)

            // ============================================================================
            // PRIVATE MEMBER FUNCTIONS
            // ============================================================================
            
            /**
             * @brief Initialize solver configuration
             */
            void initialize();
            
            /**
             * @brief Determine optimal block size for Cholesky on current architecture
             */
            size_t determine_optimal_block_size() const;
            
            /**
             * @brief Configure BLAS threading for ARM64
             */
            void configure_blas_threading() const;
            
            // Core algorithm components
            int blocked_cholesky(double* A, size_t lda);
            int factor_diagonal_block(double* A, size_t nb, size_t lda, size_t offset);
            void solve_triangular_block(double* A, size_t lda, size_t k, size_t nb);
            void symmetric_rank_update(double* A, size_t lda, size_t k, size_t nb);
            
            // Solve components
            bool triangular_solve_cholesky(const double* L, double* x, size_t nrhs) const;
            
            // Utilities
            void validate_dimensions(size_t rows, size_t cols) const;
            bool is_matrix_valid(const double* A, size_t lda) const;
    };
}

#endif