#ifndef POLIVEM_GAUSS_ELIMINATION_HPP
#define POLIVEM_GAUSS_ELIMINATION_HPP

#include <vector>
#include <chrono>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <iostream>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

namespace LinearAlgebra {
    class GaussElimination {
        public:
            // ============================================================================
            // CONSTRUCTORS AND DESTRUCTOR
            // ============================================================================

            /**
             * @brief Construct GEPP solver for given matrix size
             * @param n Matrix dimension
             * @param block_size Block size for optimization (0 = auto-tune)
             */
            explicit GaussElimination(size_t n, size_t block_size = 0);
            
            /**
             * @brief Default destructor
             */
            ~GaussElimination() = default;
            
            // Disable copy operations (heavy objects)
            GaussElimination(const GaussElimination&) = delete;
            GaussElimination& operator=(const GaussElimination&) = delete;
            
            // Enable move operations
            GaussElimination(GaussElimination&&) = default;
            GaussElimination& operator=(GaussElimination&&) = default;

            // ============================================================================
            // MAIN SOLVER INTERFACE
            // ============================================================================
            
            /**
             * @brief Solve system Ax = b using GEPP (combined factorization + solve)
             * @param A Coefficient matrix (will be modified)
             * @param x Solution vector (input: initial guess, output: solution)
             * @param b Right-hand side vector
             * @return true if successful, false if singular matrix
             */
            bool solve(Eigen::MatrixXd& A, Eigen::VectorXd& x, const Eigen::VectorXd& b);
            
            /**
             * @brief Solve system Ax = B for multiple right-hand sides
             * @param A Coefficient matrix (will be modified)
             * @param X Solution matrix (input: initial guess, output: solutions)
             * @param B Right-hand side matrix
             * @return true if successful, false if singular matrix
             */
            bool solve(Eigen::MatrixXd& A, Eigen::MatrixXd& X, const Eigen::MatrixXd& B);
            
            /**
             * @brief In-place factorization (for advanced users)
             * 
             * TODO: Implement better profiling for factorization time
             * 
             * @param A Matrix to factorize (n x n, column-major)
             * @param lda Leading dimension of A
             * @return 0 if successful, k>0 if singular at position k
             */
            int factorize(double* A, size_t lda);
            
            /**
             * @brief Solve using pre-computed factorization
             * @param A Factorized matrix from factorize()
             * @param x Solution vector(s)
             * @param b Right-hand side vector(s)
             * @param nrhs Number of right-hand sides
             * @return true if successful
             */
            bool solve_factorized(const double* A, double* x, const double* b, size_t nrhs = 1) const;

            // ============================================================================
            // PERFORMANCE AND DIAGNOSTICS
            // ============================================================================
            
            /**
             * @brief Get factorization performance in GFLOPS
             */
            double getFactorizationGFLOPs() const;
            
            /**
             * @brief Get matrix size
             */
            size_t size() const { return matrix_size_; }
            
            /**
             * @brief Get current block size
             */
            size_t getBlockSize() const { return block_size_; }
            
            /**
             * @brief Get pivot indices from last factorization (1-based)
             */
            const std::vector<int>& getPivots() const { return pivot_indices_; }

        private:
            // ============================================================================
            // PRIVATE MEMBER VARIABLES
            // ============================================================================
            
            size_t matrix_size_;           ///< Matrix dimension
            size_t block_size_;            ///< Block size for optimization
            bool use_accelerate_;          ///< Whether to use Apple Accelerate
            
            // Storage
            std::vector<int> pivot_indices_;    ///< Pivot permutation (1-based)
            std::vector<double> work_buffer_;   ///< Temporary workspace

            // ============================================================================
            // PRIVATE MEMBER FUNCTIONS
            // ============================================================================
            
            /**
             * @brief Initialize solver configuration
             */
            void initialize();
            
            /**
             * @brief Find the best block size for matrix operations that fits 
             * in Apple Silicon's 4MB L2 cache.
             * 
             * Step 1. Divide cache by 3 (need space for matrices A, B, and C)
             * Step 2. Take square root (since we use square blocks)
             * Step 3. Round to multiple of 8 (ARM64 SIMD optimization)
             * Step 4. Clamp between 96-256 (avoid too small/large blocks)
             */
            size_t determine_optimal_block_size() const;
            
            /**
             * @brief Configure the optimal number of threads for BLAS operations 
             * to avoid overwhelming the memory system.
             * 
             * Step 1. Get number of hardware threads
             * Step 2. Clamp to 8 (reasonable limit)
             * Step 3. Configure BLAS environment variable
             */
            void configure_blas_threading() const;
            
            // Core algorithm components
            /**
             * @brief Blocked GEPP algorithm for matrix factorization.
             * 
             * This function performs the following operations:
             * 1. Panel factorization (Level-2 BLAS, but small)
             * 2. Apply pivots to right blocks
             * 3. Update trailing matrix (Level-3 BLAS - CRITICAL PATH)
             * 
             * @param A Matrix to factorize
             * @param lda Leading dimension of A
             * @return 0 if successful, k>0 if singular at position k.
             */
            int blocked_gepp(double* A, size_t lda);

            /**
             * @brief Panel factorization of matrix A with partial and scaling pivoting.
             * 
             * After factorization, matrix A contains both L and U factors stored in the
             * same memory location. The upper triangular part is stored above and on the
             * diagonal. The lower triangular part is stored below the diagonal (with 1s
             * implicitly on the diagonal). The pivot indices are stored in pivot_indices_.
             * 
             * @param A Matrix to factorize
             * @param lda Leading dimension of A
             * @param k Starting column index
             * @param jb Block size
             * @return 0 if successful, k>0 if singular at position k.
             */
            int panel_factorization(double* A, size_t lda, size_t k, size_t jb);

            /**
             * @brief Apply row interchanges to columns col_start:col_start+ncols-1
             * using pivots computed in panel k:k+jb-1
             * 
             * @param A Matrix to apply pivots to
             * @param lda Leading dimension of A
             * @param col_start Starting column index
             * @param ncols Number of columns to apply pivots to
             * @param k Starting row index
             * @param jb Block size
             */
            void apply_pivots(
                double* A, 
                size_t lda, 
                size_t col_start, 
                size_t ncols, 
                size_t k, 
                size_t jb
            ) const;
            
            /**
             * @brief Update trailing matrix A₂₂ after panel factorization.
             * 
             * This function performs the following operations:
             * 1. Triangular solve L₁₁ U₁₂ = A₁₂ (TRSM - Level-3 BLAS)
             * 2. Matrix multiply A₂₂ ← A₂₂ - L₂₁ U₁₂ (GEMM - Level-3 BLAS)
             * 
             * @param A Matrix to update
             * @param lda Leading dimension of A
             * @param k Starting column index
             * @param jb Block size
             */
            void update_trailing_matrix(double* A, size_t lda, size_t k, size_t jb) const;
            
            // Solve components  
            /**
             * @brief Apply row permutation P to right-hand side
             * @param x Right-hand side vector(s)
             * @param nrhs Number of right-hand sides
             */
            void apply_permutation(double* x, size_t nrhs) const;

            /**
             * @brief Solve using pre-computed factorization
             * Forward substitution: solve L * y = P * b
             * Backward substitution: solve U * x = y  
             * 
             * @param A Factorized matrix from factorize()
             * @param x Solution vector(s)
             * @param b Right-hand side vector(s)
             * @param nrhs Number of right-hand sides
             * @return true if successful
             */
            bool triangular_solves(const double* A, double* x, size_t nrhs) const;
            
            // Utilities
            void validate_dimensions(size_t rows, size_t cols) const;
            bool is_matrix_valid(const double* A, size_t lda) const;

    };
}

#endif // POLIVEM_GAUSS_ELIMINATION_HPP