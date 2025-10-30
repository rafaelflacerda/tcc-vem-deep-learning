#ifndef POLIVEM_POWER_METHOD_HPP
#define POLIVEM_POWER_METHOD_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <thread>

#ifdef USE_ACCELERATE
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
    #ifdef USE_LAPACK
        #include <lapack.h>
    #endif
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "utils/scope_timer.hpp"

namespace LinearAlgebra {
    /**
     * @brief High-performance Power Method implementation for dominant eigenvalue computation
     * 
     * Optimized for dense matrices with Apple Accelerate and Level 3 BLAS operations.
     * Designed to outperform Eigen's general eigenvalue solvers for specific use cases.
     */
    class PowerMethod {
        public:
            // ============================================================================
            // RESULT STRUCTURE
            // ============================================================================
            
            struct Result {
                double eigenvalue = 0.0;
                std::vector<double> eigenvector;
                int iterations = 0;
                double residual = 0.0;
                bool converged = false;
                double computation_time = 0.0;
                double initial_residual = 0.0;
                double convergence_rate = 0.0;
            };
    
            struct Config {
                double tolerance = 1e-10;
                double relative_tolerance = 1e-12;
                int max_iterations = 1000;
                bool use_block_method = false;
                size_t block_size = 4;
                bool verbose = false;
                bool normalize_eigenvector = true;
                bool use_rayleigh_quotient = true;
                double shift = 0.0;  // For shifted Power Method
            };
    
            // ============================================================================
            // CONSTRUCTORS AND INITIALIZATION
            // ============================================================================
            
            /**
             * @brief Construct Power Method solver
             * @param n Matrix size
             * @param block_size Block size for Level 3 BLAS (0 = auto-detect)
             */
            explicit PowerMethod(size_t n, size_t block_size = 0);
            
            /**
             * @brief Destructor
             */
            ~PowerMethod() = default;
    
            // Non-copyable but movable
            PowerMethod(const PowerMethod&) = delete;
            PowerMethod& operator=(const PowerMethod&) = delete;
            PowerMethod(PowerMethod&&) = default;
            PowerMethod& operator=(PowerMethod&&) = default;
    
            // ============================================================================
            // MAIN SOLVER INTERFACES
            // ============================================================================
            
            /**
             * @brief Solve standard eigenvalue problem Av = λv
             * @param A Matrix data (column-major)
             * @param lda Leading dimension of A
             * @param config Solver configuration
             * @return Result structure with eigenvalue and eigenvector
             */
            Result solve(const double* A, size_t lda);
            Result solve(const double* A, size_t lda, const Config& config);
            
            /**
             * @brief Solve generalized eigenvalue problem Av = λBv
             * @param A Matrix A data (column-major)
             * @param B Matrix B data (column-major)
             * @param lda Leading dimension
             * @param config Solver configuration
             * @return Result structure with eigenvalue and eigenvector
             */
            Result solve_generalized(const double* A, const double* B, size_t lda);
            Result solve_generalized(const double* A, const double* B, size_t lda, 
                                     const Config& config);
    
            /**
             * @brief Solve with Eigen matrices (convenience interface)
             * @param A Input matrix
             * @param config Solver configuration
             * @return Result structure
             */
            Result solve(const Eigen::MatrixXd& A);
            Result solve(const Eigen::MatrixXd& A, const Config& config);
            
            /**
             * @brief Solve generalized problem with Eigen matrices
             * @param A Matrix A
             * @param B Matrix B
             * @param config Solver configuration
             * @return Result structure
             */
            Result solve_generalized(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);
            Result solve_generalized(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, 
                                     const Config& config);
    
            /**
             * @brief Solve VEM eigenvalue problems (sparse to dense conversion)
             * @param A_h Sparse stiffness matrix
             * @param M_h Sparse mass matrix
             * @param config Solver configuration
             * @return Result structure
             */
            Result solve_vem_eigenvalue(const Eigen::SparseMatrix<double>& A_h,
                                        const Eigen::SparseMatrix<double>& M_h);
            Result solve_vem_eigenvalue(const Eigen::SparseMatrix<double>& A_h,
                                        const Eigen::SparseMatrix<double>& M_h,
                                        const Config& config);
    
            // ============================================================================
            // CONFIGURATION AND UTILITIES
            // ============================================================================
            
            /**
             * @brief Get optimal block size for current matrix
             * @return Recommended block size
             */
            size_t get_optimal_block_size() const { return block_size_; }
            
            /**
             * @brief Check if using Apple Accelerate
             * @return True if Accelerate is available and in use
             */
            bool is_using_accelerate() const { return use_accelerate_; }
            
            /**
             * @brief Get memory requirements for matrix size
             * @return Memory needed in MB
             */
            double get_memory_requirements() const;
    
        private:
            // ============================================================================
            // PRIVATE MEMBER VARIABLES
            // ============================================================================
            
            // Core configuration
            size_t matrix_size_;
            size_t block_size_;
            bool use_accelerate_;
            
            // Workspace vectors (pre-allocated)
            mutable std::vector<double> work_vector_;
            mutable std::vector<double> prev_vector_;
            mutable std::vector<double> temp_vector_;
            mutable std::vector<double> residual_vector_;
            
            // Block method workspace
            mutable std::vector<double> block_vectors_;
            mutable std::vector<double> block_result_;
            mutable std::vector<double> tau_;  // For QR factorization
            mutable std::vector<double> qr_work_;
            
            // Convergence monitoring
            mutable std::vector<double> eigenvalue_history_;
            
            // Random number generation
            mutable std::mt19937 rng_;
            
            // ============================================================================
            // INITIALIZATION METHODS
            // ============================================================================
            
            void initialize();
            void initialize_workspace();
            size_t determine_optimal_block_size() const;
            void configure_accelerate();
            void configure_blas_threading() const;
            
            // ============================================================================
            // CORE ALGORITHM IMPLEMENTATIONS
            // ============================================================================
            
            Result solve_standard_method(const double* A, size_t lda, const Config& config) const;
            Result solve_block_method(const double* A, size_t lda, const Config& config) const;
            Result solve_shifted_method(const double* A, size_t lda, const Config& config) const;
            
            // ============================================================================
            // BLAS/LAPACK OPERATIONS
            // ============================================================================
            
            void matrix_vector_multiply(const double* A, size_t lda, const double* x, double* y) const;
            void matrix_matrix_multiply(const double* A, size_t lda, const double* X, 
                                       size_t nx, double* Y) const;
            double vector_dot_product(const double* x, const double* y) const;
            double vector_norm(const double* x) const;
            void vector_scale(double* x, double alpha) const;
            void vector_copy(const double* src, double* dst) const;
            void vector_axpy(double alpha, const double* x, double* y) const;
            
            // QR factorization for block method
            int qr_factorization(double* A, size_t lda, size_t ncols, double* tau) const;
            int extract_q_matrix(double* A, size_t lda, size_t ncols, const double* tau) const;
            
            // ============================================================================
            // UTILITY AND VALIDATION METHODS
            // ============================================================================
            
            void validate_matrix(const double* A, size_t lda) const;
            void validate_config(const Config& config) const;
            bool is_matrix_valid(const double* A, size_t lda) const;
            
            void initialize_random_vector(double* v) const;
            void initialize_random_block(double* V, size_t ncols) const;
            void normalize_vector(double* v) const;
            void orthogonalize_block(double* V, size_t lda, size_t ncols) const;
            
            // Convergence checking
            bool check_convergence(double current_eigenvalue, double previous_eigenvalue,
                                  double residual, const Config& config, int iteration) const;
            double compute_residual(const double* A, size_t lda, const double* v, 
                                   double eigenvalue) const;
            double compute_relative_error(double current, double previous) const;
            
            // Result creation and management
            Result create_result(double eigenvalue, const double* eigenvector, 
                               int iterations, bool converged, double computation_time,
                               double residual, const Config& config) const;
            
            void update_convergence_history(double eigenvalue) const;
            double estimate_convergence_rate() const;
            
            // Memory and performance optimization
            void prefetch_matrix_data(const double* A, size_t lda) const;
            void optimize_memory_layout() const;
            bool should_use_block_method(const Config& config) const;
            
            // Debugging and logging
            void log_iteration(int iteration, double eigenvalue, double residual, 
                              const Config& config) const;
            void log_final_result(const Result& result, const Config& config) const;
        };
    
    
}

#endif