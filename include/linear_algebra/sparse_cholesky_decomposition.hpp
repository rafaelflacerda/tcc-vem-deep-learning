#ifndef POLIVEM_SPARSE_CHOLESKY_DECOMPOSITION_HPP
#define POLIVEM_SPARSE_CHOLESKY_DECOMPOSITION_HPP

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
#include "linear_algebra/sparse_gauss_elimination.hpp"
#include "linear_algebra/cholesky_decomposition.hpp"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

namespace LinearAlgebra {
    /**
     * @brief Sparse supernodal Cholesky decomposition for SPD matrices
     * 
     * Implements supernodal Cholesky factorization A = L·L^T for sparse SPD matrices
     * using the same supernode infrastructure as sparse LU but with simplified algorithm
     * (no pivoting required for SPD matrices).
     */
    class SparseCholeskyDecomposition {

        public:
            // ============================================================================
            // CONSTRUCTORS AND DESTRUCTOR
            // ============================================================================
            
            SparseCholeskyDecomposition();
            ~SparseCholeskyDecomposition() = default;
            
            // Disable copy operations
            SparseCholeskyDecomposition(const SparseCholeskyDecomposition&) = delete;
            SparseCholeskyDecomposition& operator=(const SparseCholeskyDecomposition&) = delete;
            
            // Enable move operations
            SparseCholeskyDecomposition(SparseCholeskyDecomposition&&) = default;
            SparseCholeskyDecomposition& operator=(SparseCholeskyDecomposition&&) = default;


            // ============================================================================
            // MAIN SOLVER INTERFACE
            // ============================================================================
            
            /**
             * @brief Analyze sparsity pattern (reusable for matrices with same pattern)
             * @param A Sparse SPD matrix in CSC format
             * @return true if analysis successful
             */
            bool analyze(const CSCMatrix& A);
            bool analyze(const Eigen::SparseMatrix<double>& A);
            
            /**
             * @brief Numerical factorization A = L·L^T
             * @param A Sparse SPD matrix (same pattern as analyzed)
             * @return 0 if successful, k>0 if not positive definite at position k
             */
            int factorize(const CSCMatrix& A);
            int factorize(const Eigen::SparseMatrix<double>& A);
            
            /**
             * @brief Solve using factorization L·L^T·x = b
             * @param x Solution vector (input: initial guess, output: solution)
             * @param b Right-hand side vector
             * @return true if successful
             */
            bool solve(Eigen::VectorXd& x, const Eigen::VectorXd& b) const;
            bool solve(Eigen::MatrixXd& X, const Eigen::MatrixXd& B) const;
            
            /**
             * @brief Combined analyze + factorize + solve
             * @param A Sparse SPD coefficient matrix
             * @param x Solution vector
             * @param b Right-hand side vector
             * @return true if successful
             */
            bool solve_system(const Eigen::SparseMatrix<double>& A, 
                            Eigen::VectorXd& x, const Eigen::VectorXd& b);

            
            // ============================================================================
            // CONFIGURATION
            // ============================================================================
            
            /**
             * @brief Set maximum number of threads
             * @param max_threads Maximum threads (0 = auto-detect)
             */
            void set_max_threads(int max_threads) { max_threads_ = max_threads; }
            
            /**
             * @brief Enable/disable row scaling for numerical stability
             * @param enable True to enable row/column scaling
             */
            void set_scaling(bool enable) { use_scaling_ = enable; }
            
            /**
             * @brief Set supernode merging threshold
             * @param threshold Minimum supernode size for merging
             */
            void set_supernode_threshold(int threshold) { supernode_threshold_ = threshold; }

            // ============================================================================
            // PERFORMANCE AND DIAGNOSTICS
            // ============================================================================
            double getFactorizationGFLOPs() const;
            
            size_t getFactorNNZ() const { return factor_nnz_; }
            size_t getOriginalNNZ() const { return original_nnz_; }
            double getFillRatio() const { return static_cast<double>(factor_nnz_) / original_nnz_; }
            
            int getNumSupernodes() const { return static_cast<int>(supernodes_.size()); }

        private:
            // ============================================================================
            // CHOLESKY-SPECIFIC SUPERNODE STRUCTURE
            // ============================================================================
            
            /**
             * @brief Supernode structure optimized for Cholesky (no pivoting)
             */
            struct CholeskySupernode {
                int start_col;                   ///< First column in supernode
                int num_cols;                    ///< Number of columns (width)
                std::vector<int> row_structure;  ///< Row indices in supernode pattern
                int front_height;                ///< Height of frontal matrix
                int front_width;                 ///< Width of frontal matrix (num_cols)
                
                // Frontal matrix storage (symmetric, only lower triangle used)
                std::vector<double> frontal_matrix;  ///< Dense frontal matrix
                int front_lda;                       ///< Leading dimension of frontal matrix
                
                // Parent/child relationships in elimination tree
                std::vector<int> children;       ///< Child supernodes
                int parent;                      ///< Parent supernode (-1 if root)
                
                // Symmetric update contributions from children (simpler than LU)
                struct SymmetricUpdate {
                    std::vector<int> indices;    ///< Global row/col indices
                    std::vector<double> values;  ///< Lower triangular update matrix
                    int size;                    ///< Update matrix dimension (square)
                };
                std::vector<SymmetricUpdate> child_updates; ///< Updates from child supernodes
            };
            
            // ============================================================================
            // PRIVATE MEMBER VARIABLES
            // ============================================================================
            
            // Problem size and structure
            int n_;                          ///< Matrix dimension
            bool analyzed_;                  ///< Whether analysis is complete
            bool factorized_;               ///< Whether factorization is complete
            
            // Sparse structure (reuse from sparse LU)
            std::unique_ptr<EliminationTree> elim_tree_;
            std::vector<CholeskySupernode> supernodes_;
            std::vector<int> col_to_supernode_;  ///< Mapping from columns to supernodes
            
            // Ordering (simpler than LU - only column permutation needed)
            std::vector<int> col_perm_;      ///< Column permutation for fill reduction
            std::vector<int> col_perm_inv_;  ///< Inverse column permutation
            
            // Factorization storage (sparse L factor only - no U needed)
            CSCMatrix L_factor_;             ///< Lower triangular Cholesky factor
            
            // Configuration parameters
            int max_threads_;               ///< Maximum number of threads
            bool use_scaling_;              ///< Use row/column scaling
            int supernode_threshold_;       ///< Minimum supernode size for merging
            
            // Scaling vectors (if enabled)
            std::vector<double> scale_factors_; ///< Diagonal scaling factors
            
            // Dense Cholesky kernel for frontal matrices
            std::unique_ptr<CholeskyDecomposition> dense_kernel_;
            
            // Performance tracking
            size_t original_nnz_;           ///< Original matrix non-zeros
            size_t factor_nnz_;             ///< Factor non-zeros

            // ============================================================================
            // PRIVATE MEMBER FUNCTIONS
            // ============================================================================
            
            // Analysis phase (reuse and adapt from sparse LU)
            void compute_column_ordering(const CSCMatrix& A);
            void build_elimination_tree(const CSCMatrix& A);
            void detect_supernodes_cholesky(const CSCMatrix& A);
            void analyze_memory_requirements();
            
            // Cholesky-specific ordering (simpler than LU)
            std::vector<int> amd_ordering_cholesky(const CSCMatrix& A);
            
            // Supernode operations (adapted from sparse LU)
            void merge_supernodes_cholesky();
            bool can_merge_columns_cholesky(int col1, int col2, const CSCMatrix& A);
            void compute_supernode_structure_cholesky(const CSCMatrix& A);
            
            // Factorization phase
            void initialize_factorization(const CSCMatrix& A);
            void scale_matrix_symmetric(CSCMatrix& A);
            int factor_all_supernodes_cholesky(const CSCMatrix& A);
            int factor_supernode_cholesky(int snode_id, const CSCMatrix& A);
            
            // Frontal matrix operations (simplified for Cholesky)
            void assemble_frontal_matrix_symmetric(int snode_id, const CSCMatrix& A);
            void add_child_contributions_symmetric(int snode_id);
            int factor_dense_frontal_cholesky(CholeskySupernode& snode);
            void extract_cholesky_factor_from_frontal(const CholeskySupernode& snode);
            void generate_symmetric_update_matrix(const CholeskySupernode& snode);
            
            // Solve phase (simpler than LU - no permutation of rows)
            void apply_column_permutation(Eigen::VectorXd& x, bool forward) const;
            void forward_substitution_sparse(Eigen::VectorXd& x) const;
            void backward_substitution_sparse(Eigen::VectorXd& x) const;
            void apply_scaling_symmetric(Eigen::VectorXd& x, bool forward) const;
            
            // Utilities
            void validate_spd_matrix(const CSCMatrix& A) const;
            void clear_factorization();
            
            // Cholesky-specific helper functions
            CSCMatrix create_lower_triangular_pattern(const CSCMatrix& A) const;
            CSCMatrix apply_column_permutation_to_matrix(const CSCMatrix& A) const;
            void build_elimination_tree_cholesky(const CSCMatrix& A_lower);
            
            // Simplified sparse Cholesky methods (following sparse LU pattern)
            bool solve_sparse_cholesky_direct(const Eigen::SparseMatrix<double>& A,
                                               Eigen::VectorXd& x, const Eigen::VectorXd& b);
            bool solve_sparse_cholesky_optimized(const Eigen::SparseMatrix<double>& A,
                                                  Eigen::VectorXd& x, const Eigen::VectorXd& b);
            int factorize_sparse_cholesky_direct(const CSCMatrix& A);
            int factorize_sparse_cholesky_optimized(const CSCMatrix& A);
            void forward_substitution_simple(Eigen::VectorXd& x) const;
            void backward_substitution_simple(Eigen::VectorXd& x) const;

    };
}

#endif