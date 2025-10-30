#ifndef POLIVEM_SPARSE_GAUSS_ELIMINATION_HPP
#define POLIVEM_SPARSE_GAUSS_ELIMINATION_HPP

#include <vector>
#include <memory>
#include <map>
#include <set>
#include <queue>
#include <chrono>
#include <numeric>  
#include <unordered_set>
#include <algorithm>
#include <cstdlib>  // for setenv
#include <string>   // for std::to_string
#include <thread>   // for std::thread::hardware_concurrency 
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "linear_algebra/gauss_elimination.hpp"
#include "utils/scope_timer.hpp"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
    #include <lapacke.h>
#endif

namespace LinearAlgebra {
    /**
     * @brief Compressed Sparse Column (CSC) matrix structure
     * 
     * The CSC format stores sparse matrices efficiently by:
     * - values: Array of non-zero matrix entries
     * - row_indices: Row index for each non-zero value
     * - col_pointers: Starting index of each column in the values array
     * 
     * This format is optimal for column-wise operations and matrix-vector products.
     */
    struct CSCMatrix {
        std::vector<double> values;      ///< Non-zero values
        std::vector<int> row_indices;    ///< Row indices for each value
        std::vector<int> col_pointers;   ///< Column start pointers
        int n_rows;                      ///< Number of rows
        int n_cols;                      ///< Number of columns
        int nnz;                         ///< Number of non-zeros
        
        CSCMatrix() : n_rows(0), n_cols(0), nnz(0) {}
        CSCMatrix(const Eigen::SparseMatrix<double>& eigen_sparse);

        /**
         * @brief Convert an Eigen sparse matrix to a CSC matrix
         * 
         * This conversion preserves the sparsity pattern and numerical values
         * while optimizing the storage format for our sparse factorization algorithms.
         * 
         * @param eigen_sparse The Eigen sparse matrix to convert
         */
        void convert_from_eigen(const Eigen::SparseMatrix<double>& eigen_sparse);
    };

    /**
     * @brief Supernode structure for sparse factorization
     * 
     * Supernodes group columns with similar sparsity patterns to improve
     * cache performance and enable dense matrix operations during factorization.
     * 
     * Mathematical Background:
     * A supernode is a set of consecutive columns {j, j+1, ..., j+w-1} where:
     * - The sparsity patterns of columns j+1, ..., j+w-1 are subsets of column j
     * - This allows dense operations on w×w blocks during factorization
     * - Improves cache locality and enables BLAS-3 operations
     */
    struct Supernode {
        int start_col;                   ///< First column in supernode
        int num_cols;                    ///< Number of columns (width)
        std::vector<int> row_structure;  ///< Row indices in supernode pattern
        int front_height;                ///< Height of frontal matrix
        int front_width;                 ///< Width of frontal matrix (num_cols)
        
        // Frontal matrix storage
        std::vector<double> frontal_matrix;  ///< Dense frontal matrix
        int front_lda;                       ///< Leading dimension of frontal matrix
        
        // Parent/child relationships in elimination tree
        std::vector<int> children;       ///< Child supernodes
        int parent;                      ///< Parent supernode (-1 if root)
        
        // Update contributions from children
        struct Update {
            std::vector<int> indices;    ///< Global row/col indices
            std::vector<double> values;  ///< Update matrix values
            int width;                   ///< Update matrix dimensions
            int height;
        };
        std::vector<Update> child_updates; ///< Updates from child supernodes
    };

    /**
     * @brief Elimination tree for sparse factorization
     * 
     * The elimination tree shows which columns depend on which other columns 
     * during factorization. If we eliminate column j, it affects columns that 
     * depend on j. The tree captures these dependencies.
     * 
     * Mathematical Formulation:
     * For a matrix A, the elimination tree T is defined by:
     * - T[j] = parent of column j in the elimination tree
     * - If column k depends on column j during factorization, then k is a descendant of j
     * - The tree structure determines the optimal elimination order
     * 
     * Properties:
     * - Tree structure enables parallel elimination of independent subtrees
     * - Postorder traversal gives optimal elimination sequence
     * - Height of tree determines parallel depth of factorization
     */
    class EliminationTree {
        public:
            std::vector<int> parent;         ///< Parent of each column
            std::vector<std::vector<int>> children; ///< Children of each column
            int n;                           ///< Matrix size
            
            EliminationTree(int size) : parent(size, -1), children(size), n(size) {}

            /**
             * @brief Build the elimination tree from the pattern of the matrix
             * 
             * Algorithm:
             * 1. Analyzes the sparsity pattern of matrix A
             * 2. Determines column dependencies based on non-zero structure
             * 3. Builds parent-child relationships in the elimination tree
             * 
             * Mathematical Details:
             * For each column j, find the first column k > j that depends on j:
             * - If A[k,j] ≠ 0, then k depends on j
             * - The parent of j is the smallest such k
             * - This creates a tree structure where dependencies flow upward
             * 
             * @param A The matrix to build the elimination tree from
             */
            void build_from_pattern(const CSCMatrix& A);

            /**
             * @brief Get the postorder of the elimination tree
             * 
             * The postorder traversal ensures that all children are processed
             * before their parents, which is optimal for sparse factorization.
             * 
             * Algorithm:
             * 1. Initiates depth-first search from each unvisited node
             * 2. Ensures all nodes are visited (handles disconnected components)
             * 3. Returns elimination order for sparse factorization
             * 
             * Mathematical Significance:
             * - Postorder ensures that when we eliminate column j, all columns
             *   that depend on j have already been eliminated
             * - This minimizes fill-in during factorization
             * - Enables efficient parallel processing of independent subtrees
             * 
             * @return The postorder of the elimination tree
             */
            std::vector<int> get_postorder() const;
        private:
            /**
             * @brief Perform a depth-first search (DFS) to get the postorder of the elimination tree
             * 
             * Recursive DFS implementation that builds the postorder traversal:
             * 1. Marks current node as visited
             * 2. Recursively processes all children (depth-first)
             * 3. Adds current node to order after all children
             * 
             * @param node The current node
             * @param visited A vector to keep track of visited nodes
             * @param order The postorder of the elimination tree
             */
            void dfs_postorder(int node, std::vector<bool>& visited, std::vector<int>& order) const;
        };

        class SparseGaussElimination {
            public:
                // ============================================================================
                // CONSTRUCTORS AND DESTRUCTOR
                // ============================================================================
                
                /**
                 * @brief Default constructor for sparse Gauss elimination solver
                 * 
                 * Initializes the solver with default parameters:
                 * - Pivot threshold: 0.1 (good balance between stability and performance)
                 * - Auto-threading: Uses hardware_concurrency()
                 * - Scaling enabled: Improves numerical stability
                 */
                SparseGaussElimination();
                
                /**
                 * @brief Destructor
                 * 
                 * Automatically cleans up allocated memory for factorization storage.
                 */
                ~SparseGaussElimination() = default;
                
                // Disable copy operations
                SparseGaussElimination(const SparseGaussElimination&) = delete;
                SparseGaussElimination& operator=(const SparseGaussElimination&) = delete;
                
                // Enable move operations
                SparseGaussElimination(SparseGaussElimination&&) = default;
                SparseGaussElimination& operator=(SparseGaussElimination&&) = default;
                
                // ============================================================================
                // MAIN SOLVER INTERFACE
                // ============================================================================
                
                /**
                 * @brief Analyze sparsity pattern (reusable for matrices with same pattern)
                 * 
                 * The analysis phase is the most expensive part of sparse factorization
                 * and can be reused for multiple matrices with identical sparsity patterns.
                 * 
                 * Algorithm Steps:
                 * 1. Compute column ordering (AMD/COLAMD) to minimize fill-in
                 * 2. Build elimination tree to determine dependencies
                 * 3. Detect supernodes for dense block operations
                 * 4. Analyze memory requirements for factorization
                 * 
                 * Mathematical Background:
                 * - Column ordering minimizes the number of non-zeros in L and U factors
                 * - Supernode detection groups columns for cache-efficient operations
                 * - Memory analysis ensures sufficient workspace allocation
                 * 
                 * Performance Notes:
                 * - Analysis time: O(n²) in worst case, but typically O(n log n)
                 * - Memory overhead: O(n) for elimination tree and supernode structures
                 * - Reusable: Same analysis can be used for multiple factorizations
                 * 
                 * @param A Sparse matrix in CSC format
                 * @return true if analysis successful, false otherwise
                 */
                bool analyze(const CSCMatrix& A);
                
                /**
                 * @brief Analyze sparsity pattern from Eigen sparse matrix
                 * 
                 * Convenience method that converts Eigen sparse matrix to CSC format
                 * and then calls the main analyze method.
                 * 
                 * @param A Eigen sparse matrix
                 * @return true if analysis successful, false otherwise
                 */
                bool analyze(const Eigen::SparseMatrix<double>& A);
                
                /**
                 * @brief Numerical factorization
                 * 
                 * Performs the actual LU factorization of the matrix A = L*U.
                 * This phase requires the matrix to have been analyzed first.
                 * 
                 * Mathematical Formulation:
                 * Factorizes A = P_r * L * U * P_c^T where:
                 * - P_r, P_c are row and column permutation matrices
                 * - L is unit lower triangular
                 * - U is upper triangular
                 * 
                 * Algorithm:
                 * 1. Apply scaling and permutations to A
                 * 2. Factor each supernode using dense operations
                 * 3. Generate update matrices for child contributions
                 * 4. Extract L and U factors from frontal matrices
                 * 
                 * Numerical Stability:
                 * - Uses partial pivoting with threshold-based selection
                 * - Row/column scaling improves condition number
                 * - Supernode structure reduces roundoff error accumulation
                 * 
                 * Performance Characteristics:
                 * - Factorization time: O(n³) worst case, but much better for sparse matrices
                 * - Memory usage: O(n²) in worst case, typically O(n log n)
                 * - Parallel efficiency: Good for matrices with wide elimination trees
                 * 
                 * @param A Sparse matrix (same pattern as analyzed)
                 * @return 0 if successful, k>0 if singular at position k
                 */
                int factorize(const CSCMatrix& A);
                
                /**
                 * @brief Numerical factorization from Eigen sparse matrix
                 * 
                 * Convenience method that converts Eigen sparse matrix to CSC format
                 * and then calls the main factorize method.
                 * 
                 * @param A Eigen sparse matrix
                 * @return 0 if successful, k>0 if singular at position k
                 */
                int factorize(const Eigen::SparseMatrix<double>& A);
                
                /**
                 * @brief Solve using factorization
                 * 
                 * Solves the linear system Ax = b using the precomputed LU factorization.
                 * This is the fastest phase and can be called multiple times with different
                 * right-hand sides.
                 * 
                 * Mathematical Formulation:
                 * Solves P_r * L * U * P_c^T * x = b by:
                 * 1. Forward substitution: L * y = P_r * b
                 * 2. Backward substitution: U * z = y
                 * 3. Apply permutations: x = P_c * z
                 * 
                 * Algorithm:
                 * 1. Apply row permutation to right-hand side
                 * 2. Forward substitution through L factor (level-by-level)
                 * 3. Backward substitution through U factor (level-by-level)
                 * 4. Apply column permutation to solution
                 * 5. Apply scaling factors if enabled
                 * 
                 * Performance Characteristics:
                 * - Solve time: O(n²) worst case, typically O(n log n) for sparse factors
                 * - Memory access: Cache-friendly due to supernode structure
                 * - Parallel efficiency: Good for multiple right-hand sides
                 * 
                 * @param x Solution vector (input: initial guess, output: solution)
                 * @param b Right-hand side vector
                 * @return true if successful, false otherwise
                 */
                bool solve(Eigen::VectorXd& x, const Eigen::VectorXd& b) const;
                
                /**
                 * @brief Solve multiple right-hand sides using factorization
                 * 
                 * Efficiently solves AX = B where X and B are matrices with multiple
                 * columns. This is more efficient than solving each system separately.
                 * 
                 * Mathematical Formulation:
                 * Solves P_r * L * U * P_c^T * X = B by applying the same
                 * forward/backward substitution process to each column of B.
                 * 
                 * Performance Benefits:
                 * - Better cache utilization for multiple right-hand sides
                 * - Reduced overhead from permutation and scaling operations
                 * - Enables BLAS-3 operations for dense blocks
                 * 
                 * @param X Solution matrix (input: initial guess, output: solution)
                 * @param B Right-hand side matrix
                 * @return true if successful, false otherwise
                 */
                bool solve(Eigen::MatrixXd& X, const Eigen::MatrixXd& B) const;
                
                /**
                 * @brief Combined analyze + factorize + solve
                 * 
                 * Convenience method that performs the complete solution process
                 * in one call. This is useful for one-time solves but less efficient
                 * than separate calls when solving multiple systems.
                 * 
                 * Algorithm:
                 * 1. Analyze sparsity pattern of A
                 * 2. Factorize A = L*U
                 * 3. Solve Ax = b using the factorization
                 * 
                 * Use Cases:
                 * - One-time solves where convenience is more important than performance
                 * - Prototyping and testing
                 * - Small to medium-sized problems
                 * 
                 * Performance Notes:
                 * - Analysis overhead is amortized only for single solve
                 * - Memory allocation happens at each call
                 * - No opportunity for factorization reuse
                 * 
                 * @param A Sparse coefficient matrix
                 * @param x Solution vector
                 * @param b Right-hand side vector
                 * @return true if successful, false otherwise
                 */
                bool solve_system(const Eigen::SparseMatrix<double>& A, 
                                Eigen::VectorXd& x, const Eigen::VectorXd& b);
                
                // ============================================================================
                // CONFIGURATION
                // ============================================================================
                
                /**
                 * @brief Set pivoting threshold for numerical stability
                 * 
                 * The pivot threshold controls the trade-off between numerical stability
                 * and performance during factorization.
                 * 
                 * Mathematical Background:
                 * During LU factorization, we select pivot elements that satisfy:
                 * |a_ij| ≥ threshold * max_k |a_kj|
                 * 
                 * Threshold Selection:
                 * - 0.0: No pivoting (fastest, but may be unstable)
                 * - 0.1: Good balance (default, recommended for most problems)
                 * - 0.5: Conservative pivoting (more stable, slower)
                 * - 1.0: Full pivoting (most stable, slowest)
                 * 
                 * Stability vs. Performance:
                 * - Lower threshold: Faster factorization, potential instability
                 * - Higher threshold: More stable, slower factorization
                 * - Default 0.1 provides good stability for most applications
                 * 
                 * @param threshold Pivot threshold (0.0 to 1.0, default 0.1)
                 */
                void set_pivot_threshold(double threshold) { pivot_threshold_ = threshold; }
                
                /**
                 * @brief Set maximum number of threads
                 * 
                 * Controls the level of parallelism used during factorization
                 * and solve phases.
                 * 
                 * Threading Strategy:
                 * - Factorization: Parallel elimination of independent supernodes
                 * - Solve: Parallel processing of multiple right-hand sides
                 * - Memory allocation: Thread-local workspace allocation
                 * 
                 * Performance Considerations:
                 * - Optimal thread count depends on matrix structure and hardware
                 * - Too many threads can cause cache thrashing
                 * - Memory bandwidth often limits scalability
                 * 
                 * @param max_threads Maximum threads (0 = auto-detect)
                 */
                void set_max_threads(int max_threads) { max_threads_ = max_threads; }
                
                /**
                 * @brief Enable/disable row scaling for stability
                 * 
                 * Row and column scaling can significantly improve the condition
                 * number of the matrix and numerical stability of factorization.
                 * 
                 * Scaling Methods:
                 * - Row scaling: Divide each row by its maximum element
                 * - Column scaling: Divide each column by its maximum element
                 * - Combined scaling: Apply both row and column scaling
                 * 
                 * Mathematical Effect:
                 * - Improves condition number κ(A) by balancing row/column norms
                 * - Reduces roundoff error during factorization
                 * - May increase fill-in slightly
                 * 
                 * Performance Impact:
                 * - Minimal overhead during analysis phase
                 * - Slight memory increase for scaling factors
                 * - Often improves convergence for iterative refinement
                 * 
                 * @param enable True to enable row/column scaling
                 */
                void set_scaling(bool enable) { use_scaling_ = enable; }
                
                // ============================================================================
                // PERFORMANCE AND DIAGNOSTICS
                // ============================================================================
                
                /**
                 * @brief Get analysis phase execution time
                 * 
                 * Returns the time spent in the analyze() method for the most
                 * recent analysis operation.
                 * 
                 * @return Analysis time in seconds
                 */
                double getAnalyzeTime() const { return analyze_time_; }
                
                /**
                 * @brief Get factorization phase execution time
                 * 
                 * Returns the time spent in the factorize() method for the most
                 * recent factorization operation.
                 * 
                 * @return Factorization time in seconds
                 */
                double getFactorizeTime() const { return factorize_time_; }
                
                /**
                 * @brief Get solve phase execution time
                 * 
                 * Returns the time spent in the solve() method for the most
                 * recent solve operation.
                 * 
                 * @return Solve time in seconds
                 */
                double getSolveTime() const { return solve_time_; }
                
                /**
                 * @brief Get computational complexity of factorization
                 * 
                 * Estimates the number of floating-point operations performed
                 * during factorization, useful for performance analysis.
                 * 
                 * Mathematical Definition:
                 * GFLOPs = (2 * factor_nnz - n) / 1e9
                 * 
                 * Performance Metrics:
                 * - Factor_nnz: Number of non-zeros in L and U factors
                 * - Fill-in ratio: factor_nnz / original_nnz
                 * - GFLOPs rate: GFLOPs / factorization_time
                 * 
                 * @return Factorization complexity in Giga-FLOPs
                 */
                double getFactorizationGFLOPs() const;
                
                /**
                 * @brief Get number of non-zeros in factors
                 * 
                 * Returns the total number of non-zero elements in the L and U
                 * factors after factorization.
                 * 
                 * @return Number of non-zeros in factors
                 */
                size_t getFactorNNZ() const { return factor_nnz_; }
                
                /**
                 * @brief Get number of non-zeros in original matrix
                 * 
                 * Returns the number of non-zero elements in the original
                 * matrix A.
                 * 
                 * @return Number of non-zeros in original matrix
                 */
                size_t getOriginalNNZ() const { return original_nnz_; }
                
                /**
                 * @brief Get fill-in ratio
                 * 
                 * The fill-in ratio measures how much the sparsity pattern
                 * grows during factorization.
                 * 
                 * Mathematical Definition:
                 * Fill ratio = factor_nnz / original_nnz
                 * 
                 * Interpretation:
                 * - Fill ratio = 1.0: No fill-in (diagonal matrix)
                 * - Fill ratio = 2.0: Doubled non-zeros
                 * - Fill ratio > 10: Significant fill-in (may need reordering)
                 * 
                 * @return Fill-in ratio (factor_nnz / original_nnz)
                 */
                double getFillRatio() const { return static_cast<double>(factor_nnz_) / original_nnz_; }
                
                /**
                 * @brief Get number of supernodes
                 * 
                 * Returns the number of supernodes detected during analysis.
                 * More supernodes generally indicate better cache performance.
                 * 
                 * @return Number of supernodes
                 */
                int getNumSupernodes() const { return static_cast<int>(supernodes_.size()); }
                
            private:
                // ============================================================================
                // PRIVATE MEMBER VARIABLES
                // ============================================================================
                
                // Problem size and structure
                int n_;                          ///< Matrix dimension
                bool analyzed_;                  ///< Whether analysis is complete
                bool factorized_;               ///< Whether factorization is complete
                
                // Sparse structure
                std::unique_ptr<EliminationTree> elim_tree_;
                std::vector<Supernode> supernodes_;
                std::vector<int> col_to_supernode_;  ///< Mapping from columns to supernodes
                
                // Permutation vectors
                std::vector<int> row_perm_;      ///< Row permutation (P_r)
                std::vector<int> col_perm_;      ///< Column permutation (P_c) 
                std::vector<int> row_perm_inv_;  ///< Inverse row permutation
                std::vector<int> col_perm_inv_;  ///< Inverse column permutation
                
                // Factorization storage (sparse L and U)
                CSCMatrix L_factor_;             ///< Lower triangular factor
                CSCMatrix U_factor_;             ///< Upper triangular factor
                
                // Configuration parameters
                double pivot_threshold_;         ///< Pivoting threshold
                int max_threads_;               ///< Maximum number of threads
                bool use_scaling_;              ///< Use row/column scaling
                
                // Scaling vectors (if enabled)
                std::vector<double> row_scale_;  ///< Row scaling factors
                std::vector<double> col_scale_;  ///< Column scaling factors
                
                // Dense frontal operations kernel
                std::unique_ptr<GaussElimination> dense_kernel_;
                
                // Performance tracking
                mutable double analyze_time_;    ///< Analysis time (seconds)
                mutable double factorize_time_;  ///< Factorization time (seconds)
                mutable double solve_time_;      ///< Solve time (seconds)
                size_t original_nnz_;           ///< Original matrix non-zeros
                size_t factor_nnz_;             ///< Factor non-zeros
                
                // ============================================================================
                // PRIVATE MEMBER FUNCTIONS
                // ============================================================================
                
                // Analysis phase
                /**
                 * @brief Compute optimal column ordering to minimize fill-in
                 * 
                 * Applies advanced ordering algorithms (AMD/COLAMD) to reduce the number
                 * of non-zeros introduced during factorization.
                 * 
                 * Mathematical Background:
                 * - Fill-in occurs when eliminating column j affects column k > j
                 * - Good ordering minimizes the number of such dependencies
                 * - AMD (Approximate Minimum Degree) provides near-optimal ordering
                 * 
                 * Algorithm:
                 * 1. Build elimination graph from matrix pattern
                 * 2. Compute approximate minimum degree for each column
                 * 3. Eliminate columns in order of increasing degree
                 * 4. Update degrees of remaining columns
                 * 
                 * Performance Impact:
                 * - Ordering time: O(n log n) typically
                 * - Can reduce fill-in by 2-10x for many matrices
                 * - Critical for large sparse systems
                 * 
                 * @param A Sparse matrix in CSC format
                 */
                void compute_column_ordering(const CSCMatrix& A);
                
                /**
                 * @brief Build elimination tree from matrix sparsity pattern
                 * 
                 * Constructs the elimination tree that shows dependencies between
                 * columns during factorization.
                 * 
                 * Mathematical Definition:
                 * For each column j, find the first column k > j that depends on j:
                 * - If A[k,j] ≠ 0, then k depends on j
                 * - The parent of j is the smallest such k
                 * - Creates tree structure where dependencies flow upward
                 * 
                 * Algorithm:
                 * 1. Scan each column j from left to right
                 * 2. For each non-zero A[k,j], find the first k > j
                 * 3. Set parent[j] = k if k is smaller than current parent
                 * 4. Build children arrays from parent relationships
                 * 
                 * Tree Properties:
                 * - Height determines parallel depth of factorization
                 * - Independent subtrees can be processed in parallel
                 * - Postorder traversal gives optimal elimination sequence
                 * 
                 * @param A Sparse matrix in CSC format
                 */
                void build_elimination_tree(const CSCMatrix& A);
                
                /**
                 * @brief Detect supernodes for dense block operations
                 * 
                 * Groups consecutive columns with similar sparsity patterns to enable
                 * efficient dense matrix operations during factorization.
                 * 
                 * Supernode Definition:
                 * A supernode is a set of consecutive columns {j, j+1, ..., j+w-1} where:
                 * - The sparsity pattern of column j+1 is a subset of column j
                 * - This pattern continues for all columns in the supernode
                 * - Enables dense operations on w×w blocks
                 * 
                 * Detection Algorithm:
                 * 1. Start with column j = 0
                 * 2. Check if column j+1 has subset pattern of column j
                 * 3. Extend supernode while pattern condition holds
                 * 4. Create supernode and move to next ungrouped column
                 * 
                 * Benefits:
                 * - Better cache performance through dense operations
                 * - Enables BLAS-3 operations (matrix-matrix)
                 * - Reduces memory access overhead
                 * 
                 * @param A Sparse matrix in CSC format
                 */
                void detect_supernodes(const CSCMatrix& A);
                
                /**
                 * @brief Analyze memory requirements for factorization
                 * 
                 * Estimates the memory needed for L and U factors, workspace arrays,
                 * and auxiliary data structures.
                 * 
                 * Memory Components:
                 * - L and U factors: O(factor_nnz) storage
                 * - Frontal matrices: O(max_supernode_size²) workspace
                 * - Elimination tree: O(n) storage
                 * - Supernode structures: O(n) storage
                 * 
                 * Estimation Strategy:
                 * 1. Use elimination tree to estimate fill-in
                 * 2. Consider supernode structure for workspace
                 * 3. Account for permutation and scaling arrays
                 * 4. Add safety margin for numerical stability
                 * 
                 * @return Memory estimate in bytes
                 */
                void analyze_memory_requirements();
                
                // Ordering algorithms
                /**
                 * @brief Compute COLAMD ordering for sparse matrices
                 * 
                 * COLAMD (Column Approximate Minimum Degree) is an ordering algorithm
                 * specifically designed for sparse matrices that provides excellent
                 * fill-in reduction with reasonable computational cost.
                 * 
                 * Algorithm Features:
                 * - Approximates minimum degree ordering
                 * - Handles unsymmetric matrices well
                 * - Provides better results than AMD for many problems
                 * - Time complexity: O(n log n) typically
                 * 
                 * Mathematical Principle:
                 * - Computes approximate minimum degree for each column
                 * - Eliminates columns in order of increasing degree
                 * - Updates degrees of remaining columns efficiently
                 * 
                 * @param A Sparse matrix in CSC format
                 * @return Column permutation vector
                 */
                std::vector<int> colamd_ordering(const CSCMatrix& A);
                
                /**
                 * @brief Compute AMD ordering for symmetric matrices
                 * 
                 * AMD (Approximate Minimum Degree) is optimized for symmetric matrices
                 * and provides excellent fill-in reduction for symmetric problems.
                 * 
                 * Algorithm Features:
                 * - Exploits symmetry for better performance
                 * - Provides near-optimal ordering for many problems
                 * - Time complexity: O(n log n) typically
                 * - Memory efficient for symmetric patterns
                 * 
                 * Use Cases:
                 * - Symmetric positive definite matrices
                 * - Laplacian matrices from finite element methods
                 * - Graph adjacency matrices
                 * 
                 * @param A Sparse matrix in CSC format (should be symmetric)
                 * @return Column permutation vector
                 */
                std::vector<int> amd_ordering(const CSCMatrix& A);
                
                // Supernode operations
                /**
                 * @brief Merge compatible supernodes for better performance
                 * 
                 * Combines supernodes that can be processed together to improve
                 * cache performance and enable larger dense operations.
                 * 
                 * Merging Criteria:
                 * - Supernodes must have compatible sparsity patterns
                 * - Combined size should not exceed cache limits
                 * - Memory overhead should be reasonable
                 * 
                 * Benefits:
                 * - Larger dense blocks for BLAS operations
                 * - Better cache utilization
                 * - Reduced overhead from supernode management
                 * 
                 * Algorithm:
                 * 1. Identify pairs of adjacent supernodes
                 * 2. Check compatibility using can_merge_columns()
                 * 3. Merge compatible pairs
                 * 4. Update elimination tree and dependencies
                 */
                void merge_supernodes();
                
                /**
                 * @brief Check if two columns can be merged into a supernode
                 * 
                 * Determines whether columns col1 and col2 can be grouped together
                 * based on their sparsity patterns and numerical properties.
                 * 
                 * Compatibility Conditions:
                 * - Pattern of col2 must be subset of col1's pattern
                 * - Numerical properties should be similar
                 * - Combined size should fit in cache
                 * 
                 * Mathematical Check:
                 * - For each row i: if A[i,col2] ≠ 0, then A[i,col1] ≠ 0
                 * - This ensures dense operations can be performed
                 * 
                 * @param col1 First column index
                 * @param col2 Second column index
                 * @param A Original sparse matrix
                 * @return true if columns can be merged
                 */
                bool can_merge_columns(int col1, int col2, const CSCMatrix& A);
                
                /**
                 * @brief Compute detailed structure of supernodes
                 * 
                 * Analyzes the internal structure of each supernode to optimize
                 * memory layout and computation patterns.
                 * 
                 * Structure Analysis:
                 * - Row structure within each supernode
                 * - Memory layout for dense operations
                 * - Update matrix generation patterns
                 * 
                 * Memory Optimization:
                 * - Align supernode data for cache efficiency
                 * - Minimize memory fragmentation
                 * - Optimize for BLAS operations
                 * 
                 * @param A Sparse matrix in CSC format
                 */
                void compute_supernode_structure(const CSCMatrix& A);
                
                // Factorization phase
                /**
                 * @brief Initialize factorization data structures
                 * 
                 * Sets up all necessary arrays and matrices for the factorization
                 * process, including workspace allocation and initialization.
                 * 
                 * Initialization Steps:
                 * 1. Allocate memory for L and U factors
                 * 2. Set up permutation arrays
                 * 3. Initialize scaling factors
                 * 4. Prepare supernode workspace
                 * 
                 * Memory Management:
                 * - Pre-allocate all necessary arrays
                 * - Use memory pools for efficiency
                 * - Align data for cache performance
                 * 
                 * @param A Sparse matrix in CSC format
                 */
                void initialize_factorization(const CSCMatrix& A);
                
                /**
                 * @brief Apply row and column scaling to matrix
                 * 
                 * Scales the matrix to improve numerical stability and condition
                 * number before factorization.
                 * 
                 * Scaling Methods:
                 * - Row scaling: Divide each row by its maximum element
                 * - Column scaling: Divide each column by its maximum element
                 * - Combined scaling: Apply both row and column scaling
                 * 
                 * Mathematical Effect:
                 * - Improves condition number κ(A)
                 * - Reduces roundoff error during factorization
                 * - May increase fill-in slightly
                 * 
                 * Implementation:
                 * - Store scaling factors for later use
                 * - Apply scaling in-place to avoid memory overhead
                 * - Handle zero rows/columns gracefully
                 * 
                 * @param A Matrix to be scaled (modified in-place)
                 */
                void scale_matrix(CSCMatrix& A);
                
                /**
                 * @brief Factorize all supernodes in elimination order
                 * 
                 * Performs the main factorization loop, processing each supernode
                 * according to the elimination tree order.
                 * 
                 * Factorization Process:
                 * 1. Process supernodes in postorder traversal
                 * 2. Assemble frontal matrix for each supernode
                 * 3. Add contributions from child supernodes
                 * 4. Factorize dense frontal matrix
                 * 5. Extract L and U factors
                 * 6. Generate update matrices for parents
                 * 
                 * Numerical Stability:
                 * - Use partial pivoting with threshold selection
                 * - Apply scaling to improve condition
                 * - Monitor pivot quality and adjust if needed
                 * 
                 * @param A Original sparse matrix
                 * @return 0 if successful, k>0 if singular at position k
                 */
                int factor_all_supernodes(const CSCMatrix& A);
                
                /**
                 * @brief Factorize a single supernode
                 * 
                 * Performs the factorization of one supernode, including assembly
                 * of the frontal matrix and extraction of factors.
                 * 
                 * Algorithm Steps:
                 * 1. Assemble frontal matrix from original matrix
                 * 2. Add contributions from child supernodes
                 * 3. Apply row/column permutations
                 * 4. Factorize dense frontal matrix
                 * 5. Extract L and U factors
                 * 6. Generate update matrices for parent supernodes
                 * 
                 * Numerical Considerations:
                 * - Use threshold-based pivoting for stability
                 * - Monitor condition number of frontal matrix
                 * - Handle near-singular cases gracefully
                 * 
                 * @param snode_id Index of supernode to factorize
                 * @param A Original sparse matrix
                 * @return 0 if successful, k>0 if singular
                 */
                int factor_supernode(int snode_id, const CSCMatrix& A);
                
                // Frontal matrix operations
                /**
                 * @brief Assemble frontal matrix for a supernode
                 * 
                 * Constructs the dense frontal matrix that represents the current
                 * supernode and its interactions with the original matrix.
                 * 
                 * Frontal Matrix Structure:
                 * - Dense block representing supernode variables
                 * - Includes contributions from original matrix
                 * - Prepared for dense factorization
                 * 
                 * Assembly Process:
                 * 1. Extract relevant rows and columns from A
                 * 2. Apply row and column permutations
                 * 3. Set up dense storage layout
                 * 4. Initialize with original matrix entries
                 * 
                 * Memory Layout:
                 * - Column-major storage for BLAS compatibility
                 * - Aligned for optimal cache performance
                 * - Includes workspace for factorization
                 * 
                 * @param snode_id Index of supernode
                 * @param A Original sparse matrix
                 */
                void assemble_frontal_matrix(int snode_id, const CSCMatrix& A);
                
                /**
                 * @brief Add contributions from child supernodes
                 * 
                 * Incorporates the update matrices from child supernodes into
                 * the current frontal matrix.
                 * 
                 * Update Process:
                 * 1. For each child supernode, get update matrix
                 * 2. Map child indices to current supernode indices
                 * 3. Add update contributions to frontal matrix
                 * 4. Accumulate multiple updates efficiently
                 * 
                 * Mathematical Formulation:
                 * Frontal_matrix += Σ(child_updates)
                 * 
                 * Performance Optimization:
                 * - Use dense matrix addition for efficiency
                 * - Minimize index mapping overhead
                 * - Batch multiple updates when possible
                 * 
                 * @param snode_id Index of current supernode
                 */
                void add_child_contributions(int snode_id);
                
                /**
                 * @brief Factorize dense frontal matrix
                 * 
                 * Performs LU factorization of the dense frontal matrix using
                 * optimized dense linear algebra operations.
                 * 
                 * Factorization Method:
                 * - Use LAPACK-style LU factorization
                 * - Apply partial pivoting for stability
                 * - Store L and U factors in-place
                 * 
                 * Numerical Stability:
                 * - Monitor pivot quality during factorization
                 * - Use threshold-based pivot selection
                 * - Handle near-singular cases
                 * 
                 * Performance:
                 * - Use BLAS-3 operations for efficiency
                 * - Optimize for cache performance
                 * - Enable parallel processing for large blocks
                 * 
                 * @param snode Supernode containing frontal matrix
                 * @return 0 if successful, k>0 if singular
                 */
                int factor_dense_frontal(Supernode& snode);
                
                /**
                 * @brief Extract L and U factors from frontal matrix
                 * 
                 * Copies the L and U factors from the factored frontal matrix
                 * into the sparse storage format.
                 * 
                 * Extraction Process:
                 * 1. Copy L factor (lower triangular part)
                 * 2. Copy U factor (upper triangular part)
                 * 3. Apply row and column permutations
                 * 4. Store in CSC format for efficiency
                 * 
                 * Storage Format:
                 * - L stored in CSC format (unit diagonal implied)
                 * - U stored in CSC format
                 * - Permutation arrays stored separately
                 * 
                 * Memory Management:
                 * - Allocate sparse storage for factors
                 * - Compress storage to minimize memory usage
                 * - Maintain sparsity pattern information
                 * 
                 * @param snode Supernode with factored frontal matrix
                 */
                void extract_factors_from_frontal(const Supernode& snode);
                
                /**
                 * @brief Generate update matrix for parent supernodes
                 * 
                 * Creates the update matrix that will be added to parent supernodes
                 * during the factorization process.
                 * 
                 * Update Matrix Generation:
                 * 1. Extract relevant rows from factored frontal matrix
                 * 2. Apply appropriate scaling and permutations
                 * 3. Store in compact format for efficient addition
                 * 4. Prepare for transmission to parent supernodes
                 * 
                 * Mathematical Formulation:
                 * Update = P * L * U * P^T (relevant rows only)
                 * 
                 * Storage Optimization:
                 * - Store only non-zero elements
                 * - Use compressed format for efficiency
                 * - Minimize memory overhead
                 * 
                 * @param snode Supernode that generated the update
                 */
                void generate_update_matrix(const Supernode& snode);
                
                // Solve phase
                /**
                 * @brief Apply row permutation to vector
                 * 
                 * Applies the row permutation P_r to the vector x, where
                 * P_r is the row permutation matrix from factorization.
                 * 
                 * Mathematical Operation:
                 * x_permuted = P_r * x
                 * 
                 * Implementation:
                 * - Use index-based permutation for efficiency
                 * - Avoid matrix-vector multiplication overhead
                 * - Apply permutation in-place when possible
                 * 
                 * @param x Vector to be permuted (modified in-place)
                 */
                void apply_row_permutation(Eigen::VectorXd& x) const;
                
                /**
                 * @brief Apply column permutation to vector
                 * 
                 * Applies the column permutation P_c to the vector x, where
                 * P_c is the column permutation matrix from factorization.
                 * 
                 * Mathematical Operation:
                 * x_permuted = P_c * x
                 * 
                 * Implementation:
                 * - Use index-based permutation for efficiency
                 * - Apply permutation in-place when possible
                 * - Handle both forward and inverse permutations
                 * 
                 * @param x Vector to be permuted (modified in-place)
                 */
                void apply_col_permutation(Eigen::VectorXd& x) const;
                
                /**
                 * @brief Perform forward substitution L * y = b
                 * 
                 * Solves the lower triangular system L * y = b using the
                 * L factor from factorization.
                 * 
                 * Mathematical Formulation:
                 * L * y = b, where L is unit lower triangular
                 * 
                 * Algorithm:
                 * 1. Process supernodes in elimination order
                 * 2. For each supernode, solve dense triangular system
                 * 3. Update right-hand side for remaining variables
                 * 4. Use level scheduling for parallel efficiency
                 * 
                 * Performance:
                 * - Use dense operations within supernodes
                 * - Enable parallel processing of independent supernodes
                 * - Optimize for cache performance
                 * 
                 * @param x Right-hand side b (input), solution y (output)
                 */
                void forward_substitution(Eigen::VectorXd& x) const;
                
                /**
                 * @brief Perform backward substitution U * z = y
                 * 
                 * Solves the upper triangular system U * z = y using the
                 * U factor from factorization.
                 * 
                 * Mathematical Formulation:
                 * U * z = y, where U is upper triangular
                 * 
                 * Algorithm:
                 * 1. Process supernodes in reverse elimination order
                 * 2. For each supernode, solve dense triangular system
                 * 3. Use level scheduling for parallel efficiency
                 * 4. Apply updates from child supernodes
                 * 
                 * Performance:
                 * - Use dense operations within supernodes
                 * - Enable parallel processing of independent supernodes
                 * - Optimize for cache performance
                 * 
                 * @param x Right-hand side y (input), solution z (output)
                 */
                void backward_substitution(Eigen::VectorXd& x) const;
                
                /**
                 * @brief Apply scaling factors to vector
                 * 
                 * Applies or removes the scaling factors used during factorization
                 * to maintain numerical consistency.
                 * 
                 * Scaling Operations:
                 * - Forward: Apply scaling factors (for solve phase)
                 * - Reverse: Remove scaling factors (for output phase)
                 * 
                 * Mathematical Formulation:
                 * - Forward: x_scaled[i] = x[i] / row_scale[i]
                 * - Reverse: x_unscaled[i] = x[i] * row_scale[i]
                 * 
                 * Implementation:
                 * - Use vectorized operations for efficiency
                 * - Handle zero scaling factors gracefully
                 * - Apply scaling in-place
                 * 
                 * @param x Vector to be scaled (modified in-place)
                 * @param forward true for forward scaling, false for reverse
                 */
                void apply_scaling(Eigen::VectorXd& x, bool forward) const;
                
                // Level scheduling and supernode-based solves
                /**
                 * @brief Compute level schedule for parallel processing
                 * 
                 * Determines the order in which supernodes can be processed
                 * in parallel during forward/backward substitution.
                 * 
                 * Level Scheduling:
                 * - Level 0: Supernodes with no dependencies
                 * - Level k: Supernodes that depend only on levels < k
                 * - Independent supernodes within each level can be processed in parallel
                 * 
                 * Algorithm:
                 * 1. Build dependency graph from elimination tree
                 * 2. Compute levels using topological sorting
                 * 3. Group supernodes by level
                 * 4. Return level-by-level schedule
                 * 
                 * Parallel Efficiency:
                 * - Number of levels determines parallel depth
                 * - Width of each level determines parallel width
                 * - Optimal for matrices with wide elimination trees
                 * 
                 * @param forward true for forward substitution, false for backward
                 * @return Vector of level schedules
                 */
                std::vector<std::vector<int>> compute_level_schedule(bool forward) const;
                
                /**
                 * @brief Solve forward substitution for a single supernode
                 * 
                 * Performs forward substitution L * y = b for variables in the
                 * specified supernode.
                 * 
                 * Algorithm:
                 * 1. Extract relevant variables from right-hand side
                 * 2. Solve dense triangular system for supernode
                 * 3. Update remaining variables in right-hand side
                 * 4. Apply updates to dependent supernodes
                 * 
                 * Performance:
                 * - Use dense BLAS operations for efficiency
                 * - Minimize memory access overhead
                 * - Enable vectorization within supernode
                 * 
                 * @param snode_id Index of supernode to process
                 * @param x Right-hand side and solution vector
                 */
                void solve_supernode_forward(int snode_id, Eigen::VectorXd& x) const;
                
                /**
                 * @brief Solve backward substitution for a single supernode
                 * 
                 * Performs backward substitution U * z = y for variables in the
                 * specified supernode.
                 * 
                 * Algorithm:
                 * 1. Extract relevant variables from right-hand side
                 * 2. Solve dense triangular system for supernode
                 * 3. Apply updates to dependent supernodes
                 * 4. Update solution vector
                 * 
                 * Performance:
                 * - Use dense BLAS operations for efficiency
                 * - Minimize memory access overhead
                 * - Enable vectorization within supernode
                 * 
                 * @param snode_id Index of supernode to process
                 * @param x Right-hand side and solution vector
                 */
                void solve_supernode_backward(int snode_id, Eigen::VectorXd& x) const;
                
                /**
                 * @brief Solve forward substitution for multiple right-hand sides
                 * 
                 * Efficiently solves L * Y = B for multiple right-hand sides
                 * using the same supernode structure.
                 * 
                 * Performance Benefits:
                 * - Better cache utilization for multiple vectors
                 * - Reduced overhead from supernode processing
                 * - Enables BLAS-3 operations for dense blocks
                 * 
                 * Algorithm:
                 * 1. Process each supernode in elimination order
                 * 2. Solve dense triangular systems for all right-hand sides
                 * 3. Update remaining variables efficiently
                 * 4. Apply updates to dependent supernodes
                 * 
                 * @param snode_id Index of supernode to process
                 * @param X Right-hand side and solution matrix
                 */
                void solve_supernode_forward_multiple(int snode_id, Eigen::MatrixXd& X) const;
                
                /**
                 * @brief Solve backward substitution for multiple right-hand sides
                 * 
                 * Efficiently solves U * Z = Y for multiple right-hand sides
                 * using the same supernode structure.
                 * 
                 * Performance Benefits:
                 * - Better cache utilization for multiple vectors
                 * - Reduced overhead from supernode processing
                 * - Enables BLAS-3 operations for dense blocks
                 * 
                 * Algorithm:
                 * 1. Process each supernode in reverse elimination order
                 * 2. Solve dense triangular systems for all right-hand sides
                 * 3. Apply updates efficiently to all vectors
                 * 4. Update solution matrix
                 * 
                 * @param snode_id Index of supernode to process
                 * @param X Right-hand side and solution matrix
                 */
                void solve_supernode_backward_multiple(int snode_id, Eigen::MatrixXd& X) const;
                
                /**
                 * @brief Finalize CSC factors for efficient storage
                 * 
                 * Converts the factors from internal format to final CSC format
                 * for efficient storage and access.
                 * 
                 * Finalization Steps:
                 * 1. Compress sparse storage to minimize memory
                 * 2. Optimize data layout for cache performance
                 * 3. Validate sparsity patterns
                 * 4. Prepare for efficient solve operations
                 * 
                 * Memory Optimization:
                 * - Remove any temporary storage
                 * - Align data for optimal access patterns
                 * - Compress index arrays where possible
                 */
                void finalize_csc_factors();
                
                // Utilities
                /**
                 * @brief Validate matrix properties and dimensions
                 * 
                 * Performs comprehensive validation of the input matrix to ensure
                 * it meets the requirements for sparse factorization.
                 * 
                 * Validation Checks:
                 * - Matrix dimensions are positive
                 * - Non-zero count is reasonable
                 * - Matrix is not empty
                 * - Data arrays are consistent
                 * 
                 * Error Handling:
                 * - Throw exceptions for invalid matrices
                 * - Provide detailed error messages
                 * - Check for common issues (empty matrices, etc.)
                 * 
                 * @param A Matrix to validate
                 * @throws std::invalid_argument if matrix is invalid
                 */
                void validate_matrix(const CSCMatrix& A) const;
                
                /**
                 * @brief Clear all factorization data
                 * 
                 * Releases all memory associated with the current factorization
                 * and resets the solver to its initial state.
                 * 
                 * Cleanup Operations:
                 * - Deallocate L and U factors
                 * - Clear supernode structures
                 * - Reset elimination tree
                 * - Free workspace arrays
                 * 
                 * State Reset:
                 * - Set analyzed_ = false
                 * - Set factorized_ = false
                 * - Reset performance counters
                 * 
                 * Memory Management:
                 * - Return all memory to system
                 * - Clear any cached data structures
                 * - Reset memory pools if used
                 */
                void clear_factorization();
                
                /**
                 * @brief Allocate workspace for factorization
                 * 
                 * Sets up temporary storage arrays needed during the factorization
                 * process.
                 * 
                 * Workspace Components:
                 * - Frontal matrix storage
                 * - Update matrix buffers
                 * - Temporary vectors for solve phase
                 * - Index mapping arrays
                 * 
                 * Allocation Strategy:
                 * - Pre-allocate all necessary arrays
                 * - Use memory pools for efficiency
                 * - Align data for cache performance
                 * - Handle allocation failures gracefully
                 */
                void allocate_workspace();
                
                // True sparse LU implementation for large matrices
                /**
                 * @brief Solve large sparse systems using true sparse LU
                 * 
                 * For very large matrices, uses a simplified sparse LU approach
                 * that avoids the complexity of supernodes but maintains efficiency.
                 * 
                 * Use Cases:
                 * - Matrices too large for supernode approach
                 * - Memory-constrained environments
                 * - Matrices with poor supernode structure
                 * 
                 * Algorithm:
                 * 1. Simple sparse LU factorization
                 * 2. Basic fill-in minimization
                 * 3. Standard forward/backward substitution
                 * 
                 * Performance Characteristics:
                 * - Lower memory overhead
                 * - Simpler implementation
                 * - May be slower for well-structured matrices
                 * 
                 * @param A Sparse coefficient matrix
                 * @param x Solution vector
                 * @param b Right-hand side vector
                 * @return true if successful, false otherwise
                 */
                bool solve_sparse_lu(const Eigen::SparseMatrix<double>& A, 
                                   Eigen::VectorXd& x, const Eigen::VectorXd& b);
                
                // Simplified sparse factorization without complex supernodes
                /**
                 * @brief Simplified sparse factorization for large matrices
                 * 
                 * Performs basic sparse LU factorization without the complexity
                 * of supernodes, suitable for very large or memory-constrained problems.
                 * 
                 * Algorithm:
                 * 1. Basic column ordering (AMD)
                 * 2. Simple sparse LU with partial pivoting
                 * 3. Basic fill-in minimization
                 * 4. Standard triangular solve
                 * 
                 * Trade-offs:
                 * - Simpler implementation and debugging
                 * - Lower memory overhead
                 * - May be slower for well-structured matrices
                 * - Less cache-friendly
                 * 
                 * @param A Sparse matrix in CSC format
                 * @return 0 if successful, k>0 if singular at position k
                 */
                int factorize_sparse_direct(const CSCMatrix& A);
                
                /**
                 * @brief Extract L and U factors from simplified factorization
                 * 
                 * Converts the factors from the simplified factorization approach
                 * into the standard sparse storage format.
                 * 
                 * Extraction Process:
                 * 1. Copy L factor to sparse storage
                 * 2. Copy U factor to sparse storage
                 * 3. Apply permutations
                 * 4. Compress storage format
                 * 
                 * Storage Format:
                 * - L stored in CSC format (unit diagonal implied)
                 * - U stored in CSC format
                 * - Permutation arrays stored separately
                 * 
                 * @param A_factored Matrix after simplified factorization
                 */
                void extract_lu_factors_direct(const CSCMatrix& A_factored);
    };
}

#endif // POLIVEM_SPARSE_GAUSS_ELIMINATION_HPP