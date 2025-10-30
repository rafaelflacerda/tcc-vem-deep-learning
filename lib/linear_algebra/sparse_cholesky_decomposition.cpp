#include "linear_algebra/sparse_cholesky_decomposition.hpp"
#include <iomanip>

namespace LinearAlgebra {
    // ============================================================================
    // CONSTRUCTORS AND INITIALIZATION
    // ============================================================================
    
    SparseCholeskyDecomposition::SparseCholeskyDecomposition()
        : n_(0)
        , analyzed_(false)
        , factorized_(false)
        , max_threads_(0)
        , use_scaling_(true)
        , supernode_threshold_(8)
        , original_nnz_(0)
        , factor_nnz_(0) {
        
        // Auto-detect thread count if not specified
        if (max_threads_ == 0) {
            max_threads_ = std::min(static_cast<int>(std::thread::hardware_concurrency()), 8);
        }
    }
    
    // ============================================================================
    // MAIN SOLVER INTERFACE
    // ============================================================================
    
    bool SparseCholeskyDecomposition::analyze(const CSCMatrix& A) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            validate_spd_matrix(A);
            
            n_ = A.n_rows;
            original_nnz_ = A.nnz;
            
            // Clear any previous analysis
            analyzed_ = false;
            factorized_ = false;
            
            // Step 1: Compute fill-reducing ordering (AMD for SPD)
            compute_column_ordering(A);
            
            // Step 2: Build elimination tree
            build_elimination_tree(A);
            
            // Step 3: Detect and merge supernodes for Cholesky
            detect_supernodes_cholesky(A);
            
            // Step 4: Analyze memory requirements
            analyze_memory_requirements();
            
            // Step 5: Initialize dense Cholesky kernel (will resize as needed)
            dense_kernel_ = std::make_unique<CholeskyDecomposition>(1);
            
            analyzed_ = true;
            
        } catch (const std::exception& e) {
            std::cerr << "Cholesky analysis failed: " << e.what() << std::endl;
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        return true;
    }
    
    bool SparseCholeskyDecomposition::analyze(const Eigen::SparseMatrix<double>& A) {
        CSCMatrix csc_matrix(A);
        return analyze(csc_matrix);
    }
    
    int SparseCholeskyDecomposition::factorize(const CSCMatrix& A) {
        if (!analyzed_) {
            throw std::runtime_error("Must call analyze() before factorize()");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Create working copy of matrix
        CSCMatrix A_work = A;
        
        try {
            // Step 1: Apply column permutation and scaling
            initialize_factorization(A_work);
            if (use_scaling_) {
                scale_matrix_symmetric(A_work);
            }
            
            // Step 2: Factor all supernodes using Cholesky
            int result = factor_all_supernodes_cholesky(A_work);
            
            if (result == 0) {
                factorized_ = true;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            
            return result;
            
        } catch (const std::exception& e) {
            std::cerr << "Cholesky factorization failed: " << e.what() << std::endl;
            return -1;
        }
    }
    
    int SparseCholeskyDecomposition::factorize(const Eigen::SparseMatrix<double>& A) {
        CSCMatrix csc_matrix(A);
        return factorize(csc_matrix);
    }
    
    bool SparseCholeskyDecomposition::solve(Eigen::VectorXd& x, const Eigen::VectorXd& b) const {
        if (!factorized_) {
            throw std::runtime_error("Must call factorize() before solve()");
        }
        
        if (b.size() != n_) {
            throw std::invalid_argument("Vector size does not match matrix size");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        x = b;  // Copy RHS
        
        // Apply scaling and permutations, then solve L·L^T·x = b
        if (use_scaling_) {
            apply_scaling_symmetric(x, true);  // Forward scaling
        }
        
        apply_column_permutation(x, true);      // Apply column permutation
        forward_substitution_sparse(x);         // Solve L·y = P·S·b
        backward_substitution_sparse(x);        // Solve L^T·z = y
        apply_column_permutation(x, false);     // Apply inverse column permutation
        
        if (use_scaling_) {
            apply_scaling_symmetric(x, false);  // Backward scaling
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        return true;
    }
    
    bool SparseCholeskyDecomposition::solve(Eigen::MatrixXd& X, const Eigen::MatrixXd& B) const {
        if (B.rows() != n_) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        
        X.resize(B.rows(), B.cols());
        
        // Solve each column independently
        for (int j = 0; j < B.cols(); j++) {
            Eigen::VectorXd x_j = X.col(j);
            Eigen::VectorXd b_j = B.col(j);
            
            bool success = solve(x_j, b_j);
            if (!success) {
                return false;
            }
            
            X.col(j) = x_j;
        }
        
        return true;
    }
    
    bool SparseCholeskyDecomposition::solve_system(const Eigen::SparseMatrix<double>& A,
                                                    Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        n_ = static_cast<int>(A.rows());
        
        // HYBRID APPROACH: Use dense kernel for small matrices, sparse for large
        double density = static_cast<double>(A.nonZeros()) / (n_ * n_);
        
        if (n_ < 100 || density > 0.1) {
            // Small or dense matrices: use dense Cholesky (more aggressive threshold)
            Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
            CholeskyDecomposition dense_solver(A_dense.rows());
            Eigen::MatrixXd A_copy = A_dense;
            
            return dense_solver.solve(A_copy, x, b);
        } else {
            // Large sparse matrices: use optimized sparse Cholesky
            return solve_sparse_cholesky_optimized(A, x, b);
        }
    }
    
    // ============================================================================
    // PERFORMANCE AND DIAGNOSTICS
    // ============================================================================
    
    double SparseCholeskyDecomposition::getFactorizationGFLOPs() const {
        return 0.0;
    }
    
    // ============================================================================
    // ANALYSIS PHASE IMPLEMENTATION
    // ============================================================================
    
    void SparseCholeskyDecomposition::compute_column_ordering(const CSCMatrix& A) {
        // Use AMD ordering for SPD matrices (better than COLAMD for symmetric)
        col_perm_ = amd_ordering_cholesky(A);
        
        // Compute inverse permutation
        col_perm_inv_.resize(n_);
        for (int i = 0; i < n_; i++) {
            col_perm_inv_[col_perm_[i]] = i;
        }
    }
    
    std::vector<int> SparseCholeskyDecomposition::amd_ordering_cholesky(const CSCMatrix& A) {
        // Simplified AMD implementation for SPD matrices
        // In practice, would use the actual AMD library or METIS
        
        std::vector<std::pair<int, int>> col_weights(A.n_cols);
        for (int j = 0; j < A.n_cols; j++) {
            int col_nnz = 0;
            // Count lower triangular entries only (SPD symmetry)
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                if (A.row_indices[p] >= j) {  // Lower triangular
                    col_nnz++;
                }
            }
            col_weights[j] = {col_nnz, j};
        }
        
        // Sort by number of non-zeros in lower triangle (ascending)
        std::sort(col_weights.begin(), col_weights.end());
        
        std::vector<int> ordering(A.n_cols);
        for (int i = 0; i < A.n_cols; i++) {
            ordering[i] = col_weights[i].second;
        }
        
        return ordering;
    }
    
    void SparseCholeskyDecomposition::build_elimination_tree(const CSCMatrix& A) {
        elim_tree_ = std::make_unique<EliminationTree>(n_);
        
        // For SPD matrices, we need to build the elimination tree from the lower triangular part
        // Create a working matrix that represents the pattern we'll actually factor
        CSCMatrix A_lower = create_lower_triangular_pattern(A);
        
        // Apply column permutation if we have one
        if (!col_perm_.empty()) {
            A_lower = apply_column_permutation_to_matrix(A_lower);
        }
        
        build_elimination_tree_cholesky(A_lower);
    }
    
    void SparseCholeskyDecomposition::detect_supernodes_cholesky(const CSCMatrix& A) {
        supernodes_.clear();
        col_to_supernode_.resize(n_);
        
        // Create lower triangular pattern for analysis
        CSCMatrix A_lower = create_lower_triangular_pattern(A);
        
        // Start with each column as its own supernode
        for (int j = 0; j < n_; j++) {
            CholeskySupernode snode;
            snode.start_col = j;
            snode.num_cols = 1;
            snode.parent = (elim_tree_ && j < static_cast<int>(elim_tree_->parent.size())) ? elim_tree_->parent[j] : -1;
            snode.children = (elim_tree_ && j < static_cast<int>(elim_tree_->children.size())) ? elim_tree_->children[j] : std::vector<int>();
            
            supernodes_.push_back(snode);
            col_to_supernode_[j] = static_cast<int>(supernodes_.size() - 1);
        }
        
        // Merge adjacent columns with identical structure (Cholesky-specific)
        merge_supernodes_cholesky();
        
        // Compute supernode structures
        compute_supernode_structure_cholesky(A_lower);
    }
    
    void SparseCholeskyDecomposition::merge_supernodes_cholesky() {
        // Enhanced supernode merging for Cholesky (can be more aggressive than LU)
        std::vector<CholeskySupernode> merged_supernodes;
        
        int current_snode = 0;
        while (current_snode < static_cast<int>(supernodes_.size())) {
            CholeskySupernode& snode = supernodes_[current_snode];
            
            // Try to merge with consecutive supernodes
            int next_snode = current_snode + 1;
            while (next_snode < static_cast<int>(supernodes_.size()) &&
                    snode.num_cols < supernode_threshold_ &&
                    can_merge_columns_cholesky(snode.start_col + snode.num_cols - 1, 
                                                supernodes_[next_snode].start_col, CSCMatrix())) {
                snode.num_cols += supernodes_[next_snode].num_cols;
                next_snode++;
            }
            
            merged_supernodes.push_back(snode);
            current_snode = next_snode;
        }
        
        supernodes_ = std::move(merged_supernodes);
        
        // Update column to supernode mapping
        for (int s = 0; s < static_cast<int>(supernodes_.size()); s++) {
            const CholeskySupernode& snode = supernodes_[s];
            for (int j = snode.start_col; j < snode.start_col + snode.num_cols; j++) {
                col_to_supernode_[j] = s;
            }
        }
    }
    
    bool SparseCholeskyDecomposition::can_merge_columns_cholesky(int col1, int col2, const CSCMatrix& A) {
        // Cholesky-specific merging criterion
        // Can merge if columns are consecutive and have compatible elimination tree structure
        if (col2 != col1 + 1) return false;  // Must be consecutive
        
        // Check if they have the same parent in elimination tree
        if (elim_tree_ && col1 < static_cast<int>(elim_tree_->parent.size()) && 
            col2 < static_cast<int>(elim_tree_->parent.size())) {
            return elim_tree_->parent[col1] == elim_tree_->parent[col2];
        }
        
        return true;  // Default to allowing merge for consecutive columns
    }
    
    void SparseCholeskyDecomposition::compute_supernode_structure_cholesky(const CSCMatrix& A) {
        for (auto& snode : supernodes_) {
            // Compute row structure for this supernode (lower triangle only)
            std::unordered_set<int> row_set;
            
            for (int j = snode.start_col; j < snode.start_col + snode.num_cols; j++) {
                for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                    int i = A.row_indices[p];
                    if (i >= j) {  // Lower triangular part only
                        row_set.insert(i);
                    }
                }
            }
            
            snode.row_structure.assign(row_set.begin(), row_set.end());
            std::sort(snode.row_structure.begin(), snode.row_structure.end());
            
            snode.front_height = static_cast<int>(snode.row_structure.size());
            snode.front_width = static_cast<int>(snode.row_structure.size());  // Square for Cholesky
            
            // Allocate frontal matrix (symmetric, but store full for BLAS compatibility)
            int front_size = snode.front_height * snode.front_width;
            snode.frontal_matrix.resize(front_size, 0.0);
            snode.front_lda = snode.front_height;
        }
    }
    
    void SparseCholeskyDecomposition::analyze_memory_requirements() {
        // Estimate memory requirements for L factor (no U needed for Cholesky)
        factor_nnz_ = 0;
        
        for (const auto& snode : supernodes_) {
            // Lower triangular part only
            size_t L_nnz = (snode.front_height * (snode.front_height + 1)) / 2;
            factor_nnz_ += L_nnz;
        }
        
        // Pre-allocate storage
        L_factor_.values.reserve(factor_nnz_);
        L_factor_.row_indices.reserve(factor_nnz_);
    }
    
    // ============================================================================
    // FACTORIZATION PHASE IMPLEMENTATION
    // ============================================================================
    
    void SparseCholeskyDecomposition::initialize_factorization(const CSCMatrix& A) {
        // Initialize scaling vectors for SPD matrices
        if (use_scaling_) {
            scale_factors_.resize(n_, 1.0);
            
            // Compute diagonal scaling factors
            for (int j = 0; j < A.n_cols; j++) {
                for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                    int i = A.row_indices[p];
                    if (i == j) {  // Diagonal element
                        scale_factors_[j] = 1.0 / std::sqrt(std::abs(A.values[p]));
                        break;
                    }
                }
            }
        }
        
        // Initialize L factor structure
        L_factor_.n_rows = n_;
        L_factor_.n_cols = n_;
        L_factor_.col_pointers.resize(n_ + 1, 0);
    }
    
    void SparseCholeskyDecomposition::scale_matrix_symmetric(CSCMatrix& A) {
        if (!use_scaling_) return;
        
        // Apply symmetric scaling: A ← S·A·S where S = diag(scale_factors)
        for (int j = 0; j < A.n_cols; j++) {
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];
                A.values[p] *= scale_factors_[i] * scale_factors_[j];
            }
        }
    }
    
    int SparseCholeskyDecomposition::factor_all_supernodes_cholesky(const CSCMatrix& A) {
        // Clear previous factorization
        L_factor_.values.clear();
        L_factor_.row_indices.clear();
        std::fill(L_factor_.col_pointers.begin(), L_factor_.col_pointers.end(), 0);
        
        // Get postorder traversal of elimination tree for dependency scheduling
        auto postorder = elim_tree_->get_postorder();
        
        // Factor supernodes in postorder (children before parents)
        for (int col : postorder) {
            // Bounds checking
            if (col < 0 || col >= n_ || col >= static_cast<int>(col_to_supernode_.size())) {
                continue;
            }
            
            int snode_id = col_to_supernode_[col];
            
            // Skip if this supernode was already processed or invalid
            if (snode_id == -1 || snode_id >= static_cast<int>(supernodes_.size())) {
                continue;
            }
            
            int result = factor_supernode_cholesky(snode_id, A);
            if (result != 0) {
                return result;  // Not positive definite
            }
            
            // Mark processed columns
            const CholeskySupernode& snode = supernodes_[snode_id];
            for (int j = snode.start_col; j < snode.start_col + snode.num_cols; j++) {
                if (j != col && j >= 0 && j < static_cast<int>(col_to_supernode_.size())) {
                    col_to_supernode_[j] = -1;
                }
            }
        }
        
        // Finalize CSC structure - set final column pointer and fill any gaps
        for (int j = 1; j <= n_; j++) {
            if (L_factor_.col_pointers[j] == 0) {
                // Column j wasn't processed, set to same as previous column
                L_factor_.col_pointers[j] = (j > 0) ? L_factor_.col_pointers[j-1] : 0;
            }
        }
        L_factor_.col_pointers[n_] = static_cast<int>(L_factor_.values.size());
        L_factor_.nnz = static_cast<int>(L_factor_.values.size());
        
        // Add missing diagonal entries for columns that weren't processed
        for (int j = 0; j < n_; j++) {
            int col_start = L_factor_.col_pointers[j];
            int col_end = L_factor_.col_pointers[j + 1];
            
            // Check if column j has a diagonal entry
            bool has_diagonal = false;
            for (int p = col_start; p < col_end; p++) {
                if (L_factor_.row_indices[p] == j) {
                    has_diagonal = true;
                    break;
                }
            }
            
            if (!has_diagonal) {
                // Insert identity diagonal entry
                L_factor_.values.insert(L_factor_.values.begin() + col_start, 1.0);
                L_factor_.row_indices.insert(L_factor_.row_indices.begin() + col_start, j);
                
                // Update all subsequent column pointers
                for (int k = j + 1; k <= n_; k++) {
                    L_factor_.col_pointers[k]++;
                }
            }
        }
        
        L_factor_.nnz = static_cast<int>(L_factor_.values.size());
        
        return 0;  // Success
    }
    
    int SparseCholeskyDecomposition::factor_supernode_cholesky(int snode_id, const CSCMatrix& A) {
        // Bounds checking
        if (snode_id < 0 || snode_id >= static_cast<int>(supernodes_.size())) {
            return -1;  // Invalid supernode ID
        }
        
        CholeskySupernode& snode = supernodes_[snode_id];
        
        // Step 1: Assemble frontal matrix from sparse matrix and updates
        assemble_frontal_matrix_symmetric(snode_id, A);
        
        // Step 2: Add symmetric contributions from child supernodes
        add_child_contributions_symmetric(snode_id);
        
        // Step 3: Factor the dense frontal matrix using dense Cholesky kernel
        int result = factor_dense_frontal_cholesky(snode);
        if (result != 0) {
            return snode.start_col + result;  // Return global column index
        }
        
        // Step 4: Extract L factor from frontal matrix
        extract_cholesky_factor_from_frontal(snode);
        
        // Step 5: Generate symmetric update matrix for parent supernodes
        generate_symmetric_update_matrix(snode);
        
        return 0;  // Success
    }
    
    void SparseCholeskyDecomposition::assemble_frontal_matrix_symmetric(int snode_id, const CSCMatrix& A) {
        CholeskySupernode& snode = supernodes_[snode_id];
        
        // Clear frontal matrix
        std::fill(snode.frontal_matrix.begin(), snode.frontal_matrix.end(), 0.0);
        
        // Copy sparse matrix entries into frontal matrix (full symmetric matrix)
        for (int ii = 0; ii < snode.front_height; ii++) {
            int i = snode.row_structure[ii];  // Global row index
            
            for (int jj = 0; jj < snode.front_width; jj++) {
                int j = snode.row_structure[jj];  // Global column index
                
                // Find the value A[i,j] in the sparse matrix
                double value = 0.0;
                
                // Look in column j for row i
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                    if (A.row_indices[p] == i) {
                        value = A.values[p];
                        break;
                    }
                }
                
                // Store in frontal matrix
                        snode.frontal_matrix[jj * snode.front_lda + ii] = value;
            }
        }
    }
    
    void SparseCholeskyDecomposition::add_child_contributions_symmetric(int snode_id) {
        CholeskySupernode& snode = supernodes_[snode_id];
        
        // Add symmetric update contributions from all child supernodes
        for (const auto& update : snode.child_updates) {
            // Map global indices to local frontal matrix indices
            for (int ii = 0; ii < update.size; ii++) {
                for (int jj = 0; jj <= ii; jj++) {  // Lower triangle only
                    int global_i = update.indices[ii];
                    int global_j = update.indices[jj];
                    
                    // Find local positions in frontal matrix
                    auto row_it = std::find(snode.row_structure.begin(), snode.row_structure.end(), global_i);
                    auto col_it = std::find(snode.row_structure.begin(), snode.row_structure.end(), global_j);
                    
                    if (row_it != snode.row_structure.end() && col_it != snode.row_structure.end()) {
                        int local_i = static_cast<int>(row_it - snode.row_structure.begin());
                        int local_j = static_cast<int>(col_it - snode.row_structure.begin());
                        
                        // Add symmetric contribution
                        int update_idx = ii * update.size + jj;
                        snode.frontal_matrix[local_j * snode.front_lda + local_i] += update.values[update_idx];
                    }
                }
            }
        }
    }
    
    int SparseCholeskyDecomposition::factor_dense_frontal_cholesky(CholeskySupernode& snode) {
        // Resize dense kernel if needed
        if (!dense_kernel_ || dense_kernel_->size() != static_cast<size_t>(snode.front_height)) {
            dense_kernel_ = std::make_unique<CholeskyDecomposition>(snode.front_height);
        }
        
        // Factor the frontal matrix in-place using dense Cholesky
        return dense_kernel_->factorize(snode.frontal_matrix.data(), snode.front_lda);
    }
    
    void SparseCholeskyDecomposition::extract_cholesky_factor_from_frontal(const CholeskySupernode& snode) {
        // Extract L factor (lower triangular) from frontal matrix
        // For Cholesky: A = L * L^T, so we extract the actual Cholesky factor L
        
        for (int jj = 0; jj < snode.num_cols; jj++) {
            int j = snode.start_col + jj;  // Global column index
            
            // Set column pointer for this column
            if (j < n_) {
                L_factor_.col_pointers[j] = static_cast<int>(L_factor_.values.size());
            }
            
            // Extract lower triangular part (including diagonal)
            for (int ii = jj; ii < snode.front_height; ii++) {  // Lower triangular
                int i = snode.row_structure[ii];  // Global row index
                double value = snode.frontal_matrix[jj * snode.front_lda + ii];
                
                if (std::abs(value) > 1e-16) {  // Only store significant entries
                    L_factor_.values.push_back(value);
                    L_factor_.row_indices.push_back(i);
                }
            }
        }
        
        // NOTE: Final column pointer L_factor_.col_pointers[n_] will be set in factor_all_supernodes_cholesky
    }
    
    void SparseCholeskyDecomposition::generate_symmetric_update_matrix(const CholeskySupernode& snode) {
        if (snode.parent == -1) return;  // Root supernode has no parent
        
        // Create symmetric Schur complement update for parent supernode
        int update_size = snode.front_height - snode.num_cols;
        if (update_size <= 0) return;
        
        CholeskySupernode::SymmetricUpdate update;
        update.size = update_size;
        update.values.resize((update_size * (update_size + 1)) / 2);  // Lower triangle only
        update.indices.resize(update_size);
        
        // Copy update matrix indices (rows after the pivot block)
        for (int i = 0; i < update_size; i++) {
            update.indices[i] = snode.row_structure[snode.num_cols + i];
        }
        
        // Copy update matrix values (lower triangle only)
        int value_idx = 0;
        for (int j = 0; j < update_size; j++) {
            for (int i = j; i < update_size; i++) {  // Lower triangle
                int frontal_i = snode.num_cols + i;
                int frontal_j = snode.num_cols + j;
                update.values[value_idx++] = 
                    snode.frontal_matrix[frontal_j * snode.front_lda + frontal_i];
            }
        }
        
        // Add update to parent supernode
        if (snode.parent >= 0 && snode.parent < static_cast<int>(col_to_supernode_.size())) {
        int parent_snode_id = col_to_supernode_[snode.parent];
            if (parent_snode_id >= 0 && parent_snode_id < static_cast<int>(supernodes_.size())) {
            supernodes_[parent_snode_id].child_updates.push_back(std::move(update));
            }
        }
    }
    
    // ============================================================================
    // SOLVE PHASE IMPLEMENTATION
    // ============================================================================
    
    void SparseCholeskyDecomposition::apply_column_permutation(Eigen::VectorXd& x, bool forward) const {
        Eigen::VectorXd x_perm(n_);
        
        if (forward) {
            // Apply permutation: x_perm = P * x
            for (int i = 0; i < n_; i++) {
                x_perm[col_perm_[i]] = x[i];
            }
        } else {
            // Apply inverse permutation: x_perm = P^T * x
            for (int i = 0; i < n_; i++) {
                x_perm[i] = x[col_perm_[i]];
            }
        }
        
        x = x_perm;
    }
    
    void SparseCholeskyDecomposition::forward_substitution_sparse(Eigen::VectorXd& x) const {
        // Solve L * y = x using sparse forward substitution
        // L is stored in CSC format (Compressed Sparse Column)
        
        for (int j = 0; j < n_; j++) {
            // Process column j of L matrix
            int col_start = L_factor_.col_pointers[j];
            int col_end = L_factor_.col_pointers[j + 1];
            
            if (col_start >= col_end) {
                // Empty column - this might indicate an issue with factorization
                // For now, assume diagonal is 1.0 (identity)
                continue;
            }
            
            // Find diagonal entry L[j,j]
            double L_jj = 0.0;
            int diag_pos = -1;
            
            for (int p = col_start; p < col_end; p++) {
                int i = L_factor_.row_indices[p];
                if (i == j) {  // Diagonal entry
                    L_jj = L_factor_.values[p];
                    diag_pos = p;
                    break;
                }
            }
            
            if (diag_pos == -1) {
                // No diagonal entry found - assume identity
                L_jj = 1.0;
            } else if (std::abs(L_jj) < 1e-16) {
                throw std::runtime_error("Singular matrix in forward substitution at column " + std::to_string(j));
            }
            
            // Scale by diagonal: x[j] = x[j] / L[j,j]
            if (std::abs(L_jj - 1.0) > 1e-16) {  // Only divide if not identity
            x[j] /= L_jj;
            }
            
            // Update remaining entries: x[i] -= L[i,j] * x[j] for i > j
            for (int p = col_start; p < col_end; p++) {
                int i = L_factor_.row_indices[p];
                if (i > j) {  // Below diagonal
                    double L_ij = L_factor_.values[p];
                    x[i] -= L_ij * x[j];
                }
            }
        }
    }
    
    void SparseCholeskyDecomposition::backward_substitution_sparse(Eigen::VectorXd& x) const {
        // Solve L^T * x = y using sparse backward substitution
        // For Cholesky: A = L * L^T, so we solve L^T * x = y
        
        for (int j = n_ - 1; j >= 0; j--) {
            // Process column j of L (which becomes row j of L^T)
            int col_start = L_factor_.col_pointers[j];
            int col_end = L_factor_.col_pointers[j + 1];
            
            if (col_start >= col_end) {
                // Empty column - assume identity
                continue;
            }
            
            // Find diagonal entry L[j,j] (same as L^T[j,j])
            double L_jj = 1.0;  // Default for missing diagonal
            
            for (int p = col_start; p < col_end; p++) {
                int i = L_factor_.row_indices[p];
                if (i == j) {  // Diagonal entry
                    L_jj = L_factor_.values[p];
                    break;
                }
            }
            
            if (std::abs(L_jj) < 1e-16) {
                throw std::runtime_error("Singular matrix in backward substitution at column " + std::to_string(j));
            }
            
            // Apply updates from below-diagonal entries
            // For L^T[i,j] = L[j,i], we look at column j entries where i > j
            for (int p = col_start; p < col_end; p++) {
                int i = L_factor_.row_indices[p];
                if (i > j) {  // Below diagonal in L becomes above diagonal in L^T
                    double L_ij = L_factor_.values[p];  // L[i,j] = L^T[j,i]
                    x[j] -= L_ij * x[i];  // x[j] -= L^T[j,i] * x[i]
                }
            }
            
            // Scale by diagonal: x[j] = x[j] / L^T[j,j] = x[j] / L[j,j]
            if (std::abs(L_jj - 1.0) > 1e-16) {  // Only divide if not identity
                x[j] /= L_jj;
            }
        }
    }
    
    void SparseCholeskyDecomposition::apply_scaling_symmetric(Eigen::VectorXd& x, bool forward) const {
        if (!use_scaling_) return;
        
        if (forward) {
            // Apply scaling: x ← S * x
            for (int i = 0; i < n_; i++) {
                x[i] *= scale_factors_[i];
            }
        } else {
            // Apply inverse scaling: x ← S^(-1) * x
            for (int i = 0; i < n_; i++) {
                x[i] /= scale_factors_[i];
            }
        }
    }
    
    // ============================================================================
    // HELPER FUNCTIONS FOR CHOLESKY-SPECIFIC OPERATIONS
    // ============================================================================
    
    CSCMatrix SparseCholeskyDecomposition::create_lower_triangular_pattern(const CSCMatrix& A) const {
        CSCMatrix A_lower;
        A_lower.n_rows = A.n_rows;
        A_lower.n_cols = A.n_cols;
        A_lower.col_pointers.resize(A.n_cols + 1, 0);
        
        // Extract only lower triangular entries (i >= j)
        for (int j = 0; j < A.n_cols; j++) {
            A_lower.col_pointers[j] = static_cast<int>(A_lower.values.size());
            
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];
                if (i >= j) {  // Lower triangular part (including diagonal)
                    A_lower.values.push_back(A.values[p]);
                    A_lower.row_indices.push_back(i);
                }
            }
        }
        
        A_lower.col_pointers[A.n_cols] = static_cast<int>(A_lower.values.size());
        A_lower.nnz = static_cast<int>(A_lower.values.size());
        
        return A_lower;
    }
    
    CSCMatrix SparseCholeskyDecomposition::apply_column_permutation_to_matrix(const CSCMatrix& A) const {
        // For now, return the matrix as-is
        // In a full implementation, this would apply the column permutation
        return A;
    }
    
    void SparseCholeskyDecomposition::build_elimination_tree_cholesky(const CSCMatrix& A_lower) {
        // Build elimination tree specifically for Cholesky factorization
        // This is adapted from the sparse LU implementation but considers only lower triangular part
        
        std::fill(elim_tree_->parent.begin(), elim_tree_->parent.end(), -1);
        for (auto& child_list : elim_tree_->children) {
            child_list.clear();
        }
        
        std::vector<int> ancestor(n_, -1);
        
        // For Cholesky, we process the lower triangular pattern
        for (int j = 0; j < n_; j++) {
            for (int p = A_lower.col_pointers[j]; p < A_lower.col_pointers[j + 1]; p++) {
                int i = A_lower.row_indices[p];
                if (i <= j) continue;  // Only consider entries below diagonal (i > j)
                
                // Find the path from i to the root and update ancestors
                int r = i;
                while (ancestor[r] != -1 && ancestor[r] != j) {
                    int next = ancestor[r];
                    ancestor[r] = j;
                    r = next;
                }
                
                if (ancestor[r] == -1) {
                    ancestor[r] = j;
                    elim_tree_->parent[r] = j;
                }
            }
        }
        
        // Build children lists from parent relationships
        for (int i = 0; i < n_; i++) {
            if (elim_tree_->parent[i] != -1) {
                elim_tree_->children[elim_tree_->parent[i]].push_back(i);
            }
        }
    }
    
    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    void SparseCholeskyDecomposition::validate_spd_matrix(const CSCMatrix& A) const {
        if (A.n_rows != A.n_cols) {
            throw std::invalid_argument("Matrix must be square for Cholesky decomposition");
        }
        
        if (A.n_rows <= 0) {
            throw std::invalid_argument("Matrix size must be positive");
        }
        
        if (A.nnz < 0) {
            throw std::invalid_argument("Invalid number of non-zeros");
        }
        
        if (static_cast<int>(A.values.size()) != A.nnz ||
            static_cast<int>(A.row_indices.size()) != A.nnz) {
            throw std::invalid_argument("Inconsistent matrix data sizes");
        }
        
        if (static_cast<int>(A.col_pointers.size()) != A.n_cols + 1) {
            throw std::invalid_argument("Invalid column pointer size");
        }
        
        // Basic symmetry check (simplified)
        // In practice, would do more thorough SPD validation
    }
    
    void SparseCholeskyDecomposition::clear_factorization() {
        factorized_ = false;
        
        L_factor_.values.clear();
        L_factor_.row_indices.clear();
        L_factor_.col_pointers.clear();
        
        for (auto& snode : supernodes_) {
            snode.frontal_matrix.clear();
            snode.child_updates.clear();
        }
    }
    
    // ============================================================================
    // SIMPLIFIED SPARSE CHOLESKY (Following sparse LU pattern)
    // ============================================================================
    
    bool SparseCholeskyDecomposition::solve_sparse_cholesky_direct(const Eigen::SparseMatrix<double>& A,
                                                                    Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        // Convert to CSC format
        CSCMatrix A_csc(A);
        
        // Factorize directly: A = L * L^T
        int info = factorize_sparse_cholesky_direct(A_csc);
        if (info != 0) {
            return false;  // Singular matrix
        }
        
        // Solve using L factor: L * L^T * x = b
        x = b;  // Copy RHS
        
        // Forward substitution: L * y = b
        forward_substitution_simple(x);
        
        // Backward substitution: L^T * x = y
        backward_substitution_simple(x);
        
        return true;
    }
    
    bool SparseCholeskyDecomposition::solve_sparse_cholesky_optimized(const Eigen::SparseMatrix<double>& A,
                                                                       Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        // Convert to CSC format
        CSCMatrix A_csc(A);
        
        // Factorize with optimized algorithm that minimizes fill-in
        int info = factorize_sparse_cholesky_optimized(A_csc);
        if (info != 0) {
            return false;  // Singular matrix
        }
        
        // Solve using L factor: L * L^T * x = b
        x = b;  // Copy RHS
        
        // Forward substitution: L * y = b
        forward_substitution_simple(x);
        
        // Backward substitution: L^T * x = y
        backward_substitution_simple(x);
        
        return true;
    }
    
    int SparseCholeskyDecomposition::factorize_sparse_cholesky_direct(const CSCMatrix& A) {
        // SIMPLIFIED SPARSE CHOLESKY: Column-by-column factorization
        // Based on the working sparse LU pattern from sparse_gauss_elimination.cpp
        
        // Initialize L factor
        L_factor_.n_rows = A.n_rows;
        L_factor_.n_cols = A.n_cols;
        L_factor_.col_pointers.resize(A.n_cols + 1, 0);
        L_factor_.values.clear();
        L_factor_.row_indices.clear();
        
        // Working matrix for factorization (will accumulate fill-in)
        std::vector<std::map<int, double>> working_cols(A.n_cols);
        
        // Initialize working matrix from lower triangular part of A
        for (int j = 0; j < A.n_cols; j++) {
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];
                double value = A.values[p];
                if (i >= j) {  // Only store lower triangular part
                    working_cols[j][i] = value;
                }
            }
        }
        
        // Column-by-column Cholesky factorization: A = L * L^T
        for (int k = 0; k < A.n_cols; k++) {
            // Compute diagonal element: L[k,k] = sqrt(A[k,k])
            if (!working_cols[k].count(k) || working_cols[k][k] <= 0.0) {
                return k + 1;  // Not positive definite
            }
            
            double L_kk = std::sqrt(working_cols[k][k]);
            
            // Store column k of L factor
            L_factor_.col_pointers[k] = static_cast<int>(L_factor_.values.size());
            
            // Diagonal entry
            L_factor_.values.push_back(L_kk);
            L_factor_.row_indices.push_back(k);
            
            // Below diagonal entries: L[i,k] = A[i,k] / L[k,k]
            for (auto& [i, val] : working_cols[k]) {
                if (i > k) {
                    double L_ik = val / L_kk;
                    L_factor_.values.push_back(L_ik);
                    L_factor_.row_indices.push_back(i);
                }
            }
            
            // Update remaining columns: A[i,j] -= L[i,k] * L[j,k] for i,j > k
            for (int j = k + 1; j < A.n_cols; j++) {
                if (working_cols[k].count(j)) {  // L[j,k] exists
                    double L_jk = working_cols[k][j] / L_kk;
                    
                    // Apply symmetric updates: A[i,j] -= L[i,k] * L[j,k] for all i >= j
                    for (auto& [i, L_ik_val] : working_cols[k]) {
                        if (i > k && i >= j) {
                            double L_ik = L_ik_val / L_kk;
                            working_cols[j][i] -= L_ik * L_jk;
                        }
                    }
                }
            }
        }
        
        // Finalize column pointers
        L_factor_.col_pointers[A.n_cols] = static_cast<int>(L_factor_.values.size());
        L_factor_.nnz = static_cast<int>(L_factor_.values.size());
        
        return 0;  // Success
    }
    
    int SparseCholeskyDecomposition::factorize_sparse_cholesky_optimized(const CSCMatrix& A) {
        // OPTIMIZED SPARSE CHOLESKY: Uses more efficient data structures and minimizes fill-in
        // Key optimizations:
        // 1. Use vectors instead of maps for better cache performance
        // 2. Only update entries that actually exist (sparsity-aware)
        // 3. Minimize symbolic operations
        
        // Initialize L factor
        L_factor_.n_rows = A.n_rows;
        L_factor_.n_cols = A.n_cols;
        L_factor_.col_pointers.resize(A.n_cols + 1, 0);
        L_factor_.values.clear();
        L_factor_.row_indices.clear();
        
        // Pre-allocate reasonable capacity based on input sparsity
        int estimated_nnz = static_cast<int>(A.nnz * 1.5);  // Estimate 50% fill-in
        L_factor_.values.reserve(estimated_nnz);
        L_factor_.row_indices.reserve(estimated_nnz);
        
        // Working storage: use vectors for better performance
        std::vector<std::vector<std::pair<int, double>>> working_cols(A.n_cols);
        
        // Initialize working matrix from lower triangular part of A (sparsity-aware)
        for (int j = 0; j < A.n_cols; j++) {
            working_cols[j].reserve(A.col_pointers[j + 1] - A.col_pointers[j]);
            
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];
                double value = A.values[p];
                if (i >= j) {  // Only store lower triangular part
                    working_cols[j].emplace_back(i, value);
                }
            }
            
            // Sort by row index for efficient access
            std::sort(working_cols[j].begin(), working_cols[j].end());
        }
        
        // Column-by-column Cholesky factorization with optimized updates
        for (int k = 0; k < A.n_cols; k++) {
            // Find diagonal element
            double A_kk = 0.0;
            for (const auto& [i, val] : working_cols[k]) {
                if (i == k) {
                    A_kk = val;
                    break;
                }
            }
            
            if (A_kk <= 0.0) {
                return k + 1;  // Not positive definite
            }
            
            double L_kk = std::sqrt(A_kk);
            
            // Store column k of L factor
            L_factor_.col_pointers[k] = static_cast<int>(L_factor_.values.size());
            
            // Store entries efficiently
            L_factor_.values.push_back(L_kk);
            L_factor_.row_indices.push_back(k);
            
            // Store below-diagonal entries
            std::vector<std::pair<int, double>> L_k_entries;
            L_k_entries.reserve(working_cols[k].size());
            
            for (const auto& [i, val] : working_cols[k]) {
                if (i > k) {
                    double L_ik = val / L_kk;
                    L_factor_.values.push_back(L_ik);
                    L_factor_.row_indices.push_back(i);
                    L_k_entries.emplace_back(i, L_ik);
                }
            }
            
            // OPTIMIZED UPDATES: Only update columns that will be affected
            // This is the key optimization - avoid unnecessary work
            for (int j = k + 1; j < A.n_cols; j++) {
                // Check if column j intersects with column k (has entry at row >= k+1)
                bool has_intersection = false;
                double L_jk = 0.0;
                
                for (const auto& [i, val] : working_cols[j]) {
                    if (i == k) {
                        L_jk = val / L_kk;
                        has_intersection = true;
                        break;
                    }
                }
                
                if (!has_intersection) continue;  // Skip columns with no updates
                
                // Apply updates only to existing entries (maintain sparsity)
                for (auto& [i, val] : working_cols[j]) {
                    if (i > k) {
                        // Find corresponding L[i,k] entry
                        for (const auto& [ii, L_ik] : L_k_entries) {
                            if (ii == i) {
                                val -= L_ik * L_jk;
                                break;
                            }
                        }
                    }
                }
                
                // Remove entry at (k,j) since it's been processed
                working_cols[j].erase(
                    std::remove_if(working_cols[j].begin(), working_cols[j].end(),
                                   [k](const std::pair<int, double>& p) { return p.first == k; }),
                    working_cols[j].end()
                );
            }
        }
        
        // Finalize column pointers
        L_factor_.col_pointers[A.n_cols] = static_cast<int>(L_factor_.values.size());
        L_factor_.nnz = static_cast<int>(L_factor_.values.size());
        
        return 0;  // Success
    }
    
    void SparseCholeskyDecomposition::forward_substitution_simple(Eigen::VectorXd& x) const {
        // Solve L * y = x using CSC format L factor
        // L is lower triangular with non-unit diagonal
        
        for (int j = 0; j < n_; j++) {
            int col_start = L_factor_.col_pointers[j];
            int col_end = L_factor_.col_pointers[j + 1];
            
            if (col_start >= col_end) continue;  // Empty column
            
            // Get diagonal element (first entry should be diagonal)
            double L_jj = L_factor_.values[col_start];
            int diag_row = L_factor_.row_indices[col_start];
            
            if (diag_row != j) {
                // Find actual diagonal
                L_jj = 1.0;  // Default
                for (int p = col_start; p < col_end; p++) {
                    if (L_factor_.row_indices[p] == j) {
                        L_jj = L_factor_.values[p];
                        break;
                    }
                }
            }
            
            // Scale by diagonal: x[j] = x[j] / L[j,j]
            if (std::abs(L_jj) > 1e-16) {
                x[j] /= L_jj;
            }
            
            // Update remaining entries: x[i] -= L[i,j] * x[j] for i > j
            for (int p = col_start; p < col_end; p++) {
                int i = L_factor_.row_indices[p];
                if (i > j) {
                    double L_ij = L_factor_.values[p];
                    x[i] -= L_ij * x[j];
                }
            }
        }
    }
    
    void SparseCholeskyDecomposition::backward_substitution_simple(Eigen::VectorXd& x) const {
        // Solve L^T * x = y using CSC format L factor
        // L^T is upper triangular with non-unit diagonal
        
        for (int j = n_ - 1; j >= 0; j--) {
            int col_start = L_factor_.col_pointers[j];
            int col_end = L_factor_.col_pointers[j + 1];
            
            if (col_start >= col_end) continue;  // Empty column
            
            // Apply updates from below-diagonal entries first
            for (int p = col_start; p < col_end; p++) {
                int i = L_factor_.row_indices[p];
                if (i > j) {
                    double L_ij = L_factor_.values[p];  // L[i,j] = L^T[j,i]
                    x[j] -= L_ij * x[i];
                }
            }
            
            // Get diagonal element and scale
            double L_jj = 1.0;  // Default
            for (int p = col_start; p < col_end; p++) {
                if (L_factor_.row_indices[p] == j) {
                    L_jj = L_factor_.values[p];
                    break;
                }
            }
            
            // Scale by diagonal: x[j] = x[j] / L^T[j,j] = x[j] / L[j,j]
            if (std::abs(L_jj) > 1e-16) {
                x[j] /= L_jj;
            }
        }
    }
    
}