#include "linear_algebra/sparse_gauss_elimination.hpp"

namespace LinearAlgebra {
    // ============================================================================
    // CSCMatrix IMPLEMENTATION
    // ============================================================================
    
    CSCMatrix::CSCMatrix(const Eigen::SparseMatrix<double>& eigen_sparse) {
        convert_from_eigen(eigen_sparse);
    }
    
    void CSCMatrix::convert_from_eigen(const Eigen::SparseMatrix<double>& eigen_sparse) {
        if (!eigen_sparse.isCompressed()) {
            throw std::runtime_error("Input matrix must be compressed");
        }
        
        n_rows = static_cast<int>(eigen_sparse.rows());
        n_cols = static_cast<int>(eigen_sparse.cols());
        nnz = static_cast<int>(eigen_sparse.nonZeros());
        
        // Copy values and row indices
        values.resize(nnz);
        row_indices.resize(nnz);
        col_pointers.resize(n_cols + 1);
        
        const double* eigen_values = eigen_sparse.valuePtr();
        const int* eigen_inner = eigen_sparse.innerIndexPtr();
        const int* eigen_outer = eigen_sparse.outerIndexPtr();
        
        std::copy(eigen_values, eigen_values + nnz, values.begin());
        std::copy(eigen_inner, eigen_inner + nnz, row_indices.begin());
        std::copy(eigen_outer, eigen_outer + n_cols + 1, col_pointers.begin());
    }

    // ============================================================================
    // ELIMINATION TREE IMPLEMENTATION
    // ============================================================================
    
    void EliminationTree::build_from_pattern(const CSCMatrix& A) {
        std::fill(parent.begin(), parent.end(), -1);
        for (auto& child_list : children) {
            child_list.clear();
        }
        
        std::vector<int> ancestor(n, -1);
        
        for (int j = 0; j < n; j++) {
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];
                if (i >= j) continue;  // Only consider upper triangular part
                
                int r = i;
                while (ancestor[r] != -1 && ancestor[r] != j) {
                    int next = ancestor[r];
                    ancestor[r] = j;
                    r = next;
                }
                
                if (ancestor[r] == -1) {
                    ancestor[r] = j;
                    parent[r] = j;
                }
            }
        }
        
        // Build children lists
        for (int i = 0; i < n; i++) {
            if (parent[i] != -1) {
                children[parent[i]].push_back(i);
            }
        }
    }
    
    std::vector<int> EliminationTree::get_postorder() const {
        std::vector<int> order;
        std::vector<bool> visited(n, false);
        
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                dfs_postorder(i, visited, order);
            }
        }
        
        return order;
    }
    
    void EliminationTree::dfs_postorder(int node, std::vector<bool>& visited, std::vector<int>& order) const {
        visited[node] = true;
        
        for (int child : children[node]) {
            if (!visited[child]) {
                dfs_postorder(child, visited, order);
            }
        }
        
        order.push_back(node);
    }

    // ============================================================================
    // SPARSE GAUSS ELIMINATION IMPLEMENTATION
    // ============================================================================
    
    SparseGaussElimination::SparseGaussElimination()
        : n_(0)
        , analyzed_(false)
        , factorized_(false)
        , pivot_threshold_(0.1)
        , max_threads_(0)
        , use_scaling_(true)
        , analyze_time_(0.0)
        , factorize_time_(0.0)
        , solve_time_(0.0)
        , original_nnz_(0)
        , factor_nnz_(0) {
        
        // Auto-detect thread count if not specified
        if (max_threads_ == 0) {
            max_threads_ = std::min(static_cast<int>(std::thread::hardware_concurrency()), 8);
        }
        
        // Configure optimal BLAS threading for Apple Accelerate
        #ifdef USE_ACCELERATE
            // Set optimal thread count for Accelerate
            setenv("VECLIB_MAXIMUM_THREADS", std::to_string(max_threads_).c_str(), 1);
        #else
            // Configure OpenBLAS threading
            setenv("OPENBLAS_NUM_THREADS", std::to_string(max_threads_).c_str(), 1);
        #endif
    }
    
    // ============================================================================
    // MAIN SOLVER INTERFACE
    // ============================================================================
    
    bool SparseGaussElimination::analyze(const CSCMatrix& A) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            validate_matrix(A);
            
            n_ = A.n_rows;
            original_nnz_ = A.nnz;
            
            // Clear any previous analysis
            analyzed_ = false;
            factorized_ = false;
            
            // Step 1: Compute fill-reducing ordering
            compute_column_ordering(A);
            
            // Step 2: Build elimination tree
            build_elimination_tree(A);
            
            // Step 3: Detect and merge supernodes
            detect_supernodes(A);
            
            // Step 4: Analyze memory requirements
            analyze_memory_requirements();
            
            // Step 5: Initialize dense kernel
            dense_kernel_ = std::make_unique<GaussElimination>(1);  // Will resize as needed
            
            analyzed_ = true;
            
        } catch (const std::exception& e) {
            std::cerr << "Analysis failed: " << e.what() << std::endl;
            return false;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        analyze_time_ = std::chrono::duration<double>(end_time - start_time).count();
        
        return true;
    }
    
    bool SparseGaussElimination::analyze(const Eigen::SparseMatrix<double>& A) {
        CSCMatrix csc_matrix(A);
        return analyze(csc_matrix);
    }
    
    int SparseGaussElimination::factorize(const CSCMatrix& A) {
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
                scale_matrix(A_work);
            }
            
            // Step 2: Factor all supernodes
            int result = factor_all_supernodes(A_work);
            
            if (result == 0) {
                // Step 3: Finalize CSC structure for factors
                finalize_csc_factors();
                factorized_ = true;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            factorize_time_ = std::chrono::duration<double>(end_time - start_time).count();
            
            return result;
            
        } catch (const std::exception& e) {
            std::cerr << "Factorization failed: " << e.what() << std::endl;
            return -1;
        }
    }
    
    int SparseGaussElimination::factorize(const Eigen::SparseMatrix<double>& A) {
        CSCMatrix csc_matrix(A);
        return factorize(csc_matrix);
    }
    
    bool SparseGaussElimination::solve(Eigen::VectorXd& x, const Eigen::VectorXd& b) const {
        if (!factorized_) {
            throw std::runtime_error("Must call factorize() before solve()");
        }
        
        if (b.size() != n_) {
            throw std::invalid_argument("Vector size does not match matrix size");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        x = b;  // Copy RHS
        
        // Apply scaling and permutations, then solve
        if (use_scaling_) {
            apply_scaling(x, true);  // Forward scaling
        }
        
        apply_col_permutation(x);
        forward_substitution(x);
        backward_substitution(x);
        apply_row_permutation(x);
        
        if (use_scaling_) {
            apply_scaling(x, false);  // Backward scaling
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        solve_time_ = std::chrono::duration<double>(end_time - start_time).count();
        
        return true;
    }
    
    bool SparseGaussElimination::solve(Eigen::MatrixXd& X, const Eigen::MatrixXd& B) const {
        if (B.rows() != n_) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        
        X.resize(B.rows(), B.cols());
        X = B;  // Copy RHS
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Apply scaling and permutations
        if (use_scaling_) {
            for (int j = 0; j < X.cols(); j++) {
                Eigen::VectorXd x_col = X.col(j);
                apply_scaling(x_col, true);  // Forward scaling
                X.col(j) = x_col;
            }
        }
        
        for (int j = 0; j < X.cols(); j++) {
            Eigen::VectorXd x_col = X.col(j);
            apply_col_permutation(x_col);
            X.col(j) = x_col;
        }
        
        // Forward substitution using Level-3 BLAS for multiple RHS
        auto forward_schedule = compute_level_schedule(true);
        for (const auto& level : forward_schedule) {
            for (int snode_id : level) {
                if (X.cols() >= 4) {
                    // Use Level-3 BLAS optimized version for multiple RHS
                    solve_supernode_forward_multiple(snode_id, X);
                } else {
                    // Use single RHS version for few columns
                    for (int j = 0; j < X.cols(); j++) {
                        Eigen::VectorXd x_col = X.col(j);
                        solve_supernode_forward(snode_id, x_col);
                        X.col(j) = x_col;
                    }
                }
            }
        }
        
        // Backward substitution using Level-3 BLAS for multiple RHS
        auto backward_schedule = compute_level_schedule(false);
        for (const auto& level : backward_schedule) {
            for (int snode_id : level) {
                if (X.cols() >= 4) {
                    // Use Level-3 BLAS optimized version for multiple RHS
                    solve_supernode_backward_multiple(snode_id, X);
                } else {
                    // Use single RHS version for few columns
                    for (int j = 0; j < X.cols(); j++) {
                        Eigen::VectorXd x_col = X.col(j);
                        solve_supernode_backward(snode_id, x_col);
                        X.col(j) = x_col;
                    }
                }
            }
        }
        
        // Apply remaining transformations
        for (int j = 0; j < X.cols(); j++) {
            Eigen::VectorXd x_col = X.col(j);
            apply_row_permutation(x_col);
            X.col(j) = x_col;
        }
        
        if (use_scaling_) {
            for (int j = 0; j < X.cols(); j++) {
                Eigen::VectorXd x_col = X.col(j);
                apply_scaling(x_col, false);  // Backward scaling
                X.col(j) = x_col;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        solve_time_ += std::chrono::duration<double>(end_time - start_time).count();
        
        return true;
    }
    
    bool SparseGaussElimination::solve_system(const Eigen::SparseMatrix<double>& A,
                                                Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        // HYBRID SPARSE APPROACH: Choose algorithm based on matrix size and sparsity
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        n_ = static_cast<int>(A.rows());
        original_nnz_ = static_cast<size_t>(A.nonZeros());
        double density = static_cast<double>(original_nnz_) / (n_ * n_);
        
        bool success = false;
        
        // STRATEGY: Use different algorithms based on matrix characteristics
        if (n_ < 500 || density > 0.1) {
            // SMALL OR DENSE MATRICES: Use BLAS-optimized dense kernel
            Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
            LinearAlgebra::GaussElimination dense_solver(A_dense.rows());
            Eigen::MatrixXd A_copy = A_dense;
            
            success = dense_solver.solve(A_copy, x, b);
            factor_nnz_ = static_cast<size_t>(n_ * n_);  // Dense factorization
            
        } else {
            // LARGE SPARSE MATRICES: Use true sparse LU factorization
            success = solve_sparse_lu(A, x, b);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        solve_time_ = std::chrono::duration<double>(end_time - start_time).count();
        
        return success;
    }
    
    // ============================================================================
    // PERFORMANCE AND DIAGNOSTICS
    // ============================================================================
    
    double SparseGaussElimination::getFactorizationGFLOPs() const {
        if (factorize_time_ <= 0.0) return 0.0;
        
        // Estimate flops: roughly nnz(L+U) * average_supernode_ops
        double estimated_flops = 0.0;
        for (const auto& snode : supernodes_) {
            double front_flops = (2.0 * snode.front_width * snode.front_width * snode.front_height) / 3.0;
            estimated_flops += front_flops;
        }
        
        return estimated_flops / (factorize_time_ * 1e9);
    }
    
    // ============================================================================
    // ANALYSIS PHASE IMPLEMENTATION
    // ============================================================================
    
    void SparseGaussElimination::compute_column_ordering(const CSCMatrix& A) {
        // Use COLAMD ordering for now (simplified version)
        col_perm_ = colamd_ordering(A);
        
        // Compute inverse permutation
        col_perm_inv_.resize(n_);
        for (int i = 0; i < n_; i++) {
            col_perm_inv_[col_perm_[i]] = i;
        }
        
        // Initialize row permutation as identity
        row_perm_.resize(n_);
        row_perm_inv_.resize(n_);
        std::iota(row_perm_.begin(), row_perm_.end(), 0);
        std::iota(row_perm_inv_.begin(), row_perm_inv_.end(), 0);
    }
    
    std::vector<int> SparseGaussElimination::colamd_ordering(const CSCMatrix& A) {
        // Simplified COLAMD implementation
        // In practice, you would use the actual COLAMD library
        
        std::vector<std::pair<int, int>> col_weights(A.n_cols);
        for (int j = 0; j < A.n_cols; j++) {
            int col_nnz = A.col_pointers[j + 1] - A.col_pointers[j];
            col_weights[j] = {col_nnz, j};
        }
        
        // Sort by number of non-zeros (ascending)
        std::sort(col_weights.begin(), col_weights.end());
        
        std::vector<int> ordering(A.n_cols);
        for (int i = 0; i < A.n_cols; i++) {
            ordering[i] = col_weights[i].second;
        }
        
        return ordering;
    }
    
    std::vector<int> SparseGaussElimination::amd_ordering(const CSCMatrix& A) {
        // Placeholder for AMD ordering
        // Would implement approximate minimum degree here
        return colamd_ordering(A);  // Fallback to COLAMD
    }
    
    void SparseGaussElimination::build_elimination_tree(const CSCMatrix& A) {
        elim_tree_ = std::make_unique<EliminationTree>(n_);
        
        // Apply column permutation to build tree on permuted matrix
        CSCMatrix A_perm = A;  // Create permuted version
        // ... (permutation logic would go here)
        
        elim_tree_->build_from_pattern(A_perm);
    }
    
    void SparseGaussElimination::detect_supernodes(const CSCMatrix& A) {
        supernodes_.clear();
        col_to_supernode_.resize(n_);
        
        // Start with each column as its own supernode
        for (int j = 0; j < n_; j++) {
            Supernode snode;
            snode.start_col = j;
            snode.num_cols = 1;
            snode.parent = elim_tree_->parent[j];
            snode.children = elim_tree_->children[j];
            
            supernodes_.push_back(snode);
            col_to_supernode_[j] = static_cast<int>(supernodes_.size() - 1);
        }
        
        // Merge adjacent columns with identical structure
        merge_supernodes();
        
        // Compute supernode structures
        compute_supernode_structure(A);
    }
    
    void SparseGaussElimination::merge_supernodes() {
        // Simplified supernode merging
        std::vector<Supernode> merged_supernodes;
        
        int current_snode = 0;
        while (current_snode < static_cast<int>(supernodes_.size())) {
            Supernode& snode = supernodes_[current_snode];
            
            // Try to merge with consecutive supernodes
            int next_snode = current_snode + 1;
            while (next_snode < static_cast<int>(supernodes_.size()) &&
                    can_merge_columns(snode.start_col + snode.num_cols - 1, 
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
            const Supernode& snode = supernodes_[s];
            for (int j = snode.start_col; j < snode.start_col + snode.num_cols; j++) {
                col_to_supernode_[j] = s;
            }
        }
    }
    
    bool SparseGaussElimination::can_merge_columns(int col1, int col2, const CSCMatrix& A) {
        // Simplified merging criterion
        // In practice, would check if columns have identical structure below diagonal
        return (col2 == col1 + 1);  // Only merge consecutive columns for now
    }
    
    void SparseGaussElimination::compute_supernode_structure(const CSCMatrix& A) {
        for (auto& snode : supernodes_) {
            // Compute row structure for this supernode
            std::unordered_set<int> row_set;
            
            for (int j = snode.start_col; j < snode.start_col + snode.num_cols; j++) {
                for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                    int i = A.row_indices[p];
                    if (i >= j) {  // Lower triangular part
                        row_set.insert(i);
                    }
                }
            }
            
            snode.row_structure.assign(row_set.begin(), row_set.end());
            std::sort(snode.row_structure.begin(), snode.row_structure.end());
            
            snode.front_height = static_cast<int>(snode.row_structure.size());
            snode.front_width = snode.num_cols;
            
            // Allocate frontal matrix
            int front_size = snode.front_height * snode.front_width;
            snode.frontal_matrix.resize(front_size, 0.0);
            snode.front_lda = snode.front_height;
        }
    }
    
    void SparseGaussElimination::analyze_memory_requirements() {
        // Estimate memory requirements for L and U factors
        factor_nnz_ = 0;
        
        for (const auto& snode : supernodes_) {
            // Lower triangular part
            size_t L_nnz = snode.front_height * snode.front_width;
            // Upper triangular part  
            size_t U_nnz = snode.front_width * snode.front_width;
            
            factor_nnz_ += L_nnz + U_nnz;
        }
        
        // Pre-allocate storage
        L_factor_.values.reserve(factor_nnz_);
        L_factor_.row_indices.reserve(factor_nnz_);
        U_factor_.values.reserve(factor_nnz_);
        U_factor_.row_indices.reserve(factor_nnz_);
    }
    
    // ============================================================================
    // FACTORIZATION PHASE IMPLEMENTATION
    // ============================================================================
    
    void SparseGaussElimination::initialize_factorization(const CSCMatrix& A) {
        // Initialize scaling vectors
        if (use_scaling_) {
            row_scale_.resize(n_, 1.0);
            col_scale_.resize(n_, 1.0);
            
            // Compute row and column scaling factors
            for (int j = 0; j < A.n_cols; j++) {
                double col_max = 0.0;
                for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                    int i = A.row_indices[p];
                    double abs_val = std::abs(A.values[p]);
                    col_max = std::max(col_max, abs_val);
                    row_scale_[i] = std::max(row_scale_[i], abs_val);
                }
                col_scale_[j] = col_max;
            }
            
            // Invert scaling factors
            for (int i = 0; i < n_; i++) {
                if (row_scale_[i] > 0.0) row_scale_[i] = 1.0 / row_scale_[i];
                if (col_scale_[i] > 0.0) col_scale_[i] = 1.0 / col_scale_[i];
            }
        }
        
        // Initialize L and U factor structures
        L_factor_.n_rows = n_;
        L_factor_.n_cols = n_;
        L_factor_.col_pointers.resize(n_ + 1, 0);
        
        U_factor_.n_rows = n_;
        U_factor_.n_cols = n_;
        U_factor_.col_pointers.resize(n_ + 1, 0);
    }
    
    void SparseGaussElimination::scale_matrix(CSCMatrix& A) {
        if (!use_scaling_) return;
        
        for (int j = 0; j < A.n_cols; j++) {
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];
                A.values[p] *= row_scale_[i] * col_scale_[j];
            }
        }
    }
    
    int SparseGaussElimination::factor_all_supernodes(const CSCMatrix& A) {
        // Get postorder traversal of elimination tree for dependency scheduling
        auto postorder = elim_tree_->get_postorder();
        
        // Factor supernodes in postorder (children before parents)
        for (int col : postorder) {
            int snode_id = col_to_supernode_[col];
            
            // Skip if this supernode was already processed
            if (snode_id == -1) continue;
            
            int result = factor_supernode(snode_id, A);
            if (result != 0) {
                return result;  // Singular matrix
            }
            
            // Mark processed columns
            const Supernode& snode = supernodes_[snode_id];
            for (int j = snode.start_col; j < snode.start_col + snode.num_cols; j++) {
                if (j != col) col_to_supernode_[j] = -1;
            }
        }
        
        return 0;  // Success
    }
    
    int SparseGaussElimination::factor_supernode(int snode_id, const CSCMatrix& A) {
        Supernode& snode = supernodes_[snode_id];
        
        // Step 1: Assemble frontal matrix from sparse matrix and updates
        assemble_frontal_matrix(snode_id, A);
        
        // Step 2: Add contributions from child supernodes
        add_child_contributions(snode_id);
        
        // Step 3: Factor the dense frontal matrix using our dense kernel
        int result = factor_dense_frontal(snode);
        if (result != 0) {
            return snode.start_col + result;  // Return global column index
        }
        
        // Step 4: Extract L and U factors from frontal matrix
        extract_factors_from_frontal(snode);
        
        // Step 5: Generate update matrix for parent supernodes
        generate_update_matrix(snode);
        
        return 0;  // Success
    }
    
    void SparseGaussElimination::assemble_frontal_matrix(int snode_id, const CSCMatrix& A) {
        Supernode& snode = supernodes_[snode_id];
        
        // Clear frontal matrix
        std::fill(snode.frontal_matrix.begin(), snode.frontal_matrix.end(), 0.0);
        
        // Copy sparse matrix entries into frontal matrix
        for (int jj = 0; jj < snode.num_cols; jj++) {
            int j = snode.start_col + jj;  // Global column index
            
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];  // Global row index
                double value = A.values[p];
                
                // Find position in frontal matrix
                auto it = std::find(snode.row_structure.begin(), snode.row_structure.end(), i);
                if (it != snode.row_structure.end()) {
                    int ii = static_cast<int>(it - snode.row_structure.begin());  // Local row index
                    snode.frontal_matrix[jj * snode.front_lda + ii] = value;
                }
            }
        }
    }
    
    void SparseGaussElimination::add_child_contributions(int snode_id) {
        Supernode& snode = supernodes_[snode_id];
        
        // Add update contributions from all child supernodes
        for (const auto& update : snode.child_updates) {
            // Map global indices to local frontal matrix indices
            for (int k = 0; k < update.width * update.height; k++) {
                int global_i = update.indices[k / update.width];
                int global_j = update.indices[k % update.width];
                
                // Find local positions
                auto row_it = std::find(snode.row_structure.begin(), snode.row_structure.end(), global_i);
                if (row_it == snode.row_structure.end()) continue;
                
                int col_offset = global_j - snode.start_col;
                if (col_offset < 0 || col_offset >= snode.num_cols) continue;
                
                int local_i = static_cast<int>(row_it - snode.row_structure.begin());
                int local_j = col_offset;
                
                snode.frontal_matrix[local_j * snode.front_lda + local_i] += update.values[k];
            }
        }
    }
    
    int SparseGaussElimination::factor_dense_frontal(Supernode& snode) {
        // Use our existing dense GaussElimination kernel
        if (!dense_kernel_ || dense_kernel_->size() != static_cast<size_t>(snode.front_height)) {
            dense_kernel_ = std::make_unique<GaussElimination>(snode.front_height);
        }
        
        // Factor the frontal matrix in-place
        return dense_kernel_->factorize(snode.frontal_matrix.data(), snode.front_lda);
    }
    
    void SparseGaussElimination::extract_factors_from_frontal(const Supernode& snode) {
        // FIXED: Extract L and U factors with correct CSC indexing
        // We'll build temporary column-wise storage and then properly construct CSC
        
        // Extract L factor (unit lower triangular)
        for (int jj = 0; jj < snode.num_cols; jj++) {
            int j = snode.start_col + jj;  // Global column index
            
            // Store starting position for this column
            if (j < n_) {
                L_factor_.col_pointers[j] = static_cast<int>(L_factor_.values.size());
            }
            
            // Add diagonal entry (1.0 for unit lower triangular)
            L_factor_.values.push_back(1.0);
            L_factor_.row_indices.push_back(snode.row_structure[jj]);
            
            // Add below-diagonal entries
            for (int ii = jj + 1; ii < snode.front_height; ii++) {
                int i = snode.row_structure[ii];  // Global row index
                double value = snode.frontal_matrix[jj * snode.front_lda + ii];
                
                if (std::abs(value) > 1e-16) {
                    L_factor_.values.push_back(value);
                    L_factor_.row_indices.push_back(i);
                }
            }
        }
        
        // Extract U factor (upper triangular with non-unit diagonal)
        for (int jj = 0; jj < snode.num_cols; jj++) {
            int j = snode.start_col + jj;  // Global column index
            
            // Store starting position for this column
            if (j < n_) {
                U_factor_.col_pointers[j] = static_cast<int>(U_factor_.values.size());
            }
            
            // Add upper triangular entries (including diagonal)
            for (int ii = 0; ii <= jj; ii++) {
                int i = snode.row_structure[ii];  // Global row index
                double value = snode.frontal_matrix[jj * snode.front_lda + ii];
                
                if (std::abs(value) > 1e-16) {
                    U_factor_.values.push_back(value);
                    U_factor_.row_indices.push_back(i);
                }
            }
        }
    }
    
    void SparseGaussElimination::generate_update_matrix(const Supernode& snode) {
        if (snode.parent == -1) return;  // Root supernode has no parent
        
        // Create Schur complement update for parent supernode
        int update_size = snode.front_height - snode.num_cols;
        if (update_size <= 0) return;
        
        Supernode::Update update;
        update.width = update_size;
        update.height = update_size;
        update.values.resize(update_size * update_size);
        update.indices.resize(update_size);
        
        // Copy update matrix indices (rows after the pivot block)
        for (int i = 0; i < update_size; i++) {
            update.indices[i] = snode.row_structure[snode.num_cols + i];
        }
        
        // Copy update matrix values
        for (int j = 0; j < update_size; j++) {
            for (int i = 0; i < update_size; i++) {
                int frontal_i = snode.num_cols + i;
                int frontal_j = snode.num_cols + j;
                update.values[j * update_size + i] = 
                    snode.frontal_matrix[frontal_j * snode.front_lda + frontal_i];
            }
        }
        
        // Add update to parent supernode
        int parent_snode_id = col_to_supernode_[snode.parent];
        if (parent_snode_id >= 0) {
            supernodes_[parent_snode_id].child_updates.push_back(std::move(update));
        }
    }
    
    // ============================================================================
    // SOLVE PHASE IMPLEMENTATION
    // ============================================================================
    
    void SparseGaussElimination::apply_row_permutation(Eigen::VectorXd& x) const {
        Eigen::VectorXd x_perm(n_);
        for (int i = 0; i < n_; i++) {
            x_perm[i] = x[row_perm_[i]];
        }
        x = x_perm;
    }
    
    void SparseGaussElimination::apply_col_permutation(Eigen::VectorXd& x) const {
        Eigen::VectorXd x_perm(n_);
        for (int i = 0; i < n_; i++) {
            x_perm[col_perm_[i]] = x[i];
        }
        x = x_perm;
    }
    
    void SparseGaussElimination::forward_substitution(Eigen::VectorXd& x) const {
        // Solve L * y = x using CSC format L factor
        // L is unit lower triangular, so L[j,j] = 1.0
        
        for (int j = 0; j < n_; j++) {
            // Process column j of L
            for (int p = L_factor_.col_pointers[j]; p < L_factor_.col_pointers[j + 1]; p++) {
                int i = L_factor_.row_indices[p];
                double L_ij = L_factor_.values[p];
                
                if (i == j) {
                    // Diagonal entry (should be 1.0 for unit lower triangular)
                    continue;
                } else if (i > j) {
                    // Below diagonal: x[i] -= L[i,j] * x[j]
                    x[i] -= L_ij * x[j];
                }
            }
        }
    }
    
    void SparseGaussElimination::backward_substitution(Eigen::VectorXd& x) const {
        // Solve U * x = y using CSC format U factor
        // U is upper triangular with non-unit diagonal
        
        for (int j = n_ - 1; j >= 0; j--) {
            // Find diagonal entry U[j,j]
            double U_jj = 1.0;
            for (int p = U_factor_.col_pointers[j]; p < U_factor_.col_pointers[j + 1]; p++) {
                int i = U_factor_.row_indices[p];
                if (i == j) {
                    U_jj = U_factor_.values[p];
                    break;
                }
            }
            
            // Diagonal solve: x[j] /= U[j,j]
            if (std::abs(U_jj) > 1e-16) {
                x[j] /= U_jj;
            }
            
            // Update other entries: x[i] -= U[i,j] * x[j] for i < j
            for (int p = U_factor_.col_pointers[j]; p < U_factor_.col_pointers[j + 1]; p++) {
                int i = U_factor_.row_indices[p];
                if (i < j) {
                    double U_ij = U_factor_.values[p];
                    x[i] -= U_ij * x[j];
                }
            }
        }
    }
    
    void SparseGaussElimination::apply_scaling(Eigen::VectorXd& x, bool forward) const {
        if (!use_scaling_) return;
        
        if (forward) {
            for (int i = 0; i < n_; i++) {
                x[i] *= row_scale_[i];
            }
        } else {
            for (int i = 0; i < n_; i++) {
                x[i] *= col_scale_[i];
            }
        }
    }
    
    // ============================================================================
    // UTILITY FUNCTIONS
    // ============================================================================
    
    void SparseGaussElimination::validate_matrix(const CSCMatrix& A) const {
        if (A.n_rows != A.n_cols) {
            throw std::invalid_argument("Matrix must be square");
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
    }
    
    void SparseGaussElimination::clear_factorization() {
        factorized_ = false;
        
        L_factor_.values.clear();
        L_factor_.row_indices.clear();
        L_factor_.col_pointers.clear();
        
        U_factor_.values.clear();
        U_factor_.row_indices.clear();
        U_factor_.col_pointers.clear();
        
        for (auto& snode : supernodes_) {
            snode.frontal_matrix.clear();
            snode.child_updates.clear();
        }
    }
    
    // ============================================================================
    // LEVEL SCHEDULING AND SUPERNODE-BASED TRIANGULAR SOLVES
    // ============================================================================
    
    void SparseGaussElimination::finalize_csc_factors() {
        // FIXED: Properly finalize CSC structure for L and U factors
        
        // Finalize L factor - ensure all column pointers are set correctly
        for (int j = 1; j <= n_; j++) {
            if (j >= static_cast<int>(L_factor_.col_pointers.size()) || L_factor_.col_pointers[j] == 0) {
                // If column j wasn't processed, it has same start as previous column
                if (j < static_cast<int>(L_factor_.col_pointers.size())) {
                    L_factor_.col_pointers[j] = (j > 0) ? L_factor_.col_pointers[j-1] : 0;
                }
            }
        }
        // Ensure final pointer is set to total number of entries
        if (L_factor_.col_pointers.size() > n_) {
            L_factor_.col_pointers[n_] = static_cast<int>(L_factor_.values.size());
        }
        L_factor_.nnz = static_cast<int>(L_factor_.values.size());
        
        // Finalize U factor - ensure all column pointers are set correctly  
        for (int j = 1; j <= n_; j++) {
            if (j >= static_cast<int>(U_factor_.col_pointers.size()) || U_factor_.col_pointers[j] == 0) {
                // If column j wasn't processed, it has same start as previous column
                if (j < static_cast<int>(U_factor_.col_pointers.size())) {
                    U_factor_.col_pointers[j] = (j > 0) ? U_factor_.col_pointers[j-1] : 0;
                }
            }
        }
        // Ensure final pointer is set to total number of entries
        if (U_factor_.col_pointers.size() > n_) {
            U_factor_.col_pointers[n_] = static_cast<int>(U_factor_.values.size());
        }
        U_factor_.nnz = static_cast<int>(U_factor_.values.size());
    }
    
    std::vector<std::vector<int>> SparseGaussElimination::compute_level_schedule(bool forward) const {
        std::vector<std::vector<int>> levels;
        std::vector<int> level_of_snode(supernodes_.size(), -1);
        std::vector<int> dependencies(supernodes_.size(), 0);
        
        if (forward) {
            // Forward scheduling: children must be processed before parents
            for (size_t s = 0; s < supernodes_.size(); s++) {
                const auto& snode = supernodes_[s];
                dependencies[s] = static_cast<int>(snode.children.size());
            }
            
            // Start with leaf nodes (no children)
            std::queue<int> ready_queue;
            for (size_t s = 0; s < supernodes_.size(); s++) {
                if (dependencies[s] == 0) {
                    ready_queue.push(static_cast<int>(s));
                    level_of_snode[s] = 0;
                }
            }
            
            // Build levels bottom-up
            while (!ready_queue.empty()) {
                std::vector<int> current_level;
                int level_size = static_cast<int>(ready_queue.size());
                
                for (int i = 0; i < level_size; i++) {
                    int snode_id = ready_queue.front();
                    ready_queue.pop();
                    current_level.push_back(snode_id);
                    
                    // Update parent dependencies
                    const auto& snode = supernodes_[snode_id];
                    if (snode.parent != -1) {
                        int parent_snode = col_to_supernode_[snode.parent];
                        if (parent_snode >= 0) {
                            dependencies[parent_snode]--;
                            if (dependencies[parent_snode] == 0) {
                                ready_queue.push(parent_snode);
                                level_of_snode[parent_snode] = static_cast<int>(levels.size()) + 1;
                            }
                        }
                    }
                }
                
                if (!current_level.empty()) {
                    levels.push_back(current_level);
                }
            }
        } else {
            // Backward scheduling: parents must be processed before children
            for (size_t s = 0; s < supernodes_.size(); s++) {
                const auto& snode = supernodes_[s];
                dependencies[s] = (snode.parent != -1) ? 1 : 0;
            }
            
            // Start with root nodes (no parent)
            std::queue<int> ready_queue;
            for (size_t s = 0; s < supernodes_.size(); s++) {
                if (dependencies[s] == 0) {
                    ready_queue.push(static_cast<int>(s));
                    level_of_snode[s] = 0;
                }
            }
            
            // Build levels top-down
            while (!ready_queue.empty()) {
                std::vector<int> current_level;
                int level_size = static_cast<int>(ready_queue.size());
                
                for (int i = 0; i < level_size; i++) {
                    int snode_id = ready_queue.front();
                    ready_queue.pop();
                    current_level.push_back(snode_id);
                    
                    // Update children dependencies
                    const auto& snode = supernodes_[snode_id];
                    for (int child_col : snode.children) {
                        int child_snode = col_to_supernode_[child_col];
                        if (child_snode >= 0) {
                            dependencies[child_snode]--;
                            if (dependencies[child_snode] == 0) {
                                ready_queue.push(child_snode);
                                level_of_snode[child_snode] = static_cast<int>(levels.size()) + 1;
                            }
                        }
                    }
                }
                
                if (!current_level.empty()) {
                    levels.push_back(current_level);
                }
            }
        }
        
        return levels;
    }
    
    void SparseGaussElimination::solve_supernode_forward(int snode_id, Eigen::VectorXd& x) const {
        const Supernode& snode = supernodes_[snode_id];
        
        // Extract relevant portion of solution vector
        std::vector<double> x_local(snode.front_height);
        for (int i = 0; i < snode.front_height; i++) {
            int global_row = snode.row_structure[i];
            x_local[i] = x[global_row];
        }
        
        // Forward substitution: L * y = x_local, where L is unit lower triangular
        
        // OPTIMIZATION: For larger supernodes, prefer Level-3 BLAS (DTRSM) over Level-2 (DTRSV)
        if (snode.front_width >= 32) {
            // Use DTRSM (Level-3 BLAS) - more efficient for large blocks
            // Treat single RHS as matrix with nrhs=1
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                       snode.front_height, 1, 1.0,
                       snode.frontal_matrix.data(), snode.front_lda,
                       x_local.data(), snode.front_height);
        } else {
            // Use DTRSV (Level-2 BLAS) for smaller supernodes
            cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                       snode.front_height, snode.frontal_matrix.data(), snode.front_lda,
                       x_local.data(), 1);
        }
        
        // Update: Apply L21 * y1 to remaining entries (Level-3 BLAS GEMV or GEMM)
        if (snode.front_height > snode.front_width) {
            int remaining_rows = snode.front_height - snode.front_width;
            
            // y2 = y2 - L21 * y1 (GEMV - Level-2 BLAS)
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                       remaining_rows, snode.front_width, -1.0,
                       snode.frontal_matrix.data() + snode.front_width, snode.front_lda,
                       x_local.data(), 1, 1.0,
                       x_local.data() + snode.front_width, 1);
        }
        
        // Copy result back to global solution vector
        for (int i = 0; i < snode.front_height; i++) {
            int global_row = snode.row_structure[i];
            x[global_row] = x_local[i];
        }
    }
    
    void SparseGaussElimination::solve_supernode_backward(int snode_id, Eigen::VectorXd& x) const {
        const Supernode& snode = supernodes_[snode_id];
        
        // Extract relevant portion of solution vector
        std::vector<double> x_local(snode.front_width); // Only need pivot block for backward solve
        for (int i = 0; i < snode.front_width; i++) {
            int global_row = snode.row_structure[i];
            x_local[i] = x[global_row];
        }
        
        // Backward substitution: U11 * x1 = y1, where U11 is upper triangular
        
        // OPTIMIZATION: For larger supernodes, prefer Level-3 BLAS (DTRSM) over Level-2 (DTRSV)
        if (snode.front_width >= 32) {
            // Use DTRSM (Level-3 BLAS) - more efficient for large blocks
            // Treat single RHS as matrix with nrhs=1
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                       snode.front_width, 1, 1.0,
                       snode.frontal_matrix.data(), snode.front_lda,
                       x_local.data(), snode.front_width);
        } else {
            // Use DTRSV (Level-2 BLAS) for smaller supernodes
            cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                       snode.front_width, snode.frontal_matrix.data(), snode.front_lda,
                       x_local.data(), 1);
        }
        
        // Update remaining entries: x2 = x2 - U12^T * x1 (GEMV - Level-2 BLAS)
        if (snode.front_height > snode.front_width) {
            int remaining_rows = snode.front_height - snode.front_width;
            
            // Extract x2 (remaining part)
            std::vector<double> x2_local(remaining_rows);
            for (int i = 0; i < remaining_rows; i++) {
                int global_row = snode.row_structure[snode.front_width + i];
                x2_local[i] = x[global_row];
            }
            
            // x2 = x2 - U12^T * x1 (GEMV with transpose)
            cblas_dgemv(CblasColMajor, CblasTrans,
                       snode.front_width, remaining_rows, -1.0,
                       snode.frontal_matrix.data() + snode.front_width * snode.front_lda, snode.front_lda,
                       x_local.data(), 1, 1.0,
                       x2_local.data(), 1);
            
            // Copy x2 back to global solution vector
            for (int i = 0; i < remaining_rows; i++) {
                int global_row = snode.row_structure[snode.front_width + i];
                x[global_row] = x2_local[i];
            }
        }
        
        // Copy x1 result back to global solution vector
        for (int i = 0; i < snode.front_width; i++) {
            int global_row = snode.row_structure[i];
            x[global_row] = x_local[i];
        }
    }
    
    void SparseGaussElimination::solve_supernode_forward_multiple(int snode_id, Eigen::MatrixXd& X) const {
        const Supernode& snode = supernodes_[snode_id];
        int nrhs = static_cast<int>(X.cols());
        
        // Extract relevant portion of solution matrix
        Eigen::MatrixXd X_local(snode.front_height, nrhs);
        for (int i = 0; i < snode.front_height; i++) {
            int global_row = snode.row_structure[i];
            for (int j = 0; j < nrhs; j++) {
                X_local(i, j) = X(global_row, j);
            }
        }
        
        // Forward substitution: L * Y = X_local, where L is unit lower triangular
        // Use DTRSM (Level-3 BLAS) - highly efficient for multiple RHS
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                   snode.front_height, nrhs, 1.0,
                   snode.frontal_matrix.data(), snode.front_lda,
                   X_local.data(), snode.front_height);
        
        // Update: Apply L21 * Y1 to remaining entries (Level-3 BLAS GEMM)
        if (snode.front_height > snode.front_width) {
            int remaining_rows = snode.front_height - snode.front_width;
            
            // Y2 = Y2 - L21 * Y1 (GEMM - Level-3 BLAS)
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       remaining_rows, nrhs, snode.front_width, -1.0,
                       snode.frontal_matrix.data() + snode.front_width, snode.front_lda,
                       X_local.data(), snode.front_height, 1.0,
                       X_local.data() + snode.front_width, snode.front_height);
        }
        
        // Copy result back to global solution matrix
        for (int i = 0; i < snode.front_height; i++) {
            int global_row = snode.row_structure[i];
            for (int j = 0; j < nrhs; j++) {
                X(global_row, j) = X_local(i, j);
            }
        }
    }
    
    void SparseGaussElimination::solve_supernode_backward_multiple(int snode_id, Eigen::MatrixXd& X) const {
        const Supernode& snode = supernodes_[snode_id];
        int nrhs = static_cast<int>(X.cols());
        
        // Extract relevant portion of solution matrix (pivot block only)
        Eigen::MatrixXd X1_local(snode.front_width, nrhs);
        for (int i = 0; i < snode.front_width; i++) {
            int global_row = snode.row_structure[i];
            for (int j = 0; j < nrhs; j++) {
                X1_local(i, j) = X(global_row, j);
            }
        }
        
        // Backward substitution: U11 * X1 = Y1, where U11 is upper triangular
        // Use DTRSM (Level-3 BLAS) - highly efficient for multiple RHS
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                   snode.front_width, nrhs, 1.0,
                   snode.frontal_matrix.data(), snode.front_lda,
                   X1_local.data(), snode.front_width);
        
        // Update remaining entries: X2 = X2 - U12^T * X1 (GEMM - Level-3 BLAS)
        if (snode.front_height > snode.front_width) {
            int remaining_rows = snode.front_height - snode.front_width;
            
            // Extract X2 (remaining part)
            Eigen::MatrixXd X2_local(remaining_rows, nrhs);
            for (int i = 0; i < remaining_rows; i++) {
                int global_row = snode.row_structure[snode.front_width + i];
                for (int j = 0; j < nrhs; j++) {
                    X2_local(i, j) = X(global_row, j);
                }
            }
            
            // X2 = X2 - U12^T * X1 (GEMM with transpose - Level-3 BLAS)
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                       remaining_rows, nrhs, snode.front_width, -1.0,
                       snode.frontal_matrix.data() + snode.front_width * snode.front_lda, snode.front_lda,
                       X1_local.data(), snode.front_width, 1.0,
                       X2_local.data(), remaining_rows);
            
            // Copy X2 back to global solution matrix
            for (int i = 0; i < remaining_rows; i++) {
                int global_row = snode.row_structure[snode.front_width + i];
                for (int j = 0; j < nrhs; j++) {
                    X(global_row, j) = X2_local(i, j);
                }
            }
        }
        
        // Copy X1 result back to global solution matrix
        for (int i = 0; i < snode.front_width; i++) {
            int global_row = snode.row_structure[i];
            for (int j = 0; j < nrhs; j++) {
                X(global_row, j) = X1_local(i, j);
            }
        }
    }
    
    // ============================================================================
    // TRUE SPARSE LU IMPLEMENTATION FOR LARGE MATRICES
    // ============================================================================
    
    bool SparseGaussElimination::solve_sparse_lu(const Eigen::SparseMatrix<double>& A, 
                                                  Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        // Convert to our CSC format
        CSCMatrix A_csc(A);
        
        // Factorize directly in CSC format
        int info = factorize_sparse_direct(A_csc);
        if (info != 0) {
            return false;  // Singular matrix
        }
        
        // Solve using CSC factors
        x = b;  // Copy RHS
        
        // Forward substitution: L * y = b
        forward_substitution(x);
        
        // Backward substitution: U * x = y  
        backward_substitution(x);
        
        return true;
    }
    
    int SparseGaussElimination::factorize_sparse_direct(const CSCMatrix& A) {
        // SIMPLIFIED SPARSE LU: Column-by-column factorization with fill-in tracking
        
        // Initialize L and U factors
        L_factor_.n_rows = A.n_rows;
        L_factor_.n_cols = A.n_cols;
        L_factor_.col_pointers.resize(A.n_cols + 1, 0);
        L_factor_.values.clear();
        L_factor_.row_indices.clear();
        
        U_factor_.n_rows = A.n_rows;
        U_factor_.n_cols = A.n_cols;
        U_factor_.col_pointers.resize(A.n_cols + 1, 0);
        U_factor_.values.clear();
        U_factor_.row_indices.clear();
        
        // Working matrix for factorization (will accumulate fill-in)
        std::vector<std::map<int, double>> working_cols(A.n_cols);
        
        // Initialize working matrix from A
        for (int j = 0; j < A.n_cols; j++) {
            for (int p = A.col_pointers[j]; p < A.col_pointers[j + 1]; p++) {
                int i = A.row_indices[p];
                double value = A.values[p];
                working_cols[j][i] = value;
            }
        }
        
        // Column-by-column LU factorization
        for (int k = 0; k < A.n_cols; k++) {
            // Find pivot in column k
            double pivot_val = 0.0;
            int pivot_row = k;
            
            for (auto& [i, val] : working_cols[k]) {
                if (i >= k && std::abs(val) > std::abs(pivot_val)) {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
            
            if (std::abs(pivot_val) < 1e-14) {
                return k + 1;  // Singular at column k
            }
            
            // Extract column k of L (below diagonal) and U (above and on diagonal)
            L_factor_.col_pointers[k] = static_cast<int>(L_factor_.values.size());
            U_factor_.col_pointers[k] = static_cast<int>(U_factor_.values.size());
            
            // Store L factor (unit lower triangular)
            L_factor_.values.push_back(1.0);  // Diagonal is always 1
            L_factor_.row_indices.push_back(k);
            
            for (auto& [i, val] : working_cols[k]) {
                if (i > k) {
                    // Below diagonal: L[i,k] = A[i,k] / A[k,k]
                    double L_ik = val / pivot_val;
                    L_factor_.values.push_back(L_ik);
                    L_factor_.row_indices.push_back(i);
                }
            }
            
            // Store U factor (upper triangular)
            for (auto& [i, val] : working_cols[k]) {
                if (i <= k) {
                    U_factor_.values.push_back(val);
                    U_factor_.row_indices.push_back(i);
                }
            }
            
            // Update remaining columns: A[i,j] -= L[i,k] * U[k,j] for i,j > k
            for (int j = k + 1; j < A.n_cols; j++) {
                if (working_cols[j].count(k)) {  // U[k,j] exists
                    double U_kj = working_cols[j][k];
                    
                    // Apply updates: A[i,j] -= L[i,k] * U[k,j] for all i > k
                    for (auto& [i, L_ik] : working_cols[k]) {
                        if (i > k) {
                            double L_ik_val = L_ik / pivot_val;
                            working_cols[j][i] -= L_ik_val * U_kj;
                        }
                    }
                }
            }
        }
        
        // Finalize column pointers
        L_factor_.col_pointers[A.n_cols] = static_cast<int>(L_factor_.values.size());
        U_factor_.col_pointers[A.n_cols] = static_cast<int>(U_factor_.values.size());
        L_factor_.nnz = static_cast<int>(L_factor_.values.size());
        U_factor_.nnz = static_cast<int>(U_factor_.values.size());
        
        factor_nnz_ = L_factor_.nnz + U_factor_.nnz;
        
        return 0;  // Success
    }
    
    void SparseGaussElimination::extract_lu_factors_direct(const CSCMatrix& A_factored) {
        // This method is now integrated into factorize_sparse_direct
        // Left as placeholder for future optimizations
    }
}