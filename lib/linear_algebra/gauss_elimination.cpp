#include "linear_algebra/gauss_elimination.hpp"

namespace LinearAlgebra {
    // ============================================================================
    // CONSTRUCTORS AND INITIALIZATION
    // ============================================================================

    GaussElimination::GaussElimination(size_t n, size_t block_size)
            : matrix_size_(n)
            , block_size_(block_size == 0 ? determine_optimal_block_size() : block_size)
            , use_accelerate_(false)
            , pivot_indices_(n)
            , work_buffer_(n) {
            
            if (n == 0) {
                throw std::invalid_argument("Matrix size must be positive");
            }
            
            initialize();
        }

    void GaussElimination::initialize() {
        configure_blas_threading();
        
        #ifdef USE_ACCELERATE
        use_accelerate_ = true;
        #endif
        
        // Pre-allocate workspace
        work_buffer_.resize(matrix_size_);
        pivot_indices_.resize(matrix_size_);
    }

    size_t GaussElimination::determine_optimal_block_size() const {
        // ARM64-optimized block size based on cache hierarchy
        // L2 cache: ~4MB on Apple Silicon, target 3 blocks (A, B, C for GEMM)
        constexpr size_t L2_CACHE_SIZE = 4 * 1024 * 1024;
        constexpr size_t ELEMENT_SIZE = sizeof(double);
        
        size_t target_elements = L2_CACHE_SIZE / (3 * ELEMENT_SIZE);
        size_t candidate = static_cast<size_t>(std::sqrt(target_elements));
        
        // Round to multiple of 8 for SIMD alignment
        candidate = ((candidate + 7) / 8) * 8;
        
        // Clamp to reasonable range for ARM64
        return std::clamp(candidate, size_t(96), size_t(256));
    }
    
    void GaussElimination::configure_blas_threading() const {
        // // Optimal threading for ARM64 to avoid memory bandwidth saturation
        // int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        // int optimal_threads = std::min(hw_threads, 8);
        
        // #ifdef USE_ACCELERATE
        // // Configure Apple Accelerate using setenv (safer than putenv)
        // setenv("VECLIB_MAXIMUM_THREADS", std::to_string(optimal_threads).c_str(), 1);
        // #else
        // // Configure OpenBLAS using setenv
        // setenv("OPENBLAS_NUM_THREADS", std::to_string(optimal_threads).c_str(), 1);
        // #endif

        return;
    }

    // ============================================================================
    // MAIN SOLVER INTERFACE
    // ============================================================================

    bool GaussElimination::solve(Eigen::MatrixXd& A, Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        validate_dimensions(A.rows(), A.cols());
        
        if (A.rows() != b.size() || A.rows() != x.size()) {
            throw std::invalid_argument("Matrix and vector dimensions must match");
        }
        
        // Copy b to x for in-place solve
        x = b;
        
        // Factorize matrix
        int info = factorize(A.data(), A.outerStride());
        if (info != 0) {
            return false; // Singular matrix
        }
        
        // Solve using factorization
        return solve_factorized(A.data(), x.data(), b.data(), 1);
    }
    
    bool GaussElimination::solve(Eigen::MatrixXd& A, Eigen::MatrixXd& X, const Eigen::MatrixXd& B) {
        validate_dimensions(A.rows(), A.cols());
        
        if (A.rows() != B.rows() || B.rows() != X.rows() || B.cols() != X.cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        
        // Copy B to X for in-place solve
        X = B;
        
        // Factorize matrix
        int info = factorize(A.data(), A.outerStride());
        if (info != 0) {
            return false; // Singular matrix
        }
        
        // Solve using factorization
        return solve_factorized(A.data(), X.data(), B.data(), B.cols());
    }

    int GaussElimination::factorize(double* A, size_t lda) {
        if (!is_matrix_valid(A, lda)) {
            return -1;
        }
        
        int result = blocked_gepp(A, lda);
        
        return result;
    }

    bool GaussElimination::solve_factorized(const double* A, double* x, 
        const double* b, size_t nrhs) const {

        const size_t n = matrix_size_;

        // Copy b to x
        cblas_dcopy(static_cast<int>(n * nrhs), b, 1, x, 1);

        // Apply permutation and triangular solves
        apply_permutation(x, nrhs);
        bool success = triangular_solves(A, x, nrhs);

        return success;
        }

    // ============================================================================
    // CORE FACTORIZATION ALGORITHM
    // ============================================================================

    int GaussElimination::blocked_gepp(double* A, size_t lda) {
        const size_t n = matrix_size_;
        
        // Main blocked factorization loop
        for (size_t k = 0; k < n; k += block_size_) {
            size_t jb = std::min(block_size_, n - k);  // Current block width
            
            // STEP 1: Panel factorization (Level-2 BLAS, but small)
            int panel_info = panel_factorization(A, lda, k, jb);
            if (panel_info != 0) {
                return static_cast<int>(k + panel_info); // Singular at position
            }
            
            // STEP 2: Apply pivots to right blocks
            if (k + jb < n) {
                apply_pivots(A, lda, k + jb, n - k - jb, k, jb);
            }
            
            // STEP 3: Update trailing matrix (Level-3 BLAS - CRITICAL PATH)
            if (k + jb < n) {
                update_trailing_matrix(A, lda, k, jb);
            }
        }
        
        return 0; // Success
    }

    int GaussElimination::panel_factorization(double* A, size_t lda, size_t k, size_t jb) {
        const size_t n = matrix_size_;
        
        // Factor panel columns k:k+jb-1 using unblocked algorithm
        for (size_t j = k; j < k + jb; j++) {
            // Find pivot in column j (partial pivoting)
            size_t pivot_row = j;
            double max_val = std::abs(A[j * lda + j]);
            
            for (size_t i = j + 1; i < n; i++) {
                double val = std::abs(A[j * lda + i]);
                if (val > max_val) {
                    max_val = val;
                    pivot_row = i;
                }
            }
            
            pivot_indices_[j] = static_cast<int>(pivot_row + 1); // 1-based indexing
            
            // Check for singularity
            if (max_val == 0.0) {
                return static_cast<int>(j - k + 1); // Singular at relative position
            }
            
            // Apply row swap to entire matrix
            if (pivot_row != j) {
                cblas_dswap(static_cast<int>(n), A + j, static_cast<int>(lda), 
                           A + pivot_row, static_cast<int>(lda));
            }
            
            // Scale column below diagonal
            if (j < n - 1) {
                double alpha = 1.0 / A[j * lda + j];
                cblas_dscal(static_cast<int>(n - j - 1), alpha, A + j * lda + j + 1, 1);
            }
            
            // Rank-1 update of trailing submatrix within panel
            if (j < k + jb - 1 && j < n - 1) {
                cblas_dger(CblasColMajor, 
                          static_cast<int>(n - j - 1), 
                          static_cast<int>(k + jb - j - 1), 
                          -1.0,
                          A + j * lda + j + 1, 1,
                          A + (j + 1) * lda + j, static_cast<int>(lda),
                          A + (j + 1) * lda + j + 1, static_cast<int>(lda));
            }
        }
        
        return 0; // Success
    }

    void GaussElimination::apply_pivots(double* A, size_t lda, size_t col_start, size_t ncols, size_t k, size_t jb) const {
        #ifdef USE_ACCELERATE
            // Use optimized LAPACK DLASWP if available
            long n_cols = static_cast<long>(ncols);
            long lda_int = static_cast<long>(lda);
            long k1 = static_cast<long>(k + 1);          // 1-based
            long k2 = static_cast<long>(k + jb);         // 1-based
            long incx = 1;
            
            // Validate parameters before calling dlaswp_
            if (n_cols <= 0 || lda_int <= 0 || k1 <= 0 || k2 <= 0 || k2 < k1) {
                return;
            }
            
            // Check if we have enough pivot indices
            if (k + jb > pivot_indices_.size()) {
                return;
            }
            
            // Convert to long and ensure we have the full range
            std::vector<long> pivot_long(pivot_indices_.size());
            for (size_t i = 0; i < pivot_indices_.size(); ++i) {
                pivot_long[i] = static_cast<long>(pivot_indices_[i]);
            }
            
            dlaswp_(&n_cols, A + col_start * lda, &lda_int, &k1, &k2, 
                   pivot_long.data(), &incx);
        #else
            // Manual implementation with vectorized swaps
            for (size_t i = k; i < k + jb; i++) {
                int pivot_row = pivot_indices_[i] - 1; // Convert to 0-based
                if (pivot_row != static_cast<int>(i)) {
                    // Validate parameters
                    if (pivot_row < 0 || pivot_row >= static_cast<int>(matrix_size_) ||
                        static_cast<int>(i) >= static_cast<int>(matrix_size_)) {
                        continue;
                    }
                    
                    cblas_dswap(static_cast<int>(ncols), 
                               A + col_start * lda + i, static_cast<int>(lda),
                               A + col_start * lda + pivot_row, static_cast<int>(lda));
                }
            }
        #endif
    }

    void GaussElimination::update_trailing_matrix(double* A, size_t lda, size_t k, size_t jb) const {
        const size_t n = matrix_size_;
        
        if (k + jb >= n) {
            return; // No trailing matrix
        }
        
        size_t remaining = n - k - jb;
        
        // STEP 1: Triangular solve L₁₁ U₁₂ = A₁₂ (TRSM - Level-3 BLAS)
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                   static_cast<int>(jb), static_cast<int>(remaining), 1.0,
                   A + k * lda + k, static_cast<int>(lda),           // L₁₁
                   A + (k + jb) * lda + k, static_cast<int>(lda));   // A₁₂ → U₁₂
        
        // STEP 2: Matrix multiply A₂₂ ← A₂₂ - L₂₁ U₁₂ (GEMM - Level-3 BLAS)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                   static_cast<int>(remaining), static_cast<int>(remaining), 
                   static_cast<int>(jb), -1.0,
                   A + k * lda + k + jb, static_cast<int>(lda),     // L₂₁
                   A + (k + jb) * lda + k, static_cast<int>(lda),   // U₁₂
                   1.0,
                   A + (k + jb) * lda + k + jb, static_cast<int>(lda)); // A₂₂
    }

    void GaussElimination::apply_permutation(double* x, size_t nrhs) const {
        const size_t n = matrix_size_;
        
        // Apply row permutation P to right-hand side
        for (size_t k = 0; k < n; k++) {
            int pivot = pivot_indices_[k] - 1; // Convert to 0-based
            if (pivot != static_cast<int>(k)) {
                for (size_t j = 0; j < nrhs; j++) {
                    std::swap(x[j * n + k], x[j * n + pivot]);
                }
            }
        }
    }

    bool GaussElimination::triangular_solves(const double* A, double* x, size_t nrhs) const {
        const size_t n = matrix_size_;
        
        // Forward substitution: solve L * y = P * b
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                   static_cast<int>(n), static_cast<int>(nrhs), 1.0, 
                   A, static_cast<int>(n), x, static_cast<int>(n));
        
        // Backward substitution: solve U * x = y
        cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                   static_cast<int>(n), static_cast<int>(nrhs), 1.0,
                   A, static_cast<int>(n), x, static_cast<int>(n));
        
        return true;
    }

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================

    void GaussElimination::validate_dimensions(size_t rows, size_t cols) const {
        if (rows != matrix_size_ || cols != matrix_size_) {
            throw std::invalid_argument("Matrix dimensions must match solver size");
        }
        
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
    }

    bool GaussElimination::is_matrix_valid(const double* A, size_t lda) const {
        if (!A) {
            return false; // Null pointer
        }
        
        if (lda < matrix_size_) {
            return false; // Leading dimension too small
        }
        
        // Check for reasonable matrix size limits - be more conservative
        const size_t MAX_MATRIX_SIZE = 1000; // Reduced limit
        if (matrix_size_ > MAX_MATRIX_SIZE) {
            std::cerr << "Warning: Matrix size " << matrix_size_ 
                      << " exceeds recommended limit of " << MAX_MATRIX_SIZE << std::endl;
            return false;
        }
        
        // Estimate memory usage (in MB) - be more conservative
        size_t memory_needed = matrix_size_ * matrix_size_ * sizeof(double) / (1024 * 1024);
        if (memory_needed > 200) { // Reduced from 1000MB to 200MB
            std::cerr << "Warning: Matrix requires approximately " << memory_needed 
                      << " MB of memory" << std::endl;
            return false;
        }
        
        return true;
    }

}