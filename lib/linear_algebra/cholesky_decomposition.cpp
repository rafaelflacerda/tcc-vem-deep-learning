#include "linear_algebra/cholesky_decomposition.hpp"

namespace LinearAlgebra {
    // ============================================================================
    // CONSTRUCTORS AND INITIALIZATION
    // ============================================================================
    
    CholeskyDecomposition::CholeskyDecomposition(size_t n, size_t block_size)
        : matrix_size_(n)
        , block_size_(block_size == 0 ? determine_optimal_block_size() : block_size)
        , use_accelerate_(false)
        , work_buffer_(n)
        , factorize_time_(0.0)
        , solve_time_(0.0) {
        
        if (n == 0) {
            throw std::invalid_argument("Matrix size must be positive");
        }
        
        initialize();
    }
    
    void CholeskyDecomposition::initialize() {
        configure_blas_threading();
        
        #ifdef USE_ACCELERATE
        use_accelerate_ = true;
        #endif
        
        // Pre-allocate workspace
        work_buffer_.resize(matrix_size_);
    }
    
    size_t CholeskyDecomposition::determine_optimal_block_size() const {
        // ARM64-optimized block size for DSYRK performance
        // Similar to GEMM tuning but optimized for symmetric rank-k updates
        constexpr size_t L2_CACHE_SIZE = 4 * 1024 * 1024;
        constexpr size_t ELEMENT_SIZE = sizeof(double);
        
        // DSYRK accesses: A (m×k) + C (m×m), optimize for C fitting in cache
        size_t target_elements = L2_CACHE_SIZE / (2 * ELEMENT_SIZE);
        size_t candidate = static_cast<size_t>(std::sqrt(target_elements));
        
        // Round to multiple of 8 for SIMD alignment
        candidate = ((candidate + 7) / 8) * 8;
        
        // Cholesky often benefits from smaller blocks than LU
        return std::clamp(candidate, size_t(64), size_t(192));
    }
    
    void CholeskyDecomposition::configure_blas_threading() const {
        // Optimal threading for ARM64 to avoid memory bandwidth saturation
        int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
        int optimal_threads = std::min(hw_threads, 8);
        
        #ifdef USE_ACCELERATE
        // Configure Apple Accelerate using setenv (safer than putenv)
        setenv("VECLIB_MAXIMUM_THREADS", std::to_string(optimal_threads).c_str(), 1);
        #else
        // Configure OpenBLAS using setenv
        setenv("OPENBLAS_NUM_THREADS", std::to_string(optimal_threads).c_str(), 1);
        #endif
    }

    // ============================================================================
    // MAIN SOLVER INTERFACE
    // ============================================================================
    
     int CholeskyDecomposition::factorize(double* A, size_t lda) {
         if (!is_matrix_valid(A, lda)) {
             return -1;
         }

         auto start_time = std::chrono::high_resolution_clock::now();
         
         int result = blocked_cholesky(A, lda);
         
         auto end_time = std::chrono::high_resolution_clock::now();
         factorize_time_ = std::chrono::duration<double>(end_time - start_time).count();
         
         return result;
     }
    
    int CholeskyDecomposition::factorize(Eigen::MatrixXd& A) {
        validate_dimensions(A.rows(), A.cols());
        
        if (A.rows() != matrix_size_) {
            throw std::invalid_argument("Matrix size does not match solver size");
        }
        
        return factorize(A.data(), A.outerStride());
    }
    
    bool CholeskyDecomposition::solve(Eigen::MatrixXd& A, Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        validate_dimensions(A.rows(), A.cols());
        
        if (A.rows() != b.size() || A.rows() != x.size()) {
            throw std::invalid_argument("Matrix and vector dimensions must match");
        }
        
        // Copy b to x for in-place solve
        x = b;
        
        // Factorize matrix
        int info = factorize(A);
        if (info != 0) {
            return false; // Not positive definite
        }
        
        // Solve using factorization
        return solve_factorized(A.data(), x.data(), b.data(), 1);
    }
    
    bool CholeskyDecomposition::solve(Eigen::MatrixXd& A, Eigen::MatrixXd& X, const Eigen::MatrixXd& B) {
        validate_dimensions(A.rows(), A.cols());
        
        if (A.rows() != B.rows() || B.rows() != X.rows() || B.cols() != X.cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }
        
        // Copy B to X for in-place solve
        X = B;
        
        // Factorize matrix
        int info = factorize(A);
        if (info != 0) {
            return false; // Not positive definite
        }
        
        // Solve using factorization
        return solve_factorized(A.data(), X.data(), B.data(), B.cols());
    }
    
    bool CholeskyDecomposition::solve_system(const Eigen::MatrixXd& A, Eigen::VectorXd& x, const Eigen::VectorXd& b) {
        // Create working copy to avoid modifying input
        Eigen::MatrixXd A_work = A;
        return solve(A_work, x, b);
    }

    // ============================================================================
    // CORE FACTORIZATION ALGORITHM
    // ============================================================================
    
    int CholeskyDecomposition::blocked_cholesky(double* A, size_t lda) {
        const size_t n = matrix_size_;
        
        // Main blocked Cholesky loop
        for (size_t k = 0; k < n; k += block_size_) {
            size_t nb = std::min(block_size_, n - k);
            
            // STEP 1: Factor diagonal block A₁₁ = L₁₁·L₁₁^T (Level-2 BLAS)
            int diag_info = factor_diagonal_block(A, nb, lda, k);
            if (diag_info != 0) {
                return static_cast<int>(k + diag_info); // Not positive definite
            }
            
            // STEP 2: Solve triangular system L₂₁ = A₂₁·L₁₁^(-T) (Level-3 TRSM)
            if (k + nb < n) {
                solve_triangular_block(A, lda, k, nb);
            }
            
            // STEP 3: Symmetric rank-k update A₂₂ ← A₂₂ - L₂₁·L₂₁^T (Level-3 SYRK)
            if (k + nb < n) {
                symmetric_rank_update(A, lda, k, nb);
            }
        }
        
        return 0; // Success
    }
    
    int CholeskyDecomposition::factor_diagonal_block(double* A, size_t nb, size_t lda, size_t offset) {
        // Unblocked Cholesky factorization of diagonal block
        // A[offset:offset+nb-1, offset:offset+nb-1] = L₁₁·L₁₁^T
        
        for (size_t j = 0; j < nb; j++) {
            size_t global_j = offset + j;
            
            // Compute diagonal element: L[j,j] = sqrt(A[j,j] - sum(L[j,0:j-1]²))
            double diagonal_sum = 0.0;
            for (size_t k = 0; k < j; k++) {
                double L_jk = A[(offset + k) * lda + global_j];
                diagonal_sum += L_jk * L_jk;
            }
            
            double diagonal_val = A[global_j * lda + global_j] - diagonal_sum;
            if (diagonal_val <= 0.0) {
                return static_cast<int>(j + 1); // Not positive definite at position j
            }
            
            double L_jj = std::sqrt(diagonal_val);
            A[global_j * lda + global_j] = L_jj;
            
            // Compute column below diagonal: L[i,j] = (A[i,j] - dot_product) / L[j,j]
            for (size_t i = j + 1; i < nb; i++) {
                size_t global_i = offset + i;
                
                double dot_product = 0.0;
                for (size_t k = 0; k < j; k++) {
                    dot_product += A[(offset + k) * lda + global_i] * A[(offset + k) * lda + global_j];
                }
                
                A[global_j * lda + global_i] = (A[global_j * lda + global_i] - dot_product) / L_jj;
            }
        }
        
        return 0; // Success
    }
    
    void CholeskyDecomposition::solve_triangular_block(double* A, size_t lda, size_t k, size_t nb) {
        const size_t n = matrix_size_;
        
        if (k + nb >= n) return; // No trailing matrix
        
        size_t remaining = n - k - nb;
        
        // Solve L₁₁·X^T = A₂₁^T  =>  L₂₁ = X
        // This gives us L₂₁ such that L₁₁·L₂₁^T = A₂₁^T
        
        cblas_dtrsm(CblasColMajor,          // Column-major layout
                    CblasRight,              // L₁₁ is on the right
                    CblasLower,              // L₁₁ is lower triangular
                    CblasTrans,              // Use L₁₁^T in the solve
                    CblasNonUnit,            // L₁₁ has computed diagonal (not unit)
                    static_cast<int>(remaining), // Number of rows in A₂₁
                    static_cast<int>(nb),    // Number of columns in A₂₁  
                    1.0,                     // Alpha = 1.0
                    A + k * lda + k,         // L₁₁ (diagonal block)
                    static_cast<int>(lda),   // Leading dimension of L₁₁
                    A + k * lda + k + nb,    // A₂₁ (will become L₂₁)
                    static_cast<int>(lda));  // Leading dimension of A₂₁
    }
    
    void CholeskyDecomposition::symmetric_rank_update(double* A, size_t lda, size_t k, size_t nb) {
        const size_t n = matrix_size_;
        
        if (k + nb >= n) return; // No trailing matrix
        
        size_t remaining = n - k - nb;
        
        // A₂₂ ← A₂₂ - L₂₁·L₂₁^T
        // This is the CRITICAL performance path for Cholesky! (~95% of flops)
        
        cblas_dsyrk(CblasColMajor,          // Column-major layout
                    CblasLower,              // Update lower triangle only (SPD)
                    CblasNoTrans,            // L₂₁·L₂₁^T (no transpose on first operand)
                    static_cast<int>(remaining), // Dimension of A₂₂
                    static_cast<int>(nb),    // Inner dimension (rank of update)
                    -1.0,                    // Alpha = -1.0 (subtract the update)
                    A + k * lda + k + nb,    // L₂₁ matrix
                    static_cast<int>(lda),   // Leading dimension of L₂₁
                    1.0,                     // Beta = 1.0 (add to existing A₂₂)
                    A + (k + nb) * lda + k + nb, // A₂₂ (trailing block)
                    static_cast<int>(lda));  // Leading dimension of A₂₂
    }

    // ============================================================================
    // SOLVE USING FACTORIZATION
    // ============================================================================
    
     bool CholeskyDecomposition::solve_factorized(const double* A, double* x, const double* b, size_t nrhs) const {
         const size_t n = matrix_size_;
         
         auto start_time = std::chrono::high_resolution_clock::now();
         
         // Copy b to x
         cblas_dcopy(static_cast<int>(n * nrhs), b, 1, x, 1);
         
         // Perform triangular solves
         bool success = triangular_solve_cholesky(A, x, nrhs);
         
         auto end_time = std::chrono::high_resolution_clock::now();
         solve_time_ = std::chrono::duration<double>(end_time - start_time).count();
         
         return success;
     }
    
    bool CholeskyDecomposition::triangular_solve_cholesky(const double* L, double* x, size_t nrhs) const {
        const size_t n = matrix_size_;
        
        // Phase 1: Forward substitution L·y = b
        cblas_dtrsm(CblasColMajor,          // Column-major layout
                    CblasLeft,               // L is on the left: L·X = B
                    CblasLower,              // L is lower triangular
                    CblasNoTrans,            // Use L (not L^T)
                    CblasNonUnit,            // L has computed diagonal (not unit)
                    static_cast<int>(n),     // Matrix dimension
                    static_cast<int>(nrhs),  // Number of right-hand sides
                    1.0,                     // Alpha = 1.0
                    L,                       // Cholesky factor L
                    static_cast<int>(n),     // Leading dimension of L
                    x,                       // Right-hand side b / intermediate y
                    static_cast<int>(n));    // Leading dimension of x
        
        // Phase 2: Backward substitution L^T·x = y
        cblas_dtrsm(CblasColMajor,          // Column-major layout
                    CblasLeft,               // L^T is on the left: L^T·X = Y
                    CblasLower,              // L is lower (so L^T is upper)
                    CblasTrans,              // Use L^T (transpose of L)
                    CblasNonUnit,            // L^T has same diagonal as L (computed)
                    static_cast<int>(n),     // Matrix dimension
                    static_cast<int>(nrhs),  // Number of right-hand sides
                    1.0,                     // Alpha = 1.0
                    L,                       // Cholesky factor L (will be transposed)
                    static_cast<int>(n),     // Leading dimension of L
                    x,                       // Intermediate y / final solution x
                    static_cast<int>(n));    // Leading dimension of x
        
        return true;
    }
    
    // ============================================================================
    // PERFORMANCE AND UTILITIES
    // ============================================================================
    
    void CholeskyDecomposition::validate_dimensions(size_t rows, size_t cols) const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square");
        }
        if (rows != matrix_size_) {
            throw std::invalid_argument("Matrix size does not match solver size");
        }
    }
    
    bool CholeskyDecomposition::is_matrix_valid(const double* A, size_t lda) const {
        if (A == nullptr) return false;
        if (lda < matrix_size_) return false;
        
        // Check for NaN or Inf values (basic validation)
        for (size_t j = 0; j < matrix_size_; j++) {
            for (size_t i = 0; i < matrix_size_; i++) {
                double val = A[j * lda + i];
                if (!std::isfinite(val)) {
                    return false;
                }
            }
        }
        
         return true;
     }
     
     // ============================================================================
     // PERFORMANCE DIAGNOSTICS
     // ============================================================================
     
     double CholeskyDecomposition::getFactorizationGFLOPs() const {
         if (factorize_time_ <= 0.0) return 0.0;
         
         // Cholesky factorization complexity: n³/3 + O(n²) operations
         const double n = static_cast<double>(matrix_size_);
         const double flops = (n * n * n) / 3.0 + (n * n) / 2.0;
         
         return flops / (factorize_time_ * 1e9);
     }
 }