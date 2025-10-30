#include "linear_algebra/power_method.hpp"

namespace LinearAlgebra {

    // ============================================================================
    // CONSTRUCTORS AND INITIALIZATION
    // ============================================================================

    PowerMethod::PowerMethod(size_t n, size_t block_size)
        : matrix_size_(n)
        , block_size_(block_size == 0 ? determine_optimal_block_size() : block_size)
        , use_accelerate_(false)
        , rng_(std::random_device{}()) {
        
        if (n == 0) {
            throw std::invalid_argument("Matrix size must be positive");
        }
        
        initialize();
    }

    void PowerMethod::initialize() {
        configure_blas_threading();
        configure_accelerate();
        initialize_workspace();
    }

    void PowerMethod::initialize_workspace() {
        utils::ScopeTimer timer("Power Method Workspace Initialization");
        
        // Pre-allocate all working memory
        work_vector_.resize(matrix_size_);
        prev_vector_.resize(matrix_size_);
        temp_vector_.resize(matrix_size_);
        residual_vector_.resize(matrix_size_);
        
        // Block method workspace
        block_vectors_.resize(matrix_size_ * block_size_);
        block_result_.resize(matrix_size_ * block_size_);
        tau_.resize(block_size_);
        
        // QR workspace size estimation
        size_t qr_work_size = std::max(size_t(1), block_size_ * 64);
        qr_work_.resize(qr_work_size);
        
        // Convergence history
        eigenvalue_history_.reserve(1000);
    }

    size_t PowerMethod::determine_optimal_block_size() const {
        // ARM64-optimized block size based on cache hierarchy
        constexpr size_t L2_CACHE_SIZE = 4 * 1024 * 1024;
        constexpr size_t ELEMENT_SIZE = sizeof(double);
        
        // For Power Method: target matrix + 2 vectors in L2 cache
        size_t matrix_elements = matrix_size_ * matrix_size_;
        size_t vector_elements = 2 * matrix_size_;
        size_t total_elements = matrix_elements + vector_elements;
        size_t target_elements = L2_CACHE_SIZE / ELEMENT_SIZE;
        
        size_t candidate;
        if (total_elements > target_elements) {
            // Matrix doesn't fit in cache, use small blocks
            candidate = std::min(size_t(8), matrix_size_ / 8);
        } else {
            // Matrix fits, can use larger blocks
            candidate = std::min(size_t(32), matrix_size_ / 4);
        }
        
        // Round to multiple of 8 for SIMD alignment
        candidate = std::max(size_t(4), ((candidate + 7) / 8) * 8);
        
        return std::clamp(candidate, size_t(4), size_t(64));
    }

    void PowerMethod::configure_accelerate() {
        #ifdef USE_ACCELERATE
        use_accelerate_ = true;
        #endif
    }

    void PowerMethod::configure_blas_threading() const {
        // Follow the same pattern as GaussElimination
        // Currently commented out to match your implementation
        return;
    }

    // ============================================================================
    // MAIN SOLVER INTERFACES
    // ============================================================================

    PowerMethod::Result PowerMethod::solve(const double* A, size_t lda) {
        Config default_config;  // Uses default member initializers
        return solve(A, lda, default_config);
    }

    PowerMethod::Result PowerMethod::solve(const double* A, size_t lda, const Config& config) {
        utils::ScopeTimer timer("Power Method Solve");
        
        validate_matrix(A, lda);
        validate_config(config);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        Result result;
        
        if (config.shift != 0.0) {
            result = solve_shifted_method(A, lda, config);
        } else if (should_use_block_method(config)) {
            result = solve_block_method(A, lda, config);
        } else {
            result = solve_standard_method(A, lda, config);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.computation_time = duration.count() / 1000.0; // Convert to milliseconds
        
        if (config.verbose) {
            log_final_result(result, config);
        }
        
        return result;
    }

    PowerMethod::Result PowerMethod::solve_generalized(const double* A, const double* B, size_t lda) {
        Config default_config;
        return solve_generalized(A, B, lda, default_config);
    }

    PowerMethod::Result PowerMethod::solve_generalized(const double* A, const double* B, 
                                                      size_t lda, const Config& config) {
        utils::ScopeTimer timer("Power Method Generalized Solve");
        
        // For generalized eigenvalue problem Av = λBv
        // We solve B^(-1)Av = λv using inverse iteration approach
        // This is a simplified implementation - production code would use Cholesky factorization
        
        validate_matrix(A, lda);
        validate_matrix(B, lda);
        validate_config(config);
        
        // For now, convert to standard eigenvalue problem
        // A more sophisticated implementation would use LAPACK's generalized eigenvalue routines
        throw std::runtime_error("Generalized eigenvalue problems not yet implemented");
    }

    PowerMethod::Result PowerMethod::solve(const Eigen::MatrixXd& A) {
        Config default_config;
        return solve(A, default_config);
    }

    PowerMethod::Result PowerMethod::solve(const Eigen::MatrixXd& A, const Config& config) {
        utils::ScopeTimer timer("Power Method Solve (Eigen Matrix)");
        
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("Matrix must be square");
        }
        
        if (static_cast<size_t>(A.rows()) != matrix_size_) {
            throw std::invalid_argument("Matrix size must match solver configuration");
        }
        
        return solve(A.data(), A.outerStride(), config);
    }

    PowerMethod::Result PowerMethod::solve_generalized(const Eigen::MatrixXd& A, 
                                                      const Eigen::MatrixXd& B) {
        Config default_config;
        return solve_generalized(A, B, default_config);
    }

    PowerMethod::Result PowerMethod::solve_generalized(const Eigen::MatrixXd& A, 
                                                      const Eigen::MatrixXd& B, 
                                                      const Config& config) {
        if (A.rows() != A.cols() || B.rows() != B.cols()) {
            throw std::invalid_argument("Matrices must be square");
        }
        
        if (A.rows() != B.rows()) {
            throw std::invalid_argument("Matrices must have same dimensions");
        }
        
        return solve_generalized(A.data(), B.data(), A.outerStride(), config);
    }

    PowerMethod::Result PowerMethod::solve_vem_eigenvalue(const Eigen::SparseMatrix<double>& A_h,
                                                         const Eigen::SparseMatrix<double>& M_h) {
        Config default_config;
        return solve_vem_eigenvalue(A_h, M_h, default_config);
    }

    PowerMethod::Result PowerMethod::solve_vem_eigenvalue(const Eigen::SparseMatrix<double>& A_h,
                                                         const Eigen::SparseMatrix<double>& M_h,
                                                         const Config& config) {
        utils::ScopeTimer timer("Power Method VEM Eigenvalue Solve");
        
        // Convert sparse to dense for Power Method
        Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_h);
        Eigen::MatrixXd M_dense = Eigen::MatrixXd(M_h);
        
        return solve_generalized(A_dense, M_dense, config);
    }

    // ============================================================================
    // CORE ALGORITHM IMPLEMENTATIONS
    // ============================================================================

    PowerMethod::Result PowerMethod::solve_standard_method(const double* A, size_t lda, 
                                                           const Config& config) const {
        utils::ScopeTimer timer("Power Method Standard Algorithm");
        
        // Initialize random starting vector
        initialize_random_vector(work_vector_.data());
        normalize_vector(work_vector_.data());
        
        double prev_eigenvalue = 0.0;
        double current_eigenvalue = 0.0;
        eigenvalue_history_.clear();
        
        int iteration;
        for (iteration = 0; iteration < config.max_iterations; ++iteration) {
            // Copy current vector to previous
            vector_copy(work_vector_.data(), prev_vector_.data());
            
            // Matrix-vector multiplication: y = A * x
            matrix_vector_multiply(A, lda, prev_vector_.data(), work_vector_.data());
            
            // Compute Rayleigh quotient: λ = x^T * A * x / x^T * x
            if (config.use_rayleigh_quotient) {
                current_eigenvalue = vector_dot_product(prev_vector_.data(), work_vector_.data());
            } else {
                // Simple eigenvalue estimate
                double norm_Ax = vector_norm(work_vector_.data());
                current_eigenvalue = norm_Ax;
            }
            
            // Normalize the result vector
            normalize_vector(work_vector_.data());
            
            // Compute residual ||Ax - λx||
            double residual = compute_residual(A, lda, work_vector_.data(), current_eigenvalue);
            
            // Log iteration if verbose
            if (config.verbose) {
                log_iteration(iteration, current_eigenvalue, residual, config);
            }
            
            // Check convergence
            if (check_convergence(current_eigenvalue, prev_eigenvalue, residual, config, iteration)) {
                return create_result(current_eigenvalue, work_vector_.data(), iteration + 1, 
                                   true, 0.0, residual, config);
            }
            
            prev_eigenvalue = current_eigenvalue;
            update_convergence_history(current_eigenvalue);
        }
        
        // Did not converge
        double final_residual = compute_residual(A, lda, work_vector_.data(), current_eigenvalue);
        return create_result(current_eigenvalue, work_vector_.data(), iteration, 
                           false, 0.0, final_residual, config);
    }

    PowerMethod::Result PowerMethod::solve_block_method(const double* A, size_t lda, 
                                                       const Config& config) const {
        utils::ScopeTimer timer("Power Method Block Algorithm");
        
        size_t block_size = std::min(config.block_size, matrix_size_ / 2);
        
        // Initialize random block of vectors
        initialize_random_block(block_vectors_.data(), block_size);
        
        double prev_eigenvalue = 0.0;
        double current_eigenvalue = 0.0;
        eigenvalue_history_.clear();
        
        int iteration;
        for (iteration = 0; iteration < config.max_iterations; ++iteration) {
            // Matrix-matrix multiplication: Y = A * X
            matrix_matrix_multiply(A, lda, block_vectors_.data(), block_size, block_result_.data());
            
            // QR factorization: Y = Q * R
            orthogonalize_block(block_result_.data(), matrix_size_, block_size);
            
            // Extract dominant eigenvalue (largest diagonal element of R)
            current_eigenvalue = std::abs(block_result_[0]); // First diagonal element after QR
            
            // Update block vectors: X = Q
            vector_copy(block_result_.data(), block_vectors_.data());
            
            // Compute residual for the first vector
            double residual = compute_residual(A, lda, block_vectors_.data(), current_eigenvalue);
            
            // Log iteration if verbose
            if (config.verbose) {
                log_iteration(iteration, current_eigenvalue, residual, config);
            }
            
            // Check convergence
            if (check_convergence(current_eigenvalue, prev_eigenvalue, residual, config, iteration)) {
                return create_result(current_eigenvalue, block_vectors_.data(), iteration + 1, 
                                   true, 0.0, residual, config);
            }
            
            prev_eigenvalue = current_eigenvalue;
            update_convergence_history(current_eigenvalue);
        }
        
        // Did not converge
        double final_residual = compute_residual(A, lda, block_vectors_.data(), current_eigenvalue);
        return create_result(current_eigenvalue, block_vectors_.data(), iteration, 
                           false, 0.0, final_residual, config);
    }

    PowerMethod::Result PowerMethod::solve_shifted_method(const double* A, size_t lda, 
                                                         const Config& config) const {
        utils::ScopeTimer timer("Power Method Shifted Algorithm");
        
        // Shifted Power Method: (A - σI)v = λv
        // Creates a copy of the matrix with shift applied to diagonal
        std::vector<double> shifted_matrix(matrix_size_ * matrix_size_);
        
        // Copy matrix and apply shift
        for (size_t j = 0; j < matrix_size_; ++j) {
            for (size_t i = 0; i < matrix_size_; ++i) {
                shifted_matrix[j * matrix_size_ + i] = A[j * lda + i];
            }
            // Apply shift to diagonal
            shifted_matrix[j * matrix_size_ + j] -= config.shift;
        }
        
        // Solve with shifted matrix
        Config shifted_config = config;
        shifted_config.shift = 0.0; // Reset shift to avoid recursion
        
        Result result = solve_standard_method(shifted_matrix.data(), matrix_size_, shifted_config);
        
        // Adjust eigenvalue back
        if (result.converged) {
            result.eigenvalue += config.shift;
        }
        
        return result;
    }

    // ============================================================================
    // BLAS/LAPACK OPERATIONS
    // ============================================================================

    void PowerMethod::matrix_vector_multiply(const double* A, size_t lda, 
                                            const double* x, double* y) const {
        #ifdef USE_ACCELERATE
            cblas_dgemv(CblasColMajor, CblasNoTrans,
                       static_cast<int>(matrix_size_), static_cast<int>(matrix_size_),
                       1.0, A, static_cast<int>(lda), x, 1, 0.0, y, 1);
        #else
            // Fallback implementation
            for (size_t i = 0; i < matrix_size_; ++i) {
                y[i] = 0.0;
                for (size_t j = 0; j < matrix_size_; ++j) {
                    y[i] += A[j * lda + i] * x[j];
                }
            }
        #endif
    }

    void PowerMethod::matrix_matrix_multiply(const double* A, size_t lda, 
                                           const double* X, size_t nx, double* Y) const {
        #ifdef USE_ACCELERATE
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       static_cast<int>(matrix_size_), static_cast<int>(nx),
                       static_cast<int>(matrix_size_), 1.0,
                       A, static_cast<int>(lda),
                       X, static_cast<int>(matrix_size_),
                       0.0, Y, static_cast<int>(matrix_size_));
        #else
            // Fallback implementation
            for (size_t k = 0; k < nx; ++k) {
                matrix_vector_multiply(A, lda, X + k * matrix_size_, Y + k * matrix_size_);
            }
        #endif
    }

    double PowerMethod::vector_dot_product(const double* x, const double* y) const {
        #ifdef USE_ACCELERATE
            return cblas_ddot(static_cast<int>(matrix_size_), x, 1, y, 1);
        #else
            double result = 0.0;
            for (size_t i = 0; i < matrix_size_; ++i) {
                result += x[i] * y[i];
            }
            return result;
        #endif
    }

    double PowerMethod::vector_norm(const double* x) const {
        #ifdef USE_ACCELERATE
            return cblas_dnrm2(static_cast<int>(matrix_size_), x, 1);
        #else
            return std::sqrt(vector_dot_product(x, x));
        #endif
    }

    void PowerMethod::vector_scale(double* x, double alpha) const {
        #ifdef USE_ACCELERATE
            cblas_dscal(static_cast<int>(matrix_size_), alpha, x, 1);
        #else
            for (size_t i = 0; i < matrix_size_; ++i) {
                x[i] *= alpha;
            }
        #endif
    }

    void PowerMethod::vector_copy(const double* src, double* dst) const {
        #ifdef USE_ACCELERATE
            cblas_dcopy(static_cast<int>(matrix_size_), src, 1, dst, 1);
        #else
            std::copy(src, src + matrix_size_, dst);
        #endif
    }

    void PowerMethod::vector_axpy(double alpha, const double* x, double* y) const {
        #ifdef USE_ACCELERATE
            cblas_daxpy(static_cast<int>(matrix_size_), alpha, x, 1, y, 1);
        #else
            for (size_t i = 0; i < matrix_size_; ++i) {
                y[i] += alpha * x[i];
            }
        #endif
    }

    int PowerMethod::qr_factorization(double* A, size_t lda, size_t ncols, double* tau) const {
        #ifdef USE_ACCELERATE
            long m = static_cast<long>(matrix_size_);
            long n = static_cast<long>(ncols);
            long lda_long = static_cast<long>(lda);
            long lwork = static_cast<long>(qr_work_.size());
            long info = 0;
            
            dgeqrf_(&m, &n, A, &lda_long, tau, 
                   const_cast<double*>(qr_work_.data()), &lwork, &info);
            
            return static_cast<int>(info);
        #else
            // Simplified Gram-Schmidt process
            for (size_t j = 0; j < ncols; ++j) {
                // Compute norm of column j
                double norm = 0.0;
                for (size_t i = 0; i < matrix_size_; ++i) {
                    norm += A[j * lda + i] * A[j * lda + i];
                }
                norm = std::sqrt(norm);
                
                if (norm < 1e-14) {
                    return -1; // Singular matrix
                }
                
                // Normalize column j
                for (size_t i = 0; i < matrix_size_; ++i) {
                    A[j * lda + i] /= norm;
                }
                
                // Orthogonalize subsequent columns
                for (size_t k = j + 1; k < ncols; ++k) {
                    double dot = 0.0;
                    for (size_t i = 0; i < matrix_size_; ++i) {
                        dot += A[j * lda + i] * A[k * lda + i];
                    }
                    
                    for (size_t i = 0; i < matrix_size_; ++i) {
                        A[k * lda + i] -= dot * A[j * lda + i];
                    }
                }
            }
            return 0;
        #endif
    }

    int PowerMethod::extract_q_matrix(double* A, size_t lda, size_t ncols, const double* tau) const {
        #ifdef USE_ACCELERATE
            long m = static_cast<long>(matrix_size_);
            long n = static_cast<long>(ncols);
            long k = static_cast<long>(ncols);
            long lda_long = static_cast<long>(lda);
            long lwork = static_cast<long>(qr_work_.size());
            long info = 0;
            
            dorgqr_(&m, &n, &k, A, &lda_long, tau, 
                   const_cast<double*>(qr_work_.data()), &lwork, &info);
            
            return static_cast<int>(info);
        #else
            // Q matrix is already in A after Gram-Schmidt
            return 0;
        #endif
    }

    // ============================================================================
    // UTILITY AND VALIDATION METHODS
    // ============================================================================

    void PowerMethod::validate_matrix(const double* A, size_t lda) const {
        if (!A) {
            throw std::invalid_argument("Matrix pointer cannot be null");
        }
        
        if (lda < matrix_size_) {
            throw std::invalid_argument("Leading dimension too small");
        }
        
        if (!is_matrix_valid(A, lda)) {
            throw std::invalid_argument("Matrix validation failed");
        }
    }

    void PowerMethod::validate_config(const Config& config) const {
        if (config.tolerance <= 0.0 || config.tolerance >= 1.0) {
            throw std::invalid_argument("Tolerance must be between 0 and 1");
        }
        
        if (config.max_iterations <= 0) {
            throw std::invalid_argument("Maximum iterations must be positive");
        }
        
        if (config.block_size == 0 || config.block_size > matrix_size_) {
            throw std::invalid_argument("Block size must be between 1 and matrix size");
        }
    }

    bool PowerMethod::is_matrix_valid(const double* A, size_t lda) const {
        if (!A) {
            return false;
        }
        
        if (lda < matrix_size_) {
            return false;
        }
        
        // Check for reasonable matrix size limits
        const size_t MAX_MATRIX_SIZE = 10000;
        if (matrix_size_ > MAX_MATRIX_SIZE) {
            std::cerr << "Warning: Matrix size " << matrix_size_ 
                      << " exceeds recommended limit of " << MAX_MATRIX_SIZE << std::endl;
            return false;
        }
        
        // Estimate memory usage (in MB)
        size_t memory_needed = matrix_size_ * matrix_size_ * sizeof(double) / (1024 * 1024);
        if (memory_needed > 1000) {
            std::cerr << "Warning: Matrix requires approximately " << memory_needed 
                      << " MB of memory" << std::endl;
        }
        
        return true;
    }

    void PowerMethod::initialize_random_vector(double* v) const {
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (size_t i = 0; i < matrix_size_; ++i) {
            v[i] = dist(rng_);
        }
    }

    void PowerMethod::initialize_random_block(double* V, size_t ncols) const {
        std::normal_distribution<double> dist(0.0, 1.0);
        
        for (size_t j = 0; j < ncols; ++j) {
            for (size_t i = 0; i < matrix_size_; ++i) {
                V[j * matrix_size_ + i] = dist(rng_);
            }
        }
        
        // Orthogonalize the block
        orthogonalize_block(V, matrix_size_, ncols);
    }

    void PowerMethod::normalize_vector(double* v) const {
        double norm = vector_norm(v);
        if (norm > 1e-14) {
            vector_scale(v, 1.0 / norm);
        }
    }

    void PowerMethod::orthogonalize_block(double* V, size_t lda, size_t ncols) const {
        // Use QR factorization to orthogonalize
        int info = qr_factorization(V, lda, ncols, tau_.data());
        if (info == 0) {
            extract_q_matrix(V, lda, ncols, tau_.data());
        }
    }

    bool PowerMethod::check_convergence(double current_eigenvalue, double previous_eigenvalue,
                                       double residual, const Config& config, int iteration) const {
        if (iteration == 0) {
            return false; // Need at least one iteration
        }
        
        // Check residual convergence
        if (residual < config.tolerance) {
            return true;
        }
        
        // Check relative eigenvalue change
        double relative_change = compute_relative_error(current_eigenvalue, previous_eigenvalue);
        if (relative_change < config.relative_tolerance) {
            return true;
        }
        
        return false;
    }

    double PowerMethod::compute_residual(const double* A, size_t lda, const double* v, 
                                        double eigenvalue) const {
        // Compute r = Av - λv
        matrix_vector_multiply(A, lda, v, residual_vector_.data());
        vector_axpy(-eigenvalue, v, residual_vector_.data());
        
        return vector_norm(residual_vector_.data());
    }

    double PowerMethod::compute_relative_error(double current, double previous) const {
        if (std::abs(previous) < 1e-14) {
            return std::abs(current);
        }
        return std::abs(current - previous) / std::abs(previous);
    }

    PowerMethod::Result PowerMethod::create_result(double eigenvalue, const double* eigenvector,
                                                  int iterations, bool converged, 
                                                  double computation_time, double residual,
                                                  const Config& config) const {
        Result result;
        result.eigenvalue = eigenvalue;
        result.eigenvector.resize(matrix_size_);
        vector_copy(eigenvector, result.eigenvector.data());
        
        if (config.normalize_eigenvector) {
            normalize_vector(result.eigenvector.data());
        }
        
        result.iterations = iterations;
        result.converged = converged;
        result.computation_time = computation_time;
        result.residual = residual;
        result.convergence_rate = estimate_convergence_rate();
        
        return result;
    }

    void PowerMethod::update_convergence_history(double eigenvalue) const {
        eigenvalue_history_.push_back(eigenvalue);
        
        // Keep only recent history to avoid memory growth
        if (eigenvalue_history_.size() > 100) {
            eigenvalue_history_.erase(eigenvalue_history_.begin());
        }
    }

    double PowerMethod::estimate_convergence_rate() const {
        if (eigenvalue_history_.size() < 3) {
            return 0.0;
        }
        
        // Estimate convergence rate from recent history
        size_t n = eigenvalue_history_.size();
        double e1 = std::abs(eigenvalue_history_[n-1] - eigenvalue_history_[n-2]);
        double e2 = std::abs(eigenvalue_history_[n-2] - eigenvalue_history_[n-3]);
        
        if (e2 < 1e-14) {
            return 0.0;
        }
        
        return e1 / e2;
    }

    void PowerMethod::prefetch_matrix_data(const double* A, size_t lda) const {
        #ifdef __aarch64__
            // ARM64-specific prefetching
            for (size_t i = 0; i < matrix_size_; i += 64/sizeof(double)) {
                __builtin_prefetch(A + i * lda, 0, 3);
            }
        #endif
    }

    void PowerMethod::optimize_memory_layout() const {
        // Touch memory pages to avoid page faults during computation
        std::fill(work_vector_.begin(), work_vector_.end(), 0.0);
        std::fill(prev_vector_.begin(), prev_vector_.end(), 0.0);
        std::fill(temp_vector_.begin(), temp_vector_.end(), 0.0);
    }

    bool PowerMethod::should_use_block_method(const Config& config) const {
        if (!config.use_block_method) {
            return false;
        }
        
        // Use block method for larger matrices where Level 3 BLAS is beneficial
        return matrix_size_ > 100 && config.block_size > 1;
    }

    void PowerMethod::log_iteration(int iteration, double eigenvalue, double residual,
                                   const Config& config) const {
        if (iteration % 10 == 0 || residual < config.tolerance * 10) {
            std::cout << "Iteration " << iteration 
                      << ": eigenvalue = " << std::scientific << eigenvalue
                      << ", residual = " << residual << std::endl;
        }
    }

    void PowerMethod::log_final_result(const Result& result, const Config& config) const {
        std::cout << "\n=== Power Method Results ===" << std::endl;
        std::cout << "Converged: " << (result.converged ? "Yes" : "No") << std::endl;
        std::cout << "Eigenvalue: " << std::scientific << result.eigenvalue << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Computation time: " << std::fixed << result.computation_time << " ms" << std::endl;
        std::cout << "Convergence rate: " << result.convergence_rate << std::endl;
        std::cout << "Using Accelerate: " << (use_accelerate_ ? "Yes" : "No") << std::endl;
        std::cout << "=========================" << std::endl;
    }

    double PowerMethod::get_memory_requirements() const {
        size_t total_elements = 0;
        
        // Matrix storage
        total_elements += matrix_size_ * matrix_size_;
        
        // Working vectors
        total_elements += 4 * matrix_size_; // work, prev, temp, residual
        
        // Block method workspace
        if (block_size_ > 1) {
            total_elements += 2 * matrix_size_ * block_size_; // block_vectors, block_result
            total_elements += block_size_; // tau
            total_elements += qr_work_.size(); // QR workspace
        }
        
        return (total_elements * sizeof(double)) / (1024.0 * 1024.0); // MB
    }

} // namespace LinearAlgebra