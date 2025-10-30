#include "utils/matrix_helper.hpp"

namespace utils {

    // Constructor and destructor implementation
    MatrixHelper::MatrixHelper() {}
    
    MatrixHelper::~MatrixHelper() {}

    template<typename MatrixType>
    MatrixInfo MatrixHelper::check_matrix_properties(const MatrixType& matrix) {
        MatrixInfo info;

        info.num_rows = matrix.rows();
        info.num_cols = matrix.cols();

        // Check if it's sparse
        if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>> ||
            std::is_same_v<MatrixType, Eigen::SparseMatrix<float>> ||
            std::is_same_v<MatrixType, Eigen::SparseMatrix<int>>) {
            info.is_sparse = true;
        } else {
            info.is_sparse = false;
        }

        // Check symmetry
        if (matrix.rows() <= 500) {
            info.is_symmetric = check_symmetry_full(matrix, 1e-12);
        } else {
            info.is_symmetric = check_symmetry_sampled(matrix, 1e-12);
        }

        // Check trace
        info.trace_value = compute_trace(matrix);

        // Check positive definiteness
        return info;
    }

    template<typename MatrixType>
    bool MatrixHelper::check_symmetry_full(const MatrixType& matrix, double tolerance) {
        int n = matrix.rows();
    
        if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>> ||
            std::is_same_v<MatrixType, Eigen::SparseMatrix<float>> ||
            std::is_same_v<MatrixType, Eigen::SparseMatrix<int>>) {
            // Sparse matrix handling
            for (int k = 0; k < matrix.outerSize(); ++k) {
                for (typename MatrixType::InnerIterator it(matrix, k); it; ++it) {
                    int i = it.row();
                    int j = it.col();
                    double A_ij = it.value();
                    double A_ji = matrix.coeff(j, i);
                    
                    double error = std::abs(A_ij - A_ji);
                    
                    if (error > tolerance) {
                        return false;
                    }
                }
            }
        } else {
            // Dense matrix handling
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {  // Only check upper triangle
                    double A_ij = matrix.coeff(i, j);
                    double A_ji = matrix.coeff(j, i);
                    
                    double error = std::abs(A_ij - A_ji);
                    
                    if (error > tolerance) {
                        return false;
                    }
                }
            }
        }
        
        return true;
    }

    template<typename MatrixType>
    bool MatrixHelper::check_symmetry_sampled(const MatrixType& matrix, double tolerance) {
        int n = matrix.rows();
        int num_samples = std::min(10000, n * n / 100);  // Sample ~1% of entries
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n-1);
        
        for (int sample = 0; sample < num_samples; ++sample) {
            int i = dis(gen);
            int j = dis(gen);
            
            double A_ij = matrix.coeff(i, j);
            double A_ji = matrix.coeff(j, i);
            
            double error = std::abs(A_ij - A_ji);
            
            if (error > tolerance) {
                return false;
            }
        }
        
        return true;
    }

    // ============================================================================
    // HELPERS
    // ============================================================================ 
    template<typename MatrixType>
    void MatrixHelper::display_matrix(
        const MatrixType& matrix,
        const std::string& name,
        int precision,
        int max_rows,
        int max_cols,
        bool show_non_zero_elements
    ) {
        if (!name.empty()) {
            std::cout << "\n=== " << name << " ===" << std::endl;
        }
        
        int rows = matrix.rows();
        int cols = matrix.cols();
        
        // Set display limits
        int display_rows = (max_rows > 0) ? std::min(max_rows, rows) : rows;
        int display_cols = (max_cols > 0) ? std::min(max_cols, cols) : cols;
        
        // Set precision
        std::cout << std::fixed << std::setprecision(precision);
        
        // Matrix dimensions
        std::cout << "Dimensions: " << rows << " x " << cols;
        if (max_rows > 0 && rows > max_rows) std::cout << " (showing " << max_rows << " rows)";
        if (max_cols > 0 && cols > max_cols) std::cout << " (showing " << max_cols << " cols)";
        std::cout << std::endl;
        
        if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>> ||
                      std::is_same_v<MatrixType, Eigen::SparseMatrix<float>> ||
                      std::is_same_v<MatrixType, Eigen::SparseMatrix<int>>) {
            // Sparse matrix display - show full structure
            std::cout << "Sparse matrix with " << matrix.nonZeros() << " non-zero elements:" << std::endl;
            
            // Show full matrix structure with zeros
            for (int i = 0; i < display_rows; ++i) {
                for (int j = 0; j < display_cols; ++j) {
                    double val = matrix.coeff(i, j);
                    if (std::abs(val) < 1e-15) {
                        std::cout << std::setw(precision + 8) << "0";
                    } else {
                        std::cout << std::setw(precision + 8) << val;
                    }
                }
                if (cols > display_cols) std::cout << " ...";
                std::cout << std::endl;
            }
            if (rows > display_rows) {
                std::cout << "..." << std::endl;
            }
            
            // Also show non-zero elements list for debugging
            if (show_non_zero_elements) {
                std::cout << "\nNon-zero elements:" << std::endl;
                for (int k = 0; k < matrix.outerSize(); ++k) {
                    for (typename MatrixType::InnerIterator it(matrix, k); it; ++it) {
                        std::cout << "(" << it.row() << "," << it.col() << ") = " 
                                << std::setw(precision + 8) << it.value() << std::endl;
                    }
                }
            }
            
            
        } else {
            // Dense matrix display
            for (int i = 0; i < display_rows; ++i) {
                for (int j = 0; j < display_cols; ++j) {
                    std::cout << std::setw(precision + 8) << matrix.coeff(i, j);
                }
                if (cols > display_cols) std::cout << " ...";
                std::cout << std::endl;
            }
            if (rows > display_rows) {
                std::cout << "..." << std::endl;
            }
        }
        
        std::cout << std::endl;
    }

    template<typename MatrixType>
    double MatrixHelper::compute_trace(const MatrixType& matrix) {
        if constexpr (std::is_same_v<MatrixType, Eigen::SparseMatrix<double>> ||
                      std::is_same_v<MatrixType, Eigen::SparseMatrix<float>> ||
                      std::is_same_v<MatrixType, Eigen::SparseMatrix<int>>) {
            double trace = 0.0;
            for (int i = 0; i < matrix.rows(); ++i) {
                for (typename MatrixType::InnerIterator it(matrix, i); it; ++it) {
                    if (it.row() == it.col()) {
                        trace += it.value();
                    }
                }
            }
            return trace;
        } else {
            return matrix.trace();
        }
    }


    // ============================================================================
    // EXPLICIT TEMPLATE INSTANTIATIONS
    // ============================================================================ 

    // Explicit template instantiations for the types you actually use
    template MatrixInfo MatrixHelper::check_matrix_properties<Eigen::SparseMatrix<double>>(const Eigen::SparseMatrix<double>&);
    template MatrixInfo MatrixHelper::check_matrix_properties<Eigen::MatrixXd>(const Eigen::MatrixXd&);
    
    template bool MatrixHelper::check_symmetry_full<Eigen::SparseMatrix<double>>(const Eigen::SparseMatrix<double>&, double);
    template bool MatrixHelper::check_symmetry_full<Eigen::MatrixXd>(const Eigen::MatrixXd&, double);
    
    template bool MatrixHelper::check_symmetry_sampled<Eigen::SparseMatrix<double>>(const Eigen::SparseMatrix<double>&, double);
    template bool MatrixHelper::check_symmetry_sampled<Eigen::MatrixXd>(const Eigen::MatrixXd&, double);

    template void MatrixHelper::display_matrix<Eigen::SparseMatrix<double>>(const Eigen::SparseMatrix<double>&, const std::string&, int, int, int, bool);
    template void MatrixHelper::display_matrix<Eigen::MatrixXd>(const Eigen::MatrixXd&, const std::string&, int, int, int, bool);
}