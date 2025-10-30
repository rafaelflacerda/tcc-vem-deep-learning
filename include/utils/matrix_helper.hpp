#ifndef MATRIX_HELPER_HPP
#define MATRIX_HELPER_HPP


#include <random>
#include <cmath>
#include <string> 
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Sparse>

struct MatrixInfo {
    int num_rows;
    int num_cols;
    bool is_symmetric; 
    bool is_positive_definite;
    bool is_spd;
    bool is_sparse;
    double symmetry_error;
    double max_eigenvalue;
    double min_eigenvalue;
    double trace_value;
};

namespace utils {
    class MatrixHelper {
        public:

            // ============================================================================
            // CONSTRUCTOR & DESTRUCTOR
            // ============================================================================ 
            MatrixHelper();
            ~MatrixHelper();

            // ============================================================================
            // BASIC CHECKS
            // ============================================================================ 

            /**
             * @brief Check the properties of a matrix
             * @param matrix The matrix to check
             * @return A MatrixInfo struct containing the properties of the matrix
             */
            template<typename MatrixType>
            static MatrixInfo check_matrix_properties(const MatrixType& matrix);

            /**
             * @brief Check if a matrix is symmetric considering all non-zero elements. 
             * 
             * Use that only for smaller matrices (n < 500).
             * 
             * @param matrix The matrix to check
             * @param tolerance The tolerance for the symmetry check
             * @return True if the matrix is symmetric, false otherwise
             */
            template<typename MatrixType>
            static bool check_symmetry_full(const MatrixType& matrix, double tolerance);

            /**
             * @brief Check if a matrix is symmetric considering only the non-zero elements
             * by random sample (~1% of the non-zero elements).
             * 
             * @param matrix The matrix to check
             * @return True if the matrix is symmetric, false otherwise
             */
            template<typename MatrixType>
            static bool check_symmetry_sampled(const MatrixType& matrix, double tolerance);

            // ============================================================================
            // EIGENVALUES & EIGENVECTORS
            // ============================================================================ 
            template<typename MatrixType>
            static Eigen::VectorXd compute_eigenvalues(const MatrixType& matrix);

            
            // ============================================================================
            // HELPERS
            // ============================================================================ 

            /**
             * @brief Display matrix in formatted terminal output
             * @param matrix The matrix to display
             * @param name Optional name/label for the matrix
             * @param precision Number of decimal places to show
             * @param max_rows Maximum number of rows to display (0 = all)
             * @param max_cols Maximum number of columns to display (0 = all)
             */
            template<typename MatrixType>
            static void display_matrix(
                const MatrixType& matrix,
                const std::string& name = "",
                int precision = 6,
                int max_rows = 0,
                int max_cols = 0,
                bool show_non_zero_elements = false
            );

            /**
             * @brief Compute the trace of a matrix
             * Manually calculate for the sparse matrix case.
             * Use the trace() method for the dense matrix case.
             * @param matrix The matrix to compute the trace of
             * @return The trace of the matrix
             */
            template<typename MatrixType>
            static double compute_trace(const MatrixType& matrix);
            
    };
}

#endif