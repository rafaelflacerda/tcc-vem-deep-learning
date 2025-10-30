/**
 * @file test_linear_algebra.hpp
 * @brief Comprehensive test for linear algebra solvers
 * @author Paulo Akira
 * @date 2025
 */

#ifndef TEST_LINEAR_ALGEBRA_HPP
#define TEST_LINEAR_ALGEBRA_HPP

#include <iostream>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <random>
#include <iomanip>

#include "utils/scope_timer.hpp"
#include "linear_algebra/gauss_elimination.hpp"
#include "linear_algebra/sparse_gauss_elimination.hpp"
#include "linear_algebra/cholesky_decomposition.hpp"
#include "linear_algebra/sparse_cholesky_decomposition.hpp"

namespace TestLinearAlgebra {
    // Dense matrix tests
    void test_basic_solve();
    void test_performance_comparison();
    void test_performance_comparison_statistical(int num_runs = 50);
    void test_memory_usage();
    void test_multiple_solves();
    
    // Sparse matrix tests
    void test_sparse_basic_solve();
    void test_sparse_performance_comparison();
    void test_sparse_performance_comparison_statistical(int num_runs = 50);
    void test_sparse_memory_usage();
    void test_sparse_multiple_solves();
    
    // Accelerate vs non-Accelerate comparison
    void test_accelerate_comparison();
    
    // Cholesky decomposition tests
    void test_cholesky_performance_comparison_statistical(int num_runs = 50);
    
    // Sparse Cholesky decomposition tests
    void test_sparse_cholesky_performance_comparison_statistical(int num_runs = 50);
}
 
 #endif