#include <iostream>
#include <iomanip>
#include <cmath>

#include <Eigen/Dense>

#include "solver/parabolic.hpp"
#include "utils/operations.hpp"
#include "utils/integration.hpp"
#include "tests/test_time_scheme.hpp"
#include "tests/test_linear_algebra.hpp"  

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

int main() {

    // Run focused L² projection debug tests
    // L2ProjectionDebug::test_single_element_l2_projection_k1();
    
    // Test k=1 convergence
    // VEMSimulationParabolic::run_time_scheme_tests_k1();
    
    // Test k=2 convergence  
    // VEMSimulationParabolic::run_time_scheme_tests();

    // Check matrix properties
    // VEMSimulationParabolic::check_mass_matrix_properties();

    // Test VEM with 4-vertex distorted mesh
    double h_e = 0.0244; 
    double dt = utils::operations::compute_recommended_timestep(h_e);
    std::cout << "Recommended timestep: " << dt << std::endl;
    VEMSimulationParabolic::run_single_json_mesh_test(
        "data/_voronoi/voronoi_mesh_h0p0625.json",
        "data/_output/parabolic/voronoi_mesh_h0p0625_results.json",
        // "data/_serendipity/serendipity_mesh_h0p2500.json",
        // "data/_output/parabolic/serendipity_mesh_h0p2500_rk3_results_debug.json",
        // "data/_distorted_mesh/distorted_mesh_h0p500.json",
        // "data/_output/parabolic/distorted_mesh_h0p500_results_debug.json",
        1,      // VEM order
        1.0,    // Final time
        5.9536000000000014e-05,   // Time step
        true    // Verbose output
    );


    // Test linear algebra
    // try {
    //     // TestLinearAlgebra::test_performance_comparison_statistical(50);  
    //     // TestLinearAlgebra::test_sparse_performance_comparison_statistical(50);
    //     // TestLinearAlgebra::test_cholesky_performance_comparison_statistical(50);
    //     TestLinearAlgebra::test_sparse_cholesky_performance_comparison_statistical(50);
    //     
    //     std::cout << "\n All performance tests passed!" << std::endl;
    //     return 0;
    // } catch (const std::exception& e) {
    //     std::cerr << "Test failed: " << e.what() << std::endl;
    //     return 1;
    // }
    
    std::cout << "\n🎉 All JSON mesh tests completed!" << std::endl;

    return 0;
}