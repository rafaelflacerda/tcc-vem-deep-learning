/**
 * @file test_time_scheme.hpp
 * @brief Comprehensive test for VEM parabolic time integration schemes
 * @author Paulo Akira
 * @date 2024
 */

#ifndef TEST_TIME_SCHEME_HPP
#define TEST_TIME_SCHEME_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <chrono>
#include <array>
#include <filesystem>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <nlohmann/json.hpp>

#include "solver/parabolic.hpp"
#include "mesh/uniform.hpp"
#include "mesh/datasource.hpp"
#include "utils/errors.hpp"
#include "utils/boundary.hpp"
#include "utils/scope_timer.hpp"
#include "utils/time_scheme.hpp"
#include "utils/matrix_helper.hpp"
#include "models/enums.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Test configuration
struct TestConfig {
    int vem_order = 1;              // VEM polynomial order
    double final_time = 0.1;        // Final time T
    double time_step = 0.01;        // Time step size Δt
    bool verbose = true;            // Detailed output
    bool save_solution = false;     // Save solution to file
    std::string output_file = "vem_solution.dat";
    int mesh_type = 3;              // Mesh density: 3 for 3x3, 4 for 4x4, 5 for 5x5
    bool use_lumped_mass_matrix = false;
    utils::time_scheme::SchemeType scheme_type = utils::time_scheme::SchemeType::FORWARD_EULER;
    
    // Static factory method for default configuration
    static TestConfig default_config() {
        TestConfig config;
        config.vem_order = 1;
        config.final_time = 0.1;
        config.time_step = 0.01;
        config.verbose = true;
        config.save_solution = false;
        config.output_file = "vem_solution.dat";
        config.mesh_type = 3;
        config.use_lumped_mass_matrix = false;
        config.scheme_type = utils::time_scheme::SchemeType::FORWARD_EULER;
        return config;
    }
};

// Test results
struct TestResults {
    double l2_error = 0.0;
    double h1_error = 0.0;
    double max_error = 0.0;
    double convergence_rate = 0.0;
    int total_time_steps = 0;
    double total_solve_time = 0.0;
    bool test_passed = false;
};

namespace VEMSimulationParabolic {
    class TimeSchemeTest {
             
    private:
        TestConfig config_;
        TestResults results_;

    public:
        // External mesh support
        Eigen::MatrixXd external_nodes;
        Eigen::MatrixXi external_elements;
        bool use_external_mesh = false;

    public:
        TimeSchemeTest() : config_(TestConfig::default_config()) {}
        TimeSchemeTest(const TestConfig& config) : config_(config) {}

        // =========================================================================
        // MANUFACTURED SOLUTION FUNCTIONS
        // =========================================================================

        /**
         * @brief Exact solution: u(t,x,y) = e^t * sin(πx) * sin(πy)
         */
        static double exact_solution(double t, double x, double y) {
            return std::exp(t) * std::sin(M_PI * x) * std::sin(M_PI * y);
        }

        /**
         * @brief Gradient of the exact solution ∇u = (∂u/∂x, ∂u/∂y)
         */
        static Eigen::Vector2d exact_gradient(double t, double x, double y) {
            double factor = std::exp(t);
            double du_dx = M_PI * std::cos(M_PI * x) * std::sin(M_PI * y) * factor;
            double du_dy = M_PI * std::sin(M_PI * x) * std::cos(M_PI * y) * factor;
            return {du_dx, du_dy};
        }

        /**
         * @brief Initial condition: u₀(x,y) = sin(πx) * sin(πy)
         */
        static double initial_condition(const Eigen::Vector2d& point) {
            return std::sin(M_PI * point.x()) * std::sin(M_PI * point.y());
        }

        /**
         * @brief Source term: f(t,x,y) = e^t * (1 + 2π²) * sin(πx) * sin(πy)
         */
        static double source_function(const Eigen::Vector2d& point, double t) {
            double factor = 1.0 + 2.0 * M_PI * M_PI;
            return std::exp(t) * factor * std::sin(M_PI * point.x()) * std::sin(M_PI * point.y());
        }

        /**
         * @brief Homogeneous Dirichlet boundary condition
         */
        static double boundary_condition(const Eigen::Vector2d& point, double t) {
            return 0.0;  // Homogeneous Dirichlet
        }

        // ============================================================================
        // SIMULATION SETUP
        // ============================================================================

        void setup_mesh(Eigen::MatrixXd& nodes, Eigen::MatrixXi& elements) {
            if (use_external_mesh) {
                // Use externally provided mesh (e.g., from JSON)
                nodes = external_nodes;
                elements = external_elements;
                if (config_.verbose) {
                    std::cout << "Using external mesh: " << nodes.rows() << " nodes, " << elements.rows() << " elements" << std::endl;
                }
            } else {
                // Use built-in uniform mesh generation
                mesh::uniform mesh(config_.verbose);
                mesh.create_square_nxn_mesh(nodes, elements, config_.mesh_type);
            }
        }


        TestResults simulation_setup();

        // ============================================================================
        // TIME SCHEME TESTS
        // ============================================================================

        /**
         * @brief Test Backward Euler time integration scheme
         */
        TestResults test_backward_euler(
            solver::parabolic& vem_solver,
            const Eigen::VectorXd& U_initial,
            utils::boundary& boundary_handler,
            const std::function<double(const Eigen::Vector2d&, double)>& source_function,
            const std::function<double(const Eigen::Vector2d&, double)>& boundary_condition,
            double final_time,
            double time_step,
            bool verbose
        );

        /**
         * @brief Test Forward Euler time integration scheme
         */
        TestResults test_forward_euler(
            solver::parabolic& vem_solver,
            const Eigen::VectorXd& U_initial,
            utils::boundary& boundary_handler,
            const std::function<double(const Eigen::Vector2d&, double)>& source_function,
            const std::function<double(const Eigen::Vector2d&, double)>& boundary_condition,
            double final_time,
            double time_step,
            bool verbose
        );

        TestResults test_rk(
            solver::parabolic& vem_solver,
            const Eigen::VectorXd& U_initial,
            utils::boundary& boundary_handler,
            const std::function<double(const Eigen::Vector2d&, double)>& source_function,
            const std::function<double(const Eigen::Vector2d&, double)>& boundary_condition,
            double final_time,
            double time_step,
            bool verbose
        );

        // ============================================================================
        // ERROR COMPUTATION
        // ============================================================================

        /**
         * @brief Compute L² and H¹ errors at final time
         * 
         * @param vem_solver VEM solver
         * @param U_final Final solution
         * @param final_time Final time
         * @param exact_gradient Exact gradient function
         */
        void compute_final_time_errors(
            const solver::parabolic& vem_solver, 
            const Eigen::VectorXd& U_final, 
            double final_time,
            const std::function<Eigen::Vector2d(double, double, double)>& exact_gradient
        );


        // ============================================================================
        // OUTPUT AND ANALYSIS
        // ============================================================================

        /**
         * @brief Print comprehensive test results
         * 
         * @param stats Time scheme statistics
         */
        void print_test_results(const utils::time_scheme::Statistics& stats);

        /**
         * @brief Save solution data to file for visualization
         * 
         * @param vem_solver VEM solver
         * @param U_final Final solution
         * @param time_points Time points
         * @param solution_norms Solution norms
         */
        void save_solution_to_file(
            const solver::parabolic& vem_solver,
            const Eigen::VectorXd& U_final,
            const std::vector<double>& time_points,
            const std::vector<double>& solution_norms
        );

        // ============================================================================
        // CONVERGENCE STUDY
        // ============================================================================

        /**
         * @brief Run convergence study with different time steps
         */
        void convergence_study() {
            std::cout << "\n=== Time Step Convergence Study ===" << std::endl;
            
            std::vector<double> time_steps = {0.05, 0.025, 0.0125, 0.00625};
            std::vector<double> errors;
            
            std::cout << std::setw(12) << "Time Step" 
                    << std::setw(15) << "L² Error" 
                    << std::setw(15) << "Conv. Rate" << std::endl;
            std::cout << std::string(42, '-') << std::endl;
            
            for (size_t i = 0; i < time_steps.size(); ++i) {
                TestConfig study_config = config_;
                study_config.time_step = time_steps[i];
                study_config.verbose = false;  // Reduce output during study
                
                TimeSchemeTest test(study_config);
                TestResults result = test.simulation_setup();
                errors.push_back(result.l2_error);
                
                double conv_rate = 0.0;
                if (i > 0 && errors[i-1] > 1e-12 && errors[i] > 1e-12) {
                    conv_rate = std::log(errors[i-1] / errors[i]) / std::log(time_steps[i-1] / time_steps[i]);
                }
                
                std::cout << std::scientific << std::setprecision(3)
                        << std::setw(12) << time_steps[i]
                        << std::setw(15) << errors[i];
                
                if (i > 0) {
                    std::cout << std::setw(15) << conv_rate;
                } else {
                    std::cout << std::setw(15) << "---";
                }
                std::cout << std::endl;
            }
            
            std::cout << "\nExpected convergence rate for Backward Euler: O(Δt¹) ≈ 1.0" << std::endl;
        }

        // ============================================================================
        // PUBLIC TEST INTERFACE
        // ============================================================================

        /**
         * @brief Run all time scheme tests
         */
        void run_all_tests();

        /**
         * @brief Calculate convergence rates from mesh sizes and errors
         */
        struct ConvergenceData {
            std::vector<double> mesh_sizes;     // h values
            std::vector<double> l2_errors;     // L² errors
            std::vector<double> h1_errors;     // H¹ errors  
            std::vector<double> l2_rates;      // L² convergence rates
            std::vector<double> h1_rates;      // H¹ convergence rates
            double avg_l2_rate = 0.0;          // Average L² rate
            double avg_h1_rate = 0.0;          // Average H¹ rate
        };

        /**
         * @brief Compute convergence rates from error data
         */
        static ConvergenceData compute_convergence_rates(
            const std::vector<int>& mesh_types,
            const std::vector<double>& l2_errors,
            const std::vector<double>& h1_errors
        );

        /**
         * @brief Print convergence analysis table
         * 
         * @param data Convergence data
         * @param vem_order VEM order
         */
        static void print_convergence_analysis(const ConvergenceData& data, int vem_order);

        // Getters
        const TestResults& get_results() const { return results_; }
        void set_config(const TestConfig& config) { config_ = config; }
    };


    // ============================================================================
    // STANDALONE TEST FUNCTION
    // ============================================================================

    /**
     * @brief Standalone function to run time scheme tests for k=1
     */
    void run_time_scheme_tests_k1();

    /**
     * @brief Standalone function to run time scheme tests
     */
    void run_time_scheme_tests();

    /**
     * @brief Run a single JSON mesh test and save results to JSON
     * @param json_mesh_file Path to input JSON mesh file
     * @param output_json_file Path to output JSON results file
     * @param vem_order VEM polynomial order (default: 1)
     * @param final_time Final simulation time (default: 1.0)
     * @param time_step Time step size (default: 0.001)
     * @param verbose Enable verbose output (default: false)
     */
    void run_single_json_mesh_test(
        const std::string& json_mesh_file,
        const std::string& output_json_file,
        int vem_order = 1,
        double final_time = 1.0,
        double time_step = 0.001,
        bool verbose = false
    );
}



#endif // TEST_TIME_SCHEME_HPP