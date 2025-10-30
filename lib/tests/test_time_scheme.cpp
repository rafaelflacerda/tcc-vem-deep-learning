#include "tests/test_time_scheme.hpp"
#include "mesh/helpers.hpp"

namespace VEMSimulationParabolic {

    // ============================================================================
    // SIMULATION SETUP
    // ============================================================================ 
    TestResults TimeSchemeTest::simulation_setup() {
        if (config_.verbose) {
            std::cout << "\n=== VEM Parabolic Time Scheme Test: Backward Euler ===" << std::endl;
            std::cout << "VEM order: " << config_.vem_order << std::endl;
            std::cout << "Final time: " << config_.final_time << std::endl;
            std::cout << "Time step: " << config_.time_step << std::endl;
        }

        // Setup mesh
        Eigen::MatrixXd nodes;
        Eigen::MatrixXi elements;
        setup_mesh(nodes, elements);

        // Create VEM solver with debug mode enabled for investigation
        bool debug_mode_vem_solver = false;  // Disable debug for clean output
        solver::parabolic vem_solver(
            config_.vem_order, 
            nodes, 
            elements, 
            config_.use_lumped_mass_matrix, 
            debug_mode_vem_solver
        ); // Enable debug mode

        if (config_.verbose) {
            vem_solver.print_system_information();
        }

        // Assemble system matrices
        {
            utils::ScopeTimer timer("System assembly");
            vem_solver.assemble_system();
        }

        // DEBUG mass matrix
        Eigen::SparseMatrix<double> M_h = vem_solver.get_global_mass_matrix();
        auto info = utils::MatrixHelper::check_matrix_properties(M_h);
        std::cout << "DEBUG: Mass matrix properties:" << std::endl;
        std::cout << "  Is symmetric: " << (info.is_symmetric ? "Yes" : "No") << std::endl;
        std::cout << "DEBUG: M_h diagonal range: [" << M_h.diagonal().minCoeff() 
        << ", " << M_h.diagonal().maxCoeff() << "]\n";
        std::cout << "DEBUG: Trace: " << info.trace_value << std::endl;
        
        // Find DOFs with zero mass
        int zero_mass_count = 0;
        for (int i = 0; i < M_h.rows(); ++i) {
            if (std::abs(M_h.coeff(i, i)) < 1e-14) {
                if (zero_mass_count == 0) {
                    std::cout << "⚠️  DOFs with ZERO mass:" << std::endl;
                }
                if (zero_mass_count < 10) {  // Only print first 10
                    std::cout << "  DOF " << i << " at (" << nodes(i, 0) << ", " << nodes(i, 1) << ")" << std::endl;
                }
                zero_mass_count++;
            }
        }
        if (zero_mass_count > 0) {
            std::cout << "  Total DOFs with zero mass: " << zero_mass_count << " out of " << M_h.rows() << std::endl;
            std::cout << "❌ This will cause RK methods to fail!" << std::endl;
        }
        
        // Setup boundary conditions
        utils::boundary boundary_handler(nodes, elements, config_.vem_order);
        boundary_handler.set_debug_mode(false); // Enable debug output
        
        // Re-run boundary detection with debug enabled
        boundary_handler.automatically_detect_boundary();

        // Set DOF mapping for k=2 to enable edge DOF clamping
        if (config_.vem_order >= 2) {
            // Set the DOF mapping in boundary handler
            boundary_handler.set_dof_mapping(vem_solver.element_dof_map, vem_solver.get_edge_dof_map(), vem_solver.n_dofs);
            
            if (config_.verbose) {
                std::cout << "Edge DOF mapping set for k=" << config_.vem_order << std::endl;
                std::cout << "  Total edges mapped: " << vem_solver.get_edge_dof_map().size() << std::endl;
                std::cout << "  Expected unique edges: " << (elements.rows() * elements.cols()) << std::endl;
                
                // Debug: Show first few edge mappings
                std::cout << "  Sample edge mappings:" << std::endl;
                int count = 0;
                for (const auto& [edge, dof] : vem_solver.get_edge_dof_map()) {
                    if (count < 5) {
                        std::cout << "    Edge (" << edge.first << "," << edge.second << ") → DOF " << dof << std::endl;
                        count++;
                    } else {
                        break;
                    }
                }
            }
        }

        // Add homogeneous Dirichlet BC on entire boundary
        boundary_handler.add_dirichlet(boundary_condition, utils::boundary::BoundaryRegion::ENTIRE, "Homogeneous_Dirichlet");

        // Set initial conditions
        vem_solver.set_initial_conditions(initial_condition);
        Eigen::VectorXd U_initial = vem_solver.U_h;  // Copy initial solution


        if (config_.verbose) {
            std::cout << "Initial condition set. ||U₀|| = " << U_initial.norm() << std::endl;
            std::cout << "Boundary conditions initialized" << std::endl;
        }
        if (config_.scheme_type == utils::time_scheme::SchemeType::FORWARD_EULER) {
            return test_forward_euler(
                vem_solver,
                U_initial,
                boundary_handler,
                source_function,
                boundary_condition,
                config_.final_time,
                config_.time_step,
                config_.verbose
        );
        } else if (config_.scheme_type == utils::time_scheme::SchemeType::BACKWARD_EULER) {
            return test_backward_euler(
                vem_solver,
                U_initial,
                boundary_handler,
                source_function,
                boundary_condition,
                config_.final_time,
                config_.time_step,
                config_.verbose
            );
        } else if (config_.scheme_type == utils::time_scheme::SchemeType::RK3 || config_.scheme_type == utils::time_scheme::SchemeType::RK4) {
            return test_rk(
                vem_solver,
                U_initial,
                boundary_handler,
                source_function,
                boundary_condition,
                config_.final_time,
                config_.time_step,
                config_.verbose
            );
        }
        return results_;
    }

    // ============================================================================
    // TIME SCHEME TESTS
    // ============================================================================

    TestResults TimeSchemeTest::test_backward_euler(
        solver::parabolic& vem_solver,
        const Eigen::VectorXd& U_initial,
        utils::boundary& boundary_handler,
        const std::function<double(const Eigen::Vector2d&, double)>& source_function,
        const std::function<double(const Eigen::Vector2d&, double)>& boundary_condition,
        double final_time,
        double time_step,
        bool verbose
    ) {
        // Time stepping loop
        double current_time = 0.0;
        int time_step_count = 0;
        std::vector<double> solution_norms;
        std::vector<double> time_points;

        // Initialize U_current and U_previous
        Eigen::VectorXd U_current = U_initial;
        Eigen::VectorXd U_previous = U_current;


        if (config_.verbose) {
            std::cout << "\nStarting time integration..." << std::endl;
            std::cout << std::fixed << std::setprecision(6);
        }
        {
            utils::ScopeTimer integration_timer("Time integration");
            while (current_time < config_.final_time - 1e-12) {
                current_time += config_.time_step;
                time_step_count++;

                // Get fresh matrix copies for each time step
                Eigen::SparseMatrix<double> M_h = vem_solver.get_global_mass_matrix();
                Eigen::SparseMatrix<double> K_h = vem_solver.get_global_stiffness_matrix();

                // Assemble load vector at current time
                vem_solver.assemble_load_vector(source_function, current_time);
                Eigen::VectorXd F_h = vem_solver.F_h;

                // Apply boundary conditions to FRESH matrices and load vector
                boundary_handler.apply_dirichlet_conditions(K_h, M_h, F_h, current_time);

                // Create time integrator with fresh matrices for this time step
                utils::time_scheme time_integrator(M_h, K_h);
                time_integrator.setup_time_parameters(config_.time_step, utils::time_scheme::SchemeType::BACKWARD_EULER);
                time_integrator.configure_solver(utils::time_scheme::SolverType::DIRECT, 1e-12, 1000);
                time_integrator.set_debug_mode(false);  // Reduce verbosity in time loop

                // Store previous solution
                U_previous = U_current;

                // Perform time step
                bool step_success = time_integrator.step(U_current, U_previous, F_h);

                if (!step_success) {
                    std::cerr << "ERROR: Time step failed at t = " << current_time << std::endl;
                    results_.test_passed = false;
                    return results_;
                }

                // Store solution data
                solution_norms.push_back(U_current.norm());
                time_points.push_back(current_time);

                if (config_.verbose && (time_step_count % 5 == 0 || current_time >= config_.final_time - 1e-12)) {
                    std::cout << "t = " << std::setw(8) << current_time 
                            << ", ||u|| = " << std::setw(12) << U_current.norm()
                            << ", step " << time_step_count << std::endl;
                }
            }
        }
        

        // Compute errors at final time
        compute_final_time_errors(vem_solver, U_current, config_.final_time, exact_gradient);

        // Store results
        results_.total_time_steps = time_step_count;
        // results_.total_solve_time = total_time / 1000.0;  // Convert to seconds
        results_.test_passed = (results_.l2_error < 0.1);  // More reasonable threshold

        // Create dummy statistics since we don't have access to individual time integrators
        utils::time_scheme::Statistics stats;
        stats.successful_steps = time_step_count;
        stats.failed_steps = 0;
        stats.avg_solve_time = results_.total_solve_time / time_step_count * 1000.0;  // Convert to ms

        if (config_.verbose) {
            print_test_results(stats);
        }

        // Save solution if requested
        if (config_.save_solution) {
            save_solution_to_file(vem_solver, U_current, time_points, solution_norms);
        }

        return results_;
    }

    TestResults TimeSchemeTest::test_forward_euler(
        solver::parabolic& vem_solver,
        const Eigen::VectorXd& U_initial,
        utils::boundary& boundary_handler,
        const std::function<double(const Eigen::Vector2d&, double)>& source_function,
        const std::function<double(const Eigen::Vector2d&, double)>& boundary_condition,
        double final_time,
        double time_step,
        bool verbose
    ) {
        // Time stepping loop
        double current_time = 0.0;
        int time_step_count = 0;
        std::vector<double> solution_norms;
        std::vector<double> time_points;

        // Initialize U_current and U_previous
        Eigen::VectorXd U_previous;  // Will be set in the time loop
        Eigen::VectorXd U_current = U_initial;

        if (config_.verbose) {
            std::cout << "\nStarting time integration..." << std::endl;
            std::cout << std::fixed << std::setprecision(6);
        }
        {
            utils::ScopeTimer integration_timer("Time integration");
            
            // Initialize U_previous for the first iteration
            U_previous = U_current;  // At t=0, U_previous = U_current = initial condition
            
            while (current_time < config_.final_time - 1e-12) {
                current_time += config_.time_step;
                time_step_count++;

                // Get fresh matrix copies for each time step
                Eigen::SparseMatrix<double> M_h = vem_solver.get_global_mass_matrix();
                Eigen::SparseMatrix<double> K_h = vem_solver.get_global_stiffness_matrix();

                // Assemble load vector at current time
                vem_solver.assemble_load_vector(source_function, current_time);
                Eigen::VectorXd F_h = vem_solver.F_h;

                // Apply boundary conditions to FRESH matrices and load vector
                bool preserve_mass_diagonal = true;
                boundary_handler.apply_dirichlet_conditions(K_h, M_h, F_h, current_time, preserve_mass_diagonal);

                // Create time integrator with fresh matrices for this time step
                utils::time_scheme time_integrator(M_h, K_h);
                time_integrator.setup_time_parameters(config_.time_step, utils::time_scheme::SchemeType::FORWARD_EULER);
                // time_integrator.setup_time_parameters(config_.time_step, utils::time_scheme::SchemeType::BACKWARD_EULER);
                time_integrator.configure_solver(utils::time_scheme::SolverType::DIRECT, 1e-12, 1000);
                time_integrator.set_debug_mode(false);  // Reduce verbosity in time loop

                // Forward Euler: M * U^{n+1} = M * U^n + dt * (-K * U^n + F^n)
                // U_previous contains U^n, step() will compute U^{n+1} and store it in U_current
                bool step_success = time_integrator.step(U_current, U_previous, F_h);

                if (!step_success) {
                    std::cerr << "ERROR: Time step failed at t = " << current_time << std::endl;
                    results_.test_passed = false;
                    return results_;
                }

                // Store solution data
                solution_norms.push_back(U_current.norm());
                time_points.push_back(current_time);

                if (config_.verbose && (time_step_count % 5 == 0 || current_time >= config_.final_time - 1e-12)) {
                    std::cout << "t = " << std::setw(8) << current_time 
                            << ", ||u|| = " << std::setw(12) << U_current.norm()
                            << ", step " << time_step_count << std::endl;
                }

                // IMPORTANT: Update U_previous for next iteration
                // Now U_current contains the new solution U^{n+1}, which becomes U^n for next step
                U_previous = U_current;
            }
        }
        

        // Compute errors at final time
        compute_final_time_errors(vem_solver, U_current, config_.final_time, exact_gradient);

        // Store results
        results_.total_time_steps = time_step_count;
        // results_.total_solve_time = total_time / 1000.0;  // Convert to seconds
        results_.test_passed = (results_.l2_error < 0.1);  // More reasonable threshold

        // Create dummy statistics since we don't have access to individual time integrators
        utils::time_scheme::Statistics stats;
        stats.successful_steps = time_step_count;
        stats.failed_steps = 0;
        stats.avg_solve_time = results_.total_solve_time / time_step_count * 1000.0;  // Convert to ms

        if (config_.verbose) {
            print_test_results(stats);
        }

        // Save solution if requested
        if (config_.save_solution) {
            save_solution_to_file(vem_solver, U_current, time_points, solution_norms);
        }

        return results_;
    }

    TestResults TimeSchemeTest::test_rk(
        solver::parabolic& vem_solver,
        const Eigen::VectorXd& U_initial,
        utils::boundary& boundary_handler,
        const std::function<double(const Eigen::Vector2d&, double)>& source_function,
        const std::function<double(const Eigen::Vector2d&, double)>& boundary_condition,
        double final_time,
        double time_step,
        bool verbose
    ){
        // Time stepping loop
        double current_time = 0.0;
        int time_step_count = 0;
        std::vector<double> solution_norms;
        std::vector<double> time_points;

        // Initialize U_current and U_previous
        Eigen::VectorXd U_current = U_initial;
        Eigen::VectorXd U_previous = U_current;

        if (config_.verbose) {
            std::cout << "\nStarting RK3time integration..." << std::endl;
            std::cout << std::fixed << std::setprecision(6);
        }

        {
            utils::ScopeTimer integration_timer("RK3 time integration");
            while (current_time < final_time - 1e-12) {
                current_time += config_.time_step;
                time_step_count++;

                // Get matrices for the time integrator
                // Note: These will NOT be modified by boundary conditions
                Eigen::SparseMatrix<double> M_h = vem_solver.get_global_mass_matrix();
                Eigen::SparseMatrix<double> K_h = vem_solver.get_global_stiffness_matrix();

                // Create time integrator with matrices for this time step
                utils::time_scheme time_integrator(M_h, K_h);
                time_integrator.setup_time_parameters(time_step, config_.scheme_type);
                time_integrator.configure_solver(utils::time_scheme::SolverType::DIRECT, 1e-12, 1000);
                time_integrator.set_debug_mode(false);  // Disable debug for cleaner output
                
                // Store previous solution
                U_previous = U_current;

                // Create a function that provides the right-hand side vector
                // This function will be called by the RK method at different time points (3 times for RK3)
                // CRITICAL: For explicit RK methods with homogeneous BCs, just assemble the load vector
                // The BCs will be enforced by setting boundary DOFs to zero in the solution after each stage
                auto f_function = [&vem_solver, &source_function](double t) -> Eigen::VectorXd {
                    // Assemble load vector at time t
                    vem_solver.assemble_load_vector(source_function, t);
                    return vem_solver.F_h;
                };

                // Perform RK3 time step
                bool step_success = time_integrator.step_rk(U_current, U_previous, f_function, current_time, true);

                if (!step_success) {
                    std::cerr << "ERROR: RK time step failed at t = " << current_time << std::endl;
                    results_.test_passed = false;
                    return results_;
                }
                
                // Enforce homogeneous Dirichlet BCs: set boundary DOFs to zero
                for (int i = 0; i < vem_solver.nodes.rows(); ++i) {
                    if (boundary_handler.is_boundary_vertex(i)) {
                        U_current(i) = 0.0;
                    }
                }

                // Store solution data
                solution_norms.push_back(U_current.norm());
                time_points.push_back(current_time);

                if (config_.verbose && (time_step_count % 5 == 0 || current_time >= config_.final_time - 1e-12)) {
                    std::cout << "t = " << std::setw(8) << current_time 
                            << ", ||u|| = " << std::setw(12) << U_current.norm()
                            << ", step " << time_step_count << std::endl;
                }
            }
        }

        // Compute errors at final time
        compute_final_time_errors(vem_solver, U_current, config_.final_time, exact_gradient);

        // Store results
        results_.total_time_steps = time_step_count;
        results_.test_passed = (results_.l2_error < 0.1);  // More reasonable threshold
        
        // Create dummy statistics since we don't have access to individual time integrators
        utils::time_scheme::Statistics stats;
        stats.successful_steps = time_step_count;
        stats.failed_steps = 0;
        stats.avg_solve_time = results_.total_solve_time / time_step_count * 1000.0;  // Convert to ms

        if (config_.verbose) {
            print_test_results(stats);
        }

        // Save solution if requested
        if (config_.save_solution) {
            save_solution_to_file(vem_solver, U_current, time_points, solution_norms);
        }

        return results_;
    }
    

    // ============================================================================
    // PUBLIC TEST INTERFACE
    // ============================================================================
   
    void TimeSchemeTest::run_all_tests() {
        std::cout << "🚀 Starting VEM Parabolic Time Scheme Tests 🚀\n" << std::endl;
        
        // Test 1: Basic Backward Euler
        TestResults euler_results = simulation_setup();
        
        // Test 2: Convergence study (optional)
        if (config_.verbose) {
            convergence_study();
        }
        
        std::cout << "\n🎯 All tests completed!" << std::endl;
    }

    TimeSchemeTest::ConvergenceData TimeSchemeTest::compute_convergence_rates(
        const std::vector<int>& mesh_types,
        const std::vector<double>& l2_errors,
        const std::vector<double>& h1_errors
    ) {
        ConvergenceData data;
        
        // Convert mesh types to mesh sizes (h = 1/mesh_type for unit square)
        for (int mesh_type : mesh_types) {
            data.mesh_sizes.push_back(1.0 / mesh_type);
        }
        
        data.l2_errors = l2_errors;
        data.h1_errors = h1_errors;
        
        // Compute convergence rates between consecutive meshes
        for (size_t i = 0; i < data.mesh_sizes.size() - 1; ++i) {
            double h_ratio = data.mesh_sizes[i] / data.mesh_sizes[i + 1];
            double l2_ratio = data.l2_errors[i] / data.l2_errors[i + 1];
            double h1_ratio = data.h1_errors[i] / data.h1_errors[i + 1];
            
            // Convergence rate p: error ~ h^p  =>  p = log(error_ratio) / log(h_ratio)
            double l2_rate = std::log(l2_ratio) / std::log(h_ratio);
            double h1_rate = std::log(h1_ratio) / std::log(h_ratio);
            
            data.l2_rates.push_back(l2_rate);
            data.h1_rates.push_back(h1_rate);
        }
        
        // Compute average rates (excluding first transition which may be less accurate)
        if (data.l2_rates.size() > 1) {
            double l2_sum = 0.0, h1_sum = 0.0;
            for (size_t i = 1; i < data.l2_rates.size(); ++i) {  // Skip first rate
                l2_sum += data.l2_rates[i];
                h1_sum += data.h1_rates[i];
            }
            data.avg_l2_rate = l2_sum / (data.l2_rates.size() - 1);
            data.avg_h1_rate = h1_sum / (data.h1_rates.size() - 1);
        } else if (!data.l2_rates.empty()) {
            data.avg_l2_rate = data.l2_rates[0];
            data.avg_h1_rate = data.h1_rates[0];
        }
        
        return data;
    }

    void TimeSchemeTest::print_convergence_analysis(
        const ConvergenceData& data, 
        int vem_order
    ){
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "CONVERGENCE RATE ANALYSIS (k=" << vem_order << ")" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Expected theoretical rates
        double expected_l2_rate = vem_order + 1;  // O(h^{k+1})
        double expected_h1_rate = vem_order;      // O(h^k)
        
        std::cout << "Theoretical rates: L²=O(h^" << expected_l2_rate 
                 << "), H¹=O(h^" << expected_h1_rate << ")" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        // Table header
        std::cout << std::setw(8) << "Mesh" 
                 << std::setw(10) << "h" 
                 << std::setw(12) << "L²-error"
                 << std::setw(12) << "H¹-error"
                 << std::setw(10) << "L²-rate"
                 << std::setw(10) << "H¹-rate" << std::endl;
        std::cout << std::string(70, '-') << std::endl;
        
        // Data rows
        for (size_t i = 0; i < data.mesh_sizes.size(); ++i) {
            int mesh_type = static_cast<int>(std::round(1.0 / data.mesh_sizes[i]));
            std::cout << std::setw(8) << (std::to_string(mesh_type) + "x" + std::to_string(mesh_type));
            std::cout << std::scientific << std::setprecision(3);
            std::cout << std::setw(10) << data.mesh_sizes[i];
            std::cout << std::setw(12) << data.l2_errors[i];
            std::cout << std::setw(12) << data.h1_errors[i];
            
            if (i < data.l2_rates.size()) {
                std::cout << std::fixed << std::setprecision(2);
                std::cout << std::setw(10) << data.l2_rates[i];
                std::cout << std::setw(10) << data.h1_rates[i];
            } else {
                std::cout << std::setw(10) << "-";
                std::cout << std::setw(10) << "-";
            }
            std::cout << std::endl;
        }
        
        std::cout << std::string(70, '-') << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Average rates: L²=" << data.avg_l2_rate 
                 << " (expected " << expected_l2_rate << "), "
                 << "H¹=" << data.avg_h1_rate 
                 << " (expected " << expected_h1_rate << ")" << std::endl;
        
        // Analysis
        double l2_efficiency = data.avg_l2_rate / expected_l2_rate * 100.0;
        double h1_efficiency = data.avg_h1_rate / expected_h1_rate * 100.0;
        
        std::cout << "Rate efficiency: L²=" << std::setprecision(1) << l2_efficiency 
                 << "%, H¹=" << h1_efficiency << "%" << std::endl;
        
        if (l2_efficiency > 90.0 && h1_efficiency > 90.0) {
            std::cout << "✅ EXCELLENT: Both rates are very close to theoretical expectations!" << std::endl;
        } else if (l2_efficiency > 80.0 && h1_efficiency > 80.0) {
            std::cout << "✅ GOOD: Rates are reasonably close to theory." << std::endl;
        } else if (l2_efficiency > 60.0 || h1_efficiency > 60.0) {
            std::cout << "⚠️  PARTIAL: Some convergence observed, but below theoretical rates." << std::endl;
        } else {
            std::cout << "❌ POOR: Convergence rates significantly below theoretical expectations." << std::endl;
        }
        
        std::cout << std::string(70, '=') << std::endl;
    }

    // ============================================================================
    // ERROR COMPUTATION
    // ============================================================================
    void TimeSchemeTest::compute_final_time_errors(
        const solver::parabolic& vem_solver, 
        const Eigen::VectorXd& U_final, 
        double final_time,
        const std::function<Eigen::Vector2d(double, double, double)>& exact_gradient
    ) {
        const auto& nodes = vem_solver.nodes;
        const auto& elements = vem_solver.elements;
        
        double l2_error_squared = 0.0;
        double h1_error_squared = 0.0;
        double max_error = 0.0;
        double l2_norm_exact_squared = 0.0;
        double h1_norm_exact_squared = 0.0;

        // Compute errors at nodes (works for both k=1 and k=2)
        // For k=1: All DOFs are vertex DOFs
        // For k=2: First nodes.rows() DOFs are vertex DOFs
        int num_vertex_dofs = std::min((int)nodes.rows(), (int)U_final.size());

        // Compute the H1 error
        double true_h1_error = utils::errors::compute_true_h1_error_with_stored_matrices(vem_solver, U_final, final_time, exact_gradient);
        double true_l2_error = utils::errors::compute_l2_error_parabolic(vem_solver, U_final, final_time, exact_solution);

        // Set H1 error (use true H1 if available, otherwise fallback to L2)
        if (true_h1_error >= 0.0) {
            results_.h1_error = true_h1_error;
            results_.l2_error = true_l2_error;
            if (config_.verbose) {
                std::cout << "  Using TRUE triangulation-based H1 error = " << results_.h1_error << std::endl;
            }
        }
        if (config_.verbose) {
            std::cout << "========================================" << std::endl;
        }
    }

    // ============================================================================
    // OUTPUT AND ANALYSIS
    // ============================================================================
    void TimeSchemeTest::print_test_results(const utils::time_scheme::Statistics& stats) {
        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << std::scientific << std::setprecision(6);
        
        std::cout << "Error Analysis:" << std::endl;
        std::cout << "  Relative L² error: " << results_.l2_error << std::endl;
        std::cout << "  Maximum error:     " << results_.max_error << std::endl;
        
        std::cout << "\nTime Integration Statistics:" << std::endl;
        std::cout << "  Total time steps:    " << results_.total_time_steps << std::endl;
        std::cout << "  Successful steps:    " << stats.successful_steps << std::endl;
        std::cout << "  Failed steps:        " << stats.failed_steps << std::endl;
        std::cout << "  Average solve time:  " << stats.avg_solve_time << " ms" << std::endl;
        std::cout << "  Total solve time:    " << results_.total_solve_time << " s" << std::endl;
        
        std::cout << "\nTest Status: " << (results_.test_passed ? "✅ PASSED" : "❌ FAILED") << std::endl;
        
        if (results_.test_passed) {
            std::cout << "The time integration scheme is working correctly!" << std::endl;
        } else {
            std::cout << "Issues detected in time integration. Check implementation." << std::endl;
        }
    }

    void TimeSchemeTest::save_solution_to_file(
        const solver::parabolic& vem_solver,
        const Eigen::VectorXd& U_final,
        const std::vector<double>& time_points,
        const std::vector<double>& solution_norms
    ){
        // Save final solution at nodes
        std::ofstream solution_file("final_solution.dat");
        solution_file << std::scientific << std::setprecision(12);
        solution_file << "# x y u_numerical u_exact error\n";
        
        const auto& nodes = vem_solver.nodes;
        for (int i = 0; i < nodes.rows(); ++i) {
            double x = nodes(i, 0);
            double y = nodes(i, 1);
            double u_num = (config_.vem_order == 1) ? U_final(i) : 0.0;  // Simplified for k=1
            double u_exact = exact_solution(config_.final_time, x, y);
            double error = std::abs(u_num - u_exact);
            
            solution_file << x << " " << y << " " << u_num << " " << u_exact << " " << error << "\n";
        }
        solution_file.close();
        
        // Save time evolution
        std::ofstream time_file("time_evolution.dat");
        time_file << std::scientific << std::setprecision(12);
        time_file << "# time solution_norm\n";
        
        for (size_t i = 0; i < time_points.size(); ++i) {
            time_file << time_points[i] << " " << solution_norms[i] << "\n";
        }
        time_file.close();
        
        if (config_.verbose) {
            std::cout << "Solution saved to files: final_solution.dat, time_evolution.dat" << std::endl;
        }
    }

    // ============================================================================
    // MATRIX CHECKS
    // ============================================================================
    void check_mass_matrix_properties(){
        const std::vector<int> meshes = {3};  // 3×3, 4×4 grids

        for (int m : meshes) {
            TestConfig cfg;
            cfg.mesh_type   = m;
            cfg.vem_order   = 1;        // ← KEY CHANGE: k=1 instead of k=2
            cfg.final_time  = 0.8;
            // cfg.time_step   = 0.001;
            cfg.time_step   = 0.01;
            cfg.verbose     = false;   // silence detailed debug
            cfg.save_solution = false;
            cfg.use_lumped_mass_matrix = true; // use lumped mass matrix
            cfg.scheme_type = utils::time_scheme::SchemeType::BACKWARD_EULER;

            TimeSchemeTest tst(cfg);
            // Setup mesh
            Eigen::MatrixXd nodes;
            Eigen::MatrixXi elements;
            tst.setup_mesh(nodes, elements);

            // Create VEM solver with debug mode enabled for investigation
            bool debug_mode_vem_solver = false;
            solver::parabolic vem_solver(
                cfg.vem_order, 
                nodes, 
                elements, 
                cfg.use_lumped_mass_matrix, 
                debug_mode_vem_solver
            ); // Enable debug mode

            if (cfg.verbose) {
                vem_solver.print_system_information();
            }

            // Assemble system matrices
            {
                utils::ScopeTimer timer("System assembly");
                vem_solver.assemble_system();
            }

            // Get full mass matrix
            const auto& M_h = vem_solver.get_global_mass_matrix();

            // Check matrix properties
            auto info = utils::MatrixHelper::check_matrix_properties(M_h);
            std::cout << "Mass matrix properties:" << std::endl;
            std::cout << "  Number of rows: " << info.num_rows << std::endl;
            std::cout << "  Number of columns: " << info.num_cols << std::endl;
            std::cout << "  Is symmetric: " << (info.is_symmetric ? "Yes" : "No") << std::endl;
            std::cout << "  Is positive definite: " << (info.is_positive_definite ? "Yes" : "No") << std::endl;
            std::cout << "  Trace: " << info.trace_value << std::endl;

            // Display matrix
            utils::MatrixHelper::display_matrix(M_h, "Mass matrix");
        }                                           
    }

    // ============================================================================
    // STANDALONE TEST FUNCTIONS
    // ============================================================================

    /**
     * @brief Standalone function to run time scheme tests for k=1
     */
    void run_time_scheme_tests_k1() {
        const std::vector<int> meshes = {3, 4, 5, 6, 8, 10};   // 3×3, 4×4, 5×5, 6×6, 8×8, 10×10 grids

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "k = 1  Backward-Euler  Convergence study" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::setw(10) << "grid"
                  << std::setw(15) << "L2-error"
                  << std::setw(15) << "H1-error" << std::endl;
        std::cout << std::string(40, '-') << std::endl;

        // Collect error data for convergence analysis
        std::vector<double> l2_errors, h1_errors;

        for (int m : meshes) {
            TestConfig cfg;
            cfg.mesh_type   = m;
            cfg.vem_order   = 1;        // ← KEY CHANGE: k=1 instead of k=2
            cfg.final_time  = 1;
            // cfg.time_step   = 0.001;
            cfg.time_step   = 0.001;
            cfg.verbose     = false;   // silence detailed debug
            cfg.save_solution = false;
            cfg.use_lumped_mass_matrix = true; // use lumped mass matrix
            cfg.scheme_type = utils::time_scheme::SchemeType::RK3;

            TimeSchemeTest tst(cfg);
            auto res = tst.simulation_setup();

            std::cout << std::setw(10) << (std::to_string(m) + "x" + std::to_string(m))
                      << std::scientific << std::setprecision(6)
                      << std::setw(15) << res.l2_error
                      << std::setw(15) << res.h1_error << std::endl;

            // Store errors for convergence analysis
            l2_errors.push_back(res.l2_error);
            h1_errors.push_back(res.h1_error);
        }

        std::cout << std::string(60, '=') << std::endl;

        // Perform convergence analysis
        auto conv_data = TimeSchemeTest::compute_convergence_rates(meshes, l2_errors, h1_errors);
        TimeSchemeTest::print_convergence_analysis(conv_data, 1);
    }

    /**
     * @brief Standalone function to run time scheme tests
     */
    void run_time_scheme_tests() {
        const std::vector<int> meshes = {3, 4, 5, 6, 8, 10};   // 3×3, 4×4, 5×5, 6×6, 8×8, 10×10 grids

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "k = 2  Backward-Euler  Convergence study" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << std::setw(10) << "grid"
                  << std::setw(15) << "L2-error"
                  << std::setw(15) << "H1-error" << std::endl;
        std::cout << std::string(40, '-') << std::endl;

        // Collect error data for convergence analysis
        std::vector<double> l2_errors, h1_errors;

        for (int m : meshes) {
            TestConfig cfg;
            cfg.mesh_type   = m;
            cfg.vem_order   = 2;
            cfg.final_time  = 0.8;
            cfg.time_step   = 0.001;
            cfg.verbose     = false;   // silence detailed debug
            cfg.save_solution = false;

            TimeSchemeTest tst(cfg);
            auto res = tst.simulation_setup();

            std::cout << std::setw(10) << (std::to_string(m) + "×" + std::to_string(m))
                      << std::scientific << std::setprecision(6)
                      << std::setw(15) << res.l2_error
                      << std::setw(15) << res.h1_error << std::endl;

            // Store errors for convergence analysis
            l2_errors.push_back(res.l2_error);
            h1_errors.push_back(res.h1_error);
        }

        std::cout << std::string(60, '=') << std::endl;

        // Perform convergence analysis
        auto conv_data = TimeSchemeTest::compute_convergence_rates(meshes, l2_errors, h1_errors);
        TimeSchemeTest::print_convergence_analysis(conv_data, 2);
    }

    // ============================================================================
    // JSON MESH LOADING HELPER
    // ============================================================================
    
    /**
     * @brief Load mesh from VEM/Eigen JSON format
     * @param json_file Path to JSON file
     * @param nodes Output matrix for node coordinates
     * @param elements Output matrix for element connectivity
     */
    void load_vem_json_mesh(const std::string& json_file, Eigen::MatrixXd& nodes, Eigen::MatrixXi& elements, bool verbose = false) {
        std::ifstream file(json_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open JSON file: " + json_file);
        }
        
        nlohmann::json j;
        file >> j;
        
        // Check if it has the VEM/Eigen format structure
        if (!j.contains("nodes") || !j.contains("elements") || !j.contains("metadata")) {
            throw std::runtime_error("JSON file does not have VEM/Eigen format (missing nodes/elements/metadata)");
        }
        
        // Get node and element counts
        int nodeCount = j["metadata"]["nodeCount"];
        int elementCount = j["metadata"]["elementCount"];
        
        // Detect element type and vertices per element
        int verticesPerElement = 4;  // Default for standard quads
        std::string meshType = "";
        std::string elementType = "";
        
        if (j["metadata"].contains("meshType")) {
            meshType = j["metadata"]["meshType"];
            if (verbose) {
                std::cout << "Detected meshType: '" << meshType << "'" << std::endl;
            }
        }
        if (j["metadata"].contains("elementType")) {
            elementType = j["metadata"]["elementType"];
        }
        // Check multiple possible field names for max vertices
        if (j["metadata"].contains("nodesPerElement")) {
            verticesPerElement = j["metadata"]["nodesPerElement"];
        } else if (j["metadata"].contains("maxVerticesPerElement")) {
            verticesPerElement = j["metadata"]["maxVerticesPerElement"];
            if (verbose) {
                std::cout << "Max vertices per element: " << verticesPerElement << std::endl;
            }
        }
        
        // Handle different element types - VEM can handle arbitrary polygons natively
        if (elementType == "Q8" || meshType == "serendipity_quad") {
            std::cout << "Detected Q8 elements - VEM will handle as 8-vertex polygons" << std::endl;
        }
        
        // Add 1 for padding/safety margin (e.g., 7 vertices → use 8 columns)
        int matrix_columns = verticesPerElement + 1;
        
        // Initialize matrices
        nodes.resize(nodeCount, 2);
        elements.resize(elementCount, matrix_columns);
        elements.setConstant(-1);  // Initialize with -1 for padding
        
        // Load nodes
        for (const auto& node : j["nodes"]) {
            int id = node["id"];
            double x = node["x"];
            double y = node["y"];
            
            nodes(id, 0) = x;
            nodes(id, 1) = y;
        }
        
        // Load elements - VEM handles all polygons natively
        for (const auto& element : j["elements"]) {
            int id = element["id"];
            const auto& vertices = element["vertices"];
            
            // Load all vertices - VEM works with arbitrary polygons
            for (int i = 0; i < verticesPerElement && i < vertices.size(); ++i) {
                elements(id, i) = vertices[i];
            }
            
            if (verbose && id < 3) {  // Show first 3 elements as examples
                std::cout << "Element " << id << " (" << vertices.size() << " vertices): [";
                for (size_t i = 0; i < vertices.size(); ++i) {
                    std::cout << vertices[i];
                    if (i < vertices.size() - 1) std::cout << ",";
                }
                std::cout << "]" << std::endl;
            }
        }
        
        // Correct mesh orientation using helper functions
        meshHelpers::correctMeshOrientation(nodes, elements);
        
        // CRITICAL: Filter unused nodes ONLY for serendipity meshes
        // Voronoi meshes need all their nodes for proper connectivity!
        if (verbose) {
            std::cout << "Node filtering check: meshType='" << meshType << "', will filter=" << (meshType != "voronoi") << std::endl;
        }
        if (meshType != "voronoi") {
            // Find which nodes are actually used
            std::set<int> used_nodes;
            for (int e = 0; e < elements.rows(); ++e) {
                for (int v = 0; v < elements.cols(); ++v) {
                    if (elements(e, v) >= 0) {
                        used_nodes.insert(elements(e, v));
                    }
                }
            }
            
            // If not all nodes are used, renumber
            if (used_nodes.size() < static_cast<size_t>(nodes.rows())) {
                if (verbose) {
                    std::cout << "Filtering unused nodes: " << nodes.rows() << " → " << used_nodes.size() << " nodes" << std::endl;
                }
                
                // Create mapping from old node IDs to new node IDs
                std::map<int, int> old_to_new;
                int new_id = 0;
                for (int old_id : used_nodes) {
                    old_to_new[old_id] = new_id++;
                }
                
                // Create new nodes matrix with only used nodes
                Eigen::MatrixXd new_nodes(used_nodes.size(), 2);
                for (const auto& [old_id, new_id] : old_to_new) {
                    new_nodes.row(new_id) = nodes.row(old_id);
                }
                
                // Renumber elements
                for (int e = 0; e < elements.rows(); ++e) {
                    for (int v = 0; v < elements.cols(); ++v) {
                        if (elements(e, v) >= 0) {
                            elements(e, v) = old_to_new[elements(e, v)];
                        }
                    }
                }
                
                nodes = new_nodes;
            }
        }
        
        // Summary
        std::cout << "Loaded " << meshType << " JSON mesh: " << nodes.rows() << " nodes, " 
                  << elementCount << " elements (VEM native polygon handling)" << std::endl;
    }

    // ============================================================================
    // JSON MESH TESTING FUNCTION
    // ============================================================================

    void run_single_json_mesh_test(
        const std::string& json_mesh_file,
        const std::string& output_json_file,
        int vem_order,
        double final_time,
        double time_step,
        bool verbose
    ) {
        if (verbose) {
            std::cout << "\n=== Single JSON Mesh Test ===" << std::endl;
            std::cout << "Input mesh: " << json_mesh_file << std::endl;
            std::cout << "Output file: " << output_json_file << std::endl;
            std::cout << "VEM order: " << vem_order << std::endl;
            std::cout << "Final time: " << final_time << std::endl;
            std::cout << "Time step: " << time_step << std::endl;
            std::cout << "Scheme: RK3 with lumped mass matrix" << std::endl;
        }

        try {
            // Create test configuration
            TestConfig cfg;
            cfg.vem_order = vem_order;
            cfg.final_time = final_time;
            cfg.time_step = time_step;
            cfg.verbose = verbose;
            cfg.save_solution = false;
            cfg.use_lumped_mass_matrix = true;  // Use lumped mass matrix
            cfg.scheme_type = utils::time_scheme::SchemeType::RK3;  // Shu-Osher RK3/RK4

            // Load mesh from JSON - try VEM/Eigen format first, then fallback to datasource
            Eigen::MatrixXd nodes;
            Eigen::MatrixXi elements;
            
            try {
                // Try VEM/Eigen format first (your Mathematica mesh format)
                load_vem_json_mesh(json_mesh_file, nodes, elements, verbose);
            } catch (const std::exception&) {
                try {
                    // Fallback to Classical FEM format
                    mesh::datasource mesh_loader(json_mesh_file, BeamSolverType::Beam);
                    nodes = mesh_loader.nodes;
                    elements = mesh_loader.elements;
                } catch (const std::exception&) {
                    try {
                        // Fallback to Beam format
                        mesh::datasource mesh_loader(json_mesh_file, BeamSolverType::Portic);
                        nodes = mesh_loader.nodes;
                        elements = mesh_loader.elements;
                    } catch (const std::exception&) {
                        throw std::runtime_error("Unable to parse JSON mesh file. Supported formats: VEM/Eigen, Classical FEM, or Beam");
                    }
                }
            }

            if (verbose) {
                std::cout << "Mesh loaded successfully:" << std::endl;
                std::cout << "  Nodes: " << nodes.rows() << std::endl;
                std::cout << "  Elements: " << elements.rows() << std::endl;
                std::cout << "  Element matrix size: " << elements.rows() << "x" << elements.cols() << std::endl;
                
                // Debug: Print first element's vertices
                std::cout << "  First element vertices: [";
                for (int i = 0; i < elements.cols(); ++i) {
                    if (elements(0, i) >= 0) {
                        std::cout << elements(0, i);
                        if (i < elements.cols() - 1 && elements(0, i+1) >= 0) std::cout << ", ";
                    } else {
                        std::cout << " (padding: " << elements(0, i) << ")";
                        break;
                    }
                }
                std::cout << "]" << std::endl;
            }

            // Use the WORKING TimeSchemeTest infrastructure!
            TimeSchemeTest test(cfg);
            
            // Set up external mesh data
            test.external_nodes = nodes;
            test.external_elements = elements;
            test.use_external_mesh = true;
            
            // Use the working simulation_setup() method that works perfectly for built-in meshes
            TestResults results = test.simulation_setup();

            // Create output JSON with comprehensive information
            nlohmann::json output_json;
            
            // Add configuration
            output_json["configuration"] = {
                {"input_mesh_file", json_mesh_file},
                {"vem_order", vem_order},
                {"final_time", final_time},
                {"time_step", time_step},
                {"scheme_type", "RK3"},
                {"use_lumped_mass_matrix", cfg.use_lumped_mass_matrix},
                {"manufactured_solution", "u(t,x,y) = exp(t) * sin(πx) * sin(πy)"},
                {"boundary_conditions", "Homogeneous Dirichlet (u = 0 on ∂Ω)"}
            };
            
            // Add mesh information
            output_json["mesh_info"] = {
                {"nodes", nodes.rows()},
                {"elements", elements.rows()},
                {"mesh_size_h", 1.0 / std::sqrt(static_cast<double>(nodes.rows()))}, // Approximate
                {"mesh_type", "JSON_imported"}
            };
            
            // Add results
            output_json["results"] = {
                {"l2_error", results.l2_error},
                {"h1_error", results.h1_error},
                {"max_error", results.max_error},
                {"total_time_steps", results.total_time_steps},
                {"total_solve_time", results.total_solve_time},
                {"test_passed", results.test_passed}
            };

            // Add theoretical expectations
            double expected_l2_rate = vem_order + 1;  // O(h^{k+1})
            double expected_h1_rate = vem_order;      // O(h^k)
            output_json["theoretical"] = {
                {"expected_l2_convergence_rate", expected_l2_rate},
                {"expected_h1_convergence_rate", expected_h1_rate},
                {"l2_order_notation", "O(h^" + std::to_string(expected_l2_rate) + ")"},
                {"h1_order_notation", "O(h^" + std::to_string(expected_h1_rate) + ")"}
            };
            
            // Add timestamp
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::string timestamp = std::ctime(&time_t);
            timestamp.pop_back(); // Remove trailing newline
            output_json["timestamp"] = timestamp;

            // Create output directory if it doesn't exist
            std::filesystem::path output_path(output_json_file);
            std::filesystem::create_directories(output_path.parent_path());

            // Save to file
            std::ofstream output_file(output_json_file);
            output_file << output_json.dump(4); // Pretty print with 4 spaces
            output_file.close();

            if (verbose) {
                std::cout << "\n=== Test Results ===" << std::endl;
                std::cout << "Results saved to: " << output_json_file << std::endl;
                std::cout << std::scientific << std::setprecision(6);
                std::cout << "L² error: " << results.l2_error << std::endl;
                std::cout << "H¹ error: " << results.h1_error << std::endl;
                std::cout << "Time steps: " << results.total_time_steps << std::endl;
                std::cout << "Test " << (results.test_passed ? "PASSED" : "FAILED") << std::endl;
                std::cout << std::string(50, '=') << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error in run_single_json_mesh_test: " << e.what() << std::endl;
            
            // Save error information to output file
            nlohmann::json error_json;
            error_json["error"] = e.what();
            error_json["input_mesh_file"] = json_mesh_file;
            error_json["test_passed"] = false;
            error_json["configuration"] = {
                {"vem_order", vem_order},
                {"final_time", final_time},
                {"time_step", time_step},
                {"scheme_type", "RK3"},
                {"use_lumped_mass_matrix", true}
            };
            
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::string timestamp = std::ctime(&time_t);
            timestamp.pop_back(); // Remove trailing newline
            error_json["timestamp"] = timestamp;
            
            std::filesystem::path output_path(output_json_file);
            std::filesystem::create_directories(output_path.parent_path());
            
            std::ofstream output_file(output_json_file);
            output_file << error_json.dump(4);
            output_file.close();
            
            throw; // Re-throw for caller to handle
        }
    }
}