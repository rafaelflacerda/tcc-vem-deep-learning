#include "utils/time_scheme.hpp"
#include <chrono>

namespace utils {
    // ============================================================================
    // CONSTRUCTORS AND DESTRUCTOR
    // ============================================================================

    time_scheme::time_scheme(
        const Eigen::SparseMatrix<double>& M_h,
        const Eigen::SparseMatrix<double>& K_h
    )
        : M_h_(M_h),
          K_h_(K_h),
          scheme_type_(SchemeType::BACKWARD_EULER),
          solver_type_(SolverType::DIRECT),
          solver_tolerance_(1e-6),
          solver_max_iter_(1000),
          debug_mode_(false),
          is_initialized_(true)
    {
        // Validate matrix dimensions immediately
        if (M_h.rows() != M_h.cols() || K_h.rows() != K_h.cols()) {
            throw std::invalid_argument("Matrices must be square");
        }
        if (M_h.rows() != K_h.rows()) {
            throw std::invalid_argument("Mass and stiffness matrices must have same dimensions");
        }

        if (debug_mode_) {
            debug_print("Time scheme constructed with matrices");
            debug_print("Matrix size: " + std::to_string(M_h.rows()) + "x" + std::to_string(M_h.cols()));
        }
    }

    // ============================================================================
    // TIME INTEGRATION EXECUTION
    // ============================================================================

    bool time_scheme::step(
        Eigen::VectorXd& U_current,
        const Eigen::VectorXd& U_previous,
        const Eigen::VectorXd& F_rhs
    ){
        // Check that time parameters are set
        if (!is_initialized_) {
            debug_print("Error: Time parameters not set. Call setup_time_parameters() first.");
            return false;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = false;

        switch (scheme_type_) {
            case SchemeType::BACKWARD_EULER: {
                // (M + dt*K)*U^n = M*U^{n-1} + dt*F
                Eigen::SparseMatrix<double> lhs_matrix = M_h_ + dt_ * K_h_;
                Eigen::VectorXd rhs_vector = M_h_ * U_previous + dt_ * F_rhs;
                success = solve_linear_system(lhs_matrix, rhs_vector, U_current);
                break;
            }
            case SchemeType::FORWARD_EULER: {
                Eigen::VectorXd  rhs_vector =  M_h_ * U_previous + dt_ * (-K_h_ * U_previous + F_rhs);
                double min_diag = M_h_.diagonal().minCoeff();
                if (debug_mode_) {
                    debug_print("RHS norm: " + std::to_string(rhs_vector.norm()));
                    double min_diag = M_h_.diagonal().minCoeff();
                    debug_print("M_h_ min diag: " + std::to_string(min_diag));
                }
                success = solve_linear_system(M_h_, rhs_vector, U_current);
                break;
            }
            default:
                debug_print("Error: Unsupported scheme for single-level step");
                return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double solve_time = duration.count() / 1000.0; // Convert to ms

        update_statistics(success, solve_time);

        return success;
    }

    bool time_scheme::step_two_level(
        Eigen::VectorXd& U_current,
        const Eigen::VectorXd& U_previous,
        const Eigen::VectorXd& F_current,
        const Eigen::VectorXd& F_previous
    ){
        // Check that time parameters are set
        if (!is_initialized_) {
            debug_print("Error: Time parameters not set. Call setup_time_parameters() first.");
            return false;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = false;

        switch (scheme_type_) {
            case SchemeType::CRANK_NICOLSON: {
                // (M + 0.5*dt*K)*U^n = (M - 0.5*dt*K)*U^{n-1} + 0.5*dt*(F^n + F^{n-1})
                Eigen::SparseMatrix<double> lhs_matrix = M_h_ + 0.5 * dt_ * K_h_;
                Eigen::SparseMatrix<double> rhs_matrix = M_h_ - 0.5 * dt_ * K_h_;
                Eigen::VectorXd rhs_vector = rhs_matrix * U_previous + 0.5 * dt_ * (F_current + F_previous);
                success = solve_linear_system(lhs_matrix, rhs_vector, U_current);
                break;
            }
            default:
                debug_print("Error: Unsupported scheme for two-level step");
                return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double solve_time = duration.count() / 1000.0; // Convert to ms

        update_statistics(success, solve_time);

        return success;
    }

    bool time_scheme::step_rk(
        Eigen::VectorXd& U_current,
        const Eigen::VectorXd& U_previous,
        const std::function<Eigen::VectorXd(double)>& f_function,
        double current_time,
        bool use_shu_osher_from
    ) {
        switch (scheme_type_) {
            case SchemeType::RK3:
                return step_rk3(U_current, U_previous, f_function, current_time, use_shu_osher_from);
            case SchemeType::RK4:
                return step_rk4(U_current, U_previous, f_function, current_time, use_shu_osher_from);
            default:
                debug_print("Error: Unsupported scheme for RK step");
                return false;
        }
    }

    bool time_scheme::step_rk3(
        Eigen::VectorXd& U_current,
        const Eigen::VectorXd& U_previous,
        const std::function<Eigen::VectorXd(double)>& f_function,
        double current_time,
        bool use_shu_osher_from
    ) {
        // Shu-Osher from scheme
        if (use_shu_osher_from) {
            Eigen::VectorXd u0, u1, u2;
            Eigen::VectorXd L0, L1, L2;
            
            // Stage 0: Initial state
            u0 = U_previous;
            
            // Stage 1: u^(1) = u^(0) + dt * L(u^(0))
            Eigen::VectorXd f0 = f_function(current_time);
            if (!compute_ode_rhs(u0, f0, L0)) {
                return false;
            }
            u1 = u0 + dt_ * L0;
            
            // Stage 2: u^(2) = (3/4)*u^(0) + (1/4)*u^(1) + (1/4)*dt*L(u^(1))
            Eigen::VectorXd f1 = f_function(current_time + dt_);  // c1 = 1
            if (!compute_ode_rhs(u1, f1, L1)) {
                return false;
            }
            u2 = 0.75 * u0 + 0.25 * u1 + 0.25 * dt_ * L1;
            
            // Stage 3: u^(3) = (1/3)*u^(0) + (2/3)*u^(2) + (2/3)*dt*L(u^(2))
            Eigen::VectorXd f2 = f_function(current_time + 0.5 * dt_);  // c2 = 1/2
            if (!compute_ode_rhs(u2, f2, L2)) {
                return false;
            }
            U_current = (1.0/3.0) * u0 + (2.0/3.0) * u2 + (2.0/3.0) * dt_ * L2;
            
            return true;
        }

        Eigen::VectorXd k1, k2, k3;
        Eigen::VectorXd temp_vector;

        // Stage 1: k1 = dt * F(u^n, t^n)
        Eigen::VectorXd F_current = f_function(current_time);
        if (!compute_ode_rhs(U_previous, F_current, k1)) return false;
        k1 *= dt_;

        // Stage 2: k2 = dt * F(u^n + k1/2, t^n + dt/2)
        temp_vector = U_previous + k1 / 2.0;
        Eigen::VectorXd F_half = f_function(current_time + dt_ / 2.0);
        if (!compute_ode_rhs(temp_vector, F_half, k2)) return false;
        k2 *= dt_;

        // Stage 3: k3 = dt * F(u^n - k1 + 2*k2, t^n + dt)
        temp_vector = U_previous - k1 + 2.0 * k2;
        Eigen::VectorXd F_full = f_function(current_time + dt_);
        if (!compute_ode_rhs(temp_vector, F_full, k3)) return false;
        k3 *= dt_;

        // Update solution
        U_current = U_previous + (k1 + 4.0 * k2 + k3) / 6.0;

        return true;
    }

    bool time_scheme::step_rk4(
        Eigen::VectorXd& U_current,
        const Eigen::VectorXd& U_previous,
        const std::function<Eigen::VectorXd(double)>& f_function,
        double current_time,
        bool use_shu_osher_from
    ) {
        Eigen::VectorXd k1, k2, k3, k4;
        Eigen::VectorXd temp_vector;

        // Stage 1: k1 = dt * F(u^n, t^n)
        Eigen::VectorXd F_current = f_function(current_time);
        if (!compute_ode_rhs(U_previous, F_current, k1)) return false;
        k1 *= dt_;

        // Stage 2: k2 = dt * F(u^n + k1/2, t^n + dt/2)
        temp_vector = U_previous + 0.5 * k1;
        Eigen::VectorXd F_half = f_function(current_time + 0.5 * dt_);
        if (!compute_ode_rhs(temp_vector, F_half, k2)) return false;
        k2 *= dt_;

        // Stage 3: k3 = dt * F(u^n + k2/2, t^n + dt/2)
        temp_vector = U_previous + 0.5 * k2;
        F_half = f_function(current_time + 0.5 * dt_);
        if (!compute_ode_rhs(temp_vector, F_half, k3)) return false;
        k3 *= dt_;

        // Stage 4: k4 = dt * F(u^n + k3, t^n + dt) 
        temp_vector = U_previous + k3;
        Eigen::VectorXd F_full = f_function(current_time + dt_);
        if (!compute_ode_rhs(temp_vector, F_full, k4)) return false;
        k4 *= dt_;

        // Update solution
        U_current = U_previous + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        return true;
    }

    // ============================================================================
    // UTILITY METHODS
    // ============================================================================

    bool time_scheme::solve_linear_system(
        const Eigen::SparseMatrix<double>& A,
        const Eigen::VectorXd& b,
        Eigen::VectorXd& x
    ) const {

        if (solver_type_ == SolverType::DIRECT) {
            // Direct solver
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);

            if (solver.info() != Eigen::Success) {
                if (debug_mode_) {
                    debug_print("Matrix factorization failed");
                }
                return false;
            }

            x = solver.solve(b);

            if (solver.info() != Eigen::Success) {
                if (debug_mode_) {
                    debug_print("Linear system solve failed");
                }
                return false;
            }

        } else {
            // Iterative solver
            Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::DiagonalPreconditioner<double>> solver;
            solver.setMaxIterations(solver_max_iter_);
            solver.setTolerance(solver_tolerance_);
            solver.compute(A);

            if (solver.info() != Eigen::Success) {
                if (debug_mode_) {
                    debug_print("Matrix preconditioning failed");
                }
                return false;
            }

            x = solver.solve(b);

            if (solver.info() != Eigen::Success) {
                if (debug_mode_) {
                    debug_print("Iterative solver failed. Iterations: " + std::to_string(solver.iterations()) + 
                               ", Error: " + std::to_string(solver.error()));
                }
                return false;
            }

            if (debug_mode_ && solver.iterations() > solver_max_iter_ * 0.8) {
                debug_print("Solver required many iterations: " + std::to_string(solver.iterations()));
            }
        }

        return true;
    }

    bool time_scheme::compute_ode_rhs(
        const Eigen::VectorXd& u,
        const Eigen::VectorXd& f_rhs,
        Eigen::VectorXd& result
    ) const {
        // Compute K_h * u
        Eigen::VectorXd Ku = K_h_ * u;
        
        // Compute f(t) - K_h * u
        Eigen::VectorXd rhs = f_rhs - Ku;
        
        // Solve M_h * result = rhs  =>  result = M_h^(-1) * rhs
        return solve_linear_system(M_h_, rhs, result);
    }

    void time_scheme::update_statistics(bool step_successful, double solve_time) {
        statistics_.total_steps++;
        if (step_successful) {
            statistics_.successful_steps++;
        } else {
            statistics_.failed_steps++;
        }

        // Update average solve time using exponential moving average
        double alpha = 1.0 / statistics_.total_steps;
        statistics_.avg_solve_time = (1.0 - alpha) * statistics_.avg_solve_time + alpha * solve_time;
    }

    void time_scheme::debug_print(const std::string& message) const {
        if (debug_mode_) {
            std::cout << "[DEBUG] Time Scheme: " << message << std::endl;
        }
    }

}