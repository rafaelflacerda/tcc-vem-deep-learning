#ifndef TIME_SCHEME_HPP
#define TIME_SCHEME_HPP


#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace utils {

    class time_scheme {
    public:
        /**
         * @brief Time integration scheme types
         */
        enum class SchemeType {
            BACKWARD_EULER,
            CRANK_NICOLSON,
            FORWARD_EULER,
            RK3,
            RK4
        };

        /**
         * @brief Solver configuration options
         */
        enum class SolverType {
            DIRECT, // Direct solver (SparseLU)
            ITERATIVE, // Iterative solver (BiCGSTAB)
        };

        /**
         * @brief Integration statistics
         */
        struct Statistics {
            int total_steps = 0;
            int successful_steps = 0;
            int failed_steps = 0;
            double avg_solve_time = 0.0;
        };

        // ============================================================================
        // CONSTRUCTORS AND DESTRUCTOR
        // ============================================================================

        time_scheme(
            const Eigen::SparseMatrix<double>& M_h, // Mass matrix
            const Eigen::SparseMatrix<double>& K_h // Stiffness matrix
        );

        ~time_scheme() = default; // Destructor

        // ============================================================================
        // INITIALIZATION AND SETUP
        // ============================================================================

        /**
         * @brief Initialize the time scheme with new matrices
         * @param M_h Mass matrix
         * @param K_h Stiffness matrix
         */
        void initialize(
            const Eigen::SparseMatrix<double>& M_h,
            const Eigen::SparseMatrix<double>& K_h
        );

        /**
         * @brief Set the time parameters
         * @param dt Time step size
         * @param scheme Time integration scheme type
         */
        void setup_time_parameters(
            double dt, 
            SchemeType scheme
        ) {
            if (dt <= 0.0) {
                throw std::invalid_argument("Time step must be positive");
            }

            dt_ = dt;
            scheme_type_ = scheme;

            is_initialized_ = true;

            if (debug_mode_) {
                debug_print("Time parameters set:");
                debug_print("  Time step: " + std::to_string(dt_));
            }
        }

        // ============================================================================
        // SOLVER CONFIGURATION
        // ============================================================================

        void configure_solver(
            SolverType solver_type,
            double tolerance,
            int max_iterations
        ){
            solver_type_ = solver_type;
            solver_tolerance_ = tolerance;
            solver_max_iter_ = max_iterations;

            if (debug_mode_) {
                debug_print("Solver configured:");
                debug_print("  Type: " + std::string(solver_type_ == SolverType::DIRECT ? "Direct" : "Iterative"));
                debug_print("  Tolerance: " + std::to_string(solver_tolerance_));
                debug_print("  Max iterations: " + std::to_string(solver_max_iter_));
            }
        }

        /**
         * @brief Set debug mode for detailed output
         * @param enable_debug True to enable debug output, false to disable
         */
        void set_debug_mode(bool enable_debug) {
            debug_mode_ = enable_debug;
            if (debug_mode_) {
                debug_print("Debug mode enabled");
            }
        }

        /**
         * @brief Get the current statistics
         * @return Reference to the statistics object
         */
        const Statistics& get_statistics() const {
            return statistics_;
        }

        /**
         * @brief Check if scheme is unconditionally stable
         * @return True if the scheme is stable, false otherwise
         */
        bool is_stable() const {
            switch (scheme_type_) {
                case SchemeType::BACKWARD_EULER:
                case SchemeType::CRANK_NICOLSON:
                    return true;
                default:
                    return false;
            }
        }

        /**
         * @brief Get the order of the time scheme
         * @return Order of the scheme
         */
        int get_order() const {
            switch (scheme_type_) {
                case SchemeType::BACKWARD_EULER:
                    return 1;
                case SchemeType::CRANK_NICOLSON:
                    return 2;
                case SchemeType::FORWARD_EULER:
                    return 1;
                default:
                    return 0;
            }
        }

        /**
         * @brief Get the name of the time scheme
         * @return Name of the scheme
         */
        std::string get_scheme_name() const {
            switch (scheme_type_) {
                case SchemeType::BACKWARD_EULER:
                    return "Backward Euler";
                case SchemeType::CRANK_NICOLSON:
                    return "Crank-Nicolson";
                case SchemeType::FORWARD_EULER:
                    return "Forward Euler";
                default:
                    return "Unknown";
            }
        }

        // ============================================================================
        // TIME INTEGRATION EXECUTION
        // ============================================================================

        /**
         * @brief Perform single time step - PURE ALGORITHM
         * @param U_current Current solution vector
         * @param U_previous Previous solution vector
         * @param F_current Current right-hand side vector
         * @return True if the step was successful, false otherwise
         */
        bool step(
            Eigen::VectorXd& U_current,
            const Eigen::VectorXd& U_previous,
            const Eigen::VectorXd& F_current
        );

        /**
         * @brief Perform single time step with custom matrices (for Crank-Nicolson)
         * @param U_current Current solution vector
         * @param U_previous Previous solution vector
         * @param F_current Current right-hand side vector
         * @param F_previous Previous right-hand side vector
         * @return True if the step was successful, false otherwise
         */
        bool step_two_level(
            Eigen::VectorXd& U_current,
            const Eigen::VectorXd& U_previous,
            const Eigen::VectorXd& F_current,
            const Eigen::VectorXd& F_previous
        );

        /**
         * @brief Perform single time step for RK scheme
         * Centralized RK scheme with different stages
         * du/dt = M_h^(-1) [f(t) - K_h u] = F(u, t)
         * 
         * @param U_current Current solution vector
         * @param U_previous Previous solution vector
         * @param f_function Function to compute the right-hand side vector (source term)
         * @param current_time Current time
         * @param use_shu_osher_from Use Shu-Osher from scheme
         * @return True if the step was successful, false otherwise
         */
        bool step_rk(
            Eigen::VectorXd& U_current,
            const Eigen::VectorXd& U_previous,
            const std::function<Eigen::VectorXd(double)>& f_function,
            double current_time,
            bool use_shu_osher_from = false
        );

        /**
         * @brief Perform single time step for RK3 scheme
         * du/dt = M_h^(-1) [f(t) - K_h u] = F(u, t)
         * k1 = dt * F(u^n, t^n)
         * k2 = dt * F(u^n + k1/2, t^n + dt/2)
         * k3 = dt * F(u^n - k1 + 2*k2, t^n + dt)
         * u^(n+1) = u^n + (k1 + 4*k2 + k3)/6
         * 
         * @param U_current Current solution vector
         * @param U_previous Previous solution vector
         * @param f_function Function to compute the right-hand side vector (source term)
         * @param current_time Current time
         * @param use_shu_osher_from Use Shu-Osher from scheme
         * @return True if the step was successful, false otherwise
         */
        bool step_rk3(
            Eigen::VectorXd& U_current,
            const Eigen::VectorXd& U_previous,
            const std::function<Eigen::VectorXd(double)>& f_function,
            double current_time,
            bool use_shu_osher_from = false
        );

        /**
         * @brief Perform single time step for RK4 scheme
         * du/dt = M_h^(-1) [f(t) - K_h u] = F(u, t)
         * k1 = dt * F(u^n, t^n)
         * k2 = dt * F(u^n + k1/2, t^n + dt/2)
         * k3 = dt * F(u^n + k2/2, t^n + dt/2)
         * k4 = dt * F(u^n + k3, t^n + dt)
         * u^(n+1) = u^n + (k1 + 2*k2 + 2*k3 + k4)/6
         * 
         * @param U_current Current solution vector
         * @param U_previous Previous solution vector
         * @param f_function Function to compute the right-hand side vector (source term)
         * @param current_time Current time
         * @param use_shu_osher_from Use Shu-Osher from scheme
         * @return True if the step was successful, false otherwise
         */
        bool step_rk4(
            Eigen::VectorXd& U_current,
            const Eigen::VectorXd& U_previous,
            const std::function<Eigen::VectorXd(double)>& f_function,
            double current_time,
            bool use_shu_osher_from = false
        );


    private:
        // ============================================================================
        // PRIVATE MEMBER VARIABLES
        // ============================================================================

        // System matrices (using references to avoid copying)
        const Eigen::SparseMatrix<double>& M_h_;
        const Eigen::SparseMatrix<double>& K_h_;

        // Time integration parameters
        double dt_;
        SchemeType scheme_type_;

        // Solver configuration
        SolverType solver_type_;
        double solver_tolerance_;
        int solver_max_iter_;

        // Debug and state
        bool debug_mode_;
        bool is_initialized_;

        // Statistics
        Statistics statistics_;

        // ============================================================================
        // UTILITY METHODS
        // ============================================================================

        /**
         * @brief Solve the linear system Ax = b using the configured solver
         * @param A Matrix of the linear system
         * @param b Right-hand side vector of the linear system
         * @param x Solution vector
         * @return True if the system was solved successfully, false otherwise
         */
        bool solve_linear_system(
            const Eigen::SparseMatrix<double>& A,
            const Eigen::VectorXd& b,
            Eigen::VectorXd& x
        ) const;

        /**
         * @brief Update integration statistics
         * @param step_successful Whether the step was successful
         * @param solve_time Time taken for linear solve
         */
        void update_statistics(bool step_successful, double solve_time);

        /**
         * @brief Compute F(u,t) = M_h^(-1) * [f(t) - K_h * u] for RK schemes
         * @param u Current solution vector
         * @param f_rhs Right-hand side vector f(t)
         * @param result Output vector F(u,t)
         * @return True if computation successful
         */
        bool compute_ode_rhs(
            const Eigen::VectorXd& u,
            const Eigen::VectorXd& f_rhs,
            Eigen::VectorXd& result
        ) const;


        // Debugging
        void debug_print(const std::string& message) const;


    };
}

#endif
