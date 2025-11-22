# Parabolic Solver User Guide: Complete Implementation Pipeline

## Overview

This document provides a comprehensive step-by-step guide for using the Virtual Element Method (VEM) parabolic solver. The guide covers the complete pipeline from mesh setup to time integration results, including theoretical background, implementation details, and practical examples.

## Mathematical Background

### Parabolic Problem Formulation

The parabolic solver is designed to solve time-dependent partial differential equations of the form:

$$\frac{\partial u}{\partial t} - \nabla \cdot (D \nabla u) = f(\mathbf{x}, t) \quad \text{in } \Omega \times (0, T]$$

with appropriate initial and boundary conditions:

$$u(\mathbf{x}, 0) = u_0(\mathbf{x}) \quad \text{in } \Omega$$

$$u(\mathbf{x}, t) = g(\mathbf{x}, t) \quad \text{on } \Gamma_D \times (0, T]$$

$$D \nabla u \cdot \mathbf{n} = h(\mathbf{x}, t) \quad \text{on } \Gamma_N \times (0, T]$$

where:

- $\Omega \subset \mathbb{R}^2$ is the computational domain
- $u(\mathbf{x}, t)$ is the unknown solution
- $D$ is the diffusion coefficient (assumed constant = 1)
- $f(\mathbf{x}, t)$ is the source term
- $u_0(\mathbf{x})$ is the initial condition
- $g(\mathbf{x}, t)$ and $h(\mathbf{x}, t)$ are boundary conditions

### VEM Semi-Discretization

The VEM semi-discretization leads to the system of ODEs:

$$M_h \frac{d\mathbf{u}}{dt} + K_h \mathbf{u} = \mathbf{f}(t)$$

where:

- $M_h \in \mathbb{R}^{N \times N}$ is the VEM mass matrix
- $K_h \in \mathbb{R}^{N \times N}$ is the VEM stiffness matrix
- $\mathbf{u}(t) \in \mathbb{R}^N$ is the vector of degrees of freedom
- $\mathbf{f}(t) \in \mathbb{R}^N$ is the load vector
- $N$ is the total number of degrees of freedom

## Complete Implementation Pipeline

### Step 1: Mesh Setup and Geometry Definition

The first step is to define the computational mesh and geometric properties.

#### 1.1 Node Coordinates Definition

```cpp
// Define mesh nodes (coordinates)
Eigen::MatrixXd nodes(n_nodes, 2);
nodes << x1, y1,    // Node 0
         x2, y2,    // Node 1
         ...
         xn, yn;    // Node n-1
```

**Example: Unit Square with 2×2 Subdivision**

```cpp
Eigen::MatrixXd nodes(9, 2);
nodes << 0.0, 0.0,   0.5, 0.0,   1.0, 0.0,     // Bottom row
         0.0, 0.5,   0.5, 0.5,   1.0, 0.5,     // Middle row
         0.0, 1.0,   0.5, 1.0,   1.0, 1.0;     // Top row
```

#### 1.2 Element Connectivity Definition

```cpp
// Define element connectivity (node indices)
Eigen::MatrixXi elements(n_elements, n_vertices_per_element);
elements << v1, v2, v3, v4,    // Element 0
            v5, v6, v7, v8,    // Element 1
            ...
```

**Example: 4 Quadrilateral Elements**

```cpp
Eigen::MatrixXi elements(4, 4);
elements << 0, 1, 4, 3,    // Bottom-left quad
            1, 2, 5, 4,    // Bottom-right quad
            3, 4, 7, 6,    // Top-left quad
            4, 5, 8, 7;    // Top-right quad
```

#### 1.3 Polynomial Order Selection

```cpp
int polynomial_order = 1;  // Linear VEM (k=1)
// int polynomial_order = 2;  // Quadratic VEM (k=2)
```

**VEM Order Implications:**

- **k=1 (Linear)**: DOFs at vertices only
- **k=2 (Quadratic)**: DOFs at vertices + edge moments + interior moments
- **k≥3**: Higher-order polynomials with more DOF types

### Step 2: Solver Initialization and System Assembly

#### 2.1 Parabolic Solver Construction

```cpp
#include "solver/parabolic.hpp"

// Create the VEM parabolic solver
solver::parabolic vem_solver(polynomial_order, nodes, elements);

// Print solver information
std::cout << "VEM solver created with " << vem_solver.get_n_dofs() << " DOFs" << std::endl;
vem_solver.print_system_information();
```

**Constructor Operations:**

- Initializes DOF numbering for the entire mesh
- Sets up element-to-global DOF mapping
- Allocates memory for global matrices
- Computes polynomial space dimension $N_k$

#### 2.2 System Matrix Assembly

```cpp
// Assemble the complete VEM system
std::cout << "Assembling VEM system..." << std::endl;
vem_solver.assemble_system();
std::cout << "VEM system assembled successfully" << std::endl;
```

**Assembly Process:**

1. **Element Loop**: Process each element $K \in \mathcal{T}_h$
2. **Local Matrices**: Compute $M_K$ and $K_K$ for each element
3. **Global Assembly**: Accumulate local contributions into $M_h$ and $K_h$
4. **Compression**: Optimize sparse matrix storage

**VEM Local Matrix Formulation:**

- **Mass Matrix**: $M_K = (P^0)^T M_{\text{poly}} P^0 + \beta_K S_K$
- **Stiffness Matrix**: $A_K = (P^∇)^T K_{\text{poly}} P^∇ + \alpha_K S_K$

where $P^0$ and $P^∇$ are projection operators and $S_K$ is the stabilization matrix.

### Step 3: Boundary Conditions Setup

#### 3.1 Boundary Handler Initialization

```cpp
#include "utils/boundary.hpp"

// Initialize boundary condition handler
utils::boundary boundary_manager(nodes, elements, polynomial_order);
std::cout << "✓ Boundary handler initialized" << std::endl;
```

#### 3.2 Dirichlet Boundary Conditions

```cpp
// Add Dirichlet boundary condition: u = g(x,t) on Γ_D
boundary_manager.add_dirichlet(
    [](const Eigen::Vector2d& x, double t) -> double {
        return g_function(x, t);  // Your boundary function
    },
    utils::boundary::BoundaryRegion::ENTIRE,  // or BOTTOM, TOP, LEFT, RIGHT
    "Dirichlet_BC_Name"
);
```

**Example: Homogeneous Dirichlet Conditions**

```cpp
boundary_manager.add_dirichlet(
    [](const Eigen::Vector2d& x, double t) -> double {
        return 0.0;  // u = 0 on boundary
    },
    utils::boundary::BoundaryRegion::ENTIRE,
    "Homogeneous_Dirichlet"
);
```

**Example: Time-Dependent Dirichlet Conditions**

```cpp
boundary_manager.add_dirichlet(
    [](const Eigen::Vector2d& x, double t) -> double {
        return std::sin(M_PI * x(0)) * std::exp(-t);  // u = sin(πx)e^(-t)
    },
    utils::boundary::BoundaryRegion::BOTTOM,
    "Time_Dependent_Dirichlet"
);
```

#### 3.3 Neumann Boundary Conditions

```cpp
// Add Neumann boundary condition: ∂u/∂n = h(x,t) on Γ_N
boundary_manager.add_neumann(
    [](const Eigen::Vector2d& x, double t) -> double {
        return h_function(x, t);  // Your flux function
    },
    utils::boundary::BoundaryRegion::TOP,
    "Neumann_BC_Name"
);
```

#### 3.4 Boundary Condition Application

```cpp
// Initialize load vector
vem_solver.F_h = Eigen::VectorXd::Zero(vem_solver.get_n_dofs());

// Apply boundary conditions to system
double current_time = 0.0;
boundary_manager.apply_dirichlet_conditions(
    vem_solver.K_h,    // Stiffness matrix (modified)
    vem_solver.M_h,    // Mass matrix (modified)
    vem_solver.F_h,    // Load vector (modified)
    current_time       // Current time
);
```

**Boundary Condition Implementation:**

- **Dirichlet**: Modifies matrix rows/columns and RHS vector
- **Neumann**: Adds flux contributions to load vector
- **Essential vs Natural**: Dirichlet (essential) vs Neumann (natural) conditions

### Step 4: Time Integration Setup

#### 4.1 Time Scheme Initialization

```cpp
#include "utils/time_scheme.hpp"

// Create time integrator with VEM matrices
utils::time_scheme time_integrator(vem_solver.M_h, vem_solver.K_h);
```

#### 4.2 Time Parameters Configuration

```cpp
// Setup time integration parameters
double T_final = 1.0;    // Final time
double dt = 0.01;        // Time step size

time_integrator.setup_time_parameters(
    dt,
    utils::time_scheme::SchemeType::BACKWARD_EULER  // or CRANK_NICOLSON
);
```

**Time Scheme Options:**

- **BACKWARD_EULER**: First-order, A-stable, dissipative
- **CRANK_NICOLSON**: Second-order, A-stable, energy-preserving

#### 4.3 Linear Solver Configuration

```cpp
// Configure linear solver
time_integrator.configure_solver(
    utils::time_scheme::SolverType::DIRECT,  // or ITERATIVE
    1e-12,                                   // Solver tolerance
    1000                                     // Maximum iterations
);

// Enable debug output (optional)
time_integrator.set_debug_mode(true);
```

**Solver Options:**

- **DIRECT**: SparseLU factorization, robust but memory-intensive
- **ITERATIVE**: BiCGSTAB method, memory-efficient for large systems

### Step 5: Problem Functions Definition

#### 5.1 Initial Condition Function

```cpp
// Define initial condition u₀(x,y)
auto initial_condition = [](const Eigen::Vector2d& x) -> double {
    return u0_function(x);
};
```

**Examples:**

```cpp
// Smooth initial condition
auto initial_condition = [](const Eigen::Vector2d& x) -> double {
    return std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
};

// Gaussian initial condition
auto initial_condition = [](const Eigen::Vector2d& x) -> double {
    double r2 = (x(0) - 0.5) * (x(0) - 0.5) + (x(1) - 0.5) * (x(1) - 0.5);
    return std::exp(-10.0 * r2);
};
```

#### 5.2 Source Term Function

```cpp
// Define source term f(x,y,t)
auto source_function = [](const Eigen::Vector2d& x, double t) -> double {
    return f_function(x, t);
};
```

**Examples:**

```cpp
// Homogeneous source (heat equation)
auto source_function = [](const Eigen::Vector2d& x, double t) -> double {
    return 0.0;
};

// Time-dependent source
auto source_function = [](const Eigen::Vector2d& x, double t) -> double {
    return std::sin(M_PI * x(0)) * std::cos(M_PI * x(1)) * std::exp(-t);
};
```

#### 5.3 Exact Solution (for verification)

```cpp
// Define exact solution (if known) for error analysis
auto exact_solution = [](const Eigen::Vector2d& x, double t) -> double {
    return exact_function(x, t);
};
```

### Step 6: Initial Conditions Application

#### 6.1 Solution Vector Initialization

```cpp
// Initialize solution vectors
Eigen::VectorXd U_current(vem_solver.get_n_dofs());
Eigen::VectorXd U_previous(vem_solver.get_n_dofs());
```

#### 6.2 VEM Interpolation of Initial Conditions

**Method 1: Direct Vertex Evaluation (Linear VEM)**

```cpp
// For linear VEM (k=1), DOFs are vertex values
for (int i = 0; i < nodes.rows(); ++i) {
    Eigen::Vector2d vertex = nodes.row(i);
    U_current(i) = initial_condition(vertex);
}
```

**Method 2: VEM Interpolation Operator (General)**

```cpp
// Use VEM interpolation operator I_h
vem_solver.set_initial_conditions(initial_condition);
U_current = vem_solver.U_h;  // Get interpolated initial condition
```

#### 6.3 Initial Condition Verification

```cpp
U_previous = U_current;
std::cout << "Initial conditions set. Initial L2 norm: " << U_current.norm() << std::endl;
```

### Step 7: Time Integration Loop

#### 7.1 Time Loop Setup

```cpp
std::cout << "\n=== Starting Time Integration ===" << std::endl;

int num_time_steps = static_cast<int>(std::ceil(T_final / dt));
auto start_time = std::chrono::high_resolution_clock::now();

// Storage for time-dependent quantities
Eigen::VectorXd F_previous = Eigen::VectorXd::Zero(vem_solver.get_n_dofs());
```

#### 7.2 Main Time Integration Loop

```cpp
for (int step = 1; step <= num_time_steps; ++step) {
    double current_time = step * dt;

    // Progress monitoring
    if (step % (num_time_steps / 10) == 0 || step == 1) {
        std::cout << "Step " << step << "/" << num_time_steps
                  << ", t = " << std::fixed << std::setprecision(3) << current_time
                  << ", ||U|| = " << std::scientific << U_current.norm() << std::endl;
    }

    // Step 7.2.1: Assemble load vector at current time
    vem_solver.assemble_load_vector(source_function, current_time);
    Eigen::VectorXd F_current = vem_solver.get_load_vector();

    // Step 7.2.2: Apply boundary conditions
    Eigen::SparseMatrix<double> M_h_bc = vem_solver.M_h;
    Eigen::SparseMatrix<double> K_h_bc = vem_solver.K_h;
    Eigen::VectorXd F_h_bc = F_current;

    boundary_manager.apply_dirichlet_conditions(K_h_bc, M_h_bc, F_h_bc, current_time);

    // Step 7.2.3: Update time integrator (if matrices changed)
    if (step == 1) {
        time_integrator.initialize(M_h_bc, K_h_bc);
    }

    // Step 7.2.4: Perform time step
    bool success;
    if (time_integrator.get_scheme_name() == "Crank-Nicolson") {
        success = time_integrator.step_two_level(U_current, U_previous, F_current, F_previous);
    } else {
        success = time_integrator.step(U_current, U_previous, F_h_bc);
    }

    // Step 7.2.5: Check for convergence failure
    if (!success) {
        std::cerr << "Time step " << step << " failed!" << std::endl;
        break;
    }

    // Step 7.2.6: Update for next iteration
    U_previous = U_current;
    F_previous = F_current;
}
```

#### 7.3 Time Integration Schemes

**Backward Euler Implementation:**

- **Equation**: $(M_h + \Delta t K_h) \mathbf{u}^n = M_h \mathbf{u}^{n-1} + \Delta t \mathbf{f}^n$
- **Properties**: First-order accurate, unconditionally stable, dissipative
- **Usage**: `time_integrator.step(U_current, U_previous, F_current)`

**Crank-Nicolson Implementation:**

- **Equation**: $(M_h + \frac{\Delta t}{2} K_h) \mathbf{u}^n = (M_h - \frac{\Delta t}{2} K_h) \mathbf{u}^{n-1} + \frac{\Delta t}{2}(\mathbf{f}^n + \mathbf{f}^{n-1})$
- **Properties**: Second-order accurate, unconditionally stable, energy-preserving
- **Usage**: `time_integrator.step_two_level(U_current, U_previous, F_current, F_previous)`

### Step 8: Results Analysis and Post-Processing

#### 8.1 Computation Time Analysis

```cpp
auto end_time = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

std::cout << "\n=== Time Integration Complete ===" << std::endl;
std::cout << "Total computation time: " << duration.count() << " ms" << std::endl;
```

#### 8.2 Solution Quality Assessment

```cpp
std::cout << "\n=== Results Analysis ===" << std::endl;

// Final solution properties
double final_norm = U_current.norm();
std::cout << "Final solution norm: " << std::scientific << final_norm << std::endl;

// Check for conservation properties (if applicable)
double initial_norm = /* store from initial conditions */;
double conservation_ratio = final_norm / initial_norm;
std::cout << "Conservation ratio: " << conservation_ratio << std::endl;
```

#### 8.3 Error Analysis (if exact solution available)

```cpp
// Compute error against exact solution
Eigen::VectorXd U_exact(vem_solver.get_n_dofs());
for (int i = 0; i < nodes.rows(); ++i) {
    Eigen::Vector2d vertex = nodes.row(i);
    U_exact(i) = exact_solution(vertex, T_final);
}

// Error metrics
double error_l2 = (U_current - U_exact).norm();
double exact_norm = U_exact.norm();
double relative_error = error_l2 / exact_norm;

std::cout << "L2 error: " << std::scientific << error_l2 << std::endl;
std::cout << "Exact solution norm: " << std::scientific << exact_norm << std::endl;
std::cout << "Relative error: " << std::scientific << relative_error << std::endl;
```

#### 8.4 Integration Statistics

```cpp
// Display solver performance statistics
const auto& stats = time_integrator.get_statistics();
std::cout << "\n=== Time Integration Statistics ===" << std::endl;
std::cout << "Total steps: " << stats.total_steps << std::endl;
std::cout << "Successful steps: " << stats.successful_steps << std::endl;
std::cout << "Failed steps: " << stats.failed_steps << std::endl;
std::cout << "Average solve time: " << std::fixed << std::setprecision(3)
          << stats.avg_solve_time << " ms" << std::endl;
```

#### 8.5 Solution Visualization

```cpp
// Print solution values at selected points
std::cout << "\n=== Solution Values at Selected Nodes ===" << std::endl;
std::vector<int> sample_nodes = {0, 4, 8}; // Example: corners and center
for (int node : sample_nodes) {
    if (node < nodes.rows()) {
        Eigen::Vector2d pos = nodes.row(node);
        double computed = U_current(node);
        std::cout << "Node " << node << " at (" << pos(0) << ", " << pos(1) << "): "
                  << "u = " << std::scientific << computed << std::endl;
    }
}
```

## Complete Example: Heat Equation with Exponential Decay

### Problem Definition

Solve the heat equation:
$$\frac{\partial u}{\partial t} - \Delta u = 0 \quad \text{in } [0,1] \times [0,1] \times (0,1]$$

with:

- **Initial condition**: $u_0(x,y) = \sin(\pi x)\sin(\pi y)$
- **Boundary conditions**: $u = 0$ on $\partial\Omega$
- **Exact solution**: $u(x,y,t) = e^{-2\pi^2 t}\sin(\pi x)\sin(\pi y)$

### Complete Implementation

```cpp
#include <iostream>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "solver/parabolic.hpp"
#include "utils/boundary.hpp"
#include "utils/time_scheme.hpp"

int main() {
    // Step 1: Mesh setup
    Eigen::MatrixXd nodes(9, 2);
    nodes << 0.0, 0.0,   0.5, 0.0,   1.0, 0.0,
             0.0, 0.5,   0.5, 0.5,   1.0, 0.5,
             0.0, 1.0,   0.5, 1.0,   1.0, 1.0;

    Eigen::MatrixXi elements(4, 4);
    elements << 0, 1, 4, 3,
                1, 2, 5, 4,
                3, 4, 7, 6,
                4, 5, 8, 7;

    int polynomial_order = 1;

    // Step 2: Solver initialization
    solver::parabolic vem_solver(polynomial_order, nodes, elements);
    vem_solver.assemble_system();

    // Step 3: Boundary conditions
    utils::boundary boundary_manager(nodes, elements, polynomial_order);
    boundary_manager.add_dirichlet(
        [](const Eigen::Vector2d& x, double t) -> double { return 0.0; },
        utils::boundary::BoundaryRegion::ENTIRE,
        "Homogeneous_Dirichlet"
    );

    vem_solver.F_h = Eigen::VectorXd::Zero(vem_solver.get_n_dofs());
    boundary_manager.apply_dirichlet_conditions(
        vem_solver.K_h, vem_solver.M_h, vem_solver.F_h, 0.0
    );

    // Step 4: Time integration setup
    utils::time_scheme time_integrator(vem_solver.M_h, vem_solver.K_h);
    double dt = 0.01;
    time_integrator.setup_time_parameters(dt, utils::time_scheme::SchemeType::BACKWARD_EULER);
    time_integrator.configure_solver(utils::time_scheme::SolverType::DIRECT, 1e-12, 1000);

    // Step 5: Problem functions
    auto initial_condition = [](const Eigen::Vector2d& x) -> double {
        return std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
    };

    auto source_function = [](const Eigen::Vector2d& x, double t) -> double {
        return 0.0;
    };

    auto exact_solution = [](const Eigen::Vector2d& x, double t) -> double {
        return std::exp(-2.0 * M_PI * M_PI * t) *
               std::sin(M_PI * x(0)) * std::sin(M_PI * x(1));
    };

    // Step 6: Initial conditions
    Eigen::VectorXd U_current(vem_solver.get_n_dofs());
    Eigen::VectorXd U_previous(vem_solver.get_n_dofs());

    for (int i = 0; i < nodes.rows(); ++i) {
        Eigen::Vector2d vertex = nodes.row(i);
        U_current(i) = initial_condition(vertex);
    }
    U_previous = U_current;

    // Step 7: Time integration
    double T_final = 1.0;
    int num_time_steps = static_cast<int>(std::ceil(T_final / dt));

    for (int step = 1; step <= num_time_steps; ++step) {
        double current_time = step * dt;

        vem_solver.assemble_load_vector(source_function, current_time);
        Eigen::VectorXd F_h = vem_solver.get_load_vector();

        Eigen::SparseMatrix<double> M_h_bc = vem_solver.M_h;
        Eigen::SparseMatrix<double> K_h_bc = vem_solver.K_h;
        Eigen::VectorXd F_h_bc = F_h;

        boundary_manager.apply_dirichlet_conditions(K_h_bc, M_h_bc, F_h_bc, current_time);

        if (step == 1) {
            time_integrator.initialize(M_h_bc, K_h_bc);
        }

        bool success = time_integrator.step(U_current, U_previous, F_h_bc);
        if (!success) {
            std::cerr << "Time step failed!" << std::endl;
            return 1;
        }

        U_previous = U_current;
    }

    // Step 8: Results analysis
    double error_l2 = 0.0;
    for (int i = 0; i < nodes.rows(); ++i) {
        Eigen::Vector2d vertex = nodes.row(i);
        double exact = exact_solution(vertex, T_final);
        double computed = U_current(i);
        error_l2 += (computed - exact) * (computed - exact);
    }
    error_l2 = std::sqrt(error_l2);

    std::cout << "Final L2 error: " << error_l2 << std::endl;

    return 0;
}
```

## Advanced Usage Patterns

### Multi-Physics Coupling

For coupled problems, the parabolic solver can be extended:

```cpp
// Example: Heat equation with convection
auto convection_velocity = [](const Eigen::Vector2d& x, double t) -> Eigen::Vector2d {
    return Eigen::Vector2d(1.0, 0.5);  // Constant velocity field
};

// Modify source term to include convection
auto source_with_convection = [&](const Eigen::Vector2d& x, double t) -> double {
    Eigen::Vector2d v = convection_velocity(x, t);
    // Add convection term: -v · ∇u (approximated)
    return base_source(x, t) - convection_contribution(x, t, v);
};
```

### Adaptive Time Stepping

```cpp
// Monitor solution changes for adaptive time stepping
double solution_change = (U_current - U_previous).norm();
double tolerance = 1e-6;

if (solution_change > tolerance) {
    dt *= 0.8;  // Reduce time step
} else if (solution_change < tolerance / 10) {
    dt *= 1.2;  // Increase time step
}

// Update time integrator
time_integrator.setup_time_parameters(dt, scheme_type);
```

### Nonlinear Problems

For nonlinear parabolic equations, use Newton-Raphson iteration:

```cpp
// Nonlinear source term: f(u, x, t)
auto nonlinear_source = [](double u, const Eigen::Vector2d& x, double t) -> double {
    return u * u - std::sin(M_PI * x(0));  // Example: u² - sin(πx)
};

// Newton-Raphson loop within each time step
for (int newton_iter = 0; newton_iter < max_newton_iter; ++newton_iter) {
    // Assemble Jacobian and residual
    // Solve linearized system
    // Update solution
    // Check convergence
}
```

## Performance Optimization Guidelines

### Memory Management

1. **Sparse Matrix Optimization**:

   ```cpp
   // Reserve memory for sparse matrices
   M_h.reserve(estimated_nnz);
   K_h.reserve(estimated_nnz);

   // Compress after assembly
   M_h.makeCompressed();
   K_h.makeCompressed();
   ```

2. **Reuse Factorizations**:
   ```cpp
   // For multiple time steps with same matrix
   if (matrix_unchanged) {
       // Reuse previous factorization
       solver.factorize(A);  // Only if matrix changed
   }
   ```

### Computational Efficiency

1. **Solver Selection**:

   - **Small systems** (N < 10⁴): Use direct solver
   - **Large systems** (N > 10⁵): Use iterative solver with preconditioning
   - **Multiple time steps**: Direct solver with factorization reuse

2. **Time Step Selection**:
   - **Stability**: Ensure CFL condition if applicable
   - **Accuracy**: Balance between computational cost and precision
   - **Adaptive**: Monitor solution changes for optimal time stepping

### Error Control

1. **Convergence Monitoring**:

   ```cpp
   // Check residual norm
   double residual_norm = (A * U_current - F_h).norm();
   if (residual_norm > tolerance) {
       // Reduce time step or increase solver precision
   }
   ```

2. **Solution Bounds**:
   ```cpp
   // Physical bounds checking
   double u_min = U_current.minCoeff();
   double u_max = U_current.maxCoeff();
   if (u_min < physical_min || u_max > physical_max) {
       std::cerr << "Solution outside physical bounds!" << std::endl;
   }
   ```

## Troubleshooting Common Issues

### Compilation Errors

1. **Missing Includes**: Ensure all required headers are included
2. **Eigen Version**: Verify Eigen version compatibility
3. **Compiler Flags**: Use appropriate C++ standard (C++11 or later)

### Runtime Errors

1. **Matrix Singularity**: Check boundary condition application
2. **Memory Issues**: Monitor sparse matrix memory usage
3. **Convergence Failures**: Adjust solver tolerances or time step size

### Numerical Issues

1. **Oscillations**: Reduce time step size or use more dissipative scheme
2. **Energy Conservation**: Use Crank-Nicolson for conservative problems
3. **Boundary Layer Resolution**: Ensure adequate mesh resolution near boundaries

## Conclusion

This comprehensive guide provides a complete pipeline for using the VEM parabolic solver. The implementation follows VEM theory rigorously while maintaining computational efficiency. Key advantages include:

- **Flexibility**: Supports arbitrary polygonal meshes
- **Accuracy**: Optimal convergence rates for smooth solutions
- **Stability**: Unconditionally stable time integration schemes
- **Performance**: Efficient sparse matrix operations and solver options

The modular design allows for easy extension to more complex problems, making it suitable for both research and industrial applications in computational physics and engineering.
