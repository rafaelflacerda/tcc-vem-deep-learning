# Time Integration Schemes for Virtual Element Method

## Overview

This document provides a comprehensive description of the time integration schemes implemented in the `time_scheme` class. The implementation supports multiple time integration methods for solving time-dependent partial differential equations (PDEs) using the Virtual Element Method (VEM).

## Mathematical Background

### General Form of Time-Dependent PDEs

The time integration schemes are designed to solve semi-discrete systems of the form:

$$M_h \frac{du}{dt} + K_h u = f(t)$$

where:

- $M_h \in \mathbb{R}^{n \times n}$ is the mass matrix (positive definite)
- $K_h \in \mathbb{R}^{n \times n}$ is the stiffness matrix
- $u(t) \in \mathbb{R}^n$ is the solution vector at time $t$
- $f(t) \in \mathbb{R}^n$ is the right-hand side vector (forcing term)

This form arises naturally from the spatial discretization of time-dependent PDEs using the Virtual Element Method.

### Time Discretization

Let $t_n = n \cdot \Delta t$ for $n = 0, 1, 2, \ldots$ where $\Delta t > 0$ is the time step size. We denote:

- $u^n \approx u(t_n)$ as the approximate solution at time $t_n$
- $f^n = f(t_n)$ as the right-hand side at time $t_n$

## Mass Matrix Considerations

### Consistent vs. Lumped Mass Matrices

An important conceptual point: **the time integration schemes implemented here solve a coupled system of ODEs, not independent ODEs**, because we use a **consistent mass matrix** rather than a lumped one.

#### Consistent Mass Matrix (Our Implementation)

The VEM implementation uses a consistent mass matrix $M_h$ where:

$$M_{ij} = \int_{\Omega} \phi_i \phi_j \, dx$$

This results in a **full sparse matrix** with off-diagonal entries:

```cpp
// Consistent mass matrix structure:
// [m11  m12  m13  ...]  <- Row 1 couples multiple DOFs
// [m21  m22  m23  ...]  <- Row 2 couples multiple DOFs
// [m31  m32  m33  ...]  <- Row 3 couples multiple DOFs
// [... ... ...  ...]
```

**Consequences:**

- The system $M_h \frac{du}{dt} + K_h u = f(t)$ represents **coupled ODEs**
- Each degree of freedom is coupled to its neighbors through the mass matrix
- Time integration requires solving linear systems even for explicit schemes
- Higher accuracy in time integration
- More computational cost per time step

#### Lumped Mass Matrix Alternative

A lumped mass matrix would be diagonal:

$$M_{\text{lumped}} = \text{diag}(m_1, m_2, \ldots, m_n)$$

where $m_i = \sum_j M_{ij}$ (row sum) or other lumping strategies.

```cpp
// Lumped mass matrix structure:
// [m1   0   0  ...]  <- Each row has only diagonal entry
// [0   m2   0  ...]
// [0    0  m3  ...]
// [... ... ... ...]
```

**Consequences:**

- The system becomes $m_i \frac{du_i}{dt} + \sum_j K_{ij} u_j = f_i(t)$
- Each time derivative can be computed independently: $\frac{du_i}{dt} = \frac{1}{m_i}(f_i - \sum_j K_{ij} u_j)$
- Explicit schemes become truly explicit (no linear system solve)
- Lower accuracy but much faster explicit time stepping
- Loss of some conservation properties

### Impact on Time Integration Schemes

#### Forward Euler Example

**With Consistent Mass Matrix (Our Case):**
$$M_h u^{n+1} = M_h u^n - \Delta t(K_h u^n - f^n)$$

- Still requires solving $M_h u^{n+1} = \text{RHS}$
- Linear system solve needed even for "explicit" scheme

**With Lumped Mass Matrix:**
$$u_i^{n+1} = u_i^n - \frac{\Delta t}{m_i}\left(\sum_j K_{ij} u_j^n - f_i^n\right)$$

- Direct computation, no linear system solve
- Truly explicit time stepping

#### Backward Euler Example

**With Consistent Mass Matrix (Our Case):**
$$(M_h + \Delta t K_h) u^{n+1} = M_h u^n + \Delta t f^{n+1}$$

- Full system matrix to solve

**With Lumped Mass Matrix:**
$$(M_{\text{lumped}} + \Delta t K_h) u^{n+1} = M_{\text{lumped}} u^n + \Delta t f^{n+1}$$

- System matrix has diagonal mass contribution

### Why VEM Uses Consistent Mass Matrix

1. **Accuracy**: Consistent mass matrices provide higher-order accuracy in time
2. **Conservation**: Better preservation of physical conservation laws
3. **Stability**: More robust stability properties for complex geometries
4. **Consistency**: Matches the variational formulation exactly

### Performance Trade-offs

| Aspect               | Consistent Mass      | Lumped Mass        |
| -------------------- | -------------------- | ------------------ |
| **Accuracy**         | Higher               | Lower              |
| **Explicit Schemes** | Require linear solve | Direct computation |
| **Memory**           | More coupling        | Diagonal storage   |
| **Conservation**     | Excellent            | Approximate        |
| **Complex Geometry** | Robust               | May have issues    |

### Implementation Note

In our `time_scheme` class, even "explicit" schemes like Forward Euler require calling `solve_linear_system()` because of the consistent mass matrix. The term "explicit" refers to the time discretization (using $u^n$ instead of $u^{n+1}$ in the spatial operator), not the absence of linear system solves.

```cpp
// Forward Euler in our implementation still needs linear solve:
case SchemeType::FORWARD_EULER: {
    Eigen::VectorXd rhs_vector = (*M_h_) * U_previous - dt_ * ((*K_h_) * U_previous - F_rhs);
    success = solve_linear_system(*M_h_, rhs_vector, U_current);  // Linear solve needed!
    break;
}
```

This is fundamentally different from solving independent ODEs where explicit schemes would be:

```cpp
// If we had independent ODEs (lumped mass):
for (int i = 0; i < n; ++i) {
    U_current[i] = U_previous[i] - dt/mass[i] * (K_row_i_dot_U - F_rhs[i]);  // Direct computation
}
```

## Implemented Time Integration Schemes

### 1. Backward Euler Method

The Backward Euler method is a first-order implicit time integration scheme.

#### Mathematical Formulation

The backward difference approximation for the time derivative is:

$$\frac{du}{dt}\bigg|_{t=t_n} \approx \frac{u^n - u^{n-1}}{\Delta t}$$

Substituting into the semi-discrete equation:

$$M_h \frac{u^n - u^{n-1}}{\Delta t} + K_h u^n = f^n$$

Rearranging to solve for $u^n$:

$$(M_h + \Delta t \cdot K_h) u^n = M_h u^{n-1} + \Delta t \cdot f^n$$

#### Properties

- **Order of Accuracy**: First-order ($\mathcal{O}(\Delta t)$)
- **Stability**: Unconditionally stable (A-stable)
- **Implicit**: Requires solving a linear system at each time step
- **Dissipative**: Introduces numerical damping

#### Implementation Details

The linear system to solve at each time step is:
$$A u^n = b$$

where:

- $A = M_h + \Delta t \cdot K_h$
- $b = M_h u^{n-1} + \Delta t \cdot f^n$

### 2. Crank-Nicolson Method

The Crank-Nicolson method is a second-order implicit time integration scheme that uses the trapezoidal rule.

#### Mathematical Formulation

The method applies the trapezoidal rule to integrate the semi-discrete equation:

$$M_h \frac{u^n - u^{n-1}}{\Delta t} + \frac{1}{2}(K_h u^n + K_h u^{n-1}) = \frac{1}{2}(f^n + f^{n-1})$$

Rearranging:

$$\left(M_h + \frac{\Delta t}{2} K_h\right) u^n = \left(M_h - \frac{\Delta t}{2} K_h\right) u^{n-1} + \frac{\Delta t}{2}(f^n + f^{n-1})$$

#### Properties

- **Order of Accuracy**: Second-order ($\mathcal{O}(\Delta t^2)$)
- **Stability**: Unconditionally stable (A-stable)
- **Implicit**: Requires solving a linear system at each time step
- **Energy Conserving**: Preserves energy in conservative systems

#### Implementation Details

The linear system to solve at each time step is:
$$A u^n = b$$

where:

- $A = M_h + \frac{\Delta t}{2} K_h$
- $b = \left(M_h - \frac{\Delta t}{2} K_h\right) u^{n-1} + \frac{\Delta t}{2}(f^n + f^{n-1})$

## Class Architecture and Implementation

### Class Structure

The `time_scheme` class is organized into several key components:

1. **Enumerations**: Define scheme types and solver options
2. **Constructor/Destructor**: Handle initialization and cleanup
3. **Setup Methods**: Configure time parameters and solver options
4. **Integration Methods**: Perform time stepping
5. **Utility Methods**: Provide debugging and statistics

### Key Design Decisions

#### Matrix Storage Strategy

The class stores pointers to the mass and stiffness matrices rather than copying them:

```cpp
const Eigen::SparseMatrix<double>* M_h_;
const Eigen::SparseMatrix<double>* K_h_;
```

**Rationale**:

- Avoids expensive matrix copying operations
- Reduces memory footprint
- Matrices are typically large and sparse in VEM applications

#### Solver Configuration

Two solver types are supported:

1. **Direct Solver** (`SolverType::DIRECT`): Uses Eigen's `SparseLU` factorization
2. **Iterative Solver** (`SolverType::ITERATIVE`): Uses Eigen's `BiCGSTAB` method

**Selection Criteria**:

- Direct solvers: Better for small to medium systems, guaranteed convergence
- Iterative solvers: Better for large systems, memory efficient, may require preconditioning

### Method Interfaces

#### Single-Level Time Stepping

```cpp
bool step(
    Eigen::VectorXd& U_current,
    const Eigen::VectorXd& U_previous,
    const Eigen::VectorXd& F_current
);
```

Used for schemes that only require the current right-hand side (e.g., Backward Euler).

#### Two-Level Time Stepping

```cpp
bool step_two_level(
    Eigen::VectorXd& U_current,
    const Eigen::VectorXd& U_previous,
    const Eigen::VectorXd& F_current,
    const Eigen::VectorXd& F_previous
);
```

Used for schemes that require both current and previous right-hand sides (e.g., Crank-Nicolson).

## Linear System Solution

### Direct Solver Implementation

```cpp
Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
solver.compute(A);
x = solver.solve(b);
```

**Process**:

1. **Symbolic Factorization**: Analyze sparsity pattern
2. **Numerical Factorization**: Compute LU decomposition
3. **Forward/Backward Substitution**: Solve triangular systems

**Advantages**:

- Robust and reliable
- Exact solution (up to machine precision)
- Good for multiple solves with same matrix

**Disadvantages**:

- Memory intensive for large systems
- Factorization cost scales as $\mathcal{O}(n^{3/2})$ to $\mathcal{O}(n^3)$

### Iterative Solver Implementation

```cpp
Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
solver.setMaxIterations(solver_max_iter_);
solver.setTolerance(solver_tolerance_);
x = solver.solve(b);
```

**BiCGSTAB Algorithm**:

- Biconjugate Gradient Stabilized method
- Suitable for non-symmetric systems
- Converges in at most $n$ iterations (theoretically)

**Parameters**:

- `solver_tolerance_`: Convergence tolerance (default: $10^{-6}$)
- `solver_max_iter_`: Maximum iterations (default: 1000)

## Error Handling and Validation

### Input Validation

The implementation includes comprehensive validation:

1. **Matrix Dimensions**: Ensures matrices are square and compatible
2. **Time Step**: Validates $\Delta t > 0$
3. **Initialization**: Checks that setup methods are called before time stepping

### Error Recovery

- **Factorization Failures**: Detected and reported
- **Convergence Failures**: Iterative solver failures are caught
- **Statistics Tracking**: Failed steps are recorded for analysis

## Performance Considerations

### Memory Management

- **Sparse Matrix Storage**: Leverages Eigen's efficient sparse matrix format
- **In-Place Operations**: Minimizes temporary allocations
- **Matrix Reuse**: Avoids recomputing factorizations when possible

### Computational Complexity

For a system with $n$ degrees of freedom:

| Operation | Direct Solver          | Iterative Solver  |
| --------- | ---------------------- | ----------------- |
| Setup     | $\mathcal{O}(n^{3/2})$ | $\mathcal{O}(n)$  |
| Solve     | $\mathcal{O}(n^{3/2})$ | $\mathcal{O}(kn)$ |
| Memory    | $\mathcal{O}(n^{3/2})$ | $\mathcal{O}(n)$  |

where $k$ is the number of iterations for convergence.

### Statistics and Monitoring

The class tracks:

- Total number of time steps
- Successful vs. failed steps
- Average solve time per step
- Solver iteration counts (for iterative methods)

## Usage Guidelines

### Recommended Workflow

1. **Construction**: Create `time_scheme` object with mass and stiffness matrices
2. **Configuration**: Set time parameters and solver options
3. **Time Stepping**: Call appropriate step method in time loop
4. **Monitoring**: Check return values and statistics

### Scheme Selection Guidelines

| Scenario               | Recommended Scheme | Rationale             |
| ---------------------- | ------------------ | --------------------- |
| High accuracy required | Crank-Nicolson     | Second-order accuracy |
| Stability critical     | Both schemes       | Both are A-stable     |
| Energy conservation    | Crank-Nicolson     | Preserves energy      |
| Simple implementation  | Backward Euler     | First-order, robust   |

### Solver Selection Guidelines

| System Size        | Matrix Properties   | Recommended Solver           |
| ------------------ | ------------------- | ---------------------------- |
| Small ($n < 10^4$) | Any                 | Direct                       |
| Large ($n > 10^5$) | Well-conditioned    | Iterative                    |
| Medium             | Ill-conditioned     | Direct                       |
| Any                | Multiple time steps | Direct (reuse factorization) |

## Usage Example

This section provides a complete, step-by-step example of how to use the `time_scheme` class to solve time-dependent problems using both Backward Euler and Crank-Nicolson schemes.

### Problem Setup

Consider solving the semi-discrete system:
$$M_h \frac{du}{dt} + K_h u = f(t)$$

where:

- `M_h` is the mass matrix (sparse, positive definite)
- `K_h` is the stiffness matrix (sparse)
- `u(t)` is the solution vector
- `f(t)` is the right-hand side (forcing) vector

### Example 1: Backward Euler Scheme

```cpp
#include "utils/time_scheme.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

int main() {
    // Step 1: Assume we have assembled mass and stiffness matrices
    // (These would come from your VEM solver)
    Eigen::SparseMatrix<double> M_h;  // Mass matrix
    Eigen::SparseMatrix<double> K_h;  // Stiffness matrix

    // For this example, let's create simple 3x3 matrices
    int n = 3;
    M_h.resize(n, n);
    K_h.resize(n, n);

    // Create a simple mass matrix (identity)
    for (int i = 0; i < n; ++i) {
        M_h.insert(i, i) = 1.0;
    }

    // Create a simple stiffness matrix (tridiagonal)
    for (int i = 0; i < n; ++i) {
        K_h.insert(i, i) = 2.0;
        if (i > 0) K_h.insert(i, i-1) = -1.0;
        if (i < n-1) K_h.insert(i, i+1) = -1.0;
    }

    // Finalize the sparse matrices
    M_h.makeCompressed();
    K_h.makeCompressed();

    // Step 2: Create the time scheme object
    utils::time_scheme time_integrator(M_h, K_h);

    // Step 3: Configure time parameters
    double dt = 0.01;                                        // Time step
    time_integrator.setup_time_parameters(dt, utils::time_scheme::SchemeType::BACKWARD_EULER);

    // Step 4: Configure the linear solver
    time_integrator.configure_solver(
        utils::time_scheme::SolverType::DIRECT,  // Solver type
        1e-12,                                   // Tolerance
        1000                                     // Max iterations
    );

    // Step 5: Optional - Enable debug mode for detailed output
    time_integrator.set_debug_mode(true);

    // Step 6: Set up initial conditions and time loop parameters
    Eigen::VectorXd U_current(n);
    Eigen::VectorXd U_previous(n);

    // Initial condition: U(0) = [1, 0, 0]^T
    U_current << 1.0, 0.0, 0.0;

    double current_time = 0.0;
    double final_time = 0.5;
    int step_count = 0;

    std::cout << "Starting Backward Euler time integration..." << std::endl;
    std::cout << "Initial solution: " << U_current.transpose() << std::endl;

    // Step 7: Time stepping loop
    while (current_time < final_time - 1e-12) {
        current_time += dt;
        step_count++;

        // Store previous solution
        U_previous = U_current;

        // Assemble forcing vector f(t) at current time
        Eigen::VectorXd F_current(n);
        F_current << std::sin(current_time), 0.0, 0.0;  // Example forcing

        // Perform Backward Euler time step
        bool success = time_integrator.step(U_current, U_previous, F_current);

        if (!success) {
            std::cerr << "Time step failed at t = " << current_time << std::endl;
            return -1;
        }

        // Output progress
        if (step_count % 10 == 0) {
            std::cout << "t = " << current_time
                      << ", ||u|| = " << U_current.norm() << std::endl;
        }
    }

    // Step 8: Get final results and statistics
    std::cout << "\nFinal solution: " << U_current.transpose() << std::endl;

    const auto& stats = time_integrator.get_statistics();
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total steps: " << stats.total_steps << std::endl;
    std::cout << "  Successful steps: " << stats.successful_steps << std::endl;
    std::cout << "  Failed steps: " << stats.failed_steps << std::endl;
    std::cout << "  Average solve time: " << stats.avg_solve_time << " ms" << std::endl;

    return 0;
}
```

### Example 2: Crank-Nicolson Scheme

```cpp
#include "utils/time_scheme.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

int main() {
    // Step 1: Setup matrices (same as before)
    int n = 3;
    Eigen::SparseMatrix<double> M_h(n, n);
    Eigen::SparseMatrix<double> K_h(n, n);

    // Create mass matrix
    for (int i = 0; i < n; ++i) {
        M_h.insert(i, i) = 1.0;
    }

    // Create stiffness matrix
    for (int i = 0; i < n; ++i) {
        K_h.insert(i, i) = 2.0;
        if (i > 0) K_h.insert(i, i-1) = -1.0;
        if (i < n-1) K_h.insert(i, i+1) = -1.0;
    }

    M_h.makeCompressed();
    K_h.makeCompressed();

    // Step 2: Create time scheme with Crank-Nicolson
    utils::time_scheme time_integrator(M_h, K_h);

    // Step 3: Configure for Crank-Nicolson
    double dt = 0.01;
    time_integrator.setup_time_parameters(dt, utils::time_scheme::SchemeType::CRANK_NICOLSON);

    // Step 4: Configure solver (may need iterative for larger systems)
    time_integrator.configure_solver(
        utils::time_scheme::SolverType::ITERATIVE,  // Use iterative solver
        1e-10,                                      // Tighter tolerance
        5000                                        // More iterations
    );

    time_integrator.set_debug_mode(false);  // Disable debug for cleaner output

    // Step 5: Initialize solution vectors
    Eigen::VectorXd U_current(n);
    Eigen::VectorXd U_previous(n);

    // Initial condition
    U_current << 1.0, 0.0, 0.0;

    double current_time = 0.0;
    double final_time = 0.5;
    int step_count = 0;

    std::cout << "Starting Crank-Nicolson time integration..." << std::endl;
    std::cout << "Scheme order: " << time_integrator.get_order() << std::endl;
    std::cout << "Scheme name: " << time_integrator.get_scheme_name() << std::endl;

    // Step 6: Time stepping loop for Crank-Nicolson
    Eigen::VectorXd F_current(n);
    Eigen::VectorXd F_previous(n);

    // Initial forcing
    F_previous << std::sin(current_time), 0.0, 0.0;

    while (current_time < final_time - 1e-12) {
        current_time += dt;
        step_count++;

        // Store previous solution and forcing
        U_previous = U_current;
        F_previous = F_current;

        // Compute current forcing
        F_current << std::sin(current_time), 0.0, 0.0;

        // Perform Crank-Nicolson time step (requires both current and previous forcing)
        bool success = time_integrator.step_two_level(U_current, U_previous, F_current, F_previous);

        if (!success) {
            std::cerr << "Time step failed at t = " << current_time << std::endl;
            return -1;
        }

        // Output progress
        if (step_count % 10 == 0) {
            std::cout << "t = " << current_time
                      << ", ||u|| = " << U_current.norm() << std::endl;
        }
    }

    // Step 7: Final results
    std::cout << "\nFinal solution: " << U_current.transpose() << std::endl;

    const auto& stats = time_integrator.get_statistics();
    std::cout << "\nStatistics:" << std::endl;
    std::cout << "  Total steps: " << stats.total_steps << std::endl;
    std::cout << "  Successful steps: " << stats.successful_steps << std::endl;
    std::cout << "  Failed steps: " << stats.failed_steps << std::endl;
    std::cout << "  Average solve time: " << stats.avg_solve_time << " ms" << std::endl;

    return 0;
}
```

### Example 3: Integration with VEM Solver

This example shows how to integrate the `time_scheme` with a VEM solver in a practical scenario:

```cpp
#include "solver/parabolic.hpp"
#include "utils/time_scheme.hpp"
#include "utils/boundary.hpp"

void solve_parabolic_vem_problem() {
    // Step 1: Create VEM solver (assuming mesh and functions are defined)
    int vem_order = 2;  // Quadratic VEM
    Eigen::MatrixXd nodes;     // Node coordinates
    Eigen::MatrixXi elements;  // Element connectivity

    // ... (mesh setup code would go here)

    solver::parabolic vem_solver(vem_order, nodes, elements);

    // Step 2: Assemble system matrices
    vem_solver.assemble_system();

    // Step 3: Setup boundary conditions
    utils::boundary boundary_handler(nodes, elements, vem_order);
    boundary_handler.add_dirichlet([](const Eigen::Vector2d& point, double t) {
        return 0.0;  // Homogeneous Dirichlet BC
    }, utils::boundary::BoundaryRegion::ENTIRE, "Dirichlet");

    // Step 4: Set initial conditions
    vem_solver.set_initial_conditions([](const Eigen::Vector2d& point) {
        return std::sin(M_PI * point.x()) * std::sin(M_PI * point.y());
    });

    Eigen::VectorXd U_current = vem_solver.U_h;
    Eigen::VectorXd U_previous = U_current;

    // Step 5: Time stepping parameters
    double dt = 0.01;
    double final_time = 0.5;
    double current_time = 0.0;

    std::cout << "Starting VEM time integration..." << std::endl;

    // Step 6: Time stepping loop
    while (current_time < final_time - 1e-12) {
        current_time += dt;

        // Get fresh matrix copies (important for boundary conditions)
        Eigen::SparseMatrix<double> M_h = vem_solver.get_global_mass_matrix();
        Eigen::SparseMatrix<double> K_h = vem_solver.get_global_stiffness_matrix();

        // Assemble load vector at current time
        vem_solver.assemble_load_vector([](const Eigen::Vector2d& point, double t) {
            return std::exp(t) * (1.0 + 2.0 * M_PI * M_PI) *
                   std::sin(M_PI * point.x()) * std::sin(M_PI * point.y());
        }, current_time);

        Eigen::VectorXd F_h = vem_solver.F_h;

        // Apply boundary conditions
        boundary_handler.apply_dirichlet_conditions(K_h, M_h, F_h, current_time);

        // Create time integrator for this step
        utils::time_scheme time_integrator(M_h, K_h);
        time_integrator.setup_time_parameters(dt, utils::time_scheme::SchemeType::BACKWARD_EULER);
        time_integrator.configure_solver(utils::time_scheme::SolverType::DIRECT, 1e-12, 1000);

        // Store previous solution
        U_previous = U_current;

        // Perform time step
        bool success = time_integrator.step(U_current, U_previous, F_h);

        if (!success) {
            std::cerr << "Time step failed at t = " << current_time << std::endl;
            return;
        }

        // Output progress
        if (static_cast<int>(current_time / dt) % 10 == 0) {
            std::cout << "t = " << current_time
                      << ", ||u|| = " << U_current.norm() << std::endl;
        }
    }

    std::cout << "VEM time integration completed successfully!" << std::endl;
}
```

### Key Points for Successful Usage

1. **Matrix Preparation**: Ensure mass and stiffness matrices are properly assembled and compressed before creating the time scheme.

2. **Method Selection**:

   - Use `step()` for Backward Euler (requires only current forcing)
   - Use `step_two_level()` for Crank-Nicolson (requires current and previous forcing)

3. **Solver Configuration**: Choose solver type based on system size and matrix properties.

4. **Error Handling**: Always check the return value of step methods and handle failures appropriately.

5. **Statistics Monitoring**: Use `get_statistics()` to monitor performance and detect potential issues.

6. **Debug Mode**: Enable debug mode during development to get detailed information about the solution process.

7. **Boundary Conditions**: Apply boundary conditions to fresh matrix copies at each time step to ensure correctness.

8. **Time Step Size**: Choose appropriate time step size based on stability requirements and desired accuracy.

## Extending the Time Scheme

This section provides comprehensive guidelines for developers who want to add new time integration methods to the `time_scheme` class. We'll use Forward Euler as a concrete example to illustrate the process.

### Overview of Extension Process

Adding a new time integration scheme involves several steps:

1. **Mathematical Analysis**: Understand the mathematical formulation of the new scheme
2. **Enum Extension**: Add the new scheme type to the `SchemeType` enum
3. **Method Implementation**: Implement the time stepping logic
4. **Utility Method Updates**: Update helper methods for scheme properties
5. **Testing**: Validate the implementation with known solutions
6. **Documentation**: Update documentation and examples

### Step 1: Mathematical Analysis

Before implementing any new scheme, understand its mathematical formulation and properties.

#### Forward Euler Example

The Forward Euler method is an explicit first-order scheme:

**Mathematical Formulation:**
$$M_h \frac{u^n - u^{n-1}}{\Delta t} + K_h u^{n-1} = f^{n-1}$$

**Rearranged Form:**
$$M_h u^n = M_h u^{n-1} - \Delta t (K_h u^{n-1} - f^{n-1})$$

**Key Properties:**

- **Order**: First-order ($\mathcal{O}(\Delta t)$)
- **Stability**: Conditionally stable (requires $\Delta t < \Delta t_{critical}$)
- **Type**: Explicit (no linear system to solve)
- **Computational Cost**: Low per step

### Step 2: Enum Extension

Add the new scheme to the `SchemeType` enum in the header file:

```cpp
// In include/utils/time_scheme.hpp
enum class SchemeType {
    BACKWARD_EULER,
    CRANK_NICOLSON,
    FORWARD_EULER,        // Add new scheme here
    // Add more schemes as needed
};
```

### Step 3: Method Implementation

#### Option A: Explicit Schemes (Forward Euler)

For explicit schemes, extend the `step()` method since they don't require previous forcing terms:

```cpp
// In lib/utils/time_scheme.cpp
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
            // Existing implementation
            Eigen::SparseMatrix<double> lhs_matrix = *M_h_ + dt_ * (*K_h_);
            Eigen::VectorXd rhs_vector = (*M_h_) * U_previous + dt_ * F_rhs;
            success = solve_linear_system(lhs_matrix, rhs_vector, U_current);
            break;
        }

        case SchemeType::FORWARD_EULER: {
            // Forward Euler: M*U^n = M*U^{n-1} - dt*(K*U^{n-1} - F^{n-1})
            Eigen::VectorXd rhs_vector = (*M_h_) * U_previous - dt_ * ((*K_h_) * U_previous - F_rhs);
            success = solve_linear_system(*M_h_, rhs_vector, U_current);
            break;
        }

        default:
            debug_print("Error: Unsupported scheme for single-level step");
            return false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double solve_time = duration.count() / 1000.0;

    update_statistics(success, solve_time);
    return success;
}
```

#### Option B: Multi-Level Schemes

For schemes requiring multiple time levels, extend the `step_two_level()` method or create new methods:

```cpp
// Example: Adams-Bashforth 2nd order (requires two previous steps)
bool time_scheme::step_multi_level(
    Eigen::VectorXd& U_current,
    const Eigen::VectorXd& U_previous,
    const Eigen::VectorXd& U_previous2,
    const Eigen::VectorXd& F_current,
    const Eigen::VectorXd& F_previous
){
    // Implementation for multi-step methods
    // This would require additional member variables to store history
}
```

### Step 4: Utility Method Updates

Update all utility methods to handle the new scheme:

```cpp
// In include/utils/time_scheme.hpp

bool is_stable() const {
    switch (scheme_type_) {
        case SchemeType::BACKWARD_EULER:
        case SchemeType::CRANK_NICOLSON:
            return true;
        case SchemeType::FORWARD_EULER:
            return false;  // Conditionally stable
        default:
            return false;
    }
}

int get_order() const {
    switch (scheme_type_) {
        case SchemeType::BACKWARD_EULER:
        case SchemeType::FORWARD_EULER:
            return 1;
        case SchemeType::CRANK_NICOLSON:
            return 2;
        default:
            return 0;
    }
}

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
```

### Step 5: Advanced Extensions

#### Adding Adaptive Time Stepping

For schemes that benefit from adaptive time stepping, you can extend the class:

```cpp
// In include/utils/time_scheme.hpp
class time_scheme {
public:
    // ... existing members ...

    /**
     * @brief Adaptive time stepping configuration
     */
    struct AdaptiveConfig {
        bool enabled = false;
        double tolerance = 1e-6;
        double safety_factor = 0.8;
        double min_factor = 0.1;
        double max_factor = 5.0;
        double min_dt = 1e-12;
        double max_dt = 1e-1;
    };

    /**
     * @brief Enable adaptive time stepping
     */
    void enable_adaptive_stepping(const AdaptiveConfig& config);

    /**
     * @brief Adaptive time step with error estimation
     */
    bool adaptive_step(
        Eigen::VectorXd& U_current,
        const Eigen::VectorXd& U_previous,
        const Eigen::VectorXd& F_current,
        double& dt_suggested
    );

private:
    AdaptiveConfig adaptive_config_;

    /**
     * @brief Estimate local truncation error
     */
    double estimate_error(
        const Eigen::VectorXd& U_fine,
        const Eigen::VectorXd& U_coarse
    ) const;
};
```

#### Adding Higher-Order Methods

For higher-order methods like Runge-Kutta schemes:

```cpp
// Example: 4th-order Runge-Kutta
bool time_scheme::step_runge_kutta_4(
    Eigen::VectorXd& U_current,
    const Eigen::VectorXd& U_previous,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd&, double)>& rhs_function,
    double current_time
){
    // RK4 implementation
    Eigen::VectorXd k1, k2, k3, k4;
    Eigen::VectorXd temp_U;

    // k1 = dt * f(t_n, U_n)
    k1 = dt_ * rhs_function(U_previous, current_time);

    // k2 = dt * f(t_n + dt/2, U_n + k1/2)
    temp_U = U_previous + 0.5 * k1;
    k2 = dt_ * rhs_function(temp_U, current_time + 0.5 * dt_);

    // k3 = dt * f(t_n + dt/2, U_n + k2/2)
    temp_U = U_previous + 0.5 * k2;
    k3 = dt_ * rhs_function(temp_U, current_time + 0.5 * dt_);

    // k4 = dt * f(t_n + dt, U_n + k3)
    temp_U = U_previous + k3;
    k4 = dt_ * rhs_function(temp_U, current_time + dt_);

    // U_{n+1} = U_n + (k1 + 2*k2 + 2*k3 + k4)/6
    U_current = U_previous + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;

    return true;
}
```

### Step 6: Testing New Schemes

Create comprehensive tests for your new schemes:

```cpp
// Example test for Forward Euler
#include "utils/time_scheme.hpp"
#include <iostream>
#include <cmath>

void test_forward_euler() {
    std::cout << "Testing Forward Euler scheme..." << std::endl;

    // Create simple test problem: du/dt + u = 0, u(0) = 1
    // Exact solution: u(t) = exp(-t)

    int n = 3;
    Eigen::SparseMatrix<double> M_h(n, n);
    Eigen::SparseMatrix<double> K_h(n, n);

    // Identity mass matrix
    for (int i = 0; i < n; ++i) {
        M_h.insert(i, i) = 1.0;
        K_h.insert(i, i) = 1.0;  // K = I for this test
    }

    M_h.makeCompressed();
    K_h.makeCompressed();

    // Create time scheme
    utils::time_scheme time_integrator(M_h, K_h);

    // Small time step for stability (Forward Euler is conditionally stable)
    double dt = 0.001;
    time_integrator.setup_time_parameters(dt, utils::time_scheme::SchemeType::FORWARD_EULER);
    time_integrator.configure_solver(utils::time_scheme::SolverType::DIRECT, 1e-12, 1000);

    // Initial condition
    Eigen::VectorXd U_current(n);
    U_current << 1.0, 1.0, 1.0;

    double current_time = 0.0;
    double final_time = 1.0;

    // Time stepping
    while (current_time < final_time - 1e-12) {
        current_time += dt;

        Eigen::VectorXd U_previous = U_current;
        Eigen::VectorXd F_current = Eigen::VectorXd::Zero(n);  // No forcing

        bool success = time_integrator.step(U_current, U_previous, F_current);

        if (!success) {
            std::cerr << "Forward Euler step failed!" << std::endl;
            return;
        }
    }

    // Check against exact solution
    double exact_solution = std::exp(-final_time);
    double numerical_solution = U_current(0);
    double error = std::abs(exact_solution - numerical_solution);

    std::cout << "Final time: " << final_time << std::endl;
    std::cout << "Exact solution: " << exact_solution << std::endl;
    std::cout << "Numerical solution: " << numerical_solution << std::endl;
    std::cout << "Error: " << error << std::endl;

    // Test passes if error is reasonable for this time step
    if (error < 0.01) {
        std::cout << "✅ Forward Euler test PASSED" << std::endl;
    } else {
        std::cout << "❌ Forward Euler test FAILED" << std::endl;
    }
}
```

### Step 7: Usage Example for New Scheme

```cpp
// Example usage of Forward Euler
#include "utils/time_scheme.hpp"

int main() {
    // Setup matrices (same as before)
    int n = 100;
    Eigen::SparseMatrix<double> M_h(n, n);
    Eigen::SparseMatrix<double> K_h(n, n);

    // ... matrix setup code ...

    // Create time scheme with Forward Euler
    utils::time_scheme time_integrator(M_h, K_h);

    // IMPORTANT: Use small time step for stability!
    double dt = 0.0001;  // Much smaller than implicit methods
    time_integrator.setup_time_parameters(dt, utils::time_scheme::SchemeType::FORWARD_EULER);

    // Configure solver (Forward Euler still needs to solve M*U = RHS)
    time_integrator.configure_solver(utils::time_scheme::SolverType::DIRECT, 1e-12, 1000);

    // Check stability
    if (!time_integrator.is_stable()) {
        std::cout << "WARNING: Forward Euler is conditionally stable!" << std::endl;
        std::cout << "Ensure dt < dt_critical for your problem." << std::endl;
    }

    // Time stepping loop (same as other schemes)
    Eigen::VectorXd U_current(n);
    U_current.setOnes();  // Initial condition

    double current_time = 0.0;
    double final_time = 1.0;

    while (current_time < final_time - 1e-12) {
        current_time += dt;

        Eigen::VectorXd U_previous = U_current;
        Eigen::VectorXd F_current = Eigen::VectorXd::Zero(n);  // Example forcing

        bool success = time_integrator.step(U_current, U_previous, F_current);

        if (!success) {
            std::cerr << "Time step failed at t = " << current_time << std::endl;
            break;
        }
    }

    std::cout << "Forward Euler integration completed!" << std::endl;
    return 0;
}
```

### Best Practices for Extensions

1. **Stability Analysis**: Always analyze the stability properties of your new scheme and document them clearly.

2. **Error Handling**: Implement robust error checking, especially for conditionally stable schemes.

3. **Performance Considerations**: Consider the computational cost per step and memory requirements.

4. **Validation**: Test against known analytical solutions and compare with established methods.

5. **Documentation**: Update all relevant documentation, including mathematical formulations and usage examples.

6. **Backward Compatibility**: Ensure that existing code continues to work after adding new schemes.

7. **Common Pitfalls to Avoid**:

8. **Forgetting to Update Utility Methods**: Always update `is_stable()`, `get_order()`, and `get_scheme_name()`.

9. **Incorrect Time Step Handling**: Be careful with time step restrictions for conditionally stable schemes.

10. **Matrix Operations**: Ensure proper handling of sparse matrices and avoid unnecessary copies.

11. **Memory Management**: For multi-step methods, properly manage solution history storage.

12. **Numerical Precision**: Use appropriate tolerances and handle ill-conditioned systems.

### Integration with VEM Solver

When extending the time scheme for VEM applications, consider:

```cpp
// Example: Specialized method for VEM applications
bool time_scheme::step_vem_optimized(
    Eigen::VectorXd& U_current,
    const Eigen::VectorXd& U_previous,
    const Eigen::VectorXd& F_current,
    const solver::parabolic& vem_solver  // Access to VEM-specific information
){
    // VEM-specific optimizations
    // - Exploit VEM matrix structure
    // - Use element-wise operations
    // - Apply VEM-specific preconditioning

    return step(U_current, U_previous, F_current);
}
```

This comprehensive guide provides the foundation for extending the `time_scheme` class with new integration methods while maintaining code quality and performance.

## Numerical Examples

### Stability Analysis

For the model problem $\frac{du}{dt} + \lambda u = 0$ with $\lambda > 0$:

**Backward Euler**:

- Amplification factor: $G = \frac{1}{1 + \lambda \Delta t}$
- Stable for all $\Delta t > 0$ since $|G| < 1$

**Crank-Nicolson**:

- Amplification factor: $G = \frac{1 - \frac{\lambda \Delta t}{2}}{1 + \frac{\lambda \Delta t}{2}}$
- Stable for all $\Delta t > 0$ since $|G| \leq 1$

### Accuracy Comparison

For smooth solutions, the global error behaves as:

- Backward Euler: $\mathcal{O}(\Delta t)$
- Crank-Nicolson: $\mathcal{O}(\Delta t^2)$

This means halving the time step reduces the error by a factor of 2 for Backward Euler and by a factor of 4 for Crank-Nicolson.

## Future Extensions

### Potential Enhancements

1. **Adaptive Time Stepping**: Automatic $\Delta t$ selection based on error estimates
2. **Higher-Order Methods**: BDF methods, Runge-Kutta schemes
3. **Preconditioning**: Custom preconditioners for iterative solvers
4. **Parallel Solvers**: Multi-threaded linear algebra operations
5. **Non-Linear Systems**: Newton-Raphson iteration for non-linear problems

### Implementation Considerations

- **Template Specialization**: Support for different floating-point precisions
- **GPU Acceleration**: CUDA-based sparse linear algebra
- **Memory Optimization**: Block-structured matrices for VEM applications

## Conclusion

The `time_scheme` class provides a robust and flexible framework for time integration in Virtual Element Method applications. The implementation emphasizes:

- **Mathematical Rigor**: Proper implementation of established numerical methods
- **Computational Efficiency**: Optimized linear algebra operations
- **Reliability**: Comprehensive error handling and validation
- **Extensibility**: Clean architecture for future enhancements

The class successfully balances theoretical correctness with practical implementation concerns, making it suitable for both research and production VEM codes.
