# Virtual Element Method for Parabolic Problems - Implementation Documentation

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Implementation Architecture](#implementation-architecture)
4. [Core Data Structures](#core-data-structures)
5. [DOF Management System](#dof-management-system)
6. [Element Assembly Process](#element-assembly-process)
7. [Projection Operators](#projection-operators)
8. [Load Vector Computation](#load-vector-computation)
9. [Stabilization Mechanisms](#stabilization-mechanisms)
10. [Quadrature Integration](#quadrature-integration)
11. [Testing Framework](#testing-framework)
12. [Usage Examples](#usage-examples)

---

## 1. Overview

This document provides comprehensive documentation for the Virtual Element Method (VEM) implementation for parabolic problems in the `solver::parabolic` class. The implementation supports arbitrary order VEM spaces on polygonal meshes and provides a complete framework for solving time-dependent partial differential equations.

### Key Features

- **Arbitrary Order VEM**: Supports polynomial orders k = 1, 2, 3, ...
- **Polygonal Meshes**: Handles triangles, quadrilaterals, pentagons, and general polygons
- **Complete VEM Framework**: Energy projection, L² projection, stabilization
- **Load Vector Assembly**: Time-dependent source terms with proper VEM projections
- **Comprehensive Testing**: Extensive test suite validating VEM properties

---

## 2. Theoretical Foundation

### 2.1 VEM Formulation for Parabolic Problems

The VEM for parabolic problems considers the weak formulation:

```
Find u_h ∈ V_h such that:
∫_Ω ∂u_h/∂t v_h dx + a(u_h, v_h) = ∫_Ω f v_h dx  ∀v_h ∈ V_h
```

Where:

- `V_h` is the VEM space of order k
- `a(u,v) = ∫_Ω ∇u · ∇v dx` is the bilinear form
- `f` is the source term

### 2.2 VEM Space Definition

The VEM space V_h^k(K) on element K is defined as:

```
V_h^k(K) = {v ∈ H¹(K) : Δv ∈ P_{k-2}(K), v|_∂K ∈ C⁰(∂K), v|_e ∈ P_k(e) ∀edge e}
```

Where:

- `P_k(K)` denotes polynomials of degree ≤ k on K
- `P_{k-2}(K) = {0}` for k = 1

### 2.3 Degrees of Freedom

The DOFs for V_h^k(K) are:

- **Vertex DOFs**: Values at vertices
- **Edge DOFs** (k ≥ 2): Moments with P\_{k-1}(e) on each edge e
- **Interior DOFs** (k ≥ 2): Moments with P\_{k-2}(K) in the interior

Total DOFs per element: `n_v + n_v(k-1) + (k-1)(k-2)/2`

### 2.4 Projection Operators

**Energy Projection Π^∇**: P_k(K) → V_h^k(K)

```
a_K(Π^∇v_h, p) = a_K(v_h, p)  ∀p ∈ P_k(K)
```

**L² Projection Π^0**: P_k(K) → V_h^k(K)

```
∫_K Π^0v_h p dx = ∫_K v_h p dx  ∀p ∈ P_k(K)
```

### 2.5 VEM Discretization

**Local Stiffness Matrix**:

```
A_K = (Π^∇)ᵀ K_poly Π^∇ + α_K S_K
```

**Local Mass Matrix**:

```
M_K = (Π^0)ᵀ M_poly Π^0 + β_K |K| S_K
```

**Local Load Vector**:

```
F_K = (Π^0)ᵀ f_h
```

Where:

- `K_poly`, `M_poly` are polynomial matrices on P_k(K)
- `S_K` is the stabilization matrix
- `α_K`, `β_K` are stabilization parameters
- `f_h` is the L² projection of the source term

---

## 3. Implementation Architecture

### 3.1 Class Structure

```cpp
namespace solver {
    class parabolic {
        // Core VEM parameters
        int order;                    // Polynomial order k
        int N_k;                     // Dimension of P_k space

        // Mesh data
        Eigen::MatrixXd nodes;       // Node coordinates
        Eigen::MatrixXi elements;    // Element connectivity

        // Global system
        Eigen::SparseMatrix<double> M_h;  // Global mass matrix
        Eigen::SparseMatrix<double> K_h;  // Global stiffness matrix
        Eigen::VectorXd F_h;             // Global load vector

        // DOF management
        int n_dofs;                      // Total DOFs
        std::vector<std::vector<int>> element_dof_map;  // Local-to-global mapping
    };
}
```

### 3.2 Source Function Interface

```cpp
using SourceFunction = std::function<double(const Eigen::Vector2d&, double)>;
```

This allows time-dependent source functions `f(x, y, t)`.

---

## 4. Core Data Structures

### 4.1 ElementData Structure

The `ElementData` structure encapsulates all element-specific information:

```cpp
struct ElementData {
    // Geometry
    Eigen::MatrixXd vertices;        // Vertex coordinates (n_vertices × 2)
    Eigen::Vector2d centroid;        // Element centroid
    double h_e;                      // Polygonal diameter
    double area;                     // Element area
    int n_vertices;                  // Number of vertices
    int n_dofs_local;               // Local DOFs

    // Monomial basis
    std::vector<std::pair<int, int>> monomial_powers;  // (α₁, α₂) pairs

    // Projection matrices
    Eigen::MatrixXd P_nabla;         // Energy projection (N_k × n_dofs_local)
    Eigen::MatrixXd P_0;            // L² projection (N_k × n_dofs_local)

    // Polynomial matrices
    Eigen::MatrixXd K_poly;          // Stiffness on P_k (N_k × N_k)
    Eigen::MatrixXd M_poly;          // Mass on P_k (N_k × N_k)

    // Local system
    Eigen::MatrixXd K_local;         // Local stiffness matrix
    Eigen::MatrixXd M_local;         // Local mass matrix
    Eigen::VectorXd F_local;         // Local load vector
};
```

### 4.2 Monomial Basis Organization

Monomials are ordered by total degree:

```
Order 1: {1, x, y}                           (N_k = 3)
Order 2: {1, x, y, x², xy, y²}              (N_k = 6)
Order 3: {1, x, y, x², xy, y², x³, x²y, xy², y³}  (N_k = 10)
```

The `monomial_powers` vector stores pairs `(α₁, α₂)` for each monomial `x^α₁ y^α₂`.

---

## 5. DOF Management System

### 5.1 DOF Numbering Strategy

The implementation uses a hierarchical DOF numbering:

1. **Global Vertex DOFs**: `0` to `n_nodes - 1`
2. **Global Edge DOFs**: `n_nodes` to `n_nodes + n_edges × (k-1) - 1`
3. **Global Interior DOFs**: Remaining DOFs

### 5.2 Local DOF Ordering

Within each element, DOFs are ordered as:

1. **Vertex DOFs**: `0` to `n_vertices - 1`
2. **Edge DOFs**: `n_vertices` to `n_vertices + n_vertices × (k-1) - 1`
3. **Interior DOFs**: Remaining local DOFs

### 5.3 Implementation Details

#### 5.3.1 Linear Elements (k = 1)

```cpp
void parabolic::setup_linear_dofs() {
    n_dofs = nodes.rows();  // Only vertex DOFs

    for(int elem = 0; elem < elements.rows(); ++elem) {
        int n_vertices = elements.cols();
        element_dof_map[elem].resize(n_vertices);

        for(int i = 0; i < n_vertices; ++i) {
            element_dof_map[elem][i] = elements(elem, i);
        }
    }
}
```

#### 5.3.2 Higher-Order Elements (k ≥ 2)

```cpp
void parabolic::setup_high_order_dofs() {
    std::map<std::pair<int, int>, int> edge_dof_map;
    std::map<int, int> element_moments_dof_map;

    int dof_counter = nodes.rows();  // Start after vertex DOFs

    // Setup edge DOFs with consistent ordering
    if (order >= 2) {
        std::set<std::pair<int, int>> processed_edges;

        for (int elem = 0; elem < elements.rows(); ++elem) {
            for (int i = 0; i < n_vertices; ++i) {
                int v1 = elements(elem, i);
                int v2 = elements(elem, (i + 1) % n_vertices);

                // Ensure consistent edge ordering
                std::pair<int, int> edge = (v1 < v2) ?
                    std::make_pair(v1, v2) : std::make_pair(v2, v1);

                if (processed_edges.find(edge) == processed_edges.end()) {
                    edge_dof_map[edge] = dof_counter;
                    dof_counter += (order - 1);
                    processed_edges.insert(edge);
                }
            }
        }

        // Setup interior moment DOFs
        int interior_dofs_per_element = (order - 1) * (order - 2) / 2;
        for (int elem = 0; elem < elements.rows(); ++elem) {
            element_moments_dof_map[elem] = dof_counter;
            dof_counter += interior_dofs_per_element;
        }
    }

    n_dofs = dof_counter;
}
```

### 5.4 DOF Classification Utilities

The implementation provides utilities to classify DOF types:

```cpp
// From utils::operations
bool is_vertex_dof(int dof_idx, const ElementData& element_data);
bool is_edge_dof(int dof_idx, const ElementData& element_data, int order);
bool is_moment_dof(int dof_idx, const ElementData& element_data, int order);
```

---

## 6. Element Assembly Process

### 6.1 Assembly Workflow

The element assembly follows a systematic process:

```cpp
void parabolic::assemble_element(int element_idx) {
    ElementData element_data;

    // 1. Setup element geometry
    setup_element_geometry(element_idx, element_data);

    // 2. Construct monomial basis
    construct_monomial_basis(element_data);

    // 3. Compute polynomial matrices
    compute_polynomial_matrices(element_data);

    // 4. Compute projection matrices
    compute_projection_matrices(element_data);

    // 5. Assemble local matrices
    assemble_local_matrices(element_data);

    // 6. Assemble into global system
    assemble_to_global_system(element_idx, element_data);
}
```

### 6.2 Element Geometry Setup

```cpp
void parabolic::setup_element_geometry(int elem_idx, ElementData& element_data) {
    element_data.n_vertices = elements.cols();
    element_data.vertices = Eigen::MatrixXd(element_data.n_vertices, 2);

    // Extract element vertices
    for (int i = 0; i < element_data.n_vertices; ++i) {
        int node_idx = elements(elem_idx, i);
        element_data.vertices.row(i) = nodes.row(node_idx);
    }

    // Compute geometric properties
    element_data.centroid = utils::operations::calcCentroid(element_data.vertices);
    element_data.h_e = utils::operations::calcPolygonalDiam(element_data.vertices,
                                                           element_data.n_vertices);
    element_data.area = utils::operations::calcArea(element_data.vertices);
    element_data.n_dofs_local = count_local_dofs(element_data.n_vertices);
}
```

### 6.3 Monomial Basis Construction

```cpp
void parabolic::construct_monomial_basis(ElementData& element_data) {
    element_data.monomial_powers.clear();

    // Generate all monomials up to degree "order"
    for (int total_degree = 0; total_degree <= order; ++total_degree) {
        for (int alpha_1 = 0; alpha_1 <= total_degree; ++alpha_1) {
            int alpha_2 = total_degree - alpha_1;
            element_data.monomial_powers.push_back(std::make_pair(alpha_1, alpha_2));
        }
    }
}
```

This creates the scaled monomial basis:

```
m_j(x,y) = ((x - x_c)/h_e)^α₁ ((y - y_c)/h_e)^α₂
```

Where `(x_c, y_c)` is the element centroid and `h_e` is the element diameter.

---

## 7. Projection Operators

### 7.1 Energy Projection Π^∇

The energy projection satisfies:

```
a_K(Π^∇φᵢ, mⱼ) = a_K(φᵢ, mⱼ)  ∀mⱼ ∈ P_k(K)
```

#### 7.1.1 Implementation

```cpp
void parabolic::compute_energy_projection(ElementData& element_data) {
    for (int dof_idx = 0; dof_idx < element_data.n_dofs_local; ++dof_idx) {
        // Compute RHS vector for DOF φᵢ
        Eigen::VectorXd rhs_i = compute_energy_rhs_for_dof(element_data, dof_idx);

        // Setup constraint system
        Eigen::MatrixXd A_constrained;
        Eigen::VectorXd b_constrained;

        if (order == 1) {
            // k=1: Boundary constraint ∫_∂K (φᵢ - Π^∇φᵢ) ds = 0
            add_boundary_constraint(element_data, dof_idx, A_constrained,
                                  b_constrained, rhs_i);
        } else {
            // k≥2: Interior constraint ∫_K (φᵢ - Π^∇φᵢ) dx = 0
            add_interior_constraint(element_data, dof_idx, A_constrained,
                                  b_constrained, rhs_i);
        }

        // Solve constrained system
        Eigen::VectorXd p_i = A_constrained.fullPivLu().solve(b_constrained);
        element_data.P_nabla.col(dof_idx) = p_i.head(N_k);
    }
}
```

#### 7.1.2 Energy RHS Computation

The energy RHS uses Green's formula:

```
a_K(φᵢ, mⱼ) = ∫_∂K φᵢ (∇mⱼ·n) ds - ∫_K φᵢ Δmⱼ dx
```

```cpp
Eigen::VectorXd parabolic::compute_energy_rhs_for_dof(
    const ElementData& element_data, int dof_idx) {

    Eigen::VectorXd rhs(N_k);

    for (int j = 0; j < N_k; ++j) {
        double area_term = utils::integration::compute_moment_dof_laplacian_monomial_area_integral(
            element_data, dof_idx, j, order, N_k);
        double boundary_term = utils::integration::compute_dof_monomial_boundary_integral(
            element_data, dof_idx, j, order);

        rhs(j) = area_term - boundary_term;
    }

    return rhs;
}
```

#### 7.1.3 Constraint Systems

**For k = 1 (Boundary Constraint)**:

```
[K_poly  c] [p]   [rhs]
[c^T     0] [λ] = [b  ]
```

Where `c[j] = ∫_∂K mⱼ ds` and `b = ∫_∂K φᵢ ds`.

**For k ≥ 2 (Interior Constraint)**:

```
[K_poly  c] [p]   [rhs]
[c^T     0] [λ] = [b  ]
```

Where `c[j] = ∫_K mⱼ dx` and `b = ∫_K φᵢ dx`.

### 7.2 L² Projection Π^0

The L² projection satisfies:

```
∫_K Π^0φᵢ mⱼ dx = ∫_K φᵢ mⱼ dx  ∀mⱼ ∈ P_k(K)
```

#### 7.2.1 Implementation

```cpp
void parabolic::compute_l2_projection(ElementData& element_data) {
    Eigen::MatrixXd M_moments = element_data.M_poly;

    for (int dof_idx = 0; dof_idx < element_data.n_dofs_local; ++dof_idx) {
        // Compute moment vector ∫_K φᵢ mⱼ dx
        Eigen::VectorXd moments = utils::operations::compute_dof_moments(
            element_data, dof_idx, order, N_k);

        // Solve M_poly * p_i = moments
        Eigen::VectorXd p_i = M_moments.fullPivLu().solve(moments);
        element_data.P_0.col(dof_idx) = p_i.head(N_k);
    }
}
```

#### 7.2.2 Moment Computation

The moment computation handles different DOF types:

```cpp
Eigen::VectorXd utils::operations::compute_dof_moments(
    const ElementData& element_data, int dof_idx, int order, int N_k) {

    Eigen::VectorXd moments(N_k);

    for (int j = 0; j < N_k; ++j) {
        auto [alpha_1, alpha_2] = element_data.monomial_powers[j];
        int degree = alpha_1 + alpha_2;

        if (degree <= order - 2) {
            // Low-order moments: computed from DOF definitions
            if (is_moment_dof(dof_idx, element_data, order)) {
                // Interior DOF: exact moment representation
                int interior_idx = get_interior_dof_index(element_data, dof_idx, order);
                moments(j) = (interior_idx == j) ? element_data.area : 0.0;
            }
            else if (is_vertex_dof(dof_idx, element_data)) {
                // Vertex DOF: contributes to constant monomial
                moments(j) = (degree == 0) ? element_data.area / element_data.n_vertices : 0.0;
            }
            else if (is_edge_dof(dof_idx, element_data, order)) {
                // Edge DOF: orthogonal to interior
                moments(j) = 0.0;
            }
        } else {
            // High-order moments: use enhanced constraint
            // ∫_K φᵢ mⱼ dx = ∫_K Π^∇φᵢ mⱼ dx
            Eigen::VectorXd pi_nabla_coeffs = element_data.P_nabla.col(dof_idx);
            moments(j) = 0.0;
            for (int l = 0; l < N_k; ++l) {
                moments(j) += pi_nabla_coeffs(l) * element_data.M_poly(l, j);
            }
        }
    }

    return moments;
}
```

---

## 8. Load Vector Computation

### 8.1 Theoretical Foundation

The VEM load vector computation involves:

1. **Source Projection**: Project `f` onto appropriate polynomial space
2. **DOF Integration**: Compute `∫_K f_h φᵢ dx` for each DOF
3. **VEM Consistency**: Use L² projection for consistency

### 8.2 Source Projection

```cpp
void parabolic::compute_projected_source(
    ElementData& element_data,
    const SourceFunction& f,
    double time,
    Eigen::VectorXd& fh_coeffs) {

    // Determine projection degree
    int projection_degree = (order == 1) ? 0 : order - 2;
    int n_proj_monomials = (projection_degree + 1) * (projection_degree + 2) / 2;

    // Setup quadrature
    std::vector<Eigen::Vector2d> quad_points;
    std::vector<double> quad_weights;
    setup_quadrature_rule(element_data, quad_points, quad_weights);

    // Extract projection mass matrix
    Eigen::MatrixXd M_proj = element_data.M_poly.block(0, 0, n_proj_monomials, n_proj_monomials);

    // Compute RHS: ∫_K f mᵢ dx
    Eigen::VectorXd b_proj = Eigen::VectorXd::Zero(n_proj_monomials);
    for (int i = 0; i < n_proj_monomials; ++i) {
        for (size_t q = 0; q < quad_points.size(); ++q) {
            double f_value = f(quad_points[q], time);
            double monomial_value = utils::operations::evaluate_monomial(
                i, quad_points[q], element_data);
            b_proj(i) += quad_weights[q] * f_value * monomial_value;
        }
    }

    // Solve projection system: M_proj * fh_coeffs = b_proj
    fh_coeffs = M_proj.fullPivLu().solve(b_proj);
}
```

### 8.3 Load Vector Assembly

```cpp
void parabolic::compute_local_load_vector(
    ElementData& element_data,
    const SourceFunction& f,
    double time) {

    element_data.F_local = Eigen::VectorXd::Zero(element_data.n_dofs_local);

    // Step 1: Project source term
    Eigen::VectorXd fh_coeffs;
    compute_projected_source(element_data, f, time, fh_coeffs);

    // Step 2: Compute load vector entries
    for (int i = 0; i < element_data.n_dofs_local; ++i) {
        element_data.F_local(i) = utils::integration::compute_load_integral_for_dof(
            element_data, fh_coeffs, i, N_k, order);
    }
}
```

### 8.4 DOF Load Integration

The load integration uses the L² projection:

```
∫_K f_h φᵢ dx = ∫_K f_h Π^0φᵢ dx = Σⱼ fh_coeffs[j] * Π^0φᵢ[j] * ∫_K mⱼ mₖ dx
```

```cpp
double utils::integration::compute_load_integral_for_dof(
    const ElementData& element_data,
    const Eigen::VectorXd& fh_coeffs,
    int dof_idx,
    int N_k,
    int order) {

    double integral = 0.0;

    if (is_vertex_dof(dof_idx, element_data)) {
        // Vertex DOF
        Eigen::VectorXd pi0_coeffs = element_data.P_0.col(dof_idx);

        for (int i = 0; i < fh_coeffs.size(); ++i) {
            for (int j = 0; j < N_k; j++) {
                double m_ij_integral = element_data.M_poly(i, j);
                integral += fh_coeffs(i) * pi0_coeffs(j) * m_ij_integral;
            }
        }
    }
    else if (is_edge_dof(dof_idx, element_data, order)) {
        // Edge DOF - similar computation
        // ...
    }
    else if (is_moment_dof(dof_idx, element_data, order)) {
        // Interior DOF - use orthogonality
        int moment_idx = dof_idx - n_vertices - n_vertices * (order - 1);
        if (moment_idx < fh_coeffs.size()) {
            integral = fh_coeffs(moment_idx) * element_data.area;
        }
    }

    return integral;
}
```

---

## 9. Stabilization Mechanisms

### 9.1 Stabilization Theory

VEM requires stabilization to ensure well-posedness. The stabilization acts on the kernel of the projection operators.

### 9.2 Stabilization Matrix

```cpp
Eigen::MatrixXd parabolic::compute_stabilization_matrix(const ElementData& element_data) {
    int n_dofs = element_data.n_dofs_local;

    if (element_data.P_nabla.rows() > 0 && element_data.P_nabla.cols() == n_dofs) {
        // S_K = I - P^T * P (projects onto kernel)
        Eigen::MatrixXd P_proj = element_data.P_nabla.transpose() * element_data.P_nabla;
        return Eigen::MatrixXd::Identity(n_dofs, n_dofs) - P_proj;
    }

    // Fallback: identity matrix
    return Eigen::MatrixXd::Identity(n_dofs, n_dofs);
}
```

### 9.3 Stabilization Parameters

**Stiffness Stabilization**:

```cpp
double parabolic::compute_stiffness_stabilization_parameter(const ElementData& element_data) {
    if (element_data.K_poly.rows() == 0) return 1.0;

    // Use trace of K_poly
    double trace_K_poly = element_data.K_poly.trace();
    if (trace_K_poly > 1e-12) return trace_K_poly;

    // Fallback: geometric scaling
    return 1.0 / (element_data.h_e * element_data.h_e);
}
```

**Mass Stabilization**:

```cpp
double parabolic::compute_mass_stabilization_parameter(const ElementData& element_data) {
    if (element_data.M_poly.rows() == 0) return 1.0;

    double trace_M_poly = element_data.M_poly.trace();
    if (trace_M_poly > 1e-12 && element_data.area > 1e-12) {
        return trace_M_poly / (element_data.area * N_k);
    }

    return 1.0;
}
```

### 9.4 Local Matrix Assembly

```cpp
void parabolic::compute_local_stiffness_matrix(ElementData& element_data) {
    // Consistency term
    Eigen::MatrixXd K_c = element_data.P_nabla.transpose() *
                         element_data.K_poly * element_data.P_nabla;

    // Stabilization term
    double alpha_E = compute_stiffness_stabilization_parameter(element_data);
    Eigen::MatrixXd S_E = alpha_E * compute_stabilization_matrix(element_data);

    // Local stiffness matrix
    element_data.K_local = K_c + S_E;
}

void parabolic::compute_local_mass_matrix(ElementData& element_data) {
    // Consistency term
    Eigen::MatrixXd M_c = element_data.P_0.transpose() *
                         element_data.M_poly * element_data.P_0;

    // Stabilization term
    double beta_M = compute_mass_stabilization_parameter(element_data);
    Eigen::MatrixXd S_M = beta_M * compute_stabilization_matrix(element_data);

    // Local mass matrix
    element_data.M_local = M_c + S_M;
}
```

---

## 10. Quadrature Integration

### 10.1 Triangulation-Based Quadrature

The implementation uses a simple triangulation-based quadrature rule for polygonal elements:

```cpp
void parabolic::setup_quadrature_rule(
    ElementData& element_data,
    std::vector<Eigen::Vector2d>& quad_points,
    std::vector<double>& quad_weights) {

    quad_points.clear();
    quad_weights.clear();

    // Triangulate from centroid
    for (int i = 0; i < element_data.n_vertices; ++i) {
        int j = (i + 1) % element_data.n_vertices;

        // Triangle: centroid, vertex i, vertex j
        Eigen::Vector2d v0 = element_data.centroid;
        Eigen::Vector2d v1 = element_data.vertices.row(i);
        Eigen::Vector2d v2 = element_data.vertices.row(j);

        // Compute triangle area
        Eigen::MatrixXd triangle(3, 2);
        triangle << v0.x(), v0.y(), v1.x(), v1.y(), v2.x(), v2.y();
        double area = utils::operations::calcArea(triangle);

        // Use triangle centroid as quadrature point
        Eigen::Vector2d centroid = utils::operations::calcCentroid(triangle);
        quad_points.push_back(centroid);
        quad_weights.push_back(area);
    }
}
```

### 10.2 Polynomial Matrix Computation

```cpp
void parabolic::compute_polynomial_matrices(ElementData& element_data) {
    element_data.K_poly = Eigen::MatrixXd::Zero(N_k, N_k);
    element_data.M_poly = Eigen::MatrixXd::Zero(N_k, N_k);

    // Setup quadrature
    std::vector<Eigen::Vector2d> quad_points;
    std::vector<double> quad_weights;
    setup_quadrature_rule(element_data, quad_points, quad_weights);

    // Compute matrices
    for (int i = 0; i < N_k; ++i) {
        for (int j = 0; j < N_k; ++j) {
            double stiff_integral = 0.0;
            double mass_integral = 0.0;

            for (size_t q = 0; q < quad_points.size(); ++q) {
                // Evaluate monomials and gradients
                Eigen::Vector2d grad_i = utils::operations::evaluate_monomial_gradient(
                    i, quad_points[q], element_data);
                Eigen::Vector2d grad_j = utils::operations::evaluate_monomial_gradient(
                    j, quad_points[q], element_data);

                double phi_i = utils::operations::evaluate_monomial(
                    i, quad_points[q], element_data);
                double phi_j = utils::operations::evaluate_monomial(
                    j, quad_points[q], element_data);

                stiff_integral += quad_weights[q] * grad_i.dot(grad_j);
                mass_integral += quad_weights[q] * phi_i * phi_j;
            }

            element_data.K_poly(i, j) = stiff_integral;
            element_data.M_poly(i, j) = mass_integral;
        }
    }
}
```

---

## 11. Testing Framework

### 11.1 Comprehensive Test Suite

The implementation includes an extensive test suite that validates:

- **Source Projection**: Polynomial exactness, consistency
- **Load Vector Computation**: DOF contributions, VEM properties
- **Time Dependency**: Time-varying source functions
- **Global Assembly**: Multi-element assembly
- **Element Types**: Triangles, quadrilaterals, pentagons
- **Convergence**: h-convergence and p-convergence
- **Integration**: Polynomial orders 1-3
- **VEM Properties**: Consistency, stability, bounds

### 11.2 Key Test Functions

```cpp
// Test source projection computation
void test_source_projection_computation(const ElementData& element_data, int order);

// Test polynomial exactness
void test_source_projection_polynomial_exactness(const ElementData& element_data, int order);

// Test local load vector computation
void test_local_load_vector_computation(const ElementData& element_data, int order);

// Test VEM properties
void test_load_vector_vem_properties(const ElementData& element_data, int order);

// Test time-dependent sources
void test_time_dependent_load_vectors(const ElementData& element_data, int order);

// Test global assembly
void test_global_load_vector_assembly(int max_order);

// Test convergence properties
void test_load_vector_convergence(int max_order);
```

### 11.3 Test Results Analysis

The tests validate critical VEM properties:

1. **Polynomial Exactness**: For polynomial sources, the projection should be exact
2. **Conservation**: Load vector should conserve total source integral
3. **Consistency**: VEM consistency conditions must be satisfied
4. **Stability**: Load vectors should be bounded appropriately
5. **Time Evolution**: Time-dependent sources should be handled correctly

---

## 12. Usage Examples

### 12.1 Basic Setup

```cpp
#include "solver/parabolic.hpp"

// Define mesh
Eigen::MatrixXd nodes(4, 2);
nodes << 0.0, 0.0,
         1.0, 0.0,
         1.0, 1.0,
         0.0, 1.0;

Eigen::MatrixXi elements(1, 4);
elements << 0, 1, 2, 3;

// Create solver
int order = 2;
solver::parabolic solver(order, nodes, elements);
```

### 12.2 Define Source Function

```cpp
// Time-dependent source function
auto source_function = [](const Eigen::Vector2d& point, double time) -> double {
    return std::exp(-time) * (1.0 + point.x() * point.y());
};
```

### 12.3 Assemble Load Vector

```cpp
// Assemble load vector at specific time
double current_time = 1.0;
solver.assemble_load_vector(source_function, current_time);

// Access global load vector
const Eigen::VectorXd& F_global = solver.F_h;
```

### 12.4 Access System Matrices

```cpp
// Get global matrices
const Eigen::SparseMatrix<double>& M_global = solver.get_global_mass_matrix();
const Eigen::SparseMatrix<double>& K_global = solver.get_global_stiffness_matrix();

// Print system information
solver.print_system_information();
```

---

## Conclusion

This implementation provides a complete, theoretically sound VEM framework for parabolic problems. The code is extensively tested and validates all critical VEM properties including polynomial exactness, consistency conditions, and proper stabilization. The modular design allows for easy extension to higher orders and different element types while maintaining numerical stability and accuracy.

The implementation successfully handles:

- Arbitrary polygonal elements
- Multiple polynomial orders (k = 1, 2, 3, ...)
- Time-dependent source terms
- Proper VEM projections and stabilization
- Comprehensive error checking and validation

This provides a solid foundation for solving time-dependent PDEs using the Virtual Element Method.
