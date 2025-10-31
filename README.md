# POLI-VEM

A C++ library implementing the Virtual Element Method (VEM) for structural analysis.

## Project Structure

The project follows a modern C++ organization with a clear separation between interface and implementation:

```
polivem/
├── include/              # Public header files (API)
│   ├── mesh/             # Mesh generation and manipulation headers
│   ├── solver/           # Numerical solvers headers
│   ├── material/         # Material properties headers
│   ├── utils/            # Utility functions headers
│   └── models/           # Data models and enumerations
├── lib/                  # Implementation files
│   ├── mesh/             # Mesh implementation
│   ├── solver/           # Solver implementation
│   ├── material/         # Material implementation
│   └── utils/            # Utilities implementation
├── data/                 # Input/output data files
├── build/                # Build artifacts (generated)
├── CMakeLists.txt        # Main CMake configuration
└── bootstrap.sh          # Build script
```

## Requirements

- CMake 3.9.1 or higher
- C++17 compatible compiler
- Eigen3
- nlohmann_json
- OpenCV

In case the _Eigen_ library is not found by the CMake file, you can download it [here](https://eigen.tuxfamily.org/index.php?title=Main_Page). After downloading, copy the files under the Eigen subdirectory to the _include_ folder (same folder where the other libraries like _iostream_ are installed). In macOS, to find the path, you can press CMD+Click over a known library (e.g., _iostream_).

## Building and Running PoliVEM

To setup the project and install the dependencies, run the following commands:

```bash
./bootstrap.sh
```

Or manually:

```bash
cmake CMakeLists.txt
make
```

To run the solver:

```bash
./polivem
```

**Note for macOS users:** When Xcode updates, the error `'wchar.h' file not found` might occur. In that case, delete the files `cmake_install.cmake` and `CMakeCache.txt`. Then, just run the commands above again.

## Usage Examples

### 1D Beam Model

Initialize the beam object and define a discretization. In the example below, 0.45 refers to the beam length and 20 refers to the number of nodes in the discretization.

```cpp
#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"

// Initialize beam and discretization
mesh::beam bar;
bar.horizontalBarDisc(0.45, 20);

// Get nodes and elements from the discretization
Eigen::MatrixXd nodes = bar.nodes;
Eigen::MatrixXi elements = bar.elements;
```

To set a distributed load, define the extreme values:

```cpp
Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
q(0) = -1.0;
q(1) = -1.0;
```

Define the material properties:

```cpp
material::mat elastic;
elastic.setElasticModule(2.1e+11);
double E = elastic.E;
```

Initialize the solver with nodes, elements, and method order:

```cpp
int order = 5;
solver::beam1d solver(nodes, elements, order);
solver.setInertiaMoment((0.02*pow(0.003,3))/12);
```

Build the global stiffness matrix and apply static condensation:

```cpp
Eigen::MatrixXd K = solver.buildGlobalK(E);

Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");
```

Apply the distributed load vector:

```cpp
solver.setDistributedLoad(q, elements);
Eigen::VectorXd R = solver.buildGlobalDistributedLoad();
Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");
```

Apply Dirichlet boundary conditions (DBC) to restrict the displacement of the first node:

```cpp
Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1,4);
supp(0,0) = 0; // node index
supp(0,1) = 1; // restrict displacement w
supp(0,2) = 1; // restrict the rotation w'
supp(0,3) = 0; // always set 0 for horizontal bar
solver.setSupp(supp);
```

Build the final stiffness matrix and load vector:

```cpp
Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
Eigen::MatrixXd R_ = RI - KIM * KMM.inverse() * RM;

K_ = solver.applyDBCMatrix(K_);
R_ = solver.applyDBCVec(R_);
```

Solve the linear system:

```cpp
Eigen::VectorXd uh;
uh = K_.ldlt().solve(R_);
```

## Components

### Mesh Module

Handles mesh generation, manipulation, and storage. Includes implementations for:

- Beam elements
- Delaunay triangulation
- Voronoi mesh generation
- Data source handling for mesh I/O

### Solver Module

Implements numerical solvers for structural analysis:

- 1D beam solver
- 2D linear elastic solver
- Nonlinear solvers

### Material Module

Defines material properties and constitutive models:

- Elastic materials
- Material parameter handling

### Utils Module

Provides utility functions for:

- Geometric operations
- Numerical integration
- Logging

## Axisymmetric 2D Model

The axisymmetric solver implements the Virtual Element Method (VEM) for problems with rotational symmetry about the z-axis. This formulation is particularly useful for analyzing cylindrical structures, pressure vessels, and other rotationally symmetric geometries.

### Key Features

- **Virtual Element Method**: Supports arbitrary polygonal elements (triangles, quadrilaterals, pentagons, etc.)
- **Axisymmetric Formulation**: Reduces 3D problems to 2D by exploiting rotational symmetry
- **Multiple Stabilization Types**: Standard, divergence, and boundary stabilization methods
- **Strain Projection**: Accurate strain recovery using projection operators
- **Load Handling**: Body forces and boundary tractions with proper axisymmetric integration

### Mathematical Formulation

The axisymmetric formulation considers displacements in the r-z plane:

- **u_r**: Radial displacement
- **u_z**: Axial displacement

The strain vector includes:

- **ε_r**: Radial strain (∂u_r/∂r)
- **ε_z**: Axial strain (∂u_z/∂z)
- **ε_θ**: Hoop strain (u_r/r)
- **γ_rz**: Shear strain (∂u_r/∂z + ∂u_z/∂r)

### Usage Example

#### Basic Setup

```cpp
#include "solver/axisymmetric.hpp"
#include "material/mat.hpp"

// Define material properties
double E = 200e9;  // Young's modulus (Pa)
double nu = 0.3;   // Poisson's ratio

// Create element vertices (counter-clockwise ordering)
Eigen::MatrixXd element_vertices(4, 2);
element_vertices << 1.0, 0.0,    // (r1, z1)
                    2.0, 0.0,    // (r2, z2)
                    2.0, 1.0,    // (r3, z3)
                    1.0, 1.0;    // (r4, z4)
```

#### Computing Element Stiffness Matrix

```cpp
// Compute element stiffness matrix with different stabilization types
std::string stab_type = "boundary";  // Options: "standard", "divergence", "boundary"

Eigen::MatrixXd K_elem = solver::axisymmetric::compute_stiffness_matrix(
    element_vertices, E, nu, stab_type
);

std::cout << "Element stiffness matrix size: " << K_elem.rows() << "x" << K_elem.cols() << std::endl;
```

#### Load Vector Computation

```cpp
// Define body force function (returns radial and axial components)
auto body_force_func = [](double r, double z) -> std::pair<double, double> {
    double f_r = 0.0;           // Radial body force
    double f_z = -9810.0;       // Axial body force (gravity)
    return {f_r, f_z};
};

// Compute body force load vector
Eigen::VectorXd f_body = solver::axisymmetric::compute_element_load_body_force(
    element_vertices, body_force_func
);

// Define boundary traction function
auto traction_func = [](double r, double z) -> std::pair<double, double> {
    double t_r = 1000.0;        // Radial traction
    double t_z = 0.0;           // Axial traction
    return {t_r, t_z};
};

// Apply traction on specific edges (e.g., edge 0)
Eigen::VectorXi traction_edges(1);
traction_edges << 0;

Eigen::VectorXd f_traction = solver::axisymmetric::compute_element_load_boundary_traction(
    element_vertices, traction_edges, traction_func
);

// Total element load vector
Eigen::VectorXd f_total = f_body + f_traction;
```

#### Strain Recovery

```cpp
// Given nodal displacements for the element
Eigen::VectorXd displacements(8);  // 2 DOFs per vertex
displacements << u_r1, u_z1, u_r2, u_z2, u_r3, u_z3, u_r4, u_z4;

// Calculate strain at element centroid
Eigen::Vector4d strain = calculate_strain_at_centroid(element_vertices, displacements);

std::cout << "Strains at centroid:" << std::endl;
std::cout << "ε_r = " << strain[0] << std::endl;
std::cout << "ε_z = " << strain[1] << std::endl;
std::cout << "ε_θ = " << strain[2] << std::endl;
std::cout << "γ_rz = " << strain[3] << std::endl;
```

#### Patch Test Validation

The implementation includes comprehensive patch tests to verify accuracy:

```cpp
// Run VEM patch tests for axisymmetric formulation
void patch_test_vem() {
    // Material properties
    double E = 1.0;
    double nu = 0.3;

    // Domain definition
    double r_inner = 1.0, r_outer = 2.0;
    double z_min = 0.0, z_max = 1.0;

    // Mesh parameters
    int n_r = 3, n_z = 3;  // Number of elements in r and z directions

    // The function automatically:
    // 1. Generates a structured quadrilateral mesh
    // 2. Assembles global stiffness matrix
    // 3. Runs four patch tests (constant strains)
    // 4. Verifies strain recovery accuracy
}
```

### API Reference

#### Core Functions

**`compute_stiffness_matrix`**

```cpp
static Eigen::MatrixXd compute_stiffness_matrix(
    const Eigen::MatrixXd& element_vertices,
    double E,
    double nu,
    const std::string& stab_type = "standard"
);
```

Computes the complete VEM stiffness matrix including consistency and stabilization terms.

**`compute_projection_matrix`**

```cpp
static Eigen::MatrixXd compute_projection_matrix(
    const Eigen::MatrixXd& element_vertices,
    const Eigen::Matrix4d& C,
    const std::vector<Eigen::Vector4d>& base_strains
);
```

Computes the strain projection matrix that maps nodal displacements to projected strains.

**`compute_element_load_body_force`**

```cpp
static Eigen::VectorXd compute_element_load_body_force(
    const Eigen::MatrixXd& element_vertices,
    const std::function<std::pair<double, double>(double, double)>& body_force_func
);
```

Computes equivalent nodal forces from distributed body forces.

**`compute_element_load_boundary_traction`**

```cpp
static Eigen::VectorXd compute_element_load_boundary_traction(
    const Eigen::MatrixXd& element_vertices,
    const Eigen::VectorXi& edge_indices,
    const std::function<std::pair<double, double>(double, double)>& traction_func
);
```

Computes equivalent nodal forces from boundary tractions.

### Stabilization Methods

1. **Standard Stabilization**: Basic VEM stabilization ensuring stability and convergence
2. **Divergence Stabilization**: Enhanced stabilization for nearly incompressible materials
3. **Boundary Stabilization**: Edge-based stabilization for improved accuracy

### Validation

The implementation has been validated through:

- **Patch Tests**: Verification of exact strain reproduction for polynomial displacement fields
- **Convergence Studies**: Confirmation of optimal convergence rates
- **Comparison with Analytical Solutions**: Validation against known axisymmetric problems

### Performance Considerations

- **Arbitrary Polygons**: Supports elements with any number of vertices (≥3)
- **Efficient Integration**: Adaptive quadrature for vertical and non-vertical edges
- **Memory Optimization**: Sparse matrix assembly for large problems
- **Numerical Stability**: Robust projection operators and stabilization terms
