# Virtual Element Method (VEM) for Axisymmetric Analysis

## Introduction

This document explains the implementation of Virtual Element Method (VEM) for axisymmetric elasticity problems, specifically focusing on a quadrilateral element with polynomial degree k=1. The explanation covers the mathematical foundations and practical implementation details of both the consistency and stabilization terms.

## 1. Bilinear Form for the Local Stiffness Matrix

The bilinear form for the local stiffness matrix in axisymmetric elasticity is:

$$a^E_h(u_h, v_h) = \int_E \varepsilon(u_h)^T C \varepsilon(v_h) \, r \, dr \, dz$$

In VEM, this is split into two components:

- Consistency term
- Stabilization term

## 2. Consistency Term

### Mathematical Definition

The consistency term is defined as:

$$\int_E (\Pi u_h)^T C (\Pi v_h) \, r \, dr \, dz = (\Pi u_h)^T C (\Pi v_h) \int_E r \, dr \, dz$$

For k=1 (constant strain projection), the projection operators Π can be taken outside the integral, reducing it to:

$$K_c = B^T C B \cdot \int_E r \, dr \, dz$$

Where:

- B represents the projection operator Π
- C is the constitutive matrix
- The integral represents the weighted volume

### Implementation Details

We approximate the integral as:

$$\int_E r \, dr \, dz \approx r_c \cdot \text{area}$$

Where:

- $r_c$ is the radial coordinate of the element's centroid
- area is the element area

The code implementation:

```python
# Compute the consistency term K_c
K_c = np.dot(np.dot(B.T, C), B) * r_c * area
```

## 3. The B Matrix

### Role in VEM

The B matrix is the discrete representation of the projection operator Π. It maps nodal displacements to constant strains across the element.

### Structure

For a quadrilateral with 4 nodes in axisymmetric conditions, B is a 4×8 matrix:

- 4 rows for the strain components [εr, εz, εθ, γrz]
- 8 columns for the displacement DOFs [ur1, uz1, ur2, uz2, ur3, uz3, ur4, uz4]

### Why Derivatives?

The elements of B contain derivatives because they represent the strain-displacement relationships from elasticity theory:

- εr = ∂u_r/∂r
- εz = ∂u_z/∂z
- εθ = u_r/r
- γrz = ∂u_r/∂z + ∂u_z/∂r

Each entry Bij relates the jth displacement DOF to the ith strain component.

### Construction

```python
# For each node i, construct the B matrix entries
for i in range(4):
    idx_r = 2*i     # Column index for u_r of node i
    idx_z = 2*i + 1 # Column index for u_z of node i

    # εr = ∂u_r/∂r
    B[0, idx_r] = dN_dr[i]

    # εz = ∂u_z/∂z
    B[1, idx_z] = dN_dz[i]

    # εθ = u_r/r
    B[2, idx_r] = 0.25 / r_c

    # γrz = ∂u_r/∂z + ∂u_z/∂r
    B[3, idx_r] = dN_dz[i]
    B[3, idx_z] = dN_dr[i]
```

### Approximation for the Projection

Although VEM doesn't use explicit shape functions inside the element, we approximate the projection of the displacement field using shape functions Ni:

$u_r \approx \sum_{i=1}^n N_i \cdot u_{ri}$
$u_z \approx \sum_{i=1}^n N_i \cdot u_{zi}$

Where uri and uzi are the nodal displacements at node i.

Then, to get the projected strains, we apply the differential operations:

$\varepsilon_r = \frac{\partial u_r}{\partial r} \approx \sum_{i=1}^n \frac{\partial N_i}{\partial r} \cdot u_{ri}$

$\varepsilon_z = \frac{\partial u_z}{\partial z} \approx \sum_{i=1}^n \frac{\partial N_i}{\partial z} \cdot u_{zi}$
$\varepsilon_\theta = \frac{u_r}{r} \approx \sum_{i=1}^n \frac{N_i}{r} \cdot u_{ri}$

$\gamma_{rz} = \frac{\partial u_r}{\partial z} + \frac{\partial u_z}{\partial r} \approx \sum_{i=1}^n \frac{\partial N_i}{\partial z} \cdot u_{ri} + \sum_{i=1}^n \frac{\partial N_i}{\partial r} \cdot u_{zi}$

This can be written in matrix form as:

$\varepsilon = B \cdot d$

Where:

- ε is the strain vector [εr, εz, εθ, γrz]
- d is the nodal displacement vector [ur1, uz1, ur2, uz2, ur3, uz3, ur4, uz4]
- B is the strain-displacement matrix that encodes the differential relationships

To compute the derivatives needed for the B matrix, we use:

```python
def compute_derivatives():
    # Shape function derivatives in natural coordinates
    dN_dxi = np.array([-0.25, 0.25, 0.25, -0.25])
    dN_deta = np.array([-0.25, -0.25, 0.25, 0.25])

    # Compute Jacobian and transform to physical coordinates
    # ...

    return dN_dr, dN_dz
```

This allows us to compute the projection operator (B matrix) that maps displacements to constant strains. The key insight is that while we don't use shape functions to approximate the solution inside the element (as we would in FEM), we do use them to define the projection operator.

## 4. Stabilization Term

### Mathematical Definition

The stabilization term is defined as:

$$S^E = \alpha \sum_{i=1}^{m} (u_h(v_i) - \Pi u_h(v_i)) \cdot (v_h(v_i) - \Pi v_h(v_i))$$

Where:

- α ≈ trace(C) · |E| is a scaling factor
- The sum measures the difference between the virtual functions and their projections

### Implementation Details

For computational efficiency, this can be expressed in matrix form:

```python
# Compute projection matrix Π = D(D^T D)^-1 D^T
D = B  # For k=1, D equals B
DTD_inv = np.linalg.pinv(np.dot(D.T, D))
Pi = np.dot(np.dot(D, DTD_inv), D.T)

# Compute (I - Π) for the stabilization term
I = np.eye(8)
I_minus_Pi = I - Pi

# Compute stabilization parameter α ≈ trace(C)·|E|
alpha = np.trace(C) * area

# The stabilization term
K_s = alpha * I_minus_Pi
```

### Why D = B?

For VEM with polynomial degree k=1:

- Both B and D represent the projection operator Π
- Both map from the virtual element space to the space of constant strains
- This equivalence is specific to k=1 and would be different for higher-order formulations

## 5. Final Stiffness Matrix

The complete VEM stiffness matrix combines both terms:

$$K = K_c + K_s$$

This captures both the energy of the projected polynomial part (K_c) and adds stability for the non-polynomial part (K_s).

## 6. Key Differences from FEM

In VEM:

1. We only use shape functions to define the projection operator, not to approximate the solution inside the element
2. The method can handle arbitrary polygonal elements
3. The stabilization term ensures stability for the non-polynomial parts of the solution

## 7. Implementation for a Quadrilateral Element

For a quadrilateral element with vertices:

- v1=(1,0)
- v2=(2,0)
- v3=(2,1)
- v4=(1,1)

And material properties:

- Young's modulus: E=1
- Poisson's ratio: ν=0.3

The implementation computes:

1. The constitutive matrix C
2. Geometric properties (area, centroid)
3. B matrix for the projection operator
4. Consistency term K_c
5. Stabilization term K_s
6. Final stiffness matrix K

## 8. Extension to Higher-Order and Complex Elements

The approach can be extended to:

1. Higher polynomial degrees (k > 1)
2. Arbitrary polygonal elements
3. Three-dimensional problems
4. Nonlinear material behavior

For each extension, the core concepts of projection operators, consistency, and stabilization remain the same, though the specific implementations would be adapted accordingly.
