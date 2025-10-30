# Computing the VEM Projection and Stiffness Matrix for Axisymmetric Problems

Let me help you understand how to compute each term in your full projection equation and derive the stiffness matrix from it. This equation is fundamental to the VEM formulation for axisymmetric problems.

## The Full Projection Equation

The projection equation you provided is:

$$(Π\mathbf{v}_h)^T \mathbf{C}\varepsilon^p \int_E r \, dr \, dz = \int_{\partial E} \mathbf{v}_h \cdot (\mathbf{C}\varepsilon^p\mathbf{n}) r \, ds - (\sigma_r^p - \sigma_\theta^p) \int_E v_e \, dr \, dz.$$

This equation defines how to compute the projection $Π\mathbf{v}_h$ of the virtual displacement field onto polynomial spaces.

## Breaking Down Each Term

### 1. Left-Hand Side Term

$$(Π\mathbf{v}_h)^T \mathbf{C}\varepsilon^p \int_E r \, dr \, dz$$

This represents the virtual work done by the projected strain against a test strain $\varepsilon^p$, weighted by the radial coordinate $r$.

- $Π\mathbf{v}_h$ is the projection we want to compute
- $\mathbf{C}$ is the constitutive matrix for axisymmetric elasticity
- $\varepsilon^p$ is a basis strain vector (e.g., $[1,0,0,0]^T$, $[0,1,0,0]^T$, etc.)
- $\int_E r \, dr \, dz$ is the weighted volume of the element

### 2. Boundary Integral Term

$$\int_{\partial E} \mathbf{v}_h \cdot (\mathbf{C}\varepsilon^p\mathbf{n}) r \, ds$$

This boundary integral represents work done by stresses corresponding to $\varepsilon^p$ on the element boundary.

- $\mathbf{v}_h$ is the virtual displacement field on the boundary
- $\mathbf{C}\varepsilon^p$ is the stress corresponding to strain $\varepsilon^p$
- $\mathbf{n}$ is the outward normal vector
- $r\,ds$ is the weighted differential boundary segment

### 3. Volumetric Correction Term

$$(\sigma_r^p - \sigma_\theta^p) \int_E v_e \, dr \, dz$$

This term arises specifically in axisymmetric problems due to the difference between radial and hoop stresses:

- $\sigma_r^p$ is the radial stress component from $\mathbf{C}\varepsilon^p$
- $\sigma_\theta^p$ is the hoop stress component from $\mathbf{C}\varepsilon^p$
- $v_e$ is the radial component of virtual displacement $\mathbf{v}_h$

## Computation Procedure

To solve this equation for the projection $Π\mathbf{v}_h$:

### Step 1: Define Basis Strains

Create a set of basis strains $\varepsilon^p$ that span the target polynomial space:

- $\varepsilon^p_1 = [1,0,0,0]^T$ (constant radial strain)
- $\varepsilon^p_2 = [0,1,0,0]^T$ (constant axial strain)
- $\varepsilon^p_3 = [0,0,1,0]^T$ (constant hoop strain)
- $\varepsilon^p_4 = [0,0,0,1]^T$ (constant shear strain)

### Step 2: Compute Right-Hand Side for Each Basis

For each basis strain $\varepsilon^p_i$:

1. **Compute boundary integral**:

   - Calculate stress vector $\mathbf{C}\varepsilon^p_i$
   - Determine the traction vector $\mathbf{t}_i = \mathbf{C}\varepsilon^p_i\mathbf{n}$ at each point on the boundary
   - Evaluate $\mathbf{v}_h$ on the boundary (using shape functions)
   - Integrate $\mathbf{v}_h \cdot \mathbf{t}_i \, r$ along the boundary using 2-point quadrature for non-vertical edges

2. **Compute volumetric correction term**:

   - Calculate $\sigma_r^p - \sigma_\theta^p$ for the basis strain
   - Integrate $v_e \, dr \, dz$ over the element
   - Multiply these values

3. **Calculate the total right-hand side**:
   Boundary integral minus volumetric correction

### Step 3: Set Up and Solve System of Equations

For degree $k=1$ (constant strain projection):

- We have 4 unknown components of $Π\mathbf{v}_h = [\varepsilon_r, \varepsilon_z, \varepsilon_\theta, \gamma_{rz}]^T$
- We have 4 equations from the 4 basis strains
- Solve the system to determine $Π\mathbf{v}_h$

### Step 4: Express Projection as Matrix Operation

The projection operation can be expressed as a matrix $\mathbf{B}$ such that:
$$Π\mathbf{v}_h = \mathbf{B}\mathbf{d}$$

where $\mathbf{d}$ is the vector of nodal displacements for the element.

## Deriving the Stiffness Matrix

Once we have the projection operator $\mathbf{B}$, the VEM stiffness matrix has two components:

### 1. Consistency Term

$$\mathbf{K}_c = \int_E (\mathbf{B}\mathbf{d})^T \mathbf{C} (\mathbf{B}\mathbf{d}) \, r \, dr \, dz = \mathbf{d}^T \mathbf{B}^T \mathbf{C} \mathbf{B} \mathbf{d} \int_E r \, dr \, dz$$

Since $\mathbf{B}$ is constant for k=1, this simplifies to:
$$\mathbf{K}_c = \mathbf{B}^T \mathbf{C} \mathbf{B} \cdot r_c \cdot A$$

where:

- $r_c$ is the centroid's radial coordinate
- $A$ is the element area

### 2. Stabilization Term

$$\mathbf{K}_s = \alpha (\mathbf{I} - \mathbf{P})$$

where:

- $\mathbf{P} = \mathbf{B}^T(\mathbf{B}\mathbf{B}^T)^{-1}\mathbf{B}$ is the projection matrix
- $\alpha$ is a scaling factor, typically $\text{trace}(\mathbf{K}_c) / n$ where $n$ is the number of DOFs

### 3. Complete Stiffness Matrix

$$\mathbf{K} = \mathbf{K}_c + \mathbf{K}_s$$

## Key Implementation Considerations

1. **Boundary Integrals**: Use appropriate quadrature to handle r-weighting, especially for non-vertical edges

2. **Volumetric Correction**: This term is unique to axisymmetric problems and accounts for the difference between radial and hoop stresses

3. **Projection Matrix**: For k=1, the B matrix maps nodal displacements to constant strains across the element

4. **Integration Weights**: All integrals include the radial coordinate r as a weight

This approach allows you to compute the projection and build the stiffness matrix in a way that properly accounts for the axisymmetric nature of the problem, ensuring consistency and accuracy.
