# Virtual Element Method (VEM) for Linear Parabolic Diffusion on Polygonal Meshes

**Comprehensive Mathematical Derivation and Implementation Guide**  
_(based on Beirão da Veiga & Vacca, 2015, with additional commentary and proofs)_

---

## 0 · Notation and Preliminaries

| Symbol                      | Meaning                                                                                 |
| --------------------------- | --------------------------------------------------------------------------------------- |
| \(d\)                       | spatial dimension (`2` in the reference paper)                                          |
| \(k\)                       | polynomial degree of accuracy (\(k\ge1\))                                               |
| \(\mathcal T_h\)            | polygonal mesh of characteristic size \(h:=\max\_{K\in\mathcal T_h}h_K\)                |
| \(n(K)\)                    | number of vertices of element \(K\)                                                     |
| \(\mathbb P\_\ell(\omega)\) | polynomials of total degree \(\le \ell\) on domain \(\omega\)                           |
| \(a(u,v)\)                  | continuous energy bilinear form \(\displaystyle\int\_\Omega \nabla u\!\cdot\!\nabla v\) |
| \(\Pi\_{K,k}^\nabla\)       | **elliptic** (energy) projector onto \(\mathbb P_k(K)\)                                 |
| \(\Pi\_{K,k}^0\)            | \(L^2\) projector onto \(\mathbb P_k(K)\)                                               |
| \(S_K\)                     | local stabilisation bilinear form                                                       |
| \(M_K\)                     | local mass matrix                                                                       |
| \(A_K\)                     | local stiffness matrix                                                                  |

We adopt the Einstein convention for repeated coordinate indices but write out sums over element sets.

---

## 1 · Continuous Model Problem

Given \(f\in L^2\!\bigl(0,T;H^{-1}(\Omega)\bigr)\) and \(u_0\in L^2(\Omega)\),  
find

\[
\boxed{\;u\in L^2(0,T;H_0^1(\Omega)),\;u_t\in L^2(0,T;H^{-1}(\Omega))\;}
\]

such that

\[
\langle u_t(t),v\rangle + a(u(t),v)=\langle f(t),v\rangle\quad \forall v\in H_0^1(\Omega),\;\;a.e.\;t\in(0,T),
\qquad u(0)=u_0.
\]

Energy estimates give

\[
\|u(t)\|\_0^2 + 2\int_0^t \|\nabla u(s)\|\_0^2 ds = \|u_0\|\_0^2 + 2\int_0^t \langle f(s),u(s)\rangle ds ,
\]

which ensures uniqueness (by Grönwall) and the continuous dependence needed for error analysis.

---

## 2 · Mesh Assumptions

We require the usual **geometric regularity** for polygonal VEM meshes:

- **(A1) Shape‑regularity**: each \(K\) is star‑shaped w.r.t. a ball of radius \(\ge \rho_0 h_K\)  
  and \(h_e\ge\gamma_0 h_K\) for every edge \(e\subset\partial K\).

- **(A2) Bounded number of edges**: \(\exists N_0\) s.t. \(n(K)\le N_0\;\forall K\).

These conditions supply the **scaling estimates**

\[
|p|_{m,K}\simeq h_K^{-(m-\ell)} |p|_{\ell,K},\quad 0\le\ell\le m\le k+1,\;\;p\in\mathbb P\_{k+1}(K),
\]

used in projection analysis and stabilisation design.

---

## 3 · Local Augmented Space \(\widetilde V_k(K)\)

Repeating (for completeness):

\[
\widetilde V_k(K) := \{\,v\in H^1(K)\cap C^0(K):\;v|\_e\in\mathbb P_k(e),\;\Delta v\in\mathbb P_k(K)\,\}.
\]

> **Remark 3.1 (Dimension).**  
> \(|\mathbb P*k(e)|=k+1\) on an edge, hence the boundary constraints contribute  
> \(n(K)(k+1)\) scalar conditions.  
> Internal moments add \(|\mathbb P*{k-2}(K)|\) unknowns.  
> The combined set reproduces \(|\mathbb P_k(K)|\) exactly, ensuring polynomial completeness.

---

## 4 · Degrees of Freedom (DoFs)

| DoF                     | Cardinality on element \(K\) | Extraction formula (computable)   |
| ----------------------- | ---------------------------- | --------------------------------- | --- | ------------------------ | --- | -------------- |
| **D1** Vertex value     | \(n(K)\)                     | evaluation                        |
| **D2** Edge nodes       | \((k-1)\,n(K)\)              | evaluation at Gauß‑Lobatto points |
| **D3** Internal moments | \(                           | \mathbb P\_{k-2}(K)               | \)  | \(\displaystyle\frac{1}{ | K   | }\int_K v\,p\) |

Total local DoFs

\[
N_k^{\text{loc}} := n(K)\bigl[1+(k-1)\bigr] + \binom{k-1+d}{d}.
\]

---

## 5 · Computation of the Projectors

### 5.1 Elliptic Projector \(\Pi\_{K,k}^\nabla\)

Define for \(v\in\widetilde V_k(K)\):

\[
\begin{cases}
\displaystyle a*K\bigl(\Pi*{K,k}^\nabla v, q*k\bigr) = a_K(v,q_k) & \forall q_k\in\mathbb P_k(K),\\[6pt]
\displaystyle P_0\!\bigl(\Pi*{K,k}^\nabla v\bigr) = P_0 v. &
\end{cases}
\]

Because \(a*K(\cdot,\cdot)\) is SPD on \(\mathbb P_k(K)\), the system is well‑posed.  
Let \(\{\phi_i\}*{i=1}^{|\mathbb P_k(K)|}\) be a basis of \(\mathbb P_k(K)\); the coefficients \((\Pi^\nabla v)\_i\) solve

\[
\underbrace{\bigl[a_K(\phi_i,\phi_j)\bigr]_{ij}}_{\mathbf G*K}\; \mathbf c \;=\;
\underbrace{\bigl[a_K(v,\phi_j)\bigr]*{j}}\_{\mathbf b(v)},
\]

augmented by one row enforcing the average \(P_0\).  
All entries of \(\mathbf b(v)\) are computable from DoFs because  
\(a_K(v,\phi_j)\) involves \(\nabla v\) but via Green’s identity reduces to edge integrals and \(\Delta v\) (both polynomial).

### 5.2 \(L^2\) Projector \(\Pi\_{K,k}^0\)

Solve

\[
(\Pi\_{K,k}^0 v, q_k)\_K = (v,q_k)\_K,\quad\forall q_k\in\mathbb P_k(K).
\]

Again computable using D3 moments and boundary data.

---

## 6 · Local Virtual Space \(W_k(K)\)

By enforcing orthogonality of the “non‑polynomial” part to the quotient space  
\(\mathbb P*k/\mathbb P*{k-2}\), we annihilate the **kernel** of \(\Pi\_{K,k}^\nabla\) inside \(W_k(K)\).  
Consequences:

- \(\Pi\_{K,k}^\nabla\) acts as the identity on \(W_k(K)\cap\mathbb P_k(K)\).
- The stabilisation can act **only** on the kernel without harming consistency.

---

## 7 · Construction of Local Matrices

### 7.1 Split Formulation

Write any \(v_h\in W_k(K)\) as

\[
v*h = \Pi*{K,k}^\nabla v*h + (I-\Pi*{K,k}^\nabla)v*h =: v*\pi + \hat v.
\]

- **Consistency part**: computations with \(v\_\pi\) use explicit polynomial basis.
- **Stabilisation part**: \(\hat v\) has zero energy projection; we penalise it through \(S_K\).

### 7.2 Algebraic Representation

With polynomial basis \(\{\phi*i\}\) and stabilisation basis \(\{\hat\psi*\alpha\}\) (images of canonical DoFs under \(I-\Pi^\nabla\)):

\[
A*K =
\underbrace{\mathbf B_K^\top \mathbf G_K \mathbf B_K}*{\text{exact }} \;+\;
\underbrace{\alpha*K \mathbf S_K}*{\text{stab}},
\]

where

- \(\mathbf B_K\) maps DoFs to polynomial coefficients;
- \(\mathbf G_K\) is the Gram matrix \(\bigl[a_K(\phi_i,\phi_j)\bigr]\);
- \(\mathbf S_K\) is usually taken as a **scaled identity** on the kernel, i.e.  
  \(\mathbf S_K = \text{diag}(1,\dots,1)\) in the stabilisation basis.

A common scalar choice is

\[
\alpha*K = \rho*\alpha\,\mathrm{tr}(\mathbf G*K)/(k(k+1)),
\quad \rho*\alpha\in[0.1,10],
\]

ensuring estimates (9).

### 7.3 Mass Matrix

Analogously

\[
M_K = \mathbf D_K^\top \mathbf P_K \mathbf D_K + \beta_K \mathbf S_K^{(m)},
\]

with \(\mathbf P_K=[(\phi_i,\phi_j)_K]\) and a possibly different stabilisation \(\mathbf S_K^{(m)}\).  
Setting \(\beta_K=0\) yields the **reduced mass**.

---

## 8 · Assembly Algorithm (Pseudo‑code)

```text
for K in mesh:
    compute G_K, P_K           # polynomial Gram matrices
    compute B_K, D_K           # projection / moment maps
    compute S_K                # stabilisation identity on kernel
    A_K = B_K^T G_K B_K + alpha_K S_K
    M_K = D_K^T P_K D_K + beta_K S_K
    scatter A_K, M_K into global CSR matrices
```

Complexity: \(O(N*\text{dof})\) memory, \(O(N*\text{dof})\) flops per explicit assembly pass.

---

## 9 · Semidiscrete Problem

Find \(u_h:[0,T]\to\mathbb R^{N_h}\) satisfying

\[
\mathbf M\,\dot{\mathbf u}_h(t) + \mathbf A\,\mathbf u_h(t) = \mathbf f_h(t),
\qquad \mathbf u_h(0)=\mathbf u_{h,0}.
\]

- \(\mathbf M\) and \(\mathbf A\) are SPD.
- Modal decomposition uses the eigensystem  
  \(\mathbf A\,\mathbf w^{(n)}=\lambda_h^{(n)}\mathbf M\,\mathbf w^{(n)}\).

---

## 10 · Fully Discrete Schemes

### 10.1 \(\theta\)-Method (Unified View)

\[
\mathbf M\frac{\mathbf U^{n+1}-\mathbf U^{n}}{\tau} +
\bigl[(1-\theta)\mathbf A\,\mathbf U^{n} + \theta \mathbf A\,\mathbf U^{n+1}\bigr] =
(1-\theta)\mathbf f_h^{n} + \theta \mathbf f_h^{n+1},
\]

- \(\theta=0\) : explicit (conditionally stable),
- \(\theta=\tfrac12\): Crank–Nicolson (second order, A‑stable),
- \(\theta=1\) : backward Euler (first order, unconditionally stable).

### 10.2 Error Decomposition

Let \(e^{n}=u(t_n)-U^{n}\). Split

\[
e^n = \underbrace{(u(t*n)-u_h(t_n))}*{\text{spatial }O(h^{k+1})} +
\underbrace{(u*h(t_n)-U^n)}*{\text{temporal }O(\tau^p)}.
\]

A discrete Grönwall argument yields (for backward Euler)

\[
\|e^n\|_0 \,\lesssim\, h^{k+1} + \tau \sup_{0<s<t*n}\|u*{tt}(s)\|\_0.
\]

---

## 11 · Proof Sketch of Optimal Spatial Error (Thm 3.1)

1. **Strang splitting**  
   \(\|u-u_h\| \le \|u-P_h u\| + \|P_h u-u_h\|\).

2. **Approximation error**  
   Use (12).

3. **Consistency error**  
   Compare continuous and discrete weak forms with \(v_h=P_h u-u_h\).

4. **Stability**  
   Coercivity of \(m_h\) and \(a_h\) ⇒ differential inequality

   \[
   \tfrac12\frac{d}{dt}\|P*h u-u_h\|*{m_h}^2 +
   a_h(P_h u-u_h,P_h u-u_h) \le C h^{2k+2}\dots
   \]

5. **Gronwall**  
   Integrate in time and add approximation term.

---

## 12 · Conservation Properties

If \(v_h\equiv1\) belongs to \(W_h\) (true because constants lie in \(\mathbb P_k\subset W_k(K)\)),  
then testing (10) with \(1\) reproduces the **exact discrete balance** &rightarrow; no artificial mass loss/gain.

---

## 13 · Practical Tips

- **Numerical quadrature** on polygon edges: Gauß–Lobatto with \(k+1\) points suffices for exactness on \(\mathbb P\_{2k-1}(e)\).
- **Conditioning**: stabilisation scaling \(\alpha_K\sim h_K^{-2}\), \(\beta_K\sim 1\) keeps condition number nearly mesh‑independent.
- **Parallel assembly**: element matrices depend solely on local DoFs → embarrassingly parallel.
- **Adaptive time step**: solve local modal estimate \(\tau < 2/\lambda\_{\max}\) if explicit schemes are used.

---

## 14 · Extensions

| Topic                                  | Brief Description                                                           |
| -------------------------------------- | --------------------------------------------------------------------------- |
| Non‑homogeneous Dirichlet BC           | modify space \(W_h\) with lifting functions, preserve conformity            |
| Anisotropic diffusion \(\mathbf K(x)\) | replace \(a(u,v)=\int \nabla u^T\mathbf K\nabla v\) – projectors unchanged  |
| Non‑linear reaction terms              | treat by semi‑implicit linearisation, VEM only touches Laplacian part       |
| Space–time VEM                         | discretise time with polynomials → fully polygonal spatio‑temporal elements |

---

## 15 · Summary of Constants

\[
\begin{array}{lcl}
C*\text{app} &\;=& \text{approximation constant in (12)} \\
\alpha*{_},\alpha^{_} &:& \text{stability bounds for stiffness form} \\
\beta*{*},\beta^{_} &:& \text{stability bounds for mass form} \\
\rho_\alpha &:& \text{user factor for stiffness stabilisation} \\
\rho\_\beta &:& \text{user factor for mass stabilisation} \\
\end{array}
\]

Recommended defaults: \(\rho*\alpha=\rho*\beta=1\).

---

> ### Reference
>
> **L. Beirão da Veiga & G. Vacca (2015).**  
> _Virtual element methods for parabolic problems on polygonal meshes._ ESAIM: M2AN 49 (2), 577‑599.
