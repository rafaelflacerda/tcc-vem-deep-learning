#include "solver/axisymmetric.hpp"

namespace solver {

    std::vector<Eigen::Vector4d> axisymmetric::define_base_strain_vectors() {
        std::vector<Eigen::Vector4d> base_strain_vectors;
        
        // Constant radial strain (εr = 1, εz = 0, εθ = 0, γrz = 0)
        Eigen::Vector4d eps_r = Eigen::Vector4d::Zero();
        eps_r(0) = 1.0;
        base_strain_vectors.push_back(eps_r);
        
        // Constant axial strain (εr = 0, εz = 1, εθ = 0, γrz = 0)
        Eigen::Vector4d eps_z = Eigen::Vector4d::Zero();
        eps_z(1) = 1.0;
        base_strain_vectors.push_back(eps_z);
        
        // Constant hoop strain (εr = 0, εz = 0, εθ = 1, γrz = 0)
        Eigen::Vector4d eps_theta = Eigen::Vector4d::Zero();
        eps_theta(2) = 1.0;
        base_strain_vectors.push_back(eps_theta);
        
        // Constant shear strain (εr = 0, εz = 0, εθ = 0, γrz = 1)
        Eigen::Vector4d eps_rz = Eigen::Vector4d::Zero();
        eps_rz(3) = 1.0;
        base_strain_vectors.push_back(eps_rz);
        
        return base_strain_vectors;
    }

    std::pair<double, double> axisymmetric::compute_traction_vector(
    const Eigen::Matrix4d& C,
    const Eigen::Vector4d& eps_p,
    const std::pair<double, double>& normal
    ) {
        // Compute stress corresponding to strain state
        Eigen::Vector4d sigma = C * eps_p;
        
        // Extract components
        double sigma_r = sigma(0);    // Radial stress
        double sigma_z = sigma(1);    // Axial stress
        double sigma_rz = sigma(3);   // Shear stress
    
    // Extract normal vector components
    double n_r = normal.first;
    double n_z = normal.second;

        // Compute traction components (stress matrix times normal vector)
        double t_r = sigma_r * n_r + sigma_rz * n_z;
        double t_z = sigma_rz * n_r + sigma_z * n_z;
        
        return {t_r, t_z};
    }

    double axisymmetric::gauss_quadrature_boundary_integral(
    const Eigen::Matrix2d& edge_vertices,
    const std::pair<double, double>& traction_vector,
    const std::function<std::pair<double, double>(double)>& displacement_func,
    bool is_vertical
    ) {
        // Extract coordinates
        double r1 = edge_vertices(0, 0);
        double z1 = edge_vertices(0, 1);
        double r2 = edge_vertices(1, 0);
        double z2 = edge_vertices(1, 1);
        
        // Edge vector and length
    double dr = r2 - r1;
    double dz = z2 - z1;
    double edge_length = std::sqrt(dr*dr + dz*dz);

        // Extract traction components
        double t_r = traction_vector.first;
        double t_z = traction_vector.second;
        
        // Define parametric functions for r(s) and z(s)
        auto r_param = [r1, dr](double s) { return r1 + s * dr; };
        auto z_param = [z1, dz](double s) { return z1 + s * dz; };
        
        // Result variable
        double integral = 0.0;
        
        // Determine number of Gauss points based on edge type
        if (is_vertical) {
            // Use 1-point quadrature for vertical edges
            double s = 0.5;
            double weight = 1.0;
            
            // Evaluate at Gauss point
            double r_s = r_param(s);
            
            // Get displacement at Gauss point
            auto [v_r, v_z] = displacement_func(s);
            
            // Compute dot product: v_h · t
            double dot_product = v_r * t_r + v_z * t_z;
            
            // Add weighted contribution
            integral += weight * dot_product * r_s;
        } else {
            // Use 2-point Gauss quadrature for non-vertical edges
            double s1 = 0.5 - std::sqrt(3.0) / 6.0;
            double s2 = 0.5 + std::sqrt(3.0) / 6.0;
            double w1 = 0.5;
            double w2 = 0.5;
            
            // First Gauss point
            double r_s1 = r_param(s1);
            auto [v_r1, v_z1] = displacement_func(s1);
            double dot_product1 = v_r1 * t_r + v_z1 * t_z;
            integral += w1 * dot_product1 * r_s1;
            
            // Second Gauss point
            double r_s2 = r_param(s2);
            auto [v_r2, v_z2] = displacement_func(s2);
            double dot_product2 = v_r2 * t_r + v_z2 * t_z;
            integral += w2 * dot_product2 * r_s2;
        }
        
        // Scale by edge length
        integral *= edge_length;
        
        return integral;
    }

    std::function<std::pair<double, double>(double)> axisymmetric::create_displacement_function(
        const Eigen::Matrix2d& edge_vertices,
        const std::pair<int, int>& vertex_indices,
        int dof_index,
        double dof_value
    ) {
        // Extract vertex indices
        int i = vertex_indices.first;
        int j = vertex_indices.second;
        
        // Extract coordinates
        double r1 = edge_vertices(0, 0);
        double z1 = edge_vertices(0, 1);
        double r2 = edge_vertices(1, 0);
        double z2 = edge_vertices(1, 1);
        
        // Edge vector
        double dr = r2 - r1;
        double dz = z2 - z1;
        
        // Which vertex and component does this DOF correspond to?
        int vertex_idx = dof_index / 2;
        int component_idx = dof_index % 2;
        
        // Create and return the function
        return [=](double s) -> std::pair<double, double> {
            // Default zero displacement
            double v_r = 0.0;
            double v_z = 0.0;
            
            // Linear shape functions along the edge
            double N_i = (vertex_idx == i) ? (1.0 - s) : (vertex_idx == j ? s : 0.0);
            
            // Set the appropriate component
            if (component_idx == 0) {
                v_r = N_i * dof_value;
            } else {
                v_z = N_i * dof_value;
            }
            
            return {v_r, v_z};
        };
    }

    Eigen::MatrixXd axisymmetric::compute_volumetric_correction(
        const Eigen::MatrixXd& element_vertices,
        const Eigen::Matrix4d& C,
        const std::vector<Eigen::Vector4d>& base_strains
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;
        int n_strains = base_strains.size();
        
        // Initialize the result matrix
        Eigen::MatrixXd volumetric_corrections = Eigen::MatrixXd::Zero(n_dofs, n_strains);
        
        // Compute element centroid
        Eigen::Vector2d centroid = element_vertices.colwise().mean();
        
        // For each base strain, compute the stress difference (σr - σθ)
        std::vector<double> stress_differences;
        for (int k = 0; k < n_strains; ++k) {
            // Compute stresses for this basis strain
            Eigen::Vector4d sigma = C * base_strains[k];
            double sigma_r = sigma(0);     // Radial stress
            double sigma_theta = sigma(2); // Hoop stress
            
            // Calculate the difference
            double stress_diff = sigma_r - sigma_theta;
            stress_differences.push_back(stress_diff);
        }
        
        // Calculate element weighted volume using proper triangulation
        double weighted_volume = compute_weighted_volume_polygon(element_vertices);
        
        // For each radial DOF, apply the volumetric correction
        for (int i = 0; i < n_vertices; ++i) {
            int dof_idx = 2 * i;  // Radial DOF index
            
            // Apply correction for each strain type
            for (int j = 0; j < n_strains; ++j) {
                // Each vertex gets weighted_volume / n_vertices
                volumetric_corrections(dof_idx, j) = stress_differences[j] * weighted_volume / n_vertices;
            }
        }
        
        return volumetric_corrections;
    }

    Eigen::MatrixXd axisymmetric::compute_element_boundary_integrals(
        const Eigen::MatrixXd& element_vertices,
        const Eigen::Matrix4d& C,
        const std::vector<Eigen::Vector4d>& base_strains
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;
        int n_strains = base_strains.size();
        
        // Initialize the result matrix
        Eigen::MatrixXd boundary_integrals = Eigen::MatrixXd::Zero(n_dofs, n_strains);
        
        // Loop over all edges of the element
        for (int i = 0; i < n_vertices; ++i) {
            int j = (i + 1) % n_vertices;
            
            // Extract edge vertices
            Eigen::Matrix2d edge_vertices;
            edge_vertices.row(0) = element_vertices.row(i);
            edge_vertices.row(1) = element_vertices.row(j);
            
            // Compute normal vector for this edge
            double r1 = edge_vertices(0, 0);
            double z1 = edge_vertices(0, 1);
            double r2 = edge_vertices(1, 0);
            double z2 = edge_vertices(1, 1);
            
            // Edge vector
            double dr = r2 - r1;
            double dz = z2 - z1;
            
            // Edge length
            double edge_length = std::sqrt(dr*dr + dz*dz);
            
            // Skip if edge is too short
            if (edge_length < 1e-10) {
                continue;
            }
            
            // Compute outward normal vector (perpendicular to edge, pointing outward)
            // For counterclockwise ordering of vertices
            double n_r = dz / edge_length;
            double n_z = -dr / edge_length;
            std::pair<double, double> normal = {n_r, n_z};
            
            // Compute traction vectors for each base strain
            std::vector<std::pair<double, double>> traction_vectors;
            for (int k = 0; k < n_strains; ++k) {
                auto traction = compute_traction_vector(C, base_strains[k], normal);
                traction_vectors.push_back(traction);
            }
            
            // Is the edge vertical?
            bool is_vertical = std::abs(dr) < 1e-10;
            
            // For each DOF that affects this edge (the endpoints)
            for (int dof_idx = 0; dof_idx < n_dofs; ++dof_idx) {
                // Skip DOFs not at the endpoints of this edge
                int vertex_idx = dof_idx / 2;
                if (vertex_idx != i && vertex_idx != j) {
                    continue;
                }
                
                // Create displacement function for this DOF
                auto disp_func = create_displacement_function(
                    edge_vertices, {i, j}, dof_idx
                );
                
                // Compute boundary integral for each base strain
                for (size_t strain_idx = 0; strain_idx < traction_vectors.size(); ++strain_idx) {
                    auto traction = traction_vectors[strain_idx];
                    double integral = gauss_quadrature_boundary_integral(
                        edge_vertices, traction, disp_func, is_vertical
                    );
                    
                    // Add contribution to the result matrix
                    boundary_integrals(dof_idx, strain_idx) += integral;
                }
            }
        }
        
        return boundary_integrals;
    }

    Eigen::MatrixXd axisymmetric::compute_proj_system_matrix(
        const Eigen::Matrix4d& C,
        const Eigen::MatrixXd& eps_matrix,
        double weighted_volume
    ) {
        Eigen::MatrixXd proj_matrix = C * eps_matrix * weighted_volume;
        return proj_matrix;
    }

    Eigen::MatrixXd axisymmetric::compute_projection_matrix(
        const Eigen::MatrixXd& element_vertices,
        const Eigen::Matrix4d& C,
        const std::vector<Eigen::Vector4d>& base_strains
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;
        int n_strains = base_strains.size();
        
        // Compute boundary integrals
        Eigen::MatrixXd boundary_integrals = compute_element_boundary_integrals(
            element_vertices, C, base_strains
        );
        
        // Compute volumetric corrections
        Eigen::MatrixXd volumetric_corrections = compute_volumetric_correction(
            element_vertices, C, base_strains
        );
        
        // Calculate right-hand side: boundary_integrals - volumetric_corrections
        Eigen::MatrixXd rhs = boundary_integrals - volumetric_corrections;
        
        // Calculate the weighted volume using proper triangulation
        double weighted_volume = compute_weighted_volume_polygon(element_vertices);
        
        // Stack base strain vectors into a matrix (each column is a base strain)
        Eigen::MatrixXd eps_matrix(4, n_strains);
        for (int i = 0; i < n_strains; i++) {
            eps_matrix.col(i) = base_strains[i];
        }
        
        // Create the coefficient matrix for the projection system
        Eigen::MatrixXd coeff_matrix = C * eps_matrix;
        
        // Initialize the projection matrix B
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_strains, n_dofs);
        
        // For each DOF
        for (int i = 0; i < n_dofs; i++) {
            // Get the RHS for this DOF (row i of rhs matrix)
            Eigen::VectorXd scaled_rhs = rhs.row(i).transpose() / weighted_volume;
            
            // Special handling for axial DOFs to decouple shear from axial strain
            if (i % 2 == 1) {  // Axial DOF
                // Zero out the shear component (index 3) for axial DOFs
                scaled_rhs(3) = 0.0;
            }
            
            // Solve the system: coeff_matrix * x = scaled_rhs
            // Using Eigen's least squares solver
            B.col(i) = coeff_matrix.colPivHouseholderQr().solve(scaled_rhs);
        }
        
        return B;
    }

    double axisymmetric::compute_weighted_volume_polygon(
        const Eigen::MatrixXd& element_vertices
    ) {
        // Number of vertices
        int n_vertices = element_vertices.rows();
        
        // Compute centroid
        Eigen::Vector2d centroid = element_vertices.colwise().mean();
        double weighted_volume = 0.0;
        
        // Create operations object to use area calculation function
        // utils::operations operations;
        
        // Subdivide into triangles
        for (int i = 0; i < n_vertices; i++) {
            int j = (i + 1) % n_vertices;
            
            // Create triangle formed by centroid, vertex i, and vertex j
            Eigen::MatrixXd triangle(3, 2);
            triangle.row(0) = centroid;
            triangle.row(1) = element_vertices.row(i);
            triangle.row(2) = element_vertices.row(j);
            
            // For a triangle, the weighted volume can be computed exactly
            // Average r-coordinate of the triangle vertices
            double r_avg = (centroid[0] + element_vertices(i, 0) + element_vertices(j, 0)) / 3.0;
            
            // Triangle area
            double area = utils::operations::calcArea(triangle);
            
            // Contribution to weighted volume
            weighted_volume += r_avg * area;
        }
        
        return weighted_volume;
    }

    Eigen::MatrixXd axisymmetric::compute_stiffness_matrix(
        const Eigen::MatrixXd& element_vertices,
        double E,
        double nu,
        const std::string& stab_type
    ) {
        // Build constitutive matrix
        Eigen::Matrix4d C = material::mat::buildAxisymmetricElasticity(E, nu);

        // Get base strain vectors
        std::vector<Eigen::Vector4d> base_strains = define_base_strain_vectors();
        
        // Compute projection matrix using the fixed method
        Eigen::MatrixXd B = compute_projection_matrix(element_vertices, C, base_strains);
        
        // Calculate the weighted volume ∫_E r dr dz
        double r_avg = element_vertices.col(0).mean();
        double area = utils::operations::calcArea(element_vertices);
        double weighted_volume = r_avg * area;
        
        // Compute consistency term K_c = B^T · C · B · weighted_volume
        Eigen::MatrixXd K_c = B.transpose() * C * B * weighted_volume;

        // Compute standard stabilization term K_s
        int n_dofs = 2 * element_vertices.rows();
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_dofs, n_dofs);

        // Compute the projection matrix onto the space of consistent displacements
        Eigen::MatrixXd BB_T = B * B.transpose();
        Eigen::MatrixXd BB_T_inv = BB_T.inverse();
        Eigen::MatrixXd P = B.transpose() * BB_T_inv * B;

        // Compute I - P
        Eigen::MatrixXd I_minus_P = I - P;

        // Scale the standard stabilization term by the weighted volume
        double alpha = K_c.trace() / n_dofs;
        Eigen::MatrixXd K_s_standard = alpha * I_minus_P;

        // Compute the final stiffness matrix
        if (stab_type == "divergence") {
            // Compute the divergence stabilization matrix
            Eigen::MatrixXd K_div = compute_divergence_stabilization_matrix(element_vertices, E, nu);

            // Combined stabilization term
            Eigen::MatrixXd K_s = K_s_standard + K_div;

            // Complete stiffness matrix
            Eigen::MatrixXd K = K_c + K_s;

            return K;
        } else if (stab_type == "boundary"){
            // Compute the boundary stabilization term 
            Eigen::MatrixXd K_bound = compute_boundary_stabilization_matrix(element_vertices, P, E, nu);

            // Combined stabilization term
            Eigen::MatrixXd K_s = K_s_standard + K_bound;

            // Complete stiffness matrix
            Eigen::MatrixXd K = K_c + K_s;

            return K;
        } else {
            // Use only standard stabilization
            Eigen::MatrixXd K = K_c + K_s_standard;
            return K;
        }

    }

    double axisymmetric::compute_divergence_boundary_integral(
        const Eigen::MatrixXd& element_vertices,
        const Eigen::VectorXd& displacements
    ) {

        // Number of vertices
        int n_vertices = element_vertices.rows();

        // Initialize boundary integral
        double boundary_integral = 0.0;

        // Initialize edge contribution
        double edge_contribution = 0.0;

        // Loop over each edge of the element
        for (int i = 0; i < n_vertices; i++) {
            int j = (i + 1) % n_vertices;

            // Extract edge vertices
            double r_i = element_vertices(i, 0);
            double z_i = element_vertices(i, 1);
            double r_j = element_vertices(j, 0);
            double z_j = element_vertices(j, 1);

            // Extract displacements at endpoints
            double u_r_i = displacements(2*i);
            double u_z_i = displacements(2*i+1);
            double u_r_j = displacements(2*j);
            double u_z_j = displacements(2*j+1);

            // Compute edge vector and length
            double dr = r_j - r_i;
            double dz = z_j - z_i;
            double edge_length = std::sqrt(dr*dr + dz*dz);

            // Skip if edge is too short
            if (edge_length < 1e-10) {
                continue;
            }

            // Compute outward normal vector to the edge
            double n_r = dz / edge_length;
            double n_z = -dr / edge_length;

            // Check if edge is vertical (constant r)
            bool is_vertical = std::abs(dr) < VERTICAL_TOLERANCE;

            if (is_vertical){
                // For vertical edges, use single-point quadrature
                double s = 0.5; // Midpoint

                // Interpolate displacement at midpoint
                double u_r_s = (1-s) * u_r_i + s * u_r_j;
                double u_z_s = (1-s) * u_z_i + s * u_z_j;

                // Compute r at midpoint
                double r_s = r_i; // Constant for vertical edge

                // Compute dot product u·n
                double u_dot_n = u_r_s * n_r + u_z_s * n_z;

                // Compute integrand and contribution
                double integrand = 2 * M_PI * r_s * u_dot_n;
                double edge_contribution = edge_length * integrand;
            } else {
                // For non-vertical edges, use 2-point Gaussian quadrature
                double s1 = 0.5 - std::sqrt(3) / 6;
                double s2 = 0.5 + std::sqrt(3) / 6;
                double w1 = 0.5, w2 = 0.5;

                // First quadrature point   
                double r_s1 = (1-s1) * r_i + s1 * r_j;
                double u_r_s1 = (1-s1) * u_r_i + s1 * u_r_j;
                double u_z_s1 = (1-s1) * u_z_i + s1 * u_z_j;
                double u_dot_n1 = u_r_s1 * n_r + u_z_s1 * n_z;
                double integrand1 = 2 * M_PI * r_s1 * u_dot_n1;

                // Second quadrature point
                double r_s2 = (1-s2) * r_i + s2 * r_j;
                double u_r_s2 = (1-s2) * u_r_i + s2 * u_r_j;
                double u_z_s2 = (1-s2) * u_z_i + s2 * u_z_j;
                double u_dot_n2 = u_r_s2 * n_r + u_z_s2 * n_z;
                double integrand2 = 2 * M_PI * r_s2 * u_dot_n2;

                // Combine with weights
                double edge_contribution = edge_length * (w1 * integrand1 + w2 * integrand2);
            }

            // Add contribution to boundary integral
            boundary_integral += edge_contribution;
        }
        
        return boundary_integral;
    }

    // === LOADS ===
    Eigen::VectorXd axisymmetric::compute_equivalent_body_force(
        const Eigen::MatrixXd& element_vertices,
        const Eigen::Matrix4d& C,
        const Eigen::Vector4d& strain_state
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;

        // Initialize equivalent body force vector
        Eigen::VectorXd f_eq = Eigen::VectorXd::Zero(n_dofs);

        // Compute the stress tensor for the given strain state
        Eigen::Vector4d sigma = C * strain_state;

        // In axisymmetric elasticity, a constant radial strain (εr=1.0) 
        // generates a body force due to hoop stress
        // This body force is proportional to sigma_r - sigma_theta
        double radial_body_force = sigma(0) - sigma(2);
        
        // Compute the volume of the element
        double area = utils::operations::calcArea(element_vertices);
        double r_avg = element_vertices.col(0).mean();
        double weighted_volume = r_avg * area;

        // Distribute the body force to the radial DOFs
        for (int i = 0; i < n_vertices; i++){
            int dof_index = 2 * i;
            f_eq[dof_index] = radial_body_force * area / weighted_volume;
        }

        return f_eq;
    }

    Eigen::VectorXd axisymmetric::compute_element_load_body_force(
        const Eigen::MatrixXd& element_vertices,
        const std::function<std::pair<double, double>(double, double)>& body_force_func
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;

        // Compute element area
        double area = utils::operations::calcArea(element_vertices);
        
        // Initialize load vector
        Eigen::VectorXd f_body = Eigen::VectorXd::Zero(n_dofs);

        // For each shape function, compute the geometric moment
        for (int i = 0; i < n_vertices; i++){
            // Get the radial coordinate of current vertex
            double r_i = element_vertices(i, 0);

            // Compute r-weighted moment for this shape function using the formula:
            // ∫_E N_i r dr dz ≈ |E|/12 * (2r_i + r_j + r_k) for triangles
            // For a general polygon, we approximate using average r 
            // This is exact for triangles but approximate for polygons
            // TODO:  implement triangulation for general polygons
            double r_sum = 2 * r_i;
            for (int j = 0; j <  n_vertices; j++){
                if (j != i) r_sum += element_vertices(j, 0)/ (n_vertices - 1);
            }
            double moment = area / (n_vertices - 1) * r_sum;
            
            // Compute the body force at element centroid
            Eigen::Vector2d centroid = utils::operations::calcCentroid(element_vertices);
            std::pair<double, double> body_force = body_force_func(centroid[0], centroid[1]);

            // Compute the contribution to the load vector from this shape function
            f_body[2*i] = moment * body_force.first;
            f_body[2*i+1] = moment * body_force.second;
        }

        return f_body;
    }

    Eigen::VectorXd axisymmetric::compute_element_load_boundary_traction(
        const Eigen::MatrixXd& element_vertices,
        const Eigen::VectorXi& edge_indices,
        const std::function<std::pair<double, double>(double, double)>& traction_func
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;

        // Initialize load vector
        Eigen::VectorXd f_traction = Eigen::VectorXd::Zero(n_dofs);

        // Process each edge where traction is applied
        for(int idx = 0; idx < edge_indices.size(); idx++){
            int edge_start = edge_indices[idx];
            int edge_end = (edge_start + 1) % n_vertices;

            // Extract edge vertices
            double r1 = element_vertices(edge_start, 0);
            double z1 = element_vertices(edge_start, 1);
            double r2 = element_vertices(edge_end, 0);
            double z2 = element_vertices(edge_end, 1);

            // Edge vector
            double dr = r2 - r1;
            double dz = z2 - z1;

            // Edge length
            double edge_length = std::sqrt(dr*dr + dz*dz);

            // Skip if edge is too short
            if (edge_length < VERTICAL_TOLERANCE){
                continue;
            }

            // Check if edge is vertical
            bool is_vertical = std::abs(dr) < VERTICAL_TOLERANCE;

            // Parameterize the edge: r(s) = r1 + s * dr, z(s) = z1 + s * dz
            auto r_param = [r1, dr](double s) { return r1 + s * dr; };
            auto z_param = [z1, dz](double s) { return z1 + s * dz; };
            
            // For each DOF that affects this edge (the endpoints)
            std::vector<int> vertex_indices = {edge_start, edge_end};
            for (int vertex_idx : vertex_indices){
                // 2 DOFs per vertex (radial and axial)
                for (int component_idx = 0; component_idx < 2; component_idx++){
                    int dof_idx = 2 * vertex_idx + component_idx;
                    bool is_radial = (component_idx == 0); // First component is radial

                    // Create shape functions for this DOF along the edge
                    std::function<double(double)> shape_func;
                    if (vertex_idx == edge_start){
                        // Shape function decreases from 1 to 0 along the edge
                        shape_func = [](double s) { return 1.0 - s; };
                    } else {
                        // Shape function increases from 0 to 1 along the edge
                        shape_func = [](double s) { return s; };
                    }

                    // Integration points and weights for Gauss quadrature
                    double s1 = 0.5 - std::sqrt(3.0) / 6.0;
                    double s2 = 0.5 + std::sqrt(3.0) / 6.0;
                    double w1 = 0.5, w2 = 0.5;

                    // Compute the contribution to the load vector from this DOF
                    std::vector<double> s_points, weights;
                    if (is_vertical){
                        s_points = {0.5};
                        weights = {1.0};
                    } else {
                        s_points = {s1, s2};
                        weights = {w1, w2};
                    }
                    
                    // Compute the integral using quadrature
                    double integral = 0.0;
                    for (size_t i = 0; i < s_points.size(); ++i){
                        double s = s_points[i];
                        double w = weights[i];

                        // Position along the edge
                        double r_s = r_param(s);
                        double z_s = z_param(s);

                        // Shape function value at this point
                        double N_i = shape_func(s);

                        // Traction at this point
                        auto [t_r, t_z] = traction_func(r_s, z_s);

                        // Integrand: N_i * t_compoenent * r
                        double integrand;
                        if (is_radial){
                            integrand = N_i * t_r * r_s;
                        } else {
                            integrand = N_i * t_z * r_s;
                        }

                        // Add to the integral
                        integral += w * integrand;
                    }

                    // Add contribution to the load vector
                    f_traction[dof_idx] = edge_length * integral;
                    
                }
            }
        }
        
        return f_traction;
    }

    Eigen::VectorXd axisymmetric::assemble_element_load_vector(
        const Eigen::MatrixXd& element_vertices,
        const std::function<std::pair<double, double>(double, double)>& body_force_func,
        const Eigen::VectorXi& traction_edges,
        const std::function<std::pair<double, double>(double, double)>& traction_func
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;

        // Initialize global load vector
        Eigen::VectorXd f_element = Eigen::VectorXd::Zero(n_dofs);

        // Compute body force contribution
        Eigen::VectorXd f_body = compute_element_load_body_force(element_vertices, body_force_func);
        f_element += f_body;

        // Add boundary traction contribution if provided
        if (traction_edges.size() > 0 && traction_func){
            Eigen::VectorXd f_traction = compute_element_load_boundary_traction(element_vertices, traction_edges, traction_func);
            f_element += f_traction;
        }

        return f_element;
    }

    // === STABILIZATION TERMS ===
    Eigen::MatrixXd axisymmetric::compute_divergence_stabilization_matrix(
        const Eigen::MatrixXd& element_vertices,
        double E,
        double nu
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;

        // Calculate element size
        double area = utils::operations::calcArea(element_vertices);
        double h_E = std::sqrt(area);

        // Calculate weighted measure
        double weighted_measure = 2 * M_PI * compute_weighted_volume_polygon(element_vertices);
        
        // Initialize the stabilization matrix
        Eigen::MatrixXd K_div = Eigen::MatrixXd::Zero(n_dofs, n_dofs);
        
        // Compute a smoothly increasing stabilization parameter
        // Base value scales with Young's modulus
        double base_tau = 0.01 * E;
        double tau_2;

        if (nu < 0.4){
            // No enhancement needed for low Poisson's ratios
            tau_2 = 0.0;
        } else if (nu >= 0.4 && nu <0.45) {
            // Smooth linear transition to start stabilization
            double t = (nu - 0.4) / 0.05; // t in [0, 1]
            tau_2 = base_tau * t;
        } else if (nu >= 0.45){
            // Smooth quadratic scaling as we approach 0.5
            // This grows rapidly but not as abruptly as 1/(0.5-nu)
            double t = std::min(1.0, (nu - 0.45) / 0.045); // Saturates at nu = 0.495
            tau_2 = base_tau * (1 + 10 * t * t);
        }

        if (nu >= 0.495){
            tau_2 = base_tau * 2.55e3; // Modify the last value to ensure stability
        }
        
        // For each pair of DOFs, compute the contribution to K_div
        for(int i = 0; i<n_dofs; i++){
            for(int j = 0; j<n_dofs; j++){
                // Create unit displacement vectors for DOFs i and j
                Eigen::VectorXd u_i = Eigen::VectorXd::Zero(n_dofs);
                u_i[i] = 1.0;

                Eigen::VectorXd u_j = Eigen::VectorXd::Zero(n_dofs);
                u_j[j] = 1.0;
                
                // Compute boundary integrals for these unit displacements
                double boundary_integral_i = compute_divergence_boundary_integral(element_vertices, u_i);
                double boundary_integral_j = compute_divergence_boundary_integral(element_vertices, u_j);

                // Compute projected divergences
                double P_div_i = boundary_integral_i / weighted_measure;
                double P_div_j = boundary_integral_j / weighted_measure;

                // Contribution to stabilization matrix
                K_div(i, j) = tau_2 * h_E * h_E * P_div_i * P_div_j * weighted_measure;
            }
        }

        return K_div;
    }


    Eigen::MatrixXd axisymmetric::compute_boundary_stabilization_matrix(
        const Eigen::MatrixXd& element_vertices,
        const Eigen::MatrixXd& P,
        double E,
        double nu
    ) {
        // Number of vertices and DOFs
        int n_vertices = element_vertices.rows();
        int n_dofs = 2 * n_vertices;

        // Calculate element size
        double h_E = utils::operations::calcPolygonalDiam(element_vertices, n_vertices);

        // Calculate stabilization parameter τ
        double tau = material::mat::compute_stabilization_parameter(E, nu);

        // Initialize stabilization matrix
        Eigen::MatrixXd K_bound = Eigen::MatrixXd::Zero(n_dofs, n_dofs);

        // Calculate I-P
        Eigen::MatrixXd I_minus_P = Eigen::MatrixXd::Identity(n_dofs, n_dofs) - P;

        // Loop over all edges of the element
        for(int edge_idx = 0; edge_idx < n_vertices; edge_idx++){
            int j = (edge_idx + 1) % n_vertices;

            // Extract edge vertices
            double r1 = element_vertices(edge_idx, 0);
            double z1 = element_vertices(edge_idx, 1);
            double r2 = element_vertices(j, 0);
            double z2 = element_vertices(j, 1);

            // Compute edge length and parameterization
            double dr = r2 - r1;
            double dz = z2 - z1;
            double edge_length = std::sqrt(dr*dr + dz*dz);

            // Skip if edge is too short
            if (edge_length < VERTICAL_TOLERANCE){
                continue;
            }

            // Check if edge is vertical
            bool is_vertical = std::abs(dr) < VERTICAL_TOLERANCE;

            // For each pair of DOFs, compute the edge contribution to K_bound
            for(int i = 0; i < n_dofs; i++){
                for(int k = 0; k < n_dofs; k++){
                    // Create unit displacement vectors for DOFs i and k
                    Eigen::VectorXd u_i = Eigen::VectorXd::Zero(n_dofs);
                    u_i[i] = 1.0;

                    Eigen::VectorXd u_k = Eigen::VectorXd::Zero(n_dofs);
                    u_k[k] = 1.0;

                    // Compute (I-P)u for both displacement vectors
                    Eigen::VectorXd I_minus_P_u_i = I_minus_P * u_i;
                    Eigen::VectorXd I_minus_P_u_k = I_minus_P * u_k;

                    // Extract displacements at edge endpoints for both vectors
                    double u_i_r1 = I_minus_P_u_i[2*edge_idx];
                    double u_i_z1 = I_minus_P_u_i[2*edge_idx+1];
                    double u_i_r2 = I_minus_P_u_i[2*j];
                    double u_i_z2 = I_minus_P_u_i[2*j+1];

                    double u_k_r1 = I_minus_P_u_k[2*edge_idx];
                    double u_k_z1 = I_minus_P_u_k[2*edge_idx+1];
                    double u_k_r2 = I_minus_P_u_k[2*j];
                    double u_k_z2 = I_minus_P_u_k[2*j+1];

                    double edge_integral = 0.0;

                    if (is_vertical){
                        // Use 1-point quadrature for vertical edges
                        double s = 0.5;
                        double weight = 1.0;

                        // Interpolate displacements at midpoint
                        double u_i_r_s = (1-s) * u_i_r1 + s * u_i_r2;
                        double u_i_z_s = (1-s) * u_i_z1 + s * u_i_z2;
                        double u_k_r_s = (1-s) * u_k_r1 + s * u_k_r2;
                        double u_k_z_s = (1-s) * u_k_z1 + s * u_k_z2;

                        // Compute r at midpoint
                        double r_s = r1; // Constant for vertical edge

                        // Compute dot product (I-P)u_i · (I-P)u_k
                        double dot_product = u_i_r_s * u_k_r_s + u_i_z_s * u_k_z_s;

                        // Compute integrand with 2πr factor
                        double integrand = 2 * M_PI * r_s * dot_product;
                        edge_integral = weight * integrand;

                    } else {
                        // Use 2-point Gauss quadrature for non-vertical edges
                        double s1 = 0.5 - std::sqrt(3.0) / 6.0;
                        double s2 = 0.5 + std::sqrt(3.0) / 6.0;
                        double w1 = 0.5, w2 = 0.5;

                        // First Gauss point
                        double r_s1 = (1-s1) * r1 + s1 * r2;
                        double u_i_r_s1 = (1-s1) * u_i_r1 + s1 * u_i_r2;
                        double u_i_z_s1 = (1-s1) * u_i_z1 + s1 * u_i_z2;
                        double u_k_r_s1 = (1-s1) * u_k_r1 + s1 * u_k_r2;
                        double u_k_z_s1 = (1-s1) * u_k_z1 + s1 * u_k_z2;
                        double dot_product1 = u_i_r_s1 * u_k_r_s1 + u_i_z_s1 * u_k_z_s1;
                        double integrand1 = 2 * M_PI * r_s1 * dot_product1;

                        // Second Gauss point
                        double r_s2 = (1-s2) * r1 + s2 * r2;
                        double u_i_r_s2 = (1-s2) * u_i_r1 + s2 * u_i_r2;
                        double u_i_z_s2 = (1-s2) * u_i_z1 + s2 * u_i_z2;
                        double u_k_r_s2 = (1-s2) * u_k_r1 + s2 * u_k_r2;
                        double u_k_z_s2 = (1-s2) * u_k_z1 + s2 * u_k_z2;
                        double dot_product2 = u_i_r_s2 * u_k_r_s2 + u_i_z_s2 * u_k_z_s2;
                        double integrand2 = 2 * M_PI * r_s2 * dot_product2;

                        // Combine with weights
                        edge_integral = w1 * integrand1 + w2 * integrand2;
                    }

                    // Scale by edge length and add to stabilization matrix
                    K_bound(i, k) += tau / h_E * edge_length * edge_integral;
                }
            }
        }

        return K_bound;
    }
        
} // namespace solver
