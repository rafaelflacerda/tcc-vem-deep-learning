#include "solver/parabolic.hpp"

namespace solver {

    void parabolic::setup_linear_dofs() {
        // For linear elements (order = 1), DOFs are just vertex values
        n_dofs = nodes.rows();

        for(int elem = 0; elem <  elements.rows(); ++elem) {
            int n_vertices = elements.cols();
            element_dof_map[elem].resize(n_vertices);

            for(int i = 0; i < n_vertices; ++i) {
                element_dof_map[elem][i] = elements(elem, i);
            }
        }

    }

    void parabolic::setup_high_order_dofs() {
        // For higher-order elements (order > 1), we need vertex +  edge + interior moment DOFs
        std::map<std::pair<int, int>, int> edge_dof_map;
        std::map<int, int> element_moments_dof_map; 

        int dof_counter =  nodes.rows();
    
        // Setup edge DOFs
        if (order >= 2) {
            std::set<std::pair<int, int>> processed_edges;

            for (int elem = 0; elem < elements.rows(); ++elem) {
                int n_vertices = elements.cols();

                for (int i = 0; i < n_vertices; ++i) {
                    int v1 = elements(elem, i);
                    int v2 = elements(elem, (i + 1) % n_vertices);

                    // Ensure consistent ordering of edge
                    std::pair<int, int> edge = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);

                    if (processed_edges.find(edge) == processed_edges.end()) {
                        edge_dof_map[edge] = dof_counter;
                        dof_counter += (order - 1);
                        processed_edges.insert(edge);
                    }
                }
            }

            // Setup interior moment DOFs
            int interior_dofs_per_element;
            if (order == 2) {
                interior_dofs_per_element = 1;  // One area functional: (1/|E|) ∫_E v dA
            } else {
                interior_dofs_per_element = (order - 1) * (order - 2) / 2;  // General formula for k≥3
            }
            
            for (int elem = 0; elem < elements.rows(); ++elem) {
                element_moments_dof_map[elem] = dof_counter;
                dof_counter += interior_dofs_per_element;
            }

        }
        
        n_dofs = dof_counter;

        // Build element DOF maps
        for (int elem = 0; elem < elements.rows(); ++elem) {
            build_element_dof_map(elem, edge_dof_map, element_moments_dof_map);
        }

        // Store edge DOF mapping
        edge_dof_map_ = edge_dof_map;

        if (debug_mode_) {
            total_n_dof_vertex_ = nodes.rows();  // Global vertex count
            total_n_dof_edge_ = edge_dof_map.size() * (order - 1);  // Global edge DOF count
            
            // CRITICAL FIX: For k=2, we need 1 interior DOF per element
            int interior_dofs_per_element;
            if (order == 2) {
                interior_dofs_per_element = 1;  // One area functional per element
            } else {
                interior_dofs_per_element = (order - 1) * (order - 2) / 2;  // General formula for k≥3
            }
            total_n_dof_interior_ = elements.rows() * interior_dofs_per_element;
        }
    }

    void parabolic::build_element_dof_map(
        int elem, 
        const std::map<std::pair<int, int>, int>& edge_dof_map, 
        const std::map<int, int>& element_moments_dof_map
    ) {
        int n_vertices = elements.cols();
        int n_dofs_local = count_local_dofs(n_vertices);
        element_dof_map[elem].resize(n_dofs_local);

        int local_dof = 0;

        // Debug: Initialize global DOF indices for this element
        std::vector<int> debug_dof_vertex_idx;
        std::vector<int> debug_dof_edge_idx;
        std::vector<int> debug_dof_interior_idx;

        // Add vertex DOFs first
        for (int i = 0; i < n_vertices; ++i){
            element_dof_map[elem][local_dof++] = elements(elem, i);

            if (debug_mode_) {
                debug_dof_vertex_idx.push_back(elements(elem, i));
            }
        }
        if (debug_mode_) global_vertex_dof_indices_.push_back(std::make_pair(elem, debug_dof_vertex_idx));

        // Add edge DOFs ( for k >= 2)
        if (order >=2 ){
            for (int i = 0; i < n_vertices; ++i){
                int v1 = elements(elem, i);
                int v2 = elements(elem, (i + 1) % n_vertices);

                std::pair<int, int> edge = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);

                int edge_start_dof = edge_dof_map.at(edge);
                for (int j = 0; j < order - 1; ++j){
                    element_dof_map[elem][local_dof++] = edge_start_dof + j;

                    if (debug_mode_) {
                        debug_dof_edge_idx.push_back(edge_start_dof + j);
                    }
                }
            }
            if (debug_mode_) global_edge_dof_indices_.push_back(std::make_pair(elem, debug_dof_edge_idx));
        
            // Add interior moments DOFs
            int interior_start_dof = element_moments_dof_map.at(elem);
            // CRITICAL FIX: For k=2, we need 1 interior DOF (area functional)
            int interior_dofs;
            if (order == 2) {
                interior_dofs = 1;  // One area functional: (1/|E|) ∫_E v dA
            } else {
                interior_dofs = (order - 1) * (order - 2) / 2;  // General formula for k≥3
            }
            
            for (int j = 0; j < interior_dofs; ++j){
                element_dof_map[elem][local_dof++] = interior_start_dof + j;

                if (debug_mode_) {
                    debug_dof_interior_idx.push_back(interior_start_dof + j);
                }
            }

            if (debug_mode_) global_interior_dof_indices_.push_back(std::make_pair(elem, debug_dof_interior_idx));
        }

            
    }

    void parabolic::setup_global_dofs(){
        if (order == 1){
            setup_linear_dofs();
        } else {
            setup_high_order_dofs();
        }
    }

    // ============================================================================
    // MATRIX MANIPULATION
    // ============================================================================

    void parabolic::assemble_system() {
        if (debug_mode_) {
            std::cout << "Assembling VEM system matrices..." << std::endl;
        }
        
        // Initialize global matrices if not already done
        if (M_h.rows() != n_dofs || M_h.cols() != n_dofs) {
            initialize_global_matrices();
        }

        // Reset matrices to zero
        M_h.setZero();
        K_h.setZero();

        // Initialize storage for projection matrices
        element_P_nabla_.resize(elements.rows());
        element_P_0_.resize(elements.rows());
        element_data_cache_.resize(elements.rows());

        // Assemble all elements
        for (int elem_idx = 0; elem_idx < elements.rows(); ++elem_idx) {
            assemble_element(elem_idx);
        }

        // Compress sparse matrices for efficiency
        M_h.makeCompressed();
        K_h.makeCompressed();

        if (debug_mode_) {
            std::cout << "VEM system assembly completed:" << std::endl;
            std::cout << "  Matrix size: " << n_dofs << "x" << n_dofs << std::endl;
            std::cout << "  Mass matrix nnz: " << M_h.nonZeros() << std::endl;
            std::cout << "  Stiffness matrix nnz: " << K_h.nonZeros() << std::endl;
        }
    }

    void parabolic::assemble_element(int element_idx){
        ElementData element_data;

        // Setup element geometry
        setup_element_geometry(element_idx, element_data);
        
        // Debug output for first element
        if (element_idx == 0 && debug_mode_) {
            std::cout << "\n=== DEBUG: First Element (8-vertex) ===" << std::endl;
            std::cout << "Element index: " << element_idx << std::endl;
            std::cout << "Number of vertices: " << element_data.n_vertices << std::endl;
            std::cout << "Element area: " << element_data.area << std::endl;
            std::cout << "Element diameter h_e: " << element_data.h_e << std::endl;
            std::cout << "Centroid: (" << element_data.centroid(0) << ", " << element_data.centroid(1) << ")" << std::endl;
        }

        // Construct monomial basis
        construct_monomial_basis(element_data);

        // Compute polynomial matrices
        compute_polynomial_matrices(element_data);
        
        if (element_idx == 0 && debug_mode_) {
            std::cout << "M_poly matrix:\n" << element_data.M_poly << std::endl;
        }

        // Compute projection matrices
        compute_projection_matrices(element_data);
        
        if (element_idx == 0 && debug_mode_) {
            std::cout << "P_0 projection matrix:\n" << element_data.P_0 << std::endl;
            
            // Check projection quality
            Eigen::MatrixXd D_E = compute_D_matrix(element_data);
            Eigen::MatrixXd G = D_E.transpose() * D_E;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(G);
            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
            std::cout << "\n=== PROJECTION QUALITY CHECK ===" << std::endl;
            std::cout << "Number of DOFs: " << element_data.n_dofs_local << std::endl;
            std::cout << "Number of polynomial basis functions: " << N_k << std::endl;
            std::cout << "Overconstraint ratio: " << element_data.n_dofs_local << "/" << N_k << " = " 
                      << (double)element_data.n_dofs_local / N_k << std::endl;
            std::cout << "D_E^T * D_E condition number: " << cond << std::endl;
            std::cout << "(Higher ratio & condition = worse projection quality)" << std::endl;
        }

        // Store projection matrices and element data for later use
        element_P_nabla_[element_idx] = element_data.P_nabla;
        element_P_0_[element_idx] = element_data.P_0;
        element_data_cache_[element_idx] = element_data;

        // Asemble local matrices
        assemble_local_matrices(element_data);
        
        if (element_idx == 0 && debug_mode_) {
            std::cout << "Local mass matrix M_local diagonal:\n";
            for (int i = 0; i < std::min(8, (int)element_data.M_local.rows()); ++i) {
                std::cout << "  M_local[" << i << "," << i << "] = " << element_data.M_local(i, i) << std::endl;
            }
        }

        // Assemble into global matrices
        assemble_to_global_system(element_idx, element_data);
        
    }


    void parabolic::setup_element_geometry(
        int elem_idx, 
        ElementData& element_data
    ){
        // Count actual vertices for this element (skip -1 padding)
        element_data.n_vertices = 0;
        for (int i = 0; i < elements.cols(); ++i) {
            if (elements(elem_idx, i) != -1) {
                element_data.n_vertices++;
            } else {
                break; // -1 padding is always at the end
            }
        }
        
        // Initialize vertices matrix with the correct size
        element_data.vertices = Eigen::MatrixXd(element_data.n_vertices, 2);

        // Extract element vertices
        for (int i = 0; i < element_data.n_vertices; ++i){
            int node_idx = elements(elem_idx, i);
            element_data.vertices.row(i) = nodes.row(node_idx);
        }

        // Compute the centroid of the element
        element_data.centroid = utils::operations::calcCentroid(element_data.vertices);

        // Compute the polygonal diameter of the element
        element_data.h_e = utils::operations::calcPolygonalDiam(element_data.vertices, element_data.n_vertices);

        // Compute the area of the element
        element_data.area = utils::operations::calcArea(element_data.vertices);
        
        // Calculate the number of local DOFs
        element_data.n_dofs_local = count_local_dofs(element_data.n_vertices);
    }

    void parabolic::construct_monomial_basis(ElementData& element_data){
        element_data.monomial_powers.clear();

        // Standard VEM ordering: ξ powers first, then η powers
        for (int total_degree = 0; total_degree <= order; ++total_degree){
            for (int alpha_2 = 0; alpha_2 <= total_degree; ++alpha_2){  // η powers
                int alpha_1 = total_degree - alpha_2;                    // ξ powers
                element_data.monomial_powers.push_back(std::make_pair(alpha_1, alpha_2));
            }
        }
    }

    void parabolic::setup_quadrature_rule(
        ElementData& element_data, 
        std::vector<Eigen::Vector2d>& quad_points, 
        std::vector<double>& quad_weights
    ){
        quad_points.clear();
        quad_weights.clear();

        // Use proper 2D Gaussian quadrature for each triangle
        // 3-point rule for triangles (exact up to degree 2)
        std::vector<double> tri_xi = {1.0/6.0, 2.0/3.0, 1.0/6.0};
        std::vector<double> tri_eta = {1.0/6.0, 1.0/6.0, 2.0/3.0};
        std::vector<double> tri_weights = {1.0/3.0, 1.0/3.0, 1.0/3.0};

        for (int i = 0; i < element_data.n_vertices; ++i){
            int j = (i + 1) % element_data.n_vertices;

            Eigen::Vector2d v0 = element_data.centroid; 
            Eigen::Vector2d v1 = element_data.vertices.row(i);
            Eigen::Vector2d v2 = element_data.vertices.row(j);

            double tri_area = utils::operations::compute_triangle_area(v0, v1, v2);

            // Add multiple quadrature points per triangle
            for (size_t q = 0; q < tri_xi.size(); ++q) {
                Eigen::Vector2d point = tri_xi[q] * v0 + tri_eta[q] * v1 + (1.0 - tri_xi[q] - tri_eta[q]) * v2;
                quad_points.push_back(point);
                quad_weights.push_back(tri_weights[q] * tri_area);
            }
        }
    }

    void parabolic::compute_energy_projection(ElementData& element_data){
        // DEBUG: Print dimensions for k=2
        // if (debug_mode_ && order == 2) {
        //     std::cout << "\n=== ENERGY PROJECTION DEBUG (k=2) ===" << std::endl;
        //     std::cout << "element_data.n_dofs_local: " << element_data.n_dofs_local << std::endl;
        //     std::cout << "N_k: " << N_k << std::endl;
        //     std::cout << "P_nabla dimensions: " << element_data.P_nabla.rows() << "×" << element_data.P_nabla.cols() << std::endl;
        // }
        
        for (int dof_idx = 0; dof_idx < element_data.n_dofs_local; ++dof_idx){
            // Setup normalizaion constraint
            Eigen::MatrixXd A_constrained;
            Eigen::VectorXd b_constrained;

            // Compute RHS vector for DOF φi
            Eigen::VectorXd rhs_i = compute_energy_rhs_for_dof(element_data, dof_idx);

            if (order == 1){
                // TODO: put in the same pattern as for k>=2
                // For k=1, add boundary constraint: ∫_∂K (φi - Π^∇φi) ds = 0
                add_boundary_constraint(element_data, dof_idx, A_constrained, b_constrained, rhs_i);
                // Solve the constrained system
                Eigen::VectorXd p_i = A_constrained.fullPivLu().solve(b_constrained);
                // Update the energy projection matrix
                element_data.P_nabla.col(dof_idx) = p_i.head(N_k);
            } else {
                // For k≥2, add interior constraint: ∫_K (φi - Π^∇φi) dx = 0
                add_interior_constraint(element_data, dof_idx, A_constrained, b_constrained, rhs_i);
            }
        }
    }

    void parabolic::compute_projection_matrices(ElementData& element_data){
        // DEBUG: Show initialization for k=2
        // if (debug_mode_ && order == 2) {
        //     std::cout << "\n=== INITIALIZING PROJECTION MATRICES (k=2) ===" << std::endl;
        //     std::cout << "N_k: " << N_k << std::endl;
        //     std::cout << "element_data.n_dofs_local: " << element_data.n_dofs_local << std::endl;
        //     std::cout << "About to initialize P_nabla as " << N_k << "×" << element_data.n_dofs_local << std::endl;
        // }
        
        // Initialize projection matrices with correct dimensions and zeros
        // DO NOT use Identity - it's mathematically wrong and dimensionally impossible
        element_data.P_nabla = Eigen::MatrixXd::Zero(N_k, element_data.n_dofs_local);
        element_data.P_0 = Eigen::MatrixXd::Zero(N_k, element_data.n_dofs_local);

        // DEBUG: Confirm initialization for k=2
        // if (debug_mode_ && order == 2) {
        //     std::cout << "P_nabla initialized: " << element_data.P_nabla.rows() << "×" << element_data.P_nabla.cols() << std::endl;
        //     std::cout << "P_0 initialized: " << element_data.P_0.rows() << "×" << element_data.P_0.cols() << std::endl;
        // }
        
        // Step 1: Compute energy projection P^∇
        compute_energy_projection(element_data);

        // Step 2: Compute L2 projection P^0 using enhanced constraints
        compute_l2_projection(element_data);
        
        // DEBUG: Print projection matrices for order=1 to understand the issue
        if (order == 1 && debug_mode_) {
            // Only debug the first element to avoid overwhelming output
            static int debug_element_count = 0;
            if (debug_element_count == 0) {
                std::cout << "\n=== VEM PROJECTION MATRICES DEBUG (order=1, Element 0) ===" << std::endl;
                std::cout << "Element area: " << element_data.area << std::endl;
                std::cout << "Element h_e: " << element_data.h_e << std::endl;
                std::cout << "Element centroid: (" << element_data.centroid(0) << ", " << element_data.centroid(1) << ")" << std::endl;
                std::cout << "Number of local DOFs: " << element_data.n_dofs_local << std::endl;
                
                std::cout << "\nPolynomial mass matrix M_poly:" << std::endl;
                std::cout << element_data.M_poly << std::endl;
                
                std::cout << "\nPolynomial stiffness matrix K_poly:" << std::endl;
                std::cout << element_data.K_poly << std::endl;
                
                std::cout << "\nEnergy projection P_nabla:" << std::endl;
                std::cout << element_data.P_nabla << std::endl;
                
                std::cout << "\nL2 projection P_0:" << std::endl;
                std::cout << element_data.P_0 << std::endl;
                
                // Check projection properties
                Eigen::VectorXd ones = Eigen::VectorXd::Ones(element_data.n_dofs_local);
                Eigen::VectorXd P0_sum = element_data.P_0.transpose() * Eigen::Vector3d(1.0, 0.0, 0.0);
                std::cout << "\nMass conservation check (should sum to 1): " << P0_sum.sum() << std::endl;
                
                std::cout << "=== END VEM DEBUG ===" << std::endl;
            }
            debug_element_count++;
        }
    }

    void parabolic::compute_local_stiffness_matrix(ElementData& element_data) {
        // Consistency term: K_c = P_nabla^T * K_poly * P_nabla
        Eigen::MatrixXd K_c = element_data.P_nabla.transpose() * element_data.K_poly * element_data.P_nabla;

        // Stabilization term: S_K = α_E * S_E
        double alpha_E = compute_stiffness_stabilization_parameter(element_data);
        Eigen::MatrixXd S_E = compute_stabilization_matrix(element_data);
        Eigen::MatrixXd S_K = alpha_E * S_E;

        // DEBUG: Check matrix dimensions before addition
        if (debug_mode_ && order == 2) {
            std::cout << "\n=== MATRIX DIMENSIONS DEBUG ===\n";
            std::cout << "P_nabla: " << element_data.P_nabla.rows() << "×" << element_data.P_nabla.cols() << std::endl;
            std::cout << "K_poly: " << element_data.K_poly.rows() << "×" << element_data.K_poly.cols() << std::endl;
            std::cout << "K_c: " << K_c.rows() << "×" << K_c.cols() << std::endl;
            std::cout << "S_E: " << S_E.rows() << "×" << S_E.cols() << std::endl;
            std::cout << "S_K: " << S_K.rows() << "×" << S_K.cols() << std::endl;
            std::cout << "n_dofs_local: " << element_data.n_dofs_local << std::endl;
        }

        // Local stiffness matrix: K_local = K_c + S_K
        if (order == 1){
            element_data.K_local = K_c + S_K;
        } else {
            element_data.K_local = K_c + S_K;
        }

        // DEBUG: Show stiffness matrix assembly for k=2
        if (debug_mode_ && order == 2) {
            static int stiff_debug_count = 0;
            if (stiff_debug_count == 0) {
                std::cout << "\n=== LOCAL STIFFNESS MATRIX DEBUG (k=2) ===\n";
                std::cout << "Element has " << element_data.n_dofs_local << " local DOFs\n";
                
                // Show matrix components
                std::cout << "Consistency term K_c:\n" << K_c << std::endl;
                std::cout << "Stabilization parameter α_E: " << alpha_E << std::endl;
                std::cout << "Stabilization matrix S_E:\n" << S_E << std::endl;
                std::cout << "Final stiffness matrix K_local:\n" << element_data.K_local << std::endl;
                
                // Show stiffness matrix diagonal for first few DOFs
                std::cout << "\nStiffness matrix diagonal entries (first 8 DOFs):\n";
                for (int i = 0; i < std::min(8, element_data.n_dofs_local); ++i) {
                    std::cout << "  K_local[" << i << "," << i << "] = " << element_data.K_local(i, i);
                    if (i >= element_data.n_vertices) {
                        std::cout << " (edge DOF)";
                    } else {
                        std::cout << " (vertex DOF)";
                    }
                    std::cout << std::endl;
                }
                stiff_debug_count++;
            }
        }
    }

    void parabolic::compute_local_mass_matrix(ElementData& element_data) {
        if (use_lumped_mass_matrix_){
            Eigen::VectorXd s_E = compute_lumped_row_sum_vector(element_data);

            element_data.M_local = Eigen::MatrixXd::Zero(element_data.n_dofs_local, element_data.n_dofs_local);
            for (int i = 0; i < element_data.n_dofs_local; ++i){
                element_data.M_local(i, i) = s_E(i);
            }
            
            // Debug: Check mass conservation for lumped matrix
            if (debug_mode_) {
                static int debug_count = 0;
                if (debug_count < 3) {  // Show first 3 elements
                    std::cout << "\n=== LUMPED MASS DEBUG (Element " << debug_count << " with " << element_data.n_vertices << " vertices) ===" << std::endl;
                    std::cout << "Element area: " << element_data.area << std::endl;
                    std::cout << "Element h_e: " << element_data.h_e << std::endl;
                    std::cout << "Lumped mass per DOF (all): ";
                    for (int i = 0; i < element_data.n_dofs_local; ++i) {
                        std::cout << s_E(i) << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "Total lumped mass: " << s_E.sum() << std::endl;
                    std::cout << "Expected (area): " << element_data.area << std::endl;
                    std::cout << "Ratio (should be ~1): " << s_E.sum() / element_data.area << std::endl;
                    std::cout << "Mass matrix diagonal range: [" << s_E.minCoeff() << ", " << s_E.maxCoeff() << "]" << std::endl;
                    debug_count++;
                }
            }
        } else {
            // FULL CONSISTENCY TERM: M_c = P_0^T * M_poly * P_0
            Eigen::MatrixXd M_c;
            if (order == 1){
                M_c = element_data.P_nabla.transpose() * element_data.M_poly * element_data.P_nabla;
            } else {
                M_c = element_data.P_0.transpose() * element_data.M_poly * element_data.P_0;
            }

            // Stabilization term: S_M = β_M * S_E
            double beta_M = compute_mass_stabilization_parameter(element_data);
            Eigen::MatrixXd S_E = compute_stabilization_matrix(element_data);
            Eigen::MatrixXd S_M = beta_M * S_E;

            // Local mass matrix: M_local = M_c + S_M
            M_c = M_c + S_M;

            // if (use_lumped_mass_matrix_){
            //     compute_lumped_mass_matrix_naive(M_c);
            // }

            // Store the mass matrix (after lumping if needed)
            element_data.M_local = M_c;
        }

        // DEBUG: Show mass matrix assembly for k=2
        // if (debug_mode_ && order == 2) {
        //     static int mass_debug_count = 0;
        //     if (mass_debug_count == 0) {
        //         std::cout << "\n=== LOCAL MASS MATRIX DEBUG (k=2) ===\n";
        //         std::cout << "Element has " << element_data.n_dofs_local << " local DOFs\n";
        //         std::cout << "DOF breakdown: " << element_data.n_vertices << " vertex + " 
        //                  << (element_data.n_dofs_local - element_data.n_vertices) << " edge+interior\n";
                
        //         // Show matrix components
        //         std::cout << "Consistency term M_c:\n" << M_c << std::endl;
        //         std::cout << "Stabilization parameter β_M: " << beta_M << std::endl;
        //         std::cout << "Stabilization matrix S_E:\n" << S_E << std::endl;
        //         std::cout << "Final mass matrix M_local:\n" << element_data.M_local << std::endl;
                
        //         // Show mass matrix diagonal for first few DOFs
        //         std::cout << "\nMass matrix diagonal entries (first 8 DOFs):\n";
        //         for (int i = 0; i < std::min(8, element_data.n_dofs_local); ++i) {
        //             std::cout << "  M_local[" << i << "," << i << "] = " << element_data.M_local(i, i);
        //             if (i >= element_data.n_vertices) {
        //                 std::cout << " (edge DOF)";
        //             } else {
        //                 std::cout << " (vertex DOF)";
        //             }
        //             std::cout << std::endl;
        //         }
        //         mass_debug_count++;
        //     }
        // }
    }
    

    void parabolic::assemble_local_matrices(ElementData& element_data){
        compute_local_stiffness_matrix(element_data);
        compute_local_mass_matrix(element_data);
    }

    void parabolic::assemble_to_global_system(int element_idx, ElementData& element_data){
        const auto& dof_map = element_dof_map[element_idx];

        // Add local matrices to the global system
        for (int i = 0; i < element_data.n_dofs_local; ++i){
            for (int j = 0; j < element_data.n_dofs_local; ++j){
                int global_i = dof_map[i];
                int global_j = dof_map[j];

                // Add to sparse matrices (accumulating)
                M_h.coeffRef(global_i, global_j) += element_data.M_local(i, j);
                K_h.coeffRef(global_i, global_j) += element_data.K_local(i, j);
            }
        }
        
    }

    void parabolic::compute_local_load_vector(
        ElementData& element_data, 
        const SourceFunction& f, 
        double time
    ){
        // Initialize load vector - create new vector to avoid resize issues
        element_data.F_local = Eigen::VectorXd::Zero(element_data.n_dofs_local);

        // Step 1: Project source term f onto appropriate polynomial space
        Eigen::VectorXd fh_coeffs;
        compute_projected_source(element_data, f, time, fh_coeffs);

        // Step 2: Compute load vector entries
        for (int i = 0; i < element_data.n_dofs_local; ++i){
            element_data.F_local(i) = utils::integration::compute_load_integral_for_dof(element_data, fh_coeffs, i, N_k, order);
        }
    }

    void parabolic::assemble_local_to_global_load(
        int element_idx, 
        ElementData& element_data
    ){
        const auto& dof_map = element_dof_map[element_idx];

        // Add local load vector to the global system
        for (int i = 0; i < element_data.n_dofs_local; ++i){
            int global_dof = dof_map[i];
            F_h.coeffRef(global_dof) += element_data.F_local(i);
        }
    }

    void parabolic::assemble_load_vector(const SourceFunction& f, double time){
        // CRITICAL FIX: Reset global load vector to zero before assembly
        F_h = Eigen::VectorXd::Zero(n_dofs);

        // std::cout << "Assembling load vector at time t = " << time << "..." << std::endl;

        // Process each element
        for (int elem = 0; elem < elements.rows(); ++elem){
            ElementData element_data;

            setup_element_geometry(elem, element_data);

            // Construct monomial basis
            construct_monomial_basis(element_data);

            // Compute polynomial matrices first, then projection matrices
            compute_polynomial_matrices(element_data);
            compute_projection_matrices(element_data);

            // Compute local load vector
            compute_local_load_vector(element_data, f, time);

            // Assemble into global load vector
            assemble_local_to_global_load(elem, element_data);
        }
    }


    // ============================================================================
    // LHS AND RHS METHODS
    // ============================================================================

    void parabolic::compute_polynomial_matrices(ElementData& element_data){
        element_data.K_poly = Eigen::MatrixXd::Zero(N_k, N_k);
        element_data.M_poly = Eigen::MatrixXd::Zero(N_k, N_k);

        // Get element geometry properties
        double area = element_data.area;
        double cx = element_data.centroid(0);
        double cy = element_data.centroid(1);
        double h_e = element_data.h_e;
        
        // For order k=1: monomials are {1, x, y}
        if (order == 1) {
            // Mass matrix M_poly: ∫ m_i * m_j dx using exact polygonal moments
            element_data.M_poly(0, 0) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 0);  // ∫ 1*1 dx
            element_data.M_poly(0, 1) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 0);  // ∫ 1*ξ dx = 0
            element_data.M_poly(0, 2) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 1);  // ∫ 1*η dx = 0
            element_data.M_poly(1, 0) = element_data.M_poly(0, 1);  // ∫ ξ*1 dx = 0 (symmetric)
            element_data.M_poly(1, 1) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 0);  // ∫ ξ² dx
            element_data.M_poly(1, 2) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 1);  // ∫ ξ*η dx
            element_data.M_poly(2, 0) = element_data.M_poly(0, 2);  // ∫ η*1 dx = 0 (symmetric)
            element_data.M_poly(2, 1) = element_data.M_poly(1, 2);  // ∫ η*ξ dx (symmetric)
            element_data.M_poly(2, 2) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 2);  // ∫ η² dx
            
            // Stiffness matrix K_poly: ∫ ∇m_i · ∇m_j dx
            element_data.K_poly(0, 0) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(0, 1) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(0, 2) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(1, 0) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(1, 1) = area / (h_e * h_e);      // ∫ (1/h_e)*(1/h_e) dx = area/h_e²
            element_data.K_poly(1, 2) = 0.0;                     // ∇ξ ⊥ ∇η
            element_data.K_poly(2, 0) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(2, 1) = 0.0;                     // ∇ξ ⊥ ∇η
            element_data.K_poly(2, 2) = area / (h_e * h_e);      // ∫ (1/h_e)*(1/h_e) dx = area/h_e²
        }
        else if (order == 2) {
            // For order k=2: monomials are {1, ξ, η, ξ², ξη, η²}
            
            // Mass matrix computation using exact polygonal moments
            // M_poly(i,j) = ∫_K m_i * m_j dx where m_i, m_j are scaled monomials
            
            // Row 0: m_0 = 1
            element_data.M_poly(0, 0) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 0);  // ∫ 1*1 dx
            element_data.M_poly(0, 1) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 0);  // ∫ 1*ξ dx = 0
            element_data.M_poly(0, 2) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 1);  // ∫ 1*η dx = 0
            element_data.M_poly(0, 3) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 0);  // ∫ 1*ξ² dx
            element_data.M_poly(0, 4) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 1);  // ∫ 1*ξη dx
            element_data.M_poly(0, 5) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 2);  // ∫ 1*η² dx
            
            // Row 1: m_1 = ξ
            element_data.M_poly(1, 1) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 0);  // ∫ ξ*ξ dx
            element_data.M_poly(1, 2) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 1);  // ∫ ξ*η dx
            element_data.M_poly(1, 3) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 3, 0);  // ∫ ξ*ξ² dx
            element_data.M_poly(1, 4) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 1);  // ∫ ξ*ξη dx
            element_data.M_poly(1, 5) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 2);  // ∫ ξ*η² dx
            
            // Row 2: m_2 = η
            element_data.M_poly(2, 2) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 2);  // ∫ η*η dx
            element_data.M_poly(2, 3) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 1);  // ∫ η*ξ² dx
            element_data.M_poly(2, 4) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 2);  // ∫ η*ξη dx
            element_data.M_poly(2, 5) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 3);  // ∫ η*η² dx
            
            // Row 3: m_3 = ξ²
            element_data.M_poly(3, 3) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 4, 0);  // ∫ ξ²*ξ² dx
            element_data.M_poly(3, 4) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 3, 1);  // ∫ ξ²*ξη dx
            element_data.M_poly(3, 5) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 2);  // ∫ ξ²*η² dx
            
            // Row 4: m_4 = ξη
            element_data.M_poly(4, 4) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 2);  // ∫ ξη*ξη dx
            element_data.M_poly(4, 5) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 3);  // ∫ ξη*η² dx
            
            // Row 5: m_5 = η²
            element_data.M_poly(5, 5) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 4);  // ∫ η²*η² dx
            
            // Fill symmetric entries
            for (int i = 0; i < N_k; ++i) {
                for (int j = i + 1; j < N_k; ++j) {
                    element_data.M_poly(j, i) = element_data.M_poly(i, j);
                }
            }
            
            // Stiffness matrix K_poly: ∫ ∇m_i · ∇m_j dx
            element_data.K_poly(0, 0) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(0, 1) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(0, 2) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(0, 3) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(0, 4) = 0.0;                     // ∇(1) = 0
            element_data.K_poly(0, 5) = 0.0;                     // ∇(1) = 0
            
            element_data.K_poly(1, 1) = area / (h_e * h_e);      // ∫ (1/h_e)*(1/h_e) dx
            element_data.K_poly(1, 2) = 0.0;                     // ∇ξ ⊥ ∇η
            element_data.K_poly(1, 3) = 2.0 * utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 0)/ (h_e * h_e); // ∫ (1/h_e)*(2ξ/h_e) dx
            // std::cout << "K_poly(1, 3): " << element_data.K_poly(1, 3) << std::endl;
            element_data.K_poly(1, 4) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 1) / (h_e * h_e);       // ∫ (1/h_e)*(η/h_e) dx
            element_data.K_poly(1, 5) = 0.0;                     // ∇ξ ⊥ ∇(η²)_η
            
            element_data.K_poly(2, 2) = area / (h_e * h_e);      // ∫ (1/h_e)*(1/h_e) dx
            element_data.K_poly(2, 3) = 0.0;                     // ∇η ⊥ ∇(ξ²)_ξ
            element_data.K_poly(2, 4) = utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 0) / (h_e * h_e);       // ∫ (1/h_e)*(ξ/h_e) dx
            element_data.K_poly(2, 5) = 2.0 * utils::integration::compute_scaled_polygonal_moment_Ipq(element_data,0, 1) / (h_e * h_e); // ∫ (1/h_e)*(2η/h_e) dx
            // std::cout << "K_poly(2, 5): " << element_data.K_poly(2, 5) << std::endl;
            
            element_data.K_poly(3, 3) = 4.0 * utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 0) / (h_e * h_e); // ∫ (2ξ/h_e)*(2ξ/h_e) dx
            element_data.K_poly(3, 4) = 2.0 * utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 1) / (h_e * h_e); // ∫ (2ξ/h_e)*(η/h_e) dx
            element_data.K_poly(3, 5) = 0.0;                     // ∇(ξ²)_ξ ⊥ ∇(η²)_η
            
            element_data.K_poly(4, 4) = (
                utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 2, 0) + 
                utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 2)) / (h_e * h_e); // ∇(ξη) = (η/h_e, ξ/h_e)
            element_data.K_poly(4, 5) = 2.0 * utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 1, 1) / (h_e * h_e); // ∫ (ξ/h_e)*(2η/h_e) dx
            
            element_data.K_poly(5, 5) = 4.0 * utils::integration::compute_scaled_polygonal_moment_Ipq(element_data, 0, 2) / (h_e * h_e); // ∫ (2η/h_e)*(2η/h_e) dx
            
            // Fill symmetric entries
            for (int i = 0; i < N_k; ++i) {
                for (int j = i + 1; j < N_k; ++j) {
                    element_data.K_poly(j, i) = element_data.K_poly(i, j);
                }
            }
        }
        else {
            // For higher orders, use the existing quadrature-based approach as fallback
            // Setup quadrature rule
            std::vector<Eigen::Vector2d> quad_points;
            std::vector<double> quad_weights;
            setup_quadrature_rule(element_data, quad_points, quad_weights);

            // Compute polynomial matrices
            for (int i = 0; i < N_k; ++i){
                for (int j = 0; j < N_k; ++j){
                    double stiff_integral = 0.0;
                    double mass_integral = 0.0;

                    for (size_t q = 0; q < quad_points.size(); ++q){
                        // Evanluate monomials and gradients
                        Eigen::Vector2d grad_i = utils::operations::evaluate_monomial_gradient(i, quad_points[q], element_data);
                        Eigen::Vector2d grad_j = utils::operations::evaluate_monomial_gradient(j, quad_points[q], element_data);

                        double phi_i = utils::operations::evaluate_monomial(i, quad_points[q], element_data);
                        double phi_j = utils::operations::evaluate_monomial(j, quad_points[q], element_data);

                        stiff_integral += quad_weights[q] * grad_i.dot(grad_j);
                        mass_integral += quad_weights[q] * phi_i * phi_j;
                        
                    }

                    element_data.K_poly(i, j) = stiff_integral;
                    element_data.M_poly(i, j) = mass_integral;
                }
            }
        }
    }

    Eigen::VectorXd parabolic::compute_energy_rhs_for_dof(
        const ElementData& element_data, 
        int dof_idx
    ){
        Eigen::VectorXd rhs(N_k);

        for (int j = 0; j < N_k; ++j){
            if (order == 1){
                double boundary_term = utils::integration::compute_vertex_dof_monomial_boundary_integral_k1(element_data, dof_idx, j);
                rhs(j) = boundary_term;
            } else {
                double area_term = utils::integration::compute_moment_dof_laplacian_monomial_area_integral(element_data, dof_idx, j, order, N_k);
                double boundary_term = utils::integration::compute_dof_monomial_boundary_integral(element_data, dof_idx, j, order);

                rhs(j) = area_term + boundary_term;
            }
        }

        //std::cout << "rhs for dof " << dof_idx << " : " << rhs.transpose() << std::endl;

        return rhs;
    }

    void parabolic::add_boundary_constraint(
        ElementData& element_data,
        int dof_idx,
        Eigen::MatrixXd& A_constrained,
        Eigen::VectorXd& b_constrained,
        const Eigen::VectorXd& rhs_i
    ){
        // k=1 VEM Energy Projection with Vertex-Average Constraint
        // Following the Mathematica implementation algorithm
        
        // Step 1: Compute scaled vertex coordinates
        int n_vertices = element_data.n_vertices;
        Eigen::MatrixXd scaled_vertices(n_vertices, 2);
        double cx = element_data.centroid(0);
        double cy = element_data.centroid(1);
        double h_e = element_data.h_e;
        
        for (int i = 0; i < n_vertices; ++i) {
            scaled_vertices(i, 0) = (element_data.vertices(i, 0) - cx) / h_e;  // ξᵢ
            scaled_vertices(i, 1) = (element_data.vertices(i, 1) - cy) / h_e;  // ηᵢ
        }
        
        // Step 2: Compute average scaled coordinates
        double xi_bar = scaled_vertices.col(0).mean();   // ξ̄
        double eta_bar = scaled_vertices.col(1).mean();  // η̄
        
        // Step 3: Extract boundary RHS terms (already computed in compute_energy_rhs_for_dof)
        // rhs_i[0] should be 0 (constant monomial), rhs_i[1] = b₁, rhs_i[2] = b₂
        double rhs_b1 = rhs_i(1);  // Boundary integral for ξ monomial
        double rhs_b2 = rhs_i(2);  // Boundary integral for η monomial
        
        // Step 4: Solve G_high * [c₁, c₂]ᵀ = [b₁, b₂]ᵀ
        // For k=1: G_high = (|E|/h_E²) * I₂
        // Therefore: [c₁, c₂]ᵀ = (h_E²/|E|) * [b₁, b₂]ᵀ
        double area = element_data.area;
        double scaling = (h_e * h_e) / area;
        
        double c1 = scaling * rhs_b1;  // ξ coefficient
        double c2 = scaling * rhs_b2;  // η coefficient
        
        // Step 5: Apply vertex-average constraint
        // (1/N_v) ∑ᵣ (Π^∇φₛ)(Vᵣ) = (1/N_v) ∑ᵣ φₛ(Vᵣ) = 1/N_v
        // Since (Π^∇φₛ)(Vᵣ) = c₀ + c₁*ξᵣ + c₂*ηᵣ
        // We get: c₀ + c₁*ξ̄ + c₂*η̄ = 1/N_v
        // Therefore: c₀ = 1/N_v - c₁*ξ̄ - c₂*η̄
        double c0 = (1.0 / n_vertices) - c1 * xi_bar - c2 * eta_bar;
        
        // Step 6: Set up trivial constrained system that gives the correct solution
        // The calling code expects to solve A_constrained * x = b_constrained
        // We set up an identity system where x = [c0, c1, c2]
        A_constrained = Eigen::MatrixXd::Identity(N_k, N_k);
        b_constrained = Eigen::VectorXd(N_k);
        b_constrained(0) = c0;  // Constant coefficient
        b_constrained(1) = c1;  // ξ coefficient  
        b_constrained(2) = c2;  // η coefficient
        
        // Debug output for first few DOFs
        if (debug_mode_ && dof_idx < 4) {
            std::cout << "\n=== BOUNDARY CONSTRAINT DEBUG (k=1, DOF " << dof_idx << ") ===" << std::endl;
            std::cout << "Scaled vertex coordinates:" << std::endl;
            for (int i = 0; i < n_vertices; ++i) {
                std::cout << "  Vertex " << i << ": (ξ, η) = (" 
                         << std::fixed << std::setprecision(4) 
                         << scaled_vertices(i, 0) << ", " << scaled_vertices(i, 1) << ")" << std::endl;
            }
            std::cout << "Average scaled coordinates: ξ̄ = " << xi_bar << ", η̄ = " << eta_bar << std::endl;
            std::cout << "RHS: [" << rhs_b1 << ", " << rhs_b2 << "]" << std::endl;
            std::cout << "Scaling factor (h_E²/|E|): " << scaling << std::endl;
            std::cout << "Coefficients: c₀ = " << c0 << ", c₁ = " << c1 << ", c₂ = " << c2 << std::endl;
            std::cout << "=== END BOUNDARY CONSTRAINT DEBUG ===" << std::endl;
        }
    }

    void parabolic::add_interior_constraint(
        ElementData& element_data,
        int dof_idx,
        Eigen::MatrixXd& A_constrained,
        Eigen::VectorXd& b_constrained,
        const Eigen::VectorXd& rhs_i
    ){

        if (order == 2){
            // Extract the high-order system (excluding constant term)
            Eigen::MatrixXd K_poly_high = element_data.K_poly.block(1, 1, 5, 5);
            Eigen::VectorXd rhs_high = rhs_i.segment(1, 5);

            // Solve for high-order coefficients
            Eigen::VectorXd c_high = K_poly_high.fullPivLu().solve(rhs_high);

            // Check if this is the interior DOF (last DOF in local numbering)
            int n_dofs = element_data.n_dofs_local;
            double interior_value = (dof_idx == n_dofs - 1) ? 1.0 : 0.0;  // δ_{s,int}

            // Compute moment sum: Sum[cHigh[[l]] * Mpoly[[l+1, 1]], {l, 1, 5}] / area
            double moment_sum = 0.0;
            for (int l = 0; l < 5; ++l) {
                moment_sum += c_high(l) * element_data.M_poly(l + 1, 0);  // l+1 because we skip constant term
            }
            moment_sum /= element_data.area;

            // Compute c0
            double c0 = interior_value - moment_sum;

            // Store coefficients in P_nabla
            element_data.P_nabla(0, dof_idx) = c0;  // Constant term
            
            for (int l = 0; l < 5; ++l) {
                if (std::abs(c_high(l)) > 1e-14) {
                    element_data.P_nabla(l + 1, dof_idx) = c_high(l);  // Higher-order terms
                } else {
                    element_data.P_nabla(l + 1, dof_idx) = 0.0;
                }
            }
        } else { // TODO: verify this implementation for higher orders
            // Setup the augmented system
            A_constrained = Eigen::MatrixXd::Zero(N_k + 1, N_k + 1);
            b_constrained = Eigen::VectorXd::Zero(N_k + 1);

            // Copy the original system to the correct positions in the augmented system
            A_constrained.topLeftCorner(N_k, N_k) = element_data.K_poly;
            b_constrained.head(N_k) = rhs_i;

            // Compute ∫_K m_j dx for each monomial (these are the entries of M_poly first row)
            for (int j = 0; j < N_k; ++j){
                A_constrained(N_k, j) = element_data.M_poly(0, j);
                A_constrained(j, N_k) = A_constrained(N_k, j);
            }

            // RHS of constants
            b_constrained(N_k) = utils::integration::compute_dof_area_integral(element_data, dof_idx, order);
        }
        
    }

    void parabolic::compute_l2_projection(ElementData& element_data){
        // CORRECTED VEM L² PROJECTION APPROACH
        // The L² projection should satisfy: ∫_K (v_h - Π^0 v_h) q dx = 0 for all q ∈ P_k
        // This gives us: ∫_K Π^0 v_h q dx = ∫_K v_h q dx for all q ∈ P_k
        
        // For VEM, the constraint is: ∫_K φᵢ mⱼ dx = ∫_K Π^0 φᵢ mⱼ dx
        // Where Π^0 φᵢ = ∑_l P_0(l,i) m_l
        
        // This leads to: M_poly * P_0 = D
        // Where D(j,i) = ∫_K φᵢ mⱼ dx
        
        Eigen::MatrixXd G = element_data.M_poly;        // Mass matrix of scaled monomials
        Eigen::MatrixXd D(N_k, element_data.n_dofs_local);  // Constraint matrix
        
        // Initialize constraint matrix
        D.setZero();

        if (order == 1) {
            int Nv = element_data.n_vertices;
            double area = element_data.area;
        
            for (int s = 0; s < Nv; ++s) {
                Eigen::Vector2d vs = element_data.vertices.row(s);
                double xi  = (vs(0) - element_data.centroid(0)) / element_data.h_e;
                double eta = (vs(1) - element_data.centroid(1)) / element_data.h_e;
        
                D(0,s) = area / Nv;          // ∫ φ_s          = |K|/Nv
                D(1,s) = area * xi / Nv;     // ∫ φ_s ξ        = |K| ξ_s / Nv
                D(2,s) = area * eta / Nv;    // ∫ φ_s η        = |K| η_s / Nv
            }
        
            element_data.P_0 = element_data.M_poly.fullPivLu().solve(D);
            return;      // finished k = 1
        }
        
        if (order == 2){
            // Initialize D matrix to zero
            D.setZero(N_k, element_data.n_dofs_local);

            // Row 0: Only the interior-average DOF gets 1.0, all others get 0
            // For k=2, the interior DOF represents the element‐average: ∫_E φ_i dx = |E| · φ_int
            // Therefore the constant-monomial row must store |E| (the area) in that column.
            // The interior DOF is the last local DOF (vertex DOFs + edge DOFs).
            int n_vertices = element_data.n_vertices;
            int n_edges = n_vertices;  // For polygons, n_edges = n_vertices
            int interior_dof_idx = n_vertices + n_edges;  // vertex DOFs + edge DOFs = interior DOF index
            
            if (interior_dof_idx < element_data.n_dofs_local) {
                D(0, interior_dof_idx) = element_data.area;  // Interior DOF IS the constant monomial functional
            }

            // Rows 1-5: Fill via matrix product D_high = (1/|E|) * M_poly_high * P_nabla
            // Extract high rows of M_poly (rows j=1..5, all columns)
            Eigen::MatrixXd M_poly_high = element_data.M_poly.block(1, 0, 5, 6);

            // Constraint matrix: D_high = (1/area) * M_poly_high * P_nabla
            // M_poly_high is 5×6, P_nabla is 6×n_dofs_local, result should be 5×n_dofs_local
            Eigen::MatrixXd D_high =   M_poly_high * element_data.P_nabla;

            // Insert into D matrix (rows 1-5)
            D.block(1, 0, 5, D.cols()) = D_high;

            if (debug_mode_) {
                // CRITICAL: Show the computed D matrix for comparison with manual version
                // std::cout << "\nComputed D matrix (" << D.rows() << "x" << D.cols() << "):" << std::endl;
                // std::cout << std::fixed << std::setprecision(6);
                // for (int i = 0; i < D.rows(); ++i) {
                //     std::cout << "  [";
                //     for (int j = 0; j < D.cols(); ++j) {
                //         std::cout << std::setw(10) << D(i, j);
                //         if (j < D.cols() - 1) std::cout << ", ";
                //     }
                //     std::cout << "]" << std::endl;
                // }
            }

            // Solve M_poly * P_0 = D  (no extra area scaling)
            element_data.P_0 =  element_data.M_poly.fullPivLu().solve(D);
            return ;

        } 
    }

    // ============================================================================
    // STABILIZATION COMPUTATION
    // ============================================================================
    double parabolic::compute_stiffness_stabilization_parameter(const ElementData& element_data){
        /*
         * Option B (standard in VEM literature):
         *   α_E = c · trace(K_c) ,   with  c ∈ [0.1,0.3]
         *   – K_c = P_nabla^T K_poly P_nabla  (consistency part)
         *     but using trace(K_poly) gives the same |E|⁻¹ scaling.
         */
        const double c = 0.1;           // tuned once (edge/vertex balance)

        if (element_data.K_poly.rows() == 0) return 0.0; // safeguard

        return c * element_data.K_poly.trace() / static_cast<double>(element_data.K_poly.rows());
    }

    double parabolic::compute_mass_stabilization_parameter(const ElementData& element_data){
        /*
         * Option B (standard in VEM literature):
         *   β_E = c · trace(M_c) ,   with  c ∈ [0.1,0.3]
         * Using trace(M_poly) (which ≈ |E| for scaled monomials) gives the
         * correct |E| scaling and keeps edge / vertex ratios mesh-independent.
         */
        const double c = 0.1;           // tuned once (edge/vertex balance)

        if (element_data.M_poly.rows() == 0) return 0.0; // safeguard

        return c * element_data.M_poly.trace() / static_cast<double>(element_data.M_poly.rows());
    }

    Eigen::MatrixXd parabolic::compute_stabilization_matrix(const ElementData& element_data){
        int n_dofs = element_data.n_dofs_local;

        // PROPER VEM STABILIZATION MATRIX
        // S_E = (I - Π) where Π is the proper orthogonal projector onto polynomial space
        // Π = P_nabla^T * (P_nabla * P_nabla^T)^{-1} * P_nabla
        
        // Identity matrix
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n_dofs, n_dofs);
        
        // Compute the proper orthogonal projector
        // First compute P_nabla * P_nabla^T (N_k × N_k matrix)
        Eigen::MatrixXd PnablaPnablaT = element_data.P_nabla * element_data.P_nabla.transpose();
        
        // Compute its inverse (with regularization for numerical stability)
        Eigen::MatrixXd PnablaPnablaT_inv;
        double regularization = 1e-12;
        Eigen::MatrixXd regularized = PnablaPnablaT + regularization * Eigen::MatrixXd::Identity(PnablaPnablaT.rows(), PnablaPnablaT.cols());
        
        // Use LU decomposition for better numerical stability
        Eigen::FullPivLU<Eigen::MatrixXd> lu_solver(regularized);
        if (lu_solver.isInvertible()) {
            PnablaPnablaT_inv = lu_solver.inverse();
        } else {
            // Fallback to pseudoinverse using SVD
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(PnablaPnablaT, Eigen::ComputeThinU | Eigen::ComputeThinV);
            PnablaPnablaT_inv = svd.solve(Eigen::MatrixXd::Identity(PnablaPnablaT.rows(), PnablaPnablaT.cols()));
        }
        
        // Compute the proper orthogonal projector: Π = P_nabla^T * (P_nabla * P_nabla^T)^{-1} * P_nabla
        Eigen::MatrixXd Pi = element_data.P_nabla.transpose() * PnablaPnablaT_inv * element_data.P_nabla;
        
        // Stabilization matrix: S_E = I - Π
        Eigen::MatrixXd S_E = I - Pi;
        
        // Ensure the matrix is symmetric (numerical precision)
        S_E = 0.5 * (S_E + S_E.transpose());
        
        // Ensure positive semi-definiteness by zeroing out negative eigenvalues
        // This is important for VEM stability
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(S_E);
        if (eigensolver.info() == Eigen::Success) {
            Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
            Eigen::MatrixXd eigenvectors = eigensolver.eigenvectors();
            
            // Zero out negative eigenvalues (numerical precision issues)
            for (int i = 0; i < eigenvalues.size(); ++i) {
                if (eigenvalues(i) < 1e-12) {
                    eigenvalues(i) = 0.0;
                }
            }
            
            // Reconstruct the matrix: S_E = V * Λ * V^T
            S_E = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
        }
        
        return S_E;
    }

    // ============================================================================
    // MANIPULATION OF SOURCE TERMS
    // ============================================================================

    void parabolic::compute_projected_source(
        ElementData& element_data,
        const SourceFunction& f,
        double time,
        Eigen::VectorXd& fh_coeffs
    ){
        // Determine the projection degree based on VEM order
        int projection_degree;
        if (order == 1) projection_degree = 0;
        else projection_degree = order - 2;

        // Number of monomials in the projection
        int n_proj_monomials = (projection_degree + 1) * (projection_degree + 2) / 2;
        fh_coeffs = Eigen::VectorXd::Zero(n_proj_monomials);

        // Set Gaussian quadrature points and weights
        std::vector<Eigen::Vector2d> quad_points;
        std::vector<double> quad_weights;
        setup_quadrature_rule(element_data, quad_points, quad_weights);

        // Extract the mass matrix block from M_poly for the projection space
        Eigen::MatrixXd M_proj = element_data.M_poly.block(0, 0, n_proj_monomials, n_proj_monomials);
        
        // Compute the RHS (f, m_i) for each monomial m_i in the projection space
        Eigen::VectorXd b_proj = Eigen::VectorXd::Zero(n_proj_monomials);
        
        for (int i = 0; i < n_proj_monomials; ++i){
            double rhs_i = 0.0;
            
            for (size_t q = 0; q < quad_points.size(); ++q){
                double f_value = f(quad_points[q], time);
                double monomial_i_value = utils::operations::evaluate_monomial(i, quad_points[q], element_data);
                double contribution = quad_weights[q] * f_value * monomial_i_value;
                
                rhs_i += contribution;
            }
            
            b_proj(i) = rhs_i;
        }
        
        // Solve the projected system
        fh_coeffs = M_proj.fullPivLu().solve(b_proj);
    }

    // ============================================================================
    // INITIAL CONDITIONS
    // ============================================================================

    void parabolic::set_initial_conditions(const InitialFunctions& u0){
        // Initialize global solution vector
        U_h = Eigen::VectorXd::Zero(n_dofs);

        std::cout << "Setting initial condition using VEM interpolation operator I_h..." << std::endl;

        // DEBUG: Show initial condition function values at a few test points
        if (debug_mode_) {
            std::cout << "\n=== INITIAL CONDITION DEBUG ===" << std::endl;
            std::cout << "Testing initial condition function at key points:" << std::endl;
            std::vector<Eigen::Vector2d> test_points = {
                {0.25, 0.25}, {0.5, 0.5}, {0.125, 0.125}, {0.375, 0.25}
            };
            for (const auto& pt : test_points) {
                double val = u0(pt);
                std::cout << "  u0(" << pt.x() << ", " << pt.y() << ") = " << val << std::endl;
            }
            std::cout << "=== END INITIAL CONDITION DEBUG ===" << std::endl;
        }

        // Process each element to evaluate local DOF functionals
        for (int elem = 0; elem < elements.rows(); ++elem){
            // Setup element geometry information
            ElementData element_data;
            setup_element_geometry(elem, element_data);

            // Get DOF map for this element
            const auto& dof_map = element_dof_map[elem];
            int n_vertices = element_data.n_vertices;

            // Apply DOF functionals χ_i(u0) to get DOF values
            // 1. VERTEX DOFs: χ_i(u) = u(vertex_i)
            for(int v = 0; v < n_vertices; ++v){
                Eigen::Vector2d vertex = element_data.vertices.row(v);
                double dof_value = u0(vertex); // χ_i(u0) = u0(vertex_i)

                int global_dof = dof_map[v];
                U_h(global_dof) = dof_value;
            }

            // 2. EDGE DOFs (k ≥ 2): χ_i(u) = ∫_e u · L̂_j(s) ds
            if (order >= 2) {
                // DEBUG: Show edge DOF processing for first element
                if (debug_mode_ && elem == 0) {
                    std::cout << "\n=== EDGE DOF PROCESSING (Element 0) ===" << std::endl;
                    std::cout << "Processing " << n_vertices << " edges, each with " << (order-1) << " DOFs" << std::endl;
                }
                
                int dof_offset = n_vertices;

                for (int edge_idx = 0; edge_idx < n_vertices; ++edge_idx) {
                    Eigen::Vector2d v1 = element_data.vertices.row(edge_idx);
                    Eigen::Vector2d v2 = element_data.vertices.row((edge_idx + 1) % n_vertices);
                    double edge_length = (v2 - v1).norm();

                    // For each moment on this edge (j = 0, ..., k-2)
                    for (int moment_idx = 0; moment_idx < order - 1; ++moment_idx){
                        double functional_value = 0.0;
                        double orthonormal_scale = std::sqrt((2.0 * moment_idx + 1.0) / 2.0); // Move outside loop

                        // Set Gaussian quadrature points and weights
                        // int n_quad = (order + 2) / 2 + 1; // Sufficient for degree k integration
                        int n_quad = 4;
                        std::vector<double> quad_points, quad_weights;
                        utils::integration::get_gauss_quadrature_rule(n_quad, quad_points, quad_weights);

                        for (int q = 0; q < n_quad; ++q){
                            // Map point quadrature to edge
                            double xi = quad_points[q];
                            Eigen::Vector2d point = 0.5 * ((1 - xi) * v1 + (1 + xi) * v2);

                            // Evaluate u0 at quadrature point
                            double u0_value = u0(point);

                            // Evaluate orthonormal Legendre polynomial Ĺⱼ(s) at quadrature point
                            double legendre_value = utils::operations::evaluate_legendre_polynomial(moment_idx, xi);
                            double monomial_value = orthonormal_scale * legendre_value;

                            functional_value += quad_weights[q] * u0_value * monomial_value; 
                        }

                        // Apply Jacobian and normalization by edge length |e|
                        functional_value *= 0.5; 

                        // Store DOF value (avoid redundant overwrites for shared edges)
                        int global_dof = dof_map[dof_offset];
                        if (U_h(global_dof) == 0.0){
                            // Only set if not already computed
                            U_h(global_dof) = functional_value;
                        }
                        
                        // DEBUG: Show edge DOF initial condition values for first element
                        if (debug_mode_ && elem == 0 && edge_idx < 2 && moment_idx == 0) {
                            std::cout << "Edge " << edge_idx << " DOF " << global_dof << ":" << std::endl;
                            std::cout << "  Edge vertices: (" << v1.x() << "," << v1.y() << ") to (" << v2.x() << "," << v2.y() << ")" << std::endl;
                            std::cout << "  Edge length: " << edge_length << std::endl;
                            std::cout << "  Moment index: " << moment_idx << std::endl;
                            std::cout << "  Orthonormal scale: " << orthonormal_scale << std::endl;
                            std::cout << "  Raw functional value: " << functional_value << std::endl;
                            std::cout << "  Final DOF value: " << U_h(global_dof) << std::endl;
                        }

                        dof_offset++;
                    }
                }
                
                if (debug_mode_ && elem == 0) {
                    std::cout << "=== END EDGE DOF PROCESSING ===" << std::endl;
                }
            }

            if (order >= 2){
                // 3. INTERIOR MOMENT DOFs (k ≥ 2): χ_α(u) = (1/|K|) ∫_K u · m_α dx
                int n_edge_dofs = n_vertices * (order - 1);
                int dof_offset = n_vertices + n_edge_dofs;

                // Number of interior moments = dim(M_{k-2}(K))
                int n_interior_moments = (order - 1) * (order - 2) / 2;

                // construct_monomial_basis(element_data);
                double h_E = element_data.h_e;

                // Evaluate each interior moment DOF functional
                int moment_count = 0;
                for (int total_degree = 0; total_degree <= order - 2; ++total_degree){
                    for (int x_deg = 0; x_deg <= total_degree; ++x_deg){
                        int y_deg = total_degree - x_deg;

                        if (moment_count >= n_interior_moments) break;

                        double functional_value = 0.0;

                        // Simple 2D quadrature rule for integration
                        std::vector<Eigen::Vector2d> quad_points;
                        std::vector<double> quad_weights;
                        setup_quadrature_rule(element_data, quad_points, quad_weights);

                        // Evaluate ∫_K u0 · m_α dx using quadrature
                        for (size_t q = 0; q < quad_points.size(); ++q){
                            Eigen::Vector2d point = quad_points[q];

                            // Evaluate u0 at quadrature point
                            double u0_value = u0(point);

                            // Evaluate monomial m_α at quadrature point
                            double x = (point.x() - element_data.centroid.x()) / h_E;
                            double y = (point.y() - element_data.centroid.y()) / h_E;
                            double monomial_value = std::pow(x, x_deg) * std::pow(y, y_deg);

                            functional_value += quad_weights[q] * u0_value * monomial_value;
                        }

                        // DOF normalization (1/|K|)
                        functional_value *= (1.0 / element_data.area);

                        // Store DOF value
                        int global_dof = dof_map[dof_offset + moment_count];
                        U_h(global_dof) = functional_value;
                        moment_count++;
                    }
                    if (moment_count >= n_interior_moments) break;
                }
            }
        }

        std::cout << "VEM interpolation completed. Global DOF vector norm: " << U_h.norm() << std::endl;
        std::cout << "Ready for time integration with optimal initial condition." << std::endl;
    }
    
    // ============================================================================
    // PROJECTION MATRICES ACCESS FUNCTIONS
    // ============================================================================

    const Eigen::MatrixXd& parabolic::get_element_P0_matrix(int element_idx) const {
        if (element_idx < 0 || element_idx >= element_P_0_.size()) {
            throw std::out_of_range("Element index out of range for P_0 matrix access");
        }
        return element_P_0_[element_idx];
    }

    const Eigen::MatrixXd& parabolic::get_element_P_nabla_matrix(int element_idx) const {
        if (element_idx < 0 || element_idx >= element_P_nabla_.size()) {
            throw std::out_of_range("Element index out of range for P_nabla matrix access");
        }
        return element_P_nabla_[element_idx];
    }

    const ElementData& parabolic::get_element_data(int element_idx) const {
        if (element_idx < 0 || element_idx >= element_data_cache_.size()) {
            throw std::out_of_range("Element index out of range for ElementData access");
        }
        return element_data_cache_[element_idx];
    }

    bool parabolic::has_projection_matrices() const {
        return !element_P_0_.empty() && !element_P_nabla_.empty() && 
               element_P_0_.size() == elements.rows() && 
               element_P_nabla_.size() == elements.rows();
    }

    // ============================================================================
    // LUMPED MASS MATRIX
    // ============================================================================

    void parabolic::compute_lumped_mass_matrix_naive(Eigen::MatrixXd& M_c){
        // Create diagonal lumped mass matrix
        Eigen::VectorXd diagonal_entries(M_c.rows());
        
        // Compute row sums (diagonal entries of lumped matrix)
        for (int i = 0; i < M_c.rows(); ++i){
            double row_sum = 0.0;
            for (int j = 0; j < M_c.cols(); ++j){
                row_sum += M_c(i, j);
            }
            diagonal_entries(i) = row_sum;
        }
        
        M_c.setZero();
        for (int i = 0; i < M_c.rows(); ++i){
            M_c(i, i) = diagonal_entries(i);
        }
    }

    Eigen::MatrixXd parabolic::compute_unscaled_mass_matrix(const ElementData& element_data){
        Eigen::MatrixXd M_poly_E = Eigen::MatrixXd::Zero(N_k, N_k);

        for (int i = 0; i < N_k; ++i){
            for (int j = 0; j < N_k; ++j){
                auto [alpha_1_i, alpha_2_i] = element_data.monomial_powers[i];
                auto [alpha_1_j, alpha_2_j] = element_data.monomial_powers[j];

                Eigen::Vector2d origin(0.0, 0.0);

                M_poly_E(i, j) = utils::integration::compute_polygonal_moment_Ipq(
                    element_data, 
                    alpha_1_i + alpha_1_j, 
                    alpha_2_i + alpha_2_j, 
                    &origin
                );
            }
        }

        return M_poly_E;
    }

    Eigen::MatrixXd parabolic::compute_D_matrix(const ElementData& element_data){
        Eigen::MatrixXd D_E = Eigen::MatrixXd::Zero(N_k, element_data.n_dofs_local);

        // Scaled implementation
        for (int dof_idx = 0; dof_idx < element_data.n_dofs_local; ++dof_idx){
            for (int mono_idx = 0; mono_idx < N_k; ++mono_idx){
                Eigen::Vector2d vertex = element_data.vertices.row(dof_idx);
                D_E(mono_idx, dof_idx) = utils::operations::evaluate_monomial(mono_idx, vertex, element_data);
            }
        }

        // TODO: implement the edge DOFs and interior DOFs

        return D_E;
    }

    Eigen::VectorXd parabolic::compute_c_vector(const ElementData& element_data){
        Eigen::VectorXd c_E(N_k);

        // Scaled implementation
        for(int mono_idx = 0; mono_idx < N_k; ++mono_idx){
            auto [alpha_1, alpha_2] = element_data.monomial_powers[mono_idx];
            c_E(mono_idx) = utils::integration::compute_scaled_polygonal_moment_Ipq(
                element_data, 
                alpha_1, 
                alpha_2
            );
        }

        return c_E;
    }

    Eigen::VectorXd parabolic::compute_lumped_row_sum_vector(ElementData& element_data){
        Eigen::MatrixXd M_poly_E = element_data.M_poly;
        Eigen::MatrixXd D_E = compute_D_matrix(element_data);
        Eigen::VectorXd c_E = compute_c_vector(element_data);

        // DEBUG: Print the vectors
        if (debug_mode_) {
            static int lumped_debug_count = 0;
            if (lumped_debug_count < 1) {
                std::cout << "\n=== LUMPED MASS COMPUTATION DEBUG ===" << std::endl;
                std::cout << "c_E (monomial integrals): " << c_E.transpose() << std::endl;
                std::cout << "M_poly_E:\n" << M_poly_E << std::endl;
                lumped_debug_count++;
            }
        }

        // Solve M_poly_E * w_E = c_E
        Eigen::VectorXd w_E = M_poly_E.fullPivLu().solve(c_E);

        if (debug_mode_) {
            static int lumped_debug_count2 = 0;
            if (lumped_debug_count2 < 1) {
                std::cout << "w_E (solution): " << w_E.transpose() << std::endl;
                std::cout << "D_E:\n" << D_E << std::endl;
                lumped_debug_count2++;
            }
        }

        // Compute the row sum vector s_E = D_E^T * w_E
        Eigen::VectorXd s_E = D_E.transpose() * w_E;

        // CRITICAL: Scale to convert from scaled (dimensionless) coordinates to physical mass.
        // In scaled coordinates with the constant monomial, we get s_E = [1,1,...,1] (n_vertices ones).
        // The partition of unity gives: ∑_i d_i = |E|, so each DOF should get: |E| / n_vertices.
        // This is a row-sum lumping strategy that preserves total mass.
        s_E *= element_data.area / element_data.n_vertices;

        return s_E;
    }
}