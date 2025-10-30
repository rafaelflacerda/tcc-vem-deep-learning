#include "utils/integration.hpp"


namespace utils {
    void integration::setGaussParams(int integration_order){
        if(integration_order==3){
            gaussPoints = (Eigen::Matrix<double, 3, 2>() << 0.16666666666, 0.16666666666, 0.6666666666, 0.6666666666, 0.16666666666, 0.16666666666).finished();
            gaussWeights = (Eigen::Matrix<double,3, 1>()<< 0.3333333333, 0.3333333333, 0.3333333333).finished();
        }
    }

    void integration::setParamCoords(Eigen::MatrixXd coords, double s, double t){
        double x = coords(0,0) + (coords(1,0)-coords(0,0))*s + (coords(2,0)-coords(0,0))*t;
        double y = coords(0,1) + (coords(1,1)-coords(0,1))*s + (coords(2,1)-coords(0,1))*t;
        paramCoord = (Eigen::Matrix<double, 2,1>() << x, y).finished();
    }

    void integration::get_gauss_quadrature_rule(
        int required_order, 
        std::vector<double>& points, 
        std::vector<double>& weights
    ){
        // Get Gauss-Legendre quadrature rule with enough points for required precision
        // For polynomil of degree n, need at least (n+1)/2 points (rounded up)

        points.clear();
        weights.clear();

        int n_points = (required_order + 1) / 2;
        n_points = std::max(n_points, 2); // Minimum 2 points
        n_points = std::min(n_points, 10); // Maximum 10 points

        // Get points and weights from Eigen
        if (n_points == 1){
            // 1-point rule (degree of precision = 1)
            points = {0.0};
            weights = {2.0};
        } else if (n_points == 2){
            // 2-point rule (degree of precision = 3)
            points = {-1.0/std::sqrt(3.0), 1.0/std::sqrt(3.0)};
            weights = {1.0, 1.0};
        } else if (n_points == 3){
            // 3-point rule (degree of precision = 5)
            points = {-std::sqrt(3.0/5.0), 0.0, std::sqrt(3.0/5.0)};
            weights = {5.0/9.0, 8.0/9.0, 5.0/9.0};
        } else if (n_points == 4){
            // 4-point rule (degree of precision = 7)
            double sqrt_30 = std::sqrt(30.0);
            points = {-std::sqrt((3.0 + 2.0*sqrt_30/5.0)/7.0), 
                  -std::sqrt((3.0 - 2.0*sqrt_30/5.0)/7.0),
                   std::sqrt((3.0 - 2.0*sqrt_30/5.0)/7.0),
                   std::sqrt((3.0 + 2.0*sqrt_30/5.0)/7.0)};
            weights = {(18.0 - sqrt_30)/36.0, 
                    (18.0 + sqrt_30)/36.0,
                    (18.0 + sqrt_30)/36.0, 
                    (18.0 - sqrt_30)/36.0};
        } else if (n_points == 5){
            // For higher orders, use 5-point rule (degree of precision = 9)
            double sqrt_70 = std::sqrt(70.0);
            points = {-std::sqrt(5.0 + 2.0*sqrt_70/7.0)/3.0,
                    -std::sqrt(5.0 - 2.0*sqrt_70/7.0)/3.0,
                    0.0,
                    std::sqrt(5.0 - 2.0*sqrt_70/7.0)/3.0,
                    std::sqrt(5.0 + 2.0*sqrt_70/7.0)/3.0};
            weights = {(322.0 - 13.0*sqrt_70)/900.0,
                    (322.0 + 13.0*sqrt_70)/900.0,
                    128.0/225.0,
                    (322.0 + 13.0*sqrt_70)/900.0,
                    (322.0 - 13.0*sqrt_70)/900.0};
        }
    };

    // ============================================================================
    // DOF INTEGRATION METHODS
    // ============================================================================

    double integration::compute_dof_boundary_integral(
        const ElementData& element_data,
        int dof_idx,
        int order
    ){
        if (operations::is_vertex_dof(dof_idx, element_data)) {
            return compute_vertex_dof_boundary_integral(element_data, dof_idx);
        } else if (operations::is_edge_dof(dof_idx, element_data, order)) {
            return compute_edge_dof_boundary_integral(element_data, dof_idx, order);
        } 
        return 0.0;
    }

    double integration::compute_dof_area_integral(
        const ElementData& element_data,
        int dof_idx,
        int order
    ){
        if (operations::is_vertex_dof(dof_idx, element_data)) {
            return compute_vertex_dof_area_integral(element_data, dof_idx, order);
        } else if (operations::is_edge_dof(dof_idx, element_data, order)) {
            return compute_edge_dof_area_integral(element_data, dof_idx, order);
        } else if (operations::is_moment_dof(dof_idx, element_data, order)) { 
            return compute_moment_area_integral(element_data, dof_idx, order);
        }
        return 0.0;
    }

    double integration::compute_load_integral_for_dof(
        const ElementData& element_data,
        const Eigen::VectorXd& fh_coeffs,
        int dof_idx,
        int N_k,
        int order
    ){
        double integral = 0.0;
        double n_vertices = element_data.n_vertices;

        if (dof_idx < n_vertices){
            // Vertex DOF

            // Get the L^2 projection coefficients for this DOF
            Eigen::VectorXd pi0_coeffs = element_data.P_0.col(dof_idx);

            // Compute ∫_K f_h Π^0_k φ_i dx
            for (int i = 0; i < fh_coeffs.size(); ++i){
                for (int j = 0; j < N_k; j++){
                    // Compute ∫_K m_i m_j dx where m_i is from f_h and m_j from Π^0 φ_i
                    double m_ij_integral = element_data.M_poly(i, j);
                    integral += fh_coeffs(i) * pi0_coeffs(j) * m_ij_integral;
                }
            }
        } else if (order >= 2 && dof_idx < n_vertices + n_vertices * (order - 1)){
            // Edge DOF

            Eigen::VectorXd pi0_coeffs = element_data.P_0.col(dof_idx - n_vertices);

            for (int i = 0; i < fh_coeffs.size(); ++i){
                for (int j = 0; j < N_k; j++){
                    double m_ij_integral = element_data.M_poly(i, j);
                    integral += fh_coeffs(i) * pi0_coeffs(j) * m_ij_integral;
                }
            }
        } else if (order >= 2){
            // Moment DOF

            int moment_idx = dof_idx - n_vertices - n_vertices * (order - 1);

            // The interior DOF corresponds to (1/|K|) ∫_K φ_i m_α dx
            if (moment_idx < fh_coeffs.size()){
                // If the moment is within the projection space, use orthogonality
                integral = fh_coeffs(moment_idx) * element_data.area;
            } else {
                // Higher order moments of f_h are zero
                integral = 0.0;
            }
        }

        return integral;
    }


    // ============================================================================
    // VERTEX DOF INTEGRATION
    // ============================================================================

    double integration::compute_vertex_dof_boundary_integral(
        const ElementData& element_data, 
        int dof_idx
    ) {
        // For vertex DOFs: ∫_{∂K} N_i ds where N_i is the vertex shape function
        // For linear elements, each vertex contributes to adjacent edges
    
        int vertex_idx = operations::get_vertex_index(dof_idx, element_data);
        if (vertex_idx < 0) return 0.0;
        
        double total_integral = 0.0;
        
        // Vertex DOF contributes to its adjacent edges
        for (int edge_idx = 0; edge_idx < element_data.n_vertices; ++edge_idx) {
            if (operations::is_vertex_on_edge(vertex_idx, edge_idx, element_data.n_vertices)) {
                total_integral += compute_vertex_contribution_to_edge(element_data, vertex_idx, edge_idx);
            }
        }
        
        return total_integral;
    }
    
    double integration::compute_vertex_dof_area_integral(
        const ElementData& element_data, 
        int dof_idx,
        int order
    ){
        // For vertex DOFs: ∫_K N_i dx where N_i is the vertex shape function
        // For linear approximation, this is approximately area/n_vertices
        if (order == 1) {
            // Linear case: each vertex gets equal weight
            return element_data.area / element_data.n_vertices;
        } else {
            // Higher-order case: need more sophisticated integration
            // For now, use linear approximation
                return element_data.area / element_data.n_vertices;
            }
    }

    double integration::compute_vertex_contribution_to_edge(
        const ElementData& element_data,
        int vertex_idx,
        int edge_idx
    ){
        Eigen::Vector2d v0 = element_data.vertices.row(vertex_idx);
        Eigen::Vector2d v1 = element_data.vertices.row((vertex_idx + 1) % element_data.n_vertices);

        double edge_length = (v1 - v0).norm();
        
        // Linear shape function contributes half the edge length
        return edge_length / 2.0;
    }

    // ============================================================================
    // EDGE DOF INTEGRATION METHODS
    // ============================================================================

    double integration::compute_edge_dof_area_integral(
        const ElementData& element_data,
        int dof_idx, 
        int order
    ){
        if (order < 2) return 0.0;

        int local_edge_dof, edge_on_element;
        bool is_edge_dof = operations::get_edge_dof_info(element_data, dof_idx, local_edge_dof, edge_on_element, order);
        if (!is_edge_dof) return 0.0;

        // Use quadrature to integrate edge basis function over element
        return integrate_edge_dof_over_element(element_data, edge_on_element, local_edge_dof, order);
    }


    double integration::integrate_edge_dof_over_element(
        const ElementData& element_data,
        int edge_idx,
        int local_dof_idx,
        int order
    ){
        double integral = 0.0;

        // Triangulate element from centroid
        for (int tri_idx = 0; tri_idx < element_data.n_vertices; ++tri_idx){
            int next_tri_idx = (tri_idx + 1) % element_data.n_vertices;

            // Triangle vertices are centroid, vertex[tri_idx], vertex[next_tri_idx]
            Eigen::Vector2d v0 = element_data.centroid;
            Eigen::Vector2d v1 = element_data.vertices.row(tri_idx);
            Eigen::Vector2d v2 = element_data.vertices.row(next_tri_idx);

            // Calculate triangle area
            double tri_area = operations::compute_triangle_area(v0, v1, v2);

            // Use 3-point quadrature on triangle
            std::vector<Eigen::Vector2d> quad_points = {
                (v0 + v1 + v2) / 3.0, // Centroid
                (v0 + v1) / 2.0, // Edge Midpoint
                (v1 + v2) / 2.0, // Edge Midpoint
            };
            std::vector<double> quad_weights = {
                tri_area * 2.0/3.0, // Centroid
                tri_area * 1.0/6.0, // Edge Midpoint
                tri_area * 1.0/6.0, // Edge Midpoint
            };

            for (size_t i = 0; i < quad_points.size(); ++i){
                double edge_value = evaluate_edge_basis_at_point(
                    element_data,
                    quad_points[i],
                    edge_idx,
                    local_dof_idx,
                    order
                );
                integral += quad_weights[i] * edge_value;
            }
        }

        return integral;
    }
        

    double integration::evaluate_edge_basis_at_point(
        const ElementData& element_data,
        const Eigen::Vector2d& point,
        int edge_idx,
        int local_dof_idx,
        int order
    ){
        // Get edge endpoints
        Eigen::Vector2d v0 = element_data.vertices.row(edge_idx);
        Eigen::Vector2d v1 = element_data.vertices.row((edge_idx + 1) % element_data.n_vertices);

        // Check if the point lies on the edge (within tolerance)
        Eigen::Vector2d edge_vec = v1 - v0;
        Eigen::Vector2d point_vec = point - v0;
        
        // Project point onto edge line
        double t = point_vec.dot(edge_vec) / edge_vec.squaredNorm();
        
        // Check if projection parameter is within [0,1] (point is between vertices)
        if (t < 0.0 || t > 1.0) {
            return 0.0; // Point is outside the edge segment
        }
        
        // Check if point is actually on the edge (distance from edge line)
        Eigen::Vector2d projected_point = v0 + t * edge_vec;
        double distance_to_edge = (point - projected_point).norm();
        
        // Tolerance for considering a point "on the edge"
        double tolerance = 1e-10;
        if (distance_to_edge > tolerance) {
            return 0.0; // Point is not on the edge
        }
        
        // Point is on the edge, evaluate the edge basis function
        // Convert to reference coordinate s ∈ [-1,1]
        double s = 2.0 * t - 1.0;

        // Evaluate Legendre polynomial for edge DOFs
        if(local_dof_idx == 0 && order >= 2){
            // First edge DOF: P_1(s) = s (zero at both endpoints)
            return s;
        } else if (local_dof_idx == 1 && order >= 3){
            // Second edge DOF: P_2(s) = (3s²-1)/2 (zero at both endpoints)
            return (3.0 * s * s - 1.0) / 2.0;
        } else if (local_dof_idx == 2 && order >= 4){
            // Third edge DOF: P_3(s) = (5s³-3s)/2 (zero at both endpoints)
            return (5.0 * s * s * s - 3.0 * s) / 2.0;
        }

        return 0.0;
    }

    // ============================================================================
    // MOMEMNT DOF INTEGRATION METHODS
    // ============================================================================

    double integration::compute_moment_area_integral(
        const ElementData& element_data,
        int moment_dof_idx,
        int order
    ){
        // CRITICAL FIX: For k=2, we have 1 interior DOF (area functional)
        if (order < 2) return 0.0;  // Changed from order < 3 to order < 2
        
        // For k=2, the single interior DOF is the area functional
        if (order == 2) {
            // The interior DOF for k=2 is: χ_int(v) = (1/|E|) ∫_E v dA
            // So ∫_K φ_interior dx = area (since φ_interior integrates to area)
            return element_data.area;
        }

        // For k≥3, use the original logic
        int moment_idx = operations::get_interior_dof_index(element_data, moment_dof_idx, order);
        if (moment_idx == -1) return 0.0;

        // Interior DOFs correspond to moments in P_{k-2}
        // Map interior_idx to the corresponding monomial in our basis
        int monomial_idx = get_monomial_index_for_interior_dof(moment_idx, order);

        return compute_monomial_area_integral(element_data, monomial_idx);
    }

    int integration::get_monomial_index_for_interior_dof(int moment_idx, int order){
        int max_interior_degree = order - 2;
        int current_idx = 0;

        // Our monomial basis is ordered by total degree: {1, x, y, x², xy, y², ...}
        // Interior DOFs correspond to the first (k-2+1)(k-2+2)/2 monomials
        for (int total_degree = 0; total_degree <= max_interior_degree; ++total_degree){
            for (int alpha_1 = 0; alpha_1 <= total_degree; ++alpha_1){
                if (current_idx == moment_idx) return current_idx;
                current_idx++;
            }
        }

        return 0;
    }

    // ============================================================================
    // MONOMIAL INTEGRATION METHODS
    // ============================================================================

    double integration::compute_monomial_boundary_integral(
        const ElementData& element_data,
        int monomial_idx
    ){
        auto [alpha_1, alpha_2] = element_data.monomial_powers[monomial_idx];

        if(alpha_1 == 0 && alpha_2 == 0) return operations::compute_perimeter(element_data.vertices);

        // For higher-order monomials, integrate over each edge
        double integral = 0.0;

        for (int edge_idx = 0; edge_idx < element_data.n_vertices; ++edge_idx){
            integral += integrate_monomial_over_edge(element_data, edge_idx, monomial_idx);
        }

        return integral;
    }

    double integration::compute_monomial_area_integral(
        const ElementData& element_data,
        int monomial_idx
    ){
        // Choose direction based on which has lower degree to avoid division by zero
        auto [alpha_1, alpha_2] = element_data.monomial_powers[monomial_idx];
        if (alpha_1 == 0 && alpha_2 > 0){
            return compute_divergence_y_integral(element_data, alpha_1, alpha_2);
        } else {
            return compute_divergence_x_integral(element_data, alpha_1, alpha_2);
        }
    }

    double integration::integrate_monomial_over_edge(
        const ElementData& element_data,
        int edge_idx,
        int monomial_idx
    ){
        // Edge endpoints
        Eigen::Vector2d v0 = element_data.vertices.row(edge_idx);
        Eigen::Vector2d v1 = element_data.vertices.row((edge_idx + 1) % element_data.n_vertices);

        // Compute the edge length
        double edge_length = (v1 - v0).norm();

        // 2-point Gauss-Legendre Quadrature
        std::vector<double> gauss_xi;
        std::vector<double> gauss_weights;
        get_gauss_quadrature_rule(2, gauss_xi, gauss_weights);

        double integral = 0.0;

        for (size_t i = 0; i < gauss_xi.size(); ++i){
            double xi = gauss_xi[i];
            double weight = gauss_weights[i];

            // Compute the position along the edge
            Eigen::Vector2d point = 0.5 * (1.0 - xi) * v0 + 0.5 * (1.0 + xi) * v1;
            double monomial_value = operations::evaluate_monomial(monomial_idx, point, element_data);

            // Add contribution to the integral
            integral += weight * monomial_value;
        }

        return integral * edge_length * 0.5;
    }

    double integration::compute_divergence_x_integral(
        const ElementData& element_data,
        int alpha_1,
        int alpha_2
    ){
        if (alpha_1 == 0 && alpha_2 == 0) return element_data.area;

        double integral = 0.0;

        // Use x-direction divergence if alpha_1 > 0
        if (alpha_1 >= 0){
            for (int edge_idx = 0; edge_idx < element_data.n_vertices; ++edge_idx){
                integral += integrate_divergence_x_term(element_data, alpha_1 + 1, alpha_2, edge_idx);
            }
            integral /= (alpha_1 + 1);
        }
        return integral;
    }

    double integration::compute_divergence_y_integral(
        const ElementData& element_data,
        int alpha_1,
        int alpha_2
    ){
        if (alpha_1 == 0 && alpha_2 == 0) return element_data.area;

        double integral = 0.0;

        // Use y-direction divergence if alpha_2 > 0
        if (alpha_2 >= 0){
            for (int edge_idx = 0; edge_idx < element_data.n_vertices; ++edge_idx){
                integral += integrate_divergence_y_term(element_data, alpha_1, alpha_2 + 1, edge_idx);
            }
            integral /= (alpha_2 + 1);
        }
        return integral;
    }

    double integration::integrate_divergence_x_term(
        const ElementData& element_data,
        int alpha_1,
        int alpha_2,
        int edge_idx
    ){
        // Edge endpoints
        Eigen::Vector2d v0 = element_data.vertices.row(edge_idx);
        Eigen::Vector2d v1 = element_data.vertices.row((edge_idx + 1) % element_data.n_vertices);

        // Compute outward normal vector
        //TODO: Implement this in operations.hpp
        Eigen::Vector2d edge_vec = v1 - v0;
        Eigen::Vector2d normal(-edge_vec(1), edge_vec(0));
        double n_x = normal(0)/normal.norm();

        // Compute the edge length
        double edge_length = edge_vec.norm();

        // High-order Gauss-Legendre Quadrature
        std::vector<double> gauss_xi;
        std::vector<double> gauss_weights;
        get_gauss_quadrature_rule(alpha_1 + alpha_2 + 2, gauss_xi, gauss_weights);

        double edge_integral = 0.0;

        for (size_t i = 0; i < gauss_xi.size(); ++i){
            double xi = gauss_xi[i];
            double weight = gauss_weights[i];

            // Compute the position along the edge
            Eigen::Vector2d point = 0.5 * (1.0 - xi) * v0 + 0.5 * (1.0 + xi) * v1;

            // Evaluate scaled monomial at this point
            double scaled_x = (point(0) - element_data.centroid(0)) / element_data.h_e;
            double scaled_y = (point(1) - element_data.centroid(1)) / element_data.h_e;
            double monomial_value = std::pow(scaled_x, alpha_1) * std::pow(scaled_y, alpha_2);

            // Add contribution to the edge integral
            edge_integral += weight * monomial_value * n_x;

        }

        return edge_integral * edge_length * 0.5;
    }

    double integration::integrate_divergence_y_term(
        const ElementData& element_data,
        int alpha_1,
        int alpha_2,
        int edge_idx
    ){
        // Edge endpoints
        Eigen::Vector2d v0 = element_data.vertices.row(edge_idx);
        Eigen::Vector2d v1 = element_data.vertices.row((edge_idx + 1) % element_data.n_vertices);

        // Compute outward normal vector
        //TODO: Implement this in operations.hpp
        Eigen::Vector2d edge_vec = v1 - v0;
        Eigen::Vector2d normal(-edge_vec(1), edge_vec(0));
        double n_y = normal(1)/normal.norm();

        // Compute the edge length
        double edge_length = edge_vec.norm();

        // High-order Gauss-Legendre Quadrature
        std::vector<double> gauss_xi;
        std::vector<double> gauss_weights;
        get_gauss_quadrature_rule(alpha_1 + alpha_2 + 2, gauss_xi, gauss_weights);

        double edge_integral = 0.0;

        for (size_t i = 0; i < gauss_xi.size(); ++i){
            double xi = gauss_xi[i];
            double weight = gauss_weights[i];

            // Compute the position along the edge
            Eigen::Vector2d point = 0.5 * (1.0 - xi) * v0 + 0.5 * (1.0 + xi) * v1;

            // Evaluate scaled monomial at this point
            double scaled_x = (point(0) - element_data.centroid(0)) / element_data.h_e;
            double scaled_y = (point(1) - element_data.centroid(1)) / element_data.h_e;
            double monomial_value = std::pow(scaled_x, alpha_1) * std::pow(scaled_y, alpha_2);

            // Add contribution to the edge integral
            edge_integral += weight * monomial_value * n_y;

        }

        return edge_integral * edge_length * 0.5;
    }


    double integration::compute_moment_dof_laplacian_monomial_area_integral(
        const ElementData& element_data,
        int dof_idx,
        int monomial_idx,
        int order,
        int N_k
    ){
        // b_j^(s) volume part = - ∫_K φ_s Δ m_j dA
        // With our scaled monomials: Δ m_j = 0 except j in {ξ^2, η^2} (i.e., m3, m5),
        // and Δ m3 = Δ m5 = 2 / h_K^2 (returned by compute_monomial_laplacian).
        // For k = 2, only the interior mean DOF has a nonzero area integral:
        //   ∫_K φ_s dA = |K| if s = interior-mean DOF, and 0 otherwise.

        const double laplacian_mj = operations::compute_monomial_laplacian(element_data, monomial_idx);
        if (std::abs(laplacian_mj) < 1e-14) return 0.0; // j ∉ {m3, m5}

        // Vertex and edge DOFs have zero area mean → zero volume contribution
        if (operations::is_vertex_dof(dof_idx, element_data)) {
            return 0.0;
        }
        if (operations::is_edge_dof(dof_idx, element_data, order)) {
            return 0.0;
        }

        // Interior DOF(s)
        if (operations::is_moment_dof(dof_idx, element_data, order)) {
            // For k = 2 there is exactly one interior DOF: the cell-average.
            if (order == 2) {
                // volume term = - (Δ m_j) * |K|
                return -laplacian_mj * element_data.area;
            }
            // For k > 2, only the constant-moment DOF contributes to ∫_K φ_s dA.
            // If your mapping enumerates the constant moment as index 0, detect it:
            int moment_idx = operations::get_interior_dof_index(element_data, dof_idx, order);
            if (moment_idx == 0) {
                return -laplacian_mj * element_data.area; // constant-moment DOF
            }
            return 0.0; // higher interior moments have zero cell mean
        }

        return 0.0;
    }

    double integration::compute_vertex_dof_monomial_boundary_integral_k1(
        const ElementData& element_data,
        int dof_idx,
        int monomial_idx
    ){
        // k=1 version: RHS entry b^{(s)}_j = ∫_∂E φ_s ∂_n m_j ds
        // For k=1 we only have monomials m_0=1, m_1=ξ, m_2=η
        // Only m1, m2 contribute in the H1 projector RHS since ∇m_0 = 0
        if (monomial_idx == 0) return 0.0;
        if (monomial_idx != 1 && monomial_idx != 2) {
            // For k=1 we only ever need j=1,2. Others are zero by design.
            return 0.0;
        }

        // For vertex DOFs, dof_idx corresponds to vertex index
        int vertex_idx = dof_idx;
        
        const int Nv = element_data.n_vertices;
        const auto& V = element_data.vertices;
        const double hE = element_data.h_e;   // Element diameter for scaling

        // Helper lambda to compute contribution from one edge
        auto add_edge = [&](int a, int b) -> double {
            Eigen::Vector2d p1 = V.row(a);
            Eigen::Vector2d p2 = V.row(b);
            Eigen::Vector2d e  = p2 - p1;
            double L = e.norm();
            if (L == 0.0) return 0.0;

            // Outward unit normal for CCW ordering: rotate edge tangent by +90°
            Eigen::Vector2d n(e.y() / L, -e.x() / L);

            // ∂_n m_j is constant along the edge for k=1 scaled monomials
            // m_1 = ξ = (x - cx)/h_E, so ∂_n m_1 = n_x/h_E
            // m_2 = η = (y - cy)/h_E, so ∂_n m_2 = n_y/h_E
            double dn_m = (monomial_idx == 1) ? (n.x() / hE)
                         : (monomial_idx == 2) ? (n.y() / hE)
                         : 0.0;

            // For vertex basis φ_s on its adjacent edges:
            // ∫_e φ_s ds = |e|/2 (linear hat function integral)
            return 0.5 * L * dn_m;
        };

        // The two edges adjacent to vertex_idx (assuming CCW ordering)
        int s = vertex_idx;
        int sm1 = (s - 1 + Nv) % Nv;  // Previous vertex
        int sp1 = (s + 1) % Nv;       // Next vertex

        double edge_integral = 0.0;
        edge_integral += add_edge(sm1, s);   // edge e^- : (sm1 -> s)
        edge_integral += add_edge(s, sp1);   // edge e^+ : (s -> sp1)

        // Clean tiny roundoff errors
        if (std::abs(edge_integral) < 1e-14) edge_integral = 0.0;
        
        return edge_integral;
    }

    double integration::compute_vertex_dof_monomial_boundary_integral(
        const ElementData& element_data,
        int vertex_idx,
        int monomial_idx
    ){
           int n_vertices = element_data.n_vertices;
           double edge_integral = 0.0;

           // Integrate over both adjacent edges of this vertex
           for (int i =  0; i < 2; ++i){
                // Get the index of the edge
                int edge_idx = (i == 0 ) ? vertex_idx : (vertex_idx - 1 + n_vertices) % n_vertices;

                // Get the indices and coordinates of the two vertices of the edge
                int v1 = edge_idx;
                int v2 = (edge_idx + 1) % n_vertices;
                Eigen::Vector2d p1 = element_data.vertices.row(v1);
                Eigen::Vector2d p2 = element_data.vertices.row(v2);

                // Compute the outward normal vector
                Eigen::Vector2d edge = p2 - p1;
                double edge_length = edge.norm();
                // OUTWARD normal for CCW vertex order:
                Eigen::Vector2d normal(edge.y()/edge_length, -edge.x()/edge_length);

                // Get Gauss-Legendre quadrature points and weights
                std::vector<double> gauss_xi, gauss_weights;
                get_gauss_quadrature_rule(3, gauss_xi, gauss_weights);

                for (size_t j = 0; j < gauss_xi.size(); ++j){
                    double xi = gauss_xi[j];
                    Eigen::Vector2d point = p1 + 0.5 * (xi + 1.0) * edge;

                    // Evaluate basis functions at quadrature point
                    bool is_left  = (vertex_idx == v1);
                    bool is_right = (vertex_idx == v2);
                    assert(is_left || is_right && "vertex_idx is not an endpoint of this edge");

                    double L2 = 0.5*(3.0*xi*xi - 1.0);
                    double basis_value = is_left ? (-0.5*xi + 0.5*L2)  // u_{v^-}
                                                : (+0.5*xi + 0.5*L2); // u_{v^+}

                    // Evaluate monomial at quadrature point
                    Eigen::Vector2d monomial_value = operations::evaluate_monomial_gradient(monomial_idx, point, element_data);

                    // Add contribution to the edge integral
                    edge_integral += gauss_weights[j] * basis_value * monomial_value.dot(normal) * edge_length * 0.5;
                }
           }
           if (std::abs(edge_integral) < 1e-14) edge_integral = 0.0;
           return edge_integral;
    }

    double integration::compute_edge_dof_monomial_boundary_integral(
        const ElementData& element_data,
        int dof_idx,
        int monomial_idx,
        int order
    ){
        // Get the edge DOF index and local edge DOF index
        int edge_idx, local_edge_dof;
        operations::get_edge_dof_info(element_data, dof_idx, local_edge_dof, edge_idx, order);

        // Get the indices and coordinates of the two vertices of the edge
        int v1 = edge_idx;
        int v2 = (edge_idx + 1) % element_data.n_vertices;
        Eigen::Vector2d p1 = element_data.vertices.row(v1);
        Eigen::Vector2d p2 = element_data.vertices.row(v2);

        // Compute the outward normal vector
        Eigen::Vector2d edge = p2 - p1;
        double edge_length = edge.norm();
        Eigen::Vector2d normal(edge.y()/edge_length, -edge.x()/edge_length);

        // Get Gauss-Legendre quadrature points and weights
        std::vector<double> gauss_xi, gauss_weights;
        get_gauss_quadrature_rule(order + 1, gauss_xi, gauss_weights);

        double integral = 0.0;

        for (size_t i = 0; i < gauss_xi.size(); ++i){
            double xi = gauss_xi[i];
            Eigen::Vector2d point = p1 + 0.5 * (xi + 1.0) * edge;

            // Evaluate the edge basis function (typically Legendre polynomial of degree local_edge_dof + 1)
            //double basis_value = operations::evaluate_legendre_polynomial(local_edge_dof + 1, xi);
            double basis_value = 1.0- 0.5*(3.0*xi*xi - 1.0);

            // Evaluate the monomial gradient at the quadrature point
            Eigen::Vector2d monomial_value = operations::evaluate_monomial_gradient(monomial_idx, point, element_data);

            // Add contribution to the integral
            integral += gauss_weights[i] * basis_value * monomial_value.dot(normal) * edge_length * 0.5;
        }
        
        if (std::abs(integral) < 1e-14) integral = 0.0;
        return integral;
    }

    double integration::compute_dof_monomial_boundary_integral(
        const ElementData& element_data,
        int dof_idx,
        int monomial_idx,
        int order
    ){
        double integral = 0.0;

        if (operations::is_vertex_dof(dof_idx, element_data)){
            integral = compute_vertex_dof_monomial_boundary_integral(element_data, dof_idx, monomial_idx);
        } else if (operations::is_edge_dof(dof_idx, element_data, order)){
            integral = compute_edge_dof_monomial_boundary_integral(element_data, dof_idx, monomial_idx, order);
        }
        return integral;
    }
    
    // ============================================================================
    // POLYGONAL MOMENT COMPUTATION
    // ============================================================================

    double integration::compute_polygonal_moment_Ipq(
        const ElementData& element_data,
        int p,
        int q,
        const Eigen::Vector2d* centroid
    ) {
        // Validate input parameters
        if (p < 0 || q < 0) {
            throw std::invalid_argument("Powers p and q must be non-negative");
        }

        // Get or compute centroid
        Eigen::Vector2d centroid_coords;
        if (centroid != nullptr) {
            centroid_coords = *centroid;
        } else {
            centroid_coords = element_data.centroid;
        }

        double total_moment = 0.0;
        int N = element_data.n_vertices;

        // Loop over all edges
        for (int r = 0; r < N; ++r) {
            int r_next = (r + 1) % N;
            
            // Get edge vertices
            Eigen::Vector2d vertices_r = element_data.vertices.row(r);
            Eigen::Vector2d vertices_next = element_data.vertices.row(r_next);
            
            // Compute edge contribution
            double edge_contribution = compute_edge_moment_contribution(
                vertices_r, vertices_next, p, q, centroid_coords
            );
            
            total_moment += edge_contribution;
        }

        // Apply prefactor
        double prefactor = 1.0 / (p + q + 2);
        return prefactor * total_moment;
    }

    double integration::compute_scaled_polygonal_moment_Ipq(
        const ElementData& element_data,
        int p,
        int q,
        const Eigen::Vector2d* centroid
    ) {
        // First compute unscaled moment
        double unscaled = compute_polygonal_moment_Ipq(element_data, p, q, centroid);
        
        // Apply scaling: divide by h_e^(p+q)
        if (p + q == 0) {
            return unscaled;  // No scaling needed for area
        } else {
            double h_e = element_data.h_e;
            return unscaled / std::pow(h_e, p + q);
        }
    }

    double integration::compute_edge_moment_contribution(
        const Eigen::Vector2d& vertices_r,
        const Eigen::Vector2d& vertices_next,
        int p,
        int q,
        const Eigen::Vector2d& centroid
    ) {
        // Edge vectors
        double Delta_x_r = vertices_next(0) - vertices_r(0);
        double Delta_y_r = vertices_next(1) - vertices_r(1);
        
        // Shifted coordinates (relative to centroid, NOT scaled)
        double A_r = vertices_r(0) - centroid(0);
        double B_r = vertices_next(0) - centroid(0);
        double C_r = vertices_r(1) - centroid(1);
        double D_r = vertices_next(1) - centroid(1);
        
        // CRITICAL FIX: Pre-compute the fixed denominator
        int total_degree = p + q + 2;
        
        // First term: Delta_y_r * sum
        double first_term = 0.0;
        for (int i = 0; i <= p + 1; ++i) {
            for (int j = 0; j <= q; ++j) {
                // Binomial coefficients
                double binom_p1_i = operations::compute_binomial_coefficient(p + 1, i);
                double binom_q_j = operations::compute_binomial_coefficient(q, j);
                
                // Exponents
                int exp_A = p + 1 - i;
                int exp_B = i;
                int exp_C = q - j;
                int exp_D = j;
                
                // Only proceed if exponents are non-negative
                if (exp_A >= 0 && exp_C >= 0) {
                    // Compute powers (handle 0^0 = 1)
                    double A_power = (exp_A > 0) ? std::pow(A_r, exp_A) : 1.0;
                    double B_power = (exp_B > 0) ? std::pow(B_r, exp_B) : 1.0;
                    double C_power = (exp_C > 0) ? std::pow(C_r, exp_C) : 1.0;
                    double D_power = (exp_D > 0) ? std::pow(D_r, exp_D) : 1.0;
                    
                    // CORRECTED: Use the specific coefficient formula
                    int a = exp_A + exp_C;  // p + 1 - i + q - j
                    int b = exp_B + exp_D;  // i + j
                    
                    // The coefficient is (a! * b!) / (p+q+2)!
                    double coeff = operations::compute_beta_coefficient(a, b, total_degree);
                    
                    double term = binom_p1_i * binom_q_j * A_power * B_power * C_power * D_power * coeff;
                    first_term += term;
                }
            }
        }
        first_term *= Delta_y_r;
        
        // Second term: Delta_x_r * sum
        double second_term = 0.0;
        for (int i = 0; i <= p; ++i) {
            for (int j = 0; j <= q + 1; ++j) {
                // Binomial coefficients
                double binom_p_i = operations::compute_binomial_coefficient(p, i);
                double binom_q1_j = operations::compute_binomial_coefficient(q + 1, j);
                
                // Exponents
                int exp_A = p - i;
                int exp_B = i;
                int exp_C = q + 1 - j;
                int exp_D = j;
                
                // Only proceed if exponents are non-negative
                if (exp_A >= 0 && exp_C >= 0) {
                    // Compute powers (handle 0^0 = 1)
                    double A_power = (exp_A > 0) ? std::pow(A_r, exp_A) : 1.0;
                    double B_power = (exp_B > 0) ? std::pow(B_r, exp_B) : 1.0;
                    double C_power = (exp_C > 0) ? std::pow(C_r, exp_C) : 1.0;
                    double D_power = (exp_D > 0) ? std::pow(D_r, exp_D) : 1.0;
                    
                    // CORRECTED: Use the specific coefficient formula
                    int a = exp_A + exp_C;  // p - i + q + 1 - j
                    int b = exp_B + exp_D;  // i + j
                    
                    // The coefficient is (a! * b!) / (p+q+2)!
                    double coeff = operations::compute_beta_coefficient(a, b, total_degree);
                    
                    double term = binom_p_i * binom_q1_j * A_power * B_power * C_power * D_power * coeff;
                    second_term += term;
                }
            }
        }
        second_term *= Delta_x_r;
        
        // Return edge contribution
        return first_term - second_term;
    }

    // ============================================================================
    // DEBUG METHODS
    // ============================================================================
    void integration::print_dof_classification(const ElementData& element_data, int order) {
        operations::print_dof_classification(element_data, order);
    }
    
}

