#include "utils/errors.hpp"

namespace utils {
    // ============================================================================
    // ERROR COMPUTATION
    // ============================================================================

    double errors::compute_true_h1_error_with_stored_matrices(
        const solver::parabolic& vem_solver,
        const Eigen::VectorXd& U_final,
        double final_time,
        const std::function<Eigen::Vector2d(double, double, double)>& exact_gradient
    ) {
        if (!vem_solver.has_projection_matrices()) return -1.0;
        const auto& elements = vem_solver.elements;
        double h1_error_sq = 0.0;
        double h1_exact_sq = 0.0;

        // Quadrature rule based on VEM order 
        std::vector<std::array<double,3>> tri_pts;
        std::vector<double> tri_w;
        
        if (vem_solver.order >= 2) {
            // 6-point Dunavant rule (exact for degree 4 polynomials) for k>=2
            static const double w[6] = { 0.109951743655322, 0.109951743655322,
                                         0.109951743655322, 0.223381589678011,
                                         0.223381589678011, 0.223381589678011 };
            static const double a[6][2] = {
                {0.816847572980459, 0.091576213509771},
                {0.091576213509771, 0.816847572980459},
                {0.091576213509771, 0.091576213509771},
                {0.108103018168070, 0.445948490915965},
                {0.445948490915965, 0.108103018168070},
                {0.445948490915965, 0.445948490915965} };
            
            tri_pts.resize(6);
            tri_w.resize(6);
            for (int i = 0; i < 6; ++i) {
                double bary_c = 1.0 - a[i][0] - a[i][1];
                tri_pts[i] = {a[i][0], a[i][1], bary_c};
                tri_w[i] = w[i];
            }
        } else {
            // 3-point rule (exact for degree 2 polynomials) for k=1
            tri_pts = {
                {1.0/6.0, 1.0/6.0, 2.0/3.0},
                {2.0/3.0, 1.0/6.0, 1.0/6.0},
                {1.0/6.0, 2.0/3.0, 1.0/6.0}
            };
            tri_w = {1.0/3.0, 1.0/3.0, 1.0/3.0};
        }
        // TODO: Implement other quadrature rules for k > 2

        for(int e=0;e<elements.rows();++e){
            const ElementData& E = vem_solver.get_element_data(e);
            
            // For k=1, use P_nabla; for k>=2, use P_0
            const Eigen::MatrixXd* P_matrix;
            if (vem_solver.order == 1) {
                P_matrix = &vem_solver.get_element_P_nabla_matrix(e);
            } else {
                P_matrix = &vem_solver.get_element_P0_matrix(e);
            }

            // local DOF values
            Eigen::VectorXd u_loc(E.n_dofs_local);
            for(int i=0;i<E.n_dofs_local;++i) u_loc(i)=U_final(vem_solver.element_dof_map[e][i]);
            Eigen::VectorXd coeffs = (*P_matrix) * u_loc;

            int nv = E.n_vertices;
            Eigen::Vector2d c = E.centroid;
            for(int v=0; v<nv; ++v){
                Eigen::Vector2d v1 = E.vertices.row(v);
                Eigen::Vector2d v2 = E.vertices.row((v+1)%nv);
                Eigen::Vector2d v0 = c;
                double areaT = utils::operations::compute_triangle_area(v0,v1,v2);
                for(size_t q=0;q<tri_pts.size();++q){
                    double a=tri_pts[q][0], b=tri_pts[q][1], g=tri_pts[q][2];
                    Eigen::Vector2d p = a*v0 + b*v1 + g*v2;
                    // exact grad
                    Eigen::Vector2d grad_exact = exact_gradient(final_time,p(0),p(1));
                    // numerical grad
                    Eigen::Vector2d grad_num(0,0);
                    for(int k=0;k<coeffs.size();++k){
                        grad_num += coeffs(k)*utils::operations::evaluate_monomial_gradient(k,p,E);
                    }
                    Eigen::Vector2d diff = grad_exact - grad_num;
                    double w = tri_w[q]*areaT;
                    h1_error_sq += diff.squaredNorm()*w;
                    h1_exact_sq += grad_exact.squaredNorm()*w;
                }
            }
        }
        if(h1_exact_sq>1e-12) return std::sqrt(h1_error_sq/h1_exact_sq);

        return std::sqrt(h1_error_sq);
    }

    double errors::compute_l2_error_parabolic(
        const solver::parabolic& vem_solver,
        const Eigen::VectorXd& U_final,
        double final_time,
        const std::function<double(double, double, double)>& exact_solution
    ){
        // Step 1: Compute VEM interpolant u_I of exact solution
        Eigen::VectorXd u_I = compute_vem_interpolant(vem_solver, final_time, exact_solution);

        // Step 2: Compute difference vector u_h - u_I
        Eigen::VectorXd error_vector = U_final - u_I;

        // Step 3: Get global mass matrix M_h (discrete VEM bilinear form m_h(·,·))
        Eigen::SparseMatrix<double> M_h = vem_solver.get_global_mass_matrix();

        // Step 4: Compute discrete norms using VEM mass matrix
        double numerator = error_vector.transpose() * M_h * error_vector;    // m_h(u_h - u_I, u_h - u_I)
        double denominator = U_final.transpose() * M_h * U_final;            // m_h(u_h, u_h)

        // Step 5: Return relative discrete VEM error
        if (denominator > 1e-12) {
            double error = std::sqrt(numerator / denominator);

            if (vem_solver.get_debug_mode()) {
                std::cout << "  Numerator m_h(u_h - u_I, u_h - u_I) = " << numerator << std::endl;
                std::cout << "  Denominator m_h(u_h, u_h) = " << denominator << std::endl;
                std::cout << "  Error E²_{h,τ} = " << error << std::endl;
            }

            return error;
        } else {
            if (vem_solver.get_debug_mode()) {
                std::cout << "  Warning: Near-zero denominator, returning absolute error" << std::endl;
            }

            return std::sqrt(numerator);
        }
    }

    // ============================================================================
    // VEM INTERPOLANT
    // ============================================================================

    Eigen::VectorXd errors::compute_vem_interpolant(
        const solver::parabolic& vem_solver,
        double time,
        const std::function<double(double, double, double)>& exact_solution
    ){
        const Eigen::MatrixXd& nodes = vem_solver.nodes;
        const Eigen::MatrixXi& elements = vem_solver.elements;

        // Initialize interpolant DOF vector
        Eigen::VectorXd u_I = Eigen::VectorXd::Zero(vem_solver.n_dofs);

        // Process each element to compute local DOF functionals
        for (int elem = 0; elem < elements.rows(); ++elem) {
            const auto& dof_map = vem_solver.element_dof_map[elem];
            const ElementData& element_data = vem_solver.get_element_data(elem);
            int n_vertices = element_data.n_vertices;

            // 1. (D1) VERTEX DOFs: χ_i(u) = u(V_i)
            for (int v = 0; v < n_vertices; ++v) {
                Eigen::Vector2d vertex = element_data.vertices.row(v);
                double exact_value = exact_solution(time, vertex.x(), vertex.y());
                
                int global_dof = dof_map[v];
                u_I(global_dof) = exact_value;  // DOF functional: χ_i(u) = u(V_i)
            }
            
            if (vem_solver.order >= 2) {
                for (int edge_idx = 0; edge_idx < n_vertices; ++edge_idx)  {
                    Eigen::Vector2d v1 = element_data.vertices.row(edge_idx);
                    Eigen::Vector2d v2 = element_data.vertices.row((edge_idx + 1) % n_vertices);
                    for (int point_idx = 0; point_idx < (vem_solver.order - 1); ++point_idx) {
                        int local_dof = n_vertices + edge_idx * (vem_solver.order - 1) + point_idx;
                        if (local_dof < dof_map.size()) {
                            // For general k: equispaced points t = (point_idx + 1.0) / vem_solver.order
                            // For k=2: t=0.5 (midpoint)
                            double t = (point_idx + 1.0) / vem_solver.order;
                            Eigen::Vector2d point = (1.0 - t) * v1 + t * v2;
                            double value = exact_solution(time, point.x(), point.y());
                            int global_dof = dof_map[local_dof];
                            u_I(global_dof) = value;
                        }
                    }
                }

                // 3. (D3) INTERIOR DOFs (k ≥ 2): Moments ∫_K u p_α dx for p_α ∈ P_{k-2}(K)
                int interior_dof_count = (vem_solver.order - 1) * vem_solver.order / 2;
                int interior_start = n_vertices + n_vertices * (vem_solver.order - 1);

                // Generate monomial basis for P_{k-2}(K): {1, x, y, x², xy, y², ...}
                std::vector<std::pair<int, int>> interior_monomials = generate_monomial_powers(vem_solver.order - 2);

                for (int moment_idx = 0; moment_idx < interior_dof_count; ++moment_idx){
                    int local_dof = interior_start + moment_idx;
                    if (local_dof < dof_map.size()){
                        // (D3) DOF functional: χ_α(u) = ∫_K u(x,y) * p_α(x,y) dx dy
                        // where p_α is the scaled monomial corresponding to moment_idx
                        int alpha_x = interior_monomials[moment_idx].first;
                        int alpha_y = interior_monomials[moment_idx].second;

                        double interior_moment = compute_interior_moment_exact(element_data, time, alpha_x, alpha_y, exact_solution);
                        double area = utils::operations::calcArea(element_data.vertices);
                        
                        int global_dof = dof_map[local_dof];
                        u_I(global_dof) = interior_moment;
                    }
                }
            }
        }

        return u_I;
    }

    // ============================================================================
    // HELPER METHODS
    // ============================================================================

    std::vector<std::pair<int, int>> errors::generate_monomial_powers(int max_degree){
        std::vector<std::pair<int, int>> powers;

        // Add monomials x^i * y^j where i + j ≤ max_degree
        for (int total_degree = 0; total_degree <= max_degree; ++total_degree) {
            for (int i = 0; i <= total_degree; ++i) {
                int j = total_degree - i;
                powers.push_back({i, j});
            }
        }
        return powers;
    }

    double errors::compute_interior_moment_exact(
        const ElementData& element_data, 
        double time, 
        int alpha_x, 
        int alpha_y,
        const std::function<double(double, double, double)>& exact_solution
    ){
        // Use high-order triangulation-based quadrature
        // For transcendental functions like sin(πx)sin(πy), need sufficient precision

        // 7-point Dunavant rule (exact for degree 5 polynomials) - sufficient for most cases
        static const double w[7] = { 0.062969590272413, 0.062969590272413, 0.062969590272413,
            0.066197076394253, 0.066197076394253, 0.066197076394253,
            0.1125 };

        static const double a[7][2] = {
            {0.797426985353087, 0.101286507323456},
            {0.101286507323456, 0.797426985353087}, 
            {0.101286507323456, 0.101286507323456},
            {0.470142064105115, 0.059715871789770},
            {0.059715871789770, 0.470142064105115},
            {0.470142064105115, 0.470142064105115},
            {0.333333333333333, 0.333333333333333} };

        std::vector<std::array<double, 3>> triangle_points(7);
        std::vector<double> triangle_weights(7);
        for (int i = 0; i < 7; ++i) {
            double bary_c = 1.0 - a[i][0] - a[i][1];
            triangle_points[i] = {a[i][0], a[i][1], bary_c};
            triangle_weights[i] = w[i];
        }
        
        double total_integral = 0.0;

        // Split polygon into triangles from centroid
        int n_vertices = element_data.n_vertices;
        Eigen::Vector2d centroid = element_data.centroid;
        double h_K = element_data.h_e;  // Element diameter for scaling

        for (int v = 0; v < n_vertices; ++v) {
            Eigen::Vector2d v1 = element_data.vertices.row(v);
            Eigen::Vector2d v2 = element_data.vertices.row((v + 1) % n_vertices);
            Eigen::Vector2d v0 = centroid;
            
            double triangle_area = utils::operations::compute_triangle_area(v0, v1, v2);
            
            // Apply quadrature over this triangle
            for (size_t q = 0; q < triangle_points.size(); ++q) {
                double xi1 = triangle_points[q][0];
                double xi2 = triangle_points[q][1]; 
                double xi3 = triangle_points[q][2];
                
                // Physical point in triangle
                Eigen::Vector2d physical_point = xi1 * v0 + xi2 * v1 + xi3 * v2;
                
                // Evaluate exact solution u(t,x,y)
                double u_value = exact_solution(time, physical_point.x(), physical_point.y());
                
                // Evaluate scaled monomial m_α(x,y) = ((x-x_c)/h_K)^α_x * ((y-y_c)/h_K)^α_y
                double scaled_x = (physical_point.x() - centroid.x()) / h_K;
                double scaled_y = (physical_point.y() - centroid.y()) / h_K;
                double monomial_value = std::pow(scaled_x, alpha_x) * std::pow(scaled_y, alpha_y);
                
                // Add quadrature contribution
                total_integral += triangle_weights[q] * u_value * monomial_value * triangle_area;
            }
        }

        return total_integral;
    }

    double errors::compute_edge_moment_orthogonal(
        const Eigen::Vector2d& v1, 
        const Eigen::Vector2d& v2, 
        double edge_length, 
        double time, 
        int j,
        const std::function<double(double, double, double)>& exact_solution
    ){
        int quad_order = 4;
        std::vector<double> gauss_points, gauss_weights;
        get_gauss_legendre_01(quad_order, gauss_points, gauss_weights);
        
        double edge_integral = 0.0;
        for (size_t q = 0; q < gauss_points.size(); ++q) {
            double t = gauss_points[q];
            Eigen::Vector2d edge_point = (1.0 - t) * v1 + t * v2;
            double u_value = exact_solution(time, edge_point.x(), edge_point.y());
            double xi = 2.0 * t - 1.0;
            double legendre_value = utils::operations::evaluate_legendre_polynomial(j, xi);
            edge_integral += gauss_weights[q] * u_value * legendre_value;
        }

        double orthonormal_scale = std::sqrt((2.0 * j + 1.0) / 2.0);
        double solver_scaling = (edge_length / 2.0) * orthonormal_scale;
        double normalized_moment = edge_integral * solver_scaling;

        std::cout << "      Raw integral = " << edge_integral << std::endl;
        std::cout << "      Edge length = " << edge_length << std::endl;
        std::cout << "      Orthonormal scale = " << orthonormal_scale << std::endl;
        std::cout << "      Normalized moment = " << normalized_moment << std::endl;
        
        return normalized_moment;
    }

    void errors::get_gauss_legendre_01(
        int n, 
        std::vector<double>& points, 
        std::vector<double>& weights
    ){
        points.clear();
        weights.clear();
        
        // Standard Gauss-Legendre points and weights on [-1,1]
        // (These would typically come from a numerical library or precomputed tables)
        
        if (n == 3) {
            // 3-point rule (exact for polynomials up to degree 5)
            std::vector<double> xi = {-0.7745966692414834, 0.0, 0.7745966692414834};
            std::vector<double> w = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
            
            // Transform from [-1,1] to [0,1]
            for (int i = 0; i < 3; ++i) {
                points.push_back((xi[i] + 1.0) / 2.0);      // t = (ξ + 1)/2
                weights.push_back(w[i] / 2.0);              // Jacobian = 1/2
            }
        }
        else if (n == 4) {
            // 4-point rule (exact for polynomials up to degree 7)
            std::vector<double> xi = {-0.8611363115940526, -0.3399810435848563, 
                                    0.3399810435848563, 0.8611363115940526};
            std::vector<double> w = {0.3478548451374538, 0.6521451548625461,
                                    0.6521451548625461, 0.3478548451374538};
            
            for (int i = 0; i < 4; ++i) {
                points.push_back((xi[i] + 1.0) / 2.0);
                weights.push_back(w[i] / 2.0);
            }
        }
        else if (n == 5) {
            // 5-point rule (exact for polynomials up to degree 9)
            std::vector<double> xi = {-0.9061798459386640, -0.5384693101056831, 0.0,
                                    0.5384693101056831, 0.9061798459386640};
            std::vector<double> w = {0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                                    0.4786286704993665, 0.2369268850561891};
            
            for (int i = 0; i < 5; ++i) {
                points.push_back((xi[i] + 1.0) / 2.0);
                weights.push_back(w[i] / 2.0);
            }
        }
        else {
            // Fallback: 3-point rule
            get_gauss_legendre_01(3, points, weights);
        }
    }
}