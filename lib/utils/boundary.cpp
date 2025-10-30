#include "utils/boundary.hpp"

namespace utils {

    // ============================================================================
    // CONSTRUCTORS AND DESTRUCTOR
    // ============================================================================
    boundary::boundary() 
        : polynomial_order_(1), total_dofs_(0), is_initialized_(false), debug_mode_(false){}

    boundary::boundary(
        const Eigen::MatrixXd& nodes, 
        const Eigen::MatrixXi& elements, 
        int polynomial_order)
        : is_initialized_(false), debug_mode_(false){
            initialize(nodes, elements, polynomial_order);
        }

    
    // ============================================================================
    // INITIALIZATION AND SETUP
    // ============================================================================
    void boundary::initialize(
        const Eigen::MatrixXd& nodes,
        const Eigen::MatrixXi& elements,
        int polynomial_order
    ){
        nodes_ = nodes;
        elements_ = elements;
        polynomial_order_ = polynomial_order;
        
        // Clear existing data
        boundary_vertices_.clear();
        boundary_edges_.clear();
        boundary_conditions_.clear();
        boundary_segments_.clear();

        // Detect boundary features automatically
        automatically_detect_boundary();

        is_initialized_ = true;

        if (debug_mode_) {
            std::cout << "Boundary handler initialized:" << std::endl;
            std::cout << "  Nodes: " << nodes_.rows() << std::endl;
            std::cout << "  Elements: " << elements_.rows() << std::endl;
            std::cout << "  Polynomial order: " << polynomial_order_ << std::endl;
            std::cout << "  Boundary vertices: " << boundary_vertices_.size() << std::endl;
            std::cout << "  Boundary edges: " << boundary_edges_.size() << std::endl;
        }
    }

    void boundary::set_dof_mapping(
        const std::vector<std::vector<int>>& element_dof_mapping,
        const std::map<std::pair<int, int>, int>& edge_to_dof_map,
        int total_dofs
    ){
        element_dof_mapping_ = element_dof_mapping;
        edge_dof_map_ = edge_to_dof_map;
        total_dofs_ = total_dofs;

        // Re-classify boundary DOFs
        classify_boundary_dofs();

        if (debug_mode_) {
            std::cout << "DOF mapping set:" << std::endl;
            std::cout << "  Total DOFs: " << total_dofs_ << std::endl;
            std::cout << "  Dirichlet DOFs: " << dirichlet_dofs_.size() << std::endl;
            std::cout << "  Neumann DOFs: " << neumann_dofs_.size() << std::endl;
        }
    }

    void boundary::automatically_detect_boundary(){
        // Find edges that belong to only one element (boundary edges)
        std::map<std::pair<int, int>, int> edge_count;

        // Count how many elements each edge belongs to
        for (int elem = 0; elem < elements_.rows(); ++elem){
            // Count actual vertices for this element (skip -1 padding)
            int n_vertices = 0;
            for (int i = 0; i < elements_.cols(); ++i) {
                if (elements_(elem, i) != -1) {
                    n_vertices++;
                }
            }
            
            for (int i = 0; i < n_vertices; ++i){
                int v1 = elements_(elem, i);
                int v2 = elements_(elem, (i + 1) % n_vertices);
                
                // Safety check: skip invalid vertices
                if (v1 < 0 || v2 < 0 || v1 >= nodes_.rows() || v2 >= nodes_.rows()) {
                    if (debug_mode_) {
                        std::cerr << "WARNING: Invalid edge (" << v1 << "," << v2 << ") in element " << elem << std::endl;
                    }
                    continue;
                }

                // Canonical edge ordering (smaller vertex first)
                std::pair<int, int> edge = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);
                edge_count[edge]++;
            }
        }

        // Clear boundary edges and vertices objects
        boundary_edges_.clear();
        boundary_vertices_.clear();
        
        if (debug_mode_) {
            std::cout << "Edge count summary:" << std::endl;
            std::cout << "  Total unique edges: " << edge_count.size() << std::endl;
            int count_1 = 0, count_2 = 0, count_other = 0;
            for (const auto& [edge, count] : edge_count) {
                if (count == 1) count_1++;
                else if (count == 2) count_2++;
                else count_other++;
            }
            std::cout << "  Edges appearing once (boundary): " << count_1 << std::endl;
            std::cout << "  Edges appearing twice (interior): " << count_2 << std::endl;
            std::cout << "  Edges appearing >2 times (error!): " << count_other << std::endl;
        }

        // Extract boundary edges
        for (const auto& [edge, count] : edge_count){
            if (count == 1){
                boundary_edges_.insert(edge);
                boundary_vertices_.insert(edge.first);
                boundary_vertices_.insert(edge.second);
            }
        }

        // Build boundary segments
        build_boundary_segments();

        if (debug_mode_) {
            std::cout << "Boundary detection complete:" << std::endl;
            std::cout << "  Found " << boundary_vertices_.size() << " boundary vertices" << std::endl;
            std::cout << "  Found " << boundary_edges_.size() << " boundary edges" << std::endl;
        }
    }

    void boundary::set_boundary_manually(
        const std::set<int>& boundary_vertices,
        const std::set<std::pair<int, int>>& boundary_edges
    ){
        boundary_vertices_ = boundary_vertices;
        boundary_edges_ = boundary_edges;

        // Build boundary segments
        build_boundary_segments();

        if (debug_mode_) {
            std::cout << "Boundary set manually:" << std::endl;
            std::cout << "  Boundary vertices: " << boundary_vertices_.size() << std::endl;
            std::cout << "  Boundary edges: " << boundary_edges_.size() << std::endl;
        }
    }

    // ============================================================================
    // BOUNDARY CONDITION SPECIFICATION
    // ============================================================================

    void boundary::add_dirichlet(
        const std::function<double(const Eigen::Vector2d&, double)>& value_function,
        BoundaryRegion region,
        const std::string& name
    ){
        BoundaryCondition bc;
        bc.type = BCType::DIRICHLET;
        bc.value_function = value_function;
        bc.region = region;
        bc.name = name.empty() ? "Dirichlet_" + std::to_string(boundary_conditions_.size()) : name;

        // Get affected vertices and edges for this region
        bc.affected_vertices = get_vertices_for_region(region);
        bc.affected_edges = get_edges_for_region(region);
        boundary_conditions_.push_back(bc);

        // Re-classify boundary DOFs
        classify_boundary_dofs();

        if (debug_mode_) {
            std::cout << "Added Dirichlet condition '" << bc.name << "'" << std::endl;
            std::cout << "  Affected vertices: " << bc.affected_vertices.size() << std::endl;
            std::cout << "  Affected edges: " << bc.affected_edges.size() << std::endl;
        }
    }

    void boundary::add_neumann(
        const std::function<double(const Eigen::Vector2d&, double)>& flux_function,
        BoundaryRegion region,
        const std::string& name
    ){
        BoundaryCondition bc;
        bc.type = BCType::NEUMANN;
        bc.value_function = flux_function;
        bc.region = region;
        bc.name = name.empty() ? "Neumann_" + std::to_string(boundary_conditions_.size()) : name;

        // Get affected vertices and edges for this region
        bc.affected_vertices = get_vertices_for_region(region);
        bc.affected_edges = get_edges_for_region(region);
        boundary_conditions_.push_back(bc);

        // Re-classify boundary DOFs
        classify_boundary_dofs();

        if (debug_mode_) {
            std::cout << "Added Neumann condition '" << bc.name << "'" << std::endl;
            std::cout << "  Affected vertices: " << bc.affected_vertices.size() << std::endl;
            std::cout << "  Affected edges: " << bc.affected_edges.size() << std::endl;
        }
    }

    void boundary::add_robin(
        const std::function<double(const Eigen::Vector2d&, double)>& value_function,
        double alpha,
        double beta,
        BoundaryRegion region,
        const std::string& name
    ){
        BoundaryCondition bc;
        bc.type = BCType::ROBIN;
        bc.value_function = value_function;
        bc.alpha = alpha;
        bc.beta = beta;
        bc.region = region;
        bc.name = name.empty() ? "Robin_" + std::to_string(boundary_conditions_.size()) : name;

        // Get affected vertices and edges for this region
        bc.affected_vertices = get_vertices_for_region(region);
        bc.affected_edges = get_edges_for_region(region);
        boundary_conditions_.push_back(bc);

        // Re-classify boundary DOFs
        classify_boundary_dofs();

        if (debug_mode_) {
            std::cout << "Added Robin condition '" << bc.name << "'" << std::endl;
            std::cout << "  Affected vertices: " << bc.affected_vertices.size() << std::endl;
            std::cout << "  Affected edges: " << bc.affected_edges.size() << std::endl;
        }
    }

    void boundary::add_custom(
        BCType type,
        const std::function<double(const Eigen::Vector2d&, double)>& value_function,
        const std::set<int>& vertices,
        const std::set<std::pair<int, int>>& edges,
        BoundaryRegion region,
        const std::string& name
    ){
        BoundaryCondition bc;
        bc.type = type;
        bc.value_function = value_function;
        bc.affected_vertices = vertices;
        bc.affected_edges = edges;
        bc.region = region;
        bc.name = name.empty() ? "Custom_" + std::to_string(boundary_conditions_.size()) : name;

        // Add to boundary conditions
        boundary_conditions_.push_back(bc);

        // Re-classify boundary DOFs
        classify_boundary_dofs();

        if (debug_mode_) {
            std::cout << "Added custom condition '" << bc.name << "'" << std::endl;
            std::cout << "  Affected vertices: " << bc.affected_vertices.size() << std::endl;
            std::cout << "  Affected edges: " << bc.affected_edges.size() << std::endl;
        }
    }

    // ============================================================================
    // BOUNDARY CONDITION APPLICATION
    // ============================================================================

    void boundary::apply_dirichlet_conditions(
        Eigen::SparseMatrix<double>& K_h,
        Eigen::SparseMatrix<double>& M_h,
        Eigen::VectorXd& F_h,
        double time,
        bool preserve_mass_diagonal
    ){
        if (!is_initialized_){
            throw std::runtime_error("Boundary handler not initialized");
        }

        // Apply Dirichlet conditions using elimination method
        for (int dof : dirichlet_dofs_){
            // Find the boundary value for this DOF
            double boundary_value = 0.0;
            Eigen::Vector2d position = get_dof_position(dof);

            // Search through the boundary conditions to find the one that matches this DOF
            for (const auto& bc : boundary_conditions_){
                if (bc.type == BCType::DIRICHLET){
                    std::set<int> affected_dofs = get_dofs_for_region(bc.region);
                    if (affected_dofs.count(dof) > 0){
                        // Found the matching boundary condition
                        boundary_value = bc.value_function(position, time);
                        break;
                    }
                }
            }

            // Apply elimination
            apply_dirichlet_elimination(M_h, K_h, F_h, dof, boundary_value, preserve_mass_diagonal);
        }

        if (debug_mode_) {
            std::cout << "Applied Dirichlet conditions to " << dirichlet_dofs_.size() << " DOFs" << std::endl;
            std::cout << "  Matrix dimensions: " << K_h.rows() << "x" << K_h.cols() << std::endl;
            std::cout << "  Non-zero entries: " << K_h.nonZeros() << std::endl;
        }
    }

    void boundary::apply_neumann_conditions(
        Eigen::VectorXd& F_h,
        int polynomial_order,
        double time
    ){
        if (!is_initialized_) {
            std::cerr << "Error: Boundary handler not initialized!" << std::endl;
            return;
        }

        // Apply Neumann conditions by adding boundary integral contributions
        for (const auto& bc : boundary_conditions_){
            if (bc.type == BCType::NEUMANN){
                for (const auto& edge : bc.affected_edges){
                    auto contributions = compute_neumann_edge_contribution(edge, bc.value_function, polynomial_order, time);

                    // Add contributions to load vector
                    for (const auto& pair : contributions){
                        int global_dof = pair.first;
                        double contribution = pair.second;

                        F_h(global_dof) += contribution;
                    }
                }
            }
        }

        if (debug_mode_) {
            std::cout << "Applied Neumann conditions to " << neumann_dofs_.size() << " DOFs" << std::endl;
            std::cout << "  Load vector size: " << F_h.size() << std::endl;
            double total_contribution = F_h.sum();
            std::cout << "  Total flux contribution: " << total_contribution << std::endl;
        }
    }
        

    // ============================================================================
    // QUERY AND ACCESS FUNCTIONS
    // ============================================================================

    Eigen::Vector2d boundary::get_dof_position(int global_dof_idx) const{
        // For vertex DOFs - direct mapping
        if(global_dof_idx < nodes_.rows()) return nodes_.row(global_dof_idx);

        // For edge DOFs - find the edge and compute position using proper Gauss points
        for (const auto& edge_pair : edge_dof_map_){
            // Edge information
            const auto& edge = edge_pair.first;
            int edge_start_dof = edge_pair.second;
            int n_edges_dofs = polynomial_order_ - 1;

            if (global_dof_idx >= edge_start_dof && global_dof_idx < edge_start_dof + n_edges_dofs){
                int local_dof_in_edge = global_dof_idx - edge_start_dof;

                // Get endpoints of the edge
                Eigen::Vector2d v1 = nodes_.row(edge.first);
                Eigen::Vector2d v2 = nodes_.row(edge.second);

                // Use Gauss quadrature to compute position
                std::vector<double> gauss_points, gauss_weights;
                integration::get_gauss_quadrature_rule(2 * polynomial_order_, gauss_points, gauss_weights);

                // Make sure that there is enough points
                if (local_dof_in_edge < static_cast<int>(gauss_points.size())){
                    double xi = gauss_points[local_dof_in_edge];

                    // Map from reference interval [-1, 1] to physical edge
                    return 0.5 * (1 - xi) * v1 + 0.5 * (1 + xi) * v2;
                } else {
                    // Fallback: use uniform spacing if not enough Gauss points
                    double t = -1.0 + 2.0 * local_dof_in_edge / (n_edges_dofs - 1);
                    return 0.5 * (1 - t) * v1 + 0.5 * (1 + t) * v2;
                }
            }
        }

        // For debugging and visualization purposes
        try{
            return get_moment_dof_position(global_dof_idx, -1);
        } catch (const std::exception& e){
            return Eigen::Vector2d::Zero(); // Fallback
        }
    }

    Eigen::Vector2d boundary::get_moment_dof_position(
        int global_dof_idx,
        int element_idx
    ) const {
        // Get element vertices
        int n_vertices = elements_.cols();
        Eigen::Vector2d centroid(0.0, 0.0);

        for (int i = 0; i < n_vertices; ++i){
            int vertex_idx = elements_(element_idx, i);
            centroid += nodes_.row(vertex_idx);
        }

        centroid /= n_vertices;

        return centroid;
    }

    

    // ============================================================================
    // PRIVATE HELPER METHODS
    // ============================================================================

    void boundary::classify_boundary_dofs(){
        dirichlet_dofs_.clear();
        neumann_dofs_.clear();
        robin_dofs_.clear();

        for (const auto& bc : boundary_conditions_) {
            std::set<int> affected_dofs = get_dofs_for_region(bc.region);
            
            if (debug_mode_ && bc.type == BCType::DIRICHLET) {
                std::cout << "\n=== DIRICHLET DOF CLASSIFICATION ===" << std::endl;
                std::cout << "Boundary condition: " << bc.name << std::endl;
                std::cout << "Boundary vertices detected: " << boundary_vertices_.size() << std::endl;
                std::cout << "Boundary edges detected: " << boundary_edges_.size() << std::endl;
                std::cout << "Total DOFs to be clamped: " << affected_dofs.size() << std::endl;
                
                // Show vertex DOFs being clamped
                int vertex_count = 0;
                int edge_count = 0;
                for (int dof : affected_dofs) {
                    if (dof < nodes_.rows()) { // This is a vertex DOF
                        vertex_count++;
                        if (vertex_count <= 5) { // Show first 5
                            double x = nodes_(dof, 0);
                            double y = nodes_(dof, 1);
                            bool is_geometric_boundary = (std::abs(x) < 1e-10 || std::abs(x - 1.0) < 1e-10 || 
                                                         std::abs(y) < 1e-10 || std::abs(y - 1.0) < 1e-10);
                            std::cout << "  Vertex DOF " << dof << " at (" << x << ", " << y << ") - " 
                                      << (is_geometric_boundary ? "BOUNDARY" : "⚠️ INTERIOR!") << std::endl;
                        }
                    } else if (dof >= nodes_.rows()) { // This might be an edge DOF
                        edge_count++;
                        if (edge_count <= 5) { // Show first 5 edge DOFs
                            std::cout << "  Edge DOF " << dof << " (global index)" << std::endl;
                        }
                    }
                }
                if (vertex_count > 5) {
                    std::cout << "  ... and " << (vertex_count - 5) << " more vertex DOFs" << std::endl;
                }
                if (edge_count > 5) {
                    std::cout << "  ... and " << (edge_count - 5) << " more edge DOFs" << std::endl;
                }
                std::cout << "Total vertex DOFs clamped: " << vertex_count << " (expected: 8 for 3x3 grid)" << std::endl;
                std::cout << "Total edge DOFs clamped: " << edge_count << " (expected: 8 for boundary edges)" << std::endl;
                std::cout << "BREAKDOWN: " << vertex_count << " vertices + " << edge_count << " edges = " << (vertex_count + edge_count) << " total" << std::endl;
                
                // Show which boundary edges should have DOFs clamped
                std::cout << "\nBoundary edges that should have DOFs clamped:" << std::endl;
                int edge_debug_count = 0;
                for (const auto& edge : boundary_edges_) {
                    if (edge_debug_count < 5) {
                        std::cout << "  Edge (" << edge.first << "," << edge.second << ")";
                        // Try to find this edge's DOF index
                        if (edge_dof_map_.count(edge) > 0) {
                            int edge_dof = edge_dof_map_.at(edge);
                            std::cout << " → DOF " << edge_dof;
                            if (affected_dofs.count(edge_dof) > 0) {
                                std::cout << " ✅ CLAMPED";
                            } else {
                                std::cout << " ❌ NOT CLAMPED!";
                            }
                        } else {
                            std::cout << " → ❌ NOT FOUND IN MAP!";
                        }
                        std::cout << std::endl;
                    }
                    edge_debug_count++;
                }
                if (boundary_edges_.size() > 5) {
                    std::cout << "  ... and " << (boundary_edges_.size() - 5) << " more boundary edges" << std::endl;
                }
                
                // Check for incorrectly marked interior vertices
                for (int dof : affected_dofs) {
                    if (dof < nodes_.rows()) { // This is a vertex DOF
                        double x = nodes_(dof, 0);
                        double y = nodes_(dof, 1);
                        bool is_boundary = (std::abs(x) < 1e-10 || std::abs(x - 1.0) < 1e-10 || 
                                           std::abs(y) < 1e-10 || std::abs(y - 1.0) < 1e-10);
                        if (!is_boundary) {
                            std::cout << "🚨 ERROR: Interior vertex " << dof << " at (" << x << ", " << y 
                                      << ") incorrectly marked as Dirichlet!" << std::endl;
                        }
                    }
                }
                std::cout << "========================================" << std::endl;
            }
            
            switch (bc.type) {
                case BCType::DIRICHLET:
                    dirichlet_dofs_.insert(affected_dofs.begin(), affected_dofs.end());
                    break;
                case BCType::NEUMANN:
                    neumann_dofs_.insert(affected_dofs.begin(), affected_dofs.end());
                    break;
                case BCType::ROBIN:
                    robin_dofs_.insert(affected_dofs.begin(), affected_dofs.end());
                    break;
            }
        }
    }

    std::set<int> boundary::get_dofs_for_region(BoundaryRegion region) const{
        std::set<int> dofs;
    
        // Get vertices and edges for the region
        std::set<int> vertices = get_vertices_for_region(region);
        std::set<std::pair<int,int>> edges = get_edges_for_region(region);
        
        // Add vertex DOFs
        for (int vertex : vertices) {
            dofs.insert(vertex);  // For k=1, vertex DOF = vertex index
        }
        
        // Add edge DOFs (for k >= 2)
        if (polynomial_order_ >= 2) {
            for (const auto& edge : edges) {
                std::set<int> edge_dofs = get_edge_dofs(edge.first, edge.second);
                dofs.insert(edge_dofs.begin(), edge_dofs.end());
            }
        }
        
        return dofs;
    }

    std::set<int> boundary::get_vertices_for_region(BoundaryRegion region) const{
        if (region == BoundaryRegion::ENTIRE) {
            return boundary_vertices_;
        }
        
        std::set<int> vertices;
        for (int vertex : boundary_vertices_) {
            if (classify_vertex_region(vertex) == region) {
                vertices.insert(vertex);
            }
        }
        
        return vertices;
    }

    std::set<std::pair<int,int>> boundary::get_edges_for_region(BoundaryRegion region) const{
        if (region == BoundaryRegion::ENTIRE) {
            return boundary_edges_;
        }

        std::set<std::pair<int,int>> edges;
        for (const auto& edge : boundary_edges_) {
            if (classify_edge_region(edge.first, edge.second) == region) {
                edges.insert(edge);
            }
        }
        
        return edges;
    }


    boundary::BoundaryRegion boundary::classify_vertex_region(int vertex_idx) const{
        Eigen::Vector4d bbox = get_bounding_box();
        double x_min = bbox(0), x_max = bbox(1), y_min = bbox(2), y_max = bbox(3);
        double tol = 1e-10;
        
        Eigen::Vector2d vertex = nodes_.row(vertex_idx);
        
        if (std::abs(vertex.x() - x_min) < tol) return BoundaryRegion::LEFT;
        if (std::abs(vertex.x() - x_max) < tol) return BoundaryRegion::RIGHT;
        if (std::abs(vertex.y() - y_min) < tol) return BoundaryRegion::BOTTOM;
        if (std::abs(vertex.y() - y_max) < tol) return BoundaryRegion::TOP;
        
        return BoundaryRegion::CUSTOM;
    }

    boundary::BoundaryRegion boundary::classify_edge_region(int v1, int v2) const{
        // Classify based on both vertices
        BoundaryRegion r1 = classify_vertex_region(v1);
        BoundaryRegion r2 = classify_vertex_region(v2);
        
        // If both vertices are on the same boundary side, the edge is on that side
        if (r1 == r2 && r1 != BoundaryRegion::CUSTOM) {
            return r1;
        }
        
        // Check if edge is entirely on one boundary
        Eigen::Vector2d vertex1 = nodes_.row(v1);
        Eigen::Vector2d vertex2 = nodes_.row(v2);
        Eigen::Vector4d bbox = get_bounding_box();
        double tol = 1e-10;
        
        if (std::abs(vertex1.x() - bbox(0)) < tol && std::abs(vertex2.x() - bbox(0)) < tol) {
            return BoundaryRegion::LEFT;
        }
        if (std::abs(vertex1.x() - bbox(1)) < tol && std::abs(vertex2.x() - bbox(1)) < tol) {
            return BoundaryRegion::RIGHT;
        }
        if (std::abs(vertex1.y() - bbox(2)) < tol && std::abs(vertex2.y() - bbox(2)) < tol) {
            return BoundaryRegion::BOTTOM;
        }
        if (std::abs(vertex1.y() - bbox(3)) < tol && std::abs(vertex2.y() - bbox(3)) < tol) {
            return BoundaryRegion::TOP;
        }
        
        return BoundaryRegion::CUSTOM;
    }

    std::set<int> boundary::get_edge_dofs(int v1, int v2) const {
        std::set<int> edge_dofs;
        std::pair<int,int> edge = (v1 < v2) ? std::make_pair(v1, v2) : std::make_pair(v2, v1);
        
        if (edge_dof_map_.count(edge) > 0) {
            int start_dof = edge_dof_map_.at(edge);
            for (int i = 0; i < polynomial_order_ - 1; ++i) {
                edge_dofs.insert(start_dof + i);
            }
        }
        
        return edge_dofs;
    }

    void boundary::build_boundary_segments(){
        boundary_segments_.clear();

        for (const auto& edge : boundary_edges_) {
            BoundarySegment segment;

            segment.edge = edge;
            segment.start_point = nodes_.row(edge.first);
            segment.end_point = nodes_.row(edge.second);
            segment.length = (segment.end_point - segment.start_point).norm();
            segment.outward_normal = compute_outward_normal_vector(edge.first, edge.second);
            segment.region = classify_edge_region(edge.first, edge.second);
            
            boundary_segments_.push_back(segment);
            
        }
    }

    // ============================================================================
    // PRIVATE BC MANIPULATION METHODS
    // ============================================================================

    void boundary::apply_dirichlet_elimination(
        Eigen::SparseMatrix<double>& M_h,
        Eigen::SparseMatrix<double>& K_h,
        Eigen::VectorXd& F_h,
        int dof_idx,
        double value,
        bool preserve_mass_diagonal
    ){
        // Set diagonal entries: K(i,i) = 1, M(i,i) = 0 for time-dependent problems
        K_h.coeffRef(dof_idx, dof_idx) = 1.0;
        if (!preserve_mass_diagonal) {
            M_h.coeffRef(dof_idx, dof_idx) = 0.0;  // FIXED: Set to 0 for time-dependent problems
        }

        // std::cout << "DEBUG: M_h diagonal: " << M_h.coeffRef(dof_idx, dof_idx) << std::endl;

        if (K_h.outerSize() != M_h.outerSize() || K_h.innerSize() != M_h.innerSize()){
            throw std::runtime_error("Matrix dimensions do not match");
        }

        // Zero out row and column (preserve sparsity pattern)
        for (int i = 0; i < K_h.outerSize(); ++i){
            for (Eigen::SparseMatrix<double>::InnerIterator it(K_h, i); it; ++it){
                if (it.row() == dof_idx && it.col() != dof_idx) it.valueRef() = 0.0;
                if (it.row() != dof_idx && it.col() == dof_idx) it.valueRef() = 0.0;
            }

            for (Eigen::SparseMatrix<double>::InnerIterator it(M_h, i); it; ++it){
                if (it.row() == dof_idx && it.col() != dof_idx) it.valueRef() = 0.0;
                if (it.row() != dof_idx && it.col() == dof_idx) it.valueRef() = 0.0;
            }
        }

        // Modify right-hand side
        if (!preserve_mass_diagonal) { 
            F_h(dof_idx) = value;
        } else {
            F_h(dof_idx) = value * M_h.coeff(dof_idx, dof_idx);
        }
    }

    std::map<int, double> boundary::compute_neumann_edge_contribution(
        const std::pair<int, int>& edge,
        const std::function<double(const Eigen::Vector2d&, double)>& flux_function,
        int polynomial_order,
        double time
    ){
        std::map<int, double> contributions_integrals;

        // Get edge endpoints
        Eigen::Vector2d v1 = nodes_.row(edge.first);
        Eigen::Vector2d v2 = nodes_.row(edge.second);

        // Get edge length
        double length = (v2 - v1).norm();

        // Get DOFs on this edge
        std::set<int> edge_dofs = get_edge_dofs(edge.first, edge.second);
        edge_dofs.insert(edge.first);
        edge_dofs.insert(edge.second);

        // Get Gauss quadrature rule
        std::vector<double> gauss_points, gauss_weights;
        integration::get_gauss_quadrature_rule(polynomial_order + 2, gauss_points, gauss_weights);

        // Compute contribution for each DOF
        for (int dof : edge_dofs){
            double integral = 0.0;

            for (size_t q = 0; q < gauss_points.size(); ++q){
                double xi = gauss_points[q];
                Eigen::Vector2d point = 0.5 * (1 - xi) * v1 + 0.5 * (1 + xi) * v2;

                double flux_value = flux_function(point, time);
                double basis_value = evaluate_boundary_basis_function(dof, xi, edge, polynomial_order);
                
                integral += gauss_weights[q] * flux_value * basis_value * length * 0.5;
            }

            contributions_integrals[dof] = integral;
        }

        return contributions_integrals;
    }

    double boundary::evaluate_boundary_basis_function(
        int global_idx,
        double xi,
        const std::pair<int, int>& edge,
        int polynomial_order
    ) const{
        // First vertex of the edge
        if (global_idx == edge.first) return 0.5 * (1 - xi); // L_0(xi) = (1-xi)/2
        if (global_idx == edge.second) return 0.5 * (1 + xi); // L_1(xi) = (1+xi)/2

        if (polynomial_order_ >= 2){
            // Check if this DOF belongs to this edge
            std::set<int> edge_dofs = get_edge_dofs(edge.first, edge.second);

            if (edge_dofs.count(global_idx) > 0){
                // Find which local edge DOF this is (0, 1, 2, ... for order-1 DOFs per edge)
                int edge_start_dof = edge_dof_map_.at(edge);
                int local_dof_idx = global_idx - edge_start_dof;

                int legendre_degree = local_dof_idx + 2; // L_2, L_3, L_4, ... for edge DOFs

                // Evaluate Legendre polynomial
                double legendre_value = operations::evaluate_legendre_polynomial(legendre_degree, xi);

                // Apply VEM scaling factor for orthonormality
                double scaling_factor = std::sqrt((2 * legendre_degree + 1) / 2.0);
                
                return scaling_factor * legendre_value;
            }
        }
        // DOF not associated with this edge
        return 0.0;
    }
        
}