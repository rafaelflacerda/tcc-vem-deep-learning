#include "mesh/uniform.hpp"

namespace mesh {
    void uniform::create_square_nxn_mesh(
        Eigen::MatrixXd& nodes, 
        Eigen::MatrixXi& elements, 
        int n
    ){
        // Number of nodes in each direction
        int nodes_per_direction = n + 1;
        int total_nodes = nodes_per_direction * nodes_per_direction;
        int total_elements = n * n;
        
        // Element size
        double h = 1.0 / static_cast<double>(n);
        
        // Resize matrices
        nodes.resize(total_nodes, 2);
        elements.resize(total_elements, 4);
        
        // Generate nodes
        int node_idx = 0;
        for (int j = 0; j < nodes_per_direction; ++j) {      // y direction (rows)
            for (int i = 0; i < nodes_per_direction; ++i) {  // x direction (cols)
                nodes(node_idx, 0) = i * h;  // x coordinate
                nodes(node_idx, 1) = j * h;  // y coordinate
                node_idx++;
            }
        }
        
        // Generate elements
        int elem_idx = 0;
        for (int j = 0; j < n; ++j) {      // element rows
            for (int i = 0; i < n; ++i) {  // element cols
                // Bottom-left node of current element
                int bottom_left = j * nodes_per_direction + i;
                int bottom_right = bottom_left + 1;
                int top_left = bottom_left + nodes_per_direction;
                int top_right = top_left + 1;
                
                // Define element connectivity (counter-clockwise)
                elements(elem_idx, 0) = bottom_left;
                elements(elem_idx, 1) = bottom_right;
                elements(elem_idx, 2) = top_right;
                elements(elem_idx, 3) = top_left;
                elem_idx++;
            }
        }

        if (debug_) {
            std::cout << n << "x" << n << " Test mesh created:" << std::endl;
            std::cout << "  Nodes: " << nodes.rows() << " (" << nodes_per_direction 
                      << "x" << nodes_per_direction << " grid)" << std::endl;
            std::cout << "  Elements: " << elements.rows() << std::endl;
            std::cout << "  Element size h = " << std::fixed << std::setprecision(6) << h << std::endl;
        }

        diameter_ = h;
    }
}