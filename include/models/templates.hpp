/**
 * @file templates.hpp
 * @brief Defines templates used throughout the library
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#ifndef POLIVEM_MODELS_TEMPLATES_HPP
#define POLIVEM_MODELS_TEMPLATES_HPP

#include <Eigen/Dense>
#include <vector>

// Element-specific data structures
    struct ElementData {
        Eigen::MatrixXd vertices; // Coordinates of the element vertices
        Eigen::Vector2d centroid; // Centroid of the element
        double h_e; // Element polygonal diameter
        double area; // Element area
        int n_vertices; // Number of vertices
        int n_dofs_local; // Number of DOFs per vertex

        // Monomial basis powers
        std::vector<std::pair<int, int>> monomial_powers;

        // Projection matrices
        Eigen::MatrixXd P_nabla; // Energy projection matrix
        Eigen::MatrixXd P_0; // L^2 projection matrix

        // Local polynomial matrices
        Eigen::MatrixXd K_poly; // Polynomial stiffness matrix
        Eigen::MatrixXd M_poly; // Polynomial mass matrix

        // Local system matrices
        Eigen::MatrixXd K_local; // Local stiffness matrix
        Eigen::MatrixXd M_local; // Local mass matrix

        // Local load vector
        Eigen::VectorXd F_local; // Local load vector
    };

#endif