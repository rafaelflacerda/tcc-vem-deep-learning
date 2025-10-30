/**
 * @file beam.hpp
 * @brief Defines the beam class for 1D beam element mesh generation
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#ifndef POLIVEM_MESH_BEAM_HPP
#define POLIVEM_MESH_BEAM_HPP

#include <iostream>

#include <Eigen/Dense>


namespace mesh {

/**
 * @class beam
 * @brief Class for generating and manipulating 1D beam meshes
 * 
 * This class provides functionality to create and manipulate
 * one-dimensional beam element meshes for structural analysis.
 */
class beam{
    public:
        /**
         * @brief Node coordinates matrix
         * 
         * Matrix containing the coordinates of all nodes in the mesh.
         * Each row represents a node, with columns for x and y coordinates.
         */
        Eigen::MatrixXd nodes;

        /**
         * @brief Element connectivity matrix
         * 
         * Matrix defining the connectivity between nodes to form elements.
         * Each row represents an element, with columns containing the indices
         * of the nodes that form the element.
         */
        Eigen::MatrixXi elements;

        /**
         * @brief Default constructor
         */
        beam() = default;

        /**
         * @brief Discretize a horizontal bar/beam
         * 
         * Creates a uniform mesh for a horizontal beam with the specified length
         * and number of elements.
         * 
         * @param bar_length Length of the beam
         * @param num_elements Number of elements in the mesh
         */
        void horizontalBarDisc(double bar_length, int num_elements);
};
}
#endif