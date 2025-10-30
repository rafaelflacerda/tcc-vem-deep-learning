/**
 * @file linearElastic2d.hpp
 * @brief Defines the linear elastic 2D solver class
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_LINEARELASTIC2D_HPP
#define POLIVEM_LINEARELASTIC2D_HPP

#include <iostream>
#include <cmath>

#include <Eigen/Dense>

#include "utils/operations.hpp"

namespace solver {
    /**
     * @class linearElastic2d
     * @brief Solver for two-dimensional linear elastic problems using the Virtual Element Method
     * 
     * This class implements the Virtual Element Method (VEM) for solving
     * two-dimensional linear elastic problems, including plane stress and plane strain.
     */
    class linearElastic2d{
        public:
            // VEM order
            int order;

            // displacement restrictions
            Eigen::MatrixXi supp;

            // nodes (contains the coordinates)
            Eigen::MatrixXd nodes;

            // elements (contains the indices)
            Eigen::MatrixXi elements;

            // distributed load node indices (edge)
            Eigen::MatrixXi load;

            // set displacement restrictions
            void setSupp(Eigen::MatrixXi dirichlet_bc);

            // set load indices
            void setLoad(Eigen::MatrixXi load_indices);

            // Np matrix (constant polynomial space basis)
            Eigen::MatrixXd buildNp();

            // NE matrix (normal vector matrice)
            Eigen::MatrixXd buildNE(Eigen::MatrixXd coord);

            // Nv matrix (inpterpolation operator)
            Eigen::MatrixXd buildNv(int startIndex, int endIndex);

            // build G matrix
            Eigen::MatrixXd buildG(Eigen::MatrixXd coords);

            // build B matrix
            Eigen::MatrixXd buildB(Eigen::MatrixXd coords);

            // build D matrix
            Eigen::MatrixXd buildD(Eigen::MatrixXd coords);

            // build the consistency part of the stiffness matrix
            Eigen::MatrixXd buildConsistency(Eigen::MatrixXd coords, Eigen::MatrixXd C);

            // build the stabilization term of the stiffness matrix
            Eigen::MatrixXd buildStability(Eigen::MatrixXd coords, Eigen::MatrixXd Kc);

            // build the local stiffness matrix
            Eigen::MatrixXd buildLocalK(Eigen::MatrixXd coords, Eigen::MatrixXd C);

            // build the global stiffness matrix
            Eigen::MatrixXd buildGlobalK(Eigen::MatrixXd C);

            // apply Dirichlet Boundary conditions
            Eigen::MatrixXd applyDBC(Eigen::MatrixXd K);

            // apply Neumann Boundary conditions
            Eigen::VectorXd applyNBC(double qx, double qy);

            
            /**
             * @brief Constructor
             * 
             * @param nodes_coordinates Matrix of node coordinates
             * @param elements_indices Matrix of element connectivity
             * @param accuracy_order Method order (polynomial degree)
             */
            linearElastic2d(Eigen::MatrixXd nodes_coordinates, Eigen::MatrixXi elements_indices, int accuracy_order){
                nodes = nodes_coordinates;
                elements = elements_indices;
                order = accuracy_order;
            }

    };
}

#endif