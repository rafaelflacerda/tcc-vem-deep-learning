/**
 * @file beam1d.hpp
 * @brief Defines the beam1d solver class for 1D beam analysis
 * @author Paulo Akira
 * @date YYYY-MM-DD
 */

#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#ifndef POLIVEM_SOLVER_BEAM1D_HPP
#define POLIVEM_SOLVER_BEAM1D_HPP

#include <iostream>
#include <cmath>

#include <Eigen/Dense>
#include "utils/operations.hpp"
#include "models/enums.hpp"

namespace solver {

/**
 * @class beam1d
 * @brief Solver for one-dimensional beam problems using the Virtual Element Method
 * 
 * This class implements the Virtual Element Method (VEM) for solving
 * one-dimensional beam problems, including bending and deformation analysis.
 */
class beam1d {
    public:
        // model order
        int order;

        // inertial moment
        double I;

        // cross section area
        double area;

        // nodes (contains the coordinates)
        Eigen::MatrixXd nodes;

        // elements (contains the indices)
        Eigen::MatrixXi elements;

        // displacement restrictions
        Eigen::MatrixXi supp;

        // distributed load values
        Eigen::VectorXd q;

        // distrbuted load node indices
        Eigen::MatrixXi load;

        // set cross-section inertia moment
        void setInertiaMoment(double inertia_moment);

        // set cross-section area
        void setArea(double cross_sction_area);

        // set displacement restrictions
        void setSupp(Eigen::MatrixXi dirichlet_bc);

        // set load values and indices
        void setDistributedLoad(Eigen::VectorXd load_values, Eigen::MatrixXi load_indices);

        // build local stiffness matrix
        Eigen::MatrixXd buildLocalK(Eigen::MatrixXd coord, double E);

        // build local stiffness matrix considering portic structure
        Eigen::MatrixXd buildLocalKPortic(Eigen::MatrixXd coord, double E);

        // build rotation operator
        Eigen::MatrixXd buildRotationOperator(Eigen::MatrixXd coord);

        // build local mass matrix
        Eigen::MatrixXd buildLocalM(Eigen::MatrixXd coord, double rho);

        // build global stiffness matrix
        Eigen::MatrixXd buildGlobalK(double E, BeamSolverType type=BeamSolverType::Beam);

        // build global mass matrix
        Eigen::MatrixXd buildGlobalM(double rho);

        // build static condesation matrices
        Eigen::MatrixXd buildStaticCondensation(Eigen::MatrixXd K, std::string sc_type, BeamSolverType type=BeamSolverType::Beam);

        // build local distributed load vector
        Eigen::VectorXd buildLocalDistributedLoad(Eigen::MatrixXd coord);

        // build local distributed load vector considering portic structure
        Eigen::VectorXd buildLocalDistributedLoadPortic(Eigen::MatrixXd coord);

        // build global distributed load vector
        Eigen::VectorXd buildGlobalDistributedLoad(BeamSolverType type=BeamSolverType::Beam);

        // build static condensation distributed load vector
        Eigen::VectorXd buildStaticDistVector(Eigen::VectorXd fb, std::string sc_type, BeamSolverType type=BeamSolverType::Beam);

        // apply Dirichlet boundary conditions in a matrix
        Eigen::MatrixXd applyDBCMatrix(Eigen::MatrixXd K);

        // apply Dirichlet boundary condition in a vector
        Eigen::VectorXd applyDBCVec(Eigen::VectorXd R);

        // Calculate strain in beam elements based on displacement solution
        Eigen::MatrixXd calculateStrain(const Eigen::VectorXd& u, double E, int sample_points = 10, double y_top = 1.0);

        // Calculate stress in beam elements based on displacement solution
        Eigen::MatrixXd calculateStress(const Eigen::VectorXd& u, double E, int sample_points = 10, double y_top = 1.0);

        // Calculate maximum stress in beam elements (at extreme fibers)
        Eigen::MatrixXd calculateMaxStress(const Eigen::VectorXd& u, double E, double height, int sample_points = 10);

        // Calculate strain at a point within an element
        double calculateElementStrain(const Eigen::VectorXd& u_elem, double x, double L, double y);

        // Get strain and stress at a specific global coordinate point
        std::pair<double, double> getStrainStressAtPoint(const Eigen::VectorXd& u, double E, double x_global, double y);

        // Auxiliary methods for the middleware interaface
        Eigen::MatrixXd condense_matrix(Eigen::MatrixXd& KII, Eigen::MatrixXd& KIM, Eigen::MatrixXd& KMI, Eigen::MatrixXd& KMM);
        Eigen::VectorXd condense_vector(Eigen::VectorXd& RI, Eigen::VectorXd& RM, Eigen::MatrixXd& KIM, Eigen::MatrixXd& KMM);

        beam1d(Eigen::MatrixXd nodes_coordinates, Eigen::MatrixXi elements_indices, int model_order){
            nodes = nodes_coordinates;
            elements = elements_indices;
            order = model_order;
        }

};

}

#endif