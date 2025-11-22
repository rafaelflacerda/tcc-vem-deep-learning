#include <iostream>
#include <Eigen/Dense>
#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"

int main() {
    std::cout << "=== VEM 1D Beam Example ===" << std::endl;
    
    // Initialize beam and discretization
    mesh::beam bar;
    bar.horizontalBarDisc(0.45, 20);  // 0.45m length, 20 nodes
    
    Eigen::MatrixXd nodes = bar.nodes;
    Eigen::MatrixXi elements = bar.elements;
    
    std::cout << "Mesh created: " << nodes.rows() << " nodes, " 
              << elements.rows() << " elements" << std::endl;
    
    // Set distributed load (downward)
    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    q(0) = -1.0;
    q(1) = -1.0;
    
    // Define material properties (steel)
    material::mat elastic;
    elastic.setElasticModule(2.1e+11);  // 210 GPa
    double E = elastic.E;
    
    std::cout << "Young's modulus: " << E << " Pa" << std::endl;
    
    // Initialize solver
    int order = 5;
    solver::beam1d solver(nodes, elements, order);
    solver.setInertiaMoment((0.02*pow(0.003,3))/12);  // Rectangular section
    
    // Build global stiffness matrix
    Eigen::MatrixXd K = solver.buildGlobalK(E);
    
    // Static condensation
    Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
    Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
    Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
    Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");
    
    // Apply distributed load
    solver.setDistributedLoad(q, elements);
    Eigen::VectorXd R = solver.buildGlobalDistributedLoad();
    Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
    Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");
    
    // Apply boundary conditions (fixed at first node)
    Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1, 4);
    supp(0, 0) = 0;  // node index
    supp(0, 1) = 1;  // restrict displacement w
    supp(0, 2) = 1;  // restrict rotation w'
    supp(0, 3) = 0;  // always 0 for horizontal bar
    solver.setSupp(supp);
    
    // Build final system
    Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
    Eigen::VectorXd R_ = RI - KIM * KMM.inverse() * RM;
    
    K_ = solver.applyDBCMatrix(K_);
    R_ = solver.applyDBCVec(R_);
    
    // Solve
    Eigen::VectorXd uh = K_.ldlt().solve(R_);
    
    std::cout << "\n=== Solution ===" << std::endl;
    std::cout << "Displacement vector size: " << uh.size() << std::endl;
    std::cout << "Max displacement: " << uh.maxCoeff() << " m" << std::endl;
    std::cout << "Min displacement: " << uh.minCoeff() << " m" << std::endl;
    
    std::cout << "\n✅ Simulation completed successfully!" << std::endl;
    
    return 0;
}
