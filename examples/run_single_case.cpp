#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"
#include <iomanip>

int main() {
    // Initialize beam and discretization
    mesh::beam bar;
    bar.horizontalBarDisc(0.45, 20);

    // Get nodes and elements from the discretization
    Eigen::MatrixXd nodes = bar.nodes;
    Eigen::MatrixXi elements = bar.elements;

    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    q(0) = -1.0;
    q(1) = -1.0;

    material::mat elastic;
    elastic.setElasticModule(2.1e+11);
    double E = elastic.E;

    int order = 4;
    solver::beam1d solver(nodes, elements, order);
    solver.setInertiaMoment((0.02*pow(0.003,3))/12);

    Eigen::MatrixXd K = solver.buildGlobalK(E);

    Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
    Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
    Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
    Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");

    solver.setDistributedLoad(q, elements);
    Eigen::VectorXd R = solver.buildGlobalDistributedLoad();
    Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
    Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");

    Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1,4);
    supp(0,0) = 0; // node index
    supp(0,1) = 1; // restrict displacement w
    supp(0,2) = 1; // restrict the rotation w'
    supp(0,3) = 0; // always set 0 for horizontal bar
    solver.setSupp(supp);

    Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
    Eigen::MatrixXd R_ = RI - KIM * KMM.inverse() * RM;

    K_ = solver.applyDBCMatrix(K_);
    R_ = solver.applyDBCVec(R_);

    Eigen::VectorXd uh;
    uh = K_.ldlt().solve(R_);

    std::cout << std::fixed << std::setprecision(8);

    // Deslocamentos (índices pares)
    //std::cout << "Deslocamentos: ";
    //for (int j = 0; j < uh.size() / 2; ++j) {
    //    std::cout << uh(j * 2);
    //    if (j < uh.size() / 2 - 1) std::cout << ",";
    //}
    //std::cout << std::endl;

    // Rotações (índices ímpares)
    //std::cout << "Rotacoes: ";
    //for (int j = 0; j < uh.size() / 2; ++j) {
    //    std::cout << uh(j * 2 + 1);
    //    if (j < uh.size() / 2 - 1) std::cout << ",";
    //}
    //std::cout << std::endl;

    for (int i = 0; i < uh.size(); ++i) {
            std::cout << uh(i);
            if (i < uh.size() - 1) {
                std::cout << ",";
            }
    }
    std::cout << std::endl;

}