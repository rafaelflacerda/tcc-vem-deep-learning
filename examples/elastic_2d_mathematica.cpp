#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <nlohmann/json.hpp>

#include "lib/utils/operations.hpp"
#include "lib/utils/integration.hpp"
#include "lib/mesh/beam.hpp"
#include "lib/mesh/datasource.hpp"
#include "lib/solver/beam1d.hpp"
#include "lib/material/mat.hpp"
#include "lib/solver/linearElastic2d.hpp"

 
using Eigen::MatrixXd;

using json = nlohmann::json;

int main() {
    mesh::datasource ds("data/unit_square_16.json");

    Eigen::MatrixXd nodes = ds.nodes;
    Eigen::MatrixXi elements = ds.elements;
    Eigen::MatrixXi supp = ds.supp;
    Eigen::MatrixXi load = ds.load;
    double qx = ds.qx;
    double qy = ds.qy;

    std::cout << qx << std::endl;

    solver::linearElastic2d s(nodes, elements, 1);
    material::mat elastic;
    elastic.setElasticModule(7000);
    elastic.setPoissonCoef(0.3);
    Eigen::MatrixXd C = elastic.build2DElasticity();
    Eigen::MatrixXd K = s.buildGlobalK(C);
    s.setSupp(supp);
    Eigen::MatrixXd K_ = s.applyDBC(K);
    s.setLoad(load);
    Eigen::VectorXd f = s.applyNBC(qx,qy);
    
    Eigen::VectorXd uh;
    uh = K_.ldlt().solve(f);
    std::cout << uh << std::endl;

    return 0;
}
