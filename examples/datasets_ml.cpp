#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include "lib/utils/operations.hpp"
#include "lib/utils/integration.hpp"
#include "lib/mesh/beam.hpp"
#include "lib/solver/beam1d.hpp"
#include "lib/material/mat.hpp"
#include "lib/solver/linearElastic2d.hpp"
#include "lib/mesh/datasource.hpp"
#include "lib/mesh/delaunay.hpp"
#include "lib/mesh/voronoi.hpp"
#include "lib/mesh/voronoiMesh.hpp"         
#include "lib/mesh/helpers.hpp"
#include "models/enums.hpp"

using Eigen::MatrixXd;
using json = nlohmann::json;
using namespace mesh;  // Add this at the top with other using statements

void portic(){
    std::cout<<"VEM Beam"<<std::endl;
    BeamSolverType type = BeamSolverType::Portic;

    mesh::datasource ds("data/portic_8.json", type);

    Eigen::MatrixXd nodes = ds.nodes;
    Eigen::MatrixXi elements = ds.elements;
    Eigen::MatrixXi supp = ds.supp;
    Eigen::MatrixXi load = ds.load;         
    MatrixXd q = MatrixXd::Zero(2, 1);
    q(0) = ds.qx;
    q(1) = ds.qx;

    utils::operations op;
    solver::beam1d solver(nodes, elements, 5);

    // Add debug output
    std::cout << "Number of nodes: " << nodes.rows() << std::endl;
    std::cout << "Number of elements: " << elements.rows() << std::endl;
    std::cout << "Matrix K size: " << solver.buildGlobalK(1.0, type).rows() << "x" << solver.buildGlobalK(1.0, type).cols() << std::endl;

    // Samples
    int numSamples = 80;
    Eigen::VectorXd E_samples, A_samples, I_samples;
    ds.generateRandomSamples(numSamples, E_samples, A_samples, I_samples);

    for(int i = 0; i < numSamples; ++i){
        material::mat elastic;
        elastic.setElasticModule(E_samples(i));
        double E = elastic.E;
        solver.setArea(A_samples(i));
        solver.setInertiaMoment(I_samples(i));

        solver.setSupp(supp);

        Eigen::MatrixXd K = solver.buildGlobalK(E, type);
        Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII", type);
        Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM", type);
        Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI", type);
        Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM", type);

        solver.setDistributedLoad(q, load);
        Eigen::VectorXd R = solver.buildGlobalDistributedLoad(type);
        Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI", type);
        Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM", type);

        Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
        Eigen::MatrixXd R_ = RI - KIM * KMM.inverse() * RM;

        K_ = solver.applyDBCMatrix(K_);
        R_ = solver.applyDBCVec(R_);

        Eigen::VectorXd uh;
        uh = K_.ldlt().solve(R_);

        ds.saveDisplacementsToJson(uh, E_samples(i), A_samples(i), I_samples(i), "output/displacements_8_"+std::to_string(i));
    }
}

void cantileverBeam(){
    std::cout<<"VEM Beam"<<std::endl;

    mesh::beam bar;

    bar.horizontalBarDisc(200, 32);

    int numSamples = 20;
    Eigen::VectorXd E_samples, A_samples, I_samples;
    mesh::datasource::generateRandomSamples(numSamples, E_samples, A_samples, I_samples);

    Eigen::MatrixXd nodes = bar.nodes;
    Eigen::MatrixXi elements = bar.elements;

    Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    q(0) = -20.0;
    q(1) = -20.0;

    Eigen::MatrixXi supp;
    

    for(int i=0; i<numSamples; ++i){
        material::mat elastic;
        elastic.setElasticModule(E_samples(i));
        double E = elastic.E;

        int order = 4;
        solver::beam1d solver(nodes, elements, order);
        solver.setInertiaMoment(I_samples(i));

        Eigen::MatrixXd K = solver.buildGlobalK(E);
        
        solver.setDistributedLoad(q, elements);
        Eigen::VectorXd R = solver.buildGlobalDistributedLoad();

        supp = Eigen::MatrixXi::Zero(1,4);
        supp(0,0) = 0;
        supp(0,1) = 1;
        supp(0,2) = 1;
        supp(0,3) = 0;
        solver.setSupp(supp);

        if(order > 3){
            Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
            Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
            Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
            Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");

            Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
            Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");

            Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
            Eigen::MatrixXd R_ = RI - KIM * KMM.inverse() * RM;
            //R_(2) = -0.225;
            //std::cout << R_ << std::endl;

            K_ = solver.applyDBCMatrix(K_);
            R_ = solver.applyDBCVec(R_);

            Eigen::VectorXd uh;
            uh = K_.ldlt().solve(R_);

            mesh::datasource::saveDisplacementsToJson(uh, E_samples(i), 0.0, I_samples(i), "beam/displacements_32_"+std::to_string(i));
            // std::cout << uh  << std::endl;
        } else {
            Eigen::MatrixXd K_ = solver.applyDBCMatrix(K);
            Eigen::MatrixXd R_ = solver.applyDBCVec(R);

            Eigen::VectorXd uh;
            uh = K_.ldlt().solve(R_);

            mesh::datasource::saveDisplacementsToJson(uh, E_samples(i), 0.0, I_samples(i), "beam/displacements_32_"+std::to_string(i));
            //std::cout << uh  << std::endl;
        }
    }

    // Save the geometry
    mesh::datasource::saveBeamGeometryToJson(nodes, elements, supp, elements, q, "beam_64");
    
}


 
int main()
{
    portic();
    cantileverBeam();

	return 0;
}