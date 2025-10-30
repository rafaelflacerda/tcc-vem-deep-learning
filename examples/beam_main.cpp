#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "lib/utils/operations.hpp"
#include "lib/utils/integration.hpp"
#include "lib/mesh/beam.hpp"
#include "lib/solver/beam1d.hpp"
#include "lib/material/mat.hpp"
#include "lib/solver/linearElastic2d.hpp"
 
using Eigen::MatrixXd;
 
int main()
{
  std::cout<<"VEM Beam"<<std::endl;

  mesh::beam bar;

  bar.horizontalBarDisc(0.45, 20);

  Eigen::MatrixXd nodes = bar.nodes;
  Eigen::MatrixXi elements = bar.elements;

  Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
  q(0) = -1.0;
  q(1) = -1.0;

  material::mat elastic;
  elastic.setElasticModule(2.1e+11);
  double E = elastic.E;

  int order = 5;
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
  supp(0,0) = 0;
  supp(0,1) = 1;
  supp(0,2) = 1;
  supp(0,3) = 0;
  solver.setSupp(supp);
  
  Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
  Eigen::MatrixXd R_ = RI - KIM * KMM.inverse() * RM;
  //R_(2) = -0.225;
  //std::cout << R_ << std::endl;

  K_ = solver.applyDBCMatrix(K_);
  R_ = solver.applyDBCVec(R_);

  Eigen::VectorXd uh;
  uh = K_.ldlt().solve(R_);

  std::cout << uh  << std::endl;
	std::cout<< solver.buildLocalK(nodes, E) <<std::endl;

	return 0;
}
