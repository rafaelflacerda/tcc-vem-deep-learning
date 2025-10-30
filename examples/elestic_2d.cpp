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
  std::cout<<"Debugging"<<std::endl;

  Eigen::MatrixXd coords;
  coords =(Eigen::Matrix<double,4,2>()<< 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5).finished();

  Eigen::MatrixXd coord;
  coord = (Eigen::Matrix<double,2,2>()<<0.0,0.0,1.0,0.0).finished();


  utils::integration Ig;
  Ig.setGaussParams(3);
  Ig.setParamCoords(coords, 1, 1);
  // std::cout << coord << std::endl;

  utils::operations op;
  Eigen::Vector2d n = op.computerNormalVector(coord);
  // std::cout << n << std::endl;

  int order = 1;
  Eigen::MatrixXd nodes = (Eigen::Matrix<double,9,2>()<<0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0).finished();
  Eigen::MatrixXi elements = (Eigen::Matrix<int,4,4>()<<0,1,2,3, 1,4,5,2, 2,5,6,7, 3,2,7,8).finished();
  solver::linearElastic2d s(nodes, elements, order);
  Eigen::MatrixXd NE;
  // NE = s.buildNE(coord);
  // std::cout << NE << std::endl;
  // Eigen::MatrixXd Nv = s.buildNv(3,0);
  // std::cout << Nv << std::endl;
  // Eigen::MatrixXd G = s.buildG(coords);
  // std::cout << G << std::endl;
  // Eigen::MatrixXd B = s.buildB(coords);
  // std::cout << B << std::endl;
  // Eigen::MatrixXd D = s.buildD(coords);
  // std::cout << D << std::endl;

  material::mat elastic;
  elastic.setElasticModule(7000);
  elastic.setPoissonCoef(0.3);
  Eigen::MatrixXd C = elastic.build2DElasticity();

  // Eigen::MatrixXd Kc = s.buildConsistency(coords, C);
  // std::cout << Kc << std::endl;
  // Eigen::MatrixXd Ks = s.buildStability(coords, Kc);

  // Eigen::MatrixXd K = s.buildLocalK(coords, C);
  // std::cout << K << std::endl;

  // Eigen::MatrixXi e = (Eigen::Matrix<int,1,3>()<<1,4,5).finished();
  // Eigen::VectorXi dofs = op.getOrder1Indices(e);
  // std::cout << dofs << std::endl;

  Eigen::MatrixXd K = s.buildGlobalK(C);
  // std::cout << K << std::endl;

  Eigen::MatrixXi supp = (Eigen::Matrix<int, 5,3>()<<0,1,1, 1,0,1, 4,0,1, 3,1,0, 8,1,0).finished();
  s.setSupp(supp);

  Eigen::MatrixXd K_ = s.applyDBC(K);
  // std::cout << K_ << std::endl;

  Eigen::MatrixXi load = (Eigen::Matrix<int, 2,2>()<<4,5, 5,6).finished();
  s.setLoad(load);
  double qx = 2000, qy=0;
  Eigen::VectorXd f = s.applyNBC(qx,qy);
  // std::cout << f << std::endl;

  Eigen::VectorXd uh;
  uh = K_.ldlt().solve(f);
  std::cout << uh << std::endl;

  // utils::preprocessor p;
  // p.readJson();

	return 0;
}

