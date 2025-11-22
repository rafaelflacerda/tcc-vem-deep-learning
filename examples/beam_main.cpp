#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;
 
int main()
{
  std::cout << std::fixed << std::setprecision(10);
  std::cout << "--- VEM Beam Cantilever Debug (2 Elementos, K=4) ---" << std::endl;

  // --- 1. CONFIGURAÇÃO E DISCRETIZAÇÃO ---
  const double L = 1;
  const int N_ELEMENTS = 34; // Teste com 2 elementos
  const double Q0 = -1.0;  // Carregamento distribuído constante
  const int ORDER = 3;     // Ordem ideal para solução exata (P4)

  mesh::beam bar;
  bar.horizontalBarDisc(L, N_ELEMENTS);

  MatrixXd nodes = bar.nodes;
  MatrixXi elements = bar.elements;

  VectorXd q = VectorXd::Zero(2);
  q(0) = Q0;
  q(1) = Q0;

  material::mat elastic;
  elastic.setElasticModule(2.1e+11);
  double E = elastic.E;

  solver::beam1d solver(nodes, elements, ORDER);
  solver.setInertiaMoment((0.02 * std::pow(0.003, 3)) / 12.0);

  // --- 2. MONTAGEM GLOBAL INICIAL ---

  // K_local tem um tamanho [DOFs locais totais] x [DOFs locais totais]
  // Para K=4, cada elemento tem 2 DOFs nodais (w, theta) em cada nó (2*2 = 4 DOFs nodais)
  // E + N_momentos internos (1, para K=4) = 5 DOFs locais por elemento.
  // K_global (sem condensação) terá 3 nós * 2 DOFs/nó + 2 elementos * 1 momento/elemento = 8 DOFs.
  
  MatrixXd K = solver.buildGlobalK(E);
  std::cout << "\n--- 1. Matriz de Rigidez Global K (Antes da Condensação) ---" << std::endl;
  std::cout << "Dimensão: " << K.rows() << "x" << K.cols() << std::endl;
  std::cout << K << std::endl;
  // Checagem: K deve ser 8x8 (3 nós * 2 DOFs/nó + 2 momentos internos = 8 DOFs globais totais)

  // --- 3. SUB-BLOCOS DA CONDENSAÇÃO ESTÁTICA ---
  
  // I (Nodal/Interface) e M (Momento/Interno)
  MatrixXd KII = solver.buildStaticCondensation(K, "KII");
  MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
  MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
  MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");

  std::cout << "\n--- 2. Sub-bloco Nodal KII (Rigidez Nodal) ---" << std::endl;
  std::cout << "Dimensão: " << KII.rows() << "x" << KII.cols() << std::endl;
  std::cout << KII << std::endl;
  // Checagem: KII deve ser 6x6 (3 nós * 2 DOFs/nó)

  std::cout << "\n--- 3. Sub-bloco Interno KMM (Rigidez dos Momentos) ---" << std::endl;
  std::cout << "Dimensão: " << KMM.rows() << "x" << KMM.cols() << std::endl;
  std::cout << KMM << std::endl;
  // Checagem: KMM deve ser 2x2 (2 elementos * 1 momento/elemento)

  std::cout << "\n--- 4. Sub-bloco Acoplamento KIM ---" << std::endl;
  std::cout << "Dimensão: " << KIM.rows() << "x" << KIM.cols() << std::endl;
  std::cout << KIM << std::endl;
  // Checagem: KIM deve ser 6x2

  // --- 4. MONTAGEM DO VETOR DE CARGA E SUB-BLOCOS ---
  
  solver.setDistributedLoad(q, elements);
  VectorXd R = solver.buildGlobalDistributedLoad();

  std::cout << "\n--- 5. Vetor de Carga Global R (Antes da Condensação) ---" << std::endl;
  std::cout << "Dimensão: " << R.size() << std::endl;
  std::cout << R << std::endl;
  // Checagem: R deve ter 8 elementos

  VectorXd RI = solver.buildStaticDistVector(R, "RI");
  VectorXd RM = solver.buildStaticDistVector(R, "RM");

  std::cout << "\n--- 6. Vetor de Carga Nodal RI ---" << std::endl;
  std::cout << "Dimensão: " << RI.size() << std::endl;
  std::cout << RI << std::endl;
  // Checagem: RI deve ter 6 elementos (DOFs nodais)

  std::cout << "\n--- 7. Vetor de Carga de Momento RM ---" << std::endl;
  std::cout << "Dimensão: " << RM.size() << std::endl;
  std::cout << RM << std::endl;
  // Checagem: RM deve ter 2 elementos (DOFs de momento)

  // --- 5. CONDENSAÇÃO ESTÁTICA FINAL ---

  MatrixXd K_inversa = KMM.inverse();
  std::cout << "\n--- 8. Inversa da Matriz de Momentos KMM^-1 ---" << std::endl;
  std::cout << K_inversa << std::endl;
  
  // K_condensado = KII - KIM * KMM^-1 * KMI
  MatrixXd K_ = KII - KIM * K_inversa * KMI;
  
  // R_condensado = RI - KIM * KMM^-1 * RM
  VectorXd R_ = RI - KIM * K_inversa * RM;

  std::cout << "\n--- 9. Matriz de Rigidez Condensada K_final ---" << std::endl;
  std::cout << "Dimensão: " << K_.rows() << "x" << K_.cols() << std::endl;
  std::cout << K_ << std::endl;
  // Checagem: K_final deve ser 6x6

  std::cout << "\n--- 10. Vetor de Carga Condensada R_final ---" << std::endl;
  std::cout << "Dimensão: " << R_.size() << std::endl;
  std::cout << R_ << std::endl;
  // Checagem: R_final deve ter 6 elementos

  // --- 6. APLICAÇÃO DO ENGATE (DBC) ---
  
  MatrixXi supp = MatrixXi::Zero(1,4);
  supp(0,0) = 0; // Nó 0
  supp(0,1) = 1; // Restringir w
  supp(0,2) = 1; // Restringir theta
  supp(0,3) = 0;
  solver.setSupp(supp);
  
  K_ = solver.applyDBCMatrix(K_);
  R_ = solver.applyDBCVec(R_);

  std::cout << "\n--- 11. Matriz K_final após DBC (6x6) ---" << std::endl;
  std::cout << K_ << std::endl;
  
  std::cout << "\n--- 12. Vetor R_final após DBC (6x1) ---" << std::endl;
  std::cout << R_ << std::endl;

  // --- 7. SOLUÇÃO FINAL ---

  VectorXd uh;
  uh = K_.ldlt().solve(R_);

  std::cout << "\n--- 13. Solução Final (DOFs Nodais) ---" << std::endl;
  std::cout << "u_h_completo: ";
  for (int i = 0; i < uh.size(); ++i) {
      std::cout << uh(i);
      if (i < uh.size() - 1) {
          std::cout << ",";
      }
  }
  std::cout << std::endl;

  // O comando abaixo está incorreto, pois buildLocalK espera apenas as coordenadas.
  // std::cout<< solver.buildLocalK(nodes, E) <<std::endl;

  return 0;
}