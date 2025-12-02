#include "solver/linearElastic2d.hpp"
#include "material/mat.hpp"
#include "utils/operations.hpp"
#include <iomanip>

// --- ADIÇÃO 1: Biblioteca JSON ---
#include <fstream>
#include <iostream>
#include <vector>
#include "json.hpp" // Certifique-se que este arquivo está acessível
using json = nlohmann::json;

// --- ADIÇÃO 2: Função de Leitura do JSON ---
void readMeshJSON(const std::string& filename, 
                  Eigen::MatrixXd& nodes, 
                  Eigen::MatrixXi& elements, 
                  Eigen::MatrixXi& supp, 
                  Eigen::MatrixXi& load,
                  double& poisson_val) {
    
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Erro: Nao foi possivel abrir o arquivo " + filename);
    }

    json data = json::parse(f);

    // 1. Ler Poisson
    poisson_val = data["poisson"];

    // 2. Ler Nodes
    std::vector<std::vector<double>> vec_nodes = data["nodes"];
    nodes.resize(vec_nodes.size(), 2);
    for(size_t i=0; i<vec_nodes.size(); ++i) {
        nodes(i, 0) = vec_nodes[i][0];
        nodes(i, 1) = vec_nodes[i][1];
    }

    // 3. Ler Elements
    std::vector<std::vector<int>> vec_elems = data["elements"];
    if (!vec_elems.empty()) {
        elements.resize(vec_elems.size(), 3); // Triângulos (3 nós)
        for(size_t i=0; i<vec_elems.size(); ++i) {
            elements(i, 0) = vec_elems[i][0];
            elements(i, 1) = vec_elems[i][1];
            elements(i, 2) = vec_elems[i][2];
        }
    }

    // 4. Ler Supp
    std::vector<std::vector<int>> vec_supp = data["supp"];
    if (!vec_supp.empty()) {
        supp.resize(vec_supp.size(), 3);
        for(size_t i=0; i<vec_supp.size(); ++i) {
            supp(i, 0) = vec_supp[i][0];
            supp(i, 1) = vec_supp[i][1];
            supp(i, 2) = vec_supp[i][2];
        }
    }

    // 5. Ler Load
    std::vector<std::vector<int>> vec_load = data["load"];
    if (!vec_load.empty()) {
        load.resize(vec_load.size(), 2);
        for(size_t i=0; i<vec_load.size(); ++i) {
            load(i, 0) = vec_load[i][0];
            load(i, 1) = vec_load[i][1];
        }
    }
}

// ALTERAÇÃO: Adicionado argc e argv para ler do terminal
int main(int argc, char* argv[]) {

  // VERIFICAÇÃO DE ARGUMENTO (Para não dar crash se esquecer)
  if (argc < 2) {
      std::cerr << "Uso: ./run_single_case_2D <caminho_do_json>" << std::endl;
      return 1;
  }
  std::string json_path = argv[1]; // Pega o caminho passado pelo script Python

  material::mat elastic;

  utils::operations op;

  elastic.setElasticModule(1e+0);
  double E = elastic.E;

  // --- ALTERAÇÃO 1: Variáveis para receber os dados do JSON ---
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  Eigen::MatrixXi supp;
  Eigen::MatrixXi load;
  double poisson_json;

  // --- ALTERAÇÃO 2: Caminho do arquivo AGORA É DINÂMICO (json_path lido acima) ---
  
  try {
      readMeshJSON(json_path, nodes, elements, supp, load, poisson_json);
      // Removi os couts de sucesso/poisson para limpar o output do teste de velocidade
      // Se quiser ver, pode manter. Mas para benchmark é melhor menos texto.
  } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return -1;
  }

  // --- ALTERAÇÃO 3: Usar o Poisson do JSON ---
  elastic.setPoissonCoef(poisson_json); 
  Eigen::MatrixXd C = elastic.build2DElasticity();

  // --- CÓDIGO ORIGINAL CONTINUA AQUI ---
  
  double qx = 1; // Carga Unitária (ou pequena) para regime linear
  double qy = 0;

  int order = 1;
  
  solver::linearElastic2d solver(nodes, elements, order);

  solver.setSupp(supp);
  solver.setLoad(load);

  Eigen::MatrixXd K = solver.buildGlobalK(C);

  Eigen::MatrixXd K_ = solver.applyDBC(K);

  Eigen::VectorXd f =  solver.applyNBC(qx, qy);

  Eigen::VectorXd R_ = f;

  for (int i = 0; i < supp.rows(); i++) {
    int node = supp(i,0);
    if (supp(i,1) == 1) R_(2*node)   = 0.0;
    if (supp(i,2) == 1) R_(2*node+1) = 0.0;
  }

  Eigen::VectorXd uh = K_.ldlt().solve(R_);
  
  int ne = elements.rows();
  Eigen::MatrixXd sigma(ne, 3); // sigma_xx, sigma_yy, tau_xy

  for (int e = 0; e < ne; e++) {

    Eigen::MatrixXi elem = elements.row(e);
    
    Eigen::VectorXi dofs = op.getOrder1Indices(elem);

    Eigen::MatrixXd coords = op.getCoordinatesPlane(elem, nodes);

    Eigen::VectorXd ue(dofs.size());

    for (int i = 0; i < dofs.size(); i++) {
        ue(i) = uh(dofs(i));
    }

    Eigen::MatrixXd B = solver.buildB(coords);
    Eigen::MatrixXd G = solver.buildG(coords);

    Eigen::MatrixXd Bp = G.inverse() * B;

    Eigen::VectorXd strain = Bp * ue;

    Eigen::VectorXd stress = C * strain;

    sigma.row(e) = stress.transpose();
  }

  double max_sigma_xx = sigma.col(0).maxCoeff();

  double sigma0 = qx;

  double FCT = max_sigma_xx / sigma0;

  std::cout << "Max sigma_xx = " << max_sigma_xx << std::endl;

  std::cout << "FCT = " << FCT << std::endl;

}