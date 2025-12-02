#include "solver/linearElastic2d.hpp"
#include "material/mat.hpp"
#include "utils/operations.hpp"
#include <iomanip>

// Biblioteca JSON
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include "json.hpp" // Certifique-se que este arquivo está acessível
using json = nlohmann::json;

// Função de Leitura do JSON
void readMeshJSON(const std::string& filename, 
                  Eigen::MatrixXd& nodes, 
                  Eigen::MatrixXi& elements, 
                  Eigen::VectorXd& uh,
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

    // 4. Ler deslocamentos globais uh
    std::vector<double> vec_u = data["solution_u"];   // <-- nome do campo no JSON
    uh.resize(vec_u.size());
    for (size_t i = 0; i < vec_u.size(); ++i) {
        uh(i) = vec_u[i];
    }

}

// Função para Salvar o Resultado
void saveResultsJSON(const std::string& input_filename, 
                     const std::string& output_filename, 
                     const Eigen::MatrixXd& sigma) {
    // Lê o arquivo original para manter os dados de entrada (features)
    std::ifstream f(input_filename);
    json data = json::parse(f);

    // 2) sigma por elemento: (ne x 3)
    int ne = sigma.rows();
    std::vector<std::array<double,3>> sigma_vec(ne);
    for (int e = 0; e < ne; ++e) {
        sigma_vec[e] = { sigma(e,0), sigma(e,1), sigma(e,2) };
    }
    data["sigma_elements"] = sigma_vec;  // por exemplo

    // Salva no caminho de destino
    std::ofstream o(output_filename);
    o << std::setw(4) << data << std::endl;
}

// ALTERAÇÃO: Adicionado argc e argv para ler do terminal
int main(int argc, char* argv[]) {

  // VERIFICAÇÃO DE ARGUMENTO (Para não dar crash se esquecer)
  if (argc < 3) {
      std::cerr << "Uso: ./run_single_case_2D <input_json> <output_json>" << std::endl;
      return 1;
  }
  
  std::string json_path = argv[1];   // Arquivo de Entrada (Parameters)
  std::string output_path = argv[2]; // Arquivo de Saída (Dataset Treino)

  material::mat elastic;

  utils::operations op;

  elastic.setElasticModule(1e+0);
  double E = elastic.E;

  // Variáveis para receber os dados do JSON
  Eigen::MatrixXd nodes;
  Eigen::MatrixXi elements;
  Eigen::VectorXd uh;
  double poisson_json;

  // --- ALTERAÇÃO 2: Caminho do arquivo AGORA É DINÂMICO (json_path lido acima) ---
  
  try {
      readMeshJSON(json_path, nodes, elements, uh, poisson_json);
      // Removi os couts de sucesso/poisson para limpar o output do teste de velocidade
      // Se quiser ver, pode manter. Mas para benchmark é melhor menos texto.
  } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return -1;
  }

  // Usar o Poisson do JSON
  elastic.setPoissonCoef(poisson_json); 
  Eigen::MatrixXd C = elastic.build2DElasticity();

  // --- CÓDIGO ORIGINAL CONTINUA AQUI ---

  int order = 1;
  
  solver::linearElastic2d solver(nodes, elements, order);
  
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

  saveResultsJSON(json_path, output_path, sigma);

}