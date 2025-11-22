#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <cmath>
#include <sstream> // Necessário para montar o nome da pasta
#include <Eigen/Dense>
#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"

namespace fs = std::filesystem;

int main() {
    std::cout << "=== Gerando Dataset Organizado (VEM Solver) ===" << std::endl;

    srand(static_cast<unsigned>(time(nullptr)));

    // --- 1. Definição de Parâmetros ---
    double L = 1.0;
    int numElements = 44; // 45 nós
    int numSamples = 100000;
    int order = 3;        // Ordem 3 (Estável e correta para Euler-Bernoulli)

    // Intervalos
    double E_min = 50e9,  E_max = 250e9;
    double I_min = 1e-9,  I_max = 1e-5;
    double q_min = 1000,    q_max = 10000;

    // --- 2. Construção do Nome da Pasta ---
    // Lógica para sufixo "k" (ex: 1000 -> 1k)
    std::string sample_str = (numSamples >= 1000 && numSamples % 1000 == 0) 
                             ? std::to_string(numSamples / 1000) + "k" 
                             : std::to_string(numSamples);

    std::stringstream folderNameStream;
    folderNameStream << "dataset_viga1D_EngastadaLivre_" 
                     << (int)L << "Meter_" 
                     << numElements << "Elements_" 
                     << sample_str << "Samples";

    std::string folderName = folderNameStream.str();
    
    // Caminho base: assume que está rodando dentro de build/, cria em build/output/ML_datasets/...
    fs::path basePath = "output/ML_datasets"; 
    fs::path fullPath = basePath / folderName;

    // Criar diretórios recursivamente
    try {
        if (fs::exists(fullPath)) {
            std::cout << "⚠️  Aviso: A pasta '" << fullPath.string() << "' ja existe. Arquivos serao sobrescritos.\n";
        }
        fs::create_directories(fullPath);
    } catch (const std::exception& e) {
        std::cerr << "Erro ao criar diretório: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "📂 Salvando dados em: " << fullPath.string() << std::endl;

    // --- 3. Gerar Arquivo de Metadados (parameters_info.txt) ---
    std::ofstream info_file(fullPath / "parameters_info.txt");
    if (info_file.is_open()) {
        info_file << "=== Metadata do Dataset ===\n\n";
        info_file << "Tipo: Viga 1D\n";
        info_file << "BC: Engastada-Livre (Cantilever)\n";
        info_file << "Solver Order (K): " << order << "\n\n";
        
        info_file << "--- Geometria e Discretizacao ---\n";
        info_file << "Comprimento (L): " << L << " metro\n";
        info_file << "numElements: " << numElements << "\n";
        info_file << "numSamples: " << numSamples << "\n\n";
        
        info_file << "--- Intervalos de Randomizacao ---\n";
        info_file << "E_min: " << std::scientific << E_min << "\n";
        info_file << "E_max: " << std::scientific << E_max << "\n";
        info_file << "I_min: " << std::scientific << I_min << "\n";
        info_file << "I_max: " << std::scientific << I_max << "\n";
        info_file << "q_min: " << std::scientific << q_min << " (Magnitude)\n";
        info_file << "q_max: " << std::scientific << q_max << " (Magnitude)\n";
        
        info_file.close();
    } else {
        std::cerr << "Erro ao criar arquivo de info.\n";
    }

    // --- 4. Arquivos CSV ---
    // Usamos o operador / do filesystem para concatenar caminhos de forma segura
    std::ofstream csv_training(fullPath / "training_data.csv");
    csv_training << "sample_id,E,I,q,w_0.2L,theta_0.2L,w_0.3L,theta_0.3L,w_0.5L,theta_0.5L,w_0.7L,theta_0.7L,w_1.0L,theta_1.0L\n";
    csv_training << std::scientific << std::setprecision(10);

    std::ofstream csv_validation(fullPath / "full_validation_data.csv");
    csv_validation << "sample_id,x_position,E,I,q,w,theta\n";
    csv_validation << std::scientific << std::setprecision(10);

    // --- 5. Pré-cálculo dos Índices de Nós ---
    double training_positions[5] = {0.2, 0.3, 0.5, 0.7, 1.0};
    int training_node_indices[5];

    std::cout << "Indices de nos selecionados para N=" << numElements << ":" << std::endl;
    for (int p = 0; p < 5; p++) {
        training_node_indices[p] = (int)std::round(training_positions[p] * numElements);
        //std::cout << "  Pos " << training_positions[p] << "L -> No " << training_node_indices[p] << std::endl;
    }

    // --- 6. Loop Principal ---
    int successCount = 0;
    int failureCount = 0;

    for (int i = 0; i < numSamples; ++i) {
        double r1 = (double)rand() / RAND_MAX;
        double r2 = (double)rand() / RAND_MAX;
        double r3 = (double)rand() / RAND_MAX;

        double E = E_min + r1 * (E_max - E_min);
        double I = I_min + r2 * (I_max - I_min);
        double q = -(q_min + r3 * (q_max - q_min));

        // Mesh
        mesh::beam bar;
        bar.horizontalBarDisc(L, numElements);
        Eigen::MatrixXd nodes = bar.nodes;
        Eigen::MatrixXi elements = bar.elements;

        // Solver Setup
        material::mat elastic;
        elastic.setElasticModule(E);
        solver::beam1d solver(nodes, elements, order); // K=3
        solver.setInertiaMoment(I);

        Eigen::VectorXd q_vec = Eigen::VectorXd::Zero(2);
        q_vec(0) = q;
        q_vec(1) = q;
        solver.setDistributedLoad(q_vec, elements);

        // Build Matrices
        Eigen::MatrixXd K = solver.buildGlobalK(E);
        
        Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
        Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
        Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
        Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");

        Eigen::VectorXd R = solver.buildGlobalDistributedLoad();
        Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
        Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");

        // BCs
        Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1, 4);
        supp(0, 0) = 0; supp(0, 1) = 1; supp(0, 2) = 1; supp(0, 3) = 0;
        solver.setSupp(supp);

        // Condensation & Solving
        Eigen::MatrixXd K_inversa = KMM.inverse();
        Eigen::MatrixXd K_ = KII - KIM * K_inversa * KMI;
        Eigen::VectorXd R_ = RI - KIM * K_inversa * RM;

        K_ = solver.applyDBCMatrix(K_);
        R_ = solver.applyDBCVec(R_);

        Eigen::LDLT<Eigen::MatrixXd> ldlt(K_);
        if (ldlt.info() != Eigen::Success) {
            failureCount++;
            continue;
        }

        Eigen::VectorXd uh = ldlt.solve(R_);

        // Salvar Treinamento
        csv_training << i << "," << E << "," << I << "," << q;
        for (int p = 0; p < 5; p++) {
            int node_idx = training_node_indices[p];
            csv_training << "," << uh(node_idx * 2) << "," << uh(node_idx * 2 + 1);
        }
        csv_training << "\n";

        // Salvar Validação
        for (int j = 0; j < nodes.rows(); j++) {
            csv_validation << i << "," << nodes(j, 0) << "," << E << "," << I << "," << q << ","
                          << uh(j * 2) << "," << uh(j * 2 + 1) << "\n";
        }

        successCount++;

        if ((i + 1) % 100 == 0) {
            std::cout << "  [Status] Processadas " << (i + 1) << "/" << numSamples << std::endl;
        }
    }

    csv_training.close();
    csv_validation.close();

    std::cout << "\n✅ Sucesso! Arquivos salvos em: " << fullPath.string() << std::endl;
    std::cout << "   - parameters_info.txt (Metadados criados)" << std::endl;
    std::cout << "   - training_data.csv" << std::endl;
    std::cout << "   - full_validation_data.csv" << std::endl;

    return 0;
}