#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "mesh/beam.hpp"
#include "solver/beam1d.hpp"
#include "material/mat.hpp"

using Eigen::MatrixXd;

void generateDatasetForML(){
    std::cout << "=== Generating VEM Dataset for ML ===" << std::endl;

    // Beam geometry
    double L = 1.0;  // Length in meters
    int numElements = 40;  // Number of elements
    
    mesh::beam bar;
    bar.horizontalBarDisc(L, numElements);
    
    Eigen::MatrixXd nodes = bar.nodes;
    Eigen::MatrixXi elements = bar.elements;
    
    std::cout << "Mesh: " << nodes.rows() << " nodes, " 
              << elements.rows() << " elements" << std::endl;

    // Define parameter ranges
    int numSamples = 1000;  // Number of simulations
    
    std::cout << "Generating " << numSamples << " samples..." << std::endl;
    
    // Random samples for material and loading
    Eigen::VectorXd E_samples(numSamples);
    Eigen::VectorXd I_samples(numSamples);
    Eigen::VectorXd q_samples(numSamples);
    
    // Material properties variation
    double E_min = 50e9;   // 50 GPa (aluminum-like)
    double E_max = 250e9;  // 250 GPa (steel-like)
    
    double I_min = 1e-8;   // Small inertia
    double I_max = 1e-6;   // Large inertia
    
    double q_min = -1000.0;  // Light load
    double q_max = -100000.0; // Heavy load
    
    // Generate random samples
    srand(42);  // Fixed seed for reproducibility
    for(int i = 0; i < numSamples; ++i){
        double r1 = (double)rand() / RAND_MAX;
        double r2 = (double)rand() / RAND_MAX;
        double r3 = (double)rand() / RAND_MAX;
        
        E_samples(i) = E_min + r1 * (E_max - E_min);
        I_samples(i) = I_min + r2 * (I_max - I_min);
        q_samples(i) = q_min + r3 * (q_max - q_min);
    }
    
    // Boundary conditions (fixed at x=0)
    Eigen::MatrixXi supp = Eigen::MatrixXi::Zero(1, 4);
    supp(0, 0) = 0;  // node index
    supp(0, 1) = 1;  // restrict w
    supp(0, 2) = 1;  // restrict w'
    supp(0, 3) = 0;  // horizontal bar
    
    // ============================================================
    // NOVA SEÇÃO: Definir frações de comprimento e nós correspondentes
    // ============================================================
    std::vector<double> fractions = {0.2, 0.4, 0.6, 0.8, 1.0};
    std::vector<int> target_nodes;
    
    for(double frac : fractions){
        double target_x = frac * L;
        int closest_node = 0;
        double min_dist = std::abs(nodes(0, 0) - target_x);
        
        for(int j = 0; j < nodes.rows(); ++j){
            double dist = std::abs(nodes(j, 0) - target_x);
            if(dist < min_dist){
                min_dist = dist;
                closest_node = j;
            }
        }
        target_nodes.push_back(closest_node);
        std::cout << "Fração " << frac << "L encontrada no nó " << closest_node 
                  << " (x=" << nodes(closest_node, 0) << "m)" << std::endl;
    }
    // ============================================================
    
    // Create output directory
    system("mkdir -p output/ml_dataset_100k");
    
    // Open CSV file for parameters
    std::ofstream csv_file("output/ml_dataset_100k/parameters.csv");
    csv_file << "sample_id,E,I,q";
    for(double frac : fractions){
        csv_file << ",desloc_" << frac << "L";
    }
    for(double frac : fractions){
        csv_file << ",rotacao_" << frac << "L";
    }
    csv_file << "\n";
    
    // Run simulations
    int order = 5;
    for(int i = 0; i < numSamples; ++i){
        
        if(i % 20 == 0){
            std::cout << "Sample " << i << "/" << numSamples << std::endl;
        }
        
        material::mat elastic;
        elastic.setElasticModule(E_samples(i));
        double E_value = elastic.E;
        
        solver::beam1d solver(nodes, elements, order);
        solver.setInertiaMoment(I_samples(i));
        
        Eigen::MatrixXd K = solver.buildGlobalK(E_value);
        
        Eigen::MatrixXd KII = solver.buildStaticCondensation(K, "KII");
        Eigen::MatrixXd KIM = solver.buildStaticCondensation(K, "KIM");
        Eigen::MatrixXd KMI = solver.buildStaticCondensation(K, "KMI");
        Eigen::MatrixXd KMM = solver.buildStaticCondensation(K, "KMM");
        
        Eigen::VectorXd q_vec = Eigen::VectorXd::Zero(2);
        q_vec(0) = q_samples(i);
        q_vec(1) = q_samples(i);
        solver.setDistributedLoad(q_vec, elements);
        
        Eigen::VectorXd R = solver.buildGlobalDistributedLoad();
        Eigen::VectorXd RI = solver.buildStaticDistVector(R, "RI");
        Eigen::VectorXd RM = solver.buildStaticDistVector(R, "RM");
        
        solver.setSupp(supp);
        
        Eigen::MatrixXd K_ = KII - KIM * KMM.inverse() * KMI;
        Eigen::VectorXd R_ = RI - KIM * KMM.inverse() * RM;
        
        K_ = solver.applyDBCMatrix(K_);
        R_ = solver.applyDBCVec(R_);
        
        Eigen::VectorXd uh = K_.ldlt().solve(R_);
        
        // --- Salvar resultados (deslocamento) ---
        std::stringstream filename;
        filename << "output/ml_dataset_100k/sample_" 
                << std::setfill('0') << std::setw(4) << i << ".csv";

        std::ofstream file(filename.str());
        file << "node_id,x,displacement,rotation\n";

        for(int j = 0; j < nodes.rows(); ++j){
            double disp = uh(j*2);
            double rot = uh(j*2 + 1);
            file << j << "," 
                 << nodes(j, 0) << "," 
                 << disp << ","
                 << rot << "\n";
        }
        file.close();

        // ============================================================
        // NOVA SEÇÃO: Extrair deslocamentos e rotações nos pontos específicos
        // ============================================================
        csv_file << i << "," 
                 << E_samples(i) << "," 
                 << I_samples(i) << "," 
                 << q_samples(i);
        
        // Deslocamentos
        for(int node_idx : target_nodes){
            csv_file << "," << uh(node_idx * 2);
        }
        
        // Rotações
        for(int node_idx : target_nodes){
            csv_file << "," << uh(node_idx * 2 + 1);
        }
        
        csv_file << "\n";
        // ============================================================
    }
    
    csv_file.close();
    
    std::cout << "\n✅ Dataset generated successfully!" << std::endl;
    std::cout << "Location: output/ml_dataset_100k/" << std::endl;
    std::cout << "Files: " << numSamples << " displacement CSVs + parameters.csv" << std::endl;
}

int main(){
    generateDatasetForML();
    return 0;
}