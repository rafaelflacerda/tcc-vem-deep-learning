/**
 * @file test_vem_2d_simple.cpp
 * @brief Teste simples do VEM 2D - só rodar e ver resultados
 */

#include <iostream>
#include <iomanip>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "solver/linearElastic2d.hpp"
#include "material/mat.hpp"
#include "mesh/uniform.hpp"

int main() {
    std::cout << "\n=== TESTE SIMPLES VEM 2D ===\n" << std::endl;
    
    // ========================================================================
    // PARÂMETROS
    // ========================================================================
    const double L = 1.0;       // Comprimento [m]
    const double H = 0.1;       // Altura [m]
    const double E = 200e9;     // Módulo de Young [Pa]
    const double nu = 0.3;      // Poisson
    const double q = -1000.0;   // Carga [N/m]
    const int n = 50;           // Malha 10×10
    
    std::cout << "Geometria: L=" << L << "m, H=" << H << "m" << std::endl;
    std::cout << "Material: E=" << E/1e9 << " GPa, ν=" << nu << std::endl;
    std::cout << "Carga: q=" << q << " N/m" << std::endl;
    std::cout << "Malha: " << n << "×" << n << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // GERAR MALHA
    // ========================================================================
    std::cout << "Gerando malha..." << std::endl;
    
    mesh::uniform meshGen(false);  // debug=false
    Eigen::MatrixXd nodes;
    Eigen::MatrixXi elements;
    
    meshGen.create_square_nxn_mesh(nodes, elements, n);
    
    // Escalar para dimensões reais
    nodes.col(0) *= L;
    nodes.col(1) *= H;
    
    std::cout << "✓ " << nodes.rows() << " nós, " << elements.rows() << " elementos\n" << std::endl;
    
    // ========================================================================
    // CONFIGURAR SOLVER
    // ========================================================================
    std::cout << "Configurando solver..." << std::endl;
    
    solver::linearElastic2d vem(nodes, elements, 1);
    
    // Material
    material::mat elastic;
    elastic.setElasticModule(E);
    elastic.setPoissonCoef(nu);
    Eigen::MatrixXd C = elastic.build2DElasticity();
    
    // Engaste em x=0
    Eigen::MatrixXi bc(n+1, 3);  // 21 nós na borda x=0 para malha 10×10
    int bc_count = 0;
    for (int i = 0; i < nodes.rows(); i++) {
        if (std::abs(nodes(i, 0)) < 1e-6) {
            bc(bc_count, 0) = i;
            bc(bc_count, 1) = 1;  // u=0
            bc(bc_count, 2) = 1;  // v=0
            bc_count++;
        }
    }
    vem.setSupp(bc);
    
    // Carga no topo (y=H)
    Eigen::MatrixXi load(n, 2);  // 10 arestas no topo para malha 10×10
    int load_count = 0;
    for (int i = 0; i < nodes.rows(); i++) {
        if (std::abs(nodes(i, 1) - H) < 1e-6) {
            for (int j = i+1; j < nodes.rows(); j++) {
                if (std::abs(nodes(j, 1) - H) < 1e-6 && 
                    std::abs(nodes(j, 0) - nodes(i, 0) - L/n) < 1e-6) {
                    load(load_count, 0) = i;
                    load(load_count, 1) = j;
                    load_count++;
                    break;
                }
            }
        }
    }
    vem.setLoad(load);
    
    std::cout << "✓ Engaste: " << bc_count << " nós" << std::endl;
    std::cout << "✓ Carga: " << load_count << " arestas\n" << std::endl;
    
    // ========================================================================
    // RESOLVER
    // ========================================================================
    std::cout << "Resolvendo sistema..." << std::endl;
    
    Eigen::MatrixXd K = vem.buildGlobalK(C);
    K = vem.applyDBC(K);
    Eigen::VectorXd f = vem.applyNBC(0.0, q);
    Eigen::VectorXd uh = K.ldlt().solve(f);
    
    std::cout << "✓ Sistema " << K.rows() << "×" << K.cols() << " resolvido\n" << std::endl;
    
    // ========================================================================
    // MOSTRAR RESULTADOS
    // ========================================================================
    std::cout << "=== DESLOCAMENTOS (alguns pontos) ===\n" << std::endl;
    std::cout << std::setw(12) << "x [m]" 
              << std::setw(12) << "y [m]"
              << std::setw(15) << "u [m]"
              << std::setw(15) << "v [m]" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    // Mostrar alguns pontos interessantes
    for (int i = 0; i < nodes.rows(); i++) {
        double x = nodes(i, 0);
        double y = nodes(i, 1);
        
        // Apenas pontos no eixo neutro (y ≈ H/2) ou nas extremidades
        bool show = false;
        if (std::abs(y - H/2.0) < 1e-6) show = true;  // Eixo neutro
        if (std::abs(x - L) < 1e-6 && std::abs(y - H/2.0) < 1e-6) show = true;  // Ponta
        
        if (show) {
            double u = uh(2*i);
            double v = uh(2*i + 1);
            
            std::cout << std::fixed << std::setprecision(4)
                      << std::setw(12) << x
                      << std::setw(12) << y
                      << std::scientific << std::setprecision(3)
                      << std::setw(15) << u
                      << std::setw(15) << v << std::endl;
        }
    }
    
    // ========================================================================
    // VALORES EXTREMOS
    // ========================================================================
    std::cout << "\n=== VALORES EXTREMOS ===" << std::endl;
    
    double max_u = 0.0, max_v = 0.0;
    double min_u = 0.0, min_v = 0.0;
    
    for (int i = 0; i < nodes.rows(); i++) {
        double u = uh(2*i);
        double v = uh(2*i + 1);
        
        if (u > max_u) max_u = u;
        if (u < min_u) min_u = u;
        if (v > max_v) max_v = v;
        if (v < min_v) min_v = v;
    }
    
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "u_max: " << max_u << " m" << std::endl;
    std::cout << "u_min: " << min_u << " m" << std::endl;
    std::cout << "v_max: " << max_v << " m" << std::endl;
    std::cout << "v_min: " << min_v << " m (deflexão máxima)" << std::endl;
    
    std::cout << "\n✅ Teste concluído!\n" << std::endl;
    
    return 0;
}