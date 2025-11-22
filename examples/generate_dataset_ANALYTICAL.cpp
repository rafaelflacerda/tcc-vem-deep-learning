#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <cmath>
#include <Eigen/Dense>

namespace fs = std::filesystem;

// Função analítica para deslocamento: viga engastada sob carga distribuída uniforme
inline double w_analitico(double x, double q, double E, double I, double L) {
    return (q * pow(x, 2) * (6*pow(L, 2) - 4*L*x + pow(x, 2))) / (24.0 * E * I);
}

// Gera ruído entre 0.97 e 1.03 (excluindo 1.0)
inline double gerar_ruido() {
    double noise = 0.97 + 0.06 * ((double)rand() / RAND_MAX); // [0.97, 1.03]
    
    // Se cair muito perto de 1.0, regenerar
    while (noise > 0.99 && noise < 1.01) {
        noise = 0.97 + 0.06 * ((double)rand() / RAND_MAX);
    }
    
    return noise;
}

int main() {
    std::cout << "=== Gerando Dataset Analítico (Apenas Deslocamentos) ===" << std::endl;

    srand(static_cast<unsigned>(time(nullptr)));

    // Parâmetros
    double L = 1.0;
    int numElements = 40;
    int numSamples = 1000;

    double E_min = 50e9,  E_max = 250e9;
    double I_min = 1e-8,  I_max = 1e-5;
    double q_min = 10, q_max = 1000;

    // Criar pasta
    fs::create_directories("output/generated_dataset_analytical");

    // ========== ARQUIVO 1: TREINAMENTO (5 pontos) ==========
    std::ofstream csv_training("output/generated_dataset_analytical/training_data.csv");
    csv_training << "sample_id,E,I,q,w_0.2L,w_0.4L,w_0.6L,w_0.8L,w_1.0L\n";
    csv_training << std::scientific << std::setprecision(10);

    // ========== ARQUIVO 2: VALIDAÇÃO (todos os nós) ==========
    std::ofstream csv_validation("output/generated_dataset_analytical/full_validation_data.csv");
    csv_validation << "sample_id,x_position,E,I,q,w\n";
    csv_validation << std::scientific << std::setprecision(10);

    // Posições normalizadas para treinamento
    double training_positions[5] = {0.2, 0.4, 0.6, 0.8, 1.0};

    int successCount = 0;
    long int training_points = 0;
    long int validation_points = 0;

    for (int i = 0; i < numSamples; ++i) {
        double r1 = (double)rand() / RAND_MAX;
        double r2 = (double)rand() / RAND_MAX;
        double r3 = (double)rand() / RAND_MAX;

        double E = E_min + r1 * (E_max - E_min);
        double I = I_min + r2 * (I_max - I_min);
        double q = -(q_min + r3 * (q_max - q_min));

        // ==================== TREINAMENTO ====================
        csv_training << i << "," << E << "," << I << "," << q;

        for (int p = 0; p < 5; p++) {
            double x = training_positions[p] * L;
            double noise_factor = gerar_ruido();
            double w = w_analitico(x, q, E, I, L) * noise_factor;
            csv_training << "," << w;
        }
        csv_training << "\n";
        training_points++;

        // ==================== VALIDAÇÃO ====================
        double dx = L / numElements;
        for (int j = 0; j <= numElements; j++) {
            double x = j * dx;
            double noise_factor = gerar_ruido();
            double w = w_analitico(x, q, E, I, L) * noise_factor;

            csv_validation << i << "," << x << "," << E << "," << I << "," << q << "," << w << "\n";
            validation_points++;
        }

        successCount++;

        if ((i + 1) % 100 == 0) {
            std::cout << "Processadas " << (i + 1) << "/" << numSamples 
                      << " amostras..." << std::endl;
        }
    }

    csv_training.close();
    csv_validation.close();

    std::cout << "\n✅ Datasets analíticos gerados com sucesso!\n";
    std::cout << "   📊 TREINAMENTO (training_data.csv):\n";
    std::cout << "      - Linhas: " << training_points << "\n";
    std::cout << "      - Pontos por amostra: 5 (0.2L, 0.4L, 0.6L, 0.8L, 1.0L)\n";
    std::cout << "      - Colunas: sample_id, E, I, q, w_0.2L, w_0.4L, w_0.6L, w_0.8L, w_1.0L\n";
    std::cout << "\n   📊 VALIDAÇÃO (full_validation_data.csv):\n";
    std::cout << "      - Linhas: " << validation_points << "\n";
    std::cout << "      - Pontos por amostra: " << (validation_points / successCount) << " nós\n";
    std::cout << "      - Colunas: sample_id, x_position, E, I, q, w\n";
    std::cout << "\n   ✓ Amostras bem-sucedidas: " << successCount << "\n";
    std::cout << "\n📝 Ruído aplicado: [0.97, 1.03] excluindo [0.99, 1.01]\n";

    return 0;
}