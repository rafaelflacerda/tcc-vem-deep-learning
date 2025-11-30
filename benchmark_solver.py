import os
import glob
import subprocess
from multiprocessing import Pool, cpu_count
import time
import sys

# --- CONFIGURAÇÃO ---
# Caminho exato onde você compilou seu C++
EXECUTAVEL = "./build/bin/run_single_case_2D"

# Caminho onde estão os JSONs de teste (10 amostras)
PASTA_DATASET = "/Users/rafaelflacerda/Desktop/tcc-vem-deep-learning/00_URGENTE/malha/parameters/meshes_10_samples"

def processar_caso(arquivo_json):
    """
    Roda um caso individual.
    """
    # Adicionado para usar no print
    nome_arq = os.path.basename(arquivo_json)
    
    try:
        # Adicionado: Print de início com flush para aparecer na hora
        print(f"[INICIANDO] {nome_arq}...")
        sys.stdout.flush()

        # Roda o C++ passando o arquivo como argumento
        # capture_output=True esconde o print do FCT para não bagunçar o terminal
        start = time.time()
        result = subprocess.run([EXECUTAVEL, arquivo_json], capture_output=True, text=True)
        end = time.time()
        
        if result.returncode == 0:
            tempo = end - start
            # Adicionado: Print de conclusão
            print(f"[CONCLUÍDO] {nome_arq} em {tempo:.4f}s")
            sys.stdout.flush()
            return tempo # Retorna o tempo que levou
        else:
            print(f"Erro no C++: {result.stderr}")
            return None
    except Exception as e:
        print(f"Erro Python: {e}")
        return None

def main():
    # 1. Pegar arquivos
    arquivos = glob.glob(os.path.join(PASTA_DATASET, "mesh_data_*.json"))
    arquivos.sort()
    total_arquivos = len(arquivos)
    print(f"Encontrados {total_arquivos} arquivos para teste.")

    # 2. Benchmark Serial (1 Núcleo)
    print("\n--- INICIANDO TESTE SERIAL (1 POR VEZ) ---")
    start_serial = time.time()
    for arq in arquivos:
        processar_caso(arq)
    end_serial = time.time()
    tempo_serial = end_serial - start_serial
    print(f"Tempo Total Serial: {tempo_serial:.4f} s")

    # 3. Benchmark Paralelo (Multiprocessing)
    # Usando 10 workers (pois temos 10 arquivos e seu mac aguenta)
    num_workers = min(10, cpu_count()) 
    print(f"\n--- INICIANDO TESTE PARALELO ({num_workers} WORKERS) ---")
    
    start_paralelo = time.time()
    with Pool(num_workers) as pool:
        pool.map(processar_caso, arquivos)
    end_paralelo = time.time()
    tempo_paralelo = end_paralelo - start_paralelo
    
    print(f"Tempo Total Paralelo: {tempo_paralelo:.4f} s")
    
    # 4. Conclusão
    speedup = tempo_serial / tempo_paralelo
    print(f"\n>>> SPEEDUP: {speedup:.2f}x mais rápido")

if __name__ == "__main__":
    main()