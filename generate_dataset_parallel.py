import os
import glob
import subprocess
from multiprocessing import Pool, cpu_count
import time
import sys

# --- CONFIGURAÇÃO ---
# 1. Caminho do Executável C++
EXECUTAVEL = "./build/bin/run_single_case_2D"

# 2. Definição das Pastas (Ajuste o número de samples aqui)
SAMPLES = "500" # Mude para 100, 1000, etc.

BASE_PATH = "/Users/rafaelflacerda/Desktop/tcc-vem-deep-learning/00_URGENTE/malha"
INPUT_DIR = os.path.join(BASE_PATH, "parameters", f"meshes_{SAMPLES}_samples")
OUTPUT_DIR = os.path.join(BASE_PATH, "training_dataset", f"meshes_{SAMPLES}_samples")

def processar_caso(arquivo_input):
    """
    Função executada por cada núcleo da CPU.
    Recebe o arquivo de entrada, define o de saída e chama o C++.
    """
    # Pega o nome do arquivo (ex: mesh_data_0.json)
    nome_arq = os.path.basename(arquivo_input)
    
    # Define o nome do arquivo de saída (ex: result_data_0.json)
    # Trocamos 'mesh' por 'result' para diferenciar
    nome_saida = nome_arq.replace("mesh_data", "result_data")
    
    # Cria o caminho completo de saída
    arquivo_output = os.path.join(OUTPUT_DIR, nome_saida)
    
    try:
        # Print de início
        # print(f"[PROCESSANDO] {nome_arq}...") # Comentei para poluir menos se forem muitos
        
        start = time.time()
        
        # --- AQUI ESTÁ A MUDANÇA PRINCIPAL ---
        # Passamos 3 argumentos para o C++: [executável, input, output]
        cmd = [EXECUTAVEL, arquivo_input, arquivo_output]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        end = time.time()
        
        if result.returncode == 0:
            tempo = end - start
            # Print de sucesso
            print(f"[OK] {nome_saida} gerado em {tempo:.4f}s")
            sys.stdout.flush()
            return True
        else:
            # Print de erro do C++
            print(f"[ERRO C++] {nome_arq}: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERRO PYTHON] {nome_arq}: {e}")
        return False

def main():
    # 1. Garantir que a pasta de saída existe
    if not os.path.exists(OUTPUT_DIR):
        print(f"Criando pasta de saída: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Pegar arquivos de entrada
    if not os.path.exists(INPUT_DIR):
        print(f"Erro: Pasta de entrada não encontrada: {INPUT_DIR}")
        return

    arquivos = glob.glob(os.path.join(INPUT_DIR, "mesh_data_*.json"))
    arquivos.sort() # Opcional, só para organizar a lista interna
    total_arquivos = len(arquivos)
    
    if total_arquivos == 0:
        print("Nenhum arquivo .json encontrado na entrada.")
        return

    print(f"--- INICIANDO GERAÇÃO DE DATASET ---")
    print(f"Entrada: {INPUT_DIR}")
    print(f"Saída:   {OUTPUT_DIR}")
    print(f"Total de casos: {total_arquivos}")

    # 3. Configurar Paralelismo
    # Usa quase todos os núcleos (deixa 2 livres para o sistema respirar)
    num_workers = max(1, cpu_count() - 2) 
    print(f"Utilizando {num_workers} processos paralelos.")
    
    start_total = time.time()
    
    # 4. Disparar o Processamento
    with Pool(num_workers) as pool:
        pool.map(processar_caso, arquivos)
        
    end_total = time.time()
    tempo_total = end_total - start_total
    
    print("\n" + "="*40)
    print(f" CONCLUÍDO!")
    print(f" Tempo Total: {tempo_total:.2f} s")
    print(f" Média: {(tempo_total/total_arquivos):.4f} s/caso")
    print("="*40)

if __name__ == "__main__":
    main()