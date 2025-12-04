import json
import os
from pathlib import Path

def calcular_e_salvar_lambda_fct():
    # 1. Obter o diretório base do script atual (para portabilidade)
    # Assume que o script está em '.../tcc-vem-deep-learning/00_FINAL_SCRIPTS/'
    diretorio_script = Path(__file__).resolve().parent 
    diretorio_base = diretorio_script.parent
    
    # 2. Construir o caminho de destino
    # Modifiquei para buscar a pasta 'meshes_2500_samples' (seu caminho original)
    # ou 'meshes_5_samples' (o que apareceu no seu log de erro anterior)
    
    # *** VERIFIQUE QUAL PASTA VOCÊ QUER PROCESSAR AGORA ***
    pasta_alvo_relativa = Path("00_URGENTE") / "malha" / "training_dataset_json" / "meshes_2500_samples" 
    
    # Caminho completo
    pasta_destino = diretorio_base / pasta_alvo_relativa
    
    # Verificar se a pasta existe
    if not pasta_destino.is_dir():
        print(f"Erro: O diretório de destino não foi encontrado em: {pasta_destino}")
        # Se a pasta 'meshes_2500_samples' não existir, tente a 'meshes_5_samples'
        pasta_destino = diretorio_base / Path("00_URGENTE") / "malha" / "training_dataset_json" / "meshes_2500_samples"
        if not pasta_destino.is_dir():
            print(f"Erro: Nenhuma das pastas de destino (2500 ou 5 samples) foi encontrada.")
            return

    print(f"Processando arquivos em: {pasta_destino}")
    
    # 3. Iterar sobre os arquivos na pasta destino
    for caminho_completo in pasta_destino.glob("*.json"):
        nome_arquivo = caminho_completo.name
        
        try:
            with open(caminho_completo, 'r') as f:
                dados = json.load(f)

            # --- Lógica de Cálculo (CORRIGIDA) ---
            # Acessar R e H no nível superior
            lambda_fct = dados.get('lambda_fct')
            
            # Verificar se os dados são válidos
            if lambda_fct is not None:
                k_t = (3 - (3.13 * lambda_fct) + (3.66 * (lambda_fct ** 2)) - (1.53 * (lambda_fct ** 3))) / (1 - lambda_fct)
                
                # Salvar lambda_fct no nível superior
                dados['k_t'] = k_t
                print(f"Calculado FCT={k_t:.4f} para {nome_arquivo}")

                # Salvar o arquivo atualizado (mantendo a estrutura e indentação)
                with open(caminho_completo, 'w') as f:
                    # Usamos sort_keys=False para tentar manter a ordem original
                    json.dump(dados, f, indent=4, sort_keys=False)
            
            else:
                print(f"Aviso: R  ou H  inválido(s) ou zero em {nome_arquivo}. Cálculo ignorado.")
            # ----------------------------------------------------------------

        except Exception as e:
            print(f"Erro ao processar {nome_arquivo}: {e}")

# Execução
calcular_e_salvar_lambda_fct()