"""
Script para consolidar todos os arquivos progress.csv dos experimentos Ray Tune/ASHA
em um único arquivo CSV, incluindo os hiperparâmetros do params.json como colunas.

Uso: Coloque este script na pasta raiz onde estão as subpastas dos experimentos
     e execute: python consolidar_experimentos.py
"""

import os
import json
import pandas as pd
from pathlib import Path


def encontrar_experimentos(pasta_raiz: Path) -> list[Path]:
    """
    Encontra todas as subpastas que contêm progress.csv e params.json.
    """
    experimentos = []
    
    for pasta in pasta_raiz.rglob("*"):
        if pasta.is_dir():
            progress_csv = pasta / "progress.csv"
            params_json = pasta / "params.json"
            
            if progress_csv.exists() and params_json.exists():
                experimentos.append(pasta)
    
    return experimentos


def carregar_params(params_path: Path) -> dict:
    """
    Carrega os parâmetros do arquivo JSON.
    """
    with open(params_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def processar_experimento(pasta_experimento: Path) -> pd.DataFrame | None:
    """
    Processa um único experimento: lê o CSV, adiciona os parâmetros como colunas,
    e adiciona uma coluna identificadora com o nome da pasta.
    """
    progress_csv = pasta_experimento / "progress.csv"
    params_json = pasta_experimento / "params.json"
    
    try:
        # Carrega o CSV
        df = pd.read_csv(progress_csv)
        
        if df.empty:
            print(f"  ⚠️  CSV vazio em: {pasta_experimento.name}")
            return None
        
        # Carrega os parâmetros
        params = carregar_params(params_json)
        
        # Adiciona cada parâmetro como uma coluna (mesmo valor em todas as linhas)
        for param_nome, param_valor in params.items():
            df[f"param_{param_nome}"] = param_valor
        
        # Adiciona coluna com o nome da pasta do experimento para identificação
        df["experiment_folder"] = pasta_experimento.name
        
        return df
    
    except Exception as e:
        print(f"  ❌ Erro ao processar {pasta_experimento.name}: {e}")
        return None


def consolidar_experimentos(pasta_raiz: Path, nome_saida: str = "consolidated_experiments.csv"):
    """
    Função principal que consolida todos os experimentos em um único CSV.
    """
    print("=" * 60)
    print("CONSOLIDADOR DE EXPERIMENTOS RAY TUNE / ASHA")
    print("=" * 60)
    
    # Encontra todos os experimentos
    print("\n🔍 Buscando experimentos...")
    experimentos = encontrar_experimentos(pasta_raiz)
    print(f"   Encontrados: {len(experimentos)} experimentos\n")
    
    if not experimentos:
        print("❌ Nenhum experimento encontrado!")
        print("   Certifique-se de que existem subpastas com progress.csv e params.json")
        return
    
    # Processa cada experimento
    print("📊 Processando experimentos...")
    dfs = []
    
    for i, pasta in enumerate(experimentos, 1):
        print(f"   [{i}/{len(experimentos)}] {pasta.name}")
        df = processar_experimento(pasta)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print("\n❌ Nenhum experimento foi processado com sucesso!")
        return
    
    # Concatena todos os DataFrames
    # O pandas automaticamente preenche com NaN onde colunas não existem
    print("\n🔗 Concatenando todos os dados...")
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Salva o arquivo consolidado
    caminho_saida = pasta_raiz / nome_saida
    df_final.to_csv(caminho_saida, index=False)
    
    # Estatísticas finais
    print("\n" + "=" * 60)
    print("✅ CONSOLIDAÇÃO CONCLUÍDA!")
    print("=" * 60)
    print(f"\n📁 Arquivo salvo em: {caminho_saida}")
    print(f"\n📈 Estatísticas:")
    print(f"   • Experimentos processados: {len(dfs)}")
    print(f"   • Total de linhas: {len(df_final):,}")
    print(f"   • Total de colunas: {len(df_final.columns)}")
    print(f"\n📋 Colunas no arquivo final:")
    
    # Separa colunas de parâmetros das outras
    param_cols = [c for c in df_final.columns if c.startswith("param_")]
    other_cols = [c for c in df_final.columns if not c.startswith("param_") and c != "experiment_folder"]
    
    print(f"\n   Colunas do progress.csv ({len(other_cols)}):")
    for col in other_cols:
        print(f"      - {col}")
    
    print(f"\n   Colunas de parâmetros ({len(param_cols)}):")
    for col in param_cols:
        print(f"      - {col}")
    
    print(f"\n   Coluna de identificação:")
    print(f"      - experiment_folder")


if __name__ == "__main__":
    # Obtém o diretório onde o script está localizado
    pasta_script = Path(__file__).parent.resolve()
    
    print(f"\n📂 Pasta raiz: {pasta_script}\n")
    
    # Executa a consolidação
    consolidar_experimentos(pasta_script)