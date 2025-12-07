#!/usr/bin/env python3
"""
NPZ Splitter - Divide arquivos .npz grandes em partes menores e reconstrói depois.

Útil para fazer upload de arquivos grandes no GitHub (limite de 100MB por arquivo).

Uso:
    Dividir um arquivo:    python npz_splitter.py split arquivo.npz --size 50
    Juntar um arquivo:     python npz_splitter.py merge arquivo.npz
    Dividir todos:         python npz_splitter.py split-all --size 50
    Juntar todos:          python npz_splitter.py merge-all
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


# =============================================================================
# CONFIGURAÇÃO DO PROJETO
# =============================================================================

# Obtém o diretório onde o script está localizado (funciona em qualquer dispositivo)
SCRIPT_DIR = Path(__file__).resolve().parent

# Diretório dos datasets (relativo ao script)
DATASET_DIR = SCRIPT_DIR / "dataset"

# Lista de n_samples dos datasets
DATASET_SAMPLES = [
    "10",
    "100",
    "250",
    "1000",
    "2500",
    "10000",
    "25000",
    "50000",
    "100000",
    "1000000",
]


def get_npz_path(n_samples: str) -> Path:
    """Retorna o path do arquivo NPZ original dado o número de samples."""
    return DATASET_DIR / f"beam_dataset_{n_samples}_samples_preproc.npz"


def get_chunks_dir(n_samples: str) -> Path:
    """Retorna o diretório onde os chunks serão salvos."""
    return DATASET_DIR / f"{n_samples}_samples"


def extract_n_samples(filename: str) -> str:
    """Extrai o número de samples do nome do arquivo."""
    match = re.search(r"beam_dataset_(\d+)_samples_preproc", filename)
    if match:
        return match.group(1)
    return None


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calculate_checksum(filepath: str) -> str:
    """Calcula MD5 checksum do arquivo para verificação de integridade."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# =============================================================================
# FUNÇÕES PRINCIPAIS
# =============================================================================

def split_file(filepath: str, chunk_size_mb: int = 50, output_dir: str = None):
    """
    Divide um arquivo .npz em múltiplos chunks menores.
    
    Args:
        filepath: Caminho do arquivo .npz original
        chunk_size_mb: Tamanho máximo de cada chunk em MB (default: 50MB)
        output_dir: Diretório de saída (default: mesmo do arquivo original)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    if not filepath.suffix == '.npz':
        print(f"⚠️  Aviso: O arquivo não tem extensão .npz, mas vou processar mesmo assim.")
    
    # Configurar diretório de saída
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = filepath.parent
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Nome base para os chunks
    base_name = filepath.stem
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    
    # Ler arquivo e dividir
    file_size = filepath.stat().st_size
    original_checksum = calculate_checksum(filepath)
    
    print(f"📁 Arquivo: {filepath.name}")
    print(f"📊 Tamanho: {file_size / (1024*1024):.2f} MB")
    print(f"✂️  Tamanho do chunk: {chunk_size_mb} MB")
    
    chunks_info = []
    chunk_index = 0
    
    with open(filepath, "rb") as f:
        while True:
            chunk_data = f.read(chunk_size_bytes)
            if not chunk_data:
                break
            
            chunk_filename = f"{base_name}.part{chunk_index:03d}"
            chunk_path = output_path / chunk_filename
            
            with open(chunk_path, "wb") as chunk_file:
                chunk_file.write(chunk_data)
            
            chunk_checksum = calculate_checksum(chunk_path)
            chunks_info.append({
                "index": chunk_index,
                "filename": chunk_filename,
                "size": len(chunk_data),
                "checksum": chunk_checksum
            })
            
            print(f"  ✅ Criado: {chunk_filename} ({len(chunk_data) / (1024*1024):.2f} MB)")
            chunk_index += 1
    
    # Salvar metadados para reconstrução
    metadata = {
        "original_filename": filepath.name,
        "original_size": file_size,
        "original_checksum": original_checksum,
        "chunk_size_mb": chunk_size_mb,
        "total_chunks": chunk_index,
        "chunks": chunks_info
    }
    
    metadata_path = output_path / f"{base_name}.manifest.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n🎉 Divisão completa!")
    print(f"   Total de chunks: {chunk_index}")
    print(f"   Manifesto salvo: {metadata_path.name}")
    print(f"\n💡 Para reconstruir, execute:")
    print(f"   python npz_splitter.py merge {metadata_path}")


def merge_files(manifest_path: str, output_dir: str = None, verify: bool = True):
    """
    Reconstrói o arquivo .npz original a partir dos chunks.
    
    Args:
        manifest_path: Caminho do arquivo .manifest.json
        output_dir: Diretório de saída (default: pasta dataset/)
        verify: Se True, verifica checksums após reconstrução
    """
    manifest_path = Path(manifest_path)
    
    # Aceita tanto o manifesto quanto o nome do arquivo original
    if manifest_path.suffix == '.npz':
        manifest_path = manifest_path.with_suffix('.manifest.json')
    elif not manifest_path.name.endswith('.manifest.json'):
        manifest_path = Path(str(manifest_path) + '.manifest.json')
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {manifest_path}")
    
    # Carregar metadados
    with open(manifest_path, "r") as f:
        metadata = json.load(f)
    
    chunks_dir = manifest_path.parent
    
    # Determinar diretório de saída
    if output_dir:
        output_path = Path(output_dir)
    else:
        # Por padrão, salva na pasta dataset/ (um nível acima dos chunks)
        output_path = chunks_dir.parent
    
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / metadata["original_filename"]
    
    print(f"🔧 Reconstruindo: {metadata['original_filename']}")
    print(f"   Chunks esperados: {metadata['total_chunks']}")
    print(f"   Destino: {output_file}")
    
    # Verificar se todos os chunks existem
    missing_chunks = []
    for chunk_info in metadata["chunks"]:
        chunk_path = chunks_dir / chunk_info["filename"]
        if not chunk_path.exists():
            missing_chunks.append(chunk_info["filename"])
    
    if missing_chunks:
        raise FileNotFoundError(
            f"Chunks faltando: {', '.join(missing_chunks)}\n"
            f"Certifique-se de que todos os arquivos .partXXX estão no mesmo diretório do manifesto."
        )
    
    # Verificar checksums dos chunks (opcional)
    if verify:
        print("   Verificando integridade dos chunks...")
        for chunk_info in metadata["chunks"]:
            chunk_path = chunks_dir / chunk_info["filename"]
            actual_checksum = calculate_checksum(chunk_path)
            if actual_checksum != chunk_info["checksum"]:
                raise ValueError(
                    f"Checksum inválido para {chunk_info['filename']}!\n"
                    f"Esperado: {chunk_info['checksum']}\n"
                    f"Encontrado: {actual_checksum}\n"
                    f"O arquivo pode estar corrompido."
                )
        print("   ✅ Todos os chunks verificados!")
    
    # Reconstruir arquivo
    with open(output_file, "wb") as out_f:
        for chunk_info in sorted(metadata["chunks"], key=lambda x: x["index"]):
            chunk_path = chunks_dir / chunk_info["filename"]
            print(f"   📦 Processando: {chunk_info['filename']}")
            
            with open(chunk_path, "rb") as chunk_f:
                out_f.write(chunk_f.read())
    
    # Verificar arquivo final
    final_size = output_file.stat().st_size
    
    if final_size != metadata["original_size"]:
        raise ValueError(
            f"Tamanho do arquivo reconstruído não confere!\n"
            f"Esperado: {metadata['original_size']} bytes\n"
            f"Encontrado: {final_size} bytes"
        )
    
    if verify:
        print("   Verificando arquivo final...")
        final_checksum = calculate_checksum(output_file)
        if final_checksum != metadata["original_checksum"]:
            raise ValueError(
                f"Checksum do arquivo reconstruído não confere!\n"
                f"Esperado: {metadata['original_checksum']}\n"
                f"Encontrado: {final_checksum}"
            )
        print("   ✅ Arquivo verificado!")
    
    print(f"\n🎉 Reconstrução completa!")
    print(f"   Arquivo: {output_file}")
    print(f"   Tamanho: {final_size / (1024*1024):.2f} MB")


def clean_chunks(manifest_path: str):
    """Remove os chunks após reconstrução bem-sucedida."""
    manifest_path = Path(manifest_path)
    
    if manifest_path.suffix == '.npz':
        manifest_path = manifest_path.with_suffix('.manifest.json')
    elif not manifest_path.name.endswith('.manifest.json'):
        manifest_path = Path(str(manifest_path) + '.manifest.json')
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {manifest_path}")
    
    with open(manifest_path, "r") as f:
        metadata = json.load(f)
    
    chunks_dir = manifest_path.parent
    
    print(f"🗑️  Removendo chunks de: {metadata['original_filename']}")
    
    for chunk_info in metadata["chunks"]:
        chunk_path = chunks_dir / chunk_info["filename"]
        if chunk_path.exists():
            chunk_path.unlink()
            print(f"   Removido: {chunk_info['filename']}")
    
    manifest_path.unlink()
    print(f"   Removido: {manifest_path.name}")
    print("✅ Limpeza completa!")


# =============================================================================
# FUNÇÕES DE PROCESSAMENTO EM LOTE
# =============================================================================

def split_all(chunk_size_mb: int = 50):
    """Divide todos os datasets NPZ em chunks."""
    print("=" * 60)
    print("DIVIDINDO TODOS OS DATASETS")
    print(f"Diretório base: {SCRIPT_DIR}")
    print(f"Diretório datasets: {DATASET_DIR}")
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for n_samples in DATASET_SAMPLES:
        npz_path = get_npz_path(n_samples)
        chunks_dir = get_chunks_dir(n_samples)
        
        print(f"\n{'─' * 60}")
        print(f"📦 Dataset: {n_samples} samples")
        
        if not npz_path.exists():
            print(f"   ⏭️  Arquivo não encontrado, pulando: {npz_path.name}")
            skip_count += 1
            continue
        
        # Verifica se já foi dividido
        manifest_path = chunks_dir / f"{npz_path.stem}.manifest.json"
        if manifest_path.exists():
            print(f"   ⏭️  Já dividido anteriormente, pulando.")
            print(f"      (delete {manifest_path.name} para re-dividir)")
            skip_count += 1
            continue
        
        try:
            split_file(str(npz_path), chunk_size_mb, str(chunks_dir))
            success_count += 1
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            error_count += 1
    
    print(f"\n{'=' * 60}")
    print("RESUMO")
    print(f"   ✅ Sucesso: {success_count}")
    print(f"   ⏭️  Pulados: {skip_count}")
    print(f"   ❌ Erros: {error_count}")
    print("=" * 60)


def merge_all(verify: bool = True, clean: bool = False):
    """Reconstrói todos os datasets NPZ a partir dos chunks."""
    print("=" * 60)
    print("RECONSTRUINDO TODOS OS DATASETS")
    print(f"Diretório base: {SCRIPT_DIR}")
    print(f"Diretório datasets: {DATASET_DIR}")
    print("=" * 60)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for n_samples in DATASET_SAMPLES:
        npz_path = get_npz_path(n_samples)
        chunks_dir = get_chunks_dir(n_samples)
        manifest_path = chunks_dir / f"{npz_path.stem}.manifest.json"
        
        print(f"\n{'─' * 60}")
        print(f"📦 Dataset: {n_samples} samples")
        
        # Verifica se o NPZ já existe
        if npz_path.exists():
            print(f"   ⏭️  Arquivo já existe, pulando: {npz_path.name}")
            print(f"      (delete o arquivo para reconstruir)")
            skip_count += 1
            continue
        
        if not manifest_path.exists():
            print(f"   ⏭️  Manifesto não encontrado, pulando.")
            skip_count += 1
            continue
        
        try:
            # Output é a pasta dataset/ (um nível acima dos chunks)
            merge_files(str(manifest_path), str(DATASET_DIR), verify)
            success_count += 1
            
            if clean:
                clean_chunks(str(manifest_path))
                
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            error_count += 1
    
    print(f"\n{'=' * 60}")
    print("RESUMO")
    print(f"   ✅ Sucesso: {success_count}")
    print(f"   ⏭️  Pulados: {skip_count}")
    print(f"   ❌ Erros: {error_count}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Divide arquivos .npz grandes em partes menores para upload no GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  Dividir TODOS os datasets em chunks de 50MB:
    python npz_splitter.py split-all
    
  Dividir TODOS em chunks de 90MB:
    python npz_splitter.py split-all --size 90
    
  Reconstruir TODOS os datasets:
    python npz_splitter.py merge-all
    
  Reconstruir TODOS e limpar chunks:
    python npz_splitter.py merge-all --clean

  Dividir UM arquivo específico:
    python npz_splitter.py split dataset/beam_dataset_1000_samples_preproc.npz --size 50
    
  Reconstruir UM arquivo específico:
    python npz_splitter.py merge dataset/1000_samples/beam_dataset_1000_samples_preproc.manifest.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")
    
    # Subcomando: split
    split_parser = subparsers.add_parser("split", help="Divide UM arquivo em chunks")
    split_parser.add_argument("file", help="Arquivo .npz para dividir")
    split_parser.add_argument(
        "--size", "-s",
        type=int,
        default=50,
        help="Tamanho máximo de cada chunk em MB (default: 50)"
    )
    split_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Diretório de saída (default: mesmo do arquivo)"
    )
    
    # Subcomando: merge
    merge_parser = subparsers.add_parser("merge", help="Reconstrói UM arquivo a partir dos chunks")
    merge_parser.add_argument("manifest", help="Arquivo .manifest.json ou nome base do arquivo")
    merge_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Diretório de saída (default: pasta dataset/)"
    )
    merge_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Pula verificação de checksums (mais rápido, menos seguro)"
    )
    merge_parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove chunks e manifesto após reconstrução bem-sucedida"
    )
    
    # Subcomando: split-all
    split_all_parser = subparsers.add_parser("split-all", help="Divide TODOS os datasets em chunks")
    split_all_parser.add_argument(
        "--size", "-s",
        type=int,
        default=50,
        help="Tamanho máximo de cada chunk em MB (default: 50)"
    )
    
    # Subcomando: merge-all
    merge_all_parser = subparsers.add_parser("merge-all", help="Reconstrói TODOS os datasets")
    merge_all_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Pula verificação de checksums (mais rápido, menos seguro)"
    )
    merge_all_parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove chunks e manifesto após reconstrução bem-sucedida"
    )
    
    # Subcomando: clean
    clean_parser = subparsers.add_parser("clean", help="Remove chunks e manifesto")
    clean_parser.add_argument("manifest", help="Arquivo .manifest.json ou nome base")
    
    args = parser.parse_args()
    
    if args.command == "split":
        split_file(args.file, args.size, args.output)
    
    elif args.command == "merge":
        merge_files(args.manifest, args.output, verify=not args.no_verify)
        if args.clean:
            clean_chunks(args.manifest)
    
    elif args.command == "split-all":
        split_all(args.size)
    
    elif args.command == "merge-all":
        merge_all(verify=not args.no_verify, clean=args.clean)
    
    elif args.command == "clean":
        clean_chunks(args.manifest)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()