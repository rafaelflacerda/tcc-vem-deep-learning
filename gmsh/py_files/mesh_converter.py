import numpy as np

def parse_gmsh_msh(filepath):
    """
    Parseia um arquivo .msh do Gmsh (formato 4.1) e extrai nodes e elements.
    
    Retorna:
    --------
    nodes : np.ndarray (N_nós × 2)
        Coordenadas (x, y) de cada nó
    elements : np.ndarray (N_elem × 3)
        Índices dos nós de cada elemento triangular (zero-based)
    """
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Remover whitespace e linhas vazias
    lines = [line.strip() for line in lines]
    
    nodes_dict = {}  # {node_tag: (x, y)}
    elements_list = []  # [(n1, n2, n3), ...]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # =====================================================================
        # PARSEAR NÓS
        # =====================================================================
        if line == '$Nodes':
            i += 1
            # Linha de cabeçalho: numEntityBlocks numNodes minNodeTag maxNodeTag
            header = lines[i].split()
            num_entity_blocks = int(header[0])
            total_nodes = int(header[1])
            i += 1
            
            for _ in range(num_entity_blocks):
                # Linha do bloco: entityDim entityTag parametric numNodesInBlock
                block_header = lines[i].split()
                num_nodes_in_block = int(block_header[3])
                i += 1
                
                # Ler os tags dos nós
                node_tags = []
                for _ in range(num_nodes_in_block):
                    node_tags.append(int(lines[i]))
                    i += 1
                
                # Ler as coordenadas
                for tag in node_tags:
                    coords = lines[i].split()
                    x = float(coords[0])
                    y = float(coords[1])
                    # z = float(coords[2])  # ignoramos z para 2D
                    nodes_dict[tag] = (x, y)
                    i += 1
            
            # Pular $EndNodes
            i += 1
            continue
        
        # =====================================================================
        # PARSEAR ELEMENTOS
        # =====================================================================
        if line == '$Elements':
            i += 1
            # Linha de cabeçalho: numEntityBlocks numElements minElementTag maxElementTag
            header = lines[i].split()
            num_entity_blocks = int(header[0])
            i += 1
            
            for _ in range(num_entity_blocks):
                # Linha do bloco: entityDim entityTag elementType numElementsInBlock
                block_header = lines[i].split()
                entity_dim = int(block_header[0])
                element_type = int(block_header[2])
                num_elements_in_block = int(block_header[3])
                i += 1
                
                # Só queremos elementos 2D (entityDim == 2)
                # elementType == 2 significa triângulo de 3 nós
                if entity_dim == 2 and element_type == 2:
                    for _ in range(num_elements_in_block):
                        elem_data = lines[i].split()
                        # elem_data[0] é o tag do elemento, [1:] são os nós
                        node_indices = [int(x) for x in elem_data[1:]]
                        elements_list.append(node_indices)
                        i += 1
                else:
                    # Pular elementos que não são 2D (linhas, pontos)
                    for _ in range(num_elements_in_block):
                        i += 1
            
            # Pular $EndElements
            i += 1
            continue
        
        i += 1
    
    # =========================================================================
    # CONVERTER PARA FORMATO DO SOLVER
    # =========================================================================
    
    # Criar mapeamento de tags antigos para índices novos (zero-based)
    sorted_tags = sorted(nodes_dict.keys())
    tag_to_index = {tag: idx for idx, tag in enumerate(sorted_tags)}
    
    # Criar matriz de nós
    num_nodes = len(sorted_tags)
    nodes = np.zeros((num_nodes, 2))
    for tag in sorted_tags:
        idx = tag_to_index[tag]
        nodes[idx, 0] = nodes_dict[tag][0]
        nodes[idx, 1] = nodes_dict[tag][1]
    
    # Criar matriz de elementos (convertendo tags para índices zero-based)
    num_elements = len(elements_list)
    elements = np.zeros((num_elements, 3), dtype=int)
    for i, elem in enumerate(elements_list):
        for j, node_tag in enumerate(elem):
            elements[i, j] = tag_to_index[node_tag]
    
    return nodes, elements


def check_and_fix_orientation(nodes, elements):
    """
    Verifica e corrige a orientação dos elementos para anti-horário.
    """
    fixed_count = 0
    
    for i in range(len(elements)):
        n0, n1, n2 = elements[i]
        
        # Calcular área com sinal (shoelace formula)
        x0, y0 = nodes[n0]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        
        signed_area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        
        # Se área negativa, está no sentido horário - inverter
        if signed_area < 0:
            elements[i] = [n0, n2, n1]  # inverte n1 e n2
            fixed_count += 1
    
    if fixed_count > 0:
        print(f"Corrigida orientação de {fixed_count} elementos")
    
    return elements


# =============================================================================
# EXECUTAR
# =============================================================================

# Caminho para o arquivo .msh
filepath = 'minha_malha_overkill.msh'  # <-- ALTERE PARA O NOME DO SEU ARQUIVO

# Parsear o arquivo
nodes, elements = parse_gmsh_msh(filepath)

# Verificar e corrigir orientação
elements = check_and_fix_orientation(nodes, elements)

# =============================================================================
# EXIBIR INFORMAÇÕES
# =============================================================================

print("=" * 60)
print("MALHA EXTRAÍDA DO ARQUIVO .MSH")
print("=" * 60)
print(f"Número de nós:      {len(nodes)}")
print(f"Número de elementos: {len(elements)}")
print(f"Tipo de elemento:    Triângulo (3 nós)")
print()

print("Primeiros 5 nós:")
print("  Índice |     x      |     y     ")
print("  -------|------------|------------")
for i in range(min(5, len(nodes))):
    print(f"    {i:3d}  | {nodes[i,0]:10.6f} | {nodes[i,1]:10.6f}")

print()
print("Primeiros 5 elementos:")
print("  Elem | Nó0 | Nó1 | Nó2 |")
print("  -----|-----|-----|-----|")
for i in range(min(5, len(elements))):
    e = elements[i]
    print(f"   {i:2d}  | {e[0]:3d} | {e[1]:3d} | {e[2]:3d} |")

# =============================================================================
# EXPORTAR PARA CSV
# =============================================================================

np.savetxt('nodes.csv', nodes, delimiter=',', fmt='%.10f',
           header='x,y', comments='')

np.savetxt('elements.csv', elements, delimiter=',', fmt='%d',
           header='n0,n1,n2', comments='')

print()
print("=" * 60)
print("ARQUIVOS EXPORTADOS:")
print("  nodes.csv    - Coordenadas dos nós")
print("  elements.csv - Conectividade dos elementos")
print("=" * 60)