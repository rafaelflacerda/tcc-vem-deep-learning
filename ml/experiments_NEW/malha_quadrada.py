import numpy as np

def create_rectangular_mesh(length, height, nx, ny):
    """
    Cria uma malha estruturada de quadriláteros para uma viga retangular.
    
    Parâmetros:
    -----------
    length : float
        Comprimento da viga (direção x)
    height : float
        Altura da viga (direção y)
    nx : int
        Número de elementos na direção x
    ny : int
        Número de elementos na direção y
    
    Retorna:
    --------
    nodes : np.ndarray (N_nós × 2)
        Coordenadas dos nós
    elements : np.ndarray (N_elem × 4)
        Conectividade dos elementos (sentido anti-horário)
    """
    
    # Número de nós em cada direção
    nodes_x = nx + 1
    nodes_y = ny + 1
    
    # Tamanho de cada elemento
    dx = length / nx
    dy = height / ny
    
    # Criar matriz de nós
    # A numeração vai da esquerda para direita, de baixo para cima
    # Nó 0 está em (0, 0), nó 1 em (dx, 0), etc.
    total_nodes = nodes_x * nodes_y
    nodes = np.zeros((total_nodes, 2))
    
    node_idx = 0
    for j in range(nodes_y):      # y (linhas, de baixo para cima)
        for i in range(nodes_x):  # x (colunas, da esquerda para direita)
            nodes[node_idx, 0] = i * dx  # coordenada x
            nodes[node_idx, 1] = j * dy  # coordenada y
            node_idx += 1
    
    # Criar matriz de elementos
    # Cada elemento é um quadrilátero com 4 nós em sentido anti-horário
    # Começando pelo canto inferior esquerdo
    total_elements = nx * ny
    elements = np.zeros((total_elements, 4), dtype=int)
    
    elem_idx = 0
    for j in range(ny):      # linhas de elementos
        for i in range(nx):  # colunas de elementos
            # Índices dos 4 cantos do elemento
            bottom_left = j * nodes_x + i
            bottom_right = bottom_left + 1
            top_right = bottom_left + nodes_x + 1
            top_left = bottom_left + nodes_x
            
            # Sentido anti-horário: BL -> BR -> TR -> TL
            elements[elem_idx, 0] = bottom_left
            elements[elem_idx, 1] = bottom_right
            elements[elem_idx, 2] = top_right
            elements[elem_idx, 3] = top_left
            elem_idx += 1
    
    return nodes, elements


def get_boundary_nodes(nodes, length, height, tol=1e-10):
    """
    Identifica os nós em cada borda da viga.
    
    Retorna:
    --------
    dict com as chaves:
        'left'   : índices dos nós na borda esquerda (x = 0)
        'right'  : índices dos nós na borda direita (x = length)
        'bottom' : índices dos nós na borda inferior (y = 0)
        'top'    : índices dos nós na borda superior (y = height)
    """
    n_nodes = nodes.shape[0]
    
    left = []
    right = []
    bottom = []
    top = []
    
    for i in range(n_nodes):
        x, y = nodes[i, 0], nodes[i, 1]
        
        if abs(x - 0.0) < tol:
            left.append(i)
        if abs(x - length) < tol:
            right.append(i)
        if abs(y - 0.0) < tol:
            bottom.append(i)
        if abs(y - height) < tol:
            top.append(i)
    
    return {
        'left': np.array(left),
        'right': np.array(right),
        'bottom': np.array(bottom),
        'top': np.array(top)
    }


def get_boundary_edges(boundary_nodes):
    """
    Converte nós de uma borda em arestas (pares de nós consecutivos).
    
    Parâmetros:
    -----------
    boundary_nodes : np.ndarray
        Índices dos nós em uma borda (devem estar ordenados)
    
    Retorna:
    --------
    edges : np.ndarray (N_arestas × 2)
        Cada linha é [nó_inicial, nó_final] de uma aresta
    """
    # Ordenar os nós (importante para garantir arestas corretas)
    sorted_nodes = np.sort(boundary_nodes)
    n_edges = len(sorted_nodes) - 1
    
    edges = np.zeros((n_edges, 2), dtype=int)
    for i in range(n_edges):
        edges[i, 0] = sorted_nodes[i]
        edges[i, 1] = sorted_nodes[i + 1]
    
    return edges


# =============================================================================
# CRIAR A MALHA DA VIGA
# =============================================================================

# Dimensões da viga
length = 100.0  # cm
height = 10.0   # cm

# Número de elementos (baseado em elementos de 2x2 cm)
nx = 50  # 100 cm / 2 cm = 50 elementos
ny = 5   # 10 cm / 2 cm = 5 elementos

# Gerar malha
nodes, elements = create_rectangular_mesh(length, height, nx, ny)

# Identificar nós nas bordas
boundaries = get_boundary_nodes(nodes, length, height)

print("=" * 60)
print("INFORMAÇÕES DA MALHA")
print("=" * 60)
print(f"Dimensões da viga: {length} cm × {height} cm")
print(f"Tamanho do elemento: {length/nx} cm × {height/ny} cm")
print(f"Número de elementos: {nx} × {ny} = {nx * ny}")
print(f"Número de nós: {(nx+1)} × {(ny+1)} = {(nx+1) * (ny+1)}")
print()
print("Nós por borda:")
print(f"  Esquerda (x=0):    {len(boundaries['left'])} nós")
print(f"  Direita (x={length}): {len(boundaries['right'])} nós")
print(f"  Inferior (y=0):    {len(boundaries['bottom'])} nós")
print(f"  Superior (y={height}):  {len(boundaries['top'])} nós")

# =============================================================================
# VERIFICAÇÃO VISUAL (opcional - mostra os primeiros elementos)
# =============================================================================

print()
print("=" * 60)
print("PRIMEIROS 5 NÓS:")
print("=" * 60)
for i in range(min(5, len(nodes))):
    print(f"Nó {i}: ({nodes[i, 0]:.2f}, {nodes[i, 1]:.2f})")

print()
print("=" * 60)
print("PRIMEIROS 3 ELEMENTOS:")
print("=" * 60)
for i in range(min(3, len(elements))):
    e = elements[i]
    print(f"Elemento {i}: nós [{e[0]}, {e[1]}, {e[2]}, {e[3]}]")
    print(f"  Coordenadas:")
    for j, node_idx in enumerate(e):
        print(f"    Vértice {j}: nó {node_idx} -> ({nodes[node_idx, 0]:.2f}, {nodes[node_idx, 1]:.2f})")

# =============================================================================
# EXPORTAR PARA ARQUIVO (para usar no C++)
# =============================================================================

# Salvar nós
np.savetxt('nodes.csv', nodes, delimiter=',', fmt='%.6f', 
           header='x,y', comments='')

# Salvar elementos
np.savetxt('elements.csv', elements, delimiter=',', fmt='%d',
           header='n0,n1,n2,n3', comments='')

# Salvar índices das bordas
np.savetxt('boundary_left.csv', boundaries['left'], fmt='%d')
np.savetxt('boundary_right.csv', boundaries['right'], fmt='%d')
np.savetxt('boundary_bottom.csv', boundaries['bottom'], fmt='%d')
np.savetxt('boundary_top.csv', boundaries['top'], fmt='%d')

print()
print("=" * 60)
print("ARQUIVOS EXPORTADOS:")
print("=" * 60)
print("  nodes.csv          - Coordenadas dos nós")
print("  elements.csv       - Conectividade dos elementos")
print("  boundary_left.csv  - Nós na borda esquerda")
print("  boundary_right.csv - Nós na borda direita")
print("  boundary_bottom.csv - Nós na borda inferior")
print("  boundary_top.csv   - Nós na borda superior")
