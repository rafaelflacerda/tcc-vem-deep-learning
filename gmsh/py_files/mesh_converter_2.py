import pandas as pd
import os

# Determina o caminho do diretório base
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR) 

# Determina o caminho dos arquivos a serem acessados
ELEMENTS_DIR = os.path.join(ROOT_DIR, "gmsh", "elements.csv")
NODES_DIR = os.path.join(ROOT_DIR, "gmsh", "nodes.csv")


elements = pd.read_csv(ELEMENTS_DIR)
nodes = pd.read_csv(NODES_DIR)

# Cria a lista que será a matriz elements
elements_list = elements.values.flatten().tolist()
linhas_elements = elements.shape[0]
colunas_elements = elements.shape[1]

# Cria a lista que será a matriz nodes
nodes_list = nodes.values.flatten().tolist()
linhas_nodes = nodes.shape[0]
colunas_nodes = nodes.shape[1]

# Cria listas auxiliares. Nós da esquerda (X = 0) e nós da parte de baixo (Y = 0)
left_nodes = nodes.index[nodes["x"] == 0].tolist()
bottom_nodes = nodes.index[nodes["y"] == 0].tolist()

supp = []

# União dos nós que têm algum tipo de restrição
restricted_nodes = sorted(set(left_nodes) | set(bottom_nodes))

for node in restricted_nodes:
    restr_x = 1 if node in left_nodes else 0
    restr_y = 1 if node in bottom_nodes else 0
    supp.append((node, restr_x, restr_y))

# Cria uma lista formada pelos nós da lateral direita (X = L), onde atuarão os esforços
load_nodes = nodes.index[nodes["x"] == 0.1].tolist()

# Ordenar pelos valores de y (da base ao topo)
load_nodes_sorted = sorted(load_nodes, key=lambda n: nodes.loc[n, "y"])

# Construir pares consecutivos (arestas)
load_edges = []
for i in range(len(load_nodes_sorted) - 1):
    n0 = load_nodes_sorted[i]
    n1 = load_nodes_sorted[i+1]
    load_edges.append((n0, n1))

# PRINT EXATAMENTE NO FORMATO SOLICITADO
#  n0,n1, n2,n3, n4,n5, ...
saida = ", ".join(f"{a},{b}" for a, b in load_edges)
print(saida)

# print("")

# print("Matriz de conectividade dos elementos:")
# print("{linhas_elements} linhas e {colunas_elements} colunas")
# print("")
# print(elements)
# print("")

# print("Matriz de coordenadas dos nós:")
# print("{linhas_nodes} linhas e {colunas_nodes} colunas")
# print("")
# print(nodes)
# print("")

# print("Matriz dos nós com alguma restrição (supp):")
# saida = ", ".join(f"{n},{rx},{ry}" for n, rx, ry in supp)
# print(saida)

