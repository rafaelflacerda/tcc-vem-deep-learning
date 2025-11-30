import gmsh
import sys
import numpy as np
import json

# --- FUNÇÃO AUXILIAR: EXTRATOR DE DADOS ---
def extrair_e_salvar_json(nome_arquivo, tags_bc):
    """
    Extrai geometria e condições de contorno e salva em JSON.
    tags_bc: Dicionário contendo os IDs das curvas geometricas
             {'esquerda': id, 'inferior': id, 'direita': id}
    """
    print(f"Extraindo dados para {nome_arquivo}...")
    
    # ---------------------------------------------------------
    # 1. NODES (Coordenadas)
    # ---------------------------------------------------------
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodes_array = np.array(nodeCoords).reshape(-1, 3)[:, :2] # Pega só X e Y
    
    # Mapa fundamental: ID_Gmsh -> Índice_Python (0, 1, 2...)
    tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}

    # ---------------------------------------------------------
    # 2. ELEMENTS (Conectividade 2D)
    # ---------------------------------------------------------
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)
    
    elements_list = []
    for t, tags in zip(elemTypes, elemNodeTags):
        # Type 2 = Triângulo (3 nós), Type 3 = Quad (4 nós)
        num_nos = 3 if t == 2 else 4
        if num_nos == 4: # Prioridade para Quads
            conectividade = np.array(tags, dtype=int).reshape(-1, num_nos)
            mapped = np.vectorize(tag_to_index.get)(conectividade)
            elements_list.append(mapped)

    if elements_list:
        elements_array = np.vstack(elements_list)
    else:
        elements_array = []

    # ---------------------------------------------------------
    # 3. SUPP (Restrições de Deslocamento - Dirichlet)
    # ---------------------------------------------------------
    # Estrutura temporária: { indice_no: [restr_x, restr_y] }
    supp_dict = {}

    # A. Borda Esquerda (l_esq) -> Simetria X (impede movimento em X)
    # getNodes(1, tag) pega nós da curva 1D
    tags_esq, _, _ = gmsh.model.mesh.getNodes(1, tags_bc['esquerda'])
    
    for t in tags_esq:
        idx = tag_to_index[t]
        if idx not in supp_dict: supp_dict[idx] = [0, 0]
        supp_dict[idx][0] = 1 # Trava X

    # B. Borda Inferior (l_inf) -> Simetria Y (impede movimento em Y)
    tags_inf, _, _ = gmsh.model.mesh.getNodes(1, tags_bc['inferior'])
    
    for t in tags_inf:
        idx = tag_to_index[t]
        if idx not in supp_dict: supp_dict[idx] = [0, 0]
        supp_dict[idx][1] = 1 # Trava Y

    # Converte dicionário para matriz (K x 3)
    # [indice, rx, ry]
    supp_array = []
    for idx, restr in supp_dict.items():
        supp_array.append([idx, restr[0], restr[1]])
    
    supp_array = np.array(supp_array)

    # ---------------------------------------------------------
    # 4. LOAD (Arestas com Carga - Neumann)
    # ---------------------------------------------------------
    # Queremos os elementos de linha (1D) que estão na borda direita
    # getElements(dim=1, tag=l_dir) retorna apenas as linhas dessa curva
    _, _, loadNodeTags = gmsh.model.mesh.getElements(1, tags_bc['direita'])
    
    # O Gmsh retorna uma lista plana [n1, n2, n3, n4...]. 
    # Cada aresta tem 2 nós.
    if len(loadNodeTags) > 0:
        # Pega a primeira lista de tags (geralmente só tem um tipo de linha)
        load_flat = np.array(loadNodeTags[0], dtype=int)
        load_edges_gmsh = load_flat.reshape(-1, 2)
        
        # Mapeia para índices do Python
        load_array = np.vectorize(tag_to_index.get)(load_edges_gmsh)
    else:
        load_array = []

    # ---------------------------------------------------------
    # 5. SALVAR TUDO
    # ---------------------------------------------------------
    dados_json = {
        "nodes": nodes_array.tolist(),
        "elements": elements_array.tolist(),
        "supp": supp_array.tolist() if len(supp_array) > 0 else [],
        "load": load_array.tolist() if len(load_array) > 0 else []
    }

    with open(nome_arquivo, 'w') as f:
        json.dump(dados_json, f, indent=4)

    print(f"JSON salvo: {len(nodes_array)} nós, {len(elements_array)} elems, "
          f"{len(supp_array)} restrições, {len(load_array)} arestas carregadas.")


# --- FUNÇÃO PRINCIPAL ---
def gerar_malha_completa(R, L, H, lc_furo_param, lc_borda_param, nome_json_saida, visualizar=False):
    
    gmsh.initialize()
    gmsh.model.add("Overkill_Plate")

    # --- GEOMETRIA ---
    p0 = gmsh.model.geo.addPoint(0, 0, 0, lc_furo_param) 
    p1 = gmsh.model.geo.addPoint(R, 0, 0, lc_furo_param) 
    p2 = gmsh.model.geo.addPoint(0, R, 0, lc_furo_param) 
    p3 = gmsh.model.geo.addPoint(L, 0, 0, lc_borda_param)
    p4 = gmsh.model.geo.addPoint(L, H, 0, lc_borda_param)
    p5 = gmsh.model.geo.addPoint(0, H, 0, lc_borda_param)

    # Guardamos os IDs das curvas para usar depois nas matrizes!
    c_furo = gmsh.model.geo.addCircleArc(p1, p0, p2) 
    l_inf  = gmsh.model.geo.addLine(p1, p3)  # Borda Inferior
    l_dir  = gmsh.model.geo.addLine(p3, p4)  # Borda Direita (Carga)
    l_sup  = gmsh.model.geo.addLine(p4, p5)
    l_esq  = gmsh.model.geo.addLine(p5, p2)  # Borda Esquerda

    loop = gmsh.model.geo.addCurveLoop([l_inf, l_dir, l_sup, l_esq, -c_furo])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    # --- MALHA ---
    # Campos para gradiente suave (Opcional, mas recomendado para qualidade)
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [c_furo])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("MathEval", 2)
    formula = f"{lc_furo_param} + 0.2 * F1^1.1" 
    gmsh.model.mesh.field.setString(2, "F", formula)

    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(2) 

    gmsh.model.mesh.setRecombine(2, surf) # Quads
    gmsh.option.setNumber("Mesh.Algorithm", 8)

    gmsh.model.mesh.generate(2)
    
    # --- EXPORTAR ---
    # Montamos o dicionário com as tags que a função de exportação precisa
    tags_condicoes_contorno = {
        'esquerda': l_esq,
        'inferior': l_inf,
        'direita':  l_dir
    }
    
    extrair_e_salvar_json(nome_json_saida, tags_condicoes_contorno)

    if visualizar:
        gmsh.fltk.run()
    
    gmsh.finalize()

# --- EXECUÇÃO ---
if __name__ == "__main__":
    gerar_malha_completa(
        R=20, L=100, H=100, 
        lc_furo_param=2.0,     # Um pouco mais grosso pra testar
        lc_borda_param=10.0,    
        nome_json_saida="malha_com_bcs.json",
        visualizar=True
    )