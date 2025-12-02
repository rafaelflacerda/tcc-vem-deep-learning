# sobolev_utils.py

import torch

def compute_element_centroids(nodes_case: torch.Tensor,
                              elements_case: torch.Tensor) -> torch.Tensor:
    """
    Calcula centróides de elementos triangulares de um caso.

    nodes_case: (N_nodes_case, 2)  -> coords físicas (x, y)
    elements_case: (N_elems_case, 3) -> índices dos nós (inteiros, 0-based)

    Retorna:
        centroids: (N_elems_case, 2)
    """
    # elements_case: cada linha = [n1, n2, n3]
    n1 = nodes_case[elements_case[:, 0], :]  # (N_elems_case, 2)
    n2 = nodes_case[elements_case[:, 1], :]
    n3 = nodes_case[elements_case[:, 2], :]

    centroids = (n1 + n2 + n3) / 3.0
    return centroids


def build_sigma_inputs_for_case(centroids: torch.Tensor,
                                case_params: torch.Tensor,
                                X_max: torch.Tensor) -> torch.Tensor:
    """
    Constrói o vetor de entrada X_sigma (por elemento) para passar na rede
    quando formos aplicar a loss de Sobolev nas tensões.

    centroids: (N_elems_case, 2) -> [x, y] físicos
    case_params: (4,) -> [L_phys, H_phys, R_phys, nu_phys]
    X_max: (6,) -> mesmos fatores de normalização usados no gerador de NPZ
                   (para [x, y, d, L, H, R])

    Retorna:
        X_sigma: (N_elems_case, 7) nas mesmas features do dataset:
                 [x_norm, y_norm, d_norm, L_norm, H_norm, R_norm, nu]
    """
    # separa params
    L_phys, H_phys, R_phys, nu_phys = case_params  # cada um escalar

    x = centroids[:, 0]
    y = centroids[:, 1]

    # distância à borda do furo: d = sqrt(x^2 + y^2) - R
    d = torch.sqrt(x**2 + y**2) - R_phys

    # empilha vetores físicos
    L_vec = torch.full_like(x, L_phys)
    H_vec = torch.full_like(x, H_phys)
    R_vec = torch.full_like(x, R_phys)
    nu_vec = torch.full_like(x, nu_phys)

    # monta features físicas: (N_elems_case, 7)
    X_phys = torch.stack([x, y, d, L_vec, H_vec, R_vec, nu_vec], dim=-1)

    # normalizar as 6 primeiras colunas, igual no gerador NPZ
    # X_max: (6,) -> broadcast
    X_sigma = X_phys.clone()
    X_sigma[:, :6] = X_sigma[:, :6] / (X_max + 1e-8)

    return X_sigma

def build_C_plane_stress(nu: torch.Tensor) -> torch.Tensor:
    """
    Monta a matriz de elasticidade 2D (plano de tensões) com E = 1.
    nu: tensor escalar (ou shape [...]) com o coeficiente de Poisson.
    Retorna C com shape (..., 3, 3).
    """
    # E = 1
    E = 1.0
    # Fórmula plano de tensões:
    # C = E / (1 - nu^2) * [[1,   nu,         0],
    #                       [nu,  1,          0],
    #                       [0,   0,  (1 - nu)/2]]
    denom = 1.0 - nu**2
    factor = E / denom

    C11 = factor * 1.0
    C22 = factor * 1.0
    C12 = factor * nu
    C21 = C12
    C33 = factor * (1.0 - nu) / 2.0

    # monta C com broadcast
    C = torch.zeros((*nu.shape, 3, 3), dtype=torch.float32, device=nu.device)
    C[..., 0, 0] = C11
    C[..., 0, 1] = C12
    C[..., 1, 0] = C21
    C[..., 1, 1] = C22
    C[..., 2, 2] = C33

    return C

def compute_sigma_pred_for_case(
    model,
    case_id: int,
    loader,
    device: torch.device,
) -> torch.Tensor:
    """
    Dado um caso (case_id), usa:
      - loader.case_nodes[case_id]: (N_nodes_case, 2) com [x,y] físicos
      - loader.case_elements[case_id]: (N_elem_case, 3) conectividade
      - loader.case_params[case_id]: [L,H,R,nu]
      - loader.X_max: (6,) normalização de [x,y,d,L,H,R]

    para montar X_sigma e calcular sigma_pred no centróide de cada elemento.

    Retorna sigma_pred_case com shape (N_elem_case, 3),
    em termos de tensões físicas (σ_xx, σ_yy, τ_xy) via C (plano de tensões, E=1).
    """
    # 0) Puxa tudo do loader
    nodes = loader.case_nodes[case_id].to(device)         # (N_nodes, 2) físicos
    elements = loader.case_elements[case_id].to(device)   # (N_elem, 3)
    params = loader.case_params[case_id].to(device)       # [L,H,R,nu]
    X_max = loader.X_max.to(device)                       # (6,)

    L_phys, H_phys, R_phys, nu_phys = params  # escalares

    # 1) centróides dos elementos COM grad apenas em x,y
    # nodes[elements] -> (N_elem, 3, 2); mean(dim=1) -> (N_elem, 2)
    centroids = nodes[elements].mean(dim=1)
    centroids = centroids.detach().clone().requires_grad_(True)

    x = centroids[:, 0]
    y = centroids[:, 1]

    # 2) constrói features físicas
    d = torch.sqrt(x**2 + y**2) - R_phys

    N_e = centroids.shape[0]
    L_col = L_phys * torch.ones(N_e, device=device)
    H_col = H_phys * torch.ones(N_e, device=device)
    R_col = R_phys * torch.ones(N_e, device=device)
    nu_col = nu_phys * torch.ones(N_e, device=device)

    # (N_elem, 7): [x, y, d, L, H, R, nu]
    X_phys = torch.stack([x, y, d, L_col, H_col, R_col, nu_col], dim=1)

    # 3) normalizar as 6 primeiras colunas, igual ao NPZ
    X_sigma = X_phys.clone()
    X_sigma[:, :6] = X_sigma[:, :6] / (X_max[:6] + 1e-8)

    # 4) forward na rede: u,v
    u_v = model(X_sigma)  # (N_elem, 2)
    u = u_v[:, 0]
    v = u_v[:, 1]

    # 5) derivadas físicas du/dx, du/dy, dv/dx, dv/dy via grad w.r.t. centroids (x,y)
    ones_u = torch.ones_like(u, device=device)
    ones_v = torch.ones_like(v, device=device)

    grad_u = torch.autograd.grad(
        u, centroids,
        grad_outputs=ones_u,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (N_elem, 2) -> [∂u/∂x, ∂u/∂y]

    grad_v = torch.autograd.grad(
        v, centroids,
        grad_outputs=ones_v,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # (N_elem, 2) -> [∂v/∂x, ∂v/∂y]

    du_dx = grad_u[:, 0]
    du_dy = grad_u[:, 1]
    dv_dx = grad_v[:, 0]
    dv_dy = grad_v[:, 1]

    # 6) strain de engenharia
    eps_xx   = du_dx
    eps_yy   = dv_dy
    gamma_xy = du_dy + dv_dx

    strain = torch.stack([eps_xx, eps_yy, gamma_xy], dim=1)  # (N_elem, 3)

    # 7) matriz C (plano de tensões, E = 1) e sigma_pred = C * strain
    nu_t = nu_phys.to(device)
    C = build_C_plane_stress(nu_t)             # (3,3)
    C = C.view(1, 3, 3).expand(N_e, 3, 3)      # (N_elem, 3, 3)

    strain_vec = strain.unsqueeze(-1)          # (N_elem, 3, 1)
    sigma_pred = torch.bmm(C, strain_vec)      # (N_elem, 3, 1)
    sigma_pred = sigma_pred.squeeze(-1)        # (N_elem, 3)

    return sigma_pred

def sobolev_stress_loss_for_case(
    model,
    case_id: int,
    loader,
    device: torch.device,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Calcula a loss de tensões (Sobolev) para um único caso.

    Usa:
      - loader.case_sigma[case_id]: (N_elem_case, 3)  -> sigma_true_norm (NORMALIZADA!)
      - loader.S_max: (3,)                            -> fatores de normalização por componente
      - compute_sigma_pred_for_case(...)             -> sigma_pred física

    Retorna:
      escalar (tensor) com a perda de tensões para esse caso.
    """
    # Se o loader não tiver tensões, devolve 0.
    if loader.case_sigma is None or case_id not in loader.case_sigma:
        return torch.tensor(0.0, device=device)

    # 1) Tensões verdadeiras (já NORMALIZADAS, vindo do NPZ)
    sigma_true_norm = loader.case_sigma[case_id].to(device)   # (N_elem_case, 3)

    # 2) Fatores de normalização (S_max) -> (3,)
    S_max = loader.S_max.to(device)                           # (3,)

    # 3) Predição de tensões FÍSICAS a partir do modelo
    sigma_pred_phys = compute_sigma_pred_for_case(
        model=model,
        case_id=case_id,
        loader=loader,
        device=device,
    )   # (N_elem_case, 3) físicas

    # 4) Normalizar sigma_pred para o mesmo espaço de sigma_true_norm
    sigma_pred_norm = sigma_pred_phys / (S_max + 1e-8)

    # 5) MSE componente a componente
    diff = sigma_pred_norm - sigma_true_norm                  # (N_elem_case, 3)
    loss_per_elem = (diff ** 2).mean(dim=1)                   # (N_elem_case,)

    if reduction == "mean":
        return loss_per_elem.mean()
    elif reduction == "sum":
        return loss_per_elem.sum()
    else:
        # se quiser usar "none" algum dia
        return loss_per_elem