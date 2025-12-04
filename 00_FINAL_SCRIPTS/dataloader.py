import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(npz_path, batch_size=4096, num_workers=8, train_split=0.8, seed=42):
    """
    Carrega NPZ e retorna train/val DataLoaders
    Split por casos (não por nós) para evitar data leakage.

    Se o .npz tiver:
      - Sigma, sigma_case_indices, S_max  → tensões por elemento
      - nodes, nodes_case_indices         → nós físicos
      - elements, elements_case_indices   → conectividade
    tudo isso é carregado, splitado por caso e anexado aos loaders.
    """
    # ---------- CARREGAR DADOS BÁSICOS ----------
    data = np.load(npz_path)
    X = data['X']                       # (N_nodes_total, 7) normalizado nos 6 primeiros
    Y = data['Y']                       # (N_nodes_total, 2)
    case_indices = data['case_indices'] # (N_nodes_total,)
    X_max = data['X_max'].astype(np.float32)  # (6,)

    # ---------- TENSÕES (se existirem) ----------
    has_sigma = ('Sigma' in data.files) and ('sigma_case_indices' in data.files)
    if has_sigma:
        Sigma = data['Sigma'].astype(np.float32)                  # (N_elem_total, 3) já normalizado
        sigma_case_indices = data['sigma_case_indices'].astype(np.int32)
        S_max = data['S_max'].astype(np.float32)                  # (3,)
    else:
        Sigma = None
        sigma_case_indices = None
        S_max = None

    # ---------- MALHA: nós / elementos (se existirem) ----------
    has_mesh = all(k in data.files for k in [
        'nodes', 'nodes_case_indices', 'elements', 'elements_case_indices'
    ])
    if has_mesh:
        nodes_all = data['nodes'].astype(np.float32)                  # (N_nodes_total, 2) físicos
        nodes_case_indices = data['nodes_case_indices'].astype(np.int32)
        elements_all = data['elements'].astype(np.int32)              # (N_elem_total, 3)
        elements_case_indices = data['elements_case_indices'].astype(np.int32)
    else:
        nodes_all = None
        nodes_case_indices = None
        elements_all = None
        elements_case_indices = None

    # ---------- CASOS ÚNICOS ----------
    unique_cases = np.unique(case_indices)

    # Shuffle com seed fixo (reprodutibilidade)
    np.random.seed(seed)
    np.random.shuffle(unique_cases)

    # Split 80/20
    n_train = int(train_split * len(unique_cases))
    train_cases = unique_cases[:n_train]
    val_cases = unique_cases[n_train:]

    # ---------- SPLIT NODAL (igual antes, só adicionando case_train/case_val) ----------
    train_mask = np.isin(case_indices, train_cases)
    val_mask = np.isin(case_indices, val_cases)

    X_train = torch.tensor(X[train_mask], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_mask], dtype=torch.float32)
    case_train = torch.tensor(case_indices[train_mask], dtype=torch.int64)

    X_val = torch.tensor(X[val_mask], dtype=torch.float32)
    Y_val = torch.tensor(Y[val_mask], dtype=torch.float32)
    case_val = torch.tensor(case_indices[val_mask], dtype=torch.int64)

    train_dataset = TensorDataset(X_train, Y_train, case_train)
    val_dataset = TensorDataset(X_val, Y_val, case_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ---------- PARAMETROS POR CASO (L, H, R, nu físicos) ----------
    # X está normalizado: colunas 3,4,5 = L,H,R normalizados por X_max[3:6]
    # coluna 6 = nu (não normalizado)
    case_params_all = {}
    for c in unique_cases:
        mask_c = (case_indices == c)
        # pega qualquer nó daquele caso (todas as linhas têm mesmos L,H,R,nu)
        x_sample = X[mask_c][0]   # shape (7,)
        # desfaz normalização para L,H,R
        L_phys = x_sample[3] * X_max[3]
        H_phys = x_sample[4] * X_max[4]
        R_phys = x_sample[5] * X_max[5]
        nu_phys = x_sample[6]
        case_params_all[int(c)] = np.array([L_phys, H_phys, R_phys, nu_phys], dtype=np.float32)

    # ---------- DICIONÁRIOS POR CASO (nós, elementos, tensões) ----------
    if has_mesh:
        # train
        case_nodes_train = {}
        case_elements_train = {}
        case_sigma_train = {} if has_sigma else None

        for c in train_cases:
            c_int = int(c)

            # nós desse caso
            nmask = (nodes_case_indices == c)
            case_nodes_train[c_int] = torch.tensor(
                nodes_all[nmask], dtype=torch.float32
            )  # (N_nodes_case, 2)

            # elementos desse caso
            emask = (elements_case_indices == c)
            case_elements_train[c_int] = torch.tensor(
                elements_all[emask], dtype=torch.int64
            )  # (N_elem_case, 3)

            # tensões desse caso (se existirem)
            if has_sigma:
                smask = (sigma_case_indices == c)
                case_sigma_train[c_int] = torch.tensor(
                    Sigma[smask], dtype=torch.float32
                )  # (N_elem_case, 3) normalizado

        # val
        case_nodes_val = {}
        case_elements_val = {}
        case_sigma_val = {} if has_sigma else None

        for c in val_cases:
            c_int = int(c)

            nmask = (nodes_case_indices == c)
            case_nodes_val[c_int] = torch.tensor(
                nodes_all[nmask], dtype=torch.float32
            )

            emask = (elements_case_indices == c)
            case_elements_val[c_int] = torch.tensor(
                elements_all[emask], dtype=torch.int64
            )

            if has_sigma:
                smask = (sigma_case_indices == c)
                case_sigma_val[c_int] = torch.tensor(
                    Sigma[smask], dtype=torch.float32
                )

        # anexar nos loaders
        train_loader.case_nodes = case_nodes_train
        train_loader.case_elements = case_elements_train
        train_loader.case_sigma = case_sigma_train
        train_loader.case_params = {
            int(c): torch.from_numpy(case_params_all[int(c)]) for c in train_cases
        }
        train_loader.X_max = torch.tensor(X_max, dtype=torch.float32)

        val_loader.case_nodes = case_nodes_val
        val_loader.case_elements = case_elements_val
        val_loader.case_sigma = case_sigma_val
        val_loader.case_params = {
            int(c): torch.from_numpy(case_params_all[int(c)]) for c in val_cases
        }
        val_loader.X_max = torch.tensor(X_max, dtype=torch.float32)
    else:
        # se NPZ não tiver malha, mantém None pra não quebrar nada
        train_loader.case_nodes = None
        train_loader.case_elements = None
        train_loader.case_sigma = None
        train_loader.case_params = None
        train_loader.X_max = torch.tensor(X_max, dtype=torch.float32)

        val_loader.case_nodes = None
        val_loader.case_elements = None
        val_loader.case_sigma = None
        val_loader.case_params = None
        val_loader.X_max = torch.tensor(X_max, dtype=torch.float32)

    # ---------- ATRIBUTOS PLANOS DE SIGMA (como você já tinha) ----------
    if has_sigma:
        sigma_train_mask = np.isin(sigma_case_indices, train_cases)
        sigma_val_mask = np.isin(sigma_case_indices, val_cases)

        Sigma_train = torch.tensor(Sigma[sigma_train_mask], dtype=torch.float32)
        Sigma_val = torch.tensor(Sigma[sigma_val_mask], dtype=torch.float32)

        sigma_case_indices_train = torch.tensor(
            sigma_case_indices[sigma_train_mask], dtype=torch.int32
        )
        sigma_case_indices_val = torch.tensor(
            sigma_case_indices[sigma_val_mask], dtype=torch.int32
        )

        train_loader.sigma = Sigma_train
        train_loader.sigma_case_indices = sigma_case_indices_train
        train_loader.S_max = torch.tensor(S_max, dtype=torch.float32)

        val_loader.sigma = Sigma_val
        val_loader.sigma_case_indices = sigma_case_indices_val
        val_loader.S_max = torch.tensor(S_max, dtype=torch.float32)
    else:
        train_loader.sigma = None
        train_loader.sigma_case_indices = None
        train_loader.S_max = None

        val_loader.sigma = None
        val_loader.sigma_case_indices = None
        val_loader.S_max = None

    # ---------- PRINTS ----------
    print(f"Train: {len(train_cases)} casos ({len(X_train)} nós)")
    print(f"Val:   {len(val_cases)} casos ({len(X_val)} nós)")
    if has_sigma:
        n_train_elems = train_loader.sigma.shape[0]
        n_val_elems = val_loader.sigma.shape[0]
        print(f"Tensões: {n_train_elems} elementos (train), {n_val_elems} elementos (val)")
    if has_mesh:
        print(f"Malha carregada: nodes={nodes_all.shape[0]}, elements={elements_all.shape[0]}")

    return train_loader, val_loader