import os
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ============================================================
# 🔹 Dataset personalizado com DOIS modos de carregamento
# ============================================================
class BeamDataset(Dataset):
    """
    ✅ NOVO: Carrega dados de DOIS CSVs diferentes
    - training_data.csv: 5 pontos estruturados por amostra (0.2L, 0.4L, 0.6L, 0.8L, 1.0L)
    - full_validation_data.csv: TODOS os nós (denso, sem estrutura)
    
    Modo de uso:
    - mode='training': Carrega training_data.csv (poucos pontos, bem distribuídos)
    - mode='validation': Carrega full_validation_data.csv (muitos pontos, cobertura total)
    """
    
    def __init__(self, data_dir, mode='training', n_samples=None, normalize=True, random_seed=42):
        """
        Args:
            data_dir: caminho da pasta onde estão os CSVs
            mode: 'training' ou 'validation'
            n_samples: número de amostras a carregar (None = todas)
            normalize: se True, normaliza entradas e saídas
            random_seed: seed para reproducibilidade
        """
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.data_dir = data_dir
        self.mode = mode
        self.normalize = normalize
        self.L = 1.0  # Comprimento da viga
        
        # ============================================================
        # Selecionar arquivo baseado em mode
        # ============================================================
        if mode == 'training':
            csv_file = "training_data.csv"
            print(f"📥 Modo TRAINING: Carregando {csv_file}")
            print(f"   → 5 pontos estruturados por amostra (0.2L, 0.4L, 0.6L, 0.8L, 1.0L)")
        elif mode == 'validation':
            csv_file = "full_validation_data.csv"
            print(f"📥 Modo VALIDATION: Carregando {csv_file}")
            print(f"   → Todos os nós (cobertura densa)")
        else:
            raise ValueError(f"Mode inválido: {mode}. Use 'training' ou 'validation'")
        
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"✓ CSV carregado com {len(df)} linhas")
        
        # ============================================================
        # Carregar e processar dados
        # ============================================================
        X, y = [], []
        
        if mode == 'training':
            # MODO TRAINING: Cada linha é uma amostra com 5 pontos
            # Colunas esperadas:
            # sample_id, E, I, q, w_0.2L, theta_0.2L, w_0.4L, theta_0.4L, ...
            
            positions = [0.2, 0.4, 0.6, 0.8, 1.0]
            
            if n_samples is not None:
                df = df.iloc[:n_samples]
            
            for idx, row in df.iterrows():
                E = row['E']
                I = row['I']
                q = row['q']
                
                # Iterar sobre os 5 pontos estruturados
                for pos in positions:
                    x_pos = pos * self.L
                    
                    # Deslocamento neste ponto
                    col_w = f'w_{pos:.1f}L'
                    w = row[col_w]
                    
                    # Features (mesmo padrão do generate_dataset)
                    EI_log = np.log10(E * I)
                    q_log = np.log10(abs(q) + 1e-9)
                    
                    x_scaled = x_pos / self.L
                    x2 = x_pos ** 2
                    x3 = x_pos ** 3
                    x4 = x_pos ** 4
                    
                    EI = E * I
                    inv_EI = 1.0 / EI
                    q_over_EI = q / EI
                    theoretical_disp = -(q * x_pos**2 * (6*self.L**2 - 4*self.L*x_pos + x_pos**2)) / (24 * EI)
                    
                    X.append([
                        EI_log, q_log,
                        x_scaled, x2, x3, x4,
                        EI, inv_EI, q_over_EI,
                        theoretical_disp
                    ])
                    y.append(w)
        
        elif mode == 'validation':
            # MODO VALIDATION: Cada linha é UM PONTO em UM NÍVEL de refinamento
            # Colunas esperadas:
            # sample_id, x_position, E, I, q, w, theta
            
            if n_samples is not None:
                # Filtrar apenas primeiras N amostras (sample_id)
                unique_samples = df['sample_id'].unique()[:n_samples]
                df = df[df['sample_id'].isin(unique_samples)]
            
            for idx, row in df.iterrows():
                E = row['E']
                I = row['I']
                q = row['q']
                x_pos = row['x_position']
                w = row['w']
                
                # Features (mesmo padrão)
                EI_log = np.log10(E * I)
                q_log = np.log10(abs(q) + 1e-9)
                
                x_scaled = x_pos / self.L
                x2 = x_pos ** 2
                x3 = x_pos ** 3
                x4 = x_pos ** 4
                
                EI = E * I
                inv_EI = 1.0 / EI
                q_over_EI = q / EI
                theoretical_disp = -(q * x_pos**2 * (6*self.L**2 - 4*self.L*x_pos + x_pos**2)) / (24 * EI)
                
                X.append([
                    EI_log, q_log,
                    x_scaled, x2, x3, x4,
                    EI, inv_EI, q_over_EI,
                    theoretical_disp
                ])
                y.append(w)
        
        self.X = np.array(X, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64).reshape(-1, 1)
        
        # ============================================================
        # Normalização com StandardScaler
        # ============================================================
        if normalize:
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
            
            self.X = self.scaler_x.fit_transform(self.X)
            self.y = self.scaler_y.fit_transform(self.y)
            
            print(f"✓ Normalização ativada (StandardScaler)")
        else:
            self.scaler_x = None
            self.scaler_y = None
            print(f"⚠️  Sem normalização")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


# ============================================================
# 🔹 Função auxiliar para criar dataloaders (treino/validação)
# ============================================================
def get_dataloaders(data_dir, n_samples=None, batch_size=256, normalize=True, val_split=0.2):
    """
    ✅ NOVO: Cria dataloaders usando MODO TRAINING (5 pontos estruturados)
    
    Args:
        data_dir: caminho do dataset
        n_samples: limitar número de amostras (None = todas)
        batch_size: tamanho do batch
        normalize: usar StandardScaler
        val_split: fração para validação (0.2 = 80% treino, 20% validação)
    
    Returns:
        train_loader, val_loader, dataset (com scaler_x e scaler_y)
    """
    dataset = BeamDataset(
        data_dir,
        mode='training',
        n_samples=n_samples,
        normalize=normalize,
        random_seed=42
    )

    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    generator = torch.Generator()
    generator.manual_seed(42)

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=generator
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, dataset


# ============================================================
# 🔹 Função para gerar dataset de VALIDAÇÃO DENSA (full_validation_data)
# ============================================================
def get_validation_dataset(data_dir, n_samples=None, normalize=True, scaler_x=None, scaler_y=None):
    """
    ✅ Carrega dataset COMPLETO (full_validation_data.csv) para avaliação
    
    Args:
        data_dir: caminho do dataset
        n_samples: limitar número de amostras
        normalize: se True, usa os scalers fornecidos
        scaler_x, scaler_y: scalers do treinamento (OBRIGATÓRIO se normalize=True)
    
    Returns:
        torch.DataLoader com dados densos de validação
    """
    
    class ValidationDataset(Dataset):
        def __init__(self, data_dir, n_samples, normalize, scaler_x, scaler_y):
            csv_path = os.path.join(data_dir, "full_validation_data.csv")
            df = pd.read_csv(csv_path)
            
            if n_samples is not None:
                unique_samples = df['sample_id'].unique()[:n_samples]
                df = df[df['sample_id'].isin(unique_samples)]
            
            X, y = [], []
            L = 1.0
            
            for idx, row in df.iterrows():
                E = row['E']
                I = row['I']
                q = row['q']
                x_pos = row['x_position']
                w = row['w']
                
                EI_log = np.log10(E * I)
                q_log = np.log10(abs(q) + 1e-9)
                
                x_scaled = x_pos / L
                x2 = x_pos ** 2
                x3 = x_pos ** 3
                x4 = x_pos ** 4
                
                EI = E * I
                inv_EI = 1.0 / EI
                q_over_EI = q / EI
                theoretical_disp = -(q * x_pos**2 * (6*L**2 - 4*L*x_pos + x_pos**2)) / (24 * EI)
                
                X.append([
                    EI_log, q_log,
                    x_scaled, x2, x3, x4,
                    EI, inv_EI, q_over_EI,
                    theoretical_disp
                ])
                y.append(w)
            
            self.X = np.array(X, dtype=np.float64)
            self.y = np.array(y, dtype=np.float64).reshape(-1, 1)
            
            if normalize and scaler_x is not None and scaler_y is not None:
                self.X = scaler_x.transform(self.X)
                self.y = scaler_y.transform(self.y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return (
                torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32)
            )
    
    dataset = ValidationDataset(data_dir, n_samples, normalize, scaler_x, scaler_y)
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    
    return loader


# ============================================================
# ✅ FUNÇÕES PARA INVERSE TRANSFORM (StandardScaler)
# ============================================================

def inverse_transform_predictions(predictions, scaler_y):
    """
    ✅ Converte predições de espaço transformado → espaço original
    
    Args:
        predictions: np.ndarray ou torch.Tensor em espaço transformado
        scaler_y: StandardScaler fitted
    
    Returns:
        np.ndarray em espaço original (físico)
    """
    
    if isinstance(predictions, torch.Tensor):
        pred_np = predictions.detach().cpu().numpy()
    else:
        pred_np = predictions
    
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(-1, 1)
    
    pred_original = scaler_y.inverse_transform(pred_np)
    
    return pred_original


def validate_with_inverse(model, val_loader, criterion, device, scaler_y, verbose=True):
    """
    ✅ Validação que mostra erro em DOIS espaços
    - Transformado: para otimização (scheduler usa este)
    - Original: para interpretação (usar este nos resultados!)
    """
    model.eval()
    total_loss_transformed = 0
    total_loss_original = 0
    
    with torch.no_grad():
        for xb, yb_transformed in val_loader:
            xb, yb_transformed = xb.to(device), yb_transformed.to(device)
            
            pred_transformed = model(xb)
            loss_transformed = criterion(pred_transformed, yb_transformed)
            
            yb_original = inverse_transform_predictions(yb_transformed, scaler_y)
            pred_original = inverse_transform_predictions(pred_transformed, scaler_y)
            
            loss_original = np.mean((pred_original - yb_original) ** 2)
            
            total_loss_transformed += loss_transformed.item() * xb.size(0)
            total_loss_original += loss_original * xb.size(0)
    
    avg_loss_transformed = total_loss_transformed / len(val_loader.dataset)
    avg_loss_original = total_loss_original / len(val_loader.dataset)
    
    if verbose:
        print(f"     Val Loss (transformado): {avg_loss_transformed:.6f}")
        print(f"     Val Loss (original):     {avg_loss_original:.6f} ← USE ESTE")
    
    return avg_loss_transformed, avg_loss_original


def save_checkpoint(model, optimizer, scaler_y, epoch, loss, save_path):
    """✅ Salva modelo COM scaler_y completo"""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
        "scaler_y_pickle": pickle.dumps(scaler_y),
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint salvo em: {save_path}")


def load_checkpoint(model, optimizer, load_path, device):
    """✅ Carrega modelo COM scaler_y completo"""
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler_y = pickle.loads(checkpoint["scaler_y_pickle"])
    
    print(f"✓ Checkpoint carregado de: {load_path}")
    print(f"  Época: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.6f}")
    
    return model, optimizer, scaler_y, checkpoint["epoch"]