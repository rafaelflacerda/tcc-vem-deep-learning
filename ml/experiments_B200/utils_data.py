import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from physics import analytical_solution
import pickle

class Beam1DDataset(Dataset):
    def __init__(self, csv_file, scaler_X=None, scaler_y=None, is_training=True):
        """
        Args:
            csv_file: Caminho para o CSV (training ou validation).
            scaler_X, scaler_y: Scalers fitados (se None, cria novos).
            is_training: Se True, parseia o formato 'Wide' (training_data.csv).
                         Se False, parseia o formato 'Long' (full_validation_data.csv).
        """
        self.df = pd.read_csv(csv_file)
        self.L = 1.0
        
        X_list = []
        y_list = []
        
        if is_training:
            # --- Formato TREINO (Wide: colunas w_0.2L, theta_0.2L, etc) ---
            # Posições fixas que definimos no C++
            positions_real = [9/44, 13/44, 22/44, 31/44, 44/44]
            
            positions_excel = [0.2, 0.3, 0.5, 0.7, 1.0]
            
            for idx, row in self.df.iterrows():
                E, I, q = row['E'], row['I'], row['q']
                
                for pos in positions_real:
                    x_val = pos * self.L
                    
                    idx_pos = positions_real.index(pos)
                    
                    pos_excel = positions_excel[idx_pos]
                    
                    # Inputs Físicos
                    # Calculamos a solução analítica aqui para usar como FEATURE (Viés Indutivo)
                    w_ana, th_ana = analytical_solution(E, I, q, x_val, self.L)
                    
                    E_log = np.log10(E)
                    I_log = np.log10(I)
                    
                    # Vetor de Entrada: [x, E, I, q, w_analitico, theta_analitico]
                    X_list.append([x_val, E_log, I_log, q, w_ana, th_ana])
                    
                    # Vetor de Saída (Target do VEM): [w_vem, theta_vem]
                    col_w = f"w_{pos_excel}L"
                    col_th = f"theta_{pos_excel}L"
                    y_list.append([row[col_w], row[col_th]])
                    
        else:
            # --- Formato VALIDAÇÃO (Long: colunas x_position, w, theta) ---
            for idx, row in self.df.iterrows():
                E, I, q, x_val = row['E'], row['I'], row['q'], row['x_position']
                
                w_ana, th_ana = analytical_solution(E, I, q, x_val, self.L)
                
                # Mesmo formato de entrada
                X_list.append([x_val, E, I, q, w_ana, th_ana])
                y_list.append([row['w'], row['theta']])

        self.X = np.array(X_list, dtype=np.float32)
        self.y = np.array(y_list, dtype=np.float32)
        
        # --- Normalização (StandardScaler) ---
        if scaler_X is None:
            self.scaler_X = StandardScaler()
            self.X = self.scaler_X.fit_transform(self.X)
        else:
            self.scaler_X = scaler_X
            self.X = self.scaler_X.transform(self.X)
            
        if scaler_y is None:
            self.scaler_y = StandardScaler()
            self.y = self.scaler_y.fit_transform(self.y)
        else:
            self.scaler_y = scaler_y
            self.y = self.scaler_y.transform(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def save_scalers(scaler_X, scaler_y, path="scalers.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f)

def load_scalers(path="scalers.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["scaler_X"], data["scaler_y"]