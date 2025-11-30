import torch
import torch.nn as nn
import numpy as np

class BayesianVEMNet(nn.Module):
    """
    MLP com Dropout para Uncertainty Quantification
    
    Arquitetura: 256→256→128→128→64→64
    Input: [x, y, d, L, H, R, nu] (7 features)
    Output: [u, v] (2 deslocamentos)
    """
    
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        
        self.network = nn.Sequential(
            # Bloco 1
            nn.Linear(7, 256),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Bloco 2
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Bloco 3
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Bloco 4
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Bloco 5
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Bloco 6
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            # Output (sem ativação)
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        """Forward pass padrão"""
        return self.network(x)
    
    def mc_dropout_predict(self, x, n_samples=50, device='cuda'):
        """
        Monte Carlo Dropout para Uncertainty Quantification
        
        Args:
            x: Input tensor (N, 7)
            n_samples: Número de forward passes com dropout
            device: 'cuda' ou 'cpu'
        
        Returns:
            mean: Predição média (N, 2)
            std: Incerteza epistêmica (N, 2)
        """
        self.train()  # Mantém dropout ATIVO
        predictions = []
        
        x = x.to(device)
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self(x).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_samples, N, 2)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std
    
    def count_parameters(self):
        """Conta número de parâmetros treináveis"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Teste rápido
if __name__ == "__main__":
    model = BayesianVEMNet()
    print(f"Parâmetros treináveis: {model.count_parameters():,}")
    
    # Teste forward
    x_test = torch.randn(10, 7)
    y_test = model(x_test)
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {y_test.shape}")
    
    # Teste MC Dropout
    mean, std = model.mc_dropout_predict(x_test, n_samples=50, device='cpu')
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print("✓ Modelo validado")