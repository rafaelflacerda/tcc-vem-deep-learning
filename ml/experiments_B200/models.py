import torch
import torch.nn as nn

class BeamNet(nn.Module):
    def __init__(self, input_dim=5, output_dim=2, hidden_dim=512, dropout_p=0.1):
        """
        Input Dim = 5: [x, E, I, q, L]
        Output Dim = 2: [w_vem, theta_vem]
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(), # Tanh é excelente para física/mecânica (suave e contínua)
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class BeamNetLarge(nn.Module):
    def __init__(self, input_dim=5, output_dim=2, hidden_dim=2048, dropout_p=0.15):
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)