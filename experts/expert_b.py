import torch
import torch.nn as nn

class ElementRatioExpert(nn.Module):
    """Expert 2: Element ratio vector with Residual MLP"""
    def __init__(self, input_dim):
        super().__init__()
        
        self.input_layer = nn.Linear(input_dim, 64)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(3)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        return self.output_layer(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(0.2),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.block(x)  # Residual connection