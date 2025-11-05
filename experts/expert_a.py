import torch
import torch.nn as nn

class TFIDFExpert(nn.Module):
    """Expert 1: TF-IDF chemical formula vector with MLP"""
    def __init__(self, tfidf_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(tfidf_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # 输出维度为32，便于后续融合
        )
    
    def forward(self, x):
        return self.network(x)