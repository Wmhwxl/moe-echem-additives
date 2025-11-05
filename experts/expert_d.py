import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch, Dataset
from rdkit import Chem
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, targets):
        super().__init__()
        self.graphs = []
        self.targets = []
        
        for smiles, target in zip(smiles_list, targets):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.targets.append(target)
                
    def len(self):
        return len(self.graphs)
        
    def get(self, idx):
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.targets[idx]], dtype=torch.float)
        return graph

class ExpertD(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=32):
        super(ExpertD, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Third GCN layer
        x = self.conv3(x, edge_index)
        
        # Global pooling
        return global_mean_pool(x, batch_idx)

def train_gnn_expert(train_file, test_file, device='cuda'):
    # 读取数据
    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)
    
    # 准备数据集
    train_dataset = MoleculeDataset(
        train_df['SMILES'].values,
        train_df['γ (meV/A2)'].values * 0.01  # 缩放到相同范围
    )
    
    # 创建数据加载器
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 初始化模型
    model = ExpertD(input_dim=1, hidden_dim=64, output_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_loss = float('inf')
    patience = 15
    counter = 0

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_save_path = os.path.join(parent_dir, 'best_gnn_expert.pth')
    
    for epoch in range(200):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            output = model(batch)
            targets = batch.y.unsqueeze(1).expand(-1, output.size(1))  # [batch_size, output_dim]
            loss = criterion(output, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        
        # 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            # 使用完整路径保存最佳模型
            torch.save(model.state_dict(), model_save_path)
            # print(f"Saved best model to: {model_save_path}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break
    
    return model
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edge_index = [[], []]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [a, b]
        edge_index[1] += [b, a]
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

# 在主训练流程中使用
if __name__ == "__main__":
    import os
    
    # Get the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')
    
    # Create full paths to data files
    train_file = os.path.join(data_dir, 'Data_DFTtrain_with_SMILES_with_SMILES.xlsx')
    test_file = os.path.join(data_dir, 'Data_DFTtest_with_SMILES.xlsx')
    
    # Verify file existence
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_gnn_expert(
        train_file=train_file,
        test_file=test_file,
        device=device
    )




