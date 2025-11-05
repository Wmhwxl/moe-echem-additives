import torch
import torch.nn as nn  # Add this import at the top
from torch.utils.data import DataLoader, TensorDataset
from utils.data_processor import MoleculeDataProcessor
from experts.expert_a import TFIDFExpert
from experts.expert_b import ElementRatioExpert
from experts.expert_c import NumericalExpert
from experts.expert_d import ExpertD  # GNN结构专家
import matplotlib
matplotlib.use('Agg')   # 服务器环境无 GUI 时需要

from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import logging
import os
import pandas as pd
from torch_geometric.data import Batch   
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU seed
    torch.cuda.manual_seed(seed)  # GPU seed (if using CUDA)
    torch.cuda.manual_seed_all(seed)  # All GPUs if using multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensures deterministic results
    torch.backends.cudnn.benchmark = False  # Disables optimizations for non-deterministic results

# Call the function to set seed at the start of the training script
set_random_seed(42)
class AttentionFusion(nn.Module):
    """Attention-based fusion layer"""
    def __init__(self, expert_dim=32, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.attention = nn.Sequential(
            nn.Linear(expert_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, expert_outputs):
        # expert_outputs shape: [batch_size, num_experts, expert_dim]
        attention_weights = []
        for i in range(expert_outputs.size(1)):
            weight = self.attention(expert_outputs[:, i, :])
            attention_weights.append(weight)
            
        attention_weights = torch.stack(attention_weights, dim=1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum of expert outputs
        weighted_sum = torch.sum(expert_outputs * attention_weights, dim=1)
        return weighted_sum

class MOESystem:
    def __init__(self, tfidf_dim, expert_dim=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = MoleculeDataProcessor()
        
        # 添加GNN专家
        self.expert_d = ExpertD(input_dim=1, hidden_dim=64, output_dim=expert_dim).to(device)
        gnn_state = torch.load("best_gnn_expert.pth")
        self.expert_d.load_state_dict(gnn_state)
        self.expert_d.eval()  # 设置为评估模式
        # Initialize experts
        element_dim = len(self.processor.elements)
        physical_dim = 6  # M, α, #H, #HA, HBA, σ
        
        self.expert_a = TFIDFExpert(tfidf_dim).to(device)
        self.expert_b = ElementRatioExpert(element_dim).to(device)
        self.expert_c = NumericalExpert(physical_dim, output_dim=expert_dim).to(device)
        
        # 修改fusion layer以适应4个专家
        self.fusion = AttentionFusion(num_experts=4).to(device)
        
        # Initialize optimizers
        self.optimizer_a = torch.optim.Adam(self.expert_a.parameters(), lr=0.001)
        self.optimizer_b = torch.optim.Adam(self.expert_b.parameters(), lr=0.001)
        # self.optimizer_c = torch.optim.Adam(self.expert_c.parameters(), lr=0.001)
        self.optimizer_fusion = torch.optim.Adam(self.fusion.parameters(), lr=0.001)
        # 添加优化器
        self.optimizer_d = torch.optim.Adam(self.expert_d.parameters(), lr=0.001)
        
        # Initialize loss function
        self.criterion = torch.nn.MSELoss()
        
        # Initialize weights for experts as nn.Parameter
        self.expert_weights = nn.Parameter(torch.ones(4, device=device) / 4)
        self.weight_optimizer = torch.optim.Adam([self.expert_weights], lr=0.01)

        self.output_layer = nn.Sequential(
            nn.Linear(expert_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, 1)  # Final output dimension
        ).to(device)
        
        # Add optimizer for output layer
        self.optimizer_output = torch.optim.Adam(self.output_layer.parameters(), lr=0.001)


    # def prepare_batch(self, batch_data):
    #     """准备批次数据"""
    #     tfidf_features = torch.FloatTensor(batch_data['formula_tfidf']).to(self.device)
    #     phys_features = torch.FloatTensor(batch_data['physical_features']).to(self.device)
    #     formula_vectors = torch.FloatTensor(batch_data['formula_vectors']).to(self.device)
    #     gnn_features = torch.FloatTensor(batch_data['gnn_features']).to(self.device)
    #     targets = torch.FloatTensor(batch_data['targets']).to(self.device)
    #     return tfidf_features, phys_features, formula_vectors, gnn_features, targets
    
    def train(self):
        """设置为训练模式"""
        self.expert_a.train()
        self.expert_b.train()
        self.expert_c  # LightGBM 本就不需要 .train()
        self.expert_d.train()    
        self.fusion.train()
        
        return self
    
    def eval(self):
        """设置为评估模式"""
        self.expert_a.eval()
        self.expert_b.eval()
        self.expert_d.eval()  # GNN专家也设置为评估模式
        self.fusion.eval()
        return self
    
    def load_state_dict(self, state_dict):
        """加载模型状态"""
        if 'expert_a' in state_dict:
            self.expert_a.load_state_dict(state_dict['expert_a'])
        if 'expert_b' in state_dict:
            self.expert_b.load_state_dict(state_dict['expert_b'])
        if 'expert_d' in state_dict:
            self.expert_d.load_state_dict(state_dict['expert_d'])
        if 'fusion' in state_dict:
            self.fusion.load_state_dict(state_dict['fusion'])
        if 'output_layer' in state_dict:
            self.output_layer.load_state_dict(state_dict['output_layer'])
        logger.info("Loaded model state successfully")

    def state_dict(self):
        """获取模型状态"""
        return {
            'expert_a': self.expert_a.state_dict(),
            'expert_b': self.expert_b.state_dict(),
            'fusion': self.fusion.state_dict(),
            'output_layer': self.output_layer.state_dict()
        }

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.expert_a.train()
        self.expert_b.train()
        self.fusion.train()

        
        total_loss = 0
        for batch in train_loader:
            # 提取批次数据
            tfidf_features, element_features, physical_features, smiles_batch, targets = [b.to(self.device) for b in batch]
            
            # 获取各个专家的预测
            expert_a_out = self.expert_a(tfidf_features)  # TF-IDF专家
            expert_b_out = self.expert_b(element_features)  # 元素比例专家
            expert_c_out = self.expert_c.predict(physical_features.cpu().numpy())  # 物理特征专家
            expert_c_out = torch.FloatTensor(expert_c_out * 0.01).to(self.device)
             # 获取GNN专家预测
            # with torch.no_grad():  # GNN专家不需要训练
            expert_d_out = self.expert_d(smiles_batch)

            # expert_c_out = torch.FloatTensor(expert_c_out).unsqueeze(1).expand(-1, expert_a_out.size(1)).to(self.device)  # [batch_size, 32]
            
            # 只融合专家A和B的输出
            expert_outputs = torch.stack([expert_a_out, expert_b_out, expert_c_out, expert_d_out], dim=1)
            fused = self.fusion(expert_outputs)
            
            
            # Transform to match target dimension
            final_pred = self.output_layer(fused)
            targets = targets * 0.01
            # 计算损失
            loss = self.criterion(final_pred, targets.unsqueeze(1))
            
            # 反向传播和优化
            self.optimizer_a.zero_grad()
            self.optimizer_b.zero_grad()
            self.optimizer_d.zero_grad()
            self.optimizer_fusion.zero_grad()
            self.weight_optimizer.zero_grad()
            self.optimizer_output.zero_grad()
            
            loss.backward()
            
            self.optimizer_a.step()
            self.optimizer_b.step()
            self.optimizer_d.step()  
            self.optimizer_fusion.step()
            self.weight_optimizer.step()
            self.optimizer_output.step()
            
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """评估模型"""
        self.expert_a.eval()
        self.expert_b.eval()
        self.fusion.eval()
        self.output_layer.eval()
        # expert_d_out = gnn_model(gnn_features)

        # expert_outputs = torch.stack([expert_a_out, expert_b_out, expert_c_out, expert_d_out], dim=1)
        
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            # 提取批次数据
            tfidf_features, element_features, physical_features,smiles_batch, targets = [b.to(self.device) for b in batch]
            
            # 获取各个专家的预测
            expert_a_out = self.expert_a(tfidf_features)
            expert_b_out = self.expert_b(element_features)
            expert_c_out = self.expert_c.predict(physical_features.cpu().numpy())  # 物理特征专家
            expert_c_out = torch.FloatTensor(expert_c_out * 0.01).to(self.device)
            # expert_c_out = torch.FloatTensor(expert_c_out).unsqueeze(1).expand(-1, expert_a_out.size(1)).to(self.device)  # [batch_size, 32]
            # with torch.no_grad():  # GNN专家不需要训练
            expert_d_out = self.expert_d(smiles_batch)
            expert_outputs = torch.stack([expert_a_out, expert_b_out, expert_c_out, expert_d_out], dim=1)
            fused = self.fusion(expert_outputs)
            final_pred = self.output_layer(fused)
            
            # 计算损失
            targets = targets * 0.01
            loss = self.criterion(final_pred, targets.unsqueeze(1))
            total_loss += loss.item()
            
            all_preds.extend(final_pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # 计算评估指标
        r2 = r2_score(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        return total_loss / len(val_loader), r2, rmse

    def plot_results(self, train_losses, val_losses, r2_scores):
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot R² scores
        plt.subplot(1, 2, 2)
        plt.plot(r2_scores, label='R² Score')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.title('R² Score Evolution')
        
        plt.tight_layout()
        plt.savefig('training_results.pdf', dpi=300, bbox_inches='tight')
        plt.close()

def train_moe(num_epochs=300, batch_size=8):
    # moe = MOESystem(tfidf_dim=tfidf_dim)

    # 初始化 GNN 专家模型
    # gnn_model = ExpertD().to(moe.device)

    # Get absolute paths for data files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = current_dir  # pp folder
    train_file = os.path.join(current_dir, "data", "Data_DFTtrain_with_SMILES_with_SMILES.xlsx")
    test_file = os.path.join(current_dir, "data", "Data_DFTtest_with_SMILES.xlsx")
    # expert_d_out = gnn_model(gnn_features)

    # expert_outputs = torch.stack([expert_a_out, expert_b_out, expert_c_out, expert_d_out], dim=1)
    
    # Verify file existence
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
        
    # First process training data to get TF-IDF dimension
    processor = MoleculeDataProcessor()
    train_data = processor.process_data(train_file, is_training=True)
    if not train_data:
        raise ValueError("Training data processing failed")
    
    # Get TF-IDF dimension from processed data
    tfidf_dim = train_data['formula_tfidf'].shape[1]
    
    # Now initialize MOE system with known TF-IDF dimension
    moe = MOESystem(tfidf_dim=tfidf_dim)
    
    # Process test data
    test_data = processor.process_data(test_file, is_training=False)
    if not test_data:
        raise ValueError("Test data processing failed")
    
    # # Load and process data with validation
    # logger.info("Loading training data...")
    # train_data = moe.processor.process_data(train_file, is_training=True)
    # if not train_data or any(v is None or len(v) == 0 for v in train_data.values()):
    #     raise ValueError("Training data is empty or invalid. Check data processing.")
    
    # logger.info("Loading test data...")
    # test_data = moe.processor.process_data(test_file, is_training=False)
    # if not test_data or any(v is None or len(v) == 0 for v in test_data.values()):
    #     raise ValueError("Test data is empty or invalid. Check data processing.")
    
    # Print data statistics
    logger.info(f"Training data size: {len(train_data['formula_tfidf'])} samples")
    logger.info(f"Test data size: {len(test_data['formula_tfidf'])} samples")
    logger.info(f"Feature dimensions:")
    logger.info(f"  - TF-IDF features: {train_data['formula_tfidf'].shape}")
    logger.info(f"  - Element ratio features: {train_data['formula_vectors'].shape}")
    logger.info(f"  - Physical features: {train_data['physical_features'].shape}")
    logger.info(f"SMILES features batch size: {train_data['smiles_features'].num_graphs}")
    logger.info(f"Targets shape: {train_data['targets'].shape}")
    
    # 确保所有特征的样本数量一致
    n_samples = len(train_data['targets'])
    assert len(train_data['formula_tfidf']) == n_samples, "TF-IDF features size mismatch"
    assert len(train_data['formula_vectors']) == n_samples, "Formula vectors size mismatch"
    assert len(train_data['physical_features']) == n_samples, "Physical features size mismatch"
    assert train_data['smiles_features'].num_graphs == n_samples, "SMILES features size mismatch"
    
    # 创建自定义数据集类来处理 Batch 对象
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data_dict, is_train=True):
            self.tfidf = torch.FloatTensor(data_dict['formula_tfidf'])
            self.formula = torch.FloatTensor(data_dict['formula_vectors'])
            self.physical = torch.FloatTensor(data_dict['physical_features'])
            self.smiles = data_dict['smiles_features']
            self.is_train = is_train
            if is_train:
                self.targets = torch.FloatTensor(data_dict['targets'])

        def __len__(self):
            return len(self.tfidf)

        def __getitem__(self, idx):
            if self.is_train:
                return (
                    self.tfidf[idx],
                    self.formula[idx],
                    self.physical[idx],
                    self.smiles[idx],
                    self.targets[idx]
                )
            else:
                return (
                    self.tfidf[idx],
                    self.formula[idx],
                    self.physical[idx],
                    self.smiles[idx]
                )

    # ---------- collate for TRAIN / VAL (含 target) ----------
    def geom_collate_train(batch):
        tfidf, formula, phys, smiles, target = zip(*batch)
        return (
            torch.stack(tfidf),
            torch.stack(formula),
            torch.stack(phys),
            Batch.from_data_list(list(smiles)),
            torch.stack(target),
        )

    # ---------- collate for TEST / PREDICT (无 target) ----------
    def geom_collate_test(batch):
        tfidf, formula, phys, smiles = zip(*batch)
        return (
            torch.stack(tfidf),
            torch.stack(formula),
            torch.stack(phys),
            Batch.from_data_list(list(smiles)),
        )




    # 使用自定义数据集
    train_dataset = CustomDataset(train_data, is_train=True)
    test_dataset = CustomDataset(test_data, is_train=False)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=geom_collate_train,      # <-- 加这一行
    )

    test_loader  = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=geom_collate_test,      # <-- 加这一行
    )


    # Training tracking
    train_losses = []
    train_r2s = []
    train_rmses = []
    val_predictions = []
    val_targets = []
    best_model_state = None
    best_epoch = -1

    logger.info("Pre-training LightGBM expert...")
    moe.expert_c.fit(train_data['physical_features'], train_data['targets'])

    best_r2 = -1.0          # <== 新增：记录历史最高 R²

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Train
        moe.train()
        train_loss = moe.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # Validation predictions
        moe.eval()
        epoch_preds = []
        epoch_targets = []
        
        with torch.no_grad():
            for batch in train_loader:  # 使用训练集评估
                tfidf_features, element_features, physical_features, smiles_batch, targets = [b.to(moe.device) for b in batch]
                
                expert_a_out = moe.expert_a(tfidf_features)
                expert_b_out = moe.expert_b(element_features)
                expert_c_out = moe.expert_c.predict(physical_features.cpu().numpy())
                expert_c_out = torch.FloatTensor(expert_c_out * 0.01).to(moe.device)
                expert_d_out = moe.expert_d(smiles_batch)
                expert_outputs = torch.stack([expert_a_out, expert_b_out, expert_c_out, expert_d_out], dim=1)
                fused = moe.fusion(expert_outputs)
                final_pred = moe.output_layer(fused)
                
                epoch_preds.extend(final_pred.cpu().numpy())
                epoch_targets.extend((targets * 0.01).cpu().numpy())
        
        # 计算每个epoch的R²和RMSE
        epoch_r2 = r2_score(epoch_targets, epoch_preds)
        epoch_rmse = np.sqrt(mean_squared_error(epoch_targets, epoch_preds))
        train_r2s.append(epoch_r2)
        train_rmses.append(epoch_rmse)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, R²: {epoch_r2:.4f}, RMSE: {epoch_rmse:.4f}")
        
        # Save best model
        if epoch_r2 > best_r2:               # ❶ 只要更高就覆盖；用 >= 则最后一次相等也会记录
            best_r2    = epoch_r2            # ❷ 先更新基准
            best_epoch = epoch               # ❸ 记录 epoch
            best_model_state = {             # ❹ 保存权重
                'expert_a'   : moe.expert_a.state_dict(),
                'expert_b'   : moe.expert_b.state_dict(),
                'expert_c'   : moe.expert_c,          # LightGBM 直接存引用
                'expert_d'   : moe.expert_d.state_dict(),
                'fusion'     : moe.fusion.state_dict(),
                'output_layer': moe.output_layer.state_dict(),
                'best_r2'    : best_r2,
                'epoch'      : best_epoch,
            }
            # 保存 best_preds 和 best_targets
            best_preds = epoch_preds.copy()
            best_targets = epoch_targets.copy()

            torch.save(best_model_state,
                    os.path.join(output_dir, 'best_model.pth'))
            
    # Generate predictions for test data
    moe.load_state_dict(best_model_state)
    moe.eval()
    
    # best_preds, best_targets = [], []
    # with torch.no_grad():
    #     for batch in train_loader:           # 或 val_loader，看你想画哪集
    #         tfidf, elem, phys, smiles, targets = [b.to(moe.device) for b in batch]

    #         expert_outputs = torch.stack([
    #             moe.expert_a(tfidf),
    #             moe.expert_b(elem),
    #             torch.FloatTensor(
    #                 moe.expert_c.predict(phys.cpu().numpy()) * 0.01
    #             ).to(moe.device),
    #             moe.expert_d(smiles)
    #         ], dim=1)

    #         fused = moe.fusion(expert_outputs)
    #         pred  = moe.output_layer(fused)

    #         best_preds.extend(pred.cpu().numpy())
    #         best_targets.extend((targets * 0.01).cpu().numpy())
    # sorted_idx = np.argsort(best_targets)
    # sorted_tgt = np.array(best_targets)[sorted_idx]
    # sorted_pre = np.array(best_preds)[sorted_idx]
    best_r2   = r2_score(best_targets, best_preds)
    best_rmse = np.sqrt(mean_squared_error(best_targets, best_preds))

# 绘制最终的性能图
    plt.figure(figsize=(15, 5))
    
    # 训练损失曲线
    plt.subplot(131)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.legend()
    
    # R²进展曲线
    plt.subplot(132)
    plt.plot(train_r2s, label='R² Score')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Score Evolution')
    plt.legend()
    
    # RMSE进展曲线
    plt.subplot(133)
    plt.plot(train_rmses, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))

    sorted_idx = np.argsort(best_targets)
    sorted_tgt = np.array(best_targets)[sorted_idx]
    sorted_pre = np.array(best_preds)[sorted_idx]
    x_vals     = np.arange(len(sorted_tgt))

    plt.scatter(x_vals, sorted_tgt,  alpha=0.5, c='blue',  label='True', s=30)
    plt.scatter(x_vals, sorted_pre,  alpha=0.5, c='red',   label='Pred', s=30)
    plt.plot(x_vals, sorted_tgt, 'b-', alpha=0.5, label='True Trend')
    plt.plot(x_vals, sorted_pre, 'r-', alpha=0.5, label='Pred Trend')

    plt.xlabel('Sample Index (sorted by True)')
    plt.ylabel('Values')
    plt.title(
        f'MOE Performance (Best Epoch = {best_epoch})\n'
        f'R² = {best_r2:.3f}   RMSE = {best_rmse:.3f}'
    )
    plt.legend(); plt.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_performance.pdf'), dpi=600)
    plt.close()
    

    epoch_preds = []
    epoch_targets = []
    test_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 提取批次数据
            tfidf_features, element_features, physical_features, smiles_batch = [b.to(moe.device) for b in batch]
            
            # 获取各个专家的预测
            expert_a_out = moe.expert_a(tfidf_features)
            expert_b_out = moe.expert_b(element_features)
            expert_c_out = moe.expert_c.predict(physical_features.cpu().numpy())
            expert_c_out = torch.FloatTensor(expert_c_out * 0.01).to(moe.device)
            expert_d_out = moe.expert_d(smiles_batch)  # GNN专家预测
            
            # 融合专家输出
            expert_outputs = torch.stack([expert_a_out, expert_b_out, expert_c_out, expert_d_out], dim=1)

            fused = moe.fusion(expert_outputs)
            final_pred = moe.output_layer(fused)

            epoch_preds.extend(final_pred.cpu().numpy())
            
            
            test_predictions.extend(final_pred.cpu().numpy())

    # 还原预测值的比例
    test_predictions = np.array(test_predictions) * 100  # 因为之前乘了0.01

    # Save test predictions to pp folder
    test_df = pd.read_excel(test_file)
    test_df['Predicted_γ'] = test_predictions
    test_df.to_excel(os.path.join(output_dir, "predictions.xlsx"), index=False)
    
    # Plot training loss and save to pp folder
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Evolution')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_loss.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    # Print final weights from attention fusion
    logger.info("\nTraining completed!")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best R²  : {best_model_state['best_r2']:.4f}")



if __name__ == "__main__":
    train_moe()