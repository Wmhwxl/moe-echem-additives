import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np

class NumericalExpert(nn.Module):
    """Expert 3: Numerical features with LightGBM"""
    def __init__(self, input_dim, output_dim=32):
        super().__init__()
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 8,  # 减少叶子数量
            'learning_rate': 0.1,
            'min_data_in_leaf': 3,  # 添加最小数据量限制
            'feature_fraction': 1.0,  # 使用所有特征
            'verbose': -1  # 减少输出
        }
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def fit(self, X, y):
        # 训练多个LightGBM模型，每个对应一个输出维度
        import lightgbm as lgb
        self.lgb_models = []
        
        # 使用原始目标值训练第一个模型
        train_data = lgb.Dataset(X, label=y)
        model = lgb.train(self.lgb_params, train_data, num_boost_round=50)
        self.lgb_models.append(model)
        
        # 对剩余维度，使用残差训练
        for i in range(1, self.output_dim):
            residual = y - model.predict(X)
            train_data = lgb.Dataset(X, label=residual)
            model = lgb.train(self.lgb_params, train_data, num_boost_round=50)
            self.lgb_models.append(model)
    
    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        # 获取每个模型的预测
        predictions = []
        for model in self.lgb_models:
            pred = model.predict(x)
            predictions.append(pred)
            
        # 堆叠所有预测 [batch_size, output_dim]
        return np.stack(predictions, axis=1)