import pandas as pd
import numpy as np
import torch
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import itertools
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)

class MoleculeDataProcessor:
    def __init__(self):
        self.elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Na']
        # 创建一个包含所有可能的字符组合的基础词汇表
        self.selected_features = ['AVE_PE', 'AVE_IP', 'M', '#HA', 'HBD', 'σ']
        self.feature_selection_reason = {
            'M': '分子质量直接影响分子的物理性质',
            'α': '极化率与电子分布相关',
            '#H': '氢原子数影响分子间相互作用',
            '#HA': '重原子数反映分子大小',
            'HBA': '氢键受体数影响分子间作用',
            'σ': '表面张力与界面性质相关'
        }
        self.base_vocab = set()
        for elem in self.elements:
            for i in range(1, 4):  # ngram_range=(1, 3)
                for combo in [''.join(p) for p in itertools.product(elem + '0123456789', repeat=i)]:
                    self.base_vocab.add(combo)
        
        # 使用固定词汇表初始化TfidfVectorizer
        self.tfidf = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            vocabulary=self.base_vocab,
            lowercase=False  # 保持大小写敏感
        )
        self.fitted_tfidf = False
    
    def _parse_formula(self, formula):
        """解析化学式，返回元素计数字典"""
        pattern = r'([A-Z][a-z]*)(\d*)'
        matches = re.findall(pattern, formula)
        counts = {}
        for element, count in matches:
            counts[element] = int(count) if count else 1
        return counts
    
    def _formula_to_vector(self, formula):
        """将化学式转换为元素比例向量"""
        counts = self._parse_formula(formula)
        total_atoms = sum(counts.values())
        
        # 创建元素比例向量
        vector = []
        for element in self.elements:
            ratio = counts.get(element, 0) / total_atoms
            vector.append(ratio)
            
        return vector
    
    def process_data(self, file_path, is_training=True):
        """处理Excel文件中的数据"""
        try:
            # 读取Excel文件
            df = pd.read_excel(file_path)
            logger.info(f"读取了 {len(df)} 行数据")
            
            # 基本数据验证
            required_columns = ['Formula', 'AVE_PE', 'AVE_IP', 'M', '#HA', 'HBD', 'σ']
            if is_training:
                required_columns.append('γ (meV/A2)')
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必需的列: {missing_cols}")
            
            # 使用固定词汇表的TF-IDF转换
            formulas = df['Formula'].values
            if is_training:
                formula_tfidf = self.tfidf.fit_transform(formulas).toarray()
                self.fitted_tfidf = True
            else:
                formula_tfidf = self.tfidf.transform(formulas).toarray()
            
            # 创建元素比例特征
            formula_vectors = np.array([self._formula_to_vector(f) for f in formulas])
            
            # 提取物理特征
            physical_features = df[['M', 'α', '#H', '#HA', 'HBA', 'σ']].values
            
            # 处理SMILES数据
            from experts.expert_d import smiles_to_graph
            smiles_list = df['SMILES'].values
            graphs = []
            valid_indices = []  # 记录有效的样本索引
            
            for idx, smiles in enumerate(smiles_list):
                try:
                    graph = smiles_to_graph(smiles)
                    if graph is not None:
                        graphs.append(graph)
                        valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"处理SMILES出错: {smiles}, 错误: {str(e)}")
                    continue
            
            if not graphs:
                raise ValueError("没有有效的分子图数据")
            
            # 只保留有效的样本
            valid_indices = np.array(valid_indices)
            formula_tfidf = formula_tfidf[valid_indices]
            formula_vectors = formula_vectors[valid_indices]
            physical_features = physical_features[valid_indices]
            
            # 转换为batch
            batch = Batch.from_data_list(graphs)
            
            if is_training:
                targets = df['γ (meV/A2)'].values[valid_indices]
            else:
                targets = None
                
            # 验证维度匹配
            n_samples = len(valid_indices)
            logger.info(f"有效样本数量: {n_samples}")
            logger.info(f"TF-IDF特征维度: {formula_tfidf.shape}")
            logger.info(f"元素比例特征维度: {formula_vectors.shape}")
            logger.info(f"物理特征维度: {physical_features.shape}")
            logger.info(f"分子图数量: {batch.num_graphs}")
            
            assert formula_tfidf.shape[0] == n_samples
            assert formula_vectors.shape[0] == n_samples
            assert physical_features.shape[0] == n_samples
            assert batch.num_graphs == n_samples
            
            return {
                'formula_tfidf': formula_tfidf,
                'formula_vectors': formula_vectors,
                'physical_features': physical_features,
                'smiles_features': batch,
                'targets': targets
            }
            
        except Exception as e:
            logger.error(f"处理文件出错 {file_path}: {str(e)}")
            raise