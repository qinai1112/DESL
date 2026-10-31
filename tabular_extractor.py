import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class TabularFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, out_dim=512, dropout=0.3):
        """
        input_dim: 输入指标维度（原始指标 + 缺失掩码）
        hidden_dim: 中间层大小
        out_dim: 输出维度
        dropout: Dropout 概率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (batch, input_dim)
        if x.size(0) < 2:
            # 避免 batch size 为 1 时 BatchNorm1d 崩溃
            x = self.net[0](x)
            x = torch.relu(x)
            x = self.net[3](x)
            x = self.net[4](x)
            x = torch.relu(x)
            x = self.net[6](x)
            x = self.net[7](x)
            return x
        return self.net(x)


def preprocess_tabular(df, indicator_cols):
    """
    预处理表格指标：
    1. 特殊变换：对抗体类指标进行 Log 变换和截断
    2. 用中位数填充缺失值
    3. 生成缺失掩码 (Missing Mask)
    4. 标准化 (Z-Score)
    """
    df_processed = df[indicator_cols].copy()
    
    # 🚨【核心修改1】针对抗体指标的特殊变换逻辑
    # 医疗经验：抗体类数值波动极大，建议进行 Log 变换和 99 分位数截断
    antibody_cols = ["TgAb", "TPOAb"]
    for col in antibody_cols:
        if col in df_processed.columns:
            # a. 极端值处理：将超过 99% 分位数的极高值截断（盖帽法）
            upper_limit = df_processed[col].quantile(0.99)
            df_processed[col] = df_processed[col].clip(upper=upper_limit)
            
            # b. Log1p 变换：处理偏态分布，将指数级差异线性化
            # 使用 log(x + 1) 防止零值导致报错
            df_processed[col] = np.log1p(df_processed[col])

    # 🚨【核心修改2】生成 Missing Mask
    # 为每个指标生成掩码（1 代表缺失，0 代表存在）
    masks = df_processed.isna().astype(float)
    masks.columns = [f"{c}_mask" for c in indicator_cols]

    # 🚨【核心修改3】中位数填充（基于训练集统计规律）
    medians = df_processed.median()
    df_processed = df_processed.fillna(medians)

    # 🚨【核心修改4】标准化 (Standardization)
    means = df_processed.mean()
    stds = df_processed.std().replace(0, 1) # 防止除零
    df_normalized = (df_processed - means) / stds

    # 拼接原始特征和掩码
    # 最终维度：len(indicator_cols) * 2
    combined = pd.concat([df_normalized, masks], axis=1)
    
    return combined.values, medians.to_dict()