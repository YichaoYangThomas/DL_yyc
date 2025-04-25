from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(dims: List[int]) -> nn.Sequential:
    """构建多层感知机，dims为各层维度大小列表"""
    layers = []
    for i in range(len(dims) - 2):
        layers.extend([
            nn.Linear(dims[i], dims[i+1]),
            nn.BatchNorm1d(dims[i+1]),
            nn.ReLU(True)
        ])
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class MockModel(nn.Module):
    """仅用于测试的模拟模型"""
    def __init__(self, device="cuda", batch_size=64, seq_length=17, embed_dim=256):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embed_dim = embed_dim

    def forward(self, states, actions):
        """返回随机表示，仅用于测试"""
        return torch.randn((self.batch_size, self.seq_length, self.embed_dim)).to(self.device)


class Prober(nn.Module):
    """探测器：从学习表示中提取信息"""
    def __init__(self, embed_dim: int, arch: str, output_shape: List[int]):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        
        # 解析架构字符串，构建网络层
        hidden_dims = list(map(int, arch.split("-"))) if arch else []
        layer_dims = [embed_dim] + hidden_dims + [self.output_dim]
        
        # 构建探测器网络
        layers = []
        for i in range(len(layer_dims) - 2):
            layers.extend([
                nn.Linear(layer_dims[i], layer_dims[i+1]),
                nn.ReLU(True)
            ])
        layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
        self.prober = nn.Sequential(*layers)

    def forward(self, x):
        return self.prober(x)


class Encoder(nn.Module):
    """编码器：将观察状态映射到表示空间"""
    def __init__(self, in_channels=2, input_size=(65, 65), embed_dim=256):
        super().__init__()
        
        # 构建CNN特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层：增大感受野的5x5卷积
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            
            # 后续层：常规3x3卷积
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        # 动态计算CNN输出特征数量
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *input_size)
            conv_output = self.feature_extractor(dummy_input)
            flattened_size = conv_output.numel()
        
        # 映射到表示空间
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, embed_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        embedding = self.projection(features)
        return embedding


class Predictor(nn.Module):
    """预测器：预测下一个状态表示"""
    def __init__(self, embed_dim=256, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, embedding, action):
        """预测下一状态表示"""
        combined = torch.cat([embedding, action], dim=-1)
        return self.net(combined)


class JEPAModel(nn.Module):
    """联合嵌入预测架构（JEPA）模型"""
    def __init__(self, device="cuda", embed_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.action_dim = action_dim

        # 初始化编码器和预测器
        self.encoder = Encoder(embed_dim=embed_dim).to(device)
        self.predictor = Predictor(embed_dim=embed_dim, action_dim=action_dim).to(device)

    def forward(self, states, actions):
        """
        前向传播：递归展开预测
        
        Args:
            states: [batch_size, seq_len, channels, height, width]
            actions: [batch_size, seq_len-1, action_dim]
            
        Returns:
            predictions: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len = states.shape[:2]
        
        # 编码初始状态
        current_repr = self.encoder(states[:, 0])
        predictions = [current_repr.unsqueeze(1)]
        
        # 递归预测后续状态
        for t in range(seq_len - 1):
            current_repr = self.predictor(current_repr, actions[:, t])
            predictions.append(current_repr.unsqueeze(1))
            
        # 合并所有时间步的预测
        return torch.cat(predictions, dim=1)

    def predict_future(self, init_states, actions):
        """
        展开预测未来表示序列
        
        Args:
            init_states: [batch_size, 1, channels, height, width]
            actions: [batch_size, seq_len-1, action_dim]
            
        Returns:
            predictions: [seq_len, batch_size, embed_dim]
        """
        batch_size = init_states.shape[0]
        seq_len = actions.shape[1] + 1
        
        # 编码初始状态
        current_repr = self.encoder(init_states[:, 0])
        predictions = [current_repr.unsqueeze(0)]
        
        # 递归预测未来状态
        for t in range(seq_len - 1):
            current_repr = self.predictor(current_repr, actions[:, t])
            predictions.append(current_repr.unsqueeze(0))
            
        # 沿时间维度合并预测
        return torch.cat(predictions, dim=0)