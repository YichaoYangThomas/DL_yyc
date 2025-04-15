from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.functional as F
from encs import ResNet, build_resnet
from preds import ResPredictor, RNNPredictor


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class ImprovedEncoder(nn.Module):
    def __init__(self, input_channels=2, input_size=(64, 64), repr_dim=256):
        super().__init__()
        # 使用残差连接增强特征提取
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, *input_size)
            conv_output = self.conv_net(sample_input)
            conv_output_size = conv_output.view(1, -1).size(1)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU(),
        )
        
        # 添加投影头以增强表示
        self.projection = nn.Sequential(
            nn.Linear(repr_dim, repr_dim),
            nn.LayerNorm(repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim)
        )
    
    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc(x)
        x = self.projection(x)
        return x


class ImprovedPredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=512):
        super().__init__()
        # 使用更复杂的网络结构
        self.action_encoder = nn.Linear(action_dim, hidden_dim // 4)
        self.state_encoder = nn.Linear(repr_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout防止过拟合
            nn.Linear(hidden_dim, repr_dim),
            nn.LayerNorm(repr_dim)
        )
        
        # 残差连接
        self.res_layer = nn.Linear(repr_dim, repr_dim)
    
    def forward(self, repr, action):
        # 增强动作和状态的编码
        action_enc = self.action_encoder(action)
        state_enc = self.state_encoder(repr)
        
        # 融合状态和动作
        combined = torch.cat([state_enc, action_enc], dim=-1)
        output = self.mlp(combined)
        
        # 添加残差连接，提高梯度流动
        res = self.res_layer(repr)
        output = output + res
        
        return output


class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        
        # 使用改进的编码器和预测器
        self.encoder = ImprovedEncoder(repr_dim=repr_dim).to(device)
        self.predictor = ImprovedPredictor(repr_dim=repr_dim, action_dim=action_dim).to(device)
        
        # EMA更新目标编码器
        self.target_encoder = ImprovedEncoder(repr_dim=repr_dim).to(device)
        # 初始化目标编码器为与主编码器相同的权重
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.momentum = 0.99  # EMA动量参数
    
    def _update_target_encoder(self):
        # 使用EMA更新目标编码器
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def forward(self, states, actions):
        """
        训练时的前向传播
        Args:
            states: [B, T, Ch, H, W]
            actions: [B, T-1, 2]
        Output:
            predictions: [B, T, D]
        """
        B, T, C, H, W = states.shape
        
        predictions = []
        current_repr = self.encoder(states[:, 0])  # [B, D]
        predictions.append(current_repr.unsqueeze(1))  # [B, 1, D]
        
        for t in range(T - 1):
            action = actions[:, t]
            pred_repr = self.predictor(current_repr, action)
            predictions.append(pred_repr.unsqueeze(1))
            current_repr = pred_repr  # 使用预测的表示更新当前表示
        
        predictions = torch.cat(predictions, dim=1)  # [B, T, D]
        
        return predictions
    
    def get_target_representations(self, states):
        """
        获取目标编码器的表示
        """
        B, T, C, H, W = states.shape
        states_flat = states.view(-1, C, H, W)
        target_repr = self.target_encoder(states_flat)
        return target_repr.view(B, T, -1)
    
    def predict_future(self, init_states, actions):
        """
        评估时的递归展开预测
        Args:
            init_states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]
        Returns:
            predicted_reprs: [T, B, D]
        """
        B, _, C, H, W = init_states.shape
        T_minus1 = actions.shape[1]
        T = T_minus1 + 1
        
        predicted_reprs = []
        
        # 初始状态
        current_repr = self.encoder(init_states[:, 0])  # [B, D]
        predicted_reprs.append(current_repr.unsqueeze(0))  # [1, B, D]
        
        for t in range(T_minus1):
            action = actions[:, t]  # [B, action_dim]
            # 预测下一个表示
            pred_repr = self.predictor(current_repr, action)  # [B, D]
            predicted_reprs.append(pred_repr.unsqueeze(0))  # [1, B, D]
            # 用预测的表示更新当前表示
            current_repr = pred_repr
        
        predicted_reprs = torch.cat(predicted_reprs, dim=0)  # [T, B, D]
        return predicted_reprs