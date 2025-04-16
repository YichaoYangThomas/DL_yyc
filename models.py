from typing import List, Optional
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from encs import ResNet, build_resnet
from preds import ResPredictor, RNNPredictor


def build_mlp(layer_sizes: List[int]) -> nn.Sequential:
    """构建多层感知机网络"""
    network_layers = []
    for idx in range(len(layer_sizes) - 2):
        network_layers.append(nn.Linear(layer_sizes[idx], layer_sizes[idx + 1]))
        network_layers.append(nn.BatchNorm1d(layer_sizes[idx + 1]))
        network_layers.append(nn.ReLU(True))
    network_layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    return nn.Sequential(*network_layers)


class MockModel(torch.nn.Module):
    """
    仅用于测试的模拟模型
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


class Encoder(nn.Module):
    def __init__(self, in_channels=2, img_size=(65, 65), feature_dim=256, hidden_dim=256):
        super().__init__()
        # 卷积网络部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # 计算卷积后的特征尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            conv_features = self.conv_layers(dummy_input)
            flattened_size = conv_features.view(1, -1).size(1)

        # 全连接层部分
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, feature_dim),
            nn.ReLU(),
        )

    def forward(self, input_tensor):
        features = self.conv_layers(input_tensor)
        output = self.fc_layers(features)
        return output


class Predictor(nn.Module):
    def __init__(self, feature_dim=256, action_dim=2):
        super().__init__()
        # 多层感知机预测器
        self.network = nn.Sequential(
            nn.Linear(feature_dim + action_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, state_repr, action):
        combined = torch.cat([state_repr, action], dim=-1)
        return self.network(combined)


class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        # 模型组件
        self.encoder = Encoder(feature_dim=repr_dim).to(device)
        self.predictor = Predictor(feature_dim=repr_dim, action_dim=action_dim).to(device)

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
        batch_size, seq_len, channels, height, width = states.shape
        device = states.device

        # 存储预测结果
        all_predictions = []
        
        # 编码初始状态
        current_state = self.encoder(states[:, 0])  # [B, D]
        all_predictions.append(current_state.unsqueeze(1))  # [B, 1, D]

        # 循环预测后续状态
        for step in range(seq_len - 1):
            current_action = actions[:, step]
            next_state = self.predictor(current_state, current_action)
            all_predictions.append(next_state.unsqueeze(1))
            current_state = next_state  # 更新当前状态

        # 合并所有时间步的预测
        result = torch.cat(all_predictions, dim=1)  # [B, T, D]

        return result

    def predict_future(self, init_states, actions):
        """
        根据初始状态和动作序列展开预测未来表示。

        Args:
            init_states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Returns:
            predicted_reprs: [T, B, D]
        """
        batch_size, _, channels, height, width = init_states.shape
        action_steps = actions.shape[1]
        total_steps = action_steps + 1

        # 存储预测结果
        predicted_states = []

        # 处理初始状态
        current_state = self.encoder(init_states[:, 0])  # [B, D]
        predicted_states.append(current_state.unsqueeze(0))  # [1, B, D]

        # 循环展开预测
        for step in range(action_steps):
            current_action = actions[:, step]  # [B, action_dim]
            # 预测下一个状态表示
            next_state = self.predictor(current_state, current_action)  # [B, D]
            predicted_states.append(next_state.unsqueeze(0))  # [1, B, D]
            # 更新为下一个状态
            current_state = next_state

        # 合并所有时间步的预测
        result = torch.cat(predicted_states, dim=0)  # [T, B, D]
        return result