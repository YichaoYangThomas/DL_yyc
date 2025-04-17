from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_channels=2, input_size=(65, 65), repr_dim=256, projection_hidden_dim=256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, *input_size)
            conv_output = self.conv_net(sample_input)
            conv_output_size = conv_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, repr_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        # 使用更复杂的MLP结构，逐渐扩大隐藏层
        self.mlp = nn.Sequential(
            # 第一层：从输入压缩到16维
            nn.Linear(repr_dim + action_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # 第二层：从16维扩展到32维
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # 第三层：从32维扩展到64维
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 最后一层：从64维回到原始表示维度
            nn.Linear(64, repr_dim)
        )
        
    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        batch_size = x.shape[0]
        # 确保输入形状适合BatchNorm1d
        if len(x.shape) > 2:
            x = x.view(-1, x.size(-1))
        out = self.mlp(x)
        # 如果需要，恢复原始形状
        if len(repr.shape) > 2:
            out = out.view(batch_size, -1, repr.size(-1))
        return out


class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        self.encoder = Encoder().to(device)
        self.predictor = Predictor().to(device)

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
        B, T, C, H, W = states.shape
        device = states.device

        predictions = []
        current_repr = self.encoder(states[:, 0])  # [B, D]
        predictions.append(current_repr.unsqueeze(1))  # [B, 1, D]

        for t in range(T - 1):
            action = actions[:, t]
            pred_repr = self.predictor(current_repr, action)
            predictions.append(pred_repr.unsqueeze(1))
            current_repr = pred_repr  # Update current representation with prediction

        predictions = torch.cat(predictions, dim=1)  # [B, T, D]

        return predictions

    def predict_future(self, init_states, actions):
        """
        根据初始状态和动作序列展开预测未来表示。

        Args:
            init_states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Returns:
            predicted_reprs: [T, B, D]
        """
        B, _, C, H, W = init_states.shape
        T_minus1 = actions.shape[1]
        T = T_minus1 + 1
        
        # 存储预测结果
        predicted_reprs = []
        
        # 初始状态
        current_repr = self.encoder(init_states[:, 0])  # [B, D]
        predicted_reprs.append(current_repr.unsqueeze(0))  # [1, B, D]
        
        # 循环预测
        for t in range(T_minus1):
            action = actions[:, t]  # [B, action_dim]
            # 预测下一个表示
            next_repr = self.predictor(current_repr, action)  # [B, D]
            predicted_reprs.append(next_repr.unsqueeze(0))  # [1, B, D]
            # 更新当前表示
            current_repr = next_repr
        
        # 拼接所有预测
        predicted_reprs = torch.cat(predicted_reprs, dim=0)  # [T, B, D]
        
        return predicted_reprs