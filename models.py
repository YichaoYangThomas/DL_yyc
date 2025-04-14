from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math  # Add this import


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


def off_diagonal(x):
    # 返回一个矩阵中除对角线外的所有元素
    n = x.shape[0]
    return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()


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


def vicreg_loss(x, y, sim_coef=25.0, var_coef=25.0, cov_coef=1.0):
    # 不变性损失 - 使用标准化向量的MSE损失
    sim_loss = F.mse_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    
    # 方差损失 - 确保表示中每个维度都有足够的变化
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1.0 - std_x)) + torch.mean(F.relu(1.0 - std_y))
    
    # 协方差损失 - 减少不同维度之间的相关性
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()
    
    total_loss = sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss
    return total_loss, sim_loss, var_loss, cov_loss


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn = self.conv(x)
        return x * attn


class Encoder(nn.Module):
    def __init__(self, latent_dim=128):  # 减少潜在空间维度
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 7, stride=2, padding=3),  # 减少通道数
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        
        # 简化网络结构，减少参数量
        self.layer1 = nn.Sequential(
            ResBlock(32, 64, stride=2),
        )
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2),
        )
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def create_positional_embedding(self, channels, height, width, device):
        y_embed = torch.linspace(0, 1, steps=height, device=device).unsqueeze(1).repeat(1, width)
        x_embed = torch.linspace(0, 1, steps=width, device=device).unsqueeze(0).repeat(height, 1)
        pos_embed = torch.stack((x_embed, y_embed), dim=0)
        pos_embed = pos_embed.unsqueeze(0).repeat(1, channels // 2, 1, 1)
        return pos_embed


class Predictor(nn.Module):
    def __init__(self, latent_dim=128, action_dim=2):  # 更新潜在空间维度
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.1),
            
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class JEPAModel(nn.Module):
    def __init__(self, latent_dim=128):  # 更新潜在空间维度
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.target_encoder = Encoder(latent_dim)
        self.repr_dim = latent_dim
        
        # 初始化目标编码器
        for param_q, param_k in zip(self.encoder.parameters(), 
                                  self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for param_q, param_k in zip(self.encoder.parameters(),
                                  self.target_encoder.parameters()):
            param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data
            
    def forward(self, states, actions):
        """
        During inference:
            states: [B, 1, Ch, H, W] - initial state only
            actions: [B, T-1, 2] - sequence of actions
        Returns:
            predictions: [B, T, D] - predicted representations
        """
        B = states.shape[0]
        T = actions.shape[1] + 1
        D = self.repr_dim
        
        # 获取初始嵌入
        curr_state = self.encoder(states.squeeze(1))  # [B, D]
        predictions = [curr_state]
        
        # 预测未来状态
        for t in range(T-1):
            curr_state = self.predictor(curr_state, actions[:, t])
            predictions.append(curr_state)
            
        predictions = torch.stack(predictions, dim=1)  # [B, T, D]
        return predictions
