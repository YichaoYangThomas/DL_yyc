from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 使用更强的CBAM注意力模块(通道+空间双重注意力)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, in_planes, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)  # 应用通道注意力
        x = x * self.sa(x)  # 应用空间注意力
        return x


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
        
        # CBAM注意力模块 - 更强大的注意力机制
        self.cbam1 = CBAM(32, kernel_size=5)  # 第一个CBAM注意力
        self.cbam2 = CBAM(128, kernel_size=7)  # 第二个CBAM注意力
        
        # 添加批归一化以提高训练稳定性
        self.conv_net = nn.Sequential(
            # 第一层使用更大的卷积核(5x5)以增大初始感受野
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),  # 添加批归一化
            nn.ReLU(),
            # 第一个注意力模块在这里应用（外部应用，不在Sequential中）
            nn.Dropout2d(0.1),  # 在第一层后添加少量空间Dropout
            
            # 第二层保持原有参数
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),  # 添加批归一化
            nn.ReLU(),
            
            # 第三层保持原有参数
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),  # 添加批归一化
            nn.ReLU(),
            # 第二个注意力模块在这里应用（外部应用，不在Sequential中）
            nn.Dropout2d(0.1),  # 在第三层后添加少量空间Dropout
            
            # 第四层保持原有参数
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),  # 添加批归一化
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, *input_size)
            conv_output = self.conv_net(sample_input)
            conv_output_size = conv_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),  # 在特征平铺后添加Dropout，减少过拟合
            nn.Linear(conv_output_size, repr_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # 分步执行卷积层并应用注意力机制
        # 第一层卷积+注意力
        x = self.conv_net[0](x)  # Conv2d
        x = self.conv_net[1](x)  # BatchNorm
        x = self.conv_net[2](x)  # ReLU
        x = self.cbam1(x)        # CBAM注意力
        x = self.conv_net[3](x)  # Dropout2d
        
        # 第二层卷积
        x = self.conv_net[4](x)  # Conv2d
        x = self.conv_net[5](x)  # BatchNorm
        x = self.conv_net[6](x)  # ReLU
        
        # 第三层卷积+注意力
        x = self.conv_net[7](x)  # Conv2d
        x = self.conv_net[8](x)  # BatchNorm
        x = self.conv_net[9](x)  # ReLU
        x = self.cbam2(x)        # CBAM注意力
        x = self.conv_net[10](x) # Dropout2d
        
        # 第四层卷积
        x = self.conv_net[11](x) # Conv2d
        x = self.conv_net[12](x) # BatchNorm
        x = self.conv_net[13](x) # ReLU
        
        # 全连接层
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        # 改进的MLP，添加残差连接
        self.input_proj = nn.Linear(repr_dim + action_dim, repr_dim)
        self.act1 = nn.ReLU()
        self.hidden = nn.Linear(repr_dim, repr_dim)
        self.act2 = nn.ReLU()
        
        # 增加层次以增强表达能力
        self.norm = nn.LayerNorm(repr_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.hidden.weight)

    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        # 第一层投影
        h = self.input_proj(x)
        h = self.act1(h)
        
        # 第二层投影+残差连接
        res = h
        h = self.hidden(h)
        h = h + res  # 残差连接
        h = self.norm(h)  # 标准化
        h = self.act2(h)
        
        return h


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