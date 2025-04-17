from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 基本残差模块 - 与同学代码类似的实现
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


# ResNet编码器 - 参考同学的ResNet实现
class ResNet(nn.Module):
    def __init__(self, input_channels=2, repr_dim=256, block=BasicBlock, layers=[2, 2, 2, 2]):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, repr_dim)
        self.out_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.out_relu(x)
        return x


# 带残差连接的预测器 - 参考同学的ResPredictor实现
class ResPredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(repr_dim + action_dim, repr_dim)
        self.fc2 = nn.Linear(repr_dim, repr_dim)
        self.relu = nn.ReLU()

    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        # 关键点：添加残差连接并应用ReLU激活
        out = self.relu(out + repr)
        return out


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


class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        
        # 使用ResNet编码器和ResPredictor
        self.encoder = ResNet(input_channels=2, repr_dim=repr_dim).to(device)
        self.predictor = ResPredictor(repr_dim=repr_dim, action_dim=action_dim).to(device)

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
        
        # 存储所有时间步的预测
        predictions = []
        
        # 处理初始状态
        current_repr = self.encoder(states[:, 0])  # [B, D]
        predictions.append(current_repr.unsqueeze(1))  # [B, 1, D]
        
        # 循环预测未来状态
        for t in range(T - 1):
            action = actions[:, t]  # [B, action_dim]
            # 预测下一个表示
            next_repr = self.predictor(current_repr, action)  # [B, D]
            predictions.append(next_repr.unsqueeze(1))  # [B, 1, D]
            # 更新当前表示
            current_repr = next_repr
        
        # 拼接所有预测
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