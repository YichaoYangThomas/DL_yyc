from typing import List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 一些基础网络组件
class BasicBlock(nn.Module):
    """残差基本模块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ResNetEncoder(nn.Module):
    """ResNet编码器"""
    def __init__(self, input_channels=2, repr_dim=256, layers=[2, 2, 2, 2]):
        super().__init__()
        self.in_planes = 64
        
        # 初始层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, repr_dim)
        self.out_activation = nn.ReLU()
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
            
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始层传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层传播
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 输出层处理
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.out_activation(x)
        
        return x


class ResPredictor(nn.Module):
    """残差预测器"""
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        hidden_dim = repr_dim * 2
        
        self.fc1 = nn.Linear(repr_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, repr_dim)
        self.bn2 = nn.BatchNorm1d(repr_dim)
        self.act2 = nn.ReLU()
        
    def forward(self, repr, action):
        # 连接表示和动作
        x = torch.cat([repr, action], dim=-1)
        
        # 第一层处理
        out = self.fc1(x)
        if out.dim() > 2:
            out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.bn1(out)
        out = self.act1(out)
        
        # 第二层处理
        out = self.fc2(out)
        if out.dim() > 2:
            out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.bn2(out)
        
        # 残差连接
        out = self.act2(out + repr)
        
        return out


def build_mlp(layers_dims: List[int]):
    """构建多层感知机"""
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """测试用的假模型"""
    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    """探测器模型"""
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
    """JEPA模型实现"""
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        
        # 创建编码器和预测器
        self.encoder = ResNetEncoder(input_channels=2, repr_dim=repr_dim).to(device)
        self.predictor = ResPredictor(repr_dim=repr_dim, action_dim=action_dim).to(device)

    def forward(self, states, actions):
        """
        前向传播函数
        
        Args:
            states: [B, T, Ch, H, W] - 状态序列
            actions: [B, T-1, 2] - 动作序列
            
        Returns:
            predictions: [B, T, D] - 预测的表示序列
        """
        B, T, C, H, W = states.shape
        
        # 初始化预测序列
        predictions = []
        
        # 编码初始状态
        current_repr = self.encoder(states[:, 0])  # [B, D]
        predictions.append(current_repr.unsqueeze(1))  # [B, 1, D]
        
        # 循环预测未来表示
        for t in range(T - 1):
            action = actions[:, t]  # [B, action_dim]
            pred_repr = self.predictor(current_repr, action)  # [B, D]
            predictions.append(pred_repr.unsqueeze(1))  # [B, 1, D]
            current_repr = pred_repr  # 更新当前表示
            
        # 拼接所有预测
        predictions = torch.cat(predictions, dim=1)  # [B, T, D]
        
        return predictions

    def predict_future(self, init_states, actions):
        """
        展开模型以预测未来表示
        
        Args:
            init_states: [B, 1, Ch, H, W] - 初始状态
            actions: [B, T-1, 2] - 动作序列
            
        Returns:
            predicted_reprs: [T, B, D] - 按时间顺序预测的表示
        """
        B, _, C, H, W = init_states.shape
        T_minus1 = actions.shape[1]
        T = T_minus1 + 1
        
        predicted_reprs = []
        
        # 编码初始状态
        current_repr = self.encoder(init_states[:, 0])  # [B, D]
        predicted_reprs.append(current_repr.unsqueeze(0))  # [1, B, D]
        
        # 循环预测未来表示
        for t in range(T_minus1):
            action = actions[:, t]  # [B, action_dim]
            pred_repr = self.predictor(current_repr, action)  # [B, D]
            predicted_reprs.append(pred_repr.unsqueeze(0))  # [1, B, D]
            current_repr = pred_repr  # 更新当前表示
            
        # 拼接所有预测
        predicted_reprs = torch.cat(predicted_reprs, dim=0)  # [T, B, D]
        
        return predicted_reprs