from typing import List, Tuple, Optional
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch

# ResNet相关实现（从encs.py整合）
class BasicBlock(nn.Module):
    # 残差基本模块
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """基于ResNet的编码器实现"""
    def __init__(self, input_channels: int = 2, repr_dim: int = 256, block=BasicBlock, layers: List[int] = [2, 2, 2, 2]):
        super().__init__()
        self.in_planes = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, repr_dim)
        self.out_activation = nn.ReLU(inplace=False)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
            
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        
        for _ in range(1, num_blocks):
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
        x = self.out_activation(x)
        
        return x

# 用于构建ResNet编码器的辅助函数
def build_resnet(input_channels=2, repr_dim=256):
    """构建ResNet编码器"""
    return ResNet(input_channels=input_channels, repr_dim=repr_dim, block=BasicBlock, layers=[2, 2, 2, 2])


# ViT 相关组件
class PatchEmbed(nn.Module):
    """图像分块嵌入"""
    def __init__(self, img_size=65, patch_size=5, in_channels=2, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # B, C, H/p, W/p
        x = x.flatten(2)  # B, C, N
        x = x.transpose(1, 2)  # B, N, C
        return x


class MultiheadSelfAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim=256, num_heads=8, dropout_rate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 加权聚合
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, dropout_rate=0.0):
        super().__init__()
        # Layer Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 多头自注意力
        self.attn = MultiheadSelfAttention(embed_dim, num_heads, dropout_rate)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        # 自注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """视觉Transformer编码器"""
    def __init__(self, 
                 img_size: int = 65, 
                 patch_size: int = 5, 
                 in_channels: int = 2, 
                 embed_dim: int = 256, 
                 depth: int = 6, 
                 num_heads: int = 8, 
                 mlp_ratio: int = 4,
                 dropout_rate: float = 0.1, 
                 use_cls_token: bool = True):
        super().__init__()
        # 图像分块嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # 使用cls token还是全局平均池化
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            
        # 位置编码初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout_rate)
        
        # Transformer编码器块
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        B, N, C = x.shape
        
        # 添加cls token（如果使用）
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        # 添加位置编码
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        
        # 通过Transformer编码器块
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        # 输出表示
        if self.use_cls_token:
            x = x[:, 0, :]  # 使用cls token
        else:
            x = x.mean(dim=1)  # 使用全局平均池化
            
        x = self.head(x)
        return x


# 预测器相关实现（从preds.py整合）
class ResidualPredictor(nn.Module):
    """使用残差连接的预测器"""
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(repr_dim + action_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, repr_dim)
        self.bn2 = nn.BatchNorm1d(repr_dim)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        out = self.fc1(x)
        if out.dim() > 2:
            out = self.bn1(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.bn1(out)
        out = self.act(out)
        
        out = self.fc2(out)
        if out.dim() > 2:
            out = self.bn2(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.bn2(out)
        out = self.act(out + repr)  # 残差连接
        
        return out


class RecurrentPredictor(nn.Module):
    """基于RNN的预测器"""
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=256, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(repr_dim + action_dim, hidden_dim)
        self.rnn = nn.GRU(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_projection = nn.Linear(hidden_dim, repr_dim)
        self.layer_norm = nn.LayerNorm(repr_dim)
        
    def forward(self, repr, action):
        # 连接表示和动作
        combined = torch.cat([repr, action], dim=-1).unsqueeze(1)  # [B, 1, repr_dim+action_dim]
        
        # 输入投影
        projected_input = self.input_projection(combined)
        
        # RNN处理
        rnn_output, _ = self.rnn(projected_input)
        
        # 输出投影
        output = self.output_projection(rnn_output.squeeze(1))
        
        # 残差连接和层归一化
        output = self.layer_norm(output + repr)
        
        return output


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


class EnhancedJEPAModel(nn.Module):
    """增强版JEPA模型，可配置不同的编码器和预测器"""
    def __init__(self, 
                 device="cuda", 
                 repr_dim=256, 
                 action_dim=2,
                 encoder_type="resnet",
                 predictor_type="residual"):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        
        # 选择编码器类型
        if encoder_type == "resnet":
            self.encoder = build_resnet(input_channels=2, repr_dim=repr_dim).to(device)
        elif encoder_type == "vit":
            self.encoder = VisionTransformer(in_channels=2, embed_dim=repr_dim).to(device)
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
            
        # 选择预测器类型
        if predictor_type == "residual":
            self.predictor = ResidualPredictor(repr_dim=repr_dim, action_dim=action_dim).to(device)
        elif predictor_type == "recurrent":
            self.predictor = RecurrentPredictor(repr_dim=repr_dim, action_dim=action_dim).to(device)
        else:
            raise ValueError(f"不支持的预测器类型: {predictor_type}")

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
        
        # 初始化预测序列
        predictions = []
        
        # 编码初始状态
        current_repr = self.encoder(states[:, 0])  # [B, D]
        predictions.append(current_repr.unsqueeze(1))  # [B, 1, D]
        
        # 自回归预测未来表示
        for t in range(T - 1):
            action = actions[:, t]  # [B, action_dim]
            # 预测下一个表示
            pred_repr = self.predictor(current_repr, action)  # [B, D]
            predictions.append(pred_repr.unsqueeze(1))  # [B, 1, D]
            # 更新当前表示
            current_repr = pred_repr
            
        # 连接所有预测
        predictions = torch.cat(predictions, dim=1)  # [B, T, D]
        
        return predictions

    def predict_future(self, init_states, actions):
        """
        展开模型预测未来表示。

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
        
        # 编码初始状态
        current_repr = self.encoder(init_states[:, 0])  # [B, D]
        predicted_reprs.append(current_repr.unsqueeze(0))  # [1, B, D]
        
        # 自回归预测
        for t in range(T_minus1):
            action = actions[:, t]  # [B, action_dim]
            # 预测下一个表示
            pred_repr = self.predictor(current_repr, action)  # [B, D]
            predicted_reprs.append(pred_repr.unsqueeze(0))  # [1, B, D]
            # 更新当前表示
            current_repr = pred_repr
            
        # 连接所有预测
        predicted_reprs = torch.cat(predicted_reprs, dim=0)  # [T, B, D]
        
        return predicted_reprs


# 向后兼容的JEPAModel类
class JEPAModel(EnhancedJEPAModel):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__(
            device=device, 
            repr_dim=repr_dim, 
            action_dim=action_dim,
            encoder_type="resnet",
            predictor_type="residual"
        )