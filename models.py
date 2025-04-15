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
    # 不变性损失 (使用Huber损失，对异常值更鲁棒)
    sim_loss = F.huber_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1), delta=0.1)
    
    # 方差损失 (增强到2.5以获得更好的特征分布)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(2.5 - std_x)) + torch.mean(F.relu(2.5 - std_y))
    
    # 协方差损失 (使用归一化特征)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()
    
    total_loss = sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss
    return total_loss, sim_loss, var_loss, cov_loss


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1, use_se=True):
        super().__init__()
        self.use_se = use_se
        
        # 改进的残差块结构
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        # 添加SE注意力模块
        if use_se:
            self.se = SEBlock(out_channels)
            
        # 添加随机深度
        self.survival_prob = 0.8
        self.apply_stochastic_depth = True if dropout_rate > 0 else False

    def forward(self, x):
        # 随机深度：在训练时随机跳过整个块
        if self.apply_stochastic_depth and self.training and torch.rand(1).item() > self.survival_prob:
            return self.shortcut(x)
            
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
            
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.2)
        out = self.dropout2(out)
        return out


class SEBlock(nn.Module):
    """挤压-激励注意力模块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """增强的空间注意力模块"""
    def __init__(self, channels):
        super().__init__()
        # 多尺度特征融合
        self.conv_7x7 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.conv_5x5 = nn.Conv2d(2, 1, kernel_size=5, padding=2)
        self.conv_3x3 = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conv_fuse = nn.Conv2d(3, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # 多尺度特征
        attn1 = self.conv_7x7(spatial_features)
        attn2 = self.conv_5x5(spatial_features)
        attn3 = self.conv_3x3(spatial_features)
        
        # 融合多尺度特征
        attn = torch.cat([attn1, attn2, attn3], dim=1)
        attn = self.conv_fuse(attn)
        attn = self.sigmoid(attn)
        
        return x * attn


class Encoder(nn.Module):
    def __init__(self, latent_dim=160):  # 进一步减小潜在维度，避免过拟合
        super().__init__()
        self.latent_dim = latent_dim
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 48, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2, True),
            SpatialAttention(48),  # 添加空间注意力
            nn.Dropout2d(0.1)
        )
        
        # 引入Stochastic Depth
        self.layer1 = nn.Sequential(
            ResBlock(48, 96, stride=2, dropout_rate=0.1, use_se=True),
            LayerScale(96),  # 添加层级缩放
            SpatialAttention(96)
        )
        
        self.layer2 = nn.Sequential(
            ResBlock(96, 192, stride=2, dropout_rate=0.15, use_se=True),
            LayerScale(192),
            AttentionModule(192, num_heads=4)  # 多头注意力
        )
        
        self.layer3 = nn.Sequential(
            ResBlock(192, 320, stride=2, dropout_rate=0.2, use_se=True),
            LayerScale(320),
            AttentionModule(320, num_heads=8)  # 多头注意力
        )
        
        # 全局信息集成
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveMaxPool2d((1, 1))
        )
        
        # 投影头部
        self.fc = nn.Sequential(
            nn.Linear(320*2, latent_dim*2),  # 使用平均池化和最大池化的结果
            nn.LayerNorm(latent_dim*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(latent_dim*2, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.2)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.conv1(x)
        B, C, H, W = x.size()
        device = x.device
        
        # 生成位置编码
        pos_embed = self.create_positional_embedding(C, H, W, device)
        x = x + pos_embed
        
        # 特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局信息集成（结合平均池化和最大池化）
        avg_feat = self.global_pool[0](x).view(B, -1)
        max_feat = self.global_pool[1](x).view(B, -1)
        x = torch.cat([avg_feat, max_feat], dim=1)
        
        # 特征投影
        x = self.fc(x)
        return x

    def create_positional_embedding(self, channels, height, width, device):
        """创建增强的位置编码"""
        # 生成网格位置
        y_embed = torch.linspace(-1, 1, steps=height, device=device).unsqueeze(1).repeat(1, width)
        x_embed = torch.linspace(-1, 1, steps=width, device=device).unsqueeze(0).repeat(height, 1)
        
        # 计算径向距离
        r = torch.sqrt(x_embed.pow(2) + y_embed.pow(2))
        
        # 基础位置编码 - 5个通道
        base_embed = torch.stack([x_embed, y_embed, r, torch.sin(r * math.pi), torch.cos(r * math.pi)], dim=0)
        
        # 确保位置编码的通道数与输入特征的通道数匹配
        if channels <= 5:
            # 如果需要的通道数少于5，只使用前几个通道
            pos_embed = base_embed[:channels]
        else:
            # 如果需要的通道数多于5，重复基础编码并截断到所需通道数
            repeats = (channels + 4) // 5
            pos_embed = base_embed.repeat(repeats, 1, 1)[:channels]
        
        # 增加批次维度
        pos_embed = pos_embed.unsqueeze(0)  # [1, channels, H, W]
        
        return pos_embed


class Predictor(nn.Module):
    def __init__(self, latent_dim=160, action_dim=2):
        super().__init__()
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 320),
            nn.LayerNorm(320),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),  # 增加dropout率
            
            nn.Linear(320, 320),
            nn.LayerNorm(320),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            
            nn.Linear(320, 320),  # 添加额外层，增强表达能力
            nn.LayerNorm(320),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            
            nn.Linear(320, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # 碰撞预测头部
        self.collision_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 动作调整网络
        self.action_adjust = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出[-1,1]范围的调整系数
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, state, action):
        # 特征融合
        x = torch.cat([state, action], dim=-1)
        feat = self.net(x)
        
        # 预测碰撞概率
        collision = self.collision_head(feat)
        
        # 动作调整
        action_adj = self.action_adjust(torch.cat([feat, action], dim=-1))
        scaled_action = action * (1.0 - 0.9 * collision) + action_adj * 0.1
        
        # 使用调整后的动作重新计算
        x = torch.cat([state, scaled_action], dim=-1)
        return self.net(x)


class JEPAModel(nn.Module):
    def __init__(self, latent_dim=160):
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
        
        # 添加投影头 - 用于对抗训练
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )
            
    @torch.no_grad()
    def update_target(self, momentum=0.99):
        for param_q, param_k in zip(self.encoder.parameters(),
                                  self.target_encoder.parameters()):
            param_k.data = momentum * param_k.data + (1.0 - momentum) * param_q.data
            
    def forward(self, states, actions):
        """
        训练/推理通用前向传播
        """
        B = states.shape[0]
        if states.dim() == 5:  # [B, T, C, H, W] 格式
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
        else:  # [B, C, H, W] 格式
            return self.encoder(states)


class LayerScale(nn.Module):
    """层级缩放，增强训练稳定性"""
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        
    def forward(self, x):
        # 将gamma扩展为适合x形状的张量
        if x.dim() == 4:  # 对于卷积特征图 [B, C, H, W]
            return self.gamma.view(1, -1, 1, 1) * x
        elif x.dim() == 2:  # 对于全连接层特征 [B, C]
            return self.gamma.view(1, -1) * x
        else:
            # 对于其他维度，尝试智能广播
            shape = [1] * x.dim()
            shape[1] = -1  # 假设第1维是通道维度
            return self.gamma.view(*shape) * x


class AttentionModule(nn.Module):
    """多头自注意力模块"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels, "channels必须能被num_heads整除"
        
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.1)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.Dropout2d(0.1)
        )
        
    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        
        # Self-attention
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h*w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [B, num_heads, head_dim, H*W]
        
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        k = k.permute(0, 1, 2, 3)  # [B, num_heads, head_dim, H*W]
        v = v.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        
        attn = torch.matmul(q, k) * (self.head_dim ** -0.5)  # [B, num_heads, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        out = self.proj(out)
        out = self.dropout(out)
        out = self.norm1(out + residual)
        
        # Feed Forward Network
        residual = out
        out = self.ffn(out)
        out = self.norm2(out + residual)
        
        return out
