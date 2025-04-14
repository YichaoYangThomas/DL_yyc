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
    def __init__(self, latent_dim=256):  # 增加潜在空间维度
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3),  # 增加初始通道数
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        
        # 增加ResBlock数量，深化网络
        self.layer1 = nn.Sequential(
            ResBlock(64, 128, stride=2),
            ResBlock(128, 128, stride=1),  # 添加额外ResBlock
            SelfAttention(128),  # 添加自注意力层
        )
        self.layer2 = nn.Sequential(
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256, stride=1),  # 添加额外ResBlock
            SelfAttention(256),  # 添加自注意力层
        )
        self.layer3 = nn.Sequential(
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512, stride=1),  # 添加额外ResBlock
            SelfAttention(512),  # 添加自注意力层
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 添加非线性投影头，替代简单的线性层
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x

    def create_positional_embedding(self, channels, height, width, device):
        y_embed = torch.linspace(0, 1, steps=height, device=device).unsqueeze(1).repeat(1, width)
        x_embed = torch.linspace(0, 1, steps=width, device=device).unsqueeze(0).repeat(height, 1)
        pos_embed = torch.stack((x_embed, y_embed), dim=0)
        pos_embed = pos_embed.unsqueeze(0).repeat(1, channels // 2, 1, 1)
        return pos_embed


class Predictor(nn.Module):
    def __init__(self, latent_dim=256, action_dim=2, hidden_dim=512):
        super().__init__()
        # 初始特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
        )
        
        # 添加GRU层捕捉时序依赖
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 输出层
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, state, action):
        batch_size = state.shape[0]
        
        # 每次forward都创建新的隐藏状态，避免重复使用计算图
        hidden = torch.zeros(2, batch_size, self.hidden_dim, device=state.device)
            
        # 连接状态和动作
        x = torch.cat([state, action], dim=-1)
        
        # 特征提取
        x = self.feature_net(x)
        
        # 添加序列维度
        x = x.unsqueeze(1)  # [B, 1, H]
        
        # GRU处理 - 注意这里不再保存hidden状态到self.hidden
        x, _ = self.gru(x, hidden)
        x = x.squeeze(1)  # [B, H]
        
        # 自注意力权重
        attn_weight = self.attention(x)
        attn_weight = torch.sigmoid(attn_weight)
        
        # 应用注意力
        x = x * attn_weight
        
        # 生成输出
        out = self.output_net(x)
        
        return out


class JEPAModel(nn.Module):
    def __init__(self, latent_dim=256):  # 更新潜在空间维度
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.predictor = Predictor(latent_dim)
        self.target_encoder = Encoder(latent_dim)
        self.repr_dim = latent_dim
        
        # 添加解码器用于重建任务（辅助任务）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 2 * 65 * 65),  # 更新为图像的实际尺寸2x65x65
        )
        
        # 初始化目标编码器
        for param_q, param_k in zip(self.encoder.parameters(), 
                                  self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        # 初始目标动量值
        self.base_momentum = 0.99
        self.curr_momentum = self.base_momentum
            
    @torch.no_grad()
    def update_target(self, momentum=None, step=None, total_steps=None):
        """动态动量更新策略"""
        if momentum is not None:
            self.curr_momentum = momentum
        elif step is not None and total_steps is not None:
            # 随着训练进度逐渐增加动量值
            self.curr_momentum = self.base_momentum + (0.999 - self.base_momentum) * (step / total_steps)
        
        for param_q, param_k in zip(self.encoder.parameters(),
                                  self.target_encoder.parameters()):
            param_k.data = self.curr_momentum * param_k.data + (1.0 - self.curr_momentum) * param_q.data
    
    def reconstruct(self, z):
        """重建输入图像作为辅助任务"""
        batch_size = z.shape[0]
        recon = self.decoder(z)
        recon = recon.view(batch_size, 2, 65, 65)  # 更新为图像的实际尺寸
        return recon
    
    def compute_contrastive_loss(self, z1, z2, temperature=0.1):
        """计算对比损失作为辅助任务"""
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 余弦相似度矩阵
        similarity_matrix = torch.matmul(z1, z2.T) / temperature
        
        # 正例是对角线上的元素
        positives = torch.diag(similarity_matrix)
        
        # 对所有样本计算总损失
        nll = -positives + torch.logsumexp(similarity_matrix, dim=1)
        
        # 平均损失
        nll = torch.mean(nll)
        return nll
            
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

# 添加自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 获取查询、键、值
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C/8
        key = self.key(x).view(batch_size, -1, height * width)  # B x C/8 x HW
        value = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        
        # 计算注意力权重
        energy = torch.bmm(query, key)  # B x HW x HW
        attention = F.softmax(energy, dim=2)
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        return out
