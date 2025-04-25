import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from dataset import create_wall_dataloader
from models import JEPAModel


def cosine_similarity_loss(predictions, targets):
    """基于余弦相似度的损失函数"""
    similarities = F.cosine_similarity(predictions, targets, dim=-1)
    # 将相似度转换为距离 (1-相似度)，并对每个序列求和，再取均值
    return (1 - similarities).sum(dim=1).mean()


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """Barlow Twins损失函数，防止表示坍缩"""
    # 平铺批次和时间维度
    batch_size, seq_len, embed_dim = z1.shape
    z1_flat = z1.reshape(-1, embed_dim)
    z2_flat = z2.reshape(-1, embed_dim)
    
    # 标准化
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # 计算相关矩阵
    n_samples = z1_norm.shape[0]
    corr_matrix = torch.matmul(z1_norm.T, z2_norm) / n_samples
    
    # 计算对角线和非对角线损失
    diag_loss = torch.diagonal(corr_matrix).add_(-1).pow_(2).sum()
    off_diag_loss = corr_matrix.flatten()[:-1].view(corr_matrix.size(0)-1, corr_matrix.size(1)+1)[:, 1:].pow_(2).sum()
    
    return diag_loss + lambda_param * off_diag_loss


def train_model(device, data_path="/scratch/DL25SP/train", num_epochs=10, learning_rate=1e-4, 
               jepa_weight=0.2, seed=42):
    """训练JEPA模型
    
    Args:
        device: 训练设备
        data_path: 数据路径
        num_epochs: 训练轮数
        learning_rate: 学习率
        jepa_weight: JEPA损失权重
        seed: 随机种子
    
    Returns:
        训练好的模型
    """
    print(f"加载训练数据，路径: {data_path}")
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    # 创建数据加载器
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device=device,
        train=True,
    )
    
    # 初始化模型
    print("初始化模型...")
    model = JEPAModel(device=device).to(device)
    
    # 打印模型参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {num_params:,}")
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"开始训练，共{num_epochs}轮...")
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{num_epochs}")
        
        # 批次训练
        for batch in progress_bar:
            states = batch.states  # [B,T,C,H,W]
            actions = batch.actions  # [B,T-1,2]
            
            # 前向传播
            predictions = model(states, actions)  # [B,T,D]
            
            # 计算目标表示 (无梯度)
            with torch.no_grad():
                batch_size, seq_len = states.shape[:2]
                flat_states = states.view(-1, *states.shape[2:])  # [B*T,C,H,W]
                encoded_states = model.encoder(flat_states)  # [B*T,D]
                targets = encoded_states.view(batch_size, seq_len, -1)  # [B,T,D]
            
            # 计算JEPA损失 (smooth L1损失)
            jepa_loss = F.smooth_l1_loss(predictions, targets)
            
            # 计算Barlow Twins损失 (防止表示坍缩)
            # 这里直接使用targets而不是重新编码，减少计算
            bt_loss = barlow_twins_loss(predictions, targets)
            
            # 总损失
            loss = jepa_weight * jepa_loss + (1 - jepa_weight) * bt_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'jepa': f'{jepa_loss.item():.4f}',
                'bt': f'{bt_loss.item():.4f}'
            })
        
        # 输出每轮平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"轮次 {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.8e}")
    
    # 保存模型
    print("保存模型...")
    torch.save(model.state_dict(), 'model_weights.pth')
    print("训练完成!")
    
    return model


def get_device():
    """获取可用计算设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


if __name__ == "__main__":
    device = get_device()
    model = train_model(device)