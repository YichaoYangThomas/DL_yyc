import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
import numpy as np
from dataset import create_wall_dataloader
from models import JEPAModel


def je_loss(predictions, targets):
    """计算基于余弦相似度的损失"""
    similarities = F.cosine_similarity(predictions, targets, dim=-1)
    loss_per_sample = 1 - similarities
    loss_per_sample = loss_per_sample.sum(dim=1)
    loss = loss_per_sample.mean()
    return loss


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """计算Barlow Twins损失"""
    # 获取维度
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    
    # 标准化表示
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # 计算相关矩阵
    batch_size = z1_norm.shape[0]
    corr_matrix = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # 计算对角线和非对角线损失
    diag_loss = torch.diagonal(corr_matrix).add_(-1).pow_(2).sum()
    off_diag_loss = corr_matrix.flatten()[:-1].view(corr_matrix.size(0) - 1, corr_matrix.size(1) + 1)[:, 1:].pow_(2).sum()
    
    # 合并损失
    loss = diag_loss + lambda_param * off_diag_loss
    return loss


def train_model(device):
    """训练JEPA模型"""
    # 修改为同学使用的数据路径
    data_path = "/scratch/DL25SP/train"
    print(f"加载训练数据... 路径: {data_path}")
    
    # 创建数据加载器
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device=device,
        train=True,
    )
    
    # 创建模型
    print("初始化模型...")
    model = JEPAModel(device=device).to(device)
    
    # 训练配置 - 简化为与同学一致
    num_epochs = 10
    learning_rate = 1e-4
    jepa_loss_weight = 0.8
    
    # 打印模型参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {trainable_params:,}")
    
    # 优化器 - 移除权重衰减，与同学保持一致
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"开始训练，共{num_epochs}轮...")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 设置为训练模式
        model.train()
        total_loss = 0.0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{num_epochs}")
        
        # 处理每个批次
        for batch_idx, batch in enumerate(progress_bar):
            # 获取批次数据
            states = batch.states
            actions = batch.actions
            
            # 前向传播 - 获取预测的表示 (不使用数据增强)
            predictions = model(states, actions)
            
            # 获取真实表示（无梯度）
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
            
            # 计算JEPA损失 (使用smooth L1损失)
            jepa_loss = F.smooth_l1_loss(predictions, targets)
            
            # 获取原始表示用于Barlow Twins损失
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.size(0), states.size(1), -1)
            z2 = predictions
            
            # 计算Barlow Twins损失
            bt_loss = barlow_twins_loss(z1, z2)
            
            # 总损失 - 采用同学的权重分配
            total_loss_batch = jepa_loss_weight * jepa_loss + (1 - jepa_loss_weight) * bt_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 参数更新
            optimizer.step()
            
            # 记录损失
            total_loss += total_loss_batch.item()
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"轮次 {epoch+1} 完成, 平均损失: {avg_loss:.8e}")
    
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
    # 设置随机种子确保可重复性
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
    
    # 获取设备
    device = get_device()
    
    # 开始训练
    model = train_model(device)