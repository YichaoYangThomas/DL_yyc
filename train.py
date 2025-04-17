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


def apply_data_augmentation(states):
    """应用数据增强技术到状态序列"""
    B, T, C, H, W = states.shape
    augmented_states = []
    
    for batch_idx in range(B):
        # 获取单个样本的所有帧
        sample_frames = states[batch_idx]
        
        # 随机变换参数 - 对整个序列保持一致
        do_horizontal_flip = random.random() < 0.5
        do_vertical_flip = random.random() < 0.5
        rotation_angles = [0, 90, 180, 270]
        selected_angle = random.choice(rotation_angles)
        
        # 处理序列中的每一帧
        augmented_frames = []
        for time_idx in range(T):
            frame = sample_frames[time_idx]
            
            # 应用水平翻转
            if do_horizontal_flip:
                frame = VF.hflip(frame)
            
            # 应用垂直翻转
            if do_vertical_flip:
                frame = VF.vflip(frame)
            
            # 应用旋转
            frame = VF.rotate(frame, selected_angle)
            
            # 添加随机噪声
            noise = torch.randn_like(frame) * 0.01
            frame = frame + noise
            
            # 保存增强后的帧
            augmented_frames.append(frame)
        
        # 重新堆叠帧成序列
        augmented_sequence = torch.stack(augmented_frames, dim=0)
        augmented_states.append(augmented_sequence)
    
    # 重新堆叠样本成批次
    result = torch.stack(augmented_states, dim=0)
    return result


def train_model(device):
    """训练JEPA模型"""
    # 数据路径 - 使用与同学相同的路径
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
    
    # 训练配置
    num_epochs = 10
    learning_rate = 1e-4
    jepa_loss_weight = 0.2
    
    # 打印模型参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=learning_rate/10
    )
    
    print(f"开始训练，共{num_epochs}轮...")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 设置为训练模式
        model.train()
        epoch_loss = 0.0
        
        # 创建进度条
        progress_bar = tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{num_epochs}")
        
        # 处理每个批次
        for batch_idx, batch in enumerate(progress_bar):
            # 获取批次数据
            states = batch.states
            actions = batch.actions
            
            # 数据增强
            augmented_states = apply_data_augmentation(states)
            
            # 前向传播 - 获取预测的表示
            predictions = model(augmented_states, actions)
            
            # 获取真实表示（无梯度）
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
            
            # 计算JEPA损失（使用smooth L1损失而不是je_loss，这与同学的代码一致）
            repr_loss = F.smooth_l1_loss(predictions, targets)
            
            # 获取原始表示用于Barlow Twins损失
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.size(0), states.size(1), -1)
            z2 = predictions
            
            # 计算Barlow Twins损失
            regularization_loss = barlow_twins_loss(z1, z2)
            
            # 总损失
            batch_loss = jepa_loss_weight * repr_loss + (1 - jepa_loss_weight) * regularization_loss
            
            # 反向传播
            optimizer.zero_grad()
            batch_loss.backward()
            
            # 参数更新
            optimizer.step()
            
            # 记录损失
            epoch_loss += batch_loss.item()
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'jepa': f'{repr_loss.item():.4f}',
                'bt': f'{regularization_loss.item():.4f}'
            })
        
        # 学习率调度
        scheduler.step()
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"轮次 {epoch+1} 完成, 平均损失: {avg_epoch_loss:.6f}, 学习率: {scheduler.get_last_lr()[0]:.6f}")
    
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