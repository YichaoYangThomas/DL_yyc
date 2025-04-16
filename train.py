import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import numpy as np
import os
from dataset import create_wall_dataloader
from models import EnhancedJEPAModel


def je_loss(predictions, targets):
    """计算表示之间的余弦相似度损失"""
    similarities = F.cosine_similarity(predictions, targets, dim=-1)
    loss_per_sample = 1 - similarities
    loss_per_sample = loss_per_sample.sum(dim=1)
    loss = loss_per_sample.mean()
    return loss


def mse_loss(predictions, targets, reduction='mean'):
    """带权重的均方误差损失函数"""
    # 计算每个样本和每个时间步的MSE
    squared_diff = torch.pow(predictions - targets, 2)
    # 按特征维度求和，得到每个时间步的损失
    per_sample_per_step_loss = squared_diff.mean(dim=-1)  # [B, T]
    
    if reduction == 'none':
        return per_sample_per_step_loss
    elif reduction == 'mean':
        return per_sample_per_step_loss.mean()
    elif reduction == 'sum':
        return per_sample_per_step_loss.sum()
    else:
        raise ValueError(f"不支持的reduction类型: {reduction}")


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """Barlow Twins损失函数，用于防止表示坍塌"""
    # 形状：b, t, d
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    
    # 标准化每个特征维度
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # 计算批次中样本的相关性矩阵
    batch_size = z1_norm.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # 对角线应该是1（完全相关），离对角线应该是0（不相关）
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].pow_(2).sum()
    
    # 总损失
    loss = on_diag + lambda_param * off_diag
    return loss


def vicreg_loss(z1, z2, sim_coef=25.0, std_coef=25.0, cov_coef=1.0):
    """VICReg损失函数，用于防止表示坍塌"""
    B, T, D = z1.shape
    z1_flat = z1.reshape(-1, D)
    z2_flat = z2.reshape(-1, D)
    
    # 计算不变性损失（均方差）
    sim_loss = F.mse_loss(z1_flat, z2_flat)
    
    # 计算方差损失，鼓励每个变量有高方差
    std_z1 = torch.sqrt(z1_flat.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2_flat.var(dim=0) + 1e-4)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    
    # 计算协方差损失，鼓励变量之间不相关
    N = z1_flat.shape[0]
    z1_centered = z1_flat - z1_flat.mean(dim=0)
    z2_centered = z2_flat - z2_flat.mean(dim=0)
    cov_z1 = (z1_centered.T @ z1_centered) / (N - 1)
    cov_z2 = (z2_centered.T @ z2_centered) / (N - 1)
    
    # 删除对角线上的元素（自相关）
    mask = ~torch.eye(D, device=z1_flat.device).bool()
    cov_loss = (cov_z1[mask].pow_(2).sum() / D + cov_z2[mask].pow_(2).sum() / D) / 2
    
    # 总损失
    loss = sim_coef * sim_loss + std_coef * std_loss + cov_coef * cov_loss
    return loss


def apply_augmentation(states, advanced=False):
    """对状态应用数据增强"""
    B, T, C, H, W = states.shape
    aug_states = []
    
    for b in range(B):
        sample = states[b]
        
        # 基本增强
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        
        # 高级增强选项
        brightness_factor = 1.0
        contrast_factor = 1.0
        if advanced:
            # 亮度和对比度调整
            brightness_factor = random.uniform(0.9, 1.1)
            contrast_factor = random.uniform(0.9, 1.1)
        
        aug_frames = []
        for t in range(T):
            frame = sample[t]
            
            # 应用基本增强
            if do_hflip:
                frame = VF.hflip(frame)
            if do_vflip:
                frame = VF.vflip(frame)
            frame = VF.rotate(frame, angle)
            
            # 应用高级增强
            if advanced:
                # 亮度和对比度变化（小心处理二值图像）
                if frame.max() > 1:  # 非二值图像
                    frame = VF.adjust_brightness(frame, brightness_factor)
                    frame = VF.adjust_contrast(frame, contrast_factor)
            
            # 添加少量噪声
            noise = torch.randn_like(frame) * 0.01
            frame = torch.clamp(frame + noise, 0, 1) if frame.max() <= 1 else frame + noise
            
            aug_frames.append(frame)
            
        aug_frames = torch.stack(aug_frames, dim=0)
        aug_states.append(aug_frames)
        
    aug_states = torch.stack(aug_states, dim=0)
    return aug_states


def train_model(device, model_args=None):
    """训练JEPA模型"""
    # 数据加载
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
    )
    
    # 模型配置
    if model_args is None:
        model_args = {
            'repr_dim': 256,
            'action_dim': 2,
            'encoder_type': 'resnet',
            'predictor_type': 'residual'
        }
    
    # 创建模型
    model = EnhancedJEPAModel(
        device=device,
        repr_dim=model_args['repr_dim'],
        action_dim=model_args['action_dim'],
        encoder_type=model_args['encoder_type'],
        predictor_type=model_args['predictor_type']
    ).to(device)
    
    # 训练参数
    num_epochs = 10
    learning_rate = 3e-4
    weight_decay = 1e-5
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs * len(train_loader),
        eta_min=learning_rate * 0.1
    )
    
    # 模型名称（用于保存）
    model_name = f"jepa_{model_args['encoder_type']}_{model_args['predictor_type']}"
    best_loss = float('inf')
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{num_epochs}")):
            states = batch.states
            actions = batch.actions
            
            # 应用数据增强
            use_advanced_aug = random.random() < 0.3
            states_aug = apply_augmentation(states, advanced=use_advanced_aug)
            
            # 前向传播 - 自回归预测
            predictions = model(states_aug, actions)
            
            # 计算目标表示
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.shape[0], states.shape[1], -1)
            
            # 计算JEPA损失（平滑L1或MSE）
            use_smooth_l1 = random.random() < 0.5
            if use_smooth_l1:
                jepa_loss = F.smooth_l1_loss(predictions, targets)
            else:
                jepa_loss = mse_loss(predictions, targets)
            
            # 计算Barlow Twins或VICReg损失来防止表示坍塌
            use_vicreg = random.random() < 0.5
            
            # 重新计算表示（有梯度）
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.shape[0], states.shape[1], -1)
            z2 = predictions
            
            if use_vicreg:
                reg_loss = vicreg_loss(z1, z2)
            else:
                reg_loss = barlow_twins_loss(z1, z2)
            
            # 结合损失
            jep_coef = 0.3  # 改变了损失权重
            total_loss_batch = jep_coef * jepa_loss + (1-jep_coef) * reg_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            total_loss += total_loss_batch.item()
            
            # 打印批次损失（间隔）
            if batch_idx % 50 == 0:
                print(f"批次 [{batch_idx}/{len(train_loader)}], 损失: {total_loss_batch.item():.6f}")
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"轮次 [{epoch+1}/{num_epochs}], 损失: {avg_loss:.8e}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{model_name}_best.pth')
            print(f"保存最佳模型，损失: {best_loss:.8e}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'model_weights.pth')
    print("训练完成，已保存模型权重")
    
    return model


def get_device():
    """获取设备（GPU/CPU）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 获取设备
    device = get_device()
    
    # 模型参数
    model_config = {
        'repr_dim': 256,
        'action_dim': 2,
        'encoder_type': 'resnet',  # 可以是 'resnet' 或 'vit'
        'predictor_type': 'residual'  # 可以是 'residual' 或 'recurrent'
    }
    
    # 训练模型
    model = train_model(device, model_config)