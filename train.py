import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
import numpy as np
from dataset import create_wall_dataloader
from models import JEPAModel


def barlow_twins_loss(z1, z2, lambda_coef=0.005):
    """计算Barlow Twins损失"""
    # 获取维度
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    
    # 标准化处理
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # 计算相关矩阵
    batch_size = z1_norm.shape[0]
    corr_matrix = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # 计算损失组件
    on_diag_loss = torch.diagonal(corr_matrix).add_(-1).pow_(2).sum()
    off_diag_loss = corr_matrix.flatten()[:-1].view(corr_matrix.size(0) - 1, corr_matrix.size(1) + 1)[:, 1:].pow_(2).sum()
    
    # 组合损失
    loss = on_diag_loss + lambda_coef * off_diag_loss
    return loss


def apply_data_augmentation(states):
    """应用数据增强方法"""
    B, T, C, H, W = states.shape
    augmented_batch = []
    
    for batch_idx in range(B):
        # 获取单个样本
        sample = states[batch_idx]
        
        # 随机决定变换参数
        do_horizontal_flip = random.random() < 0.5
        do_vertical_flip = random.random() < 0.5
        rotation_angle = random.choice([0, 90, 180, 270])
        
        # 处理每个时间步
        transformed_frames = []
        for time_idx in range(T):
            # 获取单帧
            frame = sample[time_idx]
            
            # 应用变换
            if do_horizontal_flip:
                frame = VF.hflip(frame)
            if do_vertical_flip:
                frame = VF.vflip(frame)
            frame = VF.rotate(frame, rotation_angle)
            
            # 添加噪声
            noise_level = 0.01
            noise = torch.randn_like(frame) * noise_level
            frame = torch.clamp(frame + noise, 0, 1) if frame.max() <= 1 else frame + noise
            
            # 保存变换后的帧
            transformed_frames.append(frame)
        
        # 堆叠时间维度
        sequence = torch.stack(transformed_frames, dim=0)
        augmented_batch.append(sequence)
    
    # 堆叠批次维度
    result = torch.stack(augmented_batch, dim=0)
    return result


def train_model(device):
    """训练模型"""
    # 配置参数
    print("准备训练数据...")
    data_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
    )
    
    # 初始化模型
    print("构建模型...")
    model = JEPAModel(device=device).to(device)
    
    # 训练配置
    training_epochs = 10
    base_lr = 1e-4
    weight_decay_rate = 1e-5
    jepa_loss_weight = 0.2
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay_rate
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=training_epochs * len(data_loader),
        eta_min=base_lr * 0.1
    )
    
    # 训练主循环
    print(f"开始训练, 共{training_epochs}轮...")
    lowest_loss = float('inf')
    
    for epoch in range(training_epochs):
        # 设置训练模式
        model.train()
        epoch_loss = 0
        
        # 创建进度条
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{training_epochs}")
        
        # 批次训练
        for batch_idx, batch_data in enumerate(progress_bar):
            # 提取批次数据
            state_batch = batch_data.states
            action_batch = batch_data.actions
            
            # 数据增强
            augmented_states = apply_data_augmentation(state_batch)
            
            # 前向传播
            predicted_repr = model(augmented_states, action_batch)
            
            # 计算目标表示
            with torch.no_grad():
                target_repr = model.encoder(
                    state_batch.view(-1, *state_batch.shape[2:])
                ).view(state_batch.shape[0], state_batch.shape[1], -1)
            
            # 计算JEPA损失
            repr_loss = F.smooth_l1_loss(predicted_repr, target_repr)
            
            # 计算正则化损失
            z1 = model.encoder(
                state_batch.view(-1, *state_batch.shape[2:])
            ).view(state_batch.shape[0], state_batch.shape[1], -1)
            z2 = predicted_repr
            reg_loss = barlow_twins_loss(z1, z2)
            
            # 总损失
            batch_loss = jepa_loss_weight * repr_loss + (1 - jepa_loss_weight) * reg_loss
            
            # 反向传播
            optimizer.zero_grad()
            batch_loss.backward()
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            epoch_loss += batch_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'repr': f'{repr_loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.6f}")
        
        # 保存最佳模型
        if avg_epoch_loss < lowest_loss:
            lowest_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'model_weights.pth')
            print(f"保存最佳模型 (损失: {lowest_loss:.6f})")
    
    print("训练完成!")
    return model


def get_device():
    """获取计算设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


if __name__ == "__main__":
    # 设置随机种子
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # 选择设备
    device = get_device()
    
    # 开始训练
    model = train_model(device)