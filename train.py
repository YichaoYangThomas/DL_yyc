import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
import numpy as np
from dataset import create_wall_dataloader
from models import JEPAModel


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """计算Barlow Twins损失函数"""
    # 展平时间维度，得到 [B*T, D]
    B, T, D = z1.shape
    z1_flat = z1.reshape(-1, D)
    z2_flat = z2.reshape(-1, D)
    
    # 标准化每个特征维度
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    # 计算批次相关性矩阵
    batch_size = z1_norm.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # 计算损失：对角线部分应接近1，非对角线部分应接近0
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].pow_(2).sum()
    
    # 总损失
    loss = on_diag + lambda_param * off_diag
    return loss


def apply_augmentation(states):
    """对状态数据应用数据增强"""
    B, T, C, H, W = states.shape
    augmented_states = []
    
    for b in range(B):
        sample = states[b]  # [T, C, H, W]
        
        # 应用随机翻转和旋转
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        angle = random.choice([0, 90, 180, 270])
        
        aug_frames = []
        for t in range(T):
            frame = sample[t]  # [C, H, W]
            
            # 水平翻转
            if do_hflip:
                frame = VF.hflip(frame)
            
            # 垂直翻转
            if do_vflip:
                frame = VF.vflip(frame)
            
            # 旋转
            frame = VF.rotate(frame, angle)
            
            # 添加少量噪声
            noise = torch.randn_like(frame) * 0.01
            frame = torch.clamp(frame + noise, 0, 1) if frame.max() <= 1 else frame + noise
            
            aug_frames.append(frame)
        
        # 堆叠时间维度
        aug_frames = torch.stack(aug_frames, dim=0)  # [T, C, H, W]
        augmented_states.append(aug_frames)
    
    # 堆叠批次维度
    augmented_states = torch.stack(augmented_states, dim=0)  # [B, T, C, H, W]
    return augmented_states


def train_model(device):
    """训练JEPA模型"""
    # 加载训练数据
    print("加载训练数据...")
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
    )
    
    # 创建模型
    print("初始化模型...")
    model = JEPAModel(device=device).to(device)
    
    # 训练参数
    num_epochs = 10
    learning_rate = 2e-4
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
    
    # 训练循环
    print(f"开始训练，共{num_epochs}轮...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 遍历批次数据
        progress_bar = tqdm(train_loader, desc=f"第{epoch+1}轮训练")
        for batch_idx, batch in enumerate(progress_bar):
            # 获取状态和动作
            states = batch.states
            actions = batch.actions
            
            # 应用数据增强
            states_aug = apply_augmentation(states)
            
            # 前向传播 - 自回归预测
            predictions = model(states_aug, actions)
            
            # 获取目标表示
            with torch.no_grad():
                targets = model.encoder(
                    states.reshape(-1, *states.shape[2:])
                ).reshape(states.shape[0], states.shape[1], -1)
            
            # 计算JEPA损失（平滑L1）
            jepa_loss = F.smooth_l1_loss(predictions, targets)
            
            # 计算Barlow Twins损失
            # 重新获取原表示（保留梯度）
            z1 = model.encoder(
                states.reshape(-1, *states.shape[2:])
            ).reshape(states.shape[0], states.shape[1], -1)
            z2 = predictions
            bt_loss = barlow_twins_loss(z1, z2)
            
            # 总损失
            loss_weight = 0.25
            total_batch_loss = loss_weight * jepa_loss + (1 - loss_weight) * bt_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_batch_loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器和学习率更新
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            total_loss += total_batch_loss.item()
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'loss': f'{total_batch_loss.item():.4f}',
                'jepa': f'{jepa_loss.item():.4f}',
                'bt': f'{bt_loss.item():.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"第{epoch+1}轮完成，平均损失: {avg_loss:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'model_weights.pth')
            print(f"保存最佳模型，损失: {best_loss:.6f}")
    
    print("训练完成！")
    return model


def get_device():
    """获取训练设备"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device


if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 获取设备
    device = get_device()
    
    # 训练模型
    model = train_model(device)