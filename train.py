import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
from dataset import create_wall_dataloader
from models import JEPAModel


def je_loss(predictions, targets, temperature=0.1):
    """
    改进的联合嵌入损失，加入温度参数
    """
    # 使用L2归一化
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    
    # 基于余弦相似度的损失
    similarities = F.cosine_similarity(predictions, targets, dim=-1) / temperature
    loss_per_sample = 1 - similarities
    loss_per_sample = loss_per_sample.sum(dim=1)
    loss = loss_per_sample.mean()
    return loss


def improved_barlow_twins_loss(z1, z2, lambda_param=0.005, scale_loss=0.025):
    """
    改进的Barlow Twins损失，调整参数和归一化
    """
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    
    # 正则化
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-6)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-6)
    
    batch_size = z1_norm.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    # 对角线和非对角线损失
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].pow_(2).sum()
    
    loss = on_diag + lambda_param * off_diag
    return loss * scale_loss


def improved_augmentation(states):
    """
    改进的数据增强方法
    """
    B, T, C, H, W = states.shape
    aug_states = []
    
    for b in range(B):
        sample = states[b]
        
        # 基本变换
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.3  # 降低垂直翻转概率
        angles = [0, 90, 180, 270]
        angle_weights = [0.6, 0.2, 0.1, 0.1]  # 加权选择角度
        angle = random.choices(angles, weights=angle_weights)[0]
        
        # 色彩增强参数
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)
        
        # 添加遮挡
        do_cutout = random.random() < 0.3
        
        aug_frames = []
        for t in range(T):
            frame = sample[t]
            
            # 几何变换
            if do_hflip:
                frame = VF.hflip(frame)
            if do_vflip:
                frame = VF.vflip(frame)
            frame = VF.rotate(frame, angle)
            
            # 色彩和噪声
            frame = frame * brightness_factor
            noise_strength = random.uniform(0.005, 0.015)
            noise = torch.randn_like(frame) * noise_strength
            frame = frame + noise
            
            # 遮挡 (Cutout)
            if do_cutout:
                mask_size = random.randint(5, 15)
                x = random.randint(0, W - mask_size)
                y = random.randint(0, H - mask_size)
                frame[:, y:y+mask_size, x:x+mask_size] = 0
            
            aug_frames.append(frame)
        
        aug_frames = torch.stack(aug_frames, dim=0)
        aug_states.append(aug_frames)
    
    aug_states = torch.stack(aug_states, dim=0)
    return aug_states


def train_model(device):
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
    )
    
    model = JEPAModel(device=device).to(device)
    
    # 训练参数
    num_epochs = 30  # 增加训练轮数
    batch_size = train_loader.batch_size
    
    # 学习率调度
    initial_lr = 2e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # 损失权重
    loss_weights = {
        'jepa': 0.3,
        'barlow': 0.7,
    }
    
    print(f"开始训练，总轮数: {num_epochs}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_jepa_loss = 0
        total_barlow_loss = 0
        
        # 学习率预热
        if epoch < 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * (epoch + 1) / 2
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{num_epochs}")):
            states = batch.states
            actions = batch.actions
            
            # 应用改进的数据增强
            if random.random() < 0.8:  # 80%的批次应用增强
                states = improved_augmentation(states)
            
            # 前向传播
            predictions = model(states, actions)
            
            # 使用目标编码器获取目标表示
            with torch.no_grad():
                model._update_target_encoder()  # 更新目标编码器
                target_reprs = model.get_target_representations(states)
            
            # 计算JEPA损失
            jepa_loss = je_loss(predictions, target_reprs, temperature=0.07)
            
            # 计算Barlow Twins损失
            bt_loss = improved_barlow_twins_loss(
                predictions, 
                target_reprs,
                lambda_param=0.0025,
                scale_loss=0.02
            )
            
            # 总损失
            total_loss_batch = loss_weights['jepa'] * jepa_loss + loss_weights['barlow'] * bt_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累计损失
            total_loss += total_loss_batch.item()
            total_jepa_loss += jepa_loss.item()
            total_barlow_loss += bt_loss.item()
            
            # 记录训练进度
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"JEPA Loss: {jepa_loss.item():.4f}, "
                      f"Barlow Loss: {bt_loss.item():.4f}, "
                      f"Total Loss: {total_loss_batch.item():.4f}")
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_jepa_loss = total_jepa_loss / len(train_loader)
        avg_barlow_loss = total_barlow_loss / len(train_loader)
        
        print(f"轮次 [{epoch+1}/{num_epochs}], 学习率: {current_lr:.6f}")
        print(f"平均损失: {avg_loss:.8f}, JEPA: {avg_jepa_loss:.8f}, Barlow: {avg_barlow_loss:.8f}")
    
    # 保存最终模型
    torch.save(model.state_dict(), 'model_weights.pth')
    print("训练完成，模型已保存")
    
    return model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    return device


if __name__ == "__main__":
    device = get_device()
    model = train_model(device)
