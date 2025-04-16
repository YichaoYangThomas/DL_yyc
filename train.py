import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from tqdm import tqdm
import random
from dataset import create_wall_dataloader
from models import JEPAModel


def je_loss(predictions, targets):
    """
    联合嵌入损失
    """
    similarities = F.cosine_similarity(predictions, targets, dim=-1)
    loss_per_sample = 1 - similarities
    loss_per_sample = loss_per_sample.sum(dim=1)
    loss = loss_per_sample.mean()
    return loss


def barlow_twins_loss(z1, z2, lambda_param=0.005):
    """
    Barlow Twins损失
    """
    # b, t, d
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)
    z2_flat = z2.view(-1, D)
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)
    
    batch_size = z1_norm.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size
    
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].pow_(2).sum()
    
    loss = on_diag + lambda_param * off_diag
    return loss


def apply_augmentation(states):
    """
    基础数据增强
    """
    B, T, C, H, W = states.shape
    aug_states = []
    
    for b in range(B):
        sample = states[b]
        
        # 基本变换
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)
        
        aug_frames = []
        for t in range(T):
            frame = sample[t]
            
            # 几何变换
            if do_hflip:
                frame = VF.hflip(frame)
            if do_vflip:
                frame = VF.vflip(frame)
            frame = VF.rotate(frame, angle)
            
            # 简单噪声
            noise = torch.randn_like(frame) * 0.01
            frame = frame + noise
            
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
    num_epochs = 30
    
    # 使用原始优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"开始训练，总轮数: {num_epochs}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"训练轮次 {epoch+1}/{num_epochs}")):
            states = batch.states
            actions = batch.actions
            
            # 应用基础数据增强
            states = apply_augmentation(states)
            
            # 前向传播
            predictions = model(states, actions)
            
            # 获取目标表示
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
            
            # 计算JEPA损失
            jepa_loss = F.smooth_l1_loss(predictions, targets)
            
            # 计算Barlow Twins损失
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.size(0), states.size(1), -1)
            z2 = predictions
            bt_loss = barlow_twins_loss(z1, z2)
            
            # 总损失 - 恢复原始权重比例
            jep_co = 0.2
            total_loss_batch = jep_co * jepa_loss + (1-jep_co) * bt_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 保留梯度裁剪以防梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累计损失
            total_loss += total_loss_batch.item()
            
            # 记录进度
            if (batch_idx + 1) % 50 == 0:
                print(f"Batch [{batch_idx+1}/{len(train_loader)}], Loss: {total_loss_batch.item():.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"轮次 [{epoch+1}/{num_epochs}], 平均损失: {avg_loss:.8e}")
    
    # 保存模型
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
