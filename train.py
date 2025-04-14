import torch
import torch.nn.functional as F
from dataset import create_wall_dataloader
from models import JEPAModel, off_diagonal
from tqdm import tqdm
import random
import math
import os
import numpy as np

def vicreg_loss(x, y, sim_coef, var_coef, cov_coef):
    # 不变性损失
    sim_loss = F.mse_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    
    # 方差损失 
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))
    
    # 协方差损失
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()
    
    return sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss, sim_loss, var_loss, cov_loss

def train(epochs=50, save_dir="./"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = JEPAModel(latent_dim=256).to(device)  # 增加潜在维度
    print(f"模型创建完成，潜在维度: {model.repr_dim}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params / 1e6:.2f}M")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # 批处理大小和梯度累积 - 减小以节省内存
    batch_size = 16  # 从32减小到16
    grad_accum_steps = 8  # 从4增加到8，保持有效批量大小
    
    # 学习率调整参数
    warmup_steps = 500
    # 添加学习率重启策略参数
    restart_epochs = [10, 20, 30, 40]  # 在这些epoch结束时重启学习率
    
    # 数据加载
    data_path = "/scratch/DL25SP/train"
    print(f"加载训练数据: {data_path}")
    
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device="cpu",  # 先加载到CPU，训练时再传输到GPU
        batch_size=batch_size,
        train=True
    )
    
    total_steps = epochs * len(train_loader)
    print(f"总训练步数: {total_steps}")
    
    best_loss = float('inf')
    step = 0
    
    # 监控变量
    total_batches = len(train_loader)
    
    # 检查保存目录是否存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model.train()
    # 创建epoch进度条
    pbar_epoch = tqdm(range(epochs), desc='训练进度')
    
    optimizer.zero_grad()  # 初始化优化器
    accumulated_loss = 0
    
    for epoch in pbar_epoch:
        epoch_loss = 0
        num_batches = 0
        
        # 创建batch进度条
        pbar_batch = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, batch in enumerate(pbar_batch):
            # 学习率调整
            if step < warmup_steps:
                curr_lr = 3e-4 * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                curr_lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
                # 添加最小学习率限制
                curr_lr = max(curr_lr, 1e-6)
            
            # 更新学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = curr_lr
                
            try:
                # 清理GPU缓存
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    
                # 将数据移至GPU
                states = batch.states.to(device)
                actions = batch.actions.to(device)
                    
                # 数据增强
                if random.random() < 0.3:  # 降低翻转概率
                    states = torch.flip(states, [3])  # 水平翻转
                    actions[:, :, 0] = -actions[:, :, 0]
                if random.random() < 0.3:  # 降低翻转概率
                    states = torch.flip(states, [4])  # 垂直翻转
                    actions[:, :, 1] = -actions[:, :, 1]
                    
                B, T, C, H, W = states.shape
                curr_states = states[:, :-1].contiguous().view(-1, C, H, W)
                next_states = states[:, 1:].contiguous().view(-1, C, H, W)
                
                # 前向传播
                pred_states = model.encoder(curr_states)
                with torch.no_grad():
                    target_states = model.target_encoder(next_states)
                
                actions_flat = actions.reshape(-1, 2)
                
                # 预测下一个状态
                pred_next = model.predictor(pred_states, actions_flat)

                # 计算VICReg损失
                total_vicreg_loss, sim_loss, var_loss, cov_loss = vicreg_loss(
                    pred_next, 
                    target_states.detach(), 
                    sim_coef=20.0,  
                    var_coef=30.0,  
                    cov_coef=2.0    
                )
                
                # 计算辅助任务损失 - 重建损失
                recon_loss = 0
                if random.random() < 0.5:  # 降低概率从0.7到0.5
                    recon = model.reconstruct(pred_states)
                    # 调整重建图像或输入图像的大小，确保尺寸匹配
                    if recon.shape[2:] != curr_states.shape[2:]:
                        # 方式1: 调整重建结果尺寸以匹配输入
                        recon = F.interpolate(recon, size=curr_states.shape[2:], mode='bilinear', align_corners=False)
                    recon_loss = F.mse_loss(recon, curr_states) * 2.0  # 权重从5.0降到2.0
                
                # 计算辅助任务损失 - 对比损失
                contrastive_loss = 0
                if random.random() < 0.5:  # 降低概率从0.7到0.5
                    # 创建两个增强视图
                    jitter = torch.randn_like(pred_states) * 0.1
                    z1 = pred_states
                    z2 = pred_states + jitter
                    contrastive_loss = model.compute_contrastive_loss(z1, z2) * 0.5  # 权重从1.0降到0.5
                
                # 总损失 = VICReg损失 + 重建损失 + 对比损失
                total_loss = total_vicreg_loss + recon_loss + contrastive_loss
                
                # 梯度累积
                loss = total_loss / grad_accum_steps
                loss.backward()
                
                accumulated_loss += loss.item() * grad_accum_steps
                
                # 梯度累积步骤完成后更新参数
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新目标编码器 - 使用动态动量
                    model.update_target(step=step, total_steps=total_steps)
                    
                    # 每100个批次记录一次损失
                    if batch_idx % 100 == 0:
                        tqdm.write(
                            f"[Epoch {epoch+1}/{epochs}][Batch {batch_idx+1}/{total_batches}] "
                            f"总损失: {accumulated_loss:.4f}, 学习率: {curr_lr:.6f}, "
                            f"VICReg: {total_vicreg_loss.item():.4f}, 重建: {recon_loss:.4f}, 对比: {contrastive_loss:.4f}, "
                            f"相似性: {sim_loss.item():.4f}, 方差: {var_loss.item():.4f}, 协方差: {cov_loss.item():.4f}"
                        )
                        accumulated_loss = 0
            
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"内存不足，跳过批次 {batch_idx}")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                else:
                    raise e
            
            epoch_loss += loss.item() * grad_accum_steps
            num_batches += 1
            step += 1
            
            pbar_batch.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{curr_lr:.6f}'})
        
        avg_epoch_loss = epoch_loss / num_batches
        pbar_epoch.set_postfix({'avg_loss': f'{avg_epoch_loss:.4f}'})
        
        # 每个epoch保存一次检查点
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
            tqdm.write(
                f"\nEpoch {epoch+1} 总结:\n"
                f"平均损失: {avg_epoch_loss:.4f} (新最佳)\n"
                f"学习率: {curr_lr:.6f}\n"
            )
        else:
            tqdm.write(
                f"\nEpoch {epoch+1} 总结:\n"
                f"平均损失: {avg_epoch_loss:.4f}\n"
                f"学习率: {curr_lr:.6f}\n"
            )
    
    print("\n训练完成!")
    print(f"最佳损失: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
