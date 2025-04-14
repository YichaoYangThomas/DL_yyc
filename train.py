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
    # 不变性损失 (使用smooth_l1_loss，对异常值更鲁棒)
    sim_loss = F.smooth_l1_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    
    # 方差损失 (增加目标阈值到2.0，增强特征表达能力)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(2.0 - std_x)) + torch.mean(F.relu(2.0 - std_y))
    
    # 协方差损失 (先归一化特征，减少规模效应)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum() + off_diagonal(cov_y).pow_(2).sum()
    
    return sim_coef * sim_loss + var_coef * var_loss + cov_coef * cov_loss, sim_loss, var_loss, cov_loss

def add_noise(tensor, noise_level=0.02):
    # 添加适量的高斯噪声，增强数据多样性
    return tensor + torch.randn_like(tensor) * noise_level

def train(epochs=50, save_dir="./checkpoints"):  # 增加训练轮数和保存目录参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = JEPAModel().to(device)
    print(f"模型创建完成，潜在维度: {model.repr_dim}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params / 1e6:.2f}M")
    
    # 优化器 - 增加权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # 批处理大小和梯度累积
    batch_size = 32
    grad_accum_steps = 16  # 增加梯度累积步数，相当于更大的批次
    
    # 学习率调整参数
    warmup_steps = 1000  # 增加预热步数
    
    # 早停策略参数
    patience = 5  # 允许5个epoch无改善
    no_improve_epochs = 0
    
    # 模型权重平均相关参数
    use_ema = True  # 启用指数移动平均
    ema_decay = 0.998
    ema_params = {}
    
    # 数据加载
    data_path = "/scratch/DL24FA/train"
    print(f"加载训练数据: {data_path}")
    
    train_loader = create_wall_dataloader(
        data_path=data_path,
        probing=False,
        device="cpu",  # 先加载到CPU，训练时再传输到GPU
        batch_size=batch_size,
        train=True
    )
    
    # 如果有验证集，可以加载验证集
    # val_loader = create_wall_dataloader(...)
    
    total_steps = epochs * len(train_loader)
    print(f"总训练步数: {total_steps}")
    
    best_loss = float('inf')
    step = 0
    
    # 监控变量
    total_batches = len(train_loader)
    
    # 检查保存目录是否存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 初始化EMA参数字典
    if use_ema:
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema_params[name] = param.data.clone()
    
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
            # 学习率调整 - 使用更平滑的调度策略
            if step < warmup_steps:
                curr_lr = 1e-4 * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                curr_lr = 1e-4 * 0.5 * (1 + math.cos(math.pi * progress))
            
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
                    
                # 增强数据增强策略
                # 1. 水平翻转 (概率0.3)
                if random.random() < 0.3:
                    states = torch.flip(states, [3])
                    actions[:, :, 0] = -actions[:, :, 0]
                
                # 2. 垂直翻转 (概率0.3)
                if random.random() < 0.3:
                    states = torch.flip(states, [4])
                    actions[:, :, 1] = -actions[:, :, 1]
                
                # 3. 添加轻微噪声 (概率0.2)
                if random.random() < 0.2:
                    states = add_noise(states, noise_level=0.01)
                
                # 4. 添加输入抖动 (概率0.15)
                if random.random() < 0.15:
                    jitter = torch.zeros_like(actions).uniform_(-0.05, 0.05)
                    actions = actions + jitter
                    
                B, T, C, H, W = states.shape
                curr_states = states[:, :-1].contiguous().view(-1, C, H, W)
                next_states = states[:, 1:].contiguous().view(-1, C, H, W)
                
                # 前向传播
                pred_states = model.encoder(curr_states)
                with torch.no_grad():
                    target_states = model.target_encoder(next_states)
                
                actions_flat = actions.reshape(-1, 2)
                
                # 获取墙壁通道用于碰撞检测
                wall_channel = next_states[:, 1:2, :, :]
                
                # 判断是否发生碰撞
                collision_mask = (wall_channel.view(-1, H * W).max(dim=1)[0] > 0).float().unsqueeze(1)
                
                # 预测下一个状态
                pred_next = model.predictor(pred_states, actions_flat)
                
                # 预测碰撞概率
                pred_collision = model.predictor.collision_head(pred_next)
                
                # 碰撞损失
                collision_loss = F.binary_cross_entropy(pred_collision, collision_mask)

                # 计算VICReg损失
                total_loss, sim_loss, var_loss, cov_loss = vicreg_loss(
                    pred_next, 
                    target_states.detach(), 
                    sim_coef=25.0, 
                    var_coef=25.0, 
                    cov_coef=1.0
                )
                
                # 组合损失 - 增加碰撞损失权重
                loss = total_loss + collision_loss * 0.2
                
                # 梯度累积
                loss = loss / grad_accum_steps
                loss.backward()
                
                accumulated_loss += loss.item() * grad_accum_steps
                
                # 梯度累积步骤完成后更新参数
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新目标编码器
                    momentum = min(0.996, 0.99 + step/total_steps * 0.006)
                    model.update_target(momentum=momentum)
                    
                    # 更新EMA参数
                    if use_ema:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if param.requires_grad:
                                    ema_params[name] = ema_params[name] * ema_decay + param.data * (1 - ema_decay)
                    
                    # 每100个批次记录一次损失
                    if batch_idx % 100 == 0:
                        tqdm.write(
                            f"[Epoch {epoch+1}/{epochs}][Batch {batch_idx+1}/{total_batches}] "
                            f"损失: {accumulated_loss:.4f}, 学习率: {curr_lr:.6f}, "
                            f"相似性损失: {sim_loss.item():.4f}, 方差损失: {var_loss.item():.4f}, "
                            f"协方差损失: {cov_loss.item():.4f}, 碰撞损失: {collision_loss.item():.4f}"
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
        
        # 如果使用EMA，保存EMA参数
        if use_ema:
            # 暂存当前参数
            original_params = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    original_params[name] = param.data.clone()
                    param.data.copy_(ema_params[name])
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'is_ema': True
            }, checkpoint_path)
            
            # 恢复原始参数
            for name, param in model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
        else:
            # 直接保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(save_dir, "best_model.pth")
            
            # 如果使用EMA，保存EMA参数作为最佳模型
            if use_ema:
                # 暂存当前参数
                original_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        original_params[name] = param.data.clone()
                        param.data.copy_(ema_params[name])
                
                # 保存模型
                torch.save(model.state_dict(), best_model_path)
                
                # 恢复原始参数
                for name, param in model.named_parameters():
                    if name in original_params:
                        param.data.copy_(original_params[name])
            else:
                # 直接保存模型
                torch.save(model.state_dict(), best_model_path)
                
            tqdm.write(
                f"\nEpoch {epoch+1} 总结:\n"
                f"平均损失: {avg_epoch_loss:.4f} (新最佳)\n"
                f"学习率: {curr_lr:.6f}\n"
            )
            no_improve_epochs = 0
        else:
            tqdm.write(
                f"\nEpoch {epoch+1} 总结:\n"
                f"平均损失: {avg_epoch_loss:.4f}\n"
                f"学习率: {curr_lr:.6f}\n"
            )
            no_improve_epochs += 1
        
        # 早停策略
        if no_improve_epochs >= patience:
            tqdm.write(f"早停：连续 {patience} 个epoch没有改善，停止训练")
            break
    
    print("\n训练完成!")
    print(f"最佳损失: {best_loss:.4f}")
    
    # 加载最佳模型
    best_model_path = os.path.join(save_dir, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    
    return model


if __name__ == "__main__":
    train()
