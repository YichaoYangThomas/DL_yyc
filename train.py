import torch
import torch.nn.functional as F
from dataset import create_wall_dataloader
from models import JEPAModel, off_diagonal
from tqdm import tqdm
import random
import math
import os
import numpy as np

def vicreg_loss(x, y, sim_coef, var_coef, cov_coef, var_threshold=2.0):
    # 不变性损失 (使用smooth_l1_loss，对异常值更鲁棒)
    sim_loss = F.smooth_l1_loss(F.normalize(x, dim=-1), F.normalize(y, dim=-1))
    
    # 方差损失 (使用可调节阈值，增强特征表达能力)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(var_threshold - std_x)) + torch.mean(F.relu(var_threshold - std_y))
    
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
    
    # 学习率调整参数 - 优化学习率策略
    warmup_steps = 1500  # 增加预热步数
    min_lr = 5e-6  # 设置最小学习率阈值
    
    # 早停策略参数
    patience = 5  # 允许5个epoch无改善
    no_improve_epochs = 0
    
    # 模型权重平均相关参数 - 优化EMA衰减率
    use_ema = True  # 启用指数移动平均
    ema_decay_min = 0.995  # 最小衰减率
    ema_decay_max = 0.9995  # 最大衰减率
    ema_params = {}
    
    # VICReg损失参数 - 微调损失权重
    sim_coef = 30.0  # 略微增加相似性损失权重
    var_coef = 20.0  # 略微减少方差损失权重
    cov_coef = 2.0   # 略微增加协方差损失权重
    
    # 碰撞损失基础权重
    collision_base_weight = 0.25  # 增加碰撞损失权重
    
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
    
    # 如果有验证集，可以加载验证集
    # val_loader = create_wall_dataloader(...)
    
    total_steps = epochs * len(train_loader)
    print(f"总训练步数: {total_steps}")
    
    best_loss = float('inf')
    step = 0
    
    # 监控变量
    total_batches = len(train_loader)
    prev_loss = float('inf')  # 添加前一个epoch的损失记录
    
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
                curr_lr = max(min_lr, 1e-4 * 0.5 * (1 + math.cos(math.pi * progress)))
            
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
                    
                # 优化数据增强策略
                # 1. 水平翻转 (概率0.25)
                if random.random() < 0.25:
                    states = torch.flip(states, [3])
                    actions[:, :, 0] = -actions[:, :, 0]
                
                # 2. 垂直翻转 (概率0.25)
                if random.random() < 0.25:
                    states = torch.flip(states, [4])
                    actions[:, :, 1] = -actions[:, :, 1]
                
                # 3. 添加轻微噪声 (概率0.15)
                if random.random() < 0.15:
                    states = add_noise(states, noise_level=0.01)
                
                # 4. 添加输入抖动 (概率0.1)
                if random.random() < 0.1:
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
                
                # 碰撞损失，根据样本比例动态调整权重
                collision_loss = F.binary_cross_entropy(pred_collision, collision_mask)
                collision_ratio = collision_mask.mean().item()
                collision_weight = collision_base_weight * (1.0 + (0.5 if collision_ratio < 0.2 else 0))

                # 计算VICReg损失
                # 动态调整方差阈值
                var_threshold = min(2.0, 1.0 + step / (total_steps * 0.2))
                
                total_loss, sim_loss, var_loss, cov_loss = vicreg_loss(
                    pred_next, 
                    target_states.detach(), 
                    sim_coef=sim_coef, 
                    var_coef=var_coef, 
                    cov_coef=cov_coef,
                    var_threshold=var_threshold
                )
                
                # 组合损失 - 使用动态权重
                loss = total_loss + collision_loss * collision_weight
                
                # 梯度累积
                loss = loss / grad_accum_steps
                loss.backward()
                
                accumulated_loss += loss.item() * grad_accum_steps
                
                # 混合样本增强训练 (15%概率)
                if random.random() < 0.15 and pred_states.size(0) > 2:
                    # 随机混合批次中的两个状态
                    idx1, idx2 = torch.randperm(pred_states.size(0))[:2]
                    mix_ratio = random.uniform(0.7, 0.9)
                    mixed_state = mix_ratio * pred_states[idx1] + (1 - mix_ratio) * pred_states[idx2]
                    # 使用混合状态进行预测
                    mixed_action = actions_flat[idx1]
                    mixed_pred = model.predictor(mixed_state.unsqueeze(0), mixed_action.unsqueeze(0)).squeeze(0)
                    # 添加到梯度计算中
                    mixed_target = mix_ratio * target_states[idx1] + (1 - mix_ratio) * target_states[idx2]
                    mixed_loss = F.mse_loss(mixed_pred, mixed_target.detach())
                    (mixed_loss / grad_accum_steps).backward()
                
                # 梯度累积步骤完成后更新参数
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪 (0.5而非1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新目标编码器 - 使用动态动量
                    momentum_min = 0.995
                    momentum_max = 0.9995
                    momentum = momentum_min + (momentum_max - momentum_min) * min(1.0, step / (total_steps * 0.5))
                    model.update_target(momentum=momentum)
                    
                    # 更新EMA参数 - 使用动态衰减率
                    if use_ema:
                        with torch.no_grad():
                            ema_decay = ema_decay_min + (ema_decay_max - ema_decay_min) * min(1.0, step / (total_steps * 0.3))
                            for name, param in model.named_parameters():
                                if param.requires_grad:
                                    ema_params[name] = ema_params[name] * ema_decay + param.data * (1 - ema_decay)
                    
                    # 周期性添加权重正则化
                    if step % 500 == 0 and step > 0:
                        with torch.no_grad():
                            for name, param in model.named_parameters():
                                if param.requires_grad and 'bn' not in name and 'bias' not in name:
                                    # 轻微收缩权重
                                    param.data.mul_(0.9999)
                    
                    # 每100个批次记录一次损失
                    if batch_idx % 100 == 0:
                        tqdm.write(
                            f"[Epoch {epoch+1}/{epochs}][Batch {batch_idx+1}/{total_batches}] "
                            f"损失: {accumulated_loss:.4f}, 学习率: {curr_lr:.6f}, "
                            f"相似性损失: {sim_loss.item():.4f}, 方差损失: {var_loss.item():.4f}, "
                            f"协方差损失: {cov_loss.item():.4f}, 碰撞损失: {collision_loss.item():.4f}, "
                            f"碰撞比例: {collision_ratio:.3f}, 碰撞权重: {collision_weight:.3f}, "
                            f"方差阈值: {var_threshold:.3f}, 动量: {momentum:.6f}"
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
        
        # 根据损失变化的程度来决定是否保存检查点
        save_thresh = 0.001  # 损失改善的阈值
        save_checkpoint = False
        
        if epoch == 0 or prev_loss - avg_epoch_loss > save_thresh:
            # 损失明显改善，保存检查点
            save_checkpoint = True
            checkpoint_path = os.path.join(save_dir, f"checkpoint_improved_epoch_{epoch+1}.pth")
        else:
            # 每5个epoch保存一次
            if (epoch + 1) % 5 == 0:
                save_checkpoint = True
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
                
        if save_checkpoint:
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
                f"EMA衰减率: {ema_decay:.6f}\n"
                f"动量: {momentum:.6f}\n"
            )
            no_improve_epochs = 0
        else:
            tqdm.write(
                f"\nEpoch {epoch+1} 总结:\n"
                f"平均损失: {avg_epoch_loss:.4f}\n"
                f"学习率: {curr_lr:.6f}\n"
                f"EMA衰减率: {ema_decay:.6f}\n"
                f"动量: {momentum:.6f}\n"
            )
            no_improve_epochs += 1
        
        # 保存当前epoch损失用于下一次比较
        prev_loss = avg_epoch_loss
        
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
