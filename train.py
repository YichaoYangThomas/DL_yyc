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

def train(epochs=50, save_dir="./"):  # 增加训练轮数和保存目录参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = JEPAModel().to(device)
    print(f"模型创建完成，潜在维度: {model.repr_dim}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params / 1e6:.2f}M")
    
    # 优化器 - 仅微调学习率，保持权重衰减不变
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-4, weight_decay=0.01)
    
    # 批处理大小和梯度累积
    batch_size = 32
    grad_accum_steps = 4
    
    # 学习率调整参数 - 适度延长预热阶段
    warmup_steps = 600
    
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
    
    # 添加早停机制
    patience = 7  # 连续7个epoch没有改善则降低学习率
    no_improve_epochs = 0
    early_stop_threshold = 15  # 连续15个epoch没有改善则停止训练
    lr_factor = 0.8  # 学习率降低因子
    
    for epoch in pbar_epoch:
        epoch_loss = 0
        num_batches = 0
        
        # 创建batch进度条
        pbar_batch = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for batch_idx, batch in enumerate(pbar_batch):
            # 学习率调整 - 保持原有的余弦退火策略，微调参数
            if step < warmup_steps:
                curr_lr = 1.2e-4 * step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                curr_lr = 1.2e-4 * 0.5 * (1 + math.cos(math.pi * progress))
            
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
                    
                # 数据增强 - 保持原有策略
                if random.random() < 0.3:
                    states = torch.flip(states, [3])  # 水平翻转
                    actions[:, :, 0] = -actions[:, :, 0]
                if random.random() < 0.3:
                    states = torch.flip(states, [4])  # 垂直翻转
                    actions[:, :, 1] = -actions[:, :, 1]
                
                # 添加轻微的高斯噪声增强数据多样性
                if random.random() < 0.2:
                    noise = torch.randn_like(states) * 0.02
                    states = states + noise
                    
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

                # 计算VICReg损失 - 稍微调整权重
                total_loss, sim_loss, var_loss, cov_loss = vicreg_loss(
                    pred_next, 
                    target_states.detach(), 
                    sim_coef=25.0, 
                    var_coef=26.0,  # 轻微增加方差损失权重
                    cov_coef=1.1    # 轻微增加协方差损失权重
                )
                
                # 梯度累积
                loss = total_loss / grad_accum_steps
                loss.backward()
                
                accumulated_loss += loss.item() * grad_accum_steps
                
                # 梯度累积步骤完成后更新参数
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # 检查梯度是否有NaN或Inf
                    valid_gradients = True
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"警告: 参数 {name} 的梯度出现NaN或Inf值！")
                                valid_gradients = False
                                break
                    
                    if valid_gradients:
                        optimizer.step()
                    else:
                        print("跳过更新，使用上一个正常的梯度")
                    
                    optimizer.zero_grad()
                    
                    # 更新目标编码器 - 微调动量系数
                    momentum = min(0.996, 0.99 + step/total_steps * 0.006)
                    model.update_target(momentum=momentum)
                    
                    # 每300个批次记录一次损失
                    if batch_idx % 100 == 0:
                        tqdm.write(
                            f"[Epoch {epoch+1}/{epochs}][Batch {batch_idx+1}/{total_batches}] "
                            f"损失: {accumulated_loss:.4f}, 学习率: {curr_lr:.6f}, "
                            f"相似性损失: {sim_loss.item():.4f}, 方差损失: {var_loss.item():.4f}, 协方差损失: {cov_loss.item():.4f}"
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
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            tqdm.write(
                f"\nEpoch {epoch+1} 总结:\n"
                f"平均损失: {avg_epoch_loss:.4f}\n"
                f"学习率: {curr_lr:.6f}\n"
            )
            
            # 如果连续多个epoch没有改善，降低学习率
            if no_improve_epochs >= patience:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_factor
                tqdm.write(f"连续{patience}个epoch没有改善，降低学习率至 {param_group['lr']:.6f}")
                no_improve_epochs = 0  # 重置计数器
            
            # 如果连续过多epoch没有改善，提前结束训练
            if no_improve_epochs >= early_stop_threshold:
                tqdm.write(f"连续{early_stop_threshold}个epoch没有改善，提前结束训练，避免过拟合")
                break
    
    print("\n训练完成!")
    print(f"最佳损失: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    train()
