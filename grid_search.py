import os
import sys
import itertools
import json
import torch
import numpy as np
from datetime import datetime
from copy import deepcopy
import importlib
import torch.nn as nn

# 导入原始模型和训练模块
from models import JEPAModel, Encoder, Predictor
from train import train_model, barlow_twins_loss
import main

# 定义要搜索的参数空间
param_grid = {
    "jepa_loss_weight": [0.1, 0.2, 0.3],  # JEPA损失权重
    "dropout_rate": [0.1, 0.2, 0.3],      # Dropout比率
    "kernel_size": [3, 5, 7],             # 第一层卷积核大小
    "channel_multiplier": [1.0, 1.25, 1.5]  # 通道数倍数
}

# 存储实验结果的目录
results_dir = "grid_search_results"
os.makedirs(results_dir, exist_ok=True)

# 创建实验记录文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(results_dir, f"grid_search_{timestamp}.json")
results_log = os.path.join(results_dir, f"grid_search_{timestamp}.log")

# 初始化结果存储
all_results = []

# 获取设备
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    return device

# 定义修改模型参数的函数，不改变模型结构
def create_model_with_params(device, jepa_loss_weight, dropout_rate, kernel_size, channel_multiplier):
    # 修改Encoder类 - 需要创建一个新类来覆盖原始参数
    class GridSearchEncoder(nn.Module):
        def __init__(self, input_channels=2, input_size=(65, 65), repr_dim=256, projection_hidden_dim=256):
            super().__init__()
            # 计算新的通道数
            ch1 = int(32 * channel_multiplier)
            ch2 = int(64 * channel_multiplier)
            ch3 = int(128 * channel_multiplier)
            ch4 = int(256 * channel_multiplier)
            
            # 设置正确的padding使得输出尺寸一致
            padding = kernel_size // 2
            
            self.conv_net = nn.Sequential(
                # 第一层使用可变的卷积核大小
                nn.Conv2d(input_channels, ch1, kernel_size=kernel_size, stride=2, padding=padding),
                nn.ReLU(),
                nn.Dropout2d(0.1),  # 保持卷积层之间的Dropout不变
                
                # 第二层保持原有参数，但通道数调整
                nn.Conv2d(ch1, ch2, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                
                # 第三层保持原有参数，但通道数调整
                nn.Conv2d(ch2, ch3, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Dropout2d(0.1),  # 保持卷积层之间的Dropout不变
                
                # 第四层保持原有参数，但通道数调整
                nn.Conv2d(ch3, ch4, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

            with torch.no_grad():
                sample_input = torch.zeros(1, input_channels, *input_size)
                conv_output = self.conv_net(sample_input)
                conv_output_size = conv_output.view(1, -1).size(1)

            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),  # 使用可变的Dropout率
                nn.Linear(conv_output_size, repr_dim),
                nn.ReLU(),
            )

        def forward(self, x):
            x = self.conv_net(x)
            x = self.fc(x)
            return x
    
    # 创建修改后的JEPAModel
    model = JEPAModel(device=device)
    
    # 替换编码器
    model.encoder = GridSearchEncoder().to(device)
    
    # 预测器保持不变
    model.predictor = Predictor(repr_dim=256, action_dim=2).to(device)
    
    return model, jepa_loss_weight

# 修改的训练函数，以接受自定义参数
def train_with_params(device, jepa_loss_weight, dropout_rate, kernel_size, channel_multiplier):
    # 创建具有自定义参数的模型
    model, jepa_weight = create_model_with_params(
        device, jepa_loss_weight, dropout_rate, kernel_size, channel_multiplier
    )
    
    # 创建训练数据加载器
    from dataset import create_wall_dataloader
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL25SP/train",
        probing=False,
        device=device,
        train=True,
    )
    
    # 训练配置
    num_epochs = 5 # 降低epochs数以加速网格搜索
    learning_rate = 1e-4
    
    # 打印模型参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"开始训练，共{num_epochs}轮...")
    
    # 训练循环
    for epoch in range(num_epochs):
        # 设置为训练模式
        model.train()
        total_loss = 0.0
        
        # 处理每个批次
        for batch_idx, batch in enumerate(train_loader):
            # 获取批次数据
            states = batch.states
            actions = batch.actions
            
            # 前向传播 - 获取预测的表示
            predictions = model(states, actions)
            
            # 获取真实表示（无梯度）
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)
            
            # 计算JEPA损失 (使用smooth L1损失)
            jepa_loss = torch.nn.functional.smooth_l1_loss(predictions, targets)
            
            # 获取原始表示用于Barlow Twins损失
            z1 = model.encoder(
                states.view(-1, *states.shape[2:])
            ).view(states.size(0), states.size(1), -1)
            z2 = predictions
            
            # 计算Barlow Twins损失
            bt_loss = barlow_twins_loss(z1, z2)
            
            # 总损失 - 使用传入的jepa_loss_weight
            total_loss_batch = jepa_weight * jepa_loss + (1 - jepa_weight) * bt_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # 参数更新
            optimizer.step()
            
            # 记录损失
            total_loss += total_loss_batch.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"轮次 {epoch+1} 完成, 平均损失: {avg_loss:.8e}")
    
    # 保存模型（使用唯一名称避免冲突）
    model_path = f'model_weights_grid_{jepa_loss_weight}_{dropout_rate}_{kernel_size}_{channel_multiplier}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"训练完成! 模型保存为: {model_path}")
    
    return model, model_path

# 评估模型函数
def evaluate_model(device, model_path):
    # 从文件名解析参数
    # 格式: model_weights_grid_0.1_0.1_5_1.25.pth
    try:
        params = model_path.replace('model_weights_grid_', '').replace('.pth', '').split('_')
        jepa_loss_weight = float(params[0])
        dropout_rate = float(params[1])
        kernel_size = int(params[2])
        channel_multiplier = float(params[3])
        
        print(f"从文件名解析的参数: jepa_loss_weight={jepa_loss_weight}, dropout_rate={dropout_rate}, "
              f"kernel_size={kernel_size}, channel_multiplier={channel_multiplier}")
        
        # 创建具有相同参数的模型
        model, _ = create_model_with_params(
            device, jepa_loss_weight, dropout_rate, kernel_size, channel_multiplier
        )
    except Exception as e:
        print(f"解析模型参数失败，使用默认模型: {str(e)}")
        model = JEPAModel(device=device).to(device)
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 加载数据
    probe_train_ds, probe_val_ds = main.load_data(device)
    
    # 创建评估器
    evaluator = main.ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    # 训练探针
    prober = evaluator.train_pred_prober()
    
    # 评估
    avg_losses = evaluator.evaluate_all(prober=prober)
    
    # 提取关键指标
    results = {key: float(val) for key, val in avg_losses.items()}
    
    return results

# 运行网格搜索
def run_grid_search():
    device = get_device()
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    # 统计总组合数
    total_combinations = len(param_combinations)
    print(f"总共有{total_combinations}种参数组合需要测试")
    
    # 遍历所有参数组合
    for i, combination in enumerate(param_combinations):
        params = dict(zip(param_names, combination))
        
        # 记录当前进度
        progress = f"[{i+1}/{total_combinations}]"
        print(f"\n{progress} 开始测试参数组合: {params}")
        
        try:
            # 训练模型
            model, model_path = train_with_params(
                device=device,
                jepa_loss_weight=params["jepa_loss_weight"],
                dropout_rate=params["dropout_rate"],
                kernel_size=params["kernel_size"],
                channel_multiplier=params["channel_multiplier"]
            )
            
            # 评估模型
            eval_results = evaluate_model(device, model_path)
            
            # 组合参数和结果
            result = {
                "params": params,
                "evaluation": eval_results,
                "model_path": model_path
            }
            
            # 添加到结果列表
            all_results.append(result)
            
            # 保存当前结果
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # 记录日志
            with open(results_log, 'a') as f:
                f.write(f"{progress} 参数: {params}, 结果: {eval_results}\n")
            
            print(f"{progress} 参数组合测试完成，评估结果: {eval_results}")
            
        except Exception as e:
            print(f"参数组合 {params} 测试失败: {str(e)}")
            with open(results_log, 'a') as f:
                f.write(f"{progress} 参数: {params}, 错误: {str(e)}\n")
    
    # 找出最佳参数组合
    if all_results:
        # 按probe_normal val loss排序
        sorted_results = sorted(all_results, key=lambda x: x["evaluation"].get("probe_normal", float('inf')))
        best_result = sorted_results[0]
        
        print("\n网格搜索完成!")
        print(f"最佳参数组合: {best_result['params']}")
        print(f"评估结果: {best_result['evaluation']}")
        print(f"模型路径: {best_result['model_path']}")
        
        # 保存最佳结果
        with open(os.path.join(results_dir, f"best_result_{timestamp}.json"), 'w') as f:
            json.dump(best_result, f, indent=2)
    else:
        print("没有成功完成的测试")

if __name__ == "__main__":
    run_grid_search() 