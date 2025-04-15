from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import JEPAModel
import glob
import os


def get_device():
    """检查GPU可用性。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    return device


def load_data(device):
    """加载评估数据。"""
    data_path = "/scratch/DL25SP"
    print(f"正在加载数据，路径: {data_path}")

    # 加载探测训练数据
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )
    print(f"已加载探测训练数据集，批次数: {len(probe_train_ds)}")

    # 加载正常验证数据
    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )
    print(f"已加载正常验证数据集，批次数: {len(probe_val_normal_ds)}")

    # 加载墙壁验证数据
    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )
    print(f"已加载墙壁验证数据集，批次数: {len(probe_val_wall_ds)}")

    # 组合验证数据集
    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """加载或初始化模型。"""
    print("正在加载模型...")
    
    # 初始化模型
    model = JEPAModel()
    
    # 检查模型权重文件是否存在
    model_path = "model_weights.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重文件 {model_path} 不存在！")
    
    try:
        # 尝试加载模型权重
        state_dict = torch.load(model_path, map_location=get_device())
        model.load_state_dict(state_dict)
        print(f"成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        raise
    
    # 将模型移动到适当的设备并设置为评估模式
    model = model.to(get_device())
    model.eval()
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数量: {total_params / 1e6:.2f}M")
    
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    """评估模型。"""
    print("\n开始评估模型...")
    
    # 创建评估器
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    # 训练探测器
    print("训练探测器...")
    prober = evaluator.train_pred_prober()

    # 评估模型
    print("评估模型在验证集上的性能...")
    avg_losses = evaluator.evaluate_all(prober=prober)

    # 打印结果
    print("\n评估结果:")
    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} 均方误差: {loss:.6f}")


if __name__ == "__main__":
    device = get_device()
    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
