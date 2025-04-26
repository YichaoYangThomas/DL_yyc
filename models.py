from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    仅用于测试的模拟模型
    """
    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class Encoder(nn.Module):
    def __init__(self, input_channels=2, input_size=(65, 65), embedding_dim=256, hidden_dim=256):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            # 第一层使用更大的卷积核(5x5)以增大初始感受野
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            
            # 第二层保持原有参数
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 第三层保持原有参数
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 第四层保持原有参数
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, *input_size)
            conv_output = self.feature_extractor(sample_input)
            flattened_size = conv_output.view(1, -1).size(1)

        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        embeddings = self.projection_head(features)
        return embeddings


class Predictor(nn.Module):
    def __init__(self, embedding_dim=256, action_dim=2):
        super().__init__()
        self.transition_network = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, state_embedding, action):
        combined_input = torch.cat([state_embedding, action], dim=-1)
        next_state_embedding = self.transition_network(combined_input)
        return next_state_embedding


class JEPAModel(nn.Module):
    def __init__(self, device="cuda", embedding_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        
        self.encoder = Encoder(embedding_dim=embedding_dim).to(device)
        self.predictor = Predictor(embedding_dim=embedding_dim, action_dim=action_dim).to(device)

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        batch_size, seq_len, channels, height, width = states.shape
        device = states.device
        
        embeddings_sequence = []
        current_embedding = self.encoder(states[:, 0])  # [B, D]
        embeddings_sequence.append(current_embedding.unsqueeze(1))  # [B, 1, D]
        
        for t in range(seq_len - 1):
            current_action = actions[:, t]
            predicted_embedding = self.predictor(current_embedding, current_action)
            embeddings_sequence.append(predicted_embedding.unsqueeze(1))
            current_embedding = predicted_embedding  # Update current embedding with prediction

        # Concatenate all embeddings along sequence dimension
        all_embeddings = torch.cat(embeddings_sequence, dim=1)  # [B, T, D]
        
        return all_embeddings

    def predict_future(self, init_states, actions):
        """
        根据初始状态和动作序列展开预测未来表示。

        Args:
            init_states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Returns:
            predicted_reprs: [T, B, D]
        """
        batch_size, _, channels, height, width = init_states.shape
        action_seq_len = actions.shape[1]
        total_seq_len = action_seq_len + 1
        
        # 存储预测结果
        predicted_embeddings = []
        
        # 初始状态
        current_embedding = self.encoder(init_states[:, 0])  # [B, D]
        predicted_embeddings.append(current_embedding.unsqueeze(0))  # [1, B, D]
        
        # 循环预测
        for t in range(action_seq_len):
            current_action = actions[:, t]  # [B, action_dim]
            # 预测下一个表示
            next_embedding = self.predictor(current_embedding, current_action)  # [B, D]
            predicted_embeddings.append(next_embedding.unsqueeze(0))  # [1, B, D]
            # 更新当前表示
            current_embedding = next_embedding
        
        # 拼接所有预测
        all_predicted_embeddings = torch.cat(predicted_embeddings, dim=0)  # [T, B, D]
        
        return all_predicted_embeddings