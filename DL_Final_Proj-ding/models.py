from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.functional as F
from encs import ResNet, build_resnet
from preds import ResPredictor, RNNPredictor


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
    Does nothing. Just for testing.
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
    def __init__(self, input_channels=2, input_size=(65, 65), repr_dim=256, projection_hidden_dim=256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, *input_size)
            conv_output = self.conv_net(sample_input)
            conv_output_size = conv_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, repr_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc(x)
        return x


class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, repr_dim)
        )

    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        return self.mlp(x)


class JEPAModel(nn.Module):
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        self.encoder = Encoder().to(device)
        self.predictor = Predictor().to(device)

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
        B, T, C, H, W = states.shape
        device = states.device

        predictions = []
        current_repr = self.encoder(states[:, 0])  # [B, D]
        predictions.append(current_repr.unsqueeze(1))  # [B, 1, D]

        for t in range(T - 1):
            action = actions[:, t]
            pred_repr = self.predictor(current_repr, action)
            predictions.append(pred_repr.unsqueeze(1))
            current_repr = pred_repr  # Update current representation with prediction

        predictions = torch.cat(predictions, dim=1)  # [B, T, D]

        return predictions

    def predict_future(self, init_states, actions):
        """
        Unroll the model to predict future representations.

        Args:
            init_states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Returns:
            predicted_reprs: [T, B, D]
        """
        B, _, C, H, W = init_states.shape
        T_minus1 = actions.shape[1]
        T = T_minus1 + 1

        predicted_reprs = []

        #initial state
        current_repr = self.encoder(init_states[:, 0])  # [B, D]
        predicted_reprs.append(current_repr.unsqueeze(0))  # [1, B, D]

        for t in range(T_minus1):
            action = actions[:, t]  # [B, action_dim]
            # Predict next representation
            pred_repr = self.predictor(current_repr, action)  # [B, D]
            predicted_reprs.append(pred_repr.unsqueeze(0))  # [1, B, D]
            # Update current representation for next step
            current_repr = pred_repr

        predicted_reprs = torch.cat(predicted_reprs, dim=0)  # [T, B, D]
        return predicted_reprs
