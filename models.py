from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mlp(layer_sizes: List[int]):
    """Create a multi-layer perceptron with batch normalization"""
    layers = []
    for i in range(len(layer_sizes) - 2):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """Test-only model that returns random values"""
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


class EncoderCNN(nn.Module):
    """CNN encoder with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels=2, img_size=(65, 65), feat_dim=256, dropout_p=0.1):
        super().__init__()
        # Progressive increase in channels: 36 -> 72 -> 144
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_channels, 36, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 72, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 144, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(144),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_p)
        )

        # Calculate output size of CNN backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *img_size)
            cnn_output = self.cnn_backbone(dummy_input)
            cnn_flat_size = cnn_output.view(1, -1).size(1)

        # Feature projector
        self.feature_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flat_size, feat_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        """Forward pass through encoder"""
        features = self.cnn_backbone(x)
        latent_repr = self.feature_proj(features)
        return latent_repr


class PredictorMLP(nn.Module):
    """MLP predictor with ELU activation"""
    def __init__(self, feat_dim=256, action_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feat_dim + action_dim, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, state_repr, action_input):
        """Predict next state representation"""
        combined_input = torch.cat([state_repr, action_input], dim=-1)
        return self.network(combined_input)


class JEPAModel(nn.Module):
    """Joint Embedding Predictive Architecture model"""
    def __init__(self, device="cuda", repr_dim=256, action_dim=2):
        super().__init__()
        self.device = device
        self.repr_dim = repr_dim
        self.action_dim = action_dim

        # Initialize encoder and predictor
        self.encoder = EncoderCNN(in_channels=2, feat_dim=repr_dim).to(device)
        self.predictor = PredictorMLP(feat_dim=repr_dim, action_dim=action_dim).to(device)

    def predict_future(self, init_states, actions):
        """
        Unroll predictions over time steps
        
        Args:
            init_states: [B, 1, Ch, H, W] - Initial observations
            actions: [B, T-1, 2] - Action sequence
            
        Returns:
            predicted_reprs: [T, B, D] - Predicted representations
        """
        batch_size, _, channels, height, width = init_states.shape
        seq_len_minus1 = actions.shape[1]
        total_steps = seq_len_minus1 + 1
        
        # Store predictions
        pred_sequence = []
        
        # Get initial state representation
        current_repr = self.encoder(init_states[:, 0])
        pred_sequence.append(current_repr.unsqueeze(0))
        
        # Predict future states
        for t in range(seq_len_minus1):
            current_action = actions[:, t]
            next_repr = self.predictor(current_repr, current_action)
            pred_sequence.append(next_repr.unsqueeze(0))
            current_repr = next_repr
        
        # Concatenate all predictions
        pred_sequence = torch.cat(pred_sequence, dim=0)
        return pred_sequence

    def forward(self, states, actions):
        """
        Forward pass through the model
        
        Args:
            states: [B, T, Ch, H, W] - Observations
            actions: [B, T-1, 2] - Actions
            
        Returns:
            predictions: [B, T, D] - Predicted state representations
        """
        batch_size, time_steps, channels, height, width = states.shape
        
        # Predictions container
        predictions = []
        
        # Process initial state
        current_repr = self.encoder(states[:, 0])
        predictions.append(current_repr.unsqueeze(1))
        
        # Recurrent prediction for subsequent states
        for t in range(time_steps - 1):
            action_t = actions[:, t]
            pred_repr = self.predictor(current_repr, action_t)
            predictions.append(pred_repr.unsqueeze(1))
            current_repr = pred_repr  # Use prediction for next step
        
        # Combine all predictions
        predictions = torch.cat(predictions, dim=1)
        return predictions


class Prober(torch.nn.Module):
    """Probes information from learned representations"""
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

        # Parse architecture string
        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        dims = [embedding] + arch_list + [self.output_dim]
        
        # Build network
        layers = []
        for i in range(len(dims) - 2):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(dims[-2], dims[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, embeddings):
        """Map embeddings to target space"""
        return self.prober(embeddings)