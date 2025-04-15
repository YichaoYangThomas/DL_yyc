import torch
import torch.nn as nn
import torch.nn.functional as F


class ResPredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(repr_dim + action_dim, repr_dim)
        self.fc2 = nn.Linear(repr_dim, repr_dim)
        self.relu = nn.ReLU()

    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out + repr)
        return out


class RNNPredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(repr_dim + action_dim, hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, repr_dim)

    def forward(self, repr, action):
        x = torch.cat([repr, action], dim=-1).unsqueeze(1)
        x = self.input_proj(x)
        out, _ = self.rnn(x)
        out = self.output_proj(out.squeeze(1))
        return out
