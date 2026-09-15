"""Red convolucional Dueling DQN (arquitectura Nature/DeepMind)."""

import torch
from torch import nn


class DuelingCNN(nn.Module):
    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_channels, 84, 84)).shape[1:].numel()
        self.value = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, 1))
        self.advantage = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = self.features(x).flatten(1)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
