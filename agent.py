"""Agente Double DQN con cabezas Dueling y actualización Polyak opcional."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import optim

from model import DuelingCNN


class D3QNAgent:
    def __init__(self, n_actions, device="cuda", learning_rate=1e-4, gamma=0.99, grad_clip=10.0):
        self.device = torch.device(device)
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.online = DuelingCNN(4, n_actions).to(self.device)
        self.target = DuelingCNN(4, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.online.parameters(), lr=learning_rate, eps=1.5e-4)

    @torch.no_grad()
    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.online.advantage[-1].out_features)
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return int(self.online(state).argmax(1).item())

    def train_step(self, batch):
        states, actions, rewards, next_states, dones, discounts = batch[:6]
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online selecciona; target evalúa.
            next_actions = self.online(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + discounts * (1.0 - dones) * next_q
        td_error = target - q
        loss_per_item = F.smooth_l1_loss(q, target, reduction="none")
        weights = batch[6] if len(batch) >= 8 else torch.ones_like(loss_per_item)
        loss = (loss_per_item * weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        indices = batch[7] if len(batch) >= 8 else None
        return float(loss.item()), float(q.detach().mean().item()), td_error.detach().abs().cpu().numpy(), indices

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path, **metadata):
        """Guarda pesos, optimizador y estado necesario para reanudar."""
        checkpoint = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            **metadata,
        }
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint.get("online", checkpoint))
        self.target.load_state_dict(checkpoint.get("target", checkpoint.get("online", checkpoint)))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint
