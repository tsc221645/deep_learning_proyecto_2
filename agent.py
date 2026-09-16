"""Agente Double DQN con cabezas Dueling y actualización Polyak opcional."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

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
        # Los hitos están expresados en steps del entorno, no en actualizaciones
        # de gradiente. Esto mantiene el calendario correcto con train_frequency=4.
        self.scheduler = MultiStepLR(
            self.optimizer,
            milestones=[4_000_000, 6_000_000],
            gamma=0.5,
        )

    @torch.no_grad()
    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.online.advantage[-1].out_features)
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return int(self.online(state).argmax(1).item())

    def train_step(self, batch):
        """Ejecuta una actualización D3QN ponderada por PER.

        ``weights`` corrige el sesgo introducido por el muestreo prioritario.
        La pérdida se calcula por transición y solo después se pondera y
        promedia, antes de llamar a ``backward``.
        """
        states, actions, rewards, next_states, dones, discounts, weights, indices = batch
        q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: online selecciona; target evalúa.
            next_actions = self.online(next_states).argmax(1)
            next_q = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rewards + discounts * (1.0 - dones) * next_q
        td_error = target - q
        elementwise_huber = F.smooth_l1_loss(q, target, reduction="none")
        weighted_huber = elementwise_huber * weights
        loss = weighted_huber.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        return float(loss.item()), float(q.detach().mean().item()), td_error.detach().abs().cpu().numpy(), indices

    def update_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def step_scheduler(self, environment_step: int):
        """Avanza el LR scheduler usando el step global del entorno."""
        self.scheduler.step(environment_step)

    def save(self, path, **metadata):
        """Guarda pesos, optimizador y estado necesario para reanudar."""
        checkpoint = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            **metadata,
        }
        torch.save(checkpoint, path)

    def load(self, path, load_optimizer=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint.get("online", checkpoint))
        self.target.load_state_dict(checkpoint.get("target", checkpoint.get("online", checkpoint)))
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint
