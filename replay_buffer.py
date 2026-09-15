"""Replay buffer circular, almacenado en RAM como uint8."""

from __future__ import annotations

import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """Memoria uniforme DQN.

    Con 1M transiciones y estados de 4x84x84, los dos arrays de frames ocupan
    aproximadamente 56 GB. Es intencional: la VRAM solo recibe un batch.
    """

    def __init__(self, capacity=1_000_000, observation_shape=(4, 84, 84), device="cuda",
                 alpha=0.6):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.states = np.empty((capacity, *observation_shape), dtype=np.uint8)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = float(alpha)
        self.position = self.size = 0

    def add(self, state, action, reward, next_state, done, discount=1.0):
        i = self.position
        self.states[i] = np.asarray(state, dtype=np.uint8)
        self.next_states[i] = np.asarray(next_state, dtype=np.uint8)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.discounts[i] = float(discount)
        self.priorities[i] = self.priorities[:self.size].max() if self.size else 1.0
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.alpha == 0.0:
            probabilities = None
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            priorities = np.maximum(self.priorities[:self.size], 1e-6)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            indices = np.random.choice(self.size, batch_size, p=probabilities)
        # La conversión ocurre aquí, nunca durante el almacenamiento.
        states = torch.as_tensor(self.states[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)
        discounts = torch.as_tensor(self.discounts[indices], device=self.device)
        if probabilities is None:
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            weights = (self.size * probabilities[indices]) ** (-float(beta))
            weights /= weights.max()
        weights = torch.as_tensor(weights, device=self.device)
        return states, actions, rewards, next_states, dones, discounts, weights, indices

    def update_priorities(self, indices, priorities):
        self.priorities[np.asarray(indices)] = np.asarray(priorities, dtype=np.float32) + 1e-6

    def __len__(self):
        return self.size


class NStepAccumulator:
    """Convierte transiciones de 1 paso en retornos n-step sin copiar frames."""

    def __init__(self, n_step=3, gamma=0.99):
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.queue = deque()

    def _make(self, count):
        state, action, _, _, _, _ = self.queue[0]
        total_reward = 0.0
        next_state = self.queue[count - 1][3]
        done = False
        steps = 0
        for i in range(count):
            _, _, reward, next_state_i, done_i, _ = self.queue[i]
            total_reward += (self.gamma ** i) * reward
            next_state = next_state_i
            steps = i + 1
            if done_i:
                done = True
                break
        # El bootstrap debe descontarse por todos los pasos acumulados.
        return state, action, total_reward, next_state, done, self.gamma ** steps

    def append(self, state, action, reward, next_state, done, boundary=None):
        """Añade una transición; ``boundary`` también puede ser truncación.

        ``done`` controla el bootstrap (solo una terminación real lo anula),
        mientras ``boundary`` indica que hay que vaciar la cola al reiniciar.
        """
        if boundary is None:
            boundary = done
        self.queue.append((state, action, reward, next_state, done, boundary))
        emitted = []
        if len(self.queue) >= self.n_step:
            emitted.append(self._make(self.n_step))
            self.queue.popleft()
        if boundary:
            while self.queue:
                emitted.append(self._make(len(self.queue)))
                self.queue.popleft()
        return emitted
