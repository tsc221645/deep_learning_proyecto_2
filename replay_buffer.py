"""Replay buffers para D3QN con almacenamiento compacto en RAM."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class SumTree:
    """Árbol binario de sumas en un array NumPy, con operaciones O(log N)."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity debe ser positiva")
        # Potencia de dos: los hijos de cada nodo i son 2*i y 2*i+1.
        self.tree_capacity = 1 << (self.capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.tree_capacity, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def add(self, priority: float, data_index: int) -> int:
        """Escribe una prioridad en la hoja asociada a ``data_index``."""
        if not 0 <= data_index < self.capacity:
            raise IndexError("data_index fuera del rango del buffer")
        tree_index = self.tree_capacity + int(data_index)
        self.update(tree_index, priority)
        return tree_index

    def update(self, tree_index: int, new_priority: float) -> None:
        """Actualiza una hoja y propaga el delta hasta la raíz en O(log N)."""
        if not self.tree_capacity <= tree_index < self.tree_capacity + self.capacity:
            raise IndexError("tree_index no corresponde a una hoja válida")
        delta = float(new_priority) - self.tree[tree_index]
        self.tree[tree_index] = float(new_priority)
        parent = tree_index >> 1
        while parent:
            self.tree[parent] += delta
            parent >>= 1

    def get_leaf(self, value: float) -> tuple[int, int, float]:
        """Devuelve ``(tree_index, data_index, priority)`` para una suma dada."""
        if self.total <= 0.0:
            raise ValueError("No se puede muestrear un SumTree vacío")
        value = min(max(float(value), 0.0), np.nextafter(self.total, 0.0))
        node = 1
        while node < self.tree_capacity:
            left = node << 1
            if value <= self.tree[left]:
                node = left
            else:
                value -= self.tree[left]
                node = left + 1
        data_index = node - self.tree_capacity
        return node, data_index, float(self.tree[node])


class ReplayBuffer:
    """Prioritized Replay Buffer con estados ``uint8`` y SumTree."""

    def __init__(self, capacity=1_000_000, observation_shape=(4, 84, 84), device="cuda",
                 alpha=0.4, priority_epsilon=1e-6):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.states = np.empty((capacity, *observation_shape), dtype=np.uint8)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.discounts = np.empty(capacity, dtype=np.float32)
        self.alpha = float(alpha)
        self.priority_epsilon = float(priority_epsilon)
        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.0
        self.position = self.size = 0

    def add(self, state, action, reward, next_state, done, discount=1.0):
        i = self.position
        self.states[i] = np.asarray(state, dtype=np.uint8)
        self.next_states[i] = np.asarray(next_state, dtype=np.uint8)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.discounts[i] = float(discount)
        self.sum_tree.add(self.max_priority, i)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size < batch_size:
            raise ValueError("No hay suficientes transiciones para el batch")
        data_indices = np.empty(batch_size, dtype=np.int64)
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        total = self.sum_tree.total
        segment = total / batch_size
        # Una muestra aleatoria por segmento reduce la varianza del batch.
        for j in range(batch_size):
            value = np.random.uniform(segment * j, segment * (j + 1))
            tree_index, data_index, priority = self.sum_tree.get_leaf(value)
            tree_indices[j] = tree_index
            data_indices[j] = data_index
            priorities[j] = priority

        states = torch.as_tensor(self.states[data_indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[data_indices], device=self.device)
        actions = torch.as_tensor(self.actions[data_indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[data_indices], device=self.device)
        dones = torch.as_tensor(self.dones[data_indices], device=self.device)
        discounts = torch.as_tensor(self.discounts[data_indices], device=self.device)
        probabilities = priorities / total
        weights = (self.size * probabilities) ** (-float(beta))
        weights /= weights.max()
        weights = torch.as_tensor(weights.astype(np.float32), device=self.device)
        return states, actions, rewards, next_states, dones, discounts, weights, tree_indices

    def update_priorities(self, tree_indices, priorities):
        for tree_index, priority in zip(tree_indices, priorities):
            scaled = (abs(float(priority)) + self.priority_epsilon) ** self.alpha
            self.sum_tree.update(int(tree_index), scaled)
            self.max_priority = max(self.max_priority, scaled)

    def __len__(self):
        return self.size


# Nombre explícito para el componente PER; se conserva ReplayBuffer para
# mantener compatibilidad con scripts y checkpoints del proyecto.
PrioritizedReplayBuffer = ReplayBuffer


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
        return state, action, total_reward, next_state, done, self.gamma ** steps

    def append(self, state, action, reward, next_state, done, boundary=None):
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
