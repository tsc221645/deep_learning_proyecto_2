"""Wrappers Atari compatibles con Gymnasium.

El orden recomendado de composición está documentado en :func:`make_atari_env`.
Los wrappers no dependen de ``gym.wrappers.atari_preprocessing`` para que el
proyecto sea explícito y fácil de estudiar en un curso de aprendizaje por
refuerzo.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Gymnasium 1.x mantiene los entornos ALE en el paquete separado ale-py y no
# siempre los registra mediante un simple ``import gymnasium``.
try:
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):
    # Permite importar este módulo para pruebas unitarias sin ale-py.
    ale_py = None


class NoopResetEnv(gym.Wrapper):
    """Ejecuta acciones NOOP aleatorias al reiniciar, como en DeepMind DQN."""

    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        meanings = env.unwrapped.get_action_meanings()
        if not meanings or meanings[0] != "NOOP":
            raise ValueError("La acción 0 del entorno debe ser NOOP")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        noops = int(self.np_random.integers(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, terminated, truncated, step_info = self.env.step(0)
            info.update(step_info)
            if terminated or truncated:
                obs, info = self.env.reset()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repite una acción y devuelve el máximo de los dos últimos frames."""

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip < 2:
            raise ValueError("skip debe ser >= 2")
        self._skip = skip
        self._obs_buffer = np.empty((2, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        obs = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # También conservamos el frame temprano: un episodio puede acabar
            # antes de llegar a los dos últimos pasos del intervalo.
            self._obs_buffer[0] = obs
            if i >= self._skip - 2:
                self._obs_buffer[i - (self._skip - 2)] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buffer[0] if i < self._skip - 2 else self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Convierte la pérdida de una vida en terminal de entrenamiento.

    Solo reinicia el juego real cuando el entorno base termina; en una pérdida
    de vida se ejecuta un NOOP para continuar desde el estado siguiente.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = int(info.get("lives", self.env.unwrapped.ale.lives()))
        life_lost = 0 < lives < self.lives
        self.lives = lives
        if life_lost and not self.was_real_done:
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = int(info.get("lives", self.env.unwrapped.ale.lives()))
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Ejecuta FIRE al iniciar juegos Atari que lo requieren."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        if "FIRE" not in meanings:
            raise ValueError("FireResetEnv requiere una acción FIRE")
        self.fire_action = meanings.index("FIRE")
        self.second_action = 2 if len(meanings) > 2 else 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, step_info = self.env.step(self.fire_action)
        info.update(step_info)
        if terminated or truncated:
            return self.env.reset(**kwargs)
        if self.second_action != 0:
            obs, _, terminated, truncated, step_info = self.env.step(self.second_action)
            info.update(step_info)
            if terminated or truncated:
                return self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """Gris y 84x84; conserva canal explícito para interoperabilidad."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width, self.height = width, height
        self.observation_space = spaces.Box(0, 255, (height, width), dtype=np.uint8)

    def observation(self, observation):
        frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)


class ClipRewardEnv(gym.RewardWrapper):
    """Reduce la recompensa a su signo, estabilizando la escala del gradiente."""

    def reward(self, reward):
        return float(np.sign(reward))


class FrameStack(gym.Wrapper):
    """Apila los últimos frames y devuelve un array ``(4, 84, 84)``."""

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames: deque[np.ndarray] = deque(maxlen=num_stack)
        shape = (num_stack, *env.observation_space.shape)
        self.observation_space = spaces.Box(0, 255, shape, dtype=np.uint8)

    def _get_obs(self):
        return np.stack(tuple(self.frames), axis=0).astype(np.uint8, copy=False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info


def make_atari_env(
    env_id: str = "ALE/SpaceInvaders-v5",
    render_mode: str | None = None,
    *,
    episodic_life: bool = True,
    clip_reward: bool = True,
) -> gym.Env:
    """Construye el pipeline Atari.

    ``episodic_life`` y ``clip_reward`` se desactivan en evaluación para que
    la puntuación sea la recompensa real de una partida completa.
    """
    env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0, render_mode=render_mode)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    env = FrameStack(env, num_stack=4)
    return env
