"""Entrenamiento D3QN: python train.py --steps 1000000 --buffer-size 1000000"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import D3QNAgent
from replay_buffer import NStepAccumulator, PrioritizedReplayBuffer
from schedules import per_beta, two_phase_epsilon
from wrappers import make_atari_env


def evaluate(agent, env_id, episodes=3):
    env = make_atari_env(env_id, episodic_life=False, clip_reward=False)
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, 0.0))
            total += reward
            done = terminated or truncated
        scores.append(total)
    env.close()
    return float(np.mean(scores))


def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = make_atari_env(args.env)
    env.action_space.seed(args.seed)
    agent = D3QNAgent(env.action_space.n, device, args.lr, args.gamma)
    replay = PrioritizedReplayBuffer(args.buffer_size, env.observation_space.shape, device, alpha=args.per_alpha)
    nstep = NStepAccumulator(args.n_step, args.gamma)
    writer = SummaryWriter(args.log_dir)
    recent = deque(maxlen=100); best_eval = -float("inf"); start_step = 0
    episode_reward = 0.0; episode = 0; last_loss = last_q = 0.0
    if args.resume:
        checkpoint = agent.load(args.resume, load_optimizer=True)
        start_step = int(checkpoint.get("step", 0))
        best_eval = float(checkpoint.get("best_eval", best_eval))
        if args.resume_step is not None:
            start_step = args.resume_step
        if args.resume_best_eval is not None:
            best_eval = args.resume_best_eval
        episode = int(checkpoint.get("episode", 0))
        print(f"Reanudando desde step={start_step:,}, episode={episode}, best_eval={best_eval:.2f}")
    obs, _ = env.reset(seed=args.seed)

    for step in range(start_step + 1, args.steps + 1):
        epsilon = two_phase_epsilon(
            step,
            phase_1_steps=args.epsilon_phase1_steps,
            phase_2_end=args.epsilon_phase2_end,
        )
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Para el bootstrap, una truncación temporal no debe tratarse como muerte.
        for transition in nstep.append(obs, action, reward, next_obs, terminated,
                                      boundary=terminated or truncated):
            replay.add(*transition)
        obs = next_obs; episode_reward += reward
        did_update = False
        if len(replay) >= args.learning_starts and step % args.train_frequency == 0:
            beta = per_beta(step, args.per_beta_start, 1.0, args.per_beta_steps)
            batch = replay.sample(args.batch_size, beta=beta)
            last_loss, last_q, td_errors, indices = agent.train_step(batch)
            replay.update_priorities(indices, td_errors)
            did_update = True
            writer.add_scalar("train/loss", last_loss, step)
            writer.add_scalar("train/q_mean", last_q, step)
            writer.add_scalar("train/per_beta", beta, step)
        if step % args.target_update == 0:
            agent.update_target()
        # Se ejecuta después de que comienza el aprendizaje real y usa el step
        # global del entorno, por lo que 4M y 6M siguen siendo hitos exactos.
        # No se llama durante el warm-up: todavía no existe un optimizer.step().
        if did_update:
            agent.step_scheduler(step)
        if terminated or truncated:
            episode += 1; recent.append(episode_reward)
            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/reward_100_mean", np.mean(recent), episode)
            print(f"step={step:,} episode={episode} reward={episode_reward:.1f} eps={epsilon:.3f}")
            obs, _ = env.reset(); episode_reward = 0.0
        metadata = {"step": step, "epsilon": epsilon, "best_eval": best_eval,
                    "episode": episode, "args": vars(args)}
        if step % args.checkpoint_interval == 0:
            agent.save(os.path.join(args.checkpoint_dir, "latest.pt"), **metadata)
            score = evaluate(agent, args.env, args.eval_episodes)
            writer.add_scalar("eval/mean_reward", score, step)
            if score > best_eval:
                best_eval = score
                best_metadata = {**metadata, "best_eval": best_eval}
                agent.save(os.path.join(args.checkpoint_dir, "best.pt"), **best_metadata)
                print(f"Nuevo mejor modelo: evaluación={score:.2f}")
        if step % 1_000_000 == 0:
            historical_path = os.path.join(args.checkpoint_dir, f"checkpoint_{step // 1_000_000}M.pt")
            historical_metadata = {"step": step, "epsilon": epsilon, "best_eval": best_eval,
                                  "episode": episode, "args": vars(args)}
            agent.save(historical_path, **historical_metadata)
            print(f"Checkpoint histórico guardado: {historical_path}")
    agent.save(os.path.join(args.checkpoint_dir, "latest.pt"), step=args.steps,
               epsilon=epsilon, best_eval=best_eval, episode=episode, args=vars(args))
    env.close(); writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="ALE/SpaceInvaders-v5"); p.add_argument("--steps", type=int, default=10_000_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-phase1-steps", type=int, default=1_000_000)
    p.add_argument("--epsilon-phase2-end", type=int, default=4_000_000)
    p.add_argument("--learning-starts", type=int, default=80_000)
    p.add_argument("--train-frequency", type=int, default=4)
    p.add_argument("--target-update", type=int, default=80_000)
    p.add_argument("--n-step", type=int, default=3, help="Longitud del retorno multi-step")
    p.add_argument("--per-alpha", type=float, default=0.4, help="0 desactiva Prioritized Replay")
    p.add_argument("--per-beta-start", type=float, default=0.4)
    p.add_argument("--per-beta-steps", type=int, default=10_000_000)
    p.add_argument("--checkpoint-interval", type=int, default=250_000); p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--checkpoint-dir", default="checkpoints"); p.add_argument("--log-dir", default="runs/spaceinvaders")
    p.add_argument("--resume", default=None, help="Checkpoint desde el que continuar, por ejemplo checkpoints/latest.pt")
    p.add_argument("--resume-step", type=int, default=None,
                   help="Sobrescribe el paso si el checkpoint antiguo no lo guardó")
    p.add_argument("--resume-best-eval", type=float, default=None,
                   help="Sobrescribe best_eval si el checkpoint antiguo no lo guardó")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--cpu", action="store_true")
    main(p.parse_args())
