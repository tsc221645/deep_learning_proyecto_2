"""Fine-tuning seguro de D3QN V2 desde ``best.pt``.

Características:
  * epsilon restart: 0.05 -> 0.01 en cada ciclo de 250k steps;
  * Adam nuevo con lr=5e-5 y CosineAnnealingWarmRestarts;
  * beta PER fijo en 1.0;
  * replay buffer nuevo con warm-up de 50k steps;
  * checkpoints separados: finetune_latest.pt y finetune_best.pt.

Ejemplo:
    python finetune_v2.py --steps 2_000_000 --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from agent import D3QNAgent
from replay_buffer import NStepAccumulator, PrioritizedReplayBuffer
from wrappers import make_atari_env


def cyclic_epsilon(step: int, cycle_steps: int = 250_000,
                   decay_steps: int = 100_000) -> float:
    """Reinicia epsilon a 0.05 y lo lleva linealmente a 0.01."""
    position = int(step) % cycle_steps
    if position < decay_steps:
        return 0.05 - 0.04 * position / max(1, decay_steps)
    return 0.01


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@torch.no_grad()
def evaluate(agent, env_id: str, episodes: int = 3) -> float:
    """Evalúa episodios completos con epsilon=0 y recompensa sin clipping."""
    env = make_atari_env(env_id, episodic_life=False, clip_reward=False)
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, epsilon=0.0))
            score += reward
            done = terminated or truncated
        scores.append(score)
    env.close()
    return float(np.mean(scores))


def save_checkpoint(agent, path, step, best_eval, episode, epsilon, updates):
    agent.save(path, step=step, best_eval=best_eval, episode=episode,
               epsilon=epsilon, gradient_updates=updates, beta=1.0)


def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    env = make_atari_env(args.env)
    agent = D3QNAgent(env.action_space.n, device, learning_rate=args.lr, gamma=args.gamma)

    # Una sesión nueva carga solo pesos; --resume restaura también optimizer y
    # scheduler para continuar exactamente el fine-tuning interrumpido.
    checkpoint = agent.load(args.checkpoint, load_optimizer=args.resume)
    print(f"Cargado best.pt: step original={checkpoint.get('step', 'desconocido')}, "
          f"best_eval={checkpoint.get('best_eval', 'desconocido')}")
    updates_per_cycle = args.cycle_steps // args.train_frequency
    if args.resume:
        # agent.__init__ crea un scheduler V2 por defecto; lo reemplazamos por
        # CosineWarmRestarts antes de restaurar su estado específico.
        scheduler_state = checkpoint.get("scheduler")
        agent.scheduler = CosineAnnealingWarmRestarts(
            agent.optimizer,
            T_0=updates_per_cycle,
            T_mult=1,
            eta_min=args.eta_min,
        )
        if scheduler_state is not None:
            agent.scheduler.load_state_dict(scheduler_state)
    else:
        agent.optimizer = Adam(agent.online.parameters(), lr=args.lr, eps=1.5e-4)
        agent.scheduler = CosineAnnealingWarmRestarts(
            agent.optimizer,
            T_0=updates_per_cycle,
            T_mult=1,
            eta_min=args.eta_min,
        )

    replay = PrioritizedReplayBuffer(
        args.buffer_size, env.observation_space.shape, device, alpha=args.per_alpha
    )
    nstep = NStepAccumulator(args.n_step, args.gamma)
    writer = SummaryWriter(args.log_dir, flush_secs=10)
    obs, _ = env.reset(seed=args.seed)
    start_step = int(checkpoint.get("step", 0)) if args.resume else 0
    episode = int(checkpoint.get("episode", 0)) if args.resume else 0
    episode_reward = 0.0
    recent = deque(maxlen=100)
    best_eval = float(checkpoint.get("best_eval", -float("inf")))
    gradient_updates = int(checkpoint.get("gradient_updates", 0)) if args.resume else 0
    started = time.perf_counter()
    last_loss = float("nan")

    print(f"Device={device}; warm-up={args.warmup_steps:,}; "
          f"T_0={updates_per_cycle:,} actualizaciones; beta=1.0")

    for step in range(start_step + 1, args.steps + 1):
        # Durante el warm-up se explota best.pt. Después, el reloj de epsilon
        # usa el step global para que sus reinicios ocurran en 250k, 500k, ...
        epsilon = 0.0 if step <= args.warmup_steps else cyclic_epsilon(
            step, args.cycle_steps, args.epsilon_decay_steps
        )
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        for transition in nstep.append(obs, action, reward, next_obs, terminated,
                                      boundary=terminated or truncated):
            replay.add(*transition)
        obs = next_obs
        episode_reward += float(reward)

        did_update = False
        if step > args.warmup_steps and len(replay) >= args.learning_starts \
                and step % args.train_frequency == 0:
            # Corrección de sesgo completa desde el primer batch.
            batch = replay.sample(args.batch_size, beta=1.0)
            last_loss, q_mean, td_errors, tree_indices = agent.train_step(batch)
            replay.update_priorities(tree_indices, td_errors)
            gradient_updates += 1
            did_update = True
            writer.add_scalar("train/loss", last_loss, step)
            writer.add_scalar("train/q_mean", q_mean, step)
            writer.add_scalar("train/beta", 1.0, step)

        if did_update:
            # El scheduler usa actualizaciones reales: 250k env steps / 4.
            virtual_update = step // args.train_frequency
            agent.scheduler.step(virtual_update)

        if step % args.target_update == 0:
            agent.update_target()

        if terminated or truncated:
            episode += 1
            recent.append(episode_reward)
            writer.add_scalar("episode/reward", episode_reward, episode)
            writer.add_scalar("episode/reward_100_mean", np.mean(recent), episode)
            obs, _ = env.reset()
            episode_reward = 0.0

        if step % args.progress_interval == 0 or step == args.steps:
            elapsed = time.perf_counter() - started
            sps = step / max(elapsed, 1e-6)
            eta = (args.steps - step) / max(sps, 1e-6)
            lr = agent.optimizer.param_groups[0]["lr"]
            print(
                f"[FineTune V2] step={step:,}/{args.steps:,} "
                f"({100 * step / args.steps:6.2f}%) | SPS={sps:,.1f} | "
                f"restante≈{format_duration(eta)} | replay={len(replay):,} | "
                f"eps={epsilon:.4f} | beta=1.0000 | loss={last_loss:.5f} | lr={lr:.2e}",
                flush=True,
            )
            writer.add_scalar("performance/sps", sps, step)
            writer.add_scalar("performance/eta_seconds", eta, step)
            writer.add_scalar("performance/replay_size", len(replay), step)
            writer.add_scalar("exploration/epsilon", epsilon, step)
            writer.add_scalar("optimizer/lr", lr, step)
            writer.flush()

        if step % args.checkpoint_interval == 0:
            save_checkpoint(agent, os.path.join(args.checkpoint_dir, "finetune_latest.pt"),
                            step, best_eval, episode, epsilon, gradient_updates)
            score = evaluate(agent, args.env, args.eval_episodes)
            writer.add_scalar("eval/mean_reward", score, step)
            if score > best_eval:
                best_eval = score
                save_checkpoint(agent, os.path.join(args.checkpoint_dir, "finetune_best.pt"),
                                step, best_eval, episode, epsilon, gradient_updates)
                print(f"Nuevo finetune_best: evaluación={score:.2f}", flush=True)

    save_checkpoint(agent, os.path.join(args.checkpoint_dir, "finetune_latest.pt"),
                    args.steps, best_eval, episode, epsilon, gradient_updates)
    env.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--resume", action="store_true",
                        help="Restaura optimizer, scheduler, step y métricas del checkpoint")
    parser.add_argument("--env", default="ALE/SpaceInvaders-v5")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--warmup-steps", type=int, default=50_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=80_000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eta-min", type=float, default=5e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cycle-steps", type=int, default=250_000)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--per-alpha", type=float, default=0.4)
    parser.add_argument("--learning-starts", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    parser.add_argument("--checkpoint-dir", default="checkpoints_finetune_v2")
    parser.add_argument("--log-dir", default="runs/finetune_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
