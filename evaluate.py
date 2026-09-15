"""Evaluación greedy exacta de 5 episodios y grabación de un MP4."""

import argparse
import os

from agent import D3QNAgent
from wrappers import make_atari_env


def main(args):
    os.makedirs(args.video_dir, exist_ok=True)
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA no está disponible; se usará CPU.")
            args.device = "cpu"
    probe = make_atari_env(args.env, episodic_life=False, clip_reward=False)
    agent = D3QNAgent(probe.action_space.n, device=args.device)
    agent.load(args.checkpoint)
    probe.close()
    env = make_atari_env(args.env, render_mode="rgb_array", episodic_life=False, clip_reward=False)
    env = __import__("gymnasium").wrappers.RecordVideo(env, args.video_dir, episode_trigger=lambda ep: ep == 0)
    scores = []
    for episode in range(5):
        obs, _ = env.reset()
        done = False; score = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, epsilon=0.0))
            score += reward; done = terminated or truncated
        scores.append(score)
        print(f"Episodio {episode + 1}/5: recompensa={score:.1f}")
    print(f"Máxima recompensa en 5 episodios: {max(scores):.1f}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--env", default="ALE/SpaceInvaders-v5"); p.add_argument("--video-dir", default="videos")
    p.add_argument("--device", default="cuda"); main(p.parse_args())
