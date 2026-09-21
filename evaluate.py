"""Evaluación greedy exacta de 5 episodios y grabación de un MP4."""

import argparse
import os
import shutil

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
    # Se graban los cinco episodios porque el de mayor puntuación solo se
    # conoce después de haber terminado la evaluación completa.
    env = __import__("gymnasium").wrappers.RecordVideo(env, args.video_dir, episode_trigger=lambda ep: True)
    scores = []
    for episode in range(5):
        # Semillas fijas hacen el ensayo reproducible y garantizan que el
        # video y la evaluación en vivo usen exactamente cinco partidas.
        obs, _ = env.reset(seed=args.seed + episode)
        done = False; score = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act(obs, epsilon=0.0))
            score += reward; done = terminated or truncated
        scores.append(score)
        print(f"Episodio {episode + 1}/5: recompensa={score:.1f}")
    print(f"Promedio en 5 episodios: {sum(scores) / len(scores):.1f}")
    best_episode = int(max(range(len(scores)), key=lambda i: scores[i]))
    best_score = scores[best_episode]
    print(f"Máxima recompensa en 5 episodios: {best_score:.1f} (episodio {best_episode + 1})")
    env.close()

    episode_video = os.path.join(args.video_dir, f"rl-video-episode-{best_episode}.mp4")
    best_video = os.path.join(args.video_dir, "best-score-episode.mp4")
    if os.path.exists(episode_video):
        shutil.copy2(episode_video, best_video)
        print(f"Video del episodio con mayor puntuación: {best_video}")
    else:
        print(f"No se encontró el video esperado: {episode_video}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints_v2/best_5m_800max.pt")
    p.add_argument("--env", default="ALE/SpaceInvaders-v5"); p.add_argument("--video-dir", default="videos")
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
