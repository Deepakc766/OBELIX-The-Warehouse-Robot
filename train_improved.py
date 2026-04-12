"""
Improved D3QN training script for OBELIX.

Key improvements over train_d3qn.py:
  1. Potential-Based Reward Shaping (PBRS) — distance-to-box gradient
  2. Curriculum learning — easy → hard
  3. PER + n-step returns (via enhanced D3QNAgent)
  4. Tuned hyperparameters
"""

import argparse
import math
import os

import numpy as np

from d3qnagent import D3QNAgent
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]




def compute_distance(env: OBELIX) -> float:
    """Euclidean distance from bot center to box center."""
    dx = env.bot_center_x - env.box_center_x
    dy = env.bot_center_y - env.box_center_y
    return math.sqrt(dx * dx + dy * dy)


def compute_shaped_reward(
    env_reward: float,
    prev_dist: float,
    curr_dist: float,
    max_dist: float,
    gamma: float,
    shaping_scale: float,
) -> float:
    """
    Potential-Based Reward Shaping (Ng et al. 1999).

    Φ(s) = -distance(bot, box) / max_distance
    Shaped reward = env_reward + shaping_scale * (γ * Φ(s') - Φ(s))

    This preserves the optimal policy while providing a smooth gradient
    toward the box.
    """
    prev_potential = -prev_dist / max_dist
    curr_potential = -curr_dist / max_dist
    shaping = gamma * curr_potential - prev_potential
    return env_reward + shaping_scale * shaping


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent: D3QNAgent,
    runs: int,
    base_seed: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    wall_obstacles: bool,
    box_speed: int,
    difficulty_levels: list[int],
) -> dict[str, float]:
    returns: list[float] = []

    for run_idx in range(runs):
        difficulty = difficulty_levels[run_idx % len(difficulty_levels)]
        seed = base_seed + run_idx

        env = OBELIX(
            scaling_factor=scaling_factor,
            arena_size=arena_size,
            max_steps=max_steps,
            wall_obstacles=wall_obstacles,
            difficulty=difficulty,
            box_speed=box_speed,
            seed=seed,
        )

        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        done = False
        total_reward = 0.0

        while not done:
            action_idx = agent.select_action(obs, rng=rng, explore=False, inference_epsilon=0.0)
            action = ACTIONS[action_idx]
            obs, reward, done = env.step(action, render=False)
            total_reward += float(reward)

        returns.append(total_reward)

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }


# ---------------------------------------------------------------------------
# Curriculum schedule
# ---------------------------------------------------------------------------

def get_curriculum_config(episode: int, total_episodes: int) -> dict:
    """
    Phase 1 (0-40%):  static box (diff=0), no walls
    Phase 2 (40-70%): static box (diff=0), with walls
    Phase 3 (70-100%): all difficulties (0,2,3), with walls
    """
    p1_end = int(total_episodes * 0.40)
    p2_end = int(total_episodes * 0.70)

    if episode <= p1_end:
        return {"wall_obstacles": False, "difficulties": [0]}
    elif episode <= p2_end:
        return {"wall_obstacles": True, "difficulties": [0]}
    else:
        return {"wall_obstacles": True, "difficulties": [0, 2, 3]}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.weights_name)
    best_path = os.path.join(args.save_dir, f"best_{args.weights_name}")

    rng = np.random.default_rng(args.seed)

    agent = D3QNAgent(
        state_dim=18,
        action_dim=5,
        hidden_dims=(args.hidden_dim1, args.hidden_dim2),
        lr=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        per_alpha=args.per_alpha,
        per_beta_start=args.per_beta_start,
        per_beta_frames=args.per_beta_frames,
        per_eps=args.per_eps,
        n_step=args.n_step,
    )

    # Resume from checkpoint if available
    if args.resume and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Resumed from {checkpoint_path}")

    # Max possible distance in the arena (diagonal)
    max_dist = math.sqrt(2) * args.arena_size

    best_mean = -float("inf")
    print(f"Starting D3QN training for {args.episodes} episodes...")
    print(f"  PER alpha={args.per_alpha}, n-step={args.n_step}, shaping_scale={args.shaping_scale}")
    print(f"  Curriculum: Phase1(no wall) → Phase2(wall, static) → Phase3(wall, all diffs)")

    for episode in range(1, args.episodes + 1):
        # Curriculum
        curriculum = get_curriculum_config(episode, args.episodes)
        difficulties = curriculum["difficulties"]
        wall_obs = curriculum["wall_obstacles"]

        difficulty = int(rng.choice(difficulties))
        env_seed = args.seed + episode

        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=wall_obs,
            difficulty=difficulty,
            box_speed=args.box_speed,
            seed=env_seed,
        )

        obs = env.reset(seed=env_seed)
        done = False
        episode_reward = 0.0       # env reward (what Codabench measures)
        episode_shaped = 0.0       # shaped reward (what agent trains on)
        episode_steps = 0
        episode_losses: list[float] = []
        found_box = False
        pushed_to_boundary = False

        render_episode = args.render_every > 0 and (episode % args.render_every == 0)

        prev_dist = compute_distance(env)

        while not done:
            action_idx = agent.select_action(obs, rng=rng, explore=True)
            action = ACTIONS[action_idx]
            next_obs, reward, done = env.step(action, render=render_episode)

            # Track milestones
            if env.enable_push and not found_box:
                found_box = True
            if done and env.enable_push and env._box_touches_boundary(env.box_center_x, env.box_center_y):
                pushed_to_boundary = True

            # Compute shaped reward for training
            curr_dist = compute_distance(env)
            shaped_reward = compute_shaped_reward(
                env_reward=float(reward),
                prev_dist=prev_dist,
                curr_dist=curr_dist,
                max_dist=max_dist,
                gamma=args.gamma,
                shaping_scale=args.shaping_scale,
            )
            prev_dist = curr_dist

            # Agent learns from shaped reward
            loss = agent.step(obs, action_idx, shaped_reward, next_obs, done)
            if loss is not None:
                episode_losses.append(loss)
            obs = next_obs
            episode_reward += float(reward)
            episode_shaped += shaped_reward
            episode_steps += 1

        avg_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")
        milestones = ""
        if found_box:
            milestones += " ✓BOX"
        if pushed_to_boundary:
            milestones += " ✓BOUNDARY"

        print(
            f"[Ep {episode:4d}] diff={difficulty} wall={'Y' if wall_obs else 'N'} "
            f"env_R={episode_reward:9.1f} shaped_R={episode_shaped:9.1f} "
            f"steps={episode_steps:4d} eps={agent.epsilon:.4f} loss={avg_loss:.5f}{milestones}"
        )

        # Save periodically
        if episode % args.save_every == 0:
            agent.save(checkpoint_path)
            print(f"  Saved checkpoint → {checkpoint_path}")

        # Evaluate periodically
        if args.eval_every > 0 and episode % args.eval_every == 0:
            # Evaluate with wall_obstacles=True (matching Codabench)
            metrics = evaluate_agent(
                agent=agent,
                runs=args.eval_runs,
                base_seed=args.seed + 10000 + episode,
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=True,
                box_speed=args.box_speed,
                difficulty_levels=[0, 2, 3],
            )
            print(
                f"  [Eval w/ wall] "
                f"mean={metrics['mean']:.1f} std={metrics['std']:.1f} "
                f"min={metrics['min']:.1f} max={metrics['max']:.1f}"
            )

            if metrics["mean"] > best_mean:
                best_mean = metrics["mean"]
                agent.save(best_path)
                print(f"  ★ New best checkpoint → {best_path} (mean={best_mean:.1f})")

    agent.save(checkpoint_path)
    print(f"\nTraining complete. Final checkpoint → {checkpoint_path}")
    if os.path.exists(best_path):
        print(f"Best checkpoint → {best_path} (mean={best_mean:.1f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train D3QN + PER + n-step + PBRS for OBELIX")

    # Training
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Environment
    p.add_argument("--scaling_factor", type=int, default=5)
    p.add_argument("--arena_size", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--box_speed", type=int, default=2)

    # Network
    p.add_argument("--hidden_dim1", type=int, default=256)
    p.add_argument("--hidden_dim2", type=int, default=128)

    # Optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer_capacity", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=int, default=500_000)
    p.add_argument("--target_update_freq", type=int, default=1000)

    # PER
    p.add_argument("--per_alpha", type=float, default=0.6)
    p.add_argument("--per_beta_start", type=float, default=0.4)
    p.add_argument("--per_beta_frames", type=int, default=500_000)
    p.add_argument("--per_eps", type=float, default=1e-5)

    # n-step
    p.add_argument("--n_step", type=int, default=3)

    # Reward shaping
    p.add_argument("--shaping_scale", type=float, default=50.0,
                   help="Scale factor for PBRS distance shaping")

    # Saving / eval
    p.add_argument("--save_dir", type=str, default=".")
    p.add_argument("--weights_name", type=str, default="obelix_d3qn_improved.pth")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_runs", type=int, default=6)
    p.add_argument("--render_every", type=int, default=0)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
