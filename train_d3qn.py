"""
D3QN Training Script for OBELIX
Run:  python train_d3qn.py --render_every 1

Key features (vs. original):
  A. PER + n-step returns  (via enhanced D3QNAgent)
  B. Forward-biased exploration  (via enhanced D3QNAgent)
  C. Potential-Based Reward Shaping (PBRS) — distance-to-box gradient
  D. Curriculum learning — easy → hard
  E. Tuned hyperparameters (5000 episodes, lr=3e-4, batch=128, buffer=200K)
"""

import argparse
import json
import math
import os

import numpy as np

from d3qnagent import D3QNAgent, ObservationStacker
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# Multi-Component Reward Shaping
#
# 3 components, all PBRS (γ·Φ(s')−Φ(s)) to preserve optimal policy:
#
#   1. BOT→BOX distance:      Guides the agent toward the box
#   2. HEADING toward box:     Rewards facing the box, not just being near it
#   3. BOX→BOUNDARY distance:  Once attached, guides pushing toward nearest wall
#
# WHY this is better than distance-only PBRS:
#   - Distance alone: agent can circle the box without facing it → never attaches
#   - Heading bonus: agent learns to TURN TOWARD the box → faster attachment
#   - Box→boundary: without this, push phase has NO gradient → random pushing
# ---------------------------------------------------------------------------

def compute_distance(env: OBELIX) -> float:
    """Euclidean distance from bot center to box center."""
    dx = env.bot_center_x - env.box_center_x
    dy = env.bot_center_y - env.box_center_y
    return math.sqrt(dx * dx + dy * dy)


def compute_box_to_boundary_dist(env: OBELIX) -> float:
    """Distance from box center to the nearest arena boundary."""
    margin = 10  # arena has a 10px border
    frame_h, frame_w = env.frame_size[0], env.frame_size[1]
    dist_left = env.box_center_x - margin
    dist_right = (frame_w - margin) - env.box_center_x
    dist_top = env.box_center_y - margin
    dist_bottom = (frame_h - margin) - env.box_center_y
    return float(min(dist_left, dist_right, dist_top, dist_bottom))


def compute_heading_similarity(env: OBELIX) -> float:
    """
    Cosine similarity between the robot's facing direction and the
    direction toward the box.  Returns value in [-1, +1].
      +1 = facing directly toward the box
       0 = perpendicular
      -1 = facing directly away
    """
    dx = env.box_center_x - env.bot_center_x
    dy = env.box_center_y - env.bot_center_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return 1.0  # on top of the box
    angle_rad = math.radians(env.facing_angle)
    fx = math.cos(angle_rad)
    fy = math.sin(angle_rad)

    tx = dx / dist
    ty = dy / dist

    # Cosine similarity
    return fx * tx + fy * ty


def compute_shaped_reward(
    env: OBELIX,
    env_reward: float,
    prev_bot_box_dist: float,
    curr_bot_box_dist: float,
    prev_box_boundary_dist: float,
    curr_box_boundary_dist: float,
    prev_heading: float,
    curr_heading: float,
    max_dist: float,
    gamma: float,
    shaping_scale: float,
) -> float:
    """
    Multi-component PBRS reward shaping.

    Component 1: Bot→Box distance (always active)
      Φ₁(s) = -dist(bot, box) / max_dist

    Component 2: Heading toward box (only when NOT attached)
      Φ₂(s) = heading_similarity (cosine, in [-1, +1])

    Component 3: Box→Boundary distance (only when attached/pushing)
      Φ₃(s) = -dist(box, nearest_boundary) / max_dist
    """
    shaped = env_reward
    phi1_prev = -prev_bot_box_dist / max_dist
    phi1_curr = -curr_bot_box_dist / max_dist
    approach_shaping = gamma * phi1_curr - phi1_prev
    shaped += shaping_scale * approach_shaping

    if not env.enable_push:
        phi2_prev = prev_heading  # in [-1, +1]
        phi2_curr = curr_heading
        heading_shaping = gamma * phi2_curr - phi2_prev
        shaped += (shaping_scale * 0.5) * heading_shaping
    else:
        phi3_prev = -prev_box_boundary_dist / max_dist
        phi3_curr = -curr_box_boundary_dist / max_dist
        push_shaping = gamma * phi3_curr - phi3_prev
        shaped += (shaping_scale * 2.0) * push_shaping  # stronger signal for pushing

    return shaped


# Curriculum schedule

def get_curriculum_config(episode: int, total_episodes: int) -> dict:
    """
    Phase 1 (ep 1-500):      static box, NO walls  — warm-up: learn to find & push
    Phase 2 (ep 501-2000):   static box, WITH walls — learn wall navigation
    Phase 3 (ep 2001-4000):  static+blink, WITH walls — handle blinking
    Phase 4 (ep 4001-8000):  all diffs, WITH walls — generalize
    """
    if episode <= 500:
        return {"wall_obstacles": False, "difficulties": [0]}
    elif episode <= 2000:
        return {"wall_obstacles": True, "difficulties": [0]}
    elif episode <= 4000:
        return {"wall_obstacles": True, "difficulties": [0, 2]}
    else:
        return {"wall_obstacles": True, "difficulties": [0, 2, 3]}


# Evaluation  

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
    n_stack: int = 4,
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

        raw_obs = env.reset(seed=seed)
        stacker = ObservationStacker(n_stack=n_stack, obs_dim=18)
        obs = stacker.reset(raw_obs)
        rng = np.random.default_rng(seed)
        done = False
        total_reward = 0.0

        while not done:
            action_idx = agent.select_action(obs, rng=rng, explore=False, inference_epsilon=0.0)
            action = ACTIONS[action_idx]
            raw_obs, reward, done = env.step(action, render=False)
            obs = stacker.push(raw_obs)
            total_reward += float(reward)

        returns.append(total_reward)

    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }


# Training loop

def train() -> None:
    parser = argparse.ArgumentParser(description="Train D3QN + PER + n-step + PBRS for OBELIX")

    # Training
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Environment
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--box_speed", type=int, default=2)

    # Network
    parser.add_argument("--hidden_dim1", type=int, default=256)
    parser.add_argument("--hidden_dim2", type=int, default=128)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_capacity", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=int, default=200_000)
    parser.add_argument("--target_update_freq", type=int, default=2000)

    # PER
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--per_beta_start", type=float, default=0.4)
    parser.add_argument("--per_beta_frames", type=int, default=200_000)
    parser.add_argument("--per_eps", type=float, default=1e-5)

    # n-step
    parser.add_argument("--n_step", type=int, default=3)

    # Observation stacking
    parser.add_argument("--n_stack", type=int, default=6,
                        help="Number of observations to stack (1=no stacking)")

    # Reward shaping
    parser.add_argument("--shaping_scale", type=float, default=10.0,
                        help="Scale factor for PBRS distance shaping")

    # Saving / eval
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--weights_name", type=str, default="obelix_d3qn_weights.pth")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_runs", type=int, default=6)
    parser.add_argument("--render_every", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.weights_name)
    best_path = os.path.join(args.save_dir, f"best_{args.weights_name}")

    rng = np.random.default_rng(args.seed)

    stacked_dim = args.n_stack * 18  # e.g. 4 * 18 = 72

    agent = D3QNAgent(
        state_dim=stacked_dim,
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

    print("D3QN Training  (PER + n-step + PBRS + Curriculum)")
    print(f"  Episodes:       {args.episodes}")
    print(f"  LR:             {args.lr}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Buffer:         {args.buffer_capacity}")
    print(f"  PER alpha:      {args.per_alpha}")
    print(f"  n-step:         {args.n_step}")
    print(f"  Obs stacking:   {args.n_stack} frames → {stacked_dim}-dim input")
    print(f"  Shaping scale:  {args.shaping_scale}")
    print(f"  Eps decay:      {args.epsilon_decay}")
    print(f"  Curriculum:     L1(nowall,1-500) -> L2(wall,501-2000) -> L3(blink+wall,2001-4000) -> L4(all+wall,4001-8000)")
    print(f"  Save to:        {checkpoint_path}")
    print(f"  Device:         {agent.device}")

    for episode in range(1, args.episodes + 1):

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

        raw_obs = env.reset(seed=env_seed)
        stacker = ObservationStacker(n_stack=args.n_stack, obs_dim=18)
        obs = stacker.reset(raw_obs)
        done = False
        episode_reward = 0.0        # actual env reward (Codabench metric)
        episode_shaped_reward = 0.0  # shaped reward (training signal)
        episode_steps = 0
        episode_losses: list[float] = []
        found_box = False
        pushed_to_boundary = False
        steps_to_detection = -1      # -1 = never found
        action_counts = [0] * 5      # per-action counts
        episode_explore_bonus = 0.0  # total exploration bonus this episode

        render_episode = args.render_every > 0 and (episode % args.render_every == 0)

        prev_dist = compute_distance(env)
        prev_box_boundary = compute_box_to_boundary_dist(env)
        prev_heading = compute_heading_similarity(env)

        while not done:
            action_idx = agent.select_action(obs, rng=rng, explore=True)
            action = ACTIONS[action_idx]
            action_counts[action_idx] += 1
            raw_next_obs, reward, done = env.step(action, render=render_episode)
            next_obs = stacker.push(raw_next_obs)

            # Track milestones
            if env.enable_push and not found_box:
                found_box = True
                steps_to_detection = episode_steps + 1
            if done and env.enable_push and env._box_touches_boundary(env.box_center_x, env.box_center_y):
                pushed_to_boundary = True
            curr_dist = compute_distance(env)
            curr_box_boundary = compute_box_to_boundary_dist(env)
            curr_heading = compute_heading_similarity(env)

            shaped_reward = compute_shaped_reward(
                env=env,
                env_reward=float(reward),
                prev_bot_box_dist=prev_dist,
                curr_bot_box_dist=curr_dist,
                prev_box_boundary_dist=prev_box_boundary,
                curr_box_boundary_dist=curr_box_boundary,
                prev_heading=prev_heading,
                curr_heading=curr_heading,
                max_dist=max_dist,
                gamma=args.gamma,
                shaping_scale=args.shaping_scale,
            )
            prev_dist = curr_dist
            prev_box_boundary = curr_box_boundary
            prev_heading = curr_heading


            # Agent learns from shaped reward
            loss = agent.step(obs, action_idx, shaped_reward, next_obs, done)
            if loss is not None:
                episode_losses.append(loss)

            obs = next_obs
            episode_reward += float(reward)
            episode_shaped_reward += shaped_reward
            episode_steps += 1

        avg_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")
        milestones = ""
        if found_box:
            milestones += "Box found"
        if pushed_to_boundary:
            milestones += "Box pushed to boundary"

        if not wall_obs:
            phase = "L1 no walls"
        elif difficulties == [0]:
            phase = "L2 wall"
        elif difficulties == [0, 2]:
            phase = "L3 blink and wall"
        else:
            phase = "L4-all"

        print(
            f"[Ep {episode:4d}/{args.episodes}] {phase} diff={difficulty} "
            f"envR={episode_reward:9.1f} shapedR={episode_shaped_reward:9.1f} "
            f"steps={episode_steps:4d} eps={agent.epsilon:.4f} loss={avg_loss:.5f}{milestones}"
        )
        log_entry = {
            "episode": episode,
            "phase": phase,
            "difficulty": difficulty,
            "wall_obstacles": wall_obs,
            "env_reward": episode_reward,
            "shaped_reward": episode_shaped_reward,
            "steps": episode_steps,
            "epsilon": agent.epsilon,
            "avg_loss": avg_loss if not math.isnan(avg_loss) else None,
            "found_box": found_box,
            "pushed_to_boundary": pushed_to_boundary,
            "steps_to_detection": steps_to_detection,
            "action_counts": action_counts,
            "explore_bonus": episode_explore_bonus,
        }
        log_path = os.path.join(args.save_dir, "training_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        if episode % args.save_every == 0:
            agent.save(checkpoint_path)
            print(f"Saved -> {checkpoint_path}")

        if args.eval_every > 0 and episode % args.eval_every == 0:
            metrics = evaluate_agent(
                agent=agent,
                runs=args.eval_runs,
                base_seed=args.seed + 10000 + episode,
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=True,          # always eval with walls
                box_speed=args.box_speed,
                difficulty_levels=[0, 2, 3],  # all diffs like Codabench
                n_stack=args.n_stack,
            )
            print(
                f"[Eval w/ wall] "
                f"mean={metrics['mean']:.1f}  std={metrics['std']:.1f}  "
                f"min={metrics['min']:.1f}  max={metrics['max']:.1f}"
            )

            if metrics["mean"] > best_mean:
                best_mean = metrics["mean"]
                agent.save(best_path)
                print(f"New best! → {best_path}  (mean={best_mean:.1f})")

            # Save eval metrics to log
            eval_entry = {
                "episode": episode,
                "eval_mean": metrics["mean"],
                "eval_std": metrics["std"],
                "eval_min": metrics["min"],
                "eval_max": metrics["max"],
            }
            eval_log_path = os.path.join(args.save_dir, "eval_log.jsonl")
            with open(eval_log_path, "a") as f:
                f.write(json.dumps(eval_entry) + "\n")

    agent.save(checkpoint_path)
    print(f"Training complete!  Final checkpoint → {checkpoint_path}")
    if os.path.exists(best_path):
        print(f"Best checkpoint → {best_path}  (mean={best_mean:.1f})")
    if os.path.exists(best_path):
        import shutil
        weights_submission = os.path.join(args.save_dir, "weights.pth")
        shutil.copy2(best_path, weights_submission)
        print(f"\nCopied best weights to {weights_submission}")


if __name__ == "__main__":
    train()
