"""
PPO Training Script for OBELIX
Run:  python train_ppo.py

Key features:
  A. PPO-Clip with GAE (on-policy, stable updates)
  B. Shared Actor-Critic backbone
  C. Potential-Based Reward Shaping (PBRS) — 3-component
  D. Curriculum learning — 4-phase (warmup → wall → blink → all)
  E. Per-episode JSONL logging (survives Ctrl+C)
  F. Periodic evaluation with walls (mirrors Codabench)
"""

import argparse
import json
import math
import os

import numpy as np

from ppo_agent import PPOAgent
from d3qnagent import ObservationStacker
from obelix import OBELIX

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


def compute_distance(env: OBELIX) -> float:
    dx = env.bot_center_x - env.box_center_x
    dy = env.bot_center_y - env.box_center_y
    return math.sqrt(dx * dx + dy * dy)


def compute_box_to_boundary_dist(env: OBELIX) -> float:
    margin = 10
    frame_h, frame_w = env.frame_size[0], env.frame_size[1]
    dist_left = env.box_center_x - margin
    dist_right = (frame_w - margin) - env.box_center_x
    dist_top = env.box_center_y - margin
    dist_bottom = (frame_h - margin) - env.box_center_y
    return float(min(dist_left, dist_right, dist_top, dist_bottom))


def compute_heading_similarity(env: OBELIX) -> float:
    dx = env.box_center_x - env.bot_center_x
    dy = env.box_center_y - env.bot_center_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1e-6:
        return 1.0
    angle_rad = math.radians(env.facing_angle)
    fx = math.cos(angle_rad)
    fy = math.sin(angle_rad)
    tx = dx / dist
    ty = dy / dist
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
    shaped = env_reward

    # Component 1: approach the box
    phi1_prev = -prev_bot_box_dist / max_dist
    phi1_curr = -curr_bot_box_dist / max_dist
    shaped += shaping_scale * (gamma * phi1_curr - phi1_prev)

    if not env.enable_push:
        # Component 2: face the box (before attachment)
        phi2_prev = prev_heading
        phi2_curr = curr_heading
        shaped += (shaping_scale * 0.5) * (gamma * phi2_curr - phi2_prev)
    else:
        # Component 3: push box toward boundary (after attachment)
        phi3_prev = -prev_box_boundary_dist / max_dist
        phi3_curr = -curr_box_boundary_dist / max_dist
        shaped += (shaping_scale * 2.0) * (gamma * phi3_curr - phi3_prev)

    return shaped


# ---------------------------------------------------------------------------
# Curriculum schedule (same as D3QN)
# ---------------------------------------------------------------------------

def get_curriculum_config(episode: int, total_episodes: int) -> dict:
    if episode <= 500:
        return {"wall_obstacles": False, "difficulties": [0]}
    elif episode <= 2000:
        return {"wall_obstacles": True, "difficulties": [0]}
    elif episode <= 4000:
        return {"wall_obstacles": True, "difficulties": [0, 2]}
    else:
        return {"wall_obstacles": True, "difficulties": [0, 2, 3]}


def evaluate_agent(
    agent: PPOAgent,
    runs: int,
    base_seed: int,
    scaling_factor: int,
    arena_size: int,
    max_steps: int,
    wall_obstacles: bool,
    box_speed: int,
    difficulty_levels: list[int],
    n_stack: int = 6,
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
        done = False
        total_reward = 0.0

        while not done:
            action_idx, _, _ = agent.select_action(obs, explore=False)
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


def train() -> None:
    parser = argparse.ArgumentParser(description="Train PPO for OBELIX")

    # Training
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    # Environment
    parser.add_argument("--scaling_factor", type=int, default=5)
    parser.add_argument("--arena_size", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--box_speed", type=int, default=2)

    # PPO hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--value_coeff", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=64)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--update_every", type=int, default=1,
                        help="Update policy every N episodes (1=every episode)")

    # Observation stacking
    parser.add_argument("--n_stack", type=int, default=6)

    # Reward shaping
    parser.add_argument("--shaping_scale", type=float, default=10.0)

    # Saving / eval
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--weights_name", type=str, default="obelix_ppo_weights.pth")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_runs", type=int, default=6)
    parser.add_argument("--render_every", type=int, default=0)

    args = parser.parse_args()

    # ---- Setup ----
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.weights_name)
    best_path = os.path.join(args.save_dir, f"best_{args.weights_name}")

    rng = np.random.default_rng(args.seed)
    stacked_dim = args.n_stack * 18

    agent = PPOAgent(
        obs_dim=stacked_dim,
        action_dim=5,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        value_coeff=args.value_coeff,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        max_grad_norm=args.max_grad_norm,
    )

    if args.resume and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Resumed from {checkpoint_path}")

    max_dist = math.sqrt(2) * args.arena_size
    best_mean = -float("inf")

 
    print("PPO Training  (Actor-Critic + GAE + PBRS + Curriculum)")
    print(f"  Episodes:       {args.episodes}")
    print(f"  LR:             {args.lr}")
    print(f"  Hidden dim:     {args.hidden_dim}")
    print(f"  Gamma:          {args.gamma}")
    print(f"  GAE lambda:     {args.gae_lambda}")
    print(f"  Clip epsilon:   {args.clip_epsilon}")
    print(f"  Entropy coeff:  {args.entropy_coeff}")
    print(f"  Value coeff:    {args.value_coeff}")
    print(f"  PPO epochs:     {args.ppo_epochs}")
    print(f"  Mini-batch:     {args.mini_batch_size}")
    print(f"  Update every:   {args.update_every} episodes")
    print(f"  Obs stacking:   {args.n_stack} frames -> {stacked_dim}-dim input")
    print(f"  Shaping scale:  {args.shaping_scale}")
    print(f"  Curriculum:     P1(1-500) -> P2(501-2000) -> P3(2001-4000) -> P4(4001-8000)")
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
        episode_reward = 0.0
        episode_shaped_reward = 0.0
        episode_steps = 0
        found_box = False
        pushed_to_boundary = False
        steps_to_detection = -1
        action_counts = [0] * 5

        render_episode = args.render_every > 0 and (episode % args.render_every == 0)

        prev_dist = compute_distance(env)
        prev_box_boundary = compute_box_to_boundary_dist(env)
        prev_heading = compute_heading_similarity(env)

        while not done:
            # Select action
            action_idx, log_prob, value = agent.select_action(obs, explore=True)
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

            # Store transition in rollout buffer
            agent.store_transition(obs, action_idx, log_prob, shaped_reward, done, value)

            obs = next_obs
            episode_reward += float(reward)
            episode_shaped_reward += shaped_reward
            episode_steps += 1

        update_info = {}
        if episode % args.update_every == 0:
            update_info = agent.update(obs)  # obs = last observation

        if not wall_obs:
            phase = "L1-no wall"
        elif difficulties == [0]:
            phase = "L2-wall"
        elif difficulties == [0, 2]:
            phase = "L3-blink"
        else:
            phase = "L4-all"

        ploss = update_info.get("policy_loss", float("nan"))
        vloss = update_info.get("value_loss", float("nan"))
        ent = update_info.get("entropy", float("nan"))

        milestones = ""
        if found_box:
            milestones += "Box found"
        if pushed_to_boundary:
            milestones += "Boundary push"

        print(
            f"[Ep {episode:4d}/{args.episodes}] {phase} diff={difficulty} "
            f"envR={episode_reward:9.1f} shapedR={episode_shaped_reward:9.1f} "
            f"steps={episode_steps:4d} ploss={ploss:.5f} vloss={vloss:.5f} "
            f"ent={ent:.4f}{milestones}"
        )

        log_entry = {
            "episode": episode,
            "phase": phase,
            "difficulty": difficulty,
            "wall_obstacles": wall_obs,
            "env_reward": episode_reward,
            "shaped_reward": episode_shaped_reward,
            "steps": episode_steps,
            # "epsilon": 0.0,  # PPO doesn't use epsilon
            "avg_loss": update_info.get("total_loss"),
            "found_box": found_box,
            "pushed_to_boundary": pushed_to_boundary,
            "steps_to_detection": steps_to_detection,
            "action_counts": action_counts,
            "explore_bonus": 0.0,
            "policy_loss": update_info.get("policy_loss"),
            "value_loss": update_info.get("value_loss"),
            "entropy": update_info.get("entropy"),
        }
        log_path = os.path.join(args.save_dir, "training_log.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if episode % args.save_every == 0:
            agent.save(checkpoint_path)
            print(f"  Saved -> {checkpoint_path}")

        if args.eval_every > 0 and episode % args.eval_every == 0:
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
                n_stack=args.n_stack,
            )
            print(
                f"  [Eval w/ wall] "
                f"mean={metrics['mean']:.1f}  std={metrics['std']:.1f}  "
                f"min={metrics['min']:.1f}  max={metrics['max']:.1f}"
            )

            if metrics["mean"] > best_mean:
                best_mean = metrics["mean"]
                agent.save(best_path)
                print(f"  New best! -> {best_path}  (mean={best_mean:.1f})")

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
    print(f"Training complete!  Final checkpoint -> {checkpoint_path}")
    if os.path.exists(best_path):
        print(f"Best checkpoint -> {best_path}  (mean={best_mean:.1f})")

    if os.path.exists(best_path):
        import shutil
        weights_submission = os.path.join(args.save_dir, "weights.pth")
        shutil.copy2(best_path, weights_submission)
        print(f"\nCopied best weights to {weights_submission}")


if __name__ == "__main__":
    train()
