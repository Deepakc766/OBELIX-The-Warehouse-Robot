"""
OBELIX Training Visualization
===============================
Generates all plots from training_log.jsonl and eval_log.jsonl.

Usage:
    python plot_results.py                         # uses ./training_log.jsonl
    python plot_results.py --log_dir ./my_run      # custom directory
    python plot_results.py --save_dir ./plots       # save PNGs to directory

Works even if training was interrupted (reads whatever episodes exist).
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.4,
    "legend.facecolor": "#16213e",
    "legend.edgecolor": "#555",
    "font.size": 10,
})

COLORS = {
    "primary": "#e94560",
    "secondary": "#0f3460",
    "accent": "#53d8fb",
    "green": "#4ecca3",
    "orange": "#f39c12",
    "purple": "#9b59b6",
    "yellow": "#f1c40f",
    "pink": "#e91e90",
}

ACTION_NAMES = ["L45", "L22", "FW", "R22", "R45"]



def load_jsonl(path: str) -> list[dict]:
    """Load a JSON-Lines file, return list of dicts."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values, window=50):
    """Exponential moving average."""
    if len(values) == 0:
        return values
    smoothed = []
    ema = values[0]
    alpha = 2.0 / (window + 1)
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed



def plot_training_reward(data, ax):
    """1. Raw training reward curve."""
    eps = [d["episode"] for d in data]
    rewards = [d["env_reward"] for d in data]
    ax.plot(eps, rewards, color=COLORS["primary"], alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Env Reward")
    ax.set_title("1. Training Reward Curve (Raw)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_smoothed_reward(data, ax):
    """2. Smoothed training reward curve."""
    eps = [d["episode"] for d in data]
    rewards = [d["env_reward"] for d in data]
    sm = smooth(rewards, window=50)
    ax.plot(eps, rewards, color=COLORS["primary"], alpha=0.15, linewidth=0.5, label="Raw")
    ax.plot(eps, sm, color=COLORS["accent"], linewidth=2, label="EMA-50")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Env Reward")
    ax.set_title("2. Smoothed Training Reward (EMA-50)")
    ax.legend()


def plot_eval_performance(eval_data, ax):
    """3. Evaluation performance across difficulty levels."""
    if not eval_data:
        ax.text(0.5, 0.5, "No eval data", ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_title("3. Evaluation Performance")
        return
    eps = [d["episode"] for d in eval_data]
    means = [d["eval_mean"] for d in eval_data]
    stds = [d["eval_std"] for d in eval_data]
    ax.plot(eps, means, color=COLORS["green"], linewidth=2, marker="o", markersize=4, label="Mean")
    ax.fill_between(eps,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     color=COLORS["green"], alpha=0.2, label="±1 std")
    ax.axhline(-500, color=COLORS["yellow"], linestyle="--", alpha=0.7, label="Target (-500)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Eval Reward")
    ax.set_title("3. Evaluation Performance (w/ walls)")
    ax.legend(fontsize=8)


def plot_reward_distribution(data, ax):
    """4. Reward distribution histogram."""
    rewards = [d["env_reward"] for d in data]
    ax.hist(rewards, bins=50, color=COLORS["accent"], alpha=0.7, edgecolor="#333")
    ax.axvline(np.mean(rewards), color=COLORS["primary"], linestyle="--", label=f"Mean: {np.mean(rewards):.0f}")
    ax.axvline(np.median(rewards), color=COLORS["green"], linestyle="--", label=f"Median: {np.median(rewards):.0f}")
    ax.set_xlabel("Env Reward")
    ax.set_ylabel("Count")
    ax.set_title("4. Reward Distribution")
    ax.legend(fontsize=8)


def plot_loss_curve(data, ax):
    """5. Loss curve."""
    eps = [d["episode"] for d in data if d.get("avg_loss") is not None]
    losses = [d["avg_loss"] for d in data if d.get("avg_loss") is not None]
    if not losses:
        ax.text(0.5, 0.5, "No loss data", ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_title("5. Loss Curve")
        return
    sm = smooth(losses, window=30)
    ax.plot(eps, losses, color=COLORS["orange"], alpha=0.2, linewidth=0.5)
    ax.plot(eps, sm, color=COLORS["orange"], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Loss")
    ax.set_title("5. Loss Curve")


def plot_success_rate(data, ax):
    """6. Success rate curve (rolling window)."""
    window = max(20, len(data) // 50)
    boundary_hits = [1 if d["pushed_to_boundary"] else 0 for d in data]
    box_finds = [1 if d["found_box"] else 0 for d in data]
    eps = [d["episode"] for d in data]

    def rolling_mean(arr, w):
        out = []
        for i in range(len(arr)):
            start = max(0, i - w + 1)
            out.append(np.mean(arr[start:i + 1]))
        return out

    ax.plot(eps, rolling_mean(box_finds, window), color=COLORS["accent"], linewidth=2, label="Box Found")
    ax.plot(eps, rolling_mean(boundary_hits, window), color=COLORS["green"], linewidth=2, label="Pushed to Boundary")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"6. Success Rate (rolling {window})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)


def plot_difficulty_learning(data, ax):
    """7. Difficulty-wise learning curve."""
    diffs = sorted(set(d["difficulty"] for d in data))
    for diff in diffs:
        sub = [d for d in data if d["difficulty"] == diff]
        if not sub:
            continue
        eps = [d["episode"] for d in sub]
        rewards = [d["env_reward"] for d in sub]
        sm = smooth(rewards, window=30)
        label = {0: "Static", 2: "Blinking", 3: "Moving"}.get(diff, f"Diff {diff}")
        ax.plot(eps, sm, linewidth=2, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Env Reward (smoothed)")
    ax.set_title("7. Difficulty-wise Learning Curve")
    ax.legend(fontsize=8)


def plot_action_distribution(data, ax):
    """9. Action distribution across training."""
    total_counts = np.zeros(5)
    for d in data:
        counts = d.get("action_counts")
        if counts:
            total_counts += np.array(counts)
    if total_counts.sum() == 0:
        ax.text(0.5, 0.5, "No action data", ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.set_title("9. Action Distribution")
        return
    pct = total_counts / total_counts.sum() * 100
    colors = [COLORS["purple"], COLORS["accent"], COLORS["green"], COLORS["accent"], COLORS["purple"]]
    bars = ax.bar(ACTION_NAMES, pct, color=colors, edgecolor="#333")
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{p:.1f}%", ha="center", va="bottom", fontsize=9, color="#eee")
    ax.set_ylabel("Percentage")
    ax.set_title("9. Action Distribution")


def plot_epsilon_decay(data, ax):
    """12. Epsilon decay over episodes."""
    eps = [d["episode"] for d in data]
    epsilons = [d["epsilon"] for d in data]
    ax.plot(eps, epsilons, color=COLORS["yellow"], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("12. Epsilon Decay")
    ax.set_ylim(-0.05, 1.05)


def plot_reward_breakdown(data, ax):
    """13. Reward breakdown: env reward vs shaped reward vs explore bonus."""
    eps = [d["episode"] for d in data]
    env_r = smooth([d["env_reward"] for d in data], 50)
    shaped_r = smooth([d["shaped_reward"] for d in data], 50)
    explore_b = smooth([d.get("explore_bonus", 0) for d in data], 50)
    # Compute PBRS component = shaped - env - explore
    pbrs = [s - e - x for s, e, x in zip(shaped_r, env_r, explore_b)]

    ax.plot(eps, env_r, color=COLORS["primary"], linewidth=1.5, label="Env Reward")
    ax.plot(eps, pbrs, color=COLORS["green"], linewidth=1.5, label="PBRS Shaping")
    ax.plot(eps, explore_b, color=COLORS["accent"], linewidth=1.5, label="Explore Bonus")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("13. Reward Breakdown")
    ax.legend(fontsize=8)


def plot_exploration_efficiency(data, ax):
    """Exploration Efficiency = 1 / steps_to_first_detection."""
    eps = [d["episode"] for d in data]
    eff = []
    for d in data:
        det = d.get("steps_to_detection", -1)
        if det > 0:
            eff.append(1.0 / det)
        else:
            eff.append(0.0)
    sm = smooth(eff, window=30)
    ax.plot(eps, eff, color=COLORS["accent"], alpha=0.2, linewidth=0.5)
    ax.plot(eps, sm, color=COLORS["accent"], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("1 / Steps to Detection")
    ax.set_title("Exploration Efficiency")


def plot_generalization(data, ax):
    """14. Generalization: compare wall vs no-wall performance."""
    wall_data = [d for d in data if d.get("wall_obstacles")]
    nowall_data = [d for d in data if not d.get("wall_obstacles")]

    if wall_data:
        eps_w = [d["episode"] for d in wall_data]
        r_w = smooth([d["env_reward"] for d in wall_data], 30)
        ax.plot(eps_w, r_w, color=COLORS["primary"], linewidth=2, label="With Walls")
    if nowall_data:
        eps_n = [d["episode"] for d in nowall_data]
        r_n = smooth([d["env_reward"] for d in nowall_data], 30)
        ax.plot(eps_n, r_n, color=COLORS["green"], linewidth=2, label="No Walls")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Env Reward (smoothed)")
    ax.set_title("14. Generalization (Wall vs No-Wall)")
    ax.legend(fontsize=8)


def plot_steps_per_episode(data, ax):
    """8. Steps per episode (proxy for trajectory length)."""
    eps = [d["episode"] for d in data]
    steps = [d["steps"] for d in data]
    sm = smooth(steps, 30)
    ax.plot(eps, steps, color=COLORS["purple"], alpha=0.2, linewidth=0.5)
    ax.plot(eps, sm, color=COLORS["purple"], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("8. Steps Per Episode (Trajectory Length)")


def plot_phase_rewards(data, ax):
    """11. Ablation/Phase-wise box plot."""
    phases = sorted(set(d["phase"] for d in data))
    phase_rewards = {p: [d["env_reward"] for d in data if d["phase"] == p] for p in phases}

    box_data = [phase_rewards[p] for p in phases]
    bp = ax.boxplot(box_data, tick_labels=phases, patch_artist=True, showfliers=False)
    colors_list = [COLORS["green"], COLORS["accent"], COLORS["orange"], COLORS["primary"]]
    for patch, color in zip(bp["boxes"], colors_list[:len(phases)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for element in ["whiskers", "caps", "medians"]:
        plt.setp(bp[element], color="#eee")
    ax.set_ylabel("Env Reward")
    ax.set_title("11. Phase-wise Reward Distribution")


def main():
    parser = argparse.ArgumentParser(description="Generate OBELIX training plots")
    parser.add_argument("--log_dir", type=str, default=".",
                        help="Directory containing training_log.jsonl & eval_log.jsonl")
    parser.add_argument("--save_dir", type=str, default="plots",
                        help="Where to save PNG plots")
    args = parser.parse_args()

    # Load data
    train_path = os.path.join(args.log_dir, "training_log.jsonl")
    eval_path = os.path.join(args.log_dir, "eval_log.jsonl")

    data = load_jsonl(train_path)
    eval_data = load_jsonl(eval_path)

    if not data:
        print(f"ERROR: No training data found at {train_path}")
        print("Run training first:  python train_d3qn.py")
        sys.exit(1)

    print(f"Loaded {len(data)} training episodes from {train_path}")
    if eval_data:
        print(f"Loaded {len(eval_data)} eval checkpoints from {eval_path}")

    os.makedirs(args.save_dir, exist_ok=True)

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(f"OBELIX D3QN Training Dashboard  ({len(data)} episodes)",
                 fontsize=18, fontweight="bold", color=COLORS["accent"])
    gs = gridspec.GridSpec(4, 4, hspace=0.35, wspace=0.3,
                           left=0.05, right=0.97, top=0.93, bottom=0.04)

    # Row 1: Rewards
    plot_training_reward(data, fig.add_subplot(gs[0, 0]))
    plot_smoothed_reward(data, fig.add_subplot(gs[0, 1]))
    plot_eval_performance(eval_data, fig.add_subplot(gs[0, 2]))
    plot_reward_distribution(data, fig.add_subplot(gs[0, 3]))

    # Row 2: Learning
    plot_loss_curve(data, fig.add_subplot(gs[1, 0]))
    plot_success_rate(data, fig.add_subplot(gs[1, 1]))
    plot_difficulty_learning(data, fig.add_subplot(gs[1, 2]))
    plot_steps_per_episode(data, fig.add_subplot(gs[1, 3]))

    # Row 3: Actions & Exploration
    plot_action_distribution(data, fig.add_subplot(gs[2, 0]))
    plot_epsilon_decay(data, fig.add_subplot(gs[2, 1]))
    plot_reward_breakdown(data, fig.add_subplot(gs[2, 2]))
    plot_exploration_efficiency(data, fig.add_subplot(gs[2, 3]))

    # Row 4: Generalization & Phase Comparison
    plot_generalization(data, fig.add_subplot(gs[3, 0]))
    plot_phase_rewards(data, fig.add_subplot(gs[3, 1]))

    # Extra: Phase transition lines on smoothed reward
    ax_phases = fig.add_subplot(gs[3, 2:])
    plot_smoothed_reward(data, ax_phases)
    # Add vertical lines for phase transitions
    phase_changes = []
    prev_phase = None
    for d in data:
        if d["phase"] != prev_phase:
            if prev_phase is not None:
                phase_changes.append((d["episode"], d["phase"]))
            prev_phase = d["phase"]
    for ep, phase in phase_changes:
        ax_phases.axvline(ep, color=COLORS["yellow"], linestyle=":", alpha=0.7)
        ax_phases.text(ep, ax_phases.get_ylim()[1] * 0.95, f" {phase}",
                      fontsize=7, color=COLORS["yellow"], rotation=90, va="top")
    ax_phases.set_title("Reward with Phase Transitions")

    dashboard_path = os.path.join(args.save_dir, "dashboard.png")
    fig.savefig(dashboard_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n Dashboard saved → {dashboard_path}")
    individual_plots = [
        ("01_training_reward", plot_training_reward),
        ("02_smoothed_reward", plot_smoothed_reward),
        ("03_eval_performance", lambda d, ax: plot_eval_performance(eval_data, ax)),
        ("04_reward_distribution", plot_reward_distribution),
        ("05_loss_curve", plot_loss_curve),
        ("06_success_rate", plot_success_rate),
        ("07_difficulty_learning", plot_difficulty_learning),
        ("08_steps_per_episode", plot_steps_per_episode),
        ("09_action_distribution", plot_action_distribution),
        ("11_phase_rewards", plot_phase_rewards),
        ("12_epsilon_decay", plot_epsilon_decay),
        ("13_reward_breakdown", plot_reward_breakdown),
        ("14_generalization", plot_generalization),
        ("15_exploration_efficiency", plot_exploration_efficiency),
    ]

    for name, plot_fn in individual_plots:
        fig_i, ax_i = plt.subplots(figsize=(8, 5))
        try:
            plot_fn(data, ax_i)
        except Exception as e:
            ax_i.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                     transform=ax_i.transAxes, color="red")
        path_i = os.path.join(args.save_dir, f"{name}.png")
        fig_i.tight_layout()
        fig_i.savefig(path_i, dpi=150, facecolor=fig_i.get_facecolor())
        plt.close(fig_i)

    print(f"{len(individual_plots)} individual plots saved → {args.save_dir}/")

    print(f"\n{'=' * 60}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 60}")
    rewards = [d["env_reward"] for d in data]
    print(f"  Episodes:        {len(data)}")
    print(f"  Mean reward:     {np.mean(rewards):.1f}")
    print(f"  Best reward:     {max(rewards):.1f}")
    print(f"  Worst reward:    {min(rewards):.1f}")
    print(f"  Last 100 mean:   {np.mean(rewards[-100:]):.1f}")
    box_rate = np.mean([d["found_box"] for d in data]) * 100
    push_rate = np.mean([d["pushed_to_boundary"] for d in data]) * 100
    print(f"  Box found rate:  {box_rate:.1f}%")
    print(f"  Push rate:       {push_rate:.1f}%")
    det_steps = [d["steps_to_detection"] for d in data if d.get("steps_to_detection", -1) > 0]
    if det_steps:
        print(f"  Avg detection:   {np.mean(det_steps):.0f} steps")
    if eval_data:
        print(f"  Best eval mean:  {max(d['eval_mean'] for d in eval_data):.1f}")



if __name__ == "__main__":
    main()
