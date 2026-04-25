"""
Visualization module for bandit simulation results.

Generates plots:
1. Ideology trajectories (users moving across ideological space)
2. Cumulative reward / learning curve
3. Arm selection distribution
4. Engagement vs ideology shift
5. Before/after ideology distribution (polarization)
6. Bandit comparison (LinUCB vs Random vs Greedy)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

import sys
sys.path.insert(0, ".")
import config


def plot_all(metrics, all_session_summaries, all_session_logs, output_dir, policy_name="LinUCB"):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    plot_ideology_trajectories(all_session_summaries, output_dir, policy_name)
    plot_learning_curve(metrics, output_dir, policy_name)
    plot_arm_selection(metrics, output_dir, policy_name)
    plot_engagement_vs_shift(all_session_logs, output_dir, policy_name)
    plot_polarization(all_session_summaries, output_dir, policy_name)
    plot_summary_dashboard(metrics, all_session_summaries, output_dir, policy_name)

    print(f"  All plots saved to {output_dir}/")


def plot_ideology_trajectories(summaries, output_dir, policy_name):
    """Plot ideology trajectories for sampled users."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    groups = {"Liberal (ψ < -0.5)": [], "Center (-0.5 ≤ ψ ≤ 0.5)": [], "Conservative (ψ > 0.5)": []}
    for s in summaries:
        ideo = s["initial_ideology"]
        if ideo < -0.5:
            groups["Liberal (ψ < -0.5)"].append(s)
        elif ideo > 0.5:
            groups["Conservative (ψ > 0.5)"].append(s)
        else:
            groups["Center (-0.5 ≤ ψ ≤ 0.5)"].append(s)

    colors = {"Liberal (ψ < -0.5)": "blue", "Center (-0.5 ≤ ψ ≤ 0.5)": "green", "Conservative (ψ > 0.5)": "red"}

    for ax, (gname, gsummaries) in zip(axes, groups.items()):
        sampled = gsummaries[:50] if len(gsummaries) > 50 else gsummaries
        for s in sampled:
            traj = s["ideology_trajectory"]
            ax.plot(range(len(traj)), traj, alpha=0.3, color=colors[gname], linewidth=0.8)

        # Plot average trajectory
        if gsummaries:
            max_len = max(len(s["ideology_trajectory"]) for s in gsummaries)
            avg_traj = []
            for step in range(max_len):
                vals = [s["ideology_trajectory"][step] for s in gsummaries if len(s["ideology_trajectory"]) > step]
                avg_traj.append(np.mean(vals))
            ax.plot(range(len(avg_traj)), avg_traj, color="black", linewidth=2.5, label="Average")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"{gname} (n={len(gsummaries)})")
        ax.set_xlabel("Step")
        ax.set_ylabel("Ideology ψ")
        ax.set_ylim(config.IDEOLOGY_RANGE)
        ax.legend()

    fig.suptitle(f"User Ideology Trajectories - {policy_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ideology_trajectories.png"), dpi=150)
    plt.close()


def plot_learning_curve(metrics, output_dir, policy_name):
    """Plot learning curves: per-step and global cumulative."""
    lc = metrics.get("learning_curve", {})
    if not lc:
        return

    steps = sorted(lc.keys())
    rewards = [lc[s] for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: per-step average reward (existing)
    axes[0].bar(steps, rewards, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Step in Session")
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title(f"Per-Step Reward - {policy_name}")
    axes[0].axhline(y=np.mean(rewards), color="red", linestyle="--", label=f"Mean={np.mean(rewards):.3f}")
    axes[0].legend()

    # Right: global learning curve (cumulative avg across users in order)
    global_lc = metrics.get("global_learning_curve", {})
    if global_lc:
        user_blocks = sorted(global_lc.keys())
        avg_rewards = [global_lc[b] for b in user_blocks]
        axes[1].plot(user_blocks, avg_rewards, color="steelblue", linewidth=1.5)
        # Add trend line
        if len(user_blocks) > 2:
            z = np.polyfit(user_blocks, avg_rewards, 1)
            trend = np.poly1d(z)
            axes[1].plot(user_blocks, trend(user_blocks), "r--", linewidth=1.5,
                         label=f"Trend (slope={z[0]:.6f})")
        axes[1].set_xlabel("Users Processed (cumulative)")
        axes[1].set_ylabel("Avg Reward per User")
        axes[1].set_title(f"Global Learning Curve - {policy_name}\n(Shows bandit improving across users)")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Global learning curve\nnot available",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title(f"Global Learning Curve - {policy_name}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curve.png"), dpi=150)
    plt.close()


def plot_arm_selection(metrics, output_dir, policy_name):
    """Plot which arms (deltas) were selected and their engagement rates."""
    arm_sel = metrics.get("arm_selection", {})
    eng_per_arm = metrics.get("engagement_per_arm", {})
    if not arm_sel:
        return

    arms = sorted(arm_sel.keys(), key=float)
    fractions = [arm_sel[a]["fraction"] for a in arms]
    engagements = [eng_per_arm.get(a, 0) for a in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(arms, fractions, color="steelblue")
    ax1.set_xlabel("Ideology Shift Δ")
    ax1.set_ylabel("Selection Frequency")
    ax1.set_title(f"Arm Selection Distribution - {policy_name}")

    ax2.bar(arms, engagements, color="coral")
    ax2.set_xlabel("Ideology Shift Δ")
    ax2.set_ylabel("Engagement Rate")
    ax2.set_title(f"Engagement Rate per Arm - {policy_name}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "arm_selection.png"), dpi=150)
    plt.close()


def plot_engagement_vs_shift(all_session_logs, output_dir, policy_name):
    """Plot engagement rate as a function of ideology distance."""
    all_steps = [step for session in all_session_logs for step in session]
    if not all_steps:
        return

    # Bin by ideology distance
    bins = defaultdict(list)
    for step in all_steps:
        dist = abs(step["psi_tweet"] - step["psi_user"])
        bin_key = round(dist * 5) / 5  # bin to 0.2
        bins[bin_key].append(step["engaged"])

    distances = sorted(bins.keys())
    engagement_rates = [np.mean(bins[d]) for d in distances]
    counts = [len(bins[d]) for d in distances]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(distances, engagement_rates, width=0.15, color="steelblue", alpha=0.8, label="Engagement Rate")
    ax2.plot(distances, counts, color="red", marker="o", markersize=4, label="Sample Count")

    ax1.set_xlabel("|ψ_tweet - ψ_user| (Ideology Distance)")
    ax1.set_ylabel("Engagement Rate", color="steelblue")
    ax2.set_ylabel("Sample Count", color="red")
    ax1.set_title(f"Engagement vs Ideology Distance - {policy_name}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engagement_vs_shift.png"), dpi=150)
    plt.close()


def plot_polarization(summaries, output_dir, policy_name):
    """Plot before/after ideology distribution."""
    initial = [s["initial_ideology"] for s in summaries]
    final = [s["final_ideology"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(-3, 3, 40)

    ax1.hist(initial, bins=bins, color="steelblue", alpha=0.7, label="Before", density=True)
    ax1.hist(final, bins=bins, color="coral", alpha=0.7, label="After", density=True)
    ax1.set_xlabel("Ideology Score ψ")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Ideology Distribution Before/After - {policy_name}")
    ax1.axvline(x=0, color="gray", linestyle="--")
    ax1.legend()

    # Shift plot
    shifts = [s["ideology_shift"] for s in summaries]
    ax2.hist(shifts, bins=40, color="green", alpha=0.7)
    ax2.set_xlabel("Ideology Shift (final - initial)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distribution of Ideology Shifts - {policy_name}")
    ax2.axvline(x=0, color="gray", linestyle="--")
    ax2.axvline(x=np.mean(shifts), color="red", linestyle="--", label=f"Mean={np.mean(shifts):.3f}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "polarization.png"), dpi=150)
    plt.close()


def plot_summary_dashboard(metrics, summaries, output_dir, policy_name):
    """Create a single dashboard figure with key metrics."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Polarization change
    ax1 = fig.add_subplot(gs[0, 0])
    p = metrics["polarization"]
    labels = ["Before", "After"]
    vals = [p["initial_mean_abs_ideology"], p["final_mean_abs_ideology"]]
    colors = ["steelblue", "coral"]
    bars = ax1.bar(labels, vals, color=colors)
    ax1.set_title("Mean |Ideology|")
    ax1.set_ylabel("Mean |ψ|")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center")

    # 2. Engagement by group
    ax2 = fig.add_subplot(gs[0, 1])
    bg = metrics.get("by_group", {})
    if bg:
        gnames = list(bg.keys())
        eng_vals = [bg[g]["avg_engagement"] for g in gnames]
        gcolors = {"liberal": "blue", "center": "green", "conservative": "red"}
        ax2.bar(gnames, eng_vals, color=[gcolors.get(g, "gray") for g in gnames])
        ax2.set_title("Engagement by Group")
        ax2.set_ylabel("Avg Engagement Rate")

    # 3. Toward center by group
    ax3 = fig.add_subplot(gs[0, 2])
    if bg:
        tc_vals = [bg[g]["pct_toward_center"] for g in gnames]
        ax3.bar(gnames, tc_vals, color=[gcolors.get(g, "gray") for g in gnames])
        ax3.set_title("% Users Moved Toward Center")
        ax3.set_ylabel("Fraction")

    # 4. Learning curve
    ax4 = fig.add_subplot(gs[1, 0])
    lc = metrics.get("learning_curve", {})
    if lc:
        steps = sorted(lc.keys())
        ax4.plot(steps, [lc[s] for s in steps], marker="o", color="steelblue")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("Avg Reward")
        ax4.set_title("Learning Curve")

    # 5. Arm selection
    ax5 = fig.add_subplot(gs[1, 1])
    arm_sel = metrics.get("arm_selection", {})
    if arm_sel:
        arms = sorted(arm_sel.keys(), key=float)
        fracs = [arm_sel[a]["fraction"] for a in arms]
        ax5.bar(arms, fracs, color="steelblue")
        ax5.set_xlabel("Δ")
        ax5.set_title("Arm Selection")

    # 6. Key metrics text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    e = metrics["engagement"]
    d = metrics["diversity"]
    text = (
        f"Policy: {policy_name}\n\n"
        f"Engagement Rate: {e['avg_engagement_rate']:.4f}\n"
        f"Avg Reward/User: {e['avg_total_reward']:.4f}\n"
        f"Total Reward: {e['total_cumulative_reward']:.1f}\n\n"
        f"Avg |Ideology Shift|: {d['avg_abs_ideology_shift']:.4f}\n"
        f"Toward Center: {d['pct_shifted_toward_center']:.1%}\n\n"
        f"Polarization Change: {p['polarization_change']:.4f}\n"
        f"({('REDUCED' if p['polarization_change'] < 0 else 'INCREASED')})"
    )
    ax6.text(0.1, 0.5, text, fontsize=12, family="monospace", verticalalignment="center",
             bbox=dict(boxstyle="round", facecolor="lightyellow"))

    fig.suptitle(f"Bandit Simulation Dashboard - {policy_name}", fontsize=16, fontweight="bold")
    plt.savefig(os.path.join(output_dir, "dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_bandit_comparison(all_metrics, output_dir):
    """Compare multiple bandit policies side by side."""
    if len(all_metrics) < 2:
        return

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    policy_names = list(all_metrics.keys())
    x = range(len(policy_names))

    # 1. Engagement rate
    vals = [all_metrics[p]["engagement"]["avg_engagement_rate"] for p in policy_names]
    axes[0].bar(x, vals, color=["steelblue", "coral", "green"][:len(policy_names)])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(policy_names, rotation=15)
    axes[0].set_title("Avg Engagement Rate")

    # 2. Avg reward
    vals = [all_metrics[p]["engagement"]["avg_total_reward"] for p in policy_names]
    axes[1].bar(x, vals, color=["steelblue", "coral", "green"][:len(policy_names)])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(policy_names, rotation=15)
    axes[1].set_title("Avg Reward/User")

    # 3. Ideology shift
    vals = [all_metrics[p]["diversity"]["avg_abs_ideology_shift"] for p in policy_names]
    axes[2].bar(x, vals, color=["steelblue", "coral", "green"][:len(policy_names)])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(policy_names, rotation=15)
    axes[2].set_title("Avg |Ideology Shift|")

    # 4. % toward center
    vals = [all_metrics[p]["diversity"]["pct_shifted_toward_center"] for p in policy_names]
    axes[3].bar(x, vals, color=["steelblue", "coral", "green"][:len(policy_names)])
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(policy_names, rotation=15)
    axes[3].set_title("% Users Toward Center")

    fig.suptitle("Bandit Policy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bandit_comparison.png"), dpi=150)
    plt.close()
    print(f"  Comparison plot saved to {output_dir}/bandit_comparison.png")
