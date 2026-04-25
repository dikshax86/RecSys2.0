"""
Evaluation Metrics for the bandit simulation.

Computes:
1. Engagement metrics (rate, cumulative reward)
2. Diversity metrics (ideology shift achieved, echo chamber reduction)
3. Polarization metrics (before/after ideology distribution)
4. Bandit learning metrics (cumulative regret, arm selection over time)
"""
import numpy as np
from collections import defaultdict


def compute_all_metrics(all_session_summaries, all_session_logs):
    """
    Compute comprehensive metrics from simulation results.

    Input:
    - all_session_summaries: list of dicts from env.get_session_summary()
    - all_session_logs: list of lists of step dicts

    Output: dict of metric groups
    """
    metrics = {}

    # --- Engagement Metrics ---
    engagement_rates = [s["engagement_rate"] for s in all_session_summaries]
    total_rewards = [s["total_reward"] for s in all_session_summaries]

    metrics["engagement"] = {
        "avg_engagement_rate": float(np.mean(engagement_rates)),
        "std_engagement_rate": float(np.std(engagement_rates)),
        "median_engagement_rate": float(np.median(engagement_rates)),
        "avg_total_reward": float(np.mean(total_rewards)),
        "total_cumulative_reward": float(np.sum(total_rewards)),
    }

    # --- Diversity Metrics ---
    abs_shifts = [s["abs_ideology_shift"] for s in all_session_summaries]
    signed_shifts = [s["ideology_shift"] for s in all_session_summaries]

    metrics["diversity"] = {
        "avg_abs_ideology_shift": float(np.mean(abs_shifts)),
        "std_abs_ideology_shift": float(np.std(abs_shifts)),
        "median_abs_ideology_shift": float(np.median(abs_shifts)),
        "avg_signed_ideology_shift": float(np.mean(signed_shifts)),
        "users_shifted_toward_center": int(sum(
            1 for s in all_session_summaries
            if abs(s["final_ideology"]) < abs(s["initial_ideology"])
        )),
        "users_shifted_away_from_center": int(sum(
            1 for s in all_session_summaries
            if abs(s["final_ideology"]) > abs(s["initial_ideology"])
        )),
        "pct_shifted_toward_center": float(np.mean([
            1 if abs(s["final_ideology"]) < abs(s["initial_ideology"]) else 0
            for s in all_session_summaries
        ])),
    }

    # --- Polarization Metrics ---
    initial_ideologies = [s["initial_ideology"] for s in all_session_summaries]
    final_ideologies = [s["final_ideology"] for s in all_session_summaries]

    metrics["polarization"] = {
        "initial_mean_abs_ideology": float(np.mean(np.abs(initial_ideologies))),
        "final_mean_abs_ideology": float(np.mean(np.abs(final_ideologies))),
        "polarization_change": float(
            np.mean(np.abs(final_ideologies)) - np.mean(np.abs(initial_ideologies))
        ),
        "initial_std": float(np.std(initial_ideologies)),
        "final_std": float(np.std(final_ideologies)),
        "initial_polarization_index": float(np.mean(np.array(initial_ideologies) ** 2)),
        "final_polarization_index": float(np.mean(np.array(final_ideologies) ** 2)),
    }

    # --- Arm Selection Metrics ---
    all_steps = [step for session in all_session_logs for step in session]
    if all_steps:
        deltas = [step["delta_selected"] for step in all_steps]
        from collections import Counter
        delta_counts = Counter(deltas)
        total = len(deltas)
        metrics["arm_selection"] = {
            str(d): {"count": c, "fraction": c / total}
            for d, c in sorted(delta_counts.items())
        }

        # Engagement rate per arm
        arm_engagement = defaultdict(list)
        for step in all_steps:
            arm_engagement[step["delta_selected"]].append(step["engaged"])
        metrics["engagement_per_arm"] = {
            str(d): float(np.mean(engs))
            for d, engs in sorted(arm_engagement.items())
        }

    # --- Learning Curve (reward over time across all users) ---
    step_rewards = defaultdict(list)
    for session in all_session_logs:
        for step in session:
            step_rewards[step["step"]].append(step["reward"])
    metrics["learning_curve"] = {
        int(step): float(np.mean(rewards))
        for step, rewards in sorted(step_rewards.items())
    }

    # --- Global Learning Curve (avg reward per user, in processing order) ---
    # This shows whether the bandit improves as it processes more users
    # (LinUCB parameters accumulate across users sequentially)
    user_total_rewards = [s["total_reward"] / max(s["steps"], 1) for s in all_session_summaries]
    block_size = max(1, len(user_total_rewards) // 50)  # ~50 points on the curve
    global_lc = {}
    for i in range(0, len(user_total_rewards), block_size):
        block = user_total_rewards[i:i + block_size]
        if block:
            global_lc[i + len(block)] = float(np.mean(block))
    metrics["global_learning_curve"] = global_lc

    # --- By Ideology Group ---
    groups = {"liberal": [], "center": [], "conservative": []}
    for s in all_session_summaries:
        ideo = s["initial_ideology"]
        if ideo < -0.5:
            groups["liberal"].append(s)
        elif ideo > 0.5:
            groups["conservative"].append(s)
        else:
            groups["center"].append(s)

    metrics["by_group"] = {}
    for gname, gsummaries in groups.items():
        if gsummaries:
            metrics["by_group"][gname] = {
                "count": len(gsummaries),
                "avg_engagement": float(np.mean([s["engagement_rate"] for s in gsummaries])),
                "avg_abs_shift": float(np.mean([s["abs_ideology_shift"] for s in gsummaries])),
                "avg_reward": float(np.mean([s["total_reward"] for s in gsummaries])),
                "pct_toward_center": float(np.mean([
                    1 if abs(s["final_ideology"]) < abs(s["initial_ideology"]) else 0
                    for s in gsummaries
                ])),
            }

    return metrics


def format_metrics_report(metrics):
    """Format metrics as a readable string report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SIMULATION RESULTS")
    lines.append("=" * 70)

    lines.append("\n--- ENGAGEMENT ---")
    e = metrics["engagement"]
    lines.append(f"  Avg engagement rate:     {e['avg_engagement_rate']:.4f}")
    lines.append(f"  Avg total reward/user:   {e['avg_total_reward']:.4f}")
    lines.append(f"  Cumulative reward:       {e['total_cumulative_reward']:.2f}")

    lines.append("\n--- DIVERSITY ---")
    d = metrics["diversity"]
    lines.append(f"  Avg |ideology shift|:    {d['avg_abs_ideology_shift']:.4f}")
    lines.append(f"  Users toward center:     {d['users_shifted_toward_center']} ({d['pct_shifted_toward_center']:.1%})")
    lines.append(f"  Users away from center:  {d['users_shifted_away_from_center']}")

    lines.append("\n--- POLARIZATION ---")
    p = metrics["polarization"]
    lines.append(f"  Initial mean |ideology|: {p['initial_mean_abs_ideology']:.4f}")
    lines.append(f"  Final mean |ideology|:   {p['final_mean_abs_ideology']:.4f}")
    lines.append(f"  Change:                  {p['polarization_change']:.4f} "
                 f"({'reduced' if p['polarization_change'] < 0 else 'increased'})")
    lines.append(f"  Polarization index:      {p['initial_polarization_index']:.4f} → {p['final_polarization_index']:.4f}")

    lines.append("\n--- ARM SELECTION ---")
    if "arm_selection" in metrics:
        for arm, info in metrics["arm_selection"].items():
            eng = metrics.get("engagement_per_arm", {}).get(arm, 0)
            lines.append(f"  Δ={arm:>5s}: selected {info['fraction']:.1%} | engagement={eng:.3f}")

    lines.append("\n--- BY IDEOLOGY GROUP ---")
    for gname, ginfo in metrics.get("by_group", {}).items():
        lines.append(f"  {gname:>12s} (n={ginfo['count']:>4d}): "
                     f"engage={ginfo['avg_engagement']:.3f}, "
                     f"shift={ginfo['avg_abs_shift']:.3f}, "
                     f"toward_center={ginfo['pct_toward_center']:.1%}")

    lines.append("\n--- LEARNING CURVE (avg reward per step) ---")
    lc = metrics.get("learning_curve", {})
    for step in sorted(lc.keys()):
        bar = "█" * int(lc[step] * 20)
        lines.append(f"  Step {step:>2d}: {lc[step]:.4f} {bar}")

    lines.append("=" * 70)
    return "\n".join(lines)
