"""
Main Pipeline: Data-Driven Narrative Bridge Bandit Simulation

Runs the COMPLETE pipeline on ALL 8164 users:
  Step 1:  Load polarity scores (160KB file)
  Step 2:  Stream & parse 39.5M tweets (12GB file)
  Step 3:  Load & merge follower+friend networks (79M + 39M edges)
  Step 4:  Build user profiles (features for all 8164 users)
  Step 5:  Build tweet pool index (tweets bucketed by ideology)
  Step 6:  Calibrate heuristic engagement model (fallback)
  Step 7:  Train behavior model from .pkl data (NEW - data-driven)
  Step 8:  Run simulation for ALL users with 3 policies (LinUCB, Random, Greedy)
           using the trained behavior model for realistic engagement
  Step 9:  Compute metrics & generate visualizations
  Step 10: Save all outputs

The key improvement: Step 7 trains a neural network on REAL user retweet
sequences (scored_rt_sequences.pkl + user_ideology_states.pkl) to learn
how users actually respond to ideological drifts. This replaces the simple
Bernoulli engagement model with learned behavioral patterns.
"""
import os
import sys
import json
import time
import csv
import random
import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from preprocessing.data_loader import load_polarity, load_tweets_streaming, load_all_networks, build_tweet_data_from_pkl
from preprocessing.feature_builder import build_user_profiles, build_tweet_pool_index, build_engagement_model
from environment.simulation_env import BanditEnvironment
from bandits.linucb import LinUCB, RandomPolicy, GreedyPolicy
from evaluation.metrics import compute_all_metrics, format_metrics_report
from evaluation.visualize import plot_all, plot_bandit_comparison
from models.behavior_model import TransitionModel
from models.train_behavior import train_from_pkl


def load_behavior_model(model_path, device="cpu"):
    """Load a trained TransitionModel from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = TransitionModel()
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    print(f"  Loaded behavior model from {model_path}")
    print(f"  Trained at epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f}, "
          f"val_acc={checkpoint['val_acc']:.3f}")
    return model


def run_bandit_simulation(env, policy, user_ids, policy_name):
    """
    Run the bandit simulation for ALL users with a given policy.

    Behind the scenes for each user:
    1. env.reset(user_id) -> build initial context from REAL user data
    2. For T=20 steps:
       a. policy.select_action(context) -> choose arm (LinUCB uses UCB score)
       b. env.step_action(arm) -> find REAL tweet, predict engagement via trained model
       c. policy.update(context, arm, reward) -> update bandit parameters (LEARNING)
    3. Record session log and summary

    Total: 8164 users x 20 steps = 163,280 bandit rounds
    """
    all_session_summaries = []
    all_session_logs = []

    total_users = len(user_ids)
    start_time = time.time()

    print(f"\n  Running {policy_name} on {total_users} users ({total_users * config.STEPS_PER_USER} total rounds)...")

    for i, user_id in enumerate(user_ids):
        # Progress reporting
        if (i + 1) % 1000 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total_users - i - 1) / rate if rate > 0 else 0
            print(f"    User {i+1}/{total_users} ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining) | "
                  f"policy={policy_name}")

        # Reset environment for this user
        context = env.reset(user_id)

        # Run T steps
        for step in range(config.STEPS_PER_USER):
            # Bandit selects arm
            arm_index, ucb_scores = policy.select_action(context)

            # Environment executes: find tweet, predict engagement via model, compute reward
            next_context, reward, done, info = env.step_action(arm_index)

            # Bandit learns from outcome
            policy.update(context, arm_index, reward)

            context = next_context
            if done:
                break

        # Record results
        all_session_summaries.append(env.get_session_summary())
        all_session_logs.append(env.get_session_log())

    elapsed = time.time() - start_time
    print(f"  {policy_name} completed: {total_users} users in {elapsed:.1f}s "
          f"({total_users * config.STEPS_PER_USER / elapsed:.0f} rounds/sec)")

    return all_session_summaries, all_session_logs


def save_session_logs_csv(all_session_logs, filepath):
    """Save all step-level logs to CSV."""
    all_steps = [step for session in all_session_logs for step in session]
    if not all_steps:
        return

    fieldnames = list(all_steps[0].keys())
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_steps)

    print(f"  Saved {len(all_steps)} step records to {filepath}")


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    total_start = time.time()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    # Resolve device
    if config.BEHAVIOR_MODEL_DEVICE == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.BEHAVIOR_MODEL_DEVICE

    # =========================================================================
    # STEP 1: Load Polarity Scores
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Loading polarity scores")
    print("  What: Reading USER_POLARITY_BARBERA.txt (160KB)")
    print("  Behind: Parsing 8164 lines -> {user_id: ideology_score} dict")
    print("=" * 70)
    step_start = time.time()
    polarity = load_polarity()
    print(f"  Time: {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 2: Build Tweet Pool
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Building tweet pool")
    step_start = time.time()

    if os.path.exists(config.TWEETS_FILE):
        print("  Mode: Streaming USER_TWEETS.txt (12GB, 39.5M tweets)")
        print("  Behind: Line-by-line streaming, extracting tweet metadata,")
        print("          detecting retweets (RT @user pattern), building tweet pool")
        tweet_data = load_tweets_streaming(polarity)
    else:
        print("  Mode: Building from .pkl files (USER_TWEETS.txt not found)")
        print("  Source: scored_rt_sequences.pkl + ideology_map.pkl")
        print("  Behind: Each retweeted user becomes a tweet at their ideology position.")
        print("          Retweet counts from real data used as popularity proxy.")
        tweet_data = build_tweet_data_from_pkl(polarity)

    print(f"  Output: {len(tweet_data['tweet_pool'])} tweets in pool")
    print(f"  Time: {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 3: Load & Merge Networks
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Loading follower + friend networks")
    print("  What: Reading FULL_FOLLOWER_NETWORK.txt (1.6GB, 79.2M edges)")
    print("        Reading FULL_FRIEND_NETWORK.txt (757MB, 39.3M edges)")
    print("  Behind: Streaming edges, keeping only edges where BOTH users")
    print("          have polarity scores (8164 users). Merging into one graph.")
    print("  Output: adjacency dict {user_id: set of connected polarity users}")
    print("=" * 70)
    step_start = time.time()
    network_adj = load_all_networks(polarity)
    print(f"  Time: {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 4: Build User Profiles
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Building user profiles")
    print("  What: Computing feature vector for each of 8164 users")
    print("  Behind: For each user -> ideology, degree, neighbor stats,")
    print("          echo chamber score, tweet activity, retweet ratio")
    print("  Output: profiles dict with 8 features per user")
    print("=" * 70)
    step_start = time.time()
    user_profiles = build_user_profiles(polarity, network_adj, tweet_data)
    print(f"  Time: {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 5: Build Tweet Pool Index
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Indexing tweet pool by ideology")
    print("  What: Organizing ~200K tweets into ideology bins (width=0.1)")
    print("  Behind: Each tweet gets bin = round(ideology / 0.1) * 0.1")
    print("          Bins sorted by popularity for fast retrieval")
    print("  Output: {bin_center: [tweets]} for fast ideology-based lookup")
    print("=" * 70)
    step_start = time.time()
    tweet_pool_index = build_tweet_pool_index(tweet_data["tweet_pool"])
    print(f"  Time: {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 6: Calibrate Heuristic Engagement Model (fallback)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Calibrating heuristic engagement model (fallback)")
    print("  What: Learning P(engage) from actual retweet patterns")
    print("  Behind: Analyzing ideology distance between connected users,")
    print("          computing distance decay from homophily patterns")
    print("  Note: This is the FALLBACK model. Step 7 trains the data-driven model.")
    print("=" * 70)
    step_start = time.time()
    engagement_model = build_engagement_model(polarity, network_adj, tweet_data)
    print(f"  Time: {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 7: Train Behavior Model from .pkl Data (NEW)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Training data-driven behavior model from real sequences")
    print("  What: Training a neural network on REAL user retweet data")
    print("  Source: scored_rt_sequences.pkl (8158 users x ~1000 RTs each)")
    print("        + user_ideology_states.pkl (ideology trajectory per user)")
    print("  Behind: Extract (user_state, drift, engaged) transitions,")
    print("          train MLP to predict P(engage) + ideology shift")
    print("  Output: behavior_model.pt (replaces simple Bernoulli simulation)")
    print("=" * 70)
    step_start = time.time()

    behavior_model = None
    if os.path.exists(config.BEHAVIOR_MODEL_PATH):
        print(f"  Found existing model at {config.BEHAVIOR_MODEL_PATH}")
        print("  Loading pre-trained model...")
        behavior_model = load_behavior_model(config.BEHAVIOR_MODEL_PATH, device)
    elif os.path.exists(config.SCORED_RT_PATH) and os.path.exists(config.IDEOLOGY_STATES_PATH):
        print("  No pre-trained model found. Training from .pkl data...")
        trained = train_from_pkl(
            scored_rt_path=config.SCORED_RT_PATH,
            ideology_states_path=config.IDEOLOGY_STATES_PATH,
            ideology_map_path=config.IDEOLOGY_MAP_PATH,
            output_path=config.BEHAVIOR_MODEL_PATH,
            neg_ratio=config.BEHAVIOR_MODEL_NEG_RATIO,
            max_samples_per_user=config.BEHAVIOR_MODEL_MAX_SAMPLES_PER_USER,
            epochs=config.BEHAVIOR_MODEL_EPOCHS,
            batch_size=config.BEHAVIOR_MODEL_BATCH_SIZE,
            device=device,
        )
        # Reload from checkpoint (ensures we use the best epoch)
        behavior_model = load_behavior_model(config.BEHAVIOR_MODEL_PATH, device)
    else:
        print("  WARNING: .pkl data files not found. Falling back to heuristic model.")
        print(f"    Expected: {config.SCORED_RT_PATH}")
        print(f"    Expected: {config.IDEOLOGY_STATES_PATH}")

    model_mode = "MODEL-BASED (learned)" if behavior_model is not None else "HEURISTIC (fallback)"
    print(f"\n  Engagement simulation mode: {model_mode}")
    print(f"  Time: {time.time() - step_start:.1f}s")

    # =========================================================================
    # STEP 8: Run Bandit Simulation (ALL users, 3 policies)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Running bandit simulation")
    print(f"  What: Simulating recommendations for ALL {len(polarity)} users")
    print(f"  Engagement model: {model_mode}")
    print(f"  Behind: Each user gets {config.STEPS_PER_USER} recommendation rounds")
    print(f"  Total rounds: {len(polarity)} x {config.STEPS_PER_USER} = {len(polarity) * config.STEPS_PER_USER}")
    print(f"  Per round:")
    print(f"    1. Build context vector (10 features from REAL user data)")
    print(f"    2. Bandit selects arm (delta in {config.ARMS})")
    print(f"    3. Find matching tweet from REAL tweet pool")
    print(f"    4. Apply narrative bridge constraint (tau={config.TAU})")
    print(f"    5. Predict engagement via {'trained neural network' if behavior_model else 'heuristic model'}")
    print(f"    6. Compute reward & update bandit (LEARNING)")
    print(f"    7. Update user ideology based on {'model-predicted shift' if behavior_model else 'fixed rate'}")
    print(f"  Policies: LinUCB (proposed), Random (baseline), Greedy/Echo (baseline)")
    print("=" * 70)

    user_ids = list(polarity.keys())
    all_policy_metrics = {}

    # Create environment with behavior model
    env = BanditEnvironment(
        user_profiles, tweet_pool_index, engagement_model,
        network_adj, polarity,
        behavior_model=behavior_model,
        device=device,
    )

    # --- LinUCB (PROPOSED METHOD) ---
    print(f"\n  {'='*50}")
    print(f"  POLICY 1: LinUCB (Proposed Method)")
    print(f"  alpha={config.ALPHA}, d={config.FEATURE_DIM}, arms={config.ARMS}")
    print(f"  {'='*50}")
    linucb = LinUCB()
    linucb_summaries, linucb_logs = run_bandit_simulation(env, linucb, user_ids, "LinUCB")

    # --- Random Policy (BASELINE) ---
    print(f"\n  {'='*50}")
    print(f"  POLICY 2: Random (Baseline - uniform random arm selection)")
    print(f"  {'='*50}")
    random_policy = RandomPolicy()
    random_summaries, random_logs = run_bandit_simulation(env, random_policy, user_ids, "Random")

    # --- Greedy/Echo Chamber Policy (BASELINE) ---
    print(f"\n  {'='*50}")
    print(f"  POLICY 3: Greedy/Echo (Baseline - always delta=0, echo chamber)")
    print(f"  {'='*50}")
    greedy_policy = GreedyPolicy()
    greedy_summaries, greedy_logs = run_bandit_simulation(env, greedy_policy, user_ids, "Greedy(Echo)")

    # =========================================================================
    # STEP 9: Compute Metrics & Generate Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: Computing metrics and generating visualizations")
    print("  What: Analyzing simulation results for all 3 policies")
    print("  Behind: Computing engagement rates, diversity scores,")
    print("          polarization changes, learning curves, per-group analysis")
    print("  Output: metrics JSON + visualization plots per policy + comparison")
    print("=" * 70)

    # Compute metrics for each policy
    policies_data = {
        "LinUCB": (linucb_summaries, linucb_logs),
        "Random": (random_summaries, random_logs),
        "Greedy(Echo)": (greedy_summaries, greedy_logs),
    }

    for pname, (summaries, logs) in policies_data.items():
        print(f"\n  --- Metrics for {pname} ---")
        metrics = compute_all_metrics(summaries, logs)
        all_policy_metrics[pname] = metrics

        report = format_metrics_report(metrics)
        print(report)

        # Save per-policy outputs
        policy_dir = os.path.join(config.OUTPUT_DIR, pname.replace("(", "").replace(")", ""))
        os.makedirs(policy_dir, exist_ok=True)

        # Save metrics JSON
        metrics_serializable = json.loads(json.dumps(metrics, default=str))
        with open(os.path.join(policy_dir, "metrics.json"), "w") as f:
            json.dump(metrics_serializable, f, indent=2)

        # Save plots
        plot_all(metrics, summaries, logs, policy_dir, pname)

        # Save session logs CSV
        save_session_logs_csv(logs, os.path.join(policy_dir, "session_logs.csv"))

    # Comparison plot
    plot_bandit_comparison(all_policy_metrics, config.OUTPUT_DIR)

    # =========================================================================
    # STEP 10: Save Summary & Parameters
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 10: Saving final outputs")
    print("=" * 70)

    # Save LinUCB learned parameters
    linucb_params = {
        "A_matrices": [a.tolist() for a in linucb.A],
        "b_vectors": [b.tolist() for b in linucb.b],
        "arm_stats": linucb.get_arm_stats(),
    }
    with open(os.path.join(config.OUTPUT_DIR, "linucb_params.json"), "w") as f:
        json.dump(linucb_params, f, indent=2, default=str)
    print(f"  Saved LinUCB parameters to {config.OUTPUT_DIR}/linucb_params.json")

    # Save overall summary
    summary = {
        "total_users": len(polarity),
        "steps_per_user": config.STEPS_PER_USER,
        "total_rounds": len(polarity) * config.STEPS_PER_USER,
        "arms": config.ARMS,
        "alpha": config.ALPHA,
        "tau": config.TAU,
        "feature_dim": config.FEATURE_DIM,
        "tweet_pool_size": len(tweet_data["tweet_pool"]),
        "network_users": len(network_adj),
        "engagement_mode": model_mode,
        "behavior_model_path": config.BEHAVIOR_MODEL_PATH if behavior_model else None,
        "total_time_seconds": time.time() - total_start,
        "linucb_arm_stats": linucb.get_arm_stats(),
    }
    with open(os.path.join(config.OUTPUT_DIR, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved simulation summary to {config.OUTPUT_DIR}/simulation_summary.json")

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  Engagement model: {model_mode}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Users processed: {len(polarity)}")
    print(f"  Total rounds: {len(polarity) * config.STEPS_PER_USER * 3} (3 policies)")
    print(f"\n  Output files in: {config.OUTPUT_DIR}/")
    print(f"    ├── LinUCB/              (metrics, logs, plots)")
    print(f"    ├── Random/              (metrics, logs, plots)")
    print(f"    ├── GreedyEcho/          (metrics, logs, plots)")
    print(f"    ├── bandit_comparison.png")
    print(f"    ├── behavior_model.pt    (trained behavior model)")
    print(f"    ├── linucb_params.json")
    print(f"    └── simulation_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
