"""
Train the User Behavior Model from REAL retweet sequence data.

Data source: scored_rt_sequences.pkl + user_ideology_states.pkl
These contain actual user retweet histories with ideology scores.

Training data construction:
  For each user at each timestep t:
    - user_state = user_ideology_states[user][t]
    - content_ideo = scored_rt_sequences[user][t][1]
    - drift = content_ideo - user_state
    - This is a POSITIVE sample (user DID engage)
    - Create N negative samples by sampling drifts the user did NOT take

  Also extracts transition targets:
    - next_state = user_ideology_states[user][t+1]
    - ideology_shift = next_state - user_state

This trains the TransitionModel which predicts:
  1. P(engage | user_state, drift) — replaces simple Bernoulli
  2. predicted_ideology_shift — replaces fixed move_rate
"""

import os
import sys
import pickle
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.behavior_model import TransitionModel, build_features


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BehaviorDataset(Dataset):
    """
    Dataset of (features, engaged, ideology_shift) samples extracted
    from real user retweet sequences.
    """

    def __init__(self, features, labels, shifts):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.shifts = torch.tensor(shifts, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.shifts[idx]


# ---------------------------------------------------------------------------
# Data extraction from .pkl files
# ---------------------------------------------------------------------------

def extract_training_data(
    scored_rt_sequences,
    user_ideology_states,
    ideology_map,
    neg_ratio=3,
    history_window=10,
    max_samples_per_user=100,
    seed=42,
):
    """
    Extract training samples from real user sequences.

    For each user at timestep t:
      Positive sample:
        - features = build_features(state[t], drift_to_content, recent_states, progress)
        - label = 1 (engaged)
        - shift = state[t+1] - state[t] (actual ideology movement)

      Negative samples (neg_ratio per positive):
        - Sample a random drift from the empirical distribution
        - features with that alternative drift
        - label = 0 (did not engage with this drift)
        - shift = 0.0 (no movement since no engagement)

    Returns: features (N, 8), labels (N,), shifts (N,)
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Collect all observed drifts for negative sampling
    all_drifts = []
    for user_id, seq in scored_rt_sequences.items():
        if user_id not in user_ideology_states:
            continue
        states = user_ideology_states[user_id]
        for t in range(min(len(seq), len(states))):
            drift = seq[t][1] - states[t]
            all_drifts.append(drift)

    all_drifts = np.array(all_drifts)
    drift_mean = np.mean(all_drifts)
    drift_std = np.std(all_drifts)
    print(f"  Empirical drift distribution: mean={drift_mean:.4f}, std={drift_std:.4f}")
    print(f"  Total observed transitions: {len(all_drifts):,}")

    # Extract samples
    all_features = []
    all_labels = []
    all_shifts = []

    user_count = 0
    for user_id, seq in scored_rt_sequences.items():
        if user_id not in user_ideology_states:
            continue

        states = user_ideology_states[user_id]
        base_ideology = ideology_map.get(user_id, 0.0) if ideology_map else 0.0
        n = min(len(seq), len(states))

        if n < 2:
            continue

        # Sample timesteps if too many
        timesteps = list(range(n - 1))
        if len(timesteps) > max_samples_per_user:
            timesteps = sorted(rng.sample(timesteps, max_samples_per_user))

        for t in timesteps:
            user_state = states[t]
            content_ideo = seq[t][1]
            drift = content_ideo - user_state

            # Skip NaN/Inf
            if not (np.isfinite(user_state) and np.isfinite(content_ideo)):
                continue

            # Recent states for context
            start = max(0, t - history_window)
            recent = [s for s in states[start:t + 1] if np.isfinite(s)]
            if not recent:
                recent = [user_state]

            progress = t / max(n - 1, 1)

            # Actual ideology shift
            next_state = states[t + 1]
            actual_shift = next_state - user_state if np.isfinite(next_state) else 0.0

            # --- Positive sample ---
            feat = build_features(user_state, drift, recent, progress)
            all_features.append(feat)
            all_labels.append(1.0)
            all_shifts.append(actual_shift)

            # --- Negative samples ---
            for _ in range(neg_ratio):
                # Sample a different drift from empirical distribution
                neg_drift = np_rng.normal(drift_mean, drift_std)
                # Make sure it's actually different from the real drift
                while abs(neg_drift - drift) < 0.05:
                    neg_drift = np_rng.normal(drift_mean, drift_std)

                neg_feat = build_features(user_state, neg_drift, recent, progress)
                all_features.append(neg_feat)
                all_labels.append(0.0)
                all_shifts.append(0.0)  # no shift when not engaged

        user_count += 1
        if user_count % 1000 == 0:
            print(f"    Processed {user_count} users, {len(all_labels):,} samples so far")

    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)
    shifts = np.array(all_shifts, dtype=np.float32)

    print(f"  Extracted {len(labels):,} samples from {user_count} users")
    print(f"  Positive: {int(labels.sum()):,}, Negative: {int((1 - labels).sum()):,}")
    print(f"  Shift range: [{shifts.min():.3f}, {shifts.max():.3f}]")

    return features, labels, shifts


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    features,
    labels,
    shifts,
    output_path,
    epochs=30,
    batch_size=2048,
    lr=1e-3,
    weight_decay=1e-4,
    val_fraction=0.1,
    device="auto",
):
    """
    Train the TransitionModel and save checkpoint.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Training on {device}")

    # Train/val split (temporal: last val_fraction of data)
    n = len(labels)
    n_val = int(n * val_fraction)
    n_train = n - n_val

    # Shuffle before split (data is ordered by user)
    idx = np.random.permutation(n)
    features, labels, shifts = features[idx], labels[idx], shifts[idx]

    train_ds = BehaviorDataset(features[:n_train], labels[:n_train], shifts[:n_train])
    val_ds = BehaviorDataset(features[n_train:], labels[n_train:], shifts[n_train:])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {n_train:,} samples, Val: {n_val:,} samples")

    model = TransitionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    engage_loss_fn = nn.BCEWithLogitsLoss()
    shift_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_loss_total = 0.0
        train_engage_loss = 0.0
        train_shift_loss = 0.0
        train_steps = 0

        for feat_batch, label_batch, shift_batch in train_dl:
            feat_batch = feat_batch.to(device)
            label_batch = label_batch.to(device)
            shift_batch = shift_batch.to(device)

            optimizer.zero_grad()

            engage_logits, pred_shift = model(feat_batch)

            # Engagement loss (all samples)
            loss_engage = engage_loss_fn(engage_logits, label_batch)

            # Shift loss (only on positive samples where engagement happened)
            pos_mask = label_batch > 0.5
            if pos_mask.any():
                loss_shift = shift_loss_fn(pred_shift[pos_mask], shift_batch[pos_mask])
            else:
                loss_shift = torch.tensor(0.0, device=device)

            loss = loss_engage + 0.5 * loss_shift

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_total += loss.item()
            train_engage_loss += loss_engage.item()
            train_shift_loss += loss_shift.item()
            train_steps += 1

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        val_steps = 0

        with torch.no_grad():
            for feat_batch, label_batch, shift_batch in val_dl:
                feat_batch = feat_batch.to(device)
                label_batch = label_batch.to(device)
                shift_batch = shift_batch.to(device)

                engage_logits, pred_shift = model(feat_batch)
                loss_engage = engage_loss_fn(engage_logits, label_batch)

                pos_mask = label_batch > 0.5
                if pos_mask.any():
                    loss_shift = shift_loss_fn(pred_shift[pos_mask], shift_batch[pos_mask])
                else:
                    loss_shift = torch.tensor(0.0, device=device)

                val_loss_total += (loss_engage + 0.5 * loss_shift).item()

                preds = (torch.sigmoid(engage_logits) > 0.5).float()
                val_correct += (preds == label_batch).sum().item()
                val_total += len(label_batch)
                val_steps += 1

        avg_train = train_loss_total / max(train_steps, 1)
        avg_val = val_loss_total / max(val_steps, 1)
        val_acc = val_correct / max(val_total, 1)

        print(f"  Epoch {epoch:>2d}/{epochs} [{time.time()-t0:.1f}s] "
              f"train={avg_train:.4f} "
              f"(engage={train_engage_loss/max(train_steps,1):.4f}, "
              f"shift={train_shift_loss/max(train_steps,1):.4f}) "
              f"val={avg_val:.4f} acc={val_acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {
                "epoch": epoch,
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_loss": best_val_loss,
                "val_acc": val_acc,
                "feature_stats": {
                    "input_dim": TransitionModel.INPUT_DIM,
                },
            }

    # Save
    if best_state is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(best_state, output_path)
        print(f"\n  Saved behavior model to {output_path}")
        print(f"  Best val loss: {best_state['val_loss']:.4f}, "
              f"acc: {best_state['val_acc']:.3f} (epoch {best_state['epoch']})")

    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_from_pkl(
    scored_rt_path,
    ideology_states_path,
    ideology_map_path,
    output_path,
    neg_ratio=3,
    max_samples_per_user=100,
    epochs=30,
    batch_size=2048,
    device="auto",
):
    """
    Full pipeline: load .pkl files -> extract data -> train -> save model.
    """
    print("\n" + "=" * 70)
    print("TRAINING USER BEHAVIOR MODEL FROM REAL DATA")
    print("=" * 70)

    # Load .pkl files
    print("\n  Loading scored_rt_sequences...")
    with open(scored_rt_path, "rb") as f:
        scored_rt_sequences = pickle.load(f)
    print(f"  Loaded {len(scored_rt_sequences)} users")

    print("  Loading user_ideology_states...")
    with open(ideology_states_path, "rb") as f:
        user_ideology_states = pickle.load(f)
    print(f"  Loaded {len(user_ideology_states)} users")

    ideology_map = None
    if ideology_map_path and os.path.exists(ideology_map_path):
        print("  Loading ideology_map...")
        with open(ideology_map_path, "rb") as f:
            ideology_map = pickle.load(f)
        print(f"  Loaded {len(ideology_map)} users")

    # Extract training data
    print("\n  Extracting training data from sequences...")
    features, labels, shifts = extract_training_data(
        scored_rt_sequences,
        user_ideology_states,
        ideology_map,
        neg_ratio=neg_ratio,
        max_samples_per_user=max_samples_per_user,
    )

    # Train
    model = train_model(
        features, labels, shifts,
        output_path=output_path,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train behavior model from .pkl data")
    parser.add_argument("--scored-rt", default=None,
                        help="Path to scored_rt_sequences.pkl")
    parser.add_argument("--states", default=None,
                        help="Path to user_ideology_states.pkl")
    parser.add_argument("--ideo-map", default=None,
                        help="Path to ideology_map.pkl")
    parser.add_argument("--output", default=None,
                        help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--neg-ratio", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Default paths
    # models/train_behavior.py -> models/ -> bandit/ -> final_PROJECT/
    bandit_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_dir = os.path.dirname(bandit_dir)  # final_PROJECT/

    scored_rt_path = args.scored_rt or os.path.join(bandit_dir, "scored_rt_sequences.pkl")
    states_path = args.states or os.path.join(
        project_dir, "RecSys", "data", "processed", "user_ideology_states.pkl"
    )
    ideo_map_path = args.ideo_map or os.path.join(
        project_dir, "RecSys", "data", "processed", "ideology_map.pkl"
    )
    output_path = args.output or os.path.join(bandit_dir, "outputs", "behavior_model.pt")

    train_from_pkl(
        scored_rt_path=scored_rt_path,
        ideology_states_path=states_path,
        ideology_map_path=ideo_map_path,
        output_path=output_path,
        neg_ratio=args.neg_ratio,
        max_samples_per_user=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
