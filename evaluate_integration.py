"""
evaluate_integration.py
-----------------------
End-to-end evaluation of the Bandit ↔ Recommender integrated system.

The key insight: the recommender must be TRAINED with the bandit's per-user deltas
so the contrastive loss learns to rank items within bandit-sized windows.

Pipeline:
  1. Load bandit policy → assign per-user δ from LinUCB
  2. Build train/val/test datasets with bandit deltas (not fixed δ=0.2)
  3. Train the recommender from scratch with bandit deltas
  4. Evaluate on test set
  5. Compare against: fixed-δ=0.2 (pre-trained) and random-δ baselines

Usage:
    python evaluate_integration.py --device cuda --epochs 30 --batch_size 2048
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

# ── Path setup (RecSys imports need to come from RecSys/, not bandit/) ────────
PROJECT_DIR = Path(__file__).resolve().parent
RECSYS_DIR  = PROJECT_DIR / "RecSys"
BANDIT_DIR  = PROJECT_DIR / "bandit"

sys.path.insert(0, str(RECSYS_DIR))

from config         import cfg, cConfig
from data.dataset   import (
    IdeologySeqDataset, build_item_vocab, NegativeSampler,
    CollateWithNegatives, ItemCatalog, PAD_IDX,
)
from evaluate        import run_evaluation
from loss.ideology_loss import IdeologyLoss
from models.recommender import IdeologyRecommender
from train           import resolve_device, load_graph_tensors, build_model


# =============================================================================
#  1.  BANDIT POLICY LOADER
# =============================================================================

def load_linucb_policy(params_path: str) -> dict:
    """Load trained LinUCB parameters (A matrices + b vectors) from JSON."""
    with open(params_path, "r") as f:
        raw = json.load(f)
    return {
        "A": [np.array(a, dtype=np.float64) for a in raw["A_matrices"]],
        "b": [np.array(b, dtype=np.float64) for b in raw["b_vectors"]],
    }


def linucb_select_arm(policy: dict, context: np.ndarray, alpha: float) -> int:
    """Run LinUCB inference: return index of the arm with highest UCB score."""
    x  = context.astype(np.float64)
    d  = policy["A"][0].shape[0]
    n  = len(policy["A"])
    best, best_score = 0, -np.inf

    for a in range(n):
        A_inv   = np.linalg.solve(policy["A"][a], np.eye(d))
        theta   = A_inv @ policy["b"][a]
        exploit = theta @ x
        explore = np.sqrt(x @ A_inv @ x)
        ucb     = exploit + alpha * explore
        if ucb > best_score:
            best, best_score = a, ucb
    return best


# =============================================================================
#  2.  BUILD USER CONTEXT (mirrors simulation_env._build_context exactly)
# =============================================================================

def build_user_context(
    ideology:       float,
    degree_norm:    float = 0.5,
    avg_nbr_ideo:   float = 0.0,
    nbr_ideo_std:   float = 0.5,
    echo_score:     float = 0.5,
    retweet_ratio:  float = 0.3,
    session_prog:   float = 0.5,
    recent_engage:  float = 0.5,
    avg_acc_delta:  float = 0.15,
) -> np.ndarray:
    """
    10-dim context vector matching BanditEnvironment._build_context():
      [0] ideology / 3.0
      [1] degree_norm
      [2] avg_neighbor_ideology / 3.0
      [3] neighbor_ideology_std
      [4] echo_chamber_score
      [5] retweet_ratio
      [6] session_progress
      [7] recent_engagement_rate
      [8] avg_abs_delta_accepted
      [9] 1.0  (bias)
    """
    return np.array([
        ideology / 3.0,
        degree_norm,
        avg_nbr_ideo / 3.0,
        nbr_ideo_std,
        echo_score,
        retweet_ratio,
        session_prog,
        recent_engage,
        avg_acc_delta,
        1.0,
    ], dtype=np.float64)


# =============================================================================
#  3.  ASSIGN BANDIT DELTAS TO EVERY USER
# =============================================================================

def assign_bandit_deltas(
    user_ideology_states: dict,
    scored_rt_sequences:  dict,
    user2idx:             dict,
    linucb_policy:        dict,
    arms:                 list,
    alpha:                float,
    graph_data            = None,
) -> dict:
    """
    For each user, build a context vector and let the bandit choose an arm.
    Returns {user_id: |delta|} — absolute delta used as ideology-window width.
    """
    degree_map = {}
    if graph_data is not None:
        ei = graph_data.edge_index.numpy()
        cnt = Counter(ei[0].tolist()) + Counter(ei[1].tolist())
        max_d = max(cnt.values()) if cnt else 1
        for uid, idx in user2idx.items():
            degree_map[uid] = cnt.get(idx, 0) / max_d

    user_deltas = {}

    for uid, states in user_ideology_states.items():
        if uid not in scored_rt_sequences:
            continue
        if not states:
            continue

        cur = states[-1] if np.isfinite(states[-1]) else 0.0
        finite = [s for s in states if np.isfinite(s)]
        mean_i = float(np.mean(finite)) if finite else 0.0
        std_i  = float(np.std(finite))  if finite else 0.5

        ctx = build_user_context(
            ideology     = cur,
            degree_norm  = degree_map.get(uid, 0.5),
            avg_nbr_ideo = mean_i,
            nbr_ideo_std = min(std_i, 1.0),
            echo_score   = min(abs(cur) / 3.0, 1.0),
        )

        arm_idx = linucb_select_arm(linucb_policy, ctx, alpha)
        user_deltas[uid] = abs(arms[arm_idx])

    return user_deltas


# =============================================================================
#  4.  BUILD DATASETS WITH BANDIT DELTAS (train + val + test)
# =============================================================================

def build_all_dataloaders(
    processed_dir:  str,
    user_deltas:    dict,
    fallback_delta: float,
    config:         cConfig,
):
    """
    Build train/val/test DataLoaders with per-user bandit-assigned deltas.
    The contrastive loss will learn to rank within bandit-sized windows.

    Returns: (train_dl, val_dl, test_dl, item_catalog)
    """
    d = Path(processed_dir)
    with open(d / "scored_rt_sequences.pkl",  "rb") as f: scored   = pickle.load(f)
    with open(d / "user_ideology_states.pkl", "rb") as f: states   = pickle.load(f)
    with open(d / "user2idx-3.pkl",           "rb") as f: user2idx = pickle.load(f)

    item2idx, idx2item, item_ideo_dict = build_item_vocab(scored, config.data.min_item_freq)

    datasets = {}
    for split in ["train", "val", "test"]:
        ds = IdeologySeqDataset(
            scored_rt_sequences  = scored,
            user_ideology_states = states,
            item2idx             = item2idx,
            item_ideo            = item_ideo_dict,
            user2idx             = user2idx,
            split                = split,
            max_seq_len          = config.data.max_seq_len,
            val_holdout          = config.data.val_holdout,
            test_holdout         = config.data.test_holdout,
            delta                = fallback_delta,
            train_target_stride  = config.data.train_target_stride,
            max_train_targets_per_user = config.data.max_train_targets_per_user,
            train_recent_window  = config.data.train_recent_window,
        )

        # Override per-sample deltas with bandit-assigned values
        overridden = 0
        for sample in ds.samples:
            uid = sample["user_id"]
            if uid in user_deltas:
                sample["delta"] = max(user_deltas[uid], 0.05)
                overridden += 1
        print(f"  [{split}] Overrode {overridden}/{len(ds.samples)} deltas with bandit values")
        datasets[split] = ds

    # Item catalog
    num_items = (max(item2idx.values()) + 1) if item2idx else 1
    item_ideo_arr = np.zeros(num_items, dtype=np.float32)
    for idx, sc in item_ideo_dict.items():
        if 0 <= idx < num_items:
            item_ideo_arr[idx] = float(sc) if np.isfinite(sc) else 0.0
    item_ideo_arr = np.nan_to_num(item_ideo_arr, nan=0.0, posinf=0.0, neginf=0.0)

    item_catalog = ItemCatalog(item2idx=item2idx, idx2item=idx2item, item_ideo=item_ideo_arr)
    neg_sampler = NegativeSampler(item_ideo_dict, strategy="hard", band=config.loss.hard_neg_band)
    collate = CollateWithNegatives(neg_sampler, item_ideo_arr, num_negatives=config.loss.num_negatives)

    common = dict(batch_size=config.train.batch_size, num_workers=0, collate_fn=collate)
    train_dl = DataLoader(datasets["train"], shuffle=True,  **common)
    val_dl   = DataLoader(datasets["val"],   shuffle=False, **common)
    test_dl  = DataLoader(datasets["test"],  shuffle=False, **common)

    return train_dl, val_dl, test_dl, item_catalog


def build_test_dataloader(
    processed_dir:  str,
    user_deltas:    dict,
    fallback_delta: float,
    config:         cConfig,
):
    """Build ONLY the test DataLoader (for evaluating pre-trained model)."""
    d = Path(processed_dir)
    with open(d / "scored_rt_sequences.pkl",  "rb") as f: scored   = pickle.load(f)
    with open(d / "user_ideology_states.pkl", "rb") as f: states   = pickle.load(f)
    with open(d / "user2idx-3.pkl",           "rb") as f: user2idx = pickle.load(f)

    item2idx, idx2item, item_ideo_dict = build_item_vocab(scored, config.data.min_item_freq)

    test_ds = IdeologySeqDataset(
        scored_rt_sequences  = scored,
        user_ideology_states = states,
        item2idx             = item2idx,
        item_ideo            = item_ideo_dict,
        user2idx             = user2idx,
        split                = "test",
        max_seq_len          = config.data.max_seq_len,
        val_holdout          = config.data.val_holdout,
        test_holdout         = config.data.test_holdout,
        delta                = fallback_delta,
    )

    overridden = 0
    for sample in test_ds.samples:
        uid = sample["user_id"]
        if uid in user_deltas:
            sample["delta"] = max(user_deltas[uid], 0.05)
            overridden += 1
    print(f"  [test] Overrode {overridden}/{len(test_ds.samples)} deltas with bandit values")

    num_items = (max(item2idx.values()) + 1) if item2idx else 1
    item_ideo_arr = np.zeros(num_items, dtype=np.float32)
    for idx, sc in item_ideo_dict.items():
        if 0 <= idx < num_items:
            item_ideo_arr[idx] = float(sc) if np.isfinite(sc) else 0.0
    item_ideo_arr = np.nan_to_num(item_ideo_arr, nan=0.0, posinf=0.0, neginf=0.0)

    item_catalog = ItemCatalog(item2idx=item2idx, idx2item=idx2item, item_ideo=item_ideo_arr)
    neg_sampler = NegativeSampler(item_ideo_dict, strategy="hard", band=0.5)
    collate = CollateWithNegatives(neg_sampler, item_ideo_arr, num_negatives=1)

    test_dl = DataLoader(test_ds, batch_size=config.train.batch_size, shuffle=False,
                         num_workers=0, collate_fn=collate)
    return test_dl, item_catalog


# =============================================================================
#  5.  TRAIN RECOMMENDER WITH BANDIT DELTAS
# =============================================================================

def train_with_bandit_deltas(
    config:          cConfig,
    train_dl:        DataLoader,
    val_dl:          DataLoader,
    item_catalog:    ItemCatalog,
    graph_x:         torch.Tensor,
    graph_edge_index: torch.Tensor,
    device:          str,
    label:           str = "bandit-integrated",
):
    """
    Train the recommender from scratch using bandit-assigned deltas.
    Returns the trained model.
    """
    model = build_model(item_catalog.num_items, config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    warmup_steps = config.train.warmup_steps
    total_steps  = config.train.num_epochs * len(train_dl)
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    loss_fn = IdeologyLoss(
        alpha_bpr=config.loss.alpha_bpr,
        alpha_contrastive=config.loss.alpha_contrastive,
    )

    checkpoint_dir = Path(config.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"best_model_{label}.pt"

    best_hit10 = -1.0
    no_improve = 0

    with torch.no_grad():
        all_graph_embs = model.graph_encoder(graph_x, graph_edge_index)

    print(f"\n  Training [{label}] for {config.train.num_epochs} epochs...")

    for epoch in range(1, config.train.num_epochs + 1):
        t0 = time.time()
        model.train()

        running_total = 0.0
        running_bpr = 0.0
        running_contr = 0.0
        steps = 0
        nan_steps = 0

        for batch in train_dl:
            optimizer.zero_grad()

            user_idx        = batch["user_idx"].to(device)
            seq_items       = batch["history_items"].to(device)
            seq_ideo        = batch["history_states"].to(device)
            pos_items       = batch["target_item"].to(device)
            pos_ideo        = batch["target_ideo"].to(device)
            neg_items       = batch["neg_item_idx"].to(device)
            neg_ideo        = batch["neg_ideo"].to(device)
            aligned_items   = batch["ideo_aligned_item_idx"].to(device)
            aligned_ideo    = batch["ideo_aligned_ideo"].to(device)
            outside_items   = batch["ideo_outside_item_idx"].to(device)
            outside_ideo    = batch["ideo_outside_ideo"].to(device)

            u_final, pos_scores, neg_scores = model(
                graph_x=graph_x, graph_edge_index=graph_edge_index,
                user_graph_idx=user_idx,
                seq_item_ids=seq_items, seq_ideo_scores=seq_ideo,
                pos_item_ids=pos_items, pos_ideo_scores=pos_ideo,
                neg_item_ids=neg_items, neg_ideo_scores=neg_ideo,
                all_graph_embs=all_graph_embs,
            )

            aligned_embs   = model.encode_items(aligned_items, aligned_ideo)
            outside_embs   = model.encode_items(outside_items, outside_ideo)
            aligned_scores = model.score(u_final, aligned_embs)
            outside_scores = model.score(u_final, outside_embs)

            total, parts = loss_fn(pos_scores, neg_scores, aligned_scores, outside_scores)

            if not torch.isfinite(total):
                nan_steps += 1
                optimizer.zero_grad()
                steps += 1
                continue

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

            has_nan_grad = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()
                nan_steps += 1
                steps += 1
                continue

            optimizer.step()
            scheduler.step()

            running_total += total.item()
            running_bpr   += parts["loss_bpr"]
            running_contr += parts["loss_contrastive"]
            steps += 1

        good_steps = steps - nan_steps
        train_loss = running_total / max(good_steps, 1)

        # Validation
        val_metrics = run_evaluation(
            model=model, dataloader=val_dl, item_catalog=item_catalog,
            graph_x=graph_x, graph_edge_index=graph_edge_index,
            device=device, k_values=config.eval.k_values, split="val",
        )
        hit10 = val_metrics.get("hit@10", 0.0)

        nan_note = f" nan={nan_steps}" if nan_steps else ""
        print(
            f"    Epoch {epoch:2d}/{config.train.num_epochs} "
            f"[{time.time()-t0:.0f}s] "
            f"loss={train_loss:.4f} "
            f"bpr={running_bpr/max(good_steps,1):.4f} "
            f"contr={running_contr/max(good_steps,1):.4f} "
            f"val_hit@10={hit10:.4f}{nan_note}"
        )

        if hit10 > best_hit10:
            best_hit10 = hit10
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_hit10": best_hit10,
                "config": config,
                "label": label,
            }, best_path)
        else:
            no_improve += 1

        if no_improve >= config.train.early_stopping:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Load best checkpoint
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"    Loaded best checkpoint from epoch {ckpt['epoch']} (hit@10={ckpt['best_hit10']:.4f})")

    model.eval()
    return model


# =============================================================================
#  6.  COMPUTE TEST LOSS
# =============================================================================

@torch.no_grad()
def compute_test_loss(
    model:             IdeologyRecommender,
    dataloader:        DataLoader,
    loss_fn:           IdeologyLoss,
    graph_x:           torch.Tensor,
    graph_edge_index:  torch.Tensor,
    all_graph_embs:    torch.Tensor,
    device:            str,
) -> dict:
    """Average loss breakdown on the given dataloader."""
    model.eval()

    sum_total = sum_bpr = sum_contr = 0.0
    n_samples = 0

    for batch in dataloader:
        user_idx        = batch["user_idx"].to(device)
        seq_items       = batch["history_items"].to(device)
        seq_ideo        = batch["history_states"].to(device)
        pos_items       = batch["target_item"].to(device)
        pos_ideo        = batch["target_ideo"].to(device)
        neg_items       = batch["neg_item_idx"].to(device)
        neg_ideo        = batch["neg_ideo"].to(device)
        aligned_items   = batch["ideo_aligned_item_idx"].to(device)
        aligned_ideo    = batch["ideo_aligned_ideo"].to(device)
        outside_items   = batch["ideo_outside_item_idx"].to(device)
        outside_ideo    = batch["ideo_outside_ideo"].to(device)

        u_final, pos_scores, neg_scores = model(
            graph_x=graph_x, graph_edge_index=graph_edge_index,
            user_graph_idx=user_idx,
            seq_item_ids=seq_items, seq_ideo_scores=seq_ideo,
            pos_item_ids=pos_items, pos_ideo_scores=pos_ideo,
            neg_item_ids=neg_items, neg_ideo_scores=neg_ideo,
            all_graph_embs=all_graph_embs,
        )

        aligned_embs   = model.encode_items(aligned_items, aligned_ideo)
        outside_embs   = model.encode_items(outside_items, outside_ideo)
        aligned_scores = model.score(u_final, aligned_embs)
        outside_scores = model.score(u_final, outside_embs)

        total, parts = loss_fn(pos_scores, neg_scores, aligned_scores, outside_scores)

        if torch.isfinite(total):
            bs = user_idx.size(0)
            sum_total += parts["loss_total"] * bs
            sum_bpr   += parts["loss_bpr"]   * bs
            sum_contr += parts["loss_contrastive"] * bs
            n_samples += bs

    n = max(n_samples, 1)
    return {
        "test_loss_total":       sum_total / n,
        "test_loss_bpr":         sum_bpr   / n,
        "test_loss_contrastive": sum_contr / n,
        "num_samples":           n_samples,
    }


# =============================================================================
#  7.  LOAD PRE-TRAINED RECOMMENDER (for baseline comparison)
# =============================================================================

def load_recommender(checkpoint_path: str, config: cConfig, device: str):
    """Load the pre-trained IdeologyRecommender from a checkpoint."""
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    model_state = ckpt["model_state"]
    num_items = model_state["tweet_encoder.item_embed.weight"].shape[0]
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}, "
          f"best_hit@10={ckpt.get('best_hit10', '?'):.4f}, num_items={num_items}")

    model = IdeologyRecommender(
        num_items         = num_items,
        num_graph_nodes   = 0,
        embed_dim         = config.tweet_enc.output_dim,
        graph_node_feat   = config.graph.node_feat_dim,
        hidden_dim        = config.graph.hidden_dim,
        num_sage_layers   = config.graph.num_layers,
        num_sasrec_layers = config.sasrec.num_layers,
        num_heads         = config.sasrec.num_heads,
        max_seq_len       = config.data.max_seq_len,
        dropout           = config.graph.dropout,
    ).to(device)

    model.load_state_dict(model_state)
    model.eval()
    return model


# =============================================================================
#  8.  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bandit + RecSys Integration")
    parser.add_argument("--device",        type=str, default="auto")
    parser.add_argument("--batch_size",    type=int, default=2048)
    parser.add_argument("--epochs",        type=int, default=30,
                        help="Epochs to train the bandit-integrated recommender")
    parser.add_argument("--checkpoint",    type=str, default=None,
                        help="Path to pre-trained recommender checkpoint (fixed-delta baseline)")
    parser.add_argument("--bandit_params", type=str, default=None,
                        help="Path to linucb_params.json")
    # parser.add_argument("--data_dir",      type=str, default=None,
    #                     help="Path to processed data directory (with .pkl files)")
    parser.add_argument("--skip_retrain",  action="store_true",
                        help="Skip retraining; only evaluate existing checkpoint with bandit deltas")
    args = parser.parse_args()

    config = copy.deepcopy(cfg)
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.epochs
    device = resolve_device(args.device if args.device != "auto" else config.train.device)
    print(f"Device: {device}")

    ckpt_path = args.checkpoint or str(Path(config.paths.checkpoint_dir) / "best_model")
    bandit_path = args.bandit_params or str(BANDIT_DIR / "outputs" / "linucb_params.json")
    # processed_dir = args.data_dir if args.data_dir else config.paths.processed_dir
    processed_dir = "/home/diksha/workspace/Term2/Projects/RS/final_PROJECT/data/processed"
    # Override config so downstream functions also use the correct path
    config.paths.processed_dir = processed_dir

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1: Load bandit policy
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 1  Load trained LinUCB policy")
    print("=" * 65)
    linucb_policy = load_linucb_policy(bandit_path)
    n_arms = len(linucb_policy["A"])
    feat_d = linucb_policy["A"][0].shape[0]
    print(f"  Arms: {n_arms}, feature_dim: {feat_d}")

    sim_summary_path = BANDIT_DIR / "outputs" / "simulation_summary.json"
    with open(sim_summary_path) as f:
        sim_summary = json.load(f)
    arms  = sim_summary["arms"]
    alpha = sim_summary["alpha"]
    print(f"  Arms values: {arms}")
    print(f"  Alpha: {alpha}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2: Assign bandit deltas to all users
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 2  Assign bandit deltas to every user")
    print("=" * 65)

    with open(Path(processed_dir) / "scored_rt_sequences.pkl",  "rb") as f: scored   = pickle.load(f)
    with open(Path(processed_dir) / "user_ideology_states.pkl", "rb") as f: states   = pickle.load(f)
    with open(Path(processed_dir) / "user2idx-3.pkl",           "rb") as f: user2idx = pickle.load(f)

    graph_data = None
    gp = Path(processed_dir) / "graph_data-3.pkl"
    if gp.exists():
        with open(gp, "rb") as f:
            graph_data = pickle.load(f)

    user_deltas = assign_bandit_deltas(
        user_ideology_states = states,
        scored_rt_sequences  = scored,
        user2idx             = user2idx,
        linucb_policy        = linucb_policy,
        arms                 = arms,
        alpha                = alpha,
        graph_data           = graph_data,
    )

    delta_counts = Counter(user_deltas.values())
    print(f"\n  Bandit delta distribution ({len(user_deltas)} users):")
    print(f"  {'|delta|':>8}  {'count':>7}  {'fraction':>10}")
    for d_val in sorted(delta_counts):
        frac = delta_counts[d_val] / len(user_deltas)
        print(f"  {d_val:8.3f}  {delta_counts[d_val]:7d}  {frac:10.4f}")
    print(f"  Mean |delta|: {np.mean(list(user_deltas.values())):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3: Load graph tensors
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 3  Load graph tensors")
    print("=" * 65)
    graph_x, graph_edge_index = load_graph_tensors(processed_dir, device)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4: Train recommender with bandit deltas (the key step)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 4  Train recommender WITH bandit-assigned deltas")
    print("=" * 65)

    if args.skip_retrain:
        print("  --skip_retrain: loading pre-trained model instead")
        model_bandit = load_recommender(ckpt_path, config, device)
    else:
        print("  Building train/val/test with bandit deltas...")
        train_dl_b, val_dl_b, test_dl_b, item_catalog_b = build_all_dataloaders(
            processed_dir, user_deltas, fallback_delta=0.15, config=config,
        )

        model_bandit = train_with_bandit_deltas(
            config=config,
            train_dl=train_dl_b, val_dl=val_dl_b,
            item_catalog=item_catalog_b,
            graph_x=graph_x, graph_edge_index=graph_edge_index,
            device=device, label="bandit-integrated",
        )

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 5: Load pre-trained fixed-delta model (baseline)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 5  Load pre-trained fixed-δ=0.2 model (baseline)")
    print("=" * 65)
    model_fixed = load_recommender(ckpt_path, config, device)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 6: Evaluate both models on their respective test sets
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 6  Evaluate on test set")
    print("=" * 65)

    loss_fn = IdeologyLoss(
        alpha_bpr=config.loss.alpha_bpr,
        alpha_contrastive=config.loss.alpha_contrastive,
    )

    all_results = {}

    # ── A) Bandit-integrated model + bandit deltas ───────────────────────
    print("\n  [A] Bandit-Integrated (retrained with bandit δ):")
    if not args.skip_retrain:
        dl_a = test_dl_b
        cat_a = item_catalog_b
    else:
        dl_a, cat_a = build_test_dataloader(processed_dir, user_deltas, 0.15, config)

    with torch.no_grad():
        graph_embs_a = model_bandit.graph_encoder(graph_x, graph_edge_index)

    loss_a = compute_test_loss(model_bandit, dl_a, loss_fn, graph_x, graph_edge_index, graph_embs_a, device)
    eval_a = run_evaluation(
        model=model_bandit, dataloader=dl_a, item_catalog=cat_a,
        graph_x=graph_x, graph_edge_index=graph_edge_index,
        device=device, k_values=config.eval.k_values, split="test",
    )
    all_results["Bandit-Integrated"] = {**loss_a, **eval_a}
    print(f"    loss={loss_a['test_loss_total']:.4f}  hit@10={eval_a.get('hit@10',0):.4f}")

    # ── B) Fixed-delta model + fixed delta ───────────────────────────────
    print("\n  [B] Fixed-δ=0.2 (pre-trained baseline):")
    dl_b, cat_b = build_test_dataloader(processed_dir, {}, 0.2, config)

    with torch.no_grad():
        graph_embs_b = model_fixed.graph_encoder(graph_x, graph_edge_index)

    loss_b = compute_test_loss(model_fixed, dl_b, loss_fn, graph_x, graph_edge_index, graph_embs_b, device)
    eval_b = run_evaluation(
        model=model_fixed, dataloader=dl_b, item_catalog=cat_b,
        graph_x=graph_x, graph_edge_index=graph_edge_index,
        device=device, k_values=config.eval.k_values, split="test",
    )
    all_results["Fixed-δ=0.2"] = {**loss_b, **eval_b}
    print(f"    loss={loss_b['test_loss_total']:.4f}  hit@10={eval_b.get('hit@10',0):.4f}")

    # ── C) Fixed model + random deltas (control) ────────────────────────
    print("\n  [C] Random-δ (random arm selection, pre-trained model):")
    rng = np.random.RandomState(42)
    random_deltas = {uid: abs(arms[rng.randint(n_arms)]) for uid in user_deltas}
    dl_c, cat_c = build_test_dataloader(processed_dir, random_deltas, 0.15, config)

    loss_c = compute_test_loss(model_fixed, dl_c, loss_fn, graph_x, graph_edge_index, graph_embs_b, device)
    eval_c = run_evaluation(
        model=model_fixed, dataloader=dl_c, item_catalog=cat_c,
        graph_x=graph_x, graph_edge_index=graph_edge_index,
        device=device, k_values=config.eval.k_values, split="test",
    )
    all_results["Random-δ"] = {**loss_c, **eval_c}
    print(f"    loss={loss_c['test_loss_total']:.4f}  hit@10={eval_c.get('hit@10',0):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 7: Print comparison table
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("INTEGRATION EVALUATION RESULTS")
    print("=" * 65)

    names = list(all_results.keys())
    header = f"{'Metric':<28}" + "".join(f" {n:>18}" for n in names)
    print(f"\n{header}")
    print("─" * len(header))

    for key in ["test_loss_total", "test_loss_bpr", "test_loss_contrastive"]:
        row = f"{key:<28}"
        for n in names:
            row += f" {all_results[n].get(key, 0):>18.4f}"
        print(row)

    print()
    metric_keys = sorted(set(
        k for r in all_results.values() for k in r
        if k not in ("test_loss_total", "test_loss_bpr", "test_loss_contrastive", "num_samples")
    ))
    for key in metric_keys:
        row = f"{key:<28}"
        for n in names:
            row += f" {all_results[n].get(key, 0):>18.4f}"
        print(row)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 8: Save results + summary
    # ══════════════════════════════════════════════════════════════════════
    eval_dir = Path(config.paths.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "integration_results.json"

    def jsonable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: jsonable(v) for k, v in obj.items()}
        return obj

    with open(out_path, "w") as f:
        json.dump(jsonable(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    b = all_results["Bandit-Integrated"]
    f_ = all_results["Fixed-δ=0.2"]
    r = all_results["Random-δ"]

    print(f"\n  {'':30} {'Bandit':>12} {'Fixed':>12} {'Random':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")
    for key, label in [
        ("test_loss_total",  "Test Loss (total)"),
        ("hit@10",           "Hit@10"),
        ("ndcg@10",          "NDCG@10"),
        ("ideo_drift@10",    "Ideo Drift@10"),
        ("ideo_in_window",   "In-Window Frac"),
        ("direction_acc",    "Direction Acc"),
    ]:
        bv = b.get(key, 0)
        fv = f_.get(key, 0)
        rv = r.get(key, 0)
        print(f"  {label:30} {bv:>12.4f} {fv:>12.4f} {rv:>12.4f}")

    delta_loss = f_["test_loss_total"] - b["test_loss_total"]
    delta_hit  = b.get("hit@10", 0) - f_.get("hit@10", 0)
    delta_wind = b.get("ideo_in_window", 0) - f_.get("ideo_in_window", 0)

    print(f"\n  Bandit-Integrated vs Fixed baseline:")
    sign = "+" if delta_loss >= 0 else ""
    print(f"    Test loss reduction: {sign}{delta_loss:.4f}  "
          f"({'BETTER' if delta_loss > 0 else 'worse'})")
    sign = "+" if delta_hit >= 0 else ""
    print(f"    Hit@10 improvement:  {sign}{delta_hit:.4f}  "
          f"({'BETTER' if delta_hit > 0 else 'worse'})")
    sign = "+" if delta_wind >= 0 else ""
    print(f"    In-window fraction:  {sign}{delta_wind:.4f}  "
          f"({'BETTER' if delta_wind > 0 else 'worse'})")


if __name__ == "__main__":
    main()
