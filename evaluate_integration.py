"""
evaluate_integration.py
-----------------------
End-to-end evaluation of the Bandit + Recommender integrated system.

Trains ONE model with bandit-assigned deltas, then evaluates with 3 delta strategies:
  A) Bandit-integrated: per-user δ from trained LinUCB (model trained with these)
  B) Fixed δ=0.2: same model, evaluated with fixed window
  C) Random δ: same model, random arm selection

Usage (Kaggle):
    python evaluate_integration.py \
        --device cuda \
        --data_dir /kaggle/input/bandit/data_processed \
        --bandit_dir /kaggle/working/RecSys2.0/bandit/outputs \
        --epochs 3 \
        --batch_size 1024

Usage (local):
    python evaluate_integration.py --device cpu --epochs 3
"""

import argparse
import copy
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
RECSYS_DIR  = PROJECT_DIR / "RecSys"

# Add RecSys to path (handles both local and Kaggle flat layouts)
if RECSYS_DIR.exists():
    sys.path.insert(0, str(RECSYS_DIR))
else:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg
from data.dataset import (
    IdeologySeqDataset, build_item_vocab, NegativeSampler,
    CollateWithNegatives, ItemCatalog, PAD_IDX,
)
from evaluate import run_evaluation
from loss.ideology_loss import IdeologyLoss
from models.recommender import IdeologyRecommender
from train import resolve_device, load_graph_tensors, build_model


# =============================================================================
#  BANDIT POLICY
# =============================================================================

def load_bandit_policy(bandit_dir: str):
    """Load LinUCB params and simulation summary."""
    bd = Path(bandit_dir)
    with open(bd / "linucb_params.json") as f:
        raw = json.load(f)
    with open(bd / "simulation_summary.json") as f:
        summary = json.load(f)

    A_matrices = [np.array(a, dtype=np.float64) for a in raw["A_matrices"]]
    b_vectors  = [np.array(b, dtype=np.float64) for b in raw["b_vectors"]]
    arms  = summary["arms"]
    alpha = summary["alpha"]

    return A_matrices, b_vectors, arms, alpha, summary


def linucb_select(A_matrices, b_vectors, context, alpha):
    """Select best arm using LinUCB UCB formula."""
    x = context.astype(np.float64)
    d = A_matrices[0].shape[0]
    best, best_s = 0, -np.inf
    for a in range(len(A_matrices)):
        A_inv = np.linalg.solve(A_matrices[a], np.eye(d))
        theta = A_inv @ b_vectors[a]
        ucb = float(theta @ x) + alpha * float(np.sqrt(x @ A_inv @ x))
        if ucb > best_s:
            best, best_s = a, ucb
    return best


def assign_bandit_deltas(states, scored, A_matrices, b_vectors, arms, alpha):
    """Assign per-user |delta| using LinUCB policy."""
    user_deltas = {}
    for uid, user_states in states.items():
        if uid not in scored or not user_states:
            continue
        cur = user_states[-1] if np.isfinite(user_states[-1]) else 0.0
        finite = [s for s in user_states if np.isfinite(s)]
        mean_i = float(np.mean(finite)) if finite else 0.0
        std_i = float(np.std(finite)) if finite else 0.5

        ctx = np.array([
            cur / 3.0,
            0.5,
            mean_i / 3.0,
            min(std_i, 1.0),
            min(abs(cur) / 3.0, 1.0),
            0.3,
            0.5,
            0.5,
            0.15,
            1.0,
        ], dtype=np.float64)

        arm_idx = linucb_select(A_matrices, b_vectors, ctx, alpha)
        user_deltas[uid] = abs(arms[arm_idx])

    return user_deltas


# =============================================================================
#  DATASET BUILDER
# =============================================================================

def build_dataset_with_deltas(scored, states, item2idx, item_ideo_dict, user2idx,
                              config, split, deltas_map, fallback_delta):
    """Build a dataset and override per-sample deltas."""
    ds = IdeologySeqDataset(
        scored_rt_sequences=scored,
        user_ideology_states=states,
        item2idx=item2idx,
        item_ideo=item_ideo_dict,
        user2idx=user2idx,
        split=split,
        max_seq_len=config.data.max_seq_len,
        val_holdout=config.data.val_holdout,
        test_holdout=config.data.test_holdout,
        delta=fallback_delta,
        train_target_stride=config.data.train_target_stride,
        max_train_targets_per_user=config.data.max_train_targets_per_user,
        train_recent_window=config.data.train_recent_window,
    )
    overridden = 0
    for sample in ds.samples:
        if sample["user_id"] in deltas_map:
            sample["delta"] = max(deltas_map[sample["user_id"]], 0.05)
            overridden += 1
    print(f"  [{split}] {len(ds.samples):,} samples, overrode {overridden:,} deltas")
    return ds


# =============================================================================
#  TRAINING
# =============================================================================

def train_model(model, train_dl, val_dl, item_catalog, graph_x, graph_edge_index,
                all_graph_embs, config, device, num_epochs):
    """Train the recommender and return best model state."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    loss_fn = IdeologyLoss(
        alpha_bpr=config.loss.alpha_bpr,
        alpha_contrastive=config.loss.alpha_contrastive,
    )

    warmup_steps = config.train.warmup_steps
    total_steps = num_epochs * len(train_dl)
    warmup_sched = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

    best_hit10 = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = running_bpr = running_contr = 0.0
        steps = nan_steps = 0

        for batch in train_dl:
            optimizer.zero_grad()

            u_final, pos_scores, neg_scores = model(
                graph_x=graph_x, graph_edge_index=graph_edge_index,
                user_graph_idx=batch["user_idx"].to(device),
                seq_item_ids=batch["history_items"].to(device),
                seq_ideo_scores=batch["history_states"].to(device),
                pos_item_ids=batch["target_item"].to(device),
                pos_ideo_scores=batch["target_ideo"].to(device),
                neg_item_ids=batch["neg_item_idx"].to(device),
                neg_ideo_scores=batch["neg_ideo"].to(device),
                all_graph_embs=all_graph_embs,
            )

            aligned_embs = model.encode_items(
                batch["ideo_aligned_item_idx"].to(device),
                batch["ideo_aligned_ideo"].to(device),
            )
            outside_embs = model.encode_items(
                batch["ideo_outside_item_idx"].to(device),
                batch["ideo_outside_ideo"].to(device),
            )
            aligned_scores = model.score(u_final, aligned_embs)
            outside_scores = model.score(u_final, outside_embs)

            total, parts = loss_fn(pos_scores, neg_scores, aligned_scores, outside_scores)

            if not torch.isfinite(total):
                optimizer.zero_grad()
                nan_steps += 1
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

            running_loss += total.item()
            running_bpr += parts["loss_bpr"]
            running_contr += parts["loss_contrastive"]
            steps += 1

        good = steps - nan_steps
        train_loss = running_loss / max(good, 1)

        val_metrics = run_evaluation(
            model=model, dataloader=val_dl, item_catalog=item_catalog,
            graph_x=graph_x, graph_edge_index=graph_edge_index,
            device=device, k_values=config.eval.k_values, split="val",
        )
        hit10 = val_metrics.get("hit@10", 0.0)

        nan_note = f" nan={nan_steps}" if nan_steps else ""
        print(
            f"    Epoch {epoch}/{num_epochs} [{time.time()-t0:.0f}s] "
            f"loss={train_loss:.4f} bpr={running_bpr/max(good,1):.4f} "
            f"contr={running_contr/max(good,1):.4f} val_hit@10={hit10:.4f}{nan_note}"
        )

        if hit10 > best_hit10:
            best_hit10 = hit10
            best_state = copy.deepcopy(model.state_dict())

    if best_state:
        model.load_state_dict(best_state)
        print(f"    Best model: hit@10={best_hit10:.4f}")

    model.eval()
    return model, best_hit10


# =============================================================================
#  EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_setting(model, scored, states, item2idx, item_ideo_dict, user2idx,
                     item_catalog, collate_fn, config, graph_x, graph_edge_index,
                     all_graph_embs, device, deltas_map, fallback_delta, label):
    """Build test set with given deltas, compute loss + retrieval metrics."""
    model.eval()

    ds = IdeologySeqDataset(
        scored_rt_sequences=scored, user_ideology_states=states,
        item2idx=item2idx, item_ideo=item_ideo_dict, user2idx=user2idx,
        split="test", max_seq_len=config.data.max_seq_len,
        val_holdout=config.data.val_holdout, test_holdout=config.data.test_holdout,
        delta=fallback_delta,
    )
    for sample in ds.samples:
        if sample["user_id"] in deltas_map:
            sample["delta"] = max(deltas_map[sample["user_id"]], 0.05)

    dl = DataLoader(ds, batch_size=config.train.batch_size, shuffle=False,
                    num_workers=0, collate_fn=collate_fn)

    loss_fn = IdeologyLoss(alpha_bpr=config.loss.alpha_bpr, alpha_contrastive=config.loss.alpha_contrastive)

    sum_total = sum_bpr = sum_contr = 0.0
    n = 0
    for batch in dl:
        u_final, pos_scores, neg_scores = model(
            graph_x=graph_x, graph_edge_index=graph_edge_index,
            user_graph_idx=batch["user_idx"].to(device),
            seq_item_ids=batch["history_items"].to(device),
            seq_ideo_scores=batch["history_states"].to(device),
            pos_item_ids=batch["target_item"].to(device),
            pos_ideo_scores=batch["target_ideo"].to(device),
            neg_item_ids=batch["neg_item_idx"].to(device),
            neg_ideo_scores=batch["neg_ideo"].to(device),
            all_graph_embs=all_graph_embs,
        )
        aligned_embs = model.encode_items(batch["ideo_aligned_item_idx"].to(device), batch["ideo_aligned_ideo"].to(device))
        outside_embs = model.encode_items(batch["ideo_outside_item_idx"].to(device), batch["ideo_outside_ideo"].to(device))
        aligned_scores = model.score(u_final, aligned_embs)
        outside_scores = model.score(u_final, outside_embs)
        total, parts = loss_fn(pos_scores, neg_scores, aligned_scores, outside_scores)
        if torch.isfinite(total):
            bs = batch["user_idx"].size(0)
            sum_total += parts["loss_total"] * bs
            sum_bpr += parts["loss_bpr"] * bs
            sum_contr += parts["loss_contrastive"] * bs
            n += bs

    loss_results = {
        "test_loss_total": sum_total / max(n, 1),
        "test_loss_bpr": sum_bpr / max(n, 1),
        "test_loss_contrastive": sum_contr / max(n, 1),
    }

    eval_metrics = run_evaluation(
        model=model, dataloader=dl, item_catalog=item_catalog,
        graph_x=graph_x, graph_edge_index=graph_edge_index,
        device=device, k_values=config.eval.k_values, split="test",
    )

    combined = {**loss_results, **eval_metrics}
    print(f"  [{label}] loss={loss_results['test_loss_total']:.4f} "
          f"hit@10={eval_metrics.get('hit@10',0):.4f} "
          f"drift={eval_metrics.get('ideo_drift@10',0):.4f} "
          f"in_win={eval_metrics.get('ideo_in_window',0):.4f}")
    return combined


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bandit + RecSys Integration Evaluation")
    parser.add_argument("--device",     type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--data_dir",   type=str, default=None,
                        help="Path to processed .pkl files")
    parser.add_argument("--bandit_dir", type=str, default=None,
                        help="Path to bandit outputs (linucb_params.json, simulation_summary.json)")
    parser.add_argument("--save_dir",   type=str, default=None,
                        help="Where to save checkpoint and results")
    args = parser.parse_args()

    # Resolve paths
    config = copy.deepcopy(cfg)
    config.train.batch_size = args.batch_size
    config.train.num_epochs = args.epochs

    data_dir = args.data_dir or config.paths.processed_dir
    bandit_dir = args.bandit_dir or str(PROJECT_DIR / "bandit" / "outputs")
    save_dir = Path(args.save_dir) if args.save_dir else PROJECT_DIR / "outputs"
    save_dir.mkdir(parents=True, exist_ok=True)


    device = resolve_device(args.device if args.device != "auto" else config.train.device)
    print(f"Device: {device}")
    print(f"Data dir: {data_dir}")
    print(f"Bandit dir: {bandit_dir}")
    print(f"Save dir: {save_dir}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 1: Load bandit policy + assign deltas
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 1: Load bandit policy & assign per-user deltas")
    print("=" * 65)

    A_matrices, b_vectors, arms, alpha, summary = load_bandit_policy(bandit_dir)
    print(f"  Arms: {arms}, Alpha: {alpha}, Feature dim: {A_matrices[0].shape[0]}")

    # Load data
    d = Path(data_dir)
    with open(d / "scored_rt_sequences.pkl", "rb") as f: scored = pickle.load(f)
    with open(d / "user_ideology_states.pkl", "rb") as f: states = pickle.load(f)
    with open(d / "user2idx-3.pkl", "rb") as f: user2idx = pickle.load(f)

    user_deltas = assign_bandit_deltas(states, scored, A_matrices, b_vectors, arms, alpha)

    counts = Counter(user_deltas.values())
    print(f"\n  Bandit delta distribution ({len(user_deltas)} users):")
    print(f"  {'|delta|':>8}  {'count':>7}  {'fraction':>10}")
    for d_val in sorted(counts):
        print(f"  {d_val:8.3f}  {counts[d_val]:7d}  {counts[d_val]/len(user_deltas):10.4f}")
    print(f"  Mean |delta|: {np.mean(list(user_deltas.values())):.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 2: Build datasets with bandit deltas
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 2: Build datasets with bandit deltas")
    print("=" * 65)

    item2idx, idx2item, item_ideo_dict = build_item_vocab(scored, config.data.min_item_freq)

    train_ds = build_dataset_with_deltas(scored, states, item2idx, item_ideo_dict, user2idx,
                                          config, "train", user_deltas, 0.15)
    val_ds = build_dataset_with_deltas(scored, states, item2idx, item_ideo_dict, user2idx,
                                        config, "val", user_deltas, 0.15)
    test_ds = build_dataset_with_deltas(scored, states, item2idx, item_ideo_dict, user2idx,
                                         config, "test", user_deltas, 0.15)

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
    train_dl = DataLoader(train_ds, shuffle=True, **common)
    val_dl = DataLoader(val_ds, shuffle=False, **common)

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 3: Load graph + train model
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 3: Train recommender with bandit deltas")
    print("=" * 65)

    graph_x, graph_edge_index = load_graph_tensors(data_dir, device)

    model = build_model(item_catalog.num_items, config).to(device)
    with torch.no_grad():
        all_graph_embs = model.graph_encoder(graph_x, graph_edge_index)

    model, best_hit10 = train_model(
        model, train_dl, val_dl, item_catalog,
        graph_x, graph_edge_index, all_graph_embs,
        config, device, args.epochs,
    )

    # Save checkpoint
    ckpt_path = save_dir / "best_model_bandit_integrated.pt"
    torch.save({"model_state": model.state_dict(), "best_hit10": best_hit10}, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 4: Evaluate with 3 delta strategies
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("STEP 4: Evaluate with different delta strategies")
    print("=" * 65)

    # A) Bandit deltas (trained with these)
    print("\n  [A] Bandit-Integrated (model trained + evaluated with bandit δ):")
    results_bandit = evaluate_setting(
        model, scored, states, item2idx, item_ideo_dict, user2idx,
        item_catalog, collate, config, graph_x, graph_edge_index,
        all_graph_embs, device, user_deltas, 0.15, "Bandit",
    )

    # B) Fixed δ=0.2 (same model, different window)
    print("\n  [B] Fixed δ=0.2 (same model, wider window):")
    results_fixed = evaluate_setting(
        model, scored, states, item2idx, item_ideo_dict, user2idx,
        item_catalog, collate, config, graph_x, graph_edge_index,
        all_graph_embs, device, {}, 0.2, "Fixed-0.2",
    )

    # C) Random deltas
    print("\n  [C] Random δ (same model, random arms):")
    rng = np.random.RandomState(42)
    random_deltas = {uid: abs(arms[rng.randint(len(arms))]) for uid in user_deltas}
    results_random = evaluate_setting(
        model, scored, states, item2idx, item_ideo_dict, user2idx,
        item_catalog, collate, config, graph_x, graph_edge_index,
        all_graph_embs, device, random_deltas, 0.15, "Random",
    )

    # ══════════════════════════════════════════════════════════════════════
    #  STEP 5: Print results table
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("INTEGRATION EVALUATION RESULTS")
    print("=" * 65)

    all_results = {
        "Bandit-Integrated": results_bandit,
        "Fixed-δ=0.2": results_fixed,
        "Random-δ": results_random,
    }

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

    # Save results
    results_path = save_dir / "integration_results.json"
    def jsonable(obj):
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: jsonable(v) for k, v in obj.items()}
        return obj
    with open(results_path, "w") as f:
        json.dump(jsonable(all_results), f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Verdict
    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    b = results_bandit
    f_ = results_fixed
    r = results_random

    print(f"\n  {'':26} {'Bandit':>10} {'Fixed':>10} {'Random':>10}")
    print(f"  {'─'*26} {'─'*10} {'─'*10} {'─'*10}")
    for key, label in [
        ("test_loss_total", "Test Loss"),
        ("hit@10", "Hit@10"),
        ("ndcg@10", "NDCG@10"),
        ("ideo_drift@10", "Ideo Drift@10"),
        ("ideo_in_window", "In-Window Frac"),
        ("direction_acc", "Direction Acc"),
    ]:
        print(f"  {label:26} {b.get(key,0):>10.4f} {f_.get(key,0):>10.4f} {r.get(key,0):>10.4f}")

    # Compare bandit vs others
    d_loss_f = b["test_loss_total"] - f_["test_loss_total"]
    d_loss_r = b["test_loss_total"] - r["test_loss_total"]
    d_hit_f = b.get("hit@10", 0) - f_.get("hit@10", 0)
    d_hit_r = b.get("hit@10", 0) - r.get("hit@10", 0)

    print(f"\n  Bandit vs Fixed:  loss {d_loss_f:+.4f} ({'better' if d_loss_f < 0 else 'worse'})  "
          f"hit@10 {d_hit_f:+.4f} ({'better' if d_hit_f > 0 else 'worse'})")
    print(f"  Bandit vs Random: loss {d_loss_r:+.4f} ({'better' if d_loss_r < 0 else 'worse'})  "
          f"hit@10 {d_hit_r:+.4f} ({'better' if d_hit_r > 0 else 'worse'})")


if __name__ == "__main__":
    main()
