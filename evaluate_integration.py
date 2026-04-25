"""
evaluate_integration.py
-----------------------
End-to-end evaluation of the Bandit ↔ Recommender integrated system.

For each test user:
  1. Bandit (LinUCB) selects a drift δ based on the user's ideology context
  2. Recommender uses that δ to define the ideology window for ranking
  3. We measure: test loss, Hit@K, NDCG@K, ideology drift, window fraction

Compares three settings:
  A) Bandit-integrated: per-user δ from trained LinUCB
  B) Fixed baseline:    δ = 0.2 for all users (RecSys default)
  C) Random baseline:   δ chosen uniformly from bandit arms

Usage:
    cd final_PROJECT
    python evaluate_integration.py [--device cpu] [--batch_size 1024]
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
    user_ideology_states: dict,          # {uid: [ideo_t0, ideo_t1, ...]}
    scored_rt_sequences:  dict,          # {uid: [(rt_uid, score), ...]}
    user2idx:             dict,          # {uid: graph_node_idx}
    linucb_policy:        dict,
    arms:                 list[float],
    alpha:                float,
    graph_data            = None,        # torch_geometric Data (optional)
) -> dict[str, float]:
    """
    For each user, build a context vector and let the bandit choose an arm.
    Returns {user_id: |delta|} — absolute delta used as ideology-window width.
    """
    # ── optional: compute degree_norm from graph ──────────────────────────
    degree_map: dict[str, float] = {}
    if graph_data is not None:
        ei = graph_data.edge_index.numpy()
        cnt = Counter(ei[0].tolist()) + Counter(ei[1].tolist())
        max_d = max(cnt.values()) if cnt else 1
        for uid, idx in user2idx.items():
            degree_map[uid] = cnt.get(idx, 0) / max_d

    user_deltas: dict[str, float] = {}

    for uid, states in user_ideology_states.items():
        if uid not in scored_rt_sequences:
            continue
        if not states:
            continue

        # current ideology = last observed state
        cur = states[-1] if np.isfinite(states[-1]) else 0.0
        finite = [s for s in states if np.isfinite(s)]
        mean_i = float(np.mean(finite))   if finite else 0.0
        std_i  = float(np.std(finite))    if finite else 0.5

        ctx = build_user_context(
            ideology      = cur,
            degree_norm   = degree_map.get(uid, 0.5),
            avg_nbr_ideo  = mean_i,
            nbr_ideo_std  = min(std_i, 1.0),
            echo_score    = min(abs(cur) / 3.0, 1.0),
        )

        arm_idx = linucb_select_arm(linucb_policy, ctx, alpha)
        user_deltas[uid] = abs(arms[arm_idx])       # window width (always ≥ 0)

    return user_deltas


# =============================================================================
#  4.  BUILD TEST-SET WITH PER-USER BANDIT DELTAS
# =============================================================================

def build_test_dataloader(
    processed_dir:  str,
    user_deltas:    dict[str, float],
    fallback_delta: float,
    config:         cConfig,
):
    """
    Build a test DataLoader whose samples carry bandit-assigned deltas.

    Returns: (test_dl, item_catalog)
    """
    d = Path(processed_dir)
    with open(d / "scored_rt_sequences.pkl",  "rb") as f:  scored  = pickle.load(f)
    with open(d / "user_ideology_states.pkl", "rb") as f:  states  = pickle.load(f)
    with open(d / "user2idx-3.pkl",           "rb") as f:  user2idx = pickle.load(f)

    item2idx, idx2item, item_ideo_dict = build_item_vocab(scored, config.data.min_item_freq)

    # Build test dataset with the fallback delta (will override per-sample)
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

    # Override each sample's delta with the bandit-assigned value
    overridden = 0
    for sample in test_ds.samples:
        uid = sample["user_id"]
        if uid in user_deltas:
            sample["delta"] = max(user_deltas[uid], 0.05)  # floor at 0.05
            overridden += 1
    print(f"  [test] Overrode {overridden}/{len(test_ds.samples)} deltas with bandit values")

    # Item catalog array
    num_items = (max(item2idx.values()) + 1) if item2idx else 1
    item_ideo_arr = np.zeros(num_items, dtype=np.float32)
    for idx, sc in item_ideo_dict.items():
        if 0 <= idx < num_items:
            item_ideo_arr[idx] = float(sc) if np.isfinite(sc) else 0.0
    item_ideo_arr = np.nan_to_num(item_ideo_arr, nan=0.0, posinf=0.0, neginf=0.0)

    item_catalog = ItemCatalog(
        item2idx  = item2idx,
        idx2item  = idx2item,
        item_ideo = item_ideo_arr,
    )

    neg_sampler = NegativeSampler(item_ideo_dict, strategy="hard", band=0.5)
    collate = CollateWithNegatives(neg_sampler, item_ideo_arr, num_negatives=1)

    test_dl = DataLoader(
        test_ds,
        batch_size  = config.train.batch_size,
        shuffle     = False,
        num_workers = 0,            # safe for local eval
        collate_fn  = collate,
    )
    return test_dl, item_catalog


# =============================================================================
#  5.  COMPUTE TEST LOSS  (BPR + Ideology-Contrastive)
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
) -> dict[str, float]:
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
            graph_x          = graph_x,
            graph_edge_index = graph_edge_index,
            user_graph_idx   = user_idx,
            seq_item_ids     = seq_items,
            seq_ideo_scores  = seq_ideo,
            pos_item_ids     = pos_items,
            pos_ideo_scores  = pos_ideo,
            neg_item_ids     = neg_items,
            neg_ideo_scores  = neg_ideo,
            all_graph_embs   = all_graph_embs,
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
#  6.  LOAD RECOMMENDER CHECKPOINT
# =============================================================================

def load_recommender(checkpoint_path: str, config: cConfig, device: str):
    """
    Load the trained IdeologyRecommender from a checkpoint.

    Handles both:
      - classic .pt file  (torch.save dict)
      - directory-based checkpoint (PyTorch >= 2.1 mmap format)
    """
    ckpt_path = Path(checkpoint_path)

    # Need num_items to build the model — peek from saved config or data
    ckpt = torch.load(
        str(ckpt_path), map_location=device, weights_only=False,
    )

    # Checkpoint saved by train.py: {epoch, model_state, optim_state, best_hit10, config}
    saved_cfg = ckpt.get("config", config)
    model_state = ckpt["model_state"]

    # Infer num_items from the embedding weight shape
    # tweet_encoder.item_embed.weight has shape (num_items, embed_dim)
    num_items = model_state["tweet_encoder.item_embed.weight"].shape[0]
    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch','?')}, "
          f"best_hit@10={ckpt.get('best_hit10', '?')}, num_items={num_items}")

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
#  7.  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bandit + RecSys Integration")
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--batch_size",   type=int, default=1024)
    parser.add_argument("--checkpoint",   type=str, default=None,
                        help="Path to recommender checkpoint (default: RecSys/checkpoints/best_model)")
    parser.add_argument("--bandit_params",type=str, default=None,
                        help="Path to linucb_params.json")
    args = parser.parse_args()

    config = copy.deepcopy(cfg)
    config.train.batch_size = args.batch_size
    device = resolve_device(args.device if args.device != "auto" else config.train.device)
    print(f"Device: {device}")

    ckpt_path = args.checkpoint or str(Path(config.paths.checkpoint_dir) / "best_model")
    bandit_path = args.bandit_params or str(BANDIT_DIR / "outputs" / "linucb_params.json")

    # ── 1. Load bandit policy ────────────────────────────────────────────
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
    arms  = sim_summary["arms"]       # [-0.3, -0.15, 0.0, 0.15, 0.3]
    alpha = sim_summary["alpha"]       # 1.5
    print(f"  Arms values: {arms}")
    print(f"  Alpha: {alpha}")

    # ── 2. Load recommender model ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2  Load trained Recommender")
    print("=" * 65)
    model = load_recommender(ckpt_path, config, device)

    # ── 3. Load graph ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3  Load graph tensors")
    print("=" * 65)
    graph_x, graph_edge_index = load_graph_tensors(config.paths.processed_dir, device)

    with torch.no_grad():
        all_graph_embs = model.graph_encoder(graph_x, graph_edge_index)

    # ── 4. Assign bandit deltas to users ─────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 4  Assign bandit deltas to every user")
    print("=" * 65)
    processed_dir = config.paths.processed_dir
    with open(Path(processed_dir) / "scored_rt_sequences.pkl",  "rb") as f: scored = pickle.load(f)
    with open(Path(processed_dir) / "user_ideology_states.pkl", "rb") as f: states = pickle.load(f)
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

    # Print delta distribution
    delta_counts = Counter(user_deltas.values())
    print(f"\n  Bandit delta distribution ({len(user_deltas)} users):")
    print(f"  {'|delta|':>8}  {'count':>7}  {'fraction':>10}")
    for d_val in sorted(delta_counts):
        frac = delta_counts[d_val] / len(user_deltas)
        print(f"  {d_val:8.3f}  {delta_counts[d_val]:7d}  {frac:10.4f}")
    print(f"  Mean |delta|: {np.mean(list(user_deltas.values())):.4f}")

    # ── 5. Build test dataloaders for each setting ───────────────────────
    print("\n" + "=" * 65)
    print("STEP 5  Build test dataloaders")
    print("=" * 65)

    loss_fn = IdeologyLoss(
        alpha_bpr         = config.loss.alpha_bpr,
        alpha_contrastive = config.loss.alpha_contrastive,
    )

    settings = {}

    # A) Bandit-integrated
    print("\n  [A] Bandit-integrated deltas:")
    dl_bandit, cat_bandit = build_test_dataloader(
        processed_dir, user_deltas, fallback_delta=0.15, config=config,
    )
    settings["Bandit-Integrated"] = (dl_bandit, cat_bandit)

    # B) Fixed delta = 0.2  (RecSys default)
    print("\n  [B] Fixed delta = 0.2 (baseline):")
    dl_fixed, cat_fixed = build_test_dataloader(
        processed_dir, {}, fallback_delta=0.2, config=config,
    )
    settings["Fixed-δ=0.2"] = (dl_fixed, cat_fixed)

    # C) Random bandit deltas
    print("\n  [C] Random deltas (baseline):")
    rng = np.random.RandomState(42)
    random_deltas = {
        uid: abs(arms[rng.randint(n_arms)])
        for uid in user_deltas
    }
    dl_random, cat_random = build_test_dataloader(
        processed_dir, random_deltas, fallback_delta=0.15, config=config,
    )
    settings["Random-δ"] = (dl_random, cat_random)

    # ── 6. Evaluate each setting ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 6  Evaluate all settings on test set")
    print("=" * 65)

    all_results = {}

    for name, (dl, catalog) in settings.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()

        # Test loss
        loss_metrics = compute_test_loss(
            model, dl, loss_fn, graph_x, graph_edge_index,
            all_graph_embs, device,
        )

        # Retrieval + ideology metrics
        eval_metrics = run_evaluation(
            model          = model,
            dataloader     = dl,
            item_catalog   = catalog,
            graph_x        = graph_x,
            graph_edge_index = graph_edge_index,
            device         = device,
            k_values       = config.eval.k_values,
            split          = "test",
        )

        combined = {**loss_metrics, **eval_metrics}
        all_results[name] = combined
        print(f"    Completed in {time.time() - t0:.1f}s")

    # ── 7. Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("INTEGRATION EVALUATION RESULTS")
    print("=" * 65)

    names = list(all_results.keys())
    header = f"{'Metric':<28}" + "".join(f" {n:>18}" for n in names)
    print(f"\n{header}")
    print("─" * len(header))

    # Loss metrics first
    for key in ["test_loss_total", "test_loss_bpr", "test_loss_contrastive"]:
        row = f"{key:<28}"
        for n in names:
            row += f" {all_results[n].get(key, 0):>18.4f}"
        print(row)

    print()

    # Retrieval + ideology metrics
    metric_keys = sorted(set(
        k for r in all_results.values() for k in r
        if k not in ("test_loss_total", "test_loss_bpr", "test_loss_contrastive", "num_samples")
    ))
    for key in metric_keys:
        row = f"{key:<28}"
        for n in names:
            row += f" {all_results[n].get(key, 0):>18.4f}"
        print(row)

    # ── 8. Save results ──────────────────────────────────────────────────
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

    # ── 9. Summary verdict ───────────────────────────────────────────────
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

    # Improvement summary
    delta_loss = f_["test_loss_total"] - b["test_loss_total"]
    delta_hit  = b.get("hit@10", 0) - f_.get("hit@10", 0)
    delta_wind = b.get("ideo_in_window", 0) - f_.get("ideo_in_window", 0)

    print(f"\n  Bandit vs Fixed baseline:")
    sign = "+" if delta_loss >= 0 else ""
    print(f"    Test loss reduction: {sign}{delta_loss:.4f}  "
          f"({'better' if delta_loss > 0 else 'worse'})")
    sign = "+" if delta_hit >= 0 else ""
    print(f"    Hit@10 change:      {sign}{delta_hit:.4f}  "
          f"({'better' if delta_hit > 0 else 'worse'})")
    sign = "+" if delta_wind >= 0 else ""
    print(f"    In-window change:   {sign}{delta_wind:.4f}  "
          f"({'better' if delta_wind > 0 else 'worse'})")


if __name__ == "__main__":
    main()
