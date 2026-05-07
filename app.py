"""
app.py — Streamlit Dashboard for Bandit ↔ Recommender Integration
=================================================================
Full pipeline demo:
  1. Select/explore a user
  2. See bandit pick a drift (UCB scores per arm)
  3. See recommender rank tweets within the ideology window
  4. View system-wide metrics and comparisons

Run:
    streamlit run app.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
RECSYS_DIR  = PROJECT_DIR / "RecSys"
BANDIT_DIR  = PROJECT_DIR / "bandit"
sys.path.insert(0, str(RECSYS_DIR))

from config import cfg
from train import resolve_device, load_graph_tensors, build_model
from models.recommender import IdeologyRecommender


# =============================================================================
#  CACHED DATA LOADERS
# =============================================================================

@st.cache_data
def load_processed_data():
    """Load all .pkl data files."""
    d = Path(cfg.paths.processed_dir)
    with open(d / "scored_rt_sequences.pkl", "rb") as f:
        scored = pickle.load(f)
    with open(d / "user_ideology_states.pkl", "rb") as f:
        states = pickle.load(f)
    with open(d / "user2idx-3.pkl", "rb") as f:
        user2idx = pickle.load(f)
    with open(d / "ideology_map.pkl", "rb") as f:
        ideology_map = pickle.load(f)
    return scored, states, user2idx, ideology_map


@st.cache_data
def load_bandit_outputs():
    """Load bandit simulation results."""
    out = BANDIT_DIR / "outputs"
    with open(out / "linucb_params.json") as f:
        params = json.load(f)
    with open(out / "simulation_summary.json") as f:
        summary = json.load(f)

    # Load per-strategy metrics
    metrics = {}
    for strategy in ["LinUCB", "Random", "GreedyEcho"]:
        mp = out / strategy / "metrics.json"
        if mp.exists():
            with open(mp) as f:
                metrics[strategy] = json.load(f)

    return params, summary, metrics


@st.cache_data
def load_integration_results():
    """Load integration evaluation results if available."""
    eval_dir = Path(cfg.paths.eval_dir)
    p = eval_dir / "integration_results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_recommender_model():
    """Load the trained recommender model."""
    ckpt_path = Path(cfg.paths.checkpoint_dir) / "best_model"
    if not ckpt_path.exists():
        ckpt_path = Path(cfg.paths.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        return None, None, None

    device = "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model_state = ckpt["model_state"]
    num_items = model_state["tweet_encoder.item_embed.weight"].shape[0]

    model = IdeologyRecommender(
        num_items=num_items, num_graph_nodes=0,
        embed_dim=cfg.tweet_enc.output_dim,
        graph_node_feat=cfg.graph.node_feat_dim,
        hidden_dim=cfg.graph.hidden_dim,
        num_sage_layers=cfg.graph.num_layers,
        num_sasrec_layers=cfg.sasrec.num_layers,
        num_heads=cfg.sasrec.num_heads,
        max_seq_len=cfg.data.max_seq_len,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    graph_x, graph_edge_index = load_graph_tensors(cfg.paths.processed_dir, device)
    with torch.no_grad():
        all_graph_embs = model.graph_encoder(graph_x, graph_edge_index)

    return model, all_graph_embs, device


# =============================================================================
#  BANDIT INFERENCE
# =============================================================================

def linucb_inference(params: dict, context: np.ndarray, alpha: float):
    """Run LinUCB and return per-arm UCB scores."""
    A_matrices = [np.array(a, dtype=np.float64) for a in params["A_matrices"]]
    b_vectors = [np.array(b, dtype=np.float64) for b in params["b_vectors"]]

    x = context.astype(np.float64)
    d = A_matrices[0].shape[0]
    n_arms = len(A_matrices)

    results = []
    for a in range(n_arms):
        A_inv = np.linalg.solve(A_matrices[a], np.eye(d))
        theta = A_inv @ b_vectors[a]
        exploit = float(theta @ x)
        explore = float(np.sqrt(x @ A_inv @ x))
        ucb = exploit + alpha * explore
        results.append({
            "arm": a,
            "exploitation": exploit,
            "exploration": explore,
            "ucb_score": ucb,
        })

    return results


def build_context_for_user(uid, states, ideology_map, user2idx, graph_data=None):
    """Build the 10-dim context vector for a user."""
    user_states = states.get(uid, [0.0])
    cur = user_states[-1] if np.isfinite(user_states[-1]) else 0.0
    finite = [s for s in user_states if np.isfinite(s)]
    mean_i = float(np.mean(finite)) if finite else 0.0
    std_i = float(np.std(finite)) if finite else 0.5

    return np.array([
        cur / 3.0,
        0.5,  # degree_norm (approximate)
        mean_i / 3.0,
        min(std_i, 1.0),
        min(abs(cur) / 3.0, 1.0),
        0.3,  # retweet_ratio (approximate)
        0.5,  # session_progress
        0.5,  # recent_engage_rate
        0.15, # avg_acc_delta
        1.0,  # bias
    ], dtype=np.float64)


# =============================================================================
#  STREAMLIT APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="Bandit × RecSys Integration",
        page_icon="🎯",
        layout="wide",
    )

    st.title("🎯 Bandit × Recommender System Integration")
    st.markdown("**Full Pipeline:** User → Bandit selects drift → Recommender ranks tweets within ideology window")

    # Load data
    scored, states, user2idx, ideology_map = load_processed_data()
    params, summary, bandit_metrics = load_bandit_outputs()
    integration_results = load_integration_results()

    arms = summary["arms"]
    alpha = summary["alpha"]

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown(f"**Arms:** {arms}")
    st.sidebar.markdown(f"**Alpha (exploration):** {alpha}")
    st.sidebar.markdown(f"**TAU (max shift):** {summary['tau']}")
    st.sidebar.markdown(f"**Total users:** {summary['total_users']}")
    st.sidebar.markdown(f"**Steps/user:** {summary['steps_per_user']}")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Per-User Pipeline",
        "📊 Bandit Performance",
        "🔗 Integration Results",
        "📈 System Overview",
    ])

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: Per-User Pipeline Demo
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.header("Per-User Pipeline Demo")

        all_users = sorted(states.keys())

        col_select, col_info = st.columns([1, 2])
        with col_select:
            user_idx_input = st.number_input(
                "User index (0 to {}):".format(len(all_users) - 1),
                min_value=0, max_value=len(all_users) - 1, value=0,
            )
            selected_user = all_users[user_idx_input]
            st.markdown(f"**User ID:** `{selected_user}`")

        user_states = states.get(selected_user, [0.0])
        current_ideo = user_states[-1] if np.isfinite(user_states[-1]) else 0.0

        with col_info:
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Ideology", f"{current_ideo:.3f}")
            c2.metric("History Length", len(user_states))
            seq_len = len(scored.get(selected_user, []))
            c3.metric("Retweet Sequence", seq_len)

        st.divider()

        # ── Step 1: Ideology Trajectory ──────────────────────────────────
        st.subheader("Step 1: User Ideology Trajectory")
        finite_states = [s for s in user_states if np.isfinite(s)]
        if finite_states:
            import plotly.graph_objects as go

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                y=finite_states, mode="lines+markers",
                name="Ideology", line=dict(color="#636EFA"),
            ))
            fig_traj.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Center")
            fig_traj.update_layout(
                height=250, margin=dict(t=30, b=30),
                xaxis_title="Time step", yaxis_title="Ideology score",
                yaxis_range=[-3, 3],
            )
            st.plotly_chart(fig_traj, use_container_width=True)

        st.divider()

        # ── Step 2: Bandit Selects Drift ─────────────────────────────────
        st.subheader("Step 2: Bandit (LinUCB) Selects Drift")

        context = build_context_for_user(selected_user, states, ideology_map, user2idx)
        arm_results = linucb_inference(params, context, alpha)

        best_arm = max(arm_results, key=lambda x: x["ucb_score"])
        selected_delta = arms[best_arm["arm"]]

        col_ctx, col_ucb = st.columns([1, 2])

        with col_ctx:
            st.markdown("**Context Vector (10-dim):**")
            ctx_labels = [
                "ideology/3", "degree_norm", "nbr_ideo/3", "nbr_std",
                "echo_score", "rt_ratio", "session_prog", "recent_engage",
                "avg_delta", "bias"
            ]
            for label, val in zip(ctx_labels, context):
                st.text(f"  {label:14s} = {val:.4f}")

        with col_ucb:
            st.markdown("**UCB Scores per Arm:**")
            import plotly.graph_objects as go

            arm_labels = [f"δ={arms[r['arm']]}" for r in arm_results]
            ucb_scores = [r["ucb_score"] for r in arm_results]
            exploits = [r["exploitation"] for r in arm_results]
            explores = [r["exploration"] for r in arm_results]

            colors = ["#2ecc71" if r["arm"] == best_arm["arm"] else "#95a5a6" for r in arm_results]

            fig_ucb = go.Figure()
            fig_ucb.add_trace(go.Bar(
                x=arm_labels, y=exploits, name="Exploitation",
                marker_color="#3498db",
            ))
            fig_ucb.add_trace(go.Bar(
                x=arm_labels, y=[alpha * e for e in explores], name=f"Exploration (α={alpha})",
                marker_color="#e74c3c",
            ))
            fig_ucb.update_layout(
                barmode="stack", height=300, margin=dict(t=30, b=30),
                yaxis_title="UCB Score",
            )
            st.plotly_chart(fig_ucb, use_container_width=True)

        st.success(f"**Bandit selects arm {best_arm['arm']}** → drift δ = **{selected_delta}** "
                   f"(UCB = {best_arm['ucb_score']:.4f})")

        st.divider()

        # ── Step 3: Recommender Uses Drift ───────────────────────────────
        st.subheader("Step 3: Recommender Ranks Tweets in Ideology Window")

        direction = -1.0 if current_ideo > 0 else 1.0
        window_lo = current_ideo + min(0, direction * abs(selected_delta))
        window_hi = current_ideo + max(0, direction * abs(selected_delta))

        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Current ideology | {current_ideo:.3f} |
        | Nudge direction | {"← left (toward center)" if direction < 0 else "→ right (toward center)"} |
        | Delta (window width) | {abs(selected_delta):.3f} |
        | **Ideology window** | **[{window_lo:.3f}, {window_hi:.3f}]** |
        """)

        # Show items in window from scored sequences
        user_seq = scored.get(selected_user, [])
        if user_seq:
            in_window = [(uid, sc) for uid, sc in user_seq if window_lo <= sc <= window_hi]
            outside = [(uid, sc) for uid, sc in user_seq if not (window_lo <= sc <= window_hi)]

            col_in, col_out = st.columns(2)
            with col_in:
                st.markdown(f"**Items in window:** {len(in_window)}")
                if in_window[:10]:
                    for uid, sc in in_window[:10]:
                        st.text(f"  {uid[:12]:12s}  ideo={sc:.3f}")
                    if len(in_window) > 10:
                        st.text(f"  ... +{len(in_window)-10} more")

            with col_out:
                st.markdown(f"**Items outside window:** {len(outside)}")
                if outside[:5]:
                    for uid, sc in outside[:5]:
                        st.text(f"  {uid[:12]:12s}  ideo={sc:.3f}")

        st.divider()

        # ── Step 4: Predicted Outcome ────────────────────────────────────
        st.subheader("Step 4: Expected Outcome")

        col_pred1, col_pred2, col_pred3 = st.columns(3)
        col_pred1.metric("Predicted New Ideology",
                         f"{current_ideo + direction * abs(selected_delta) * 0.5:.3f}",
                         f"{direction * abs(selected_delta) * 0.5:+.3f}")
        col_pred2.metric("Window Width", f"{abs(selected_delta):.3f}")
        col_pred3.metric("Direction", "Toward Center ✓" if abs(current_ideo + direction * abs(selected_delta)) < abs(current_ideo) else "Away ✗")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: Bandit Performance
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.header("Bandit Simulation Performance")

        if bandit_metrics:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # ── Arm Selection Distribution ────────────────────────────────
            st.subheader("Arm Selection Distribution (LinUCB)")
            linucb_m = bandit_metrics.get("LinUCB", {})
            arm_sel = linucb_m.get("arm_selection", {})

            if arm_sel:
                fig_arms = go.Figure(go.Bar(
                    x=list(arm_sel.keys()),
                    y=[v["fraction"] for v in arm_sel.values()],
                    marker_color=["#e74c3c", "#f39c12", "#95a5a6", "#2ecc71", "#3498db"],
                    text=[f"{v['fraction']*100:.1f}%" for v in arm_sel.values()],
                    textposition="outside",
                ))
                fig_arms.update_layout(
                    height=300, margin=dict(t=30),
                    xaxis_title="Arm (δ value)", yaxis_title="Selection Fraction",
                )
                st.plotly_chart(fig_arms, use_container_width=True)

            # ── Comparison Table ──────────────────────────────────────────
            st.subheader("Strategy Comparison")

            comp_data = []
            for name, m in bandit_metrics.items():
                comp_data.append({
                    "Strategy": name,
                    "Avg Reward/User": m["engagement"]["avg_total_reward"],
                    "Engagement Rate": m["engagement"]["avg_engagement_rate"],
                    "Avg |Ideo Shift|": m["diversity"]["avg_abs_ideology_shift"],
                    "% Toward Center": m["diversity"]["pct_shifted_toward_center"],
                    "Polarization Δ": m["polarization"]["polarization_change"],
                })

            import pandas as pd
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp.set_index("Strategy"), use_container_width=True)

            # ── Learning Curve ────────────────────────────────────────────
            st.subheader("Learning Curve (LinUCB)")
            lc = linucb_m.get("learning_curve", {})
            if lc:
                steps = sorted(lc.keys(), key=int)
                rewards = [lc[s] for s in steps]
                fig_lc = go.Figure(go.Scatter(
                    x=[int(s) for s in steps], y=rewards,
                    mode="lines+markers", line=dict(color="#636EFA"),
                ))
                fig_lc.update_layout(
                    height=300, margin=dict(t=30),
                    xaxis_title="Step", yaxis_title="Avg Reward",
                )
                st.plotly_chart(fig_lc, use_container_width=True)

            # ── Engagement per Arm ────────────────────────────────────────
            st.subheader("Engagement Rate per Arm")
            eng_arm = linucb_m.get("engagement_per_arm", {})
            if eng_arm:
                fig_eng = go.Figure(go.Bar(
                    x=list(eng_arm.keys()),
                    y=list(eng_arm.values()),
                    marker_color=["#e74c3c", "#f39c12", "#95a5a6", "#2ecc71", "#3498db"],
                ))
                fig_eng.update_layout(
                    height=300, margin=dict(t=30),
                    xaxis_title="Arm (δ)", yaxis_title="Engagement Rate",
                )
                st.plotly_chart(fig_eng, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3: Integration Results
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.header("Integration Evaluation Results")

        if integration_results:
            import pandas as pd
            import plotly.graph_objects as go

            settings = list(integration_results.keys())

            # ── Loss Comparison ───────────────────────────────────────────
            st.subheader("Test Loss Comparison")
            loss_data = []
            for name in settings:
                r = integration_results[name]
                loss_data.append({
                    "Setting": name,
                    "Total Loss": r.get("test_loss_total", 0),
                    "BPR Loss": r.get("test_loss_bpr", 0),
                    "Contrastive Loss": r.get("test_loss_contrastive", 0),
                })
            df_loss = pd.DataFrame(loss_data)
            st.dataframe(df_loss.set_index("Setting"), use_container_width=True)

            fig_loss = go.Figure()
            for metric in ["Total Loss", "BPR Loss", "Contrastive Loss"]:
                fig_loss.add_trace(go.Bar(
                    x=df_loss["Setting"], y=df_loss[metric], name=metric,
                ))
            fig_loss.update_layout(barmode="group", height=350, margin=dict(t=30))
            st.plotly_chart(fig_loss, use_container_width=True)

            # ── Retrieval Metrics ─────────────────────────────────────────
            st.subheader("Retrieval Metrics")
            ret_data = []
            for name in settings:
                r = integration_results[name]
                ret_data.append({
                    "Setting": name,
                    "Hit@5": r.get("hit@5", 0),
                    "Hit@10": r.get("hit@10", 0),
                    "Hit@20": r.get("hit@20", 0),
                    "NDCG@5": r.get("ndcg@5", 0),
                    "NDCG@10": r.get("ndcg@10", 0),
                    "NDCG@20": r.get("ndcg@20", 0),
                })
            df_ret = pd.DataFrame(ret_data)
            st.dataframe(df_ret.set_index("Setting"), use_container_width=True)

            # ── Ideology Metrics ──────────────────────────────────────────
            st.subheader("Ideology Metrics")
            ideo_data = []
            for name in settings:
                r = integration_results[name]
                ideo_data.append({
                    "Setting": name,
                    "Ideo Drift@10": r.get("ideo_drift@10", 0),
                    "In-Window Frac": r.get("ideo_in_window", 0),
                    "Direction Acc": r.get("direction_acc", 0),
                })
            df_ideo = pd.DataFrame(ideo_data)
            st.dataframe(df_ideo.set_index("Setting"), use_container_width=True)

            fig_ideo = go.Figure()
            fig_ideo.add_trace(go.Bar(x=df_ideo["Setting"], y=df_ideo["In-Window Frac"], name="In-Window"))
            fig_ideo.add_trace(go.Bar(x=df_ideo["Setting"], y=df_ideo["Direction Acc"], name="Direction Acc"))
            fig_ideo.update_layout(barmode="group", height=300, margin=dict(t=30))
            st.plotly_chart(fig_ideo, use_container_width=True)

        else:
            st.warning("No integration results found. Run `evaluate_integration.py` first.")
            st.code("python evaluate_integration.py --device cuda --epochs 30", language="bash")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4: System Overview
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.header("System Architecture Overview")

        st.markdown("""
        ### Pipeline Architecture

        ```
        ┌─────────────────────────────┐         ┌──────────────────────────────┐
        │     BANDIT (LinUCB)         │         │   RECOMMENDER (RecSys)       │
        │                             │         │                              │
        │  User context (10-dim)      │───δ───▶ │  User encoding:              │
        │  ├── ideology               │         │  ├── GraphSAGE (social)      │
        │  ├── network features       │         │  ├── SASRec (sequential)     │
        │  ├── echo chamber score     │         │  └── Fusion (MLP)            │
        │  └── engagement history     │         │                              │
        │                             │         │  Item scoring:               │
        │  UCB = θᵀx + α√(xᵀA⁻¹x)   │         │  ├── dot(user, item)         │
        │                             │         │  └── mask to [cur, cur+δ]    │
        │  Arms: [-0.3, -0.15, 0,     │         │                              │
        │         +0.15, +0.3]        │         │  Loss: BPR + Contrastive     │
        └─────────────────────────────┘         └──────────────────────────────┘
                    │                                        │
                    └────────── engagement feedback ◀────────┘
        ```
        """)

        st.divider()

        # System stats
        st.subheader("System Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Users", f"{len(states):,}")
        col2.metric("Items (tweets)", f"{len(ideology_map):,}")
        col3.metric("Bandit Arms", len(arms))
        col4.metric("Bandit Rounds", f"{summary['total_rounds']:,}")

        st.divider()

        # LinUCB learned parameters
        st.subheader("LinUCB Learned Policy")
        arm_stats = summary.get("linucb_arm_stats", {})
        if arm_stats:
            import pandas as pd
            rows = []
            for k, v in arm_stats.items():
                rows.append({
                    "Arm": f"δ={v['arm_value']}",
                    "Pulls": v["pulls"],
                    "Pull %": f"{v['pull_fraction']*100:.1f}%",
                    "θ norm": f"{v['theta_norm']:.3f}",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Arm"), use_container_width=True)

        st.divider()

        # Data files
        st.subheader("Data Files")
        st.markdown(f"""
        | File | Location |
        |---|---|
        | Scored RT sequences | `{cfg.paths.processed_dir}/scored_rt_sequences.pkl` |
        | User ideology states | `{cfg.paths.processed_dir}/user_ideology_states.pkl` |
        | Graph data | `{cfg.paths.processed_dir}/graph_data-3.pkl` |
        | LinUCB params | `bandit/outputs/linucb_params.json` |
        | Behavior model | `bandit/outputs/behavior_model.pt` |
        | Recommender checkpoint | `RecSys/checkpoints/best_model` |
        """)


if __name__ == "__main__":
    main()
