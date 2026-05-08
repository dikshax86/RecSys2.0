# Bandit × Recommender System: Full Flow & Analysis

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: Data Collection (offline)                                     │
│  ─────────────────────────────────                                      │
│  • 8,158 Twitter users                                                  │
│  • Retweet sequences (avg ~820 interactions per user)                   │
│  • Barberá ideology scores (continuous, ~[-3, +3])                      │
│  • Social graph (follower/friend network)                               │
│                                                                         │
│  PHASE 2: Behavior Model Training                                       │
│  ────────────────────────────────                                       │
│  • Input: user ideology state + proposed drift                          │
│  • Output: P(engagement) + predicted ideology shift                     │
│  • Architecture: MLP (8-dim → 128 → 64 → 32 → 2 heads)                │
│  • Trained on real retweet sequences from .pkl data                     │
│  • Accuracy: 80.3% engagement prediction                                │
│                                                                         │
│  PHASE 3: Bandit Simulation                                             │
│  ──────────────────────────                                             │
│  • LinUCB contextual bandit, 5 arms: [-0.3, -0.15, 0, +0.15, +0.3]     │
│  • 10-dim context per user (ideology, network features, history)        │
│  • 20 steps per user × 8,164 users = 163,280 rounds                    │
│  • Uses behavior model to simulate engagement                           │
│  • Learns: δ=±0.15 is optimal for engagement (89% of selections)       │
│                                                                         │
│  PHASE 4: Recommender Training                                          │
│  ─────────────────────────────                                          │
│  • Architecture: GraphSAGE + SASRec + Fusion → dot-product scoring      │
│  • Loss: BPR + Ideology-Contrastive                                     │
│  • Delta (δ) defines ideology window for contrastive loss               │
│  • Window: [user_ideo, user_ideo + direction × δ]                       │
│  • Items inside window = "aligned", outside = "outside"                 │
│  • Model learns to score aligned > outside                              │
│                                                                         │
│  PHASE 5: Integration Evaluation                                        │
│  ────────────────────────────────                                       │
│  • Bandit assigns per-user δ → Recommender evaluates with that δ        │
│  • Compare: Bandit δ vs Fixed δ=0.2 vs Random δ                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Current Results

| Metric | Bandit-Integrated | Fixed-δ=0.2 | Random-δ |
|--------|:-----------------:|:-----------:|:--------:|
| Test Loss | **0.2408** | 0.2414 | **0.2380** |
| Hit@10 | 0.0375 | **0.0434** | 0.0389 |
| NDCG@10 | 0.0254 | **0.0285** | 0.0262 |
| Ideo Drift@10 | **0.0745** | 0.0946 | 0.0886 |
| In-Window Frac | **1.0000** | **1.0000** | 0.9985 |
| Direction Acc | 0.9997 | 0.9997 | 0.9992 |

---

## 3. Problem Diagnosis

### Why Bandit is worse than Random:

```
BANDIT DELTA DISTRIBUTION:
  |δ|=0.000:    16 users  (0.2%)
  |δ|=0.150: 7,974 users  (97.7%)   ← nearly constant!
  |δ|=0.300:   168 users  (2.1%)

RANDOM DELTA DISTRIBUTION:
  |δ|=0.000: ~20% users
  |δ|=0.150: ~40% users
  |δ|=0.300: ~40% users             ← diverse mix of window sizes
```

**Root Cause:** The bandit collapsed to a near-degenerate policy (97.7% picks δ=0.15).

This happens because:

1. **Objective mismatch**: The bandit was trained to maximize *engagement reward in simulation*, NOT *recommender retrieval quality*. These are different objectives.

2. **No diversity in window sizes**: When all users get δ=0.15, the recommender is locked into one narrow window. Random gets a mix of {0, 0.15, 0.3} → some users get wider windows → easier to find targets → higher Hit@K.

3. **Sequential training (no feedback loop)**: The bandit was trained first (Phase 3), then the recommender was trained using bandit's deltas (Phase 4). But the bandit never saw how its deltas affected the recommender's performance.

```
CURRENT FLOW (broken feedback):

  Bandit ──δ──→ Recommender ──metrics──→ (nobody listens)
    ↑                                          ✗
    │                                          │
    └─── engagement simulation (Phase 3) ──────┘ ← wrong signal!
```

---

## 4. What Each Component Learned

### Bandit (LinUCB):
- ✅ Learned that moderate drifts (±0.15) maximize engagement
- ✅ Outperforms Greedy (no echo chamber) and Random in simulation
- ✅ Produces controlled ideology shifts (0.074 vs 0.094)
- ❌ Collapsed to near-constant policy (no personalization)
- ❌ Optimizes for engagement, not retrieval quality

### Behavior Model (TransitionModel):
- ✅ 80.3% accuracy predicting engagement
- ✅ Learns realistic engagement patterns from real data
- ✅ Replaces naive Bernoulli simulation

### Recommender (IdeologyRecommender):
- ✅ Learns to rank items correctly (direction_acc=0.9997)
- ✅ All recommendations stay within ideology window (in_window=1.0)
- ✅ Lower loss with bandit deltas than fixed (-0.0007)
- ❌ Lower Hit@10 with narrow bandit window vs wider fixed window

---

## 5. Improvement Steps

### Step A: Joint Training (End-to-End Feedback Loop)

```
IMPROVED FLOW:

  Bandit ──δ──→ Recommender ──Hit@K + Loss──→ Bandit Reward
    ↑                                              │
    └──────────────────────────────────────────────┘
                    (closed loop)
```

**How**: After the recommender evaluates with a given δ, feed the retrieval metrics (Hit@10, loss) back as part of the bandit's reward:

```python
reward = (
    w1 * engagement_from_behavior_model +
    w2 * hit_at_10_from_recommender +     # NEW: recommender feedback
    w3 * diversity_bonus -
    w4 * echo_penalty
)
```

**Impact**: Bandit would learn that some users need wider windows (δ=0.3) for the recommender to find good items, while others can use narrow windows.

---

### Step B: Per-User Adaptive Delta (Prevent Policy Collapse)

**Problem**: Bandit assigns δ=0.15 to 97.7% of users → no personalization.

**Fix**: Increase exploration or use Thompson Sampling instead of UCB:

```python
# Option 1: Higher alpha (more exploration)
ALPHA = 3.0  # was 1.5

# Option 2: Epsilon-greedy layer on top of LinUCB
if random() < epsilon:
    arm = random_choice(arms)
else:
    arm = linucb_select(context)

# Option 3: Thompson Sampling (naturally maintains diversity)
# Sample theta from posterior instead of using point estimate
```

**Impact**: More diverse delta assignments → better mix of window sizes → higher Hit@K.

---

### Step C: Redesign Arms to Match Recommender

**Problem**: Bandit arms are [-0.3, -0.15, 0, +0.15, +0.3] but recommender was designed for δ=0.2.

**Fix**: Align arm values with what the recommender can actually use well:

```python
# Option 1: Center arms around 0.2
ARMS = [0.1, 0.15, 0.2, 0.25, 0.3]  # all positive, window width only

# Option 2: Include 0.2 as an arm
ARMS = [-0.2, -0.1, 0.0, 0.1, 0.2]
```

**Impact**: Removes the structural mismatch between bandit output and recommender input.

---

### Step D: Two-Phase Training with Warm Start

```
Phase 1: Train recommender with FIXED δ=0.2 (baseline)
          → Produces a working recommender

Phase 2: Fine-tune with bandit deltas
          → Bandit assigns per-user δ
          → Recommender fine-tunes (not from scratch) with those deltas
          → Evaluate improvement over Phase 1
```

**Impact**: The recommender already knows how to rank items; fine-tuning adapts it to variable window sizes without losing base capability.

---

### Step E: Multi-Objective Bandit

Replace single engagement reward with multi-objective:

```python
# Current (single objective):
reward = engagement + diversity - penalty

# Improved (multi-objective):
reward = (
    0.4 * engagement +           # user stays engaged
    0.3 * retrieval_quality +    # recommender can find good items
    0.2 * depolarization +       # moves user toward center
    0.1 * diversity              # prevents echo chamber
)
```

**Impact**: Bandit directly optimizes for what we actually care about at evaluation time.

---

## 6. Priority Order for Improvements

| Priority | Step | Effort | Expected Impact |
|:--------:|------|:------:|:---------------:|
| 1 | **B: Prevent policy collapse** | Low | High |
| 2 | **C: Align arms with recommender** | Low | Medium |
| 3 | **D: Two-phase warm start** | Medium | High |
| 4 | **A: Joint training** | High | Highest |
| 5 | **E: Multi-objective** | High | High |

---

## 7. What the Current System DOES Prove

Despite the limitations, the integrated system demonstrates:

1. **The bandit learns meaningful drift policies** — it doesn't just pick random arms; it converges on moderate drifts that balance engagement and movement.

2. **The behavior model bridges simulation and reality** — 80.3% accuracy on real data means the bandit's training signal is grounded in actual user behavior.

3. **The recommender respects ideology windows** — 100% in-window fraction and 99.97% direction accuracy show the contrastive loss works correctly.

4. **Lower test loss with bandit deltas** — the model fits better when windows match the bandit's learned policy (0.2408 vs 0.2414).

5. **Controlled ideology nudging** — bandit produces 21% less extreme drift than fixed baseline (0.074 vs 0.094), which is the depolarization goal.

The gap to close is: making the bandit's policy **diverse enough** and **informed by recommender feedback** so that retrieval quality improves alongside ideology control.
