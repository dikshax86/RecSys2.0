"""
Data-Driven User Behavior Model for Bandit Simulation.

Learns P(engage | user_state, proposed_drift) from REAL retweet sequences.

Instead of a simple exponential decay (old approach), this model captures:
  - Non-linear engagement patterns learned from actual user behavior
  - Asymmetric responses (liberal vs conservative react differently to drifts)
  - Trajectory-dependent openness (users who've been drifting may continue)
  - Position-dependent sensitivity (extreme users differ from centrists)

Architecture:
  Input (8 features) --> MLP --> P(engage)

Training data comes from scored_rt_sequences.pkl + user_ideology_states.pkl:
  - Positive: real (user, timestep) = user DID engage with content at that ideology
  - Negative: sampled alternative drifts the user did NOT take
"""

import torch
import torch.nn as nn
import numpy as np


class UserBehaviorModel(nn.Module):
    """
    MLP that predicts engagement probability given user state + proposed drift.

    Input features (8-dim):
        0: user_ideology       - current ideology position (normalized)
        1: proposed_drift       - how far the content is from user (signed)
        2: abs_drift            - magnitude of proposed drift
        3: abs_ideology         - how extreme the user is
        4: recent_mean          - mean of last K ideology states
        5: recent_std           - std of last K ideology states (volatility)
        6: recent_trend         - directional momentum (last - first of window)
        7: seq_progress         - position in sequence (0=start, 1=end)

    Output: P(engage) in [0, 1]
    """

    INPUT_DIM = 8

    def __init__(self, hidden_dims=(128, 64, 32), dropout=0.2):
        super().__init__()

        layers = []
        in_dim = self.INPUT_DIM
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 8) feature tensor
        Returns: (B,) engagement logits (apply sigmoid for probability)
        """
        return self.net(x).squeeze(-1)

    def predict_proba(self, x):
        """Return P(engage) after sigmoid."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class TransitionModel(nn.Module):
    """
    Predicts BOTH engagement probability AND next ideology state.

    This gives the bandit environment two things:
      1. Whether the user engages (replaces Bernoulli with learned patterns)
      2. Where the user moves if they engage (replaces fixed 0.3 move_rate)

    Architecture: shared backbone + two heads.
    """

    INPUT_DIM = 8

    def __init__(self, hidden_dims=(128, 64, 32), dropout=0.2):
        super().__init__()

        # Shared backbone
        backbone_layers = []
        in_dim = self.INPUT_DIM
        for h_dim in hidden_dims[:-1]:
            backbone_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.backbone = nn.Sequential(*backbone_layers)

        last_hidden = hidden_dims[-1]

        # Engagement head: backbone_out -> hidden -> logit
        self.engage_head = nn.Sequential(
            nn.Linear(in_dim, last_hidden),
            nn.GELU(),
            nn.Linear(last_hidden, 1),
        )

        # Transition head: backbone_out -> hidden -> delta_ideology
        # Predicts how much ideology shifts if user engages
        self.transition_head = nn.Sequential(
            nn.Linear(in_dim, last_hidden),
            nn.GELU(),
            nn.Linear(last_hidden, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 8)
        Returns: engage_logits (B,), predicted_ideology_delta (B,)
        """
        h = self.backbone(x)
        engage_logits = self.engage_head(h).squeeze(-1)
        ideo_delta = self.transition_head(h).squeeze(-1)
        return engage_logits, ideo_delta

    def predict(self, x):
        """
        Returns: P(engage), predicted_ideology_shift
        """
        with torch.no_grad():
            logits, delta = self.forward(x)
            return torch.sigmoid(logits), delta


def build_features(user_ideology, proposed_drift, recent_states, seq_progress):
    """
    Build the 8-dim feature vector for the behavior model.

    Args:
        user_ideology: current ideology state (float)
        proposed_drift: content_ideology - user_ideology (float)
        recent_states: list of recent ideology states (last K)
        seq_progress: position in sequence, 0.0 to 1.0

    Returns: numpy array of shape (8,)
    """
    recent = np.array(recent_states) if recent_states else np.array([user_ideology])

    recent_mean = float(np.mean(recent))
    recent_std = float(np.std(recent)) if len(recent) > 1 else 0.0
    recent_trend = float(recent[-1] - recent[0]) if len(recent) > 1 else 0.0

    features = np.array([
        user_ideology / 3.0,           # normalized to ~[-1, 1]
        proposed_drift / 3.0,          # normalized
        abs(proposed_drift) / 3.0,     # magnitude
        abs(user_ideology) / 3.0,      # extremity
        recent_mean / 3.0,             # recent average
        recent_std,                     # volatility (already small)
        recent_trend / 3.0,            # momentum
        seq_progress,                   # 0 to 1
    ], dtype=np.float32)

    return features
