"""
Simulation Environment: Gym-style bandit environment.

Two modes:
  1. MODEL-BASED (new): Uses a trained TransitionModel to predict engagement
     and ideology shifts from REAL behavioral patterns learned from .pkl data.
  2. HEURISTIC (legacy): Simple exponential decay engagement model.

The model-based mode replaces the Bernoulli coin flip with a neural network
that learned P(engage | user_state, drift) from 8158 users x ~1000 interactions.
"""
import numpy as np
import random

import torch

import sys
sys.path.insert(0, ".")
import config


class BanditEnvironment:
    """
    Simulates the recommendation process for one user across T steps.

    State: context vector describing the user's current situation
    Action: arm index -> maps to ideology shift delta
    Reward: engagement + diversity bonus - penalty

    When behavior_model is provided, engagement is predicted by the trained
    neural network instead of the simple heuristic.
    """

    def __init__(self, user_profiles, tweet_pool_index, engagement_model,
                 network_adj, polarity, behavior_model=None, device="cpu"):
        self.user_profiles = user_profiles
        self.tweet_pool_index = tweet_pool_index
        self.engagement_model = engagement_model
        self.network_adj = network_adj
        self.polarity = polarity
        self.arms = config.ARMS

        # Data-driven behavior model (optional)
        self.behavior_model = behavior_model
        self.device = device
        self.use_learned_model = behavior_model is not None

        # Precompute max popularity for normalization
        max_pop = 1
        for tweets in tweet_pool_index.values():
            for t in tweets:
                if t["retweet_count"] > max_pop:
                    max_pop = t["retweet_count"]
        self.max_popularity = max_pop

    def reset(self, user_id):
        """
        Start a new session for the given user.
        Returns: initial context vector (numpy array of shape [FEATURE_DIM])
        """
        self.user_id = user_id
        profile = self.user_profiles[user_id]

        # User's ideology position (evolves during session)
        self.current_ideology = profile["ideology"]
        self.initial_ideology = profile["ideology"]

        # Session tracking
        self.step = 0
        self.total_steps = config.STEPS_PER_USER
        self.engagement_history = []  # list of 0/1
        self.delta_history = []       # list of deltas attempted
        self.ideology_trajectory = [self.current_ideology]

        # Session log (for output)
        self.session_log = []

        return self._build_context()

    def _build_context(self):
        """
        Build the context vector x for LinUCB.
        10 dimensions:
          0: user_ideology (normalized to [-1, 1])
          1: degree_norm
          2: avg_neighbor_ideology (normalized)
          3: neighbor_ideology_std
          4: echo_chamber_score
          5: retweet_ratio (user's base engagement rate)
          6: session_progress (step / total_steps)
          7: recent_engagement_rate (last 5 steps)
          8: avg_abs_delta_accepted (how much shift user has accepted so far)
          9: bias term (always 1.0)
        """
        profile = self.user_profiles[self.user_id]

        # Normalize ideology to [-1, 1]
        ideo_norm = self.current_ideology / 3.0  # since range is [-3, 3]

        # Recent engagement rate (last 5 steps)
        recent = self.engagement_history[-5:] if self.engagement_history else []
        recent_engage_rate = np.mean(recent) if recent else 0.5

        # Average accepted delta
        accepted_deltas = [
            self.delta_history[i]
            for i in range(len(self.engagement_history))
            if self.engagement_history[i] == 1
        ]
        avg_accepted_delta = np.mean([abs(d) for d in accepted_deltas]) if accepted_deltas else 0.0

        context = np.array([
            ideo_norm,
            profile["degree_norm"],
            profile["avg_neighbor_ideology"] / 3.0,
            profile["neighbor_ideology_std"],
            profile["echo_chamber_score"],
            profile["retweet_ratio"],
            self.step / max(self.total_steps, 1),
            recent_engage_rate,
            avg_accepted_delta,
            1.0,  # bias
        ], dtype=np.float64)

        return context

    def step_action(self, arm_index):
        """
        Execute one recommendation step.

        Input: arm_index (0-4) -> maps to delta from config.ARMS
        Returns: (next_context, reward, done, info)

        Behind the scenes:
        1. Compute target ideology = current + delta
        2. Apply narrative bridge constraint (tau)
        3. Find best matching tweet from pool
        4. Simulate engagement:
           - MODEL-BASED: TransitionModel predicts P(engage) + ideology shift
           - HEURISTIC: simple exponential decay + Bernoulli
        5. Compute reward
        6. Update user's ideology if engaged
        """
        delta = self.arms[arm_index]
        self.delta_history.append(delta)

        # Target ideology
        target_ideology = self.current_ideology + delta

        # Clamp to valid range
        target_ideology = max(config.IDEOLOGY_RANGE[0], min(config.IDEOLOGY_RANGE[1], target_ideology))

        # Narrative bridge constraint: enforce |shift| <= tau
        actual_delta = target_ideology - self.current_ideology
        if abs(actual_delta) > config.TAU:
            actual_delta = np.sign(actual_delta) * config.TAU
            target_ideology = self.current_ideology + actual_delta

        # Find matching tweet from pool
        tweet = self._find_tweet(target_ideology)

        if tweet is None:
            engaged = 0
            tweet_ideology = target_ideology
            tweet_id = "none"
            tweet_popularity = 0
            predicted_shift = 0.0
        else:
            tweet_ideology = tweet["ideology"]
            tweet_id = tweet["tweet_id"]
            tweet_popularity = tweet["retweet_count"]

            # Simulate engagement
            if self.use_learned_model:
                engaged, predicted_shift = self._simulate_engagement_model(tweet)
            else:
                engaged = self._simulate_engagement_heuristic(tweet)
                predicted_shift = None

        # Compute reward
        reward = self._compute_reward(engaged, actual_delta)

        # Update user state
        if engaged:
            if self.use_learned_model and predicted_shift is not None:
                # Use model-predicted ideology shift (learned from real transitions)
                # Clamp to reasonable range to prevent instability
                clamped_shift = np.clip(predicted_shift, -config.TAU, config.TAU)
                self.current_ideology += clamped_shift
            else:
                # Legacy: fixed move rate
                move_rate = 0.3
                self.current_ideology += move_rate * (tweet_ideology - self.current_ideology)

            # Clamp ideology to valid range
            self.current_ideology = np.clip(
                self.current_ideology, config.IDEOLOGY_RANGE[0], config.IDEOLOGY_RANGE[1]
            )

        self.engagement_history.append(engaged)
        self.ideology_trajectory.append(self.current_ideology)
        self.step += 1

        # Log this step
        self.session_log.append({
            "user_id": self.user_id,
            "step": self.step - 1,
            "psi_user": self.ideology_trajectory[-2],  # before update
            "delta_selected": delta,
            "actual_delta": actual_delta,
            "psi_target": target_ideology,
            "tweet_id": tweet_id,
            "psi_tweet": tweet_ideology,
            "tweet_popularity": tweet_popularity,
            "engaged": engaged,
            "reward": reward,
            "psi_user_after": self.current_ideology,
            "model_based": self.use_learned_model,
        })

        done = self.step >= self.total_steps
        next_context = self._build_context()

        info = {
            "engaged": engaged,
            "delta": actual_delta,
            "tweet_ideology": tweet_ideology,
            "ideology_before": self.ideology_trajectory[-2],
            "ideology_after": self.current_ideology,
        }

        return next_context, reward, done, info

    def _find_tweet(self, target_ideology):
        """
        Find a tweet near the target ideology from the indexed pool.

        Behind the scenes:
        - Look in the target bin and adjacent bins
        - Apply narrative bridge constraint: |tweet_ideo - current_ideo| <= tau
        - Return a tweet (weighted random from top candidates for variety)
        """
        target_bin = round(round(target_ideology / config.IDEOLOGY_BIN_WIDTH) * config.IDEOLOGY_BIN_WIDTH, 2)

        candidates = []
        for offset in [0, -config.IDEOLOGY_BIN_WIDTH, config.IDEOLOGY_BIN_WIDTH,
                        -2*config.IDEOLOGY_BIN_WIDTH, 2*config.IDEOLOGY_BIN_WIDTH]:
            bin_key = round(target_bin + offset, 2)
            bin_tweets = self.tweet_pool_index.get(bin_key, [])
            for t in bin_tweets[:50]:
                if abs(t["ideology"] - self.current_ideology) <= config.TAU + 0.1:
                    candidates.append(t)

        if not candidates:
            for offset_mult in range(3, 10):
                for sign in [-1, 1]:
                    bin_key = round(target_bin + sign * offset_mult * config.IDEOLOGY_BIN_WIDTH, 2)
                    bin_tweets = self.tweet_pool_index.get(bin_key, [])
                    for t in bin_tweets[:20]:
                        if abs(t["ideology"] - self.current_ideology) <= config.TAU + 0.3:
                            candidates.append(t)
                if candidates:
                    break

        if not candidates:
            return None

        weights = [max(1, t["retweet_count"]) for t in candidates]
        total = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(candidates, weights=weights, k=1)[0]

    def _simulate_engagement_model(self, tweet):
        """
        MODEL-BASED engagement simulation using the trained TransitionModel.

        The model was trained on real retweet sequences to learn:
          1. P(engage | user_state, drift) — from actual user behavior
          2. predicted_ideology_shift — from actual ideology transitions

        This replaces the simple Bernoulli with patterns learned from
        8158 users x ~1000 real interactions each.
        """
        from models.behavior_model import build_features

        # Build feature vector matching training format
        drift = tweet["ideology"] - self.current_ideology
        recent_states = self.ideology_trajectory[-(config.BEHAVIOR_MODEL_HISTORY_WINDOW):]
        seq_progress = self.step / max(self.total_steps, 1)

        features = build_features(self.current_ideology, drift, recent_states, seq_progress)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Model prediction
        p_engage, pred_shift = self.behavior_model.predict(feat_tensor)
        p_engage = float(p_engage.item())
        pred_shift = float(pred_shift.item())

        # Clamp probability to avoid degenerate behavior
        p_engage = max(0.05, min(0.95, p_engage))

        # Bernoulli sample with learned probability
        engaged = 1 if random.random() < p_engage else 0

        return engaged, pred_shift if engaged else 0.0

    def _simulate_engagement_heuristic(self, tweet):
        """
        HEURISTIC (legacy) engagement simulation.
        P(engage) = base_rate * exp(-decay * |distance|) * network_boost * popularity_boost
        """
        model = self.engagement_model
        profile = self.user_profiles[self.user_id]

        ideology_distance = abs(tweet["ideology"] - self.current_ideology)

        user_base = model["user_base_rates"].get(self.user_id, model["base_rate"])
        user_base = 0.2 + 0.6 * min(user_base / max(model["base_rate"] * 2, 0.01), 1.0)

        distance_factor = np.exp(-model["distance_decay"] * ideology_distance)

        is_connected = tweet["user_id"] in self.network_adj.get(self.user_id, set())
        network_factor = 1.0 + model["network_boost"] * float(is_connected)

        pop_norm = min(tweet["retweet_count"] / max(self.max_popularity, 1), 1.0)
        pop_factor = 1.0 + model["popularity_weight"] * pop_norm

        p_engage = user_base * distance_factor * network_factor * pop_factor
        p_engage = max(0.01, min(0.95, p_engage))

        engaged = 1 if random.random() < p_engage else 0
        return engaged

    def _compute_reward(self, engaged, delta):
        """
        Redesigned reward that balances engagement WITH depolarization.

        R = ENGAGE * engaged                              (base: reward engagement)
          + DIVERSITY * |delta| * engaged                  (bonus: engaged with shifted content)
          + DEPOLARIZE * toward_center * engaged           (bonus: user moved toward center)
          - ECHO_PENALTY * (delta==0) * engaged            (penalty: echo chamber behavior)
          - PENALTY * max(0, |delta| - tau)^2              (safety: penalize overshoot)

        The key insight: without the echo penalty and depolarization bonus,
        the bandit always converges to delta=0 (echo chamber) because engagement
        is highest for zero-shift content. The new terms make non-zero drifts
        competitive when they successfully move users toward the center.
        """
        r = config.REWARD_ENGAGE * engaged

        # Diversity bonus: reward engaging with ideologically shifted content
        r += config.REWARD_DIVERSITY * abs(delta) * engaged

        # Depolarization bonus: reward if user moved toward center
        if engaged and hasattr(self, 'initial_ideology'):
            # Check if current position is closer to center than starting position
            toward_center = 1.0 if abs(self.current_ideology) < abs(self.initial_ideology) else 0.0
            r += config.REWARD_DEPOLARIZE * toward_center * engaged

        # Echo chamber penalty: discourage always picking delta=0
        if abs(delta) < 0.01:
            r -= config.REWARD_ECHO_PENALTY * engaged

        # Overshoot penalty: safety fence for extreme shifts
        overshoot = max(0, abs(delta) - config.TAU)
        r -= config.REWARD_PENALTY * (overshoot ** 2)

        return r

    def get_session_log(self):
        """Return the full session log for this user."""
        return self.session_log

    def get_session_summary(self):
        """Return summary metrics for this user's session."""
        if not self.engagement_history:
            return {}
        return {
            "user_id": self.user_id,
            "initial_ideology": self.initial_ideology,
            "final_ideology": self.current_ideology,
            "ideology_shift": self.current_ideology - self.initial_ideology,
            "abs_ideology_shift": abs(self.current_ideology - self.initial_ideology),
            "engagement_rate": np.mean(self.engagement_history),
            "total_reward": sum(
                log["reward"] for log in self.session_log
            ),
            "steps": self.step,
            "ideology_trajectory": list(self.ideology_trajectory),
        }
