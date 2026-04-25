"""
LinUCB Contextual Bandit Implementation.

From paper Section 5.2:
  p_i = theta^T * x_i + alpha * sqrt(x_i^T * A^{-1} * x_i)

  - theta^T * x_i  = exploitation (predicted reward)
  - alpha * sqrt(.) = exploration bonus (curiosity bonus)

What happens behind the scenes:
- Maintains per-arm: A matrix (d x d), b vector (d x 1)
- On select: computes UCB score for each arm, picks highest
- On update: updates A and b for the chosen arm with observed reward
- A starts as identity matrix (maximum uncertainty)
- As more data comes in, A grows → A^{-1} shrinks → exploration bonus decreases
"""
import numpy as np

import sys
sys.path.insert(0, ".")
import config


class LinUCB:
    """
    LinUCB with disjoint models (separate parameters per arm).

    Each arm a has:
      A_a: (d x d) matrix = I + sum of x*x^T for all rounds where arm a was chosen
      b_a: (d x 1) vector = sum of r * x for all rounds where arm a was chosen
      theta_a: A_a^{-1} * b_a (learned parameters)
    """

    def __init__(self, n_arms=config.NUM_ARMS, d=config.FEATURE_DIM, alpha=config.ALPHA):
        self.n_arms = n_arms
        self.d = d
        self.alpha = alpha
        self.name = "LinUCB"

        # Per-arm parameters
        self.A = [np.eye(d) for _ in range(n_arms)]      # d x d identity matrices
        self.b = [np.zeros(d) for _ in range(n_arms)]     # d x 1 zero vectors

        # Tracking
        self.total_pulls = 0
        self.arm_pulls = [0] * n_arms

    def select_action(self, context):
        """
        Select arm with highest UCB score.

        Behind the scenes (for each arm):
        1. theta_a = A_a^{-1} * b_a  (solve linear system, O(d^2))
        2. exploitation = theta_a^T * x
        3. uncertainty = sqrt(x^T * A_a^{-1} * x)  (measures how "unknown" this context is for this arm)
        4. UCB = exploitation + alpha * uncertainty
        5. Pick arm with highest UCB

        Returns: arm_index, ucb_scores (for logging)
        """
        x = context.astype(np.float64)
        ucb_scores = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))  # A^{-1}
            theta_a = A_inv @ self.b[a]                          # learned params

            exploitation = theta_a @ x                           # predicted reward
            uncertainty = np.sqrt(x @ A_inv @ x)                # exploration bonus

            ucb_scores[a] = exploitation + self.alpha * uncertainty

        best_arm = np.argmax(ucb_scores)
        return best_arm, ucb_scores

    def update(self, context, arm_index, reward):
        """
        Update the chosen arm's parameters with observed reward.

        Behind the scenes:
        A_a ← A_a + x * x^T   (reduces uncertainty for this context)
        b_a ← b_a + r * x     (updates reward estimate)

        This is the learning step. After many updates:
        - A grows → A^{-1} shrinks → exploration bonus decreases for seen contexts
        - theta converges to true reward parameters
        """
        x = context.astype(np.float64)
        a = arm_index

        self.A[a] += np.outer(x, x)  # rank-1 update
        self.b[a] += reward * x

        self.total_pulls += 1
        self.arm_pulls[a] += 1

    def get_arm_stats(self):
        """Return per-arm pull counts and learned theta vectors."""
        stats = {}
        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta = A_inv @ self.b[a]
            stats[a] = {
                "arm_value": config.ARMS[a],
                "pulls": self.arm_pulls[a],
                "pull_fraction": self.arm_pulls[a] / max(self.total_pulls, 1),
                "theta_norm": np.linalg.norm(theta),
            }
        return stats


class RandomPolicy:
    """
    Baseline: selects arms uniformly at random.
    No learning, no context awareness.
    Used as a lower bound for comparison.
    """

    def __init__(self, n_arms=config.NUM_ARMS):
        self.n_arms = n_arms
        self.name = "Random"
        self.total_pulls = 0
        self.arm_pulls = [0] * n_arms

    def select_action(self, context):
        arm = np.random.randint(self.n_arms)
        scores = np.zeros(self.n_arms)
        scores[arm] = 1.0
        return arm, scores

    def update(self, context, arm_index, reward):
        self.total_pulls += 1
        self.arm_pulls[arm_index] += 1

    def get_arm_stats(self):
        stats = {}
        for a in range(self.n_arms):
            stats[a] = {
                "arm_value": config.ARMS[a],
                "pulls": self.arm_pulls[a],
                "pull_fraction": self.arm_pulls[a] / max(self.total_pulls, 1),
            }
        return stats


class GreedyPolicy:
    """
    Baseline: always selects delta=0 (no shift).
    This simulates an echo chamber recommender that only shows
    content matching the user's current ideology.
    Used to measure how much polarization increases without intervention.
    """

    def __init__(self, n_arms=config.NUM_ARMS):
        self.n_arms = n_arms
        self.name = "Greedy(Echo)"
        self.total_pulls = 0
        self.arm_pulls = [0] * n_arms
        # Find the index of delta=0
        self.zero_arm = config.ARMS.index(0.0)

    def select_action(self, context):
        scores = np.zeros(self.n_arms)
        scores[self.zero_arm] = 1.0
        return self.zero_arm, scores

    def update(self, context, arm_index, reward):
        self.total_pulls += 1
        self.arm_pulls[arm_index] += 1

    def get_arm_stats(self):
        stats = {}
        for a in range(self.n_arms):
            stats[a] = {
                "arm_value": config.ARMS[a],
                "pulls": self.arm_pulls[a],
                "pull_fraction": self.arm_pulls[a] / max(self.total_pulls, 1),
            }
        return stats
