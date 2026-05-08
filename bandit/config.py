"""
Configuration for Sequential Narrative Bridge Bandit Simulation.
All hyperparameters and paths in one place.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Dataset file paths
POLARITY_FILE = os.path.join(DATA_DIR, "USER_POLARITY_BARBERA.txt")
TWEETS_FILE = os.path.join(DATA_DIR, "USER_TWEETS.txt")
FOLLOWER_FILE = os.path.join(DATA_DIR, "FULL_FOLLOWER_NETWORK.txt")
FRIEND_FILE = os.path.join(DATA_DIR, "FULL_FRIEND_NETWORK.txt")

# Tweet file columns (0-indexed)
TWEET_COL_USER_ID = 0
TWEET_COL_TWEET_ID = 1
TWEET_COL_TEXT = 2
TWEET_COL_TIMESTAMP = 3
TWEET_COL_SOURCE = 4
TWEET_COL_RETWEET_COUNT = 5
TWEET_COL_IS_RT_FLAG = 6
TWEET_COL_FAV_COUNT = 7
TWEET_COL_HASHTAGS = 8
TWEET_COL_URLS = 9
TWEET_COL_MENTIONS = 10
TWEET_COL_IS_REPLY = 11

# Bandit arms: ideological shifts
# Arms must be within [-TAU, +TAU] so they aren't all clamped to the same value.
# Gives the bandit meaningful granularity: no-shift, small, medium, and full-TAU shifts.
ARMS = [-0.3, -0.15, 0.0, 0.15, 0.3]
NUM_ARMS = len(ARMS)

# LinUCB
ALPHA = 1.5            # exploration parameter (raised: force more exploration before converging)
FEATURE_DIM = 10       # context vector dimension

# Narrative Bridge
TAU = 0.3              # max allowed ideological shift per step (safety fence)

# Simulation
STEPS_PER_USER = 20    # T: number of recommendation rounds per user
IDEOLOGY_RANGE = (-3.0, 3.0)

# Reward function (redesigned for depolarization):
# R = REWARD_ENGAGE * engaged
#   + REWARD_DIVERSITY * |delta| * engaged        (bonus for diverse engagement)
#   + REWARD_DEPOLARIZE * toward_center * engaged  (bonus for moving user toward center)
#   - REWARD_ECHO_PENALTY * (delta==0) * engaged   (penalty for echo chamber)
#   - REWARD_PENALTY * max(0, |delta| - TAU)^2     (safety: penalize overshoot)
REWARD_ENGAGE = 1.0          # base engagement reward
REWARD_DIVERSITY = 2.0       # diversity bonus (raised from 0.3: makes non-zero delta competitive)
REWARD_DEPOLARIZE = 1.5      # bonus when user moves toward center after engaging
REWARD_ECHO_PENALTY = 0.5    # penalty for recommending delta=0 (discourages echo chamber)
REWARD_PENALTY = 0.05        # overshoot penalty (lowered: less fear of non-zero arms)

# Tweet pool
TWEET_POOL_SIZE = 200_000       # max tweets to keep in pool
IDEOLOGY_BIN_WIDTH = 0.1        # bin width for ideology indexing

# Engagement model
ENGAGEMENT_NEG_SAMPLE_RATIO = 3  # negative samples per positive

# Processing
CHUNK_SIZE = 100_000             # lines to process at a time for large files

# Behavior Model (data-driven engagement simulation)
BEHAVIOR_MODEL_PATH = os.path.join(OUTPUT_DIR, "behavior_model.pt")
SCORED_RT_PATH = os.path.join(BASE_DIR, "scored_rt_sequences.pkl")
IDEOLOGY_STATES_PATH = os.path.join(BASE_DIR, "..", "RecSys", "data", "processed", "user_ideology_states.pkl")
IDEOLOGY_MAP_PATH = os.path.join(BASE_DIR, "..", "RecSys", "data", "processed", "ideology_map.pkl")
BEHAVIOR_MODEL_EPOCHS = 30
BEHAVIOR_MODEL_BATCH_SIZE = 2048
BEHAVIOR_MODEL_NEG_RATIO = 3
BEHAVIOR_MODEL_MAX_SAMPLES_PER_USER = 100
BEHAVIOR_MODEL_DEVICE = "auto"
BEHAVIOR_MODEL_HISTORY_WINDOW = 10   # recent states for context features
