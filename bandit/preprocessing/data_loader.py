"""
Data Loader: Parses all 4 raw dataset files.

What happens behind the scenes:
1. load_polarity() - reads 8164 lines, builds {user_id: ideology_score} dict
2. load_tweets_streaming() - streams 39.5M tweets line by line (12GB file),
   extracts interactions (retweets/mentions) and builds tweet pool.
   Only keeps tweets from users who have polarity scores.
2b. build_tweet_data_from_pkl() - ALTERNATIVE when 12GB tweet file is not
    available. Builds tweet pool from scored_rt_sequences.pkl + ideology_map.pkl.
3. load_network_for_polarity_users() - streams 79M + 39M edges,
   keeps only edges where BOTH nodes are polarity users.
"""
import os
import sys
import re
import time
import random
import pickle
from collections import defaultdict

sys.path.insert(0, ".")
import config


def load_polarity():
    """
    Load USER_POLARITY_BARBERA.txt → {user_id(int): ideology_score(float)}

    Behind the scenes: 8164 lines, tab-separated, ~160KB file.
    Output: dict with 8164 entries.
    """
    polarity = {}
    with open(config.POLARITY_FILE, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    uid = int(parts[0])
                    score = float(parts[1])
                    polarity[uid] = score
                except ValueError:
                    continue
    print(f"  Loaded {len(polarity)} users with polarity scores")
    print(f"  Ideology range: [{min(polarity.values()):.3f}, {max(polarity.values()):.3f}]")
    return polarity


def load_tweets_streaming(polarity_users):
    """
    Stream USER_TWEETS.txt (12GB, 39.5M lines) and extract:
    1. tweet_pool: list of dicts with tweet metadata + ideology score
    2. interactions: list of (user_id, mentioned_user_id, engaged=True) for retweets
    3. user_tweet_counts: {user_id: total_tweet_count}
    4. user_retweet_counts: {user_id: count_of_retweets_they_made}

    Behind the scenes:
    - Streams line by line (never loads full 12GB into memory)
    - For each tweet: parse 14 tab-separated columns
    - Detect retweets via "RT @username:" pattern in tweet text
    - If retweeted user also has polarity → record as interaction
    - Keep original tweets (non-RT) for tweet pool
    - Sample tweet pool down to TWEET_POOL_SIZE for memory
    """
    polarity_set = set(polarity_users.keys())

    tweet_pool = []          # candidate tweets for recommendation
    interactions = []        # (user_id, target_user_id) retweet interactions
    user_tweet_counts = defaultdict(int)
    user_retweet_counts = defaultdict(int)
    user_mention_targets = defaultdict(set)  # who each user mentions

    # username → user_id mapping (we'll build this from tweets)
    # Since we don't have it, we use mentions column + polarity user IDs

    rt_pattern = re.compile(r"^RT @(\w+):")

    line_count = 0
    kept_tweets = 0
    found_interactions = 0
    skipped_non_polarity = 0

    start_time = time.time()

    print(f"  Streaming {config.TWEETS_FILE}...")
    print(f"  (This processes 39.5M tweets from a 12GB file - will take ~10-15 min)")

    with open(config.TWEETS_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1

            if line_count % 5_000_000 == 0:
                elapsed = time.time() - start_time
                rate = line_count / elapsed
                est_total = 39_500_000 / rate
                print(f"    Processed {line_count/1e6:.1f}M tweets "
                      f"({elapsed:.0f}s elapsed, ~{est_total - elapsed:.0f}s remaining) | "
                      f"pool={kept_tweets}, interactions={found_interactions}")

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 12:
                continue

            try:
                user_id = int(parts[config.TWEET_COL_USER_ID])
            except ValueError:
                continue

            # Only process tweets from polarity users
            if user_id not in polarity_set:
                skipped_non_polarity += 1
                continue

            user_tweet_counts[user_id] += 1
            tweet_text = parts[config.TWEET_COL_TEXT]

            try:
                retweet_count = int(parts[config.TWEET_COL_RETWEET_COUNT])
            except ValueError:
                retweet_count = 0

            # Detect retweet: "RT @username: ..."
            rt_match = rt_pattern.match(tweet_text)
            is_retweet = rt_match is not None

            if is_retweet:
                user_retweet_counts[user_id] += 1
                rt_username = rt_match.group(1).lower()
                # Record mention target
                user_mention_targets[user_id].add(rt_username)

            # Extract mentions from column 11
            mentions_raw = parts[config.TWEET_COL_MENTIONS].strip()
            if mentions_raw:
                for m in mentions_raw.split():
                    m_clean = m.strip().lower()
                    if m_clean:
                        user_mention_targets[user_id].add(m_clean)

            # Keep non-RT original tweets for the tweet pool
            if not is_retweet:
                tweet_id = parts[config.TWEET_COL_TWEET_ID]
                tweet_ideology = polarity_users[user_id]

                tweet_pool.append({
                    "tweet_id": tweet_id,
                    "user_id": user_id,
                    "ideology": tweet_ideology,
                    "retweet_count": retweet_count,
                    "text_snippet": tweet_text[:100],
                })
                kept_tweets += 1

    elapsed = time.time() - start_time
    print(f"  Done streaming tweets: {line_count} total lines in {elapsed:.1f}s")
    print(f"  Tweets from polarity users: {line_count - skipped_non_polarity}")
    print(f"  Original tweets (pool): {kept_tweets}")
    print(f"  Users who tweeted: {len(user_tweet_counts)}")

    # Now build interactions: we need to map usernames to user_ids
    # We'll do this by cross-referencing who retweeted whom
    # Since we only have usernames from RT @username, we need a username→uid map
    # We DON'T have this directly, so we use an alternative:
    # interactions based on network + ideology distance from retweet patterns

    # Sample tweet pool if too large
    if len(tweet_pool) > config.TWEET_POOL_SIZE:
        print(f"  Sampling tweet pool from {len(tweet_pool)} to {config.TWEET_POOL_SIZE}")
        tweet_pool = random.sample(tweet_pool, config.TWEET_POOL_SIZE)

    return {
        "tweet_pool": tweet_pool,
        "user_tweet_counts": dict(user_tweet_counts),
        "user_retweet_counts": dict(user_retweet_counts),
        "user_mention_targets": dict(user_mention_targets),
        "total_tweets_processed": line_count,
    }


def load_network_for_polarity_users(polarity_users, network_file, description="network"):
    """
    Stream a network file and keep only edges where BOTH nodes are polarity users.

    Behind the scenes:
    - FULL_FOLLOWER_NETWORK.txt: 79.2M lines → "follower_id \\t user_id"
    - FULL_FRIEND_NETWORK.txt: 39.3M lines → "user_id \\t friend_id"
    - Streams line by line, checks if both nodes in polarity_set
    - Returns adjacency dict: {user_id: set of connected user_ids}

    Output: adjacency dict (only among 8164 polarity users)
    """
    polarity_set = set(polarity_users.keys())
    adjacency = defaultdict(set)

    line_count = 0
    kept_edges = 0
    start_time = time.time()

    print(f"  Streaming {description}: {network_file}")

    with open(network_file, "r") as f:
        for line in f:
            line_count += 1

            if line_count % 10_000_000 == 0:
                elapsed = time.time() - start_time
                print(f"    Processed {line_count/1e6:.1f}M edges ({elapsed:.0f}s) | kept={kept_edges}")

            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            try:
                u1 = int(parts[0])
                u2 = int(parts[1])
            except ValueError:
                continue

            # Keep edge only if BOTH nodes are polarity users
            if u1 in polarity_set and u2 in polarity_set:
                adjacency[u1].add(u2)
                adjacency[u2].add(u1)
                kept_edges += 1

    elapsed = time.time() - start_time
    print(f"  Done: {line_count} edges scanned, {kept_edges} kept in {elapsed:.1f}s")
    print(f"  Users with connections: {len(adjacency)}")

    return dict(adjacency)


def load_all_networks(polarity_users):
    """Load both follower and friend networks, merge into one adjacency graph."""
    print("  Loading follower network...")
    follower_adj = load_network_for_polarity_users(
        polarity_users, config.FOLLOWER_FILE, "follower network"
    )

    print("  Loading friend network...")
    friend_adj = load_network_for_polarity_users(
        polarity_users, config.FRIEND_FILE, "friend network"
    )

    # Merge: union of both networks
    merged = defaultdict(set)
    for uid, neighbors in follower_adj.items():
        merged[uid].update(neighbors)
    for uid, neighbors in friend_adj.items():
        merged[uid].update(neighbors)

    merged = dict(merged)
    total_edges = sum(len(v) for v in merged.values()) // 2
    print(f"  Merged network: {len(merged)} users, ~{total_edges} undirected edges")

    return merged


def build_tweet_data_from_pkl(polarity, scored_rt_path=None, ideology_map_path=None):
    """
    Build tweet pool and user stats from .pkl files instead of the 12GB tweet file.

    This is the ALTERNATIVE to load_tweets_streaming() when USER_TWEETS.txt
    is not available (too large for local machine).

    Data sources:
      - scored_rt_sequences.pkl: {user_id: [(retweeted_user_id, ideology_score), ...]}
        Each entry = a real retweet interaction with the retweeted user's ideology.
      - ideology_map.pkl: {user_id: ideology_score}
        Used to assign ideology to tweet authors.

    What we construct:
      - tweet_pool: synthetic tweet entries from real retweet targets
        Each retweeted user becomes a "tweet" at their ideology position.
      - user_tweet_counts: estimated from sequence lengths
      - user_retweet_counts: from sequence lengths (all entries are retweets)

    The tweet pool captures the REAL ideology distribution of content
    that users actually interacted with.
    """
    scored_rt_path = scored_rt_path or config.SCORED_RT_PATH
    ideology_map_path = ideology_map_path or config.IDEOLOGY_MAP_PATH

    print(f"  Building tweet data from .pkl files (no 12GB file needed)")
    print(f"  Source: {scored_rt_path}")

    with open(scored_rt_path, "rb") as f:
        scored_rt_sequences = pickle.load(f)
    print(f"  Loaded scored_rt_sequences: {len(scored_rt_sequences)} users")

    ideology_map = {}
    if ideology_map_path and os.path.exists(ideology_map_path):
        with open(ideology_map_path, "rb") as f:
            ideology_map = pickle.load(f)
        print(f"  Loaded ideology_map: {len(ideology_map)} users")

    # Build tweet pool from retweet targets
    # Each (retweeted_user, ideology) becomes a tweet entry
    tweet_pool = []
    user_tweet_counts = defaultdict(int)
    user_retweet_counts = defaultdict(int)

    # Track unique retweeted users to avoid massive duplicates
    rt_user_counts = defaultdict(int)    # retweeted_user -> times retweeted
    rt_user_ideology = {}                # retweeted_user -> ideology

    polarity_int = {int(k) if not isinstance(k, int) else k: v for k, v in polarity.items()}

    for user_id_str, seq in scored_rt_sequences.items():
        # Convert user_id to int to match polarity keys
        try:
            user_id = int(user_id_str)
        except (ValueError, TypeError):
            user_id = user_id_str

        seq_len = len(seq)
        # Each entry in scored_rt_sequences is a retweet
        user_retweet_counts[user_id] = seq_len
        # Estimate total tweets as ~2x retweets (typical ratio from real data)
        user_tweet_counts[user_id] = int(seq_len * 2)

        for rt_user_id_str, ideo_score in seq:
            rt_user_counts[rt_user_id_str] += 1
            rt_user_ideology[rt_user_id_str] = ideo_score

    # Create tweet pool entries from retweeted users
    # Each unique retweeted user -> one "tweet" with their ideology
    # retweet_count = how many times they were retweeted (popularity proxy)
    tweet_id_counter = 0
    for rt_user_str, count in rt_user_counts.items():
        ideology = rt_user_ideology.get(rt_user_str, 0.0)

        # Clamp ideology to valid range
        ideology = max(config.IDEOLOGY_RANGE[0], min(config.IDEOLOGY_RANGE[1], ideology))

        try:
            rt_user_int = int(rt_user_str)
        except (ValueError, TypeError):
            rt_user_int = hash(rt_user_str) % (10**10)

        tweet_pool.append({
            "tweet_id": f"syn_{tweet_id_counter}",
            "user_id": rt_user_int,
            "ideology": ideology,
            "retweet_count": count,
            "text_snippet": f"[synthetic from pkl, author={rt_user_str}]",
        })
        tweet_id_counter += 1

    # Sample down if too large
    if len(tweet_pool) > config.TWEET_POOL_SIZE:
        print(f"  Sampling tweet pool from {len(tweet_pool)} to {config.TWEET_POOL_SIZE}")
        tweet_pool = random.sample(tweet_pool, config.TWEET_POOL_SIZE)

    print(f"  Built tweet pool: {len(tweet_pool)} tweets from {len(rt_user_counts)} unique retweeted users")
    print(f"  Users with tweet counts: {len(user_tweet_counts)}")

    # Show ideology distribution of pool
    ideologies = [t["ideology"] for t in tweet_pool]
    lib = sum(1 for i in ideologies if i < -0.5)
    cen = sum(1 for i in ideologies if -0.5 <= i <= 0.5)
    con = sum(1 for i in ideologies if i > 0.5)
    print(f"  Tweet pool ideology: liberal={lib}, center={cen}, conservative={con}")

    return {
        "tweet_pool": tweet_pool,
        "user_tweet_counts": dict(user_tweet_counts),
        "user_retweet_counts": dict(user_retweet_counts),
        "user_mention_targets": {},
        "total_tweets_processed": sum(len(v) for v in scored_rt_sequences.values()),
    }
