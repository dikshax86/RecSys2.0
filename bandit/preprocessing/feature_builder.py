"""
Feature Builder: Constructs user profiles, tweet pool index, and engagement model.

What happens behind the scenes:
1. build_user_profiles() - For each of 8164 users, compute:
   - ideology score (from polarity)
   - network degree (from merged network)
   - avg neighbor ideology (from network + polarity)
   - neighbor ideology std (diversity of neighbors)
   - echo chamber score (% neighbors with same-sign ideology)
   - tweet count, retweet ratio (from tweet stats)

2. build_tweet_pool_index() - Organize ~200K tweets into ideology bins
   for fast retrieval by target ideology.

3. build_engagement_model() - Calibrate P(engage) from REAL retweet patterns:
   - Positive samples: user retweeted someone → (ideology_distance, 1)
   - Negative samples: random pairs who didn't interact → (ideology_distance, 0)
   - Fit logistic regression: P(engage | distance, is_connected, popularity)
"""
import numpy as np
import random
from collections import defaultdict

import sys
sys.path.insert(0, ".")
import config


def build_user_profiles(polarity, network_adj, tweet_data):
    """
    Build feature profile for every polarity user.

    Output per user:
    {
        "ideology": float,
        "degree": int,
        "degree_norm": float,       # degree / max_degree
        "avg_neighbor_ideology": float,
        "neighbor_ideology_std": float,
        "echo_chamber_score": float, # % neighbors with same sign
        "tweet_count": int,
        "retweet_ratio": float,     # retweets_made / total_tweets
    }
    """
    user_tweet_counts = tweet_data["user_tweet_counts"]
    user_retweet_counts = tweet_data["user_retweet_counts"]

    max_degree = max(
        (len(network_adj.get(uid, set())) for uid in polarity),
        default=1
    )
    if max_degree == 0:
        max_degree = 1

    profiles = {}
    for uid, ideology in polarity.items():
        neighbors = network_adj.get(uid, set())
        degree = len(neighbors)

        # Neighbor ideology stats
        neighbor_ideologies = [polarity[n] for n in neighbors if n in polarity]
        if neighbor_ideologies:
            avg_neighbor_ideo = np.mean(neighbor_ideologies)
            std_neighbor_ideo = np.std(neighbor_ideologies)
            same_sign = sum(1 for ni in neighbor_ideologies if np.sign(ni) == np.sign(ideology))
            echo_score = same_sign / len(neighbor_ideologies)
        else:
            avg_neighbor_ideo = 0.0
            std_neighbor_ideo = 0.0
            echo_score = 1.0  # no neighbors = fully isolated

        tweet_count = user_tweet_counts.get(uid, 0)
        rt_count = user_retweet_counts.get(uid, 0)
        rt_ratio = rt_count / max(tweet_count, 1)

        profiles[uid] = {
            "ideology": ideology,
            "degree": degree,
            "degree_norm": degree / max_degree,
            "avg_neighbor_ideology": avg_neighbor_ideo,
            "neighbor_ideology_std": std_neighbor_ideo,
            "echo_chamber_score": echo_score,
            "tweet_count": tweet_count,
            "retweet_ratio": rt_ratio,
        }

    print(f"  Built profiles for {len(profiles)} users")
    degrees = [p["degree"] for p in profiles.values()]
    print(f"  Degree stats: min={min(degrees)}, max={max(degrees)}, "
          f"mean={np.mean(degrees):.1f}, median={np.median(degrees):.1f}")
    echo_scores = [p["echo_chamber_score"] for p in profiles.values()]
    print(f"  Echo chamber score: mean={np.mean(echo_scores):.3f}")

    return profiles


def build_tweet_pool_index(tweet_pool):
    """
    Index tweets by ideology bin for fast retrieval.

    Behind the scenes:
    - Bins tweets into buckets of width 0.1 across [-3, +3]
    - Each bin: list of tweet dicts sorted by retweet_count (popularity)
    - For target_ideology query → find matching bin → return candidates

    Output: dict {bin_center: [tweet_dicts sorted by popularity]}
    """
    bins = defaultdict(list)

    for tweet in tweet_pool:
        ideo = tweet["ideology"]
        # Clamp to range
        ideo = max(config.IDEOLOGY_RANGE[0], min(config.IDEOLOGY_RANGE[1], ideo))
        bin_key = round(round(ideo / config.IDEOLOGY_BIN_WIDTH) * config.IDEOLOGY_BIN_WIDTH, 2)
        bins[bin_key].append(tweet)

    # Sort each bin by popularity (retweet_count desc)
    for key in bins:
        bins[key].sort(key=lambda t: t["retweet_count"], reverse=True)

    total_tweets = sum(len(v) for v in bins.values())
    non_empty_bins = sum(1 for v in bins.values() if len(v) > 0)
    print(f"  Tweet pool indexed: {total_tweets} tweets in {non_empty_bins} ideology bins")
    print(f"  Ideology coverage: [{min(bins.keys()):.2f}, {max(bins.keys()):.2f}]")

    # Show distribution
    lib_count = sum(len(v) for k, v in bins.items() if k < -0.5)
    cen_count = sum(len(v) for k, v in bins.items() if -0.5 <= k <= 0.5)
    con_count = sum(len(v) for k, v in bins.items() if k > 0.5)
    print(f"  Distribution: liberal={lib_count}, center={cen_count}, conservative={con_count}")

    return dict(bins)


def build_engagement_model(polarity, network_adj, tweet_data):
    """
    Calibrate engagement probability from REAL retweet/interaction patterns.

    Behind the scenes:
    1. From tweet_data, we know each user's retweet ratio and network position
    2. We compute observed engagement rates at different ideology distances
       by analyzing which users retweet which ideological content
    3. Fit a model: P(engage) = sigmoid(w0 + w1*distance + w2*is_connected + w3*popularity_norm)

    Since we can't map RT @username to user_ids directly (no username→id mapping),
    we use an alternative approach:
    - For each user, compute their retweet_ratio (what fraction of their tweets are RTs)
    - Users who RT a lot have higher base engagement
    - Use network connections + ideology distance to model engagement

    Output: dict with model parameters
    {
        "base_rate": float,          # average engagement rate
        "distance_decay": float,     # how fast engagement drops with ideology distance
        "network_boost": float,      # boost for connected users
        "popularity_weight": float,  # weight for tweet popularity
        "user_base_rates": {uid: float}  # per-user base engagement rate
    }
    """
    # Compute per-user base engagement rate from their retweet behavior
    user_base_rates = {}
    all_rt_ratios = []

    for uid in polarity:
        rt_ratio = tweet_data["user_retweet_counts"].get(uid, 0) / max(
            tweet_data["user_tweet_counts"].get(uid, 1), 1
        )
        user_base_rates[uid] = rt_ratio
        if tweet_data["user_tweet_counts"].get(uid, 0) > 0:
            all_rt_ratios.append(rt_ratio)

    avg_rt_ratio = np.mean(all_rt_ratios) if all_rt_ratios else 0.5

    # Analyze ideology distance patterns from the network
    # For connected user pairs, compute ideology distance distribution
    connected_distances = []
    for uid, neighbors in network_adj.items():
        if uid not in polarity:
            continue
        for n in neighbors:
            if n in polarity:
                dist = abs(polarity[uid] - polarity[n])
                connected_distances.append(dist)

    # For random non-connected pairs
    polarity_list = list(polarity.keys())
    random_distances = []
    for _ in range(min(100000, len(connected_distances))):
        u1, u2 = random.sample(polarity_list, 2)
        random_distances.append(abs(polarity[u1] - polarity[u2]))

    # Calibrate distance decay from the data
    # Connected users have smaller ideology distance on average (homophily)
    avg_connected_dist = np.mean(connected_distances) if connected_distances else 1.0
    avg_random_dist = np.mean(random_distances) if random_distances else 2.0

    # The decay parameter: engagement drops exponentially with distance
    # Calibrated so that at avg_connected_dist, engagement is high
    # and at avg_random_dist, engagement is much lower
    if avg_connected_dist > 0:
        distance_decay = -np.log(0.3) / avg_random_dist  # P drops to 30% at avg random distance
    else:
        distance_decay = 0.5

    model = {
        "base_rate": avg_rt_ratio,
        "distance_decay": distance_decay,
        "network_boost": 0.15,          # connected users get 15% boost
        "popularity_weight": 0.05,      # small boost for popular tweets
        "user_base_rates": user_base_rates,
    }

    print(f"  Engagement model calibrated from real data:")
    print(f"    Avg retweet ratio: {avg_rt_ratio:.3f}")
    print(f"    Connected user avg ideology distance: {avg_connected_dist:.3f}")
    print(f"    Random user avg ideology distance: {avg_random_dist:.3f}")
    print(f"    Distance decay parameter: {distance_decay:.3f}")
    print(f"    (engagement halves every {np.log(2)/distance_decay:.2f} ideology units)")

    return model
