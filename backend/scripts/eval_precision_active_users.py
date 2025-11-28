# backend/scripts/eval_precision_active_users.py
import numpy as np, pandas as pd
from tqdm import tqdm
from backend.ml.recommender_hybrid import HybridRecommender
from backend.core.db_utils import ENGINE

K = 10
MIN_RATINGS = 5
REL_THRESHOLD = 4

def load_ratings():
    df = pd.read_sql("SELECT user_id, book_id, rating FROM ratings", ENGINE)
    return df

def main():
    ratings = load_ratings()
    # filter active users
    user_counts = ratings.groupby("user_id").size()
    active_users = user_counts[user_counts >= MIN_RATINGS].index
    ratings = ratings[ratings["user_id"].isin(active_users)]
    print(f"[INFO] Active users: {len(active_users):,}")

    # simple 80/20 split per user
    train, test = [], []
    for uid, g in ratings.groupby("user_id"):
        g = g.sample(frac=1, random_state=42)
        cut = max(1, int(0.8 * len(g)))
        train.append(g.iloc[:cut])
        test.append(g.iloc[cut:])
    train = pd.concat(train); test = pd.concat(test)
    test_pos = test[test["rating"] >= REL_THRESHOLD]

    hybrid = HybridRecommender()
    train_items = train.groupby("user_id")["book_id"].apply(set).to_dict()
    test_items = test_pos.groupby("user_id")["book_id"].apply(set).to_dict()

    precisions = []
    fallback = 0
    for uid in tqdm(test_items.keys()):
        rec = hybrid.recommend("", int(uid), top_k=K*2)
        if "[WARN]" in str(rec):  # rough count fallback users
            fallback += 1
        rec_ids = [int(b) for b in rec["book_id"] if b not in train_items.get(uid, set())][:K]
        hits = sum(b in test_items[uid] for b in rec_ids)
        precisions.append(hits / K)

    macro_p = np.mean(precisions)
    print(f"\n=== Precision@{K} (Active users) ===")
    print(f"Users evaluated: {len(precisions):,}")
    print(f"Macro Precision@{K}: {macro_p:.3f}")
    print(f"Semantic-only fallbacks: {fallback}")
if __name__ == "__main__":
    main()
