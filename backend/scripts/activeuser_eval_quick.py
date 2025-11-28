"""
Quick Standalone Evaluator — Active-User Precision@10 (ALS-only)
----------------------------------------------------------------
- Calculates Macro Precision@10 for *active* users (>= MIN_RATINGS total)
- Per-user 80/20 split (random seed fixed)
- Relevant test items: rating >= REL_THRESHOLD
- Excludes training items from recommendations
- Uses saved ALS artifacts + ID maps (no changes to your app)

Run:
  python -m backend.scripts.activeuser_eval_quick
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares

# ---- Project imports (adjust only if your paths differ)
from backend.core.config import ART_DIR
from backend.core.db_utils import ENGINE

# ---- Configurable parameters
K = 10
REL_THRESHOLD = 4.0
MIN_RATINGS = 5
RANDOM_SEED = 42


def load_artifacts():
    """Load ALS factors + ID maps. Fail fast if anything is missing."""
    uf_path = os.path.join(ART_DIR, "als_user_factors.npz")
    it_path = os.path.join(ART_DIR, "als_item_factors.npz")
    uid_map_path = os.path.join(ART_DIR, "als_uid_map.pkl")
    iid_map_path = os.path.join(ART_DIR, "als_iid_map.pkl")

    for p in [uf_path, it_path, uid_map_path, iid_map_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artifact: {p}")

    user_factors = np.load(uf_path)["data"]
    item_factors = np.load(it_path)["data"]

    with open(uid_map_path, "rb") as f:
        uid_map = pickle.load(f)
    with open(iid_map_path, "rb") as f:
        iid_map = pickle.load(f)

    # reverse maps (ALS index -> real id)
    rev_uid = {v: k for k, v in uid_map.items()}
    rev_iid = {v: k for k, v in iid_map.items()}

    # Build a tiny ALS model holder (no fitting). We just need .recommend()
    model = AlternatingLeastSquares(factors=user_factors.shape[1])
    model.user_factors = user_factors
    model.item_factors = item_factors

    print(f"[OK] ALS artifacts loaded: users={len(uid_map):,}, items={len(iid_map):,}")
    return model, uid_map, iid_map, rev_uid, rev_iid


def load_active_ratings():
    """Load ratings from DB and keep only active users (>= MIN_RATINGS)."""
    print("[INFO] Loading ratings from database ...")
    ratings = pd.read_sql("SELECT user_id, book_id, rating FROM ratings", ENGINE)

    # keep only active users
    user_counts = ratings.groupby("user_id").size()
    active_users = user_counts[user_counts >= MIN_RATINGS].index
    ratings = ratings[ratings["user_id"].isin(active_users)].copy()

    # normalize dtypes
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["book_id"] = ratings["book_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    return ratings


def split_per_user_80_20(ratings: pd.DataFrame):
    """Per-user random 80/20 split with fixed seed."""
    train, test = [], []
    for uid, g in ratings.groupby("user_id"):
        g = g.sample(frac=1.0, random_state=RANDOM_SEED)
        cut = max(1, int(0.8 * len(g)))
        train.append(g.iloc[:cut])
        test.append(g.iloc[cut:])
    train = pd.concat(train, ignore_index=True)
    test = pd.concat(test, ignore_index=True)
    return train, test


def main():
    # 1) Load data
    ratings = load_active_ratings()
    print(f"[OK] Active ratings: {len(ratings):,} rows | Users: {ratings['user_id'].nunique():,}")

    # 2) Split & define positives
    train, test = split_per_user_80_20(ratings)
    test_pos = test[test["rating"] >= REL_THRESHOLD].copy()

    # users with at least one positive in test
    eval_users = sorted(test_pos["user_id"].unique().tolist())
    print(f"[OK] Train={len(train):,}, Test positives={len(test_pos):,}, Eval users={len(eval_users):,}")

    if len(eval_users) == 0:
        print("[WARN] No users with positive items in test — nothing to evaluate.")
        return

    # 3) Build fast lookups
    test_items = test_pos.groupby("user_id")["book_id"].apply(set).to_dict()
    train_items = train.groupby("user_id")["book_id"].apply(set).to_dict()

    # 4) Load ALS artifacts
    model, uid_map, iid_map, rev_uid, rev_iid = load_artifacts()

    # 5) Build item-user matrix ON TRAIN ONLY (items x users), CSR for speed
    # rows = iidx (items), cols = uidx (users)
    mapped = ratings[["user_id", "book_id"]].copy()
    # map using only TRAIN to avoid leakage
    train_m = train.copy()
    train_m["uidx"] = train_m["user_id"].map(uid_map)
    train_m["iidx"] = train_m["book_id"].map(iid_map)
    train_m = train_m.dropna(subset=["uidx", "iidx"])
    train_m["uidx"] = train_m["uidx"].astype(int)
    train_m["iidx"] = train_m["iidx"].astype(int)

    # weight by rating (implicit confidence could be 1+rating; here raw rating is fine for user_items)
    mat_items_users = coo_matrix(
        (train_m["rating"].astype(np.float32), (train_m["iidx"], train_m["uidx"]))
    ).tocsr()  # (I x U)

    # 6) Evaluate per user
    precisions = []
    skipped_users = 0

    for u in tqdm(eval_users, desc="Evaluating users"):
        if u not in uid_map:
            skipped_users += 1
            continue
        uidx = uid_map[u]

        # Single-row CSR for this user: (1 x I) needed by implicit.recommend
        # implicit expects user_items with 1 row per user in userids; we pass only one user
        user_items_row = csr_matrix(mat_items_users.T[uidx])

        # Get N candidates (K*2 to allow filtering seen)
        res = model.recommend(uidx, user_items_row, N=K * 2, filter_items=None)

        # Handle return type: either (indices, scores) or list[(idx, score)]
        if isinstance(res, tuple) and len(res) == 2:
            item_indices = res[0].tolist()
        else:
            item_indices = [i for (i, _) in res]

        # Map back to real book_ids and exclude seen-in-train
        seen = train_items.get(u, set())
        cand_book_ids = []
        for i in item_indices:
            # skip items not in reverse map (defensive)
            if i not in rev_iid:
                continue
            bid = rev_iid[i]
            if bid not in seen:
                cand_book_ids.append(bid)
            if len(cand_book_ids) >= K:
                break

        # Hits vs. test positives
        pos = test_items.get(u, set())
        hits = sum(1 for b in cand_book_ids if b in pos)
        precisions.append(hits / K)

    macro_p = float(np.mean(precisions)) if precisions else 0.0

    print("\n=== Active-User Precision@{K} (ALS-only) ===".format(K=K))
    print(f"Users evaluated: {len(precisions):,} (skipped {skipped_users})")
    print(f"Macro Precision@{K}: {macro_p:.3f}")

    # 7) Save a tiny proof file (nice for appendix)
    out_path = os.path.join(ART_DIR, "precision_active_users.txt")
    with open(out_path, "w") as f:
        f.write(f"Active-User Precision@{K}: {macro_p:.6f}\n")
        f.write(f"Users evaluated: {len(precisions)} (skipped {skipped_users})\n")
    print(f"[OK] Saved result → {out_path}")


if __name__ == "__main__":
    main()
