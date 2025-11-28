# """
# Train ALS Collaborative Filtering Model (Database-driven)
# ---------------------------------------------------------
# Trains implicit-feedback ALS model from database ratings and saves:
#  - als_user_factors.npz
#  - als_item_factors.npz
#  - popularity.parquet
# """
#
# import os
# import numpy as np
# import pandas as pd
# from scipy.sparse import coo_matrix
# from implicit.als import AlternatingLeastSquares
# from backend.core.db_utils import load_ratings
# from backend.core.config import ART_DIR, ALS_USER_FACTORS, ALS_ITEM_FACTORS, POPULARITY_PATH
#
#
# def main():
#     os.makedirs(ART_DIR, exist_ok=True)
#
#     print("[INFO] Loading ratings data from database ...")
#     ratings = load_ratings(limit=None)  # user_id, book_id, rating
#     print(f"[OK] Loaded {len(ratings):,} ratings from DB.")
#
#     # Implicit feedback confidence: 1 + rating
#     ratings["confidence"] = 1.0 + ratings["rating"].clip(lower=0)
#
#     # Re-index users and items for matrix factorization
#     print("[INFO] Re-indexing users and items ...")
#     uid_map = {u: i for i, u in enumerate(ratings["user_id"].unique())}
#     iid_map = {b: i for i, b in enumerate(ratings["book_id"].unique())}
#     ratings["uidx"] = ratings["user_id"].map(uid_map)
#     ratings["iidx"] = ratings["book_id"].map(iid_map)
#
#     print(f"[OK] {len(uid_map):,} unique users, {len(iid_map):,} unique books.")
#
#     # Build sparse item-user confidence matrix (implicit feedback)
#     mat = coo_matrix(
#         (
#             ratings["confidence"].astype(np.float32),
#             (ratings["iidx"].astype(int), ratings["uidx"].astype(int))  # item x user for implicit ALS
#         )
#     ).tocsr()
#
#     # Train ALS model
#     print("[INFO] Training ALS collaborative filtering model ...")
#     model = AlternatingLeastSquares(
#         factors=64,
#         regularization=0.1,
#         iterations=20,
#         calculate_training_loss=True
#     )
#     model.fit(mat)
#
#     # Save latent factors
#     print("[INFO] Saving ALS latent factors ...")
#     np.savez_compressed(ALS_ITEM_FACTORS, data=model.item_factors)
#     np.savez_compressed(ALS_USER_FACTORS, data=model.user_factors)
#
#     # Compute popularity prior (for fallback recommendations)
#     print("[INFO] Computing popularity scores ...")
#     pop = ratings.groupby("book_id").size().reset_index(name="count")
#     pop["pop_score"] = (
#         (pop["count"] - pop["count"].min()) /
#         (pop["count"].max() - pop["count"].min() + 1e-9)
#     )
#     pop.to_parquet(POPULARITY_PATH, index=False)
#
#     print("[OK] ALS model and popularity data saved.")
#     print(f" users: {len(uid_map):,} | books: {len(iid_map):,}")
#
#
# if __name__ == "__main__":
#     main()


"""
Trains ALS collaborative filtering (implicit feedback) and saves:
- als_user_factors.npz
- als_item_factors.npz
- als_uid_map.pkl / als_iid_map.pkl
- popularity.parquet
"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from backend.core.config import ART_DIR, ALS_USER_FACTORS, ALS_ITEM_FACTORS, POPULARITY_PATH
from backend.core.db_utils import ENGINE  # use SQLite DB connection

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    print("[INFO] Loading ratings data from database ...")
    ratings = pd.read_sql("SELECT user_id, book_id, rating FROM ratings", ENGINE)
    print(f"[OK] Loaded {len(ratings):,} ratings from DB.")

    # Implicit feedback confidence: 1 + rating
    ratings["confidence"] = 1.0 + ratings["rating"].clip(lower=0)

    # Re-index users/items for ALS
    user_ids = ratings["user_id"].unique()
    item_ids = ratings["book_id"].unique()
    uid_map = {u: i for i, u in enumerate(user_ids)}
    iid_map = {b: i for i, b in enumerate(item_ids)}

    ratings["uidx"] = ratings["user_id"].map(uid_map)
    ratings["iidx"] = ratings["book_id"].map(iid_map)

    print(f"[OK] Unique users: {len(uid_map):,} | Unique items: {len(iid_map):,}")

    # Build item-user sparse matrix (for implicit ALS)
    mat = coo_matrix(
        (
            ratings["confidence"].astype(np.float32),
            (ratings["iidx"].astype(int), ratings["uidx"].astype(int)),
        )
    ).tocsr()

    print(f"[INFO] Matrix shape: {mat.shape} (items x users)")

    # Train ALS model
    print("[INFO] Training ALS collaborative filtering model ...")
    model = AlternatingLeastSquares(
        factors=64,
        regularization=0.1,
        iterations=20,
        calculate_training_loss=True,
    )
    model.fit(mat.T)

    # Save ALS factors
    np.savez_compressed(ALS_ITEM_FACTORS, data=model.item_factors)
    np.savez_compressed(ALS_USER_FACTORS, data=model.user_factors)
    print("[OK] Saved ALS latent factors.")

    # Save ID mappings for later lookup in HybridRecommender
    with open(os.path.join(ART_DIR, "als_uid_map.pkl"), "wb") as f:
        pickle.dump(uid_map, f)
    with open(os.path.join(ART_DIR, "als_iid_map.pkl"), "wb") as f:
        pickle.dump(iid_map, f)
    print("[OK] Saved ALS ID mapping dictionaries.")

    # Compute popularity prior (normalized rating count)
    pop = ratings.groupby("book_id").size().reset_index(name="count")
    pop["pop_score"] = (pop["count"] - pop["count"].min()) / (
        pop["count"].max() - pop["count"].min() + 1e-9
    )
    pop.to_parquet(POPULARITY_PATH, index=False)
    print("[OK] Popularity data saved.")

    print(
        f"[DONE] ALS model trained successfully.\n"
        f" Users: {len(uid_map):,} | Items: {len(iid_map):,}\n"
        f" Artifacts saved to: {ART_DIR}"
    )

if __name__ == "__main__":
    main()
