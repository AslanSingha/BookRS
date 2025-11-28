# """
# Hybrid Recommender
# ------------------
# Combines Semantic Embeddings and Collaborative Filtering (ALS)
# to generate user-personalized book recommendations.
# """
#
# import numpy as np
# import pandas as pd
# from backend.ml.recommender_semantic import SemanticRecommender
# from backend.core.config import ART_DIR
# from scipy.spatial.distance import cosine
# import os
#
#
# class HybridRecommender:
#     def __init__(self):
#         print("[INFO] Initializing Hybrid Recommender (Semantic + CF) ...")
#         self.semantic = SemanticRecommender()
#
#         # Load ALS factors (trained offline)
#         self.user_factors = np.load(os.path.join(ART_DIR, "als_user_factors.npz"))["data"]
#         self.item_factors = np.load(os.path.join(ART_DIR, "als_item_factors.npz"))["data"]
#
#     def recommend(self, query: str, user_id: int = 1, top_k: int = 10):
#         # Step 1 — Get semantic matches
#         sem_df = self.semantic.recommend(query, top_k=max(top_k, 50))
#
#         # Step 2 — Compute collaborative filtering similarity (personalization)
#         sem_df["cf_score"] = 0.0
#         if 0 <= user_id < len(self.user_factors):
#             user_vec = self.user_factors[user_id]
#             for i, row in sem_df.iterrows():
#                 # Map book_id to ALS item factor index safely
#                 book_idx = int(row["book_id"]) % len(self.item_factors)
#                 item_vec = self.item_factors[book_idx]
#                 sim = 1 - cosine(user_vec, item_vec)
#                 sem_df.at[i, "cf_score"] = sim if not np.isnan(sim) else 0
#         else:
#             print(f"[WARN] User {user_id} not found in ALS model — using semantic only.")
#
#         # Step 3 — Combine with weights (no popularity)
#         sem_df["hybrid_score"] = (
#             0.7 * sem_df["semantic_score"] +
#             0.3 * sem_df["cf_score"]
#         )
#
#         # Step 4 — Sort and return
#         sem_df = sem_df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
#         return sem_df.head(top_k)


import numpy as np
import pandas as pd
import pickle, os
from scipy.spatial.distance import cosine
from backend.ml.recommender_semantic import SemanticRecommender
from backend.core.config import ART_DIR

class HybridRecommender:
    def __init__(self):
        print("[INFO] Initializing Hybrid Recommender (Semantic + CF) ...")
        self.semantic = SemanticRecommender()

        # Load ALS artifacts
        self.user_factors = np.load(os.path.join(ART_DIR, "als_user_factors.npz"))["data"]
        self.item_factors = np.load(os.path.join(ART_DIR, "als_item_factors.npz"))["data"]

        # Load mapping dictionaries
        with open(os.path.join(ART_DIR, "als_uid_map.pkl"), "rb") as f:
            self.uid_map = pickle.load(f)
        with open(os.path.join(ART_DIR, "als_iid_map.pkl"), "rb") as f:
            self.iid_map = pickle.load(f)

        # Also build reverse map for safety
        self.rev_uid_map = {v: k for k, v in self.uid_map.items()}
        self.rev_iid_map = {v: k for k, v in self.iid_map.items()}

        print(f"[OK] ALS model loaded: {len(self.uid_map):,} users, {len(self.iid_map):,} items")

    def recommend(self, query: str, user_id: int = 1, top_k: int = 10):
        # Step 1 — Semantic matches
        sem_df = self.semantic.recommend(query, top_k=max(top_k, 50))

        # Step 2 — Collaborative personalization
        sem_df["cf_score"] = 0.0
        if user_id in self.uid_map:
            uidx = self.uid_map[user_id]
            user_vec = self.user_factors[uidx]

            for i, row in sem_df.iterrows():
                book_id = int(row["book_id"])
                if book_id in self.iid_map:
                    iidx = self.iid_map[book_id]
                    item_vec = self.item_factors[iidx]
                    sim = 1 - cosine(user_vec, item_vec)
                    sem_df.at[i, "cf_score"] = sim if not np.isnan(sim) else 0
        else:
            print(f"[WARN] User {user_id} not found in ALS model — using semantic only.")

        # Step 3 — Weighted fusion
        sem_df["hybrid_score"] = 0.7 * sem_df["semantic_score"] + 0.3 * sem_df["cf_score"]

        return sem_df.sort_values("hybrid_score", ascending=False).reset_index(drop=True).head(top_k)
