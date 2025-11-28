import numpy as np
import pandas as pd
from backend.core.config import ALS_USER_FACTORS, ALS_ITEM_FACTORS, POPULARITY_PATH

class CFModel:
    def __init__(self, user_id_to_row=None, book_id_to_row=None):
        # Load factors
        self.item_factors = np.load(ALS_ITEM_FACTORS)["data"]  # (items, k)
        self.user_factors = np.load(ALS_USER_FACTORS)["data"]  # (users, k)
        # Optional id maps (dicts). For quick demo you can infer or skip.
        self.uid_map = user_id_to_row or {}
        self.bid_map = book_id_to_row or {}

        self.pop = pd.read_parquet(POPULARITY_PATH)  # book_id, pop_score

    def score_for_user(self, user_id: int) -> pd.DataFrame:
        """Return CF scores for all items for a known ALS user. If unknown, empty frame."""
        if user_id not in self.uid_map:
            return pd.DataFrame(columns=["book_id","cf_score"])
        uidx = self.uid_map[user_id]
        u = self.user_factors[uidx]  # (k,)
        scores = self.item_factors @ u  # (items,)
        # reverse map: row → book_id (if provided)
        if self.bid_map:
            inv = {v:k for k,v in self.bid_map.items()}
            book_ids = [inv[i] for i in range(len(scores))]
        else:
            # fallback sequential IDs 0..N-1 (only for demo if your meta aligns)
            book_ids = list(range(len(scores)))
        df = pd.DataFrame({"book_id": book_ids, "cf_score": scores})
        return df

    def popularity(self) -> pd.DataFrame:
        return self.pop.copy()
