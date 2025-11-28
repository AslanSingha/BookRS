import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from backend.core.config import EMB_PATH, EMB_META


class SemanticRecommender:
    def __init__(self):
        print("[INFO] Loading semantic model and embeddings...")
        # force CPU for consistency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=str(self.device))
        self.emb = torch.load(EMB_PATH, map_location=self.device).to(self.device)

        self.meta = pd.read_parquet(EMB_META)
        print(f"[OK] Loaded {len(self.meta):,} book embeddings on {self.device}.")

    def recommend(self, query: str, top_k: int = 10):
        if not query or not query.strip():
            return pd.DataFrame(columns=["book_id", "title", "authors", "semantic_score"])
        q = self.model.encode(query, convert_to_tensor=True, device=str(self.device))
        scores = util.pytorch_cos_sim(q, self.emb)[0]
        topk = torch.topk(scores, k=min(top_k, len(self.meta)))
        idx = topk.indices.cpu().numpy()
        sc = topk.values.cpu().numpy()

        out = self.meta.iloc[idx][["book_id", "title", "authors"]].copy()

        # cols = [c for c in ["book_id", "id", "title", "authors"] if c in self.meta.columns]
        # out = self.meta.iloc[idx][cols].copy()
        # if "id" in out.columns and "book_id" not in out.columns:
        #     out = out.rename(columns={"id": "book_id"})

        out["semantic_score"] = sc.round(4)
        return out.reset_index(drop=True)
