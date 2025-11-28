import os
from dotenv import load_dotenv

load_dotenv()

# DB
DATABASE_URL = "sqlite:///./bookrs.db"


# Artifacts
ART_DIR = os.getenv("ART_DIR", "artifacts")
EMB_PATH = os.path.join(ART_DIR, "book_embeddings.pt")
EMB_META = os.path.join(ART_DIR, "emb_meta.parquet")
ALS_USER_FACTORS = os.path.join(ART_DIR, "als_user_factors.npz")
ALS_ITEM_FACTORS = os.path.join(ART_DIR, "als_item_factors.npz")
POPULARITY_PATH = os.path.join(ART_DIR, "popularity.parquet")

# Hybrid weights (tune as needed)
ALPHA = float(os.getenv("ALPHA", 0.6))   # semantic
BETA  = float(os.getenv("BETA", 0.35))   # CF (ALS)
GAMMA = float(os.getenv("GAMMA", 0.05))  # popularity prior

TOPK_DEFAULT = int(os.getenv("TOPK_DEFAULT", 10))
