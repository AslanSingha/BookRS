import torch, numpy as np, pandas as pd, os
art = "artifacts"
print("[VERIFY] Listing contents:")
for f in os.listdir(art): print(" -", f)
emb = torch.load(f"{art}/book_embeddings.pt")
meta = pd.read_parquet(f"{art}/emb_meta.parquet")
u = np.load(f"{art}/als_user_factors.npz")["data"]
i = np.load(f"{art}/als_item_factors.npz")["data"]
print(f"\nEmbeddings: {emb.shape}")
print(f"Meta rows: {len(meta):,}")
print(f"ALS users: {u.shape}, ALS items: {i.shape}")
