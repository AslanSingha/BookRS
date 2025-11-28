"""
Build Semantic Embeddings for Books (Database-driven)
-----------------------------------------------------
Generates and saves:
 - book_embeddings.pt  (torch tensor of semantic vectors)
 - emb_meta.parquet    (book_id, title, authors, combined_text in same order)
"""

import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from backend.core.db_utils import load_books
from backend.core.config import ART_DIR, EMB_PATH, EMB_META


def main():
    os.makedirs(ART_DIR, exist_ok=True)

    print("[INFO] Loading books data from database ...")
    df = load_books(columns=["book_id", "title", "authors", "description"])
    print(f"[OK] Loaded {len(df):,} books from DB.")

    # Combine text fields (weighted toward title)
    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["title"].fillna("") + " " +  # title repeated for stronger weight
        df["authors"].fillna("") + " " +
        df["description"].fillna("")
    )

    # Initialize model
    print("[INFO] Generating semantic embeddings using all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(
        df["combined_text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # Save embeddings
    torch.save(emb, EMB_PATH)
    meta = df[["book_id", "title", "authors", "combined_text"]].copy()
    meta.to_parquet(EMB_META, index=False)

    print(f"[OK] Saved embeddings → {EMB_PATH}")
    print(f"[OK] Saved metadata → {EMB_META}")
    print("[DONE] Semantic embedding build completed.")


if __name__ == "__main__":
    main()
