"""
Incremental Semantic Embedding Builder for BookRS
-------------------------------------------------
Updates existing book embeddings only for new books found in the database.
"""

import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from backend.core.db_utils import load_books
from backend.core.config import ART_DIR, EMB_PATH, EMB_META


def main():
    os.makedirs(ART_DIR, exist_ok=True)

    print("[INFO] Checking existing embeddings...")
    existing_emb, existing_meta = None, None
    if os.path.exists(EMB_PATH) and os.path.exists(EMB_META):
        existing_emb = torch.load(EMB_PATH)
        existing_meta = pd.read_parquet(EMB_META)
        print(f"[OK] Found {len(existing_meta):,} existing embeddings.")
    else:
        print("[WARN] No existing embeddings found. Full build will run.")

    # Load current books from DB
    db_books = load_books(columns=["book_id", "title", "authors", "description"])
    db_books["book_id"] = db_books["book_id"].astype(int)

    # Identify new books
    if existing_meta is not None:
        existing_ids = set(existing_meta["book_id"].astype(int))
        new_books = db_books[~db_books["book_id"].isin(existing_ids)]
    else:
        new_books = db_books

    if new_books.empty:
        print("[INFO] No new books found. Embeddings are up to date.")
        return

    print(f"[INFO] Found {len(new_books):,} new books. Generating embeddings...")

    # Combine text fields for semantic encoding
    new_books["combined_text"] = (
        new_books["title"].fillna("") + " " +
        new_books["title"].fillna("") + " " +  # weighted title
        new_books["authors"].fillna("") + " " +
        new_books["description"].fillna("")
    )

    # Load Sentence Transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    new_emb = model.encode(
        new_books["combined_text"].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True
    )

    # Append to existing embeddings
    if existing_emb is not None:
        updated_emb = torch.cat([existing_emb, new_emb], dim=0)
        updated_meta = pd.concat([existing_meta, new_books], ignore_index=True)
    else:
        updated_emb = new_emb
        updated_meta = new_books

    # Save updated files
    torch.save(updated_emb, EMB_PATH)
    updated_meta[["book_id", "title", "authors", "combined_text"]].to_parquet(EMB_META, index=False)

    print(f"[OK] Updated embeddings saved → {EMB_PATH}")
    print(f"[OK] Updated metadata saved → {EMB_META}")
    print(f"[DONE] Total books embedded: {len(updated_meta):,}")


if __name__ == "__main__":
    main()
