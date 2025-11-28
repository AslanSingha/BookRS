"""
TF-IDF Based Recommender for BookRS
-----------------------------------
Uses keyword frequency (TF-IDF) to compute similarity
between books based on title, authors, and description.
Now fully integrated with the SQLite database.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.core.db_utils import load_books


class TFIDFRecommender:
    def __init__(self, max_features=5000):
        print("[INFO] Loading books data from database for TF-IDF model ...")
        # Load only relevant columns from DB
        self.df = load_books(columns=["book_id", "title", "authors", "description"])
        print(f"[OK] Loaded {len(self.df):,} books from database.")

        # Combine text fields for TF-IDF
        self.df["combined_text"] = (
            self.df["title"].fillna("") + " " +
            self.df["authors"].fillna("") + " " +
            self.df["description"].fillna("")
        )

        print("[INFO] Building TF-IDF matrix ...")
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])
        print(f"[OK] TF-IDF model trained with {self.tfidf_matrix.shape[1]} features.")

    def recommend(self, query: str, top_k: int = 10):
        """Return top_k similar books for a given query."""
        if not query or not query.strip():
            return pd.DataFrame(columns=["book_id", "title", "authors", "tfidf_score"])

        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = sim_scores.argsort()[-top_k:][::-1]

        results = self.df.iloc[top_indices][["book_id", "title", "authors"]].copy()
        results["tfidf_score"] = sim_scores[top_indices].round(4)
        return results.reset_index(drop=True)
