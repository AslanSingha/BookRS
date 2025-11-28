"""
Popularity-based Recommender for BookRS
---------------------------------------
Ranks books based on a weighted popularity formula
that balances average rating and number of ratings.
Uses IMDb-style weighted mean to favor both quality and engagement.
Now fully integrated with the BookRS database.
"""

import pandas as pd
from backend.core.db_utils import load_books, load_ratings


class PopularityRecommender:
    def __init__(self):
        print("[INFO] Loading data from database for popularity computation ...")

        # Load books and ratings from DB
        books = load_books(columns=["book_id", "title", "authors"])
        ratings = load_ratings()

        # Rename for consistency
        books = books.rename(columns={"id": "book_id"})

        # Compute aggregate statistics
        pop = ratings.groupby("book_id").agg(
            avg_rating=("rating", "mean"),
            num_ratings=("rating", "count")
        ).reset_index()

        # Merge with books metadata
        books = books.merge(pop, on="book_id", how="left").fillna(0)

        # Weighted popularity score (IMDb-style)
        m = pop["num_ratings"].quantile(0.90)  # threshold for "popular"
        C = pop["avg_rating"].mean()           # global average rating

        def weighted_rating(x):
            v = x["num_ratings"]
            R = x["avg_rating"]
            return (v / (v + m)) * R + (m / (m + v)) * C

        books["popularity_score"] = books.apply(weighted_rating, axis=1)

        # Sort descending by popularity
        self.df = books.sort_values(by="popularity_score", ascending=False).reset_index(drop=True)
        print(f"[OK] Popularity table computed for {len(self.df):,} books.")

    def recommend(self, top_k=10):
        """Return top-k most popular books overall."""
        return self.df[
            ["book_id", "title", "authors", "avg_rating", "num_ratings", "popularity_score"]
        ].head(top_k)
