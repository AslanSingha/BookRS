"""
Database Utility Module for BookRS
----------------------------------
Provides reusable functions for loading data directly from the SQLite database.
Eliminates the need for static CSV files.
"""

import pandas as pd
from sqlalchemy import create_engine

# === Database Connection ===
DB_URL = "sqlite:///bookrs.db"
ENGINE = create_engine(DB_URL, echo=False)

# === Book Loader ===
def load_books(columns=None):
    """Load books table as DataFrame."""
    cols = "*" if columns is None else ", ".join(columns)
    query = f"SELECT {cols} FROM books"
    df = pd.read_sql(query, ENGINE)
    return df.fillna("")

# === Ratings Loader ===
def load_ratings(limit=None):
    """Load ratings table as DataFrame (optionally limited for testing)."""
    query = "SELECT user_id, book_id, rating FROM ratings"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql(query, ENGINE)
    return df

# === Users Loader ===
def load_users():
    """Load users table as DataFrame."""
    df = pd.read_sql("SELECT id, name FROM users", ENGINE)
    return df

# === Utility Function ===
def count_records():
    """Print basic table record counts for debugging."""
    counts = {}
    for table in ["books", "users", "ratings"]:
        counts[table] = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", ENGINE)["count"][0]
    return counts
