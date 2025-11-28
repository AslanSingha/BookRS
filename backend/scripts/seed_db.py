"""
Safe & Optimized Database Seeder for BookRS (SQLite)
----------------------------------------------------
This script:
  ✓ Clears existing data to prevent duplicates
  ✓ Seeds books and ratings from Goodbooks-10k dataset
  ✓ Auto-creates users based on ratings
  ✓ Prints summary statistics when finished

Usage:
    python -m backend.scripts.seed_db
"""

import pandas as pd
from tqdm import tqdm
from sqlalchemy import text
from sqlalchemy.orm import Session
from backend.core.database import SessionLocal
from backend.models.book_model import Book
from backend.models.user_model import User
from backend.models.rating_model import Rating

BOOKS_PATH = "dataset/books.csv"
RATINGS_PATH = "dataset/ratings.csv"
BATCH_SIZE = 1000   # commit size for bulk insert
MAX_RATINGS = None  # None = all rows; set to 100_000 for quick testing


def clear_existing_data(session: Session):
    """Remove old data before reseeding."""
    print("[INFO] Clearing old records ...")
    session.execute(text("DELETE FROM ratings;"))
    session.execute(text("DELETE FROM users;"))
    session.execute(text("DELETE FROM books;"))
    session.commit()
    print("[OK] All old records removed.")



def seed_books(session: Session):
    print("[INFO] Loading books.csv ...")
    df = pd.read_csv(BOOKS_PATH)
    df = df.fillna("")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Seeding books"):
        book = Book(
            book_id=int(row["book_id"]),
            title=str(row["title"])[:255],
            authors=str(row["authors"])[:255],
            description=str(row.get("description", ""))[:2000],
            avg_rating=float(row.get("average_rating", 0)) if "average_rating" in df.columns else None,
            image_url=str(row.get("image_url", ""))[:500],
        )
        session.merge(book)
    session.commit()
    print(f"[OK] {len(df):,} books inserted.")


def seed_users_and_ratings(session: Session):
    print("[INFO] Loading ratings.csv ...")
    df = pd.read_csv(RATINGS_PATH, nrows=MAX_RATINGS)
    print(f"[INFO] Seeding {len(df):,} rating rows...")

    # Create unique users
    user_ids = df["user_id"].unique()
    users = [User(id=int(u), name=f"User-{u}") for u in user_ids]
    session.bulk_save_objects(users)
    session.commit()
    print(f"[OK] {len(user_ids):,} users inserted.")

    # Insert ratings in batches
    batch = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Seeding ratings"):
        batch.append(Rating(
            user_id=int(row["user_id"]),
            book_id=int(row["book_id"]),
            rating=float(row["rating"])
        ))
        if len(batch) >= BATCH_SIZE:
            session.bulk_save_objects(batch)
            session.commit()
            batch.clear()
    if batch:
        session.bulk_save_objects(batch)
        session.commit()

    print(f"[OK] {len(df):,} ratings inserted.")


def main():
    print("=== BookRS Database Seeding (Safe Mode) ===")
    db: Session = SessionLocal()
    try:
        clear_existing_data(db)
        seed_books(db)
        seed_users_and_ratings(db)

        # Summary
        books_count = db.execute(text("SELECT COUNT(*) FROM books;")).scalar()
        users_count = db.execute(text("SELECT COUNT(*) FROM users;")).scalar()
        ratings_count = db.execute(text("SELECT COUNT(*) FROM ratings;")).scalar()

        print("\n=== ✅ Seeding Complete ===")
        print(f"Books:   {books_count:,}")
        print(f"Users:   {users_count:,}")
        print(f"Ratings: {ratings_count:,}\n")

    except Exception as e:
        print("[ERROR]", e)
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()
