"""
Run FastAPI backend for BookRS
------------------------------
Serves a simple API endpoint for book recommendation.
"""

from fastapi import FastAPI, Query
from backend.ml.recommender_hybrid import HybridRecommender

app = FastAPI(title="BookRS API", version="1.0")

# Load the hybrid model (semantic + CF)
model = HybridRecommender()


@app.get("/")
def root():
    return {"message": "Welcome to BookRS API"}


@app.get("/recommend")
def recommend(
    query: str = Query(..., description="Search keywords or topic"),
    user_id: int = Query(1, description="User ID (default=1)"),
    top_k: int = Query(10, description="Number of recommendations")
):
    """Return top-K recommended books as JSON."""
    results = model.recommend(query, user_id=user_id, top_k=top_k)
    return results.to_dict(orient="records")


# To run: uvicorn backend.scripts.run_fastapi:app --reload
