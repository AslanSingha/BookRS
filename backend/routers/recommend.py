from fastapi import APIRouter, Query
from backend.ml.recommender_hybrid import HybridRecommender
from backend.core.config import TOPK_DEFAULT

router = APIRouter(prefix="/recommend", tags=["Recommendations"])

# Load once at startup
hybrid = HybridRecommender()

@router.get("/hybrid", summary="Hybrid recommendations (semantic + CF + popularity)")
def recommend_hybrid(
    query: str = Query(..., min_length=2),
    user_id: int | None = Query(None, description="Known ALS user; fallback if None"),
    top_k: int = TOPK_DEFAULT
):
    df = hybrid.recommend(query=query, user_id=user_id, top_k=top_k)
    return df.to_dict(orient="records")






