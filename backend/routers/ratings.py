from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.core.database import SessionLocal
from backend.models.rating_model import Rating

router = APIRouter(prefix="/ratings", tags=["Ratings"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", summary="Add or update a rating")
def rate_book(user_id: int, book_id: int, rating: float, db: Session = Depends(get_db)):
    if rating < 0 or rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5.")
    existing = db.query(Rating).filter(
        Rating.user_id == user_id, Rating.book_id == book_id
    ).first()
    if existing:
        existing.rating = rating
        db.commit()
        return {"message": "Rating updated."}
    new_rating = Rating(user_id=user_id, book_id=book_id, rating=rating)
    db.add(new_rating)
    db.commit()
    return {"message": "Rating added successfully."}

@router.get("/{user_id}", summary="Get all ratings by user")
def get_user_ratings(user_id: int, db: Session = Depends(get_db)):
    ratings = db.query(Rating).filter(Rating.user_id == user_id).all()
    return ratings
