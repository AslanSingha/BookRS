from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from backend.core.database import SessionLocal
from backend.models.book_model import Book

router = APIRouter(prefix="/books", tags=["Books"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", summary="List all books")
def list_books(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    books = db.query(Book).offset(skip).limit(limit).all()
    return books

@router.get("/search", summary="Search books by keyword")
def search_books(q: str = Query(..., min_length=2), db: Session = Depends(get_db)):
    results = db.query(Book).filter(Book.title.ilike(f"%{q}%")).limit(10).all()
    return {"query": q, "results": results}
