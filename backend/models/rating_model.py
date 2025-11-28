from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime
from datetime import datetime
from backend.core.database import Base

class Rating(Base):
    __tablename__ = "ratings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    book_id = Column(Integer, ForeignKey("books.book_id"))
    rating = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
