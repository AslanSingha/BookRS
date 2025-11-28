from sqlalchemy import Column, Integer, String, Float, Text
from backend.core.database import Base

class Book(Base):
    __tablename__ = "books"
    book_id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    authors = Column(String)
    description = Column(Text)
    avg_rating = Column(Float)
    image_url = Column(String)
