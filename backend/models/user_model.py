from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from backend.core.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True)
    joined_at = Column(DateTime, default=datetime.utcnow)
