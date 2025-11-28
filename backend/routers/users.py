from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.core.database import SessionLocal
from backend.models.user_model import User

router = APIRouter(prefix="/users", tags=["Users"])

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create new user
@router.post("/", summary="Register a new user")
def create_user(name: str, email: str = None, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    new_user = User(name=name, email=email)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user": {"id": new_user.id, "name": new_user.name}}

# Get all users
@router.get("/", summary="List all users")
def list_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users
