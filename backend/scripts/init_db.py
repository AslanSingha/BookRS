from backend.core.database import Base, engine
from backend.models import book_model, user_model, rating_model

print("[INFO] Creating tables...")
Base.metadata.create_all(bind=engine)
print("[INFO] Database ready ✅")
