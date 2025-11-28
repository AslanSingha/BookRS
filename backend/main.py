from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import users, books, ratings, recommend

app = FastAPI(title="BookRS - AI-Powered Recommendation System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(books.router)
app.include_router(ratings.router)
app.include_router(recommend.router)

@app.get("/")
def root():
    return {"message": "Welcome to BookRS API"}
