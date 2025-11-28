# BookRS — AI-Powered Book Recommendation System

**BookRS** is an intelligent hybrid recommendation system that suggests books based on both user behavior and semantic similarity.  
It combines **collaborative filtering (ALS)** and **semantic search (Sentence-BERT)** to deliver personalized, accurate, and meaningful recommendations.

---

## Features
- **Hybrid Model:** Combines semantic similarity (from text embeddings) and collaborative filtering.
- **FastAPI Backend:** Efficient API architecture.
- **Gradio Interface:** Simple and elegant demo UI.
- **SQLite Database:** Lightweight, portable data storage.
- **6M+ Ratings & 10k Books:** Scalable and realistic dataset.

---

## System Architecture
- **Dataset:** Goodbooks-10k (10,000 books, 53,000 users, 6M ratings)
- **Embedding Model:** `all-MiniLM-L6-v2` (Sentence-BERT)
- **CF Model:** Alternating Least Squares (ALS) from `implicit`
- **Storage:** SQLite database
- **Interface:** Gradio (FastAPI service planned for production)

---

## Setup Instructions

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```



### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt 
```
### 3️⃣ Seed the database and build models
```bash
python -m backend.scripts.seed_db
python -m backend.scripts.build_embeddings
python -m backend.scripts.train_cf
```
### 4️⃣ Run the FastAPI backend
```bash
uvicorn backend.scripts.run_fastapi:app --reload
```
### 5️⃣ Run the Gradio prototype
```bash
python -m backend.scripts.run_gradio
```
### Access the app at:

http://127.0.0.1:7860

## Docker testing
```bash
docker composer up --build
```
## Author
### Supervisor: Mr. SOK Kimheng
### Name: Rin Singh
### Major: Information and Communication Engineering (Year 5)
### Institution: Institute of Technology of Cambodia (ITC)
