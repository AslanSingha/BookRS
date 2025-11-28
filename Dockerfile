# 1) Use the full Python image (not slim) – easier for SciPy/numpy/etc.
FROM python:3.12

# 2) Install system dependencies for scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 3) Workdir
WORKDIR /app

# 4) Copy requirements first
COPY requirements.txt .

# 5) Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6) Copy the rest of the project
COPY . .

# 7) Env
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1\
    PYTHONPATH=/app

# 8) Port
EXPOSE 8000

# 9) Run your app
CMD ["python","-m","backend.scripts.run_gradio"]


#API
#CMD ["uvicorn", "backend.scripts.run_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]

