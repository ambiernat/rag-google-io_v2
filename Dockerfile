# -----------------------------
# Base image
# -----------------------------
FROM python:3.12-slim

# -----------------------------
# Set environment variables
# -----------------------------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu


# -----------------------------
# Set work directory
# -----------------------------
WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements or your project
# -----------------------------
# COPY pyproject.toml poetry.lock* /app/  # if using poetry
# OR
COPY requirements.api.txt /app/requirements.txt

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip setuptools wheel

# If using poetry
# RUN pip install "poetry>=1.6.0" && poetry install --no-root --no-dev

# If using requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# -----------------------------
# Copy project code
# -----------------------------
COPY ./api /app/api
COPY ./retrieval /app/retrieval  # <-- ADD THIS LINE
COPY ./data /app/data 

# -----------------------------
# Expose port
# -----------------------------
EXPOSE 8000

# Download the model at build time - reduces cost of first request and ensures it's available when the container starts
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# -----------------------------
# Start Uvicorn
# -----------------------------
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

 # Add --reload for development (not recommended for production)
