# -----------------------------
# Base image
# -----------------------------
FROM python:3.12-slim

# -----------------------------
# Set environment variables
# -----------------------------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

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
COPY requirements.txt /app/ 

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip setuptools wheel

# If using poetry
# RUN pip install "poetry>=1.6.0" && poetry install --no-root --no-dev

# If using requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


# -----------------------------
# Expose port
# -----------------------------
EXPOSE 8000

# -----------------------------
# Start Uvicorn
# -----------------------------

# Do NOT COPY the whole project here for dev, we'll mount it
# Dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
 # Add --reload for development (not recommended for production)
