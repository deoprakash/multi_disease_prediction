# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

WORKDIR /code

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    pkg-config \  
    default-libmysqlclient-dev \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Enable pip cache with BuildKit
RUN --mount=type=cache,target=/root/.cache \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Load environment variables from .env file
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "app.py"]
