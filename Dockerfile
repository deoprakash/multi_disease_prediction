# # syntax=docker/dockerfile:1.4
# FROM python:3.11-slim

# WORKDIR /code

# # System dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     pkg-config \  
#     default-libmysqlclient-dev \
#     libglib2.0-0 \
#     libgl1 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev \
#     git \
#  && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# # Enable pip cache with BuildKit
# RUN --mount=type=cache,target=/root/.cache \
#     pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt

# COPY . .

# # # Load environment variables from .env file
# # ENV PYTHONUNBUFFERED=1

# EXPOSE 5000

# CMD ["python", "gunicorn app:app --bind 0.0.0.0:5000"]

FROM python:3.8-slim-buster

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose dynamic port for Render
EXPOSE $PORT

# Start the app using gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
