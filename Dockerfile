# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY app.py .
COPY static/ static/
COPY templates/ templates/
COPY video-editor-434002-b3492f400f55.json .

# Create output directory
RUN mkdir -p output && \
    chmod 777 output

# Set proper permissions for credentials
RUN chmod 600 video-editor-434002-b3492f400f55.json

# Expose port
EXPOSE 8000

# Environment variables
ENV FLASK_APP=app.py \
    FLASK_ENV=production \
    GOOGLE_APPLICATION_CREDENTIALS=/app/video-editor-434002-b3492f400f55.json \
    OPENAI_API_KEY=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

# Run the application with Flask's built-in server
CMD ["python", "app.py"]

