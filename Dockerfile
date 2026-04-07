# Use an official Python runtime as a parent image, slim version for smaller image size
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Set home directory so deepface knows where to save weights consistently
ENV HOME=/user/home
RUN mkdir -p $HOME/.deepface/weights

WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies including gunicorn for production
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Run the application using Gunicorn with Uvicorn workers
# We set a slightly longer timeout as DeepFace inference might take a few seconds on CPU
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "main:app"]
