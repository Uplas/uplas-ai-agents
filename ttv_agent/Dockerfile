# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# ENV PORT 8080 # Standard port for Cloud Run, can be overridden

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if any (e.g., for FFmpeg if doing local video processing)
# RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app/

# Expose the port the app runs on (FastAPI default is 8000 if not overridden by uvicorn command)
# Cloud Run will automatically map its external port to this.
EXPOSE 8000
# If Uvicorn is started on $PORT, then no need to EXPOSE specifically for Cloud Run.

# Command to run the Uvicorn server when the container starts
# The port should ideally be taken from the PORT environment variable for Cloud Run.
# Uvicorn default port is 8000. Cloud Run provides a PORT env var (usually 8080).
# The main:app refers to main.py file and the app = FastAPI() instance.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
