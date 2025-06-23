# uplas-ai-agents/Dockerfile
# Dockerfile for the Uplas Unified AI Agent Service

# 1. Use an official, slim Python runtime as a parent image
FROM python:3.9-slim-bookworm

# 2. Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Set the working directory in the container
WORKDIR /app

# 4. Copy the requirements file and install dependencies
# This layer is cached, so it only re-runs if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application's code into the container
COPY . .

# 6. Define the command to run the application using Gunicorn
# Gunicorn is a production-grade WSGI server.
# Cloud Run automatically sets the $PORT environment variable.
# The number of workers is a recommendation; you can adjust based on performance testing.
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:$PORT"]
