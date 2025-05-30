# uplas-ai-agents/<agent_name>/Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies that might be needed by some Python packages
# (e.g., for certain C extensions). Add more if your agent needs them.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This assumes your main.py and any sub-modules (like animation_logic for ttv_agent)
# are in the same directory as this Dockerfile.
# If shared_ai_libs is used, you'll need a strategy to include it,
# e.g., by adjusting the COPY path or building from the root of uplas-ai-agents.
# For simplicity now, assuming agent code is self-contained or shared_ai_libs is handled
# by PYTHONPATH adjustments or by copying it in during the build if the context is the repo root.
COPY . .

# Expose the port the app runs on (FastAPI default is 8000, but our agents use 8001, 8002, etc.)
# This will be overridden by the PORT env var Cloud Run provides, but it's good practice.
# The CMD below will use the PORT environment variable.
# EXPOSE 8000 # Placeholder, Cloud Run sets this.

# Define the command to run the application
# Uvicorn will listen on 0.0.0.0 and the port specified by the PORT environment variable,
# which is automatically provided by Cloud Run.
# Replace 'main:app' if your FastAPI app instance is named differently or in a different file.
# The PORT environment variable is automatically set by Cloud Run.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
