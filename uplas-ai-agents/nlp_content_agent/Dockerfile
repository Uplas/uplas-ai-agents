# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Add GCP_PROJECT_ID, GCP_LOCATION, NLP_LLM_MODEL_NAME via Cloud Run env vars

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if any (e.g., for specific Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
# This assumes your Dockerfile is in the nlp_content_agent directory
COPY . /app

# Command to run the application using Uvicorn
# PORT environment variable will be automatically set by Cloud Run.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
