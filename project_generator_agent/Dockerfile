# Dockerfile for Uplas Personalized Project Generation & Assessment Agent
# InnovateAI Version: 1.0

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set common environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# InnovateAI Note on shared_ai_libs:
# (Same note as above regarding CI/CD pre-copying shared_ai_libs)

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
