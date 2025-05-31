# Dockerfile for Uplas Groundbreaking Personalized TTV Agent
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
# Also, ensure character_assets are handled correctly by the COPY . . command
# if they are part of the build context for this agent.

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# This will copy main.py, animation_logic/, character_assets/,
# and shared_ai_libs/ (if pre-copied by CI/CD) into /app
COPY . .

# ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
