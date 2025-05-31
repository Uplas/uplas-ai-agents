# Dockerfile for Uplas NLP Content Structuring & Augmentation Agent
# InnovateAI Version: 1.0

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set common environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# InnovateAI Note on shared_ai_libs:
# This Dockerfile assumes that if this agent uses 'shared_ai_libs',
# that directory has been copied into the current build context (this agent's directory)
# by your CI/CD pipeline before this Docker build step is executed.
# e.g., your CI script might do: cp -R ../shared_ai_libs ./nlp_content_agent/shared_ai_libs

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# ENV PYTHONPATH "${PYTHONPATH}:/app" # Uncomment if needed for shared_ai_libs resolution

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
