# Dockerfile for Uplas AI Agent
# InnovateAI Version: 1.0

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set common environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# InnovateAI Note on shared_ai_libs:
# This Dockerfile assumes that if your agent uses 'shared_ai_libs',
# that directory has been copied into the current build context (this agent's directory)
# by your CI/CD pipeline before this Docker build step is executed.
# For example, your CI script might do:
#   cp -R ../shared_ai_libs ./<current_agent_directory>/shared_ai_libs
# before running `docker build .` in ./<current_agent_directory>/.

# Copy the agent-specific requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the agent's application code into the container
# This will include main.py and any other Python files/subdirectories for this agent.
# If shared_ai_libs was copied into the build context, it will be copied here too.
COPY . .

# InnovateAI Note on PYTHONPATH:
# If shared_ai_libs is copied into /app/shared_ai_libs (due to `COPY . .` and CI pre-copy),
# and your main.py (now at /app/main.py) imports from it using
# `from shared_ai_libs import ...`, Python should find it if /app is in its search path.
# Setting PYTHONPATH explicitly can ensure this:
# ENV PYTHONPATH "${PYTHONPATH}:/app"
# However, if shared_ai_libs is structured to be installable or if your imports
# are relative within a larger package structure copied to /app, this might vary.
# For the simple case of `from shared_ai_libs.main import ...` where shared_ai_libs
# is a directory at /app/shared_ai_libs, it should generally work.

# Expose the port the app runs on (Cloud Run sets PORT env var automatically)
# EXPOSE 8080 # Default, but Cloud Run will use the $PORT var for the CMD

# Command to run the Uvicorn server for the FastAPI application
# Cloud Run provides the $PORT environment variable.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
