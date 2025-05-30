# uplas-ai-agents/personalized_tutor_nlp_llm/Dockerfile
# Assumes Docker build context is the root of the uplas-ai-agents repository.

FROM python:3.9-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Set PYTHONPATH to include the root /app directory where uplas_ai_agents package will reside
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Copy shared libraries first
# This creates /app/uplas_ai_agents/shared_ai_libs/
COPY shared_ai_libs /app/uplas_ai_agents/shared_ai_libs/
# Create __init__.py files to ensure uplas_ai_agents and shared_ai_libs are treated as packages
RUN mkdir -p /app/uplas_ai_agents && touch /app/uplas_ai_agents/__init__.py
RUN touch /app/uplas_ai_agents/shared_ai_libs/__init__.py

# Copy the specific agent's requirements file
COPY personalized_tutor_nlp_llm/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the specific agent's application code into a subdirectory matching its name
# This creates /app/uplas_ai_agents/personalized_tutor_nlp_llm/
COPY personalized_tutor_nlp_llm /app/uplas_ai_agents/personalized_tutor_nlp_llm/
RUN touch /app/uplas_ai_agents/personalized_tutor_nlp_llm/__init__.py


# The application will be run from personalized_tutor_nlp_llm.main
# Uvicorn will listen on the port specified by Cloud Run's $PORT environment variable.
CMD ["uvicorn", "uplas_ai_agents.personalized_tutor_nlp_llm.main:app", "--host", "0.0.0.0", "--port", "$PORT"]
