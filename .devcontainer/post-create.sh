#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Post-create script started ---"

# Update package lists and install any OS-level dependencies if needed by Python packages
# sudo apt-get update
# sudo apt-get install -y --no-install-recommends <your-os-packages-here>
# sudo rm -rf /var/lib/apt/lists/*

# Upgrade pip
python -m pip install --upgrade pip

# Install Python dependencies for all agents and shared_ai_libs
# This assumes your requirements.txt files are in the standard locations.
echo "Installing Python dependencies for AI Tutor Agent..."
pip install -r personalized_tutor_nlp_llm/requirements.txt

echo "Installing Python dependencies for TTS Agent..."
pip install -r tts_agent/requirements.txt

echo "Installing Python dependencies for TTV Agent..."
pip install -r ttv_agent/requirements.txt

echo "Installing Python dependencies for Project Generator Agent..."
pip install -r project_generator_agent/requirements.txt

if [ -f "shared_ai_libs/requirements.txt" ]; then
    echo "Installing Python dependencies for Shared AI Libraries..."
    pip install -r shared_ai_libs/requirements.txt
fi

# Install development tools like pytest, pytest-asyncio if not in individual requirements
echo "Installing common development tools..."
pip install pytest pytest-asyncio black flake8 isort pylint

# Authenticate to Google Cloud (optional here, can be done manually or via postAttachCommand)
# echo "Attempting to configure gcloud CLI..."
# gcloud auth configure-docker $(echo $GCP_LOCATION | cut -d'-' -f1,2)-docker.pkg.dev --quiet || echo "gcloud docker config failed, ensure you are logged in if needed for local Docker pushes from Codespace."
# echo "To authenticate gcloud for other commands, run: gcloud auth application-default login"

echo "--- Post-create script finished ---"

