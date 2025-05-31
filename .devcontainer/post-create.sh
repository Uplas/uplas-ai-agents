#!/bin/bash
echo "Uplas AI Agents: Post-create script started."

# Update package list and install common tools if needed
# sudo apt-get update
# sudo apt-get install -y some-common-tool

# Install dependencies for each agent
# Assuming this script is run from the repository root
echo "Installing dependencies for Personalized AI Tutor..."
pip install -r personalized_tutor_nlp_llm/requirements.txt

echo "Installing dependencies for TTS Agent..."
pip install -r tts_agent/requirements.txt

echo "Installing dependencies for TTV Agent..."
pip install -r ttv_agent/requirements.txt # Ensure this requirements exists at this path

echo "Installing dependencies for Project Generation & Assessment Agent..."
pip install -r project_generator_agent/requirements.txt

echo "Installing dependencies for NLP Content Agent..."
pip install -r nlp_content_agent/requirements.txt # Assuming we created this

echo "Installing dependencies for Shared AI Libs (if any)..."
# If shared_ai_libs had its own requirements.txt for dev tools like linters specific to it
# pip install -r shared_ai_libs/requirements.txt

echo "Uplas AI Agents: Post-create script finished."
# You might also want to run gcloud auth application-default login here
# or provide instructions for the user to do so once the Codespace is up.
