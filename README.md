# Uplas AI Agents

Welcome to the Uplas AI Agents repository! This project houses a suite of specialized AI agents designed to power various intelligent features within the Uplas learning platform. These agents are built as FastAPI microservices, intended for cloud-native development and deployment on Google Cloud Platform (GCP).

## Project Overview

The Uplas platform aims to provide a rich, personalized, and interactive learning experience. These AI agents are the backbone of its intelligent capabilities, offering services ranging from personalized tutoring to dynamic content generation.

## AI Agents

This repository contains the following core AI agents:

1.  **Personalized NLP/LLM AI Tutor (`personalized_tutor_nlp_llm/`)**
    * **Purpose:** Provides personalized explanations, answers user questions, and offers guidance within learning modules and projects.
    * **Core Technology:** Leverages Google Cloud Vertex AI (Gemini models) for natural language understanding and generation.
    * **Features:**
        * Personalizes responses based on user profile (industry, profession, learning goals, etc.).
        * Generates analogies relevant to the user's background.
        * Supports multi-language responses.
        * Maintains conversation history for contextual understanding.
        * Can be triggered with specific context (e.g., course topic, project assessment feedback).

2.  **Text-to-Speech (TTS) Agent (`tts_agent/`)**
    * **Purpose:** Converts text content into natural-sounding speech.
    * **Core Technology:** Utilizes Google Cloud Text-to-Speech API.
    * **Features:**
        * Offers a variety of voice characters with different styles, tones, and accents.
        * Supports multi-language speech synthesis across 7 languages (English, French, Spanish, German, Portuguese, Chinese, Hindi).
        * Allows customization of speaking rate and pitch.
        * Stores generated audio files in Google Cloud Storage (GCS).

3.  **Text-to-Video (TTV) Agent (`ttv_agent/`)**
    * **Purpose:** Generates instructional videos featuring animated characters (Uncle Trevor and Susan).
    * **Core Technology:** Orchestrates script generation (via AI Tutor), audio synthesis (via TTS Agent), and video rendering via a **third-party avatar generation service**.
    * **Features:**
        * Creates 3-5 minute personalized instructional videos.
        * Features two distinct animated instructor characters: Susan and Uncle Trevor.
        * Supports dynamic attire changes for characters.
        * Generates videos in multiple languages.
        * Manages video generation as an asynchronous background job and notifies the backend via a callback upon completion.
    * **Note:** Requires integration with a chosen third-party avatar service that supports custom character animation and API control. The client for this is in `ttv_agent/animation_logic/avatar_api_client.py` and needs to be implemented based on the selected service.

4.  **AI Project Generator Agent (`project_generator_agent/`)**
    * **Purpose:** Generates personalized, real-world project ideas for learners and assesses their submissions.
    * **Core Technology:** Employs Google Cloud Vertex AI (Gemini models).
    * **Features:**
        * Generates project ideas tailored to user profiles, preferences, and desired difficulty.
        * Provides detailed project descriptions, key tasks, suggested technologies, and learning objectives in multiple languages.
        * Assesses user-submitted projects against the generated idea's criteria.
        * Provides constructive feedback on submissions.
        * Can trigger the Personalized AI Tutor if a project assessment indicates the user needs help.

5.  **Shared AI Libraries (`shared_ai_libs/`)**
    * **Purpose:** Contains common utility code, constants (e.g., supported languages), and potentially shared Pydantic models used across multiple AI agents.

## Development Environment

This project is designed for a **cloud-native development workflow**.

* **Primary Environment:** GitHub Codespaces is recommended. A configuration file for Codespaces can be found at `.devcontainer/devcontainer.json`. This ensures a consistent development environment for all contributors without requiring extensive local setup. [cite: 75, 77]
* **Local Fallback (if necessary):** If developing locally, ensure you have Python 3.9+ and `pip` installed. You'll also need to set up Google Cloud authentication (e.g., via `gcloud auth application-default login`).

## Setup and Running Agents

Each agent is a FastAPI application and resides in its own directory.

1.  **Navigate to an agent's directory:**
    ```bash
    cd <agent_directory_name>
    # e.g., cd personalized_tutor_nlp_llm
    ```

2.  **Install Dependencies:**
    Each agent has its own `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables:**
    Each agent requires specific environment variables to function correctly. These typically include:
    * `GCP_PROJECT_ID`: Your Google Cloud Project ID.
    * `GCP_LOCATION`: The GCP region for services like Vertex AI (e.g., `us-central1`).
    * **Agent-Specific Variables:**
        * **AI Tutor:** `LLM_MODEL_NAME` (e.g., `gemini-1.5-flash-001`).
        * **TTS Agent:** `TTS_AUDIO_GCS_BUCKET_NAME`.
        * **TTV Agent:** `TTV_GCS_BUCKET_NAME`, `DJANGO_TTV_CALLBACK_URL`, `AI_TUTOR_AGENT_URL`, `TTS_AGENT_URL`, `THIRD_PARTY_AVATAR_API_KEY`, `THIRD_PARTY_AVATAR_BASE_URL`.
        * **Project Generator:** `PROJECT_LLM_MODEL_NAME`, `ASSESSMENT_LLM_MODEL_NAME`, `AI_TUTOR_AGENT_URL`.
    * Refer to the top of each agent's `main.py` file for the specific environment variables it expects. For security, sensitive keys (like `THIRD_PARTY_AVATAR_API_KEY`) should be managed via secrets in your deployment environment or GitHub secrets for Actions.

4.  **Run the Agent (using Uvicorn):**
    Uvicorn is used to run the FastAPI applications. The `main.py` file in each agent directory often includes a `if __name__ == "__main__":` block for local execution.
    ```bash
    # Example for AI Tutor, from within personalized_tutor_nlp_llm/
    # Ensure required ENV VARS are set
    python main.py
    # OR directly with uvicorn:
    # uvicorn main:app --reload --port 8001
    ```
    * AI Tutor Agent: Typically runs on port `8001`.
    * TTS Agent: Typically runs on port `8002`.
    * TTV Agent: Typically runs on port `8003`.
    * Project Generator Agent: Typically runs on port `8004`.

    The specific port can be configured via the `PORT` environment variable, especially when deployed on Cloud Run.

## Testing

Each agent includes a `test_main.py` file with unit and integration tests using `pytest` and `unittest.mock`.

1.  **Install Test Dependencies:**
    Ensure `pytest` and `pytest-asyncio` are installed in your environment (they should be part of development dependencies or can be added to `requirements.txt` for testing).

2.  **Run Tests:**
    Navigate to the specific agent's directory and run:
    ```bash
    pytest
    ```

## Deployment

* **Target Platform:** Google Cloud Run is the primary deployment target for these FastAPI services.
* **Method:** Deployment is automated via GitHub Actions. Workflow files are located in `.github/workflows/`.
    * A template `deploy-agent-to-cloud-run.yml` has been outlined, which can be adapted for each agent.
* **Docker:** Each agent has a `Dockerfile` to package it as a container image. These images are pushed to Google Artifact Registry.
* **CI/CD:**
    * `ai-agent-ci.yml` (to be defined) can handle continuous integration (linting, running tests on push/PR).
    * The deployment workflow handles continuous deployment to Cloud Run upon pushes to the main branch (or other designated branches).

## Key Technologies

* Python 3.9+
* FastAPI (for building API services)
* Pydantic (for data validation)
* Google Cloud Platform (GCP):
    * Vertex AI (Gemini Models for LLM tasks)
    * Cloud Text-to-Speech API
    * Cloud Storage (GCS)
    * Cloud Run (for deploying services)
    * Artifact Registry (for Docker images)
* Docker
* GitHub Actions (for CI/CD)
* Third-Party Avatar Generation Service (to be chosen and integrated for TTV)

## Contribution

Details on contributing, coding standards, and branching strategy will be added here.

---

This README provides a good starting point. You can expand it further with more specific details as the project evolves.
