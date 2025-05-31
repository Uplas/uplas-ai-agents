# uplas-ai-agents/shared_ai_libs/main.py
from enum import Enum

# Core constants shared across Uplas AI Agents

SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

# Example of a shared Enum if needed by multiple agents
# class CommonProcessingStatus(str, Enum):
#     PENDING = "pending"
#     PROCESSING = "processing"
#     SUCCESS = "success"
#     FAILURE = "failure"

# Example of a shared Pydantic model (illustrative - decide if needed)
# from pydantic import BaseModel, Field
# from typing import Optional, List, Dict

# class BaseUserProfile(BaseModel):
#     user_id: str
#     language_preference: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)

# Add any other truly shared constants or utility functions here.
# For instance, a utility function to format GCS URLs, or a common error class.
# Keep it lean and only include what's genuinely shared and stable.

if __name__ == '__main__':
    print(f"Uplas Shared AI Libs: Loaded with {len(SUPPORTED_LANGUAGES)} supported languages.")
    print(f"Default language: {DEFAULT_LANGUAGE}")
