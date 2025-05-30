# uplas-ai-agents/shared_ai_libs/main.py

# --- Shared Constants ---

# Supported languages (BCP-47 codes)
# This list should be the single source of truth for all agents.
SUPPORTED_LANGUAGES: List[str] = [
    "en-US",  # English (US)
    "fr-FR",  # French (France)
    "es-ES",  # Spanish (Spain)
    "de-DE",  # German (Germany)
    "pt-BR",  # Portuguese (Brazil)
    "zh-CN",  # Chinese (Simplified, Mainland China)
    "hi-IN",  # Hindi (India)
]

DEFAULT_LANGUAGE: str = "en-US"

# --- Potentially Shared Pydantic Models (Example - if needed later) ---
# from pydantic import BaseModel, Field
# from typing import Optional

# class CommonDebugInfo(BaseModel):
#     processing_time_ms: Optional[float] = None
#     # Add other common debug fields if they emerge

# --- Potentially Shared Utility Functions (Example - if needed later) ---
# import logging

# def get_configured_logger(name: str, level: int = logging.INFO) -> logging.Logger:
#     logging.basicConfig(level=level)
#     logger = logging.getLogger(name)
#     # Add any standard formatting or handlers here
#     return logger

# Ensure to add typing if not already present at the top for List
from typing import List

# If you add more complex utilities that require external libraries,
# remember to add them to uplas-ai-agents/shared_ai_libs/requirements.txt

