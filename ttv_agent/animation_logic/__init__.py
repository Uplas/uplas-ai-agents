# uplas-ai-agents/ttv_agent/animation_logic/__init__.py
"""
InnovateAI: Animation Logic Sub-package for the TTV Agent.
Manages character configurations, attire selection, and communication
with the third-party avatar generation service.
"""
from .character_manager import (
    InstructorChars,
    get_character_config,
    get_avatar_service_id, # Renamed from get_character_avatar_id_from_service in its own file in main.py import
    get_voice_settings,    # Renamed from get_character_voice_settings
    get_attire_id,         # Renamed from get_character_attire_id
    CharacterConfigError
)
from .avatar_api_client import (
    ThirdPartyAvatarAPIClient,
    AvatarJobError
)

__all__ = [
    "InstructorChars",
    "get_character_config",
    "get_avatar_service_id",
    "get_voice_settings",
    "get_attire_id",
    "CharacterConfigError",
    "ThirdPartyAvatarAPIClient",
    "AvatarJobError"
]
