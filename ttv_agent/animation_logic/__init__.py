# uplas-ai-agents/ttv_agent/animation_logic/__init__.py
from .character_manager import (
    InstructorChars,
    get_character_config,
    get_avatar_service_id,
    get_voice_settings_for_character, # Renamed for clarity
    get_attire_id_for_character,      # Renamed for clarity
    CharacterConfigError
)
from .avatar_api_client import ThirdPartyAvatarAPIClient, AvatarJobError

__all__ = [
    "InstructorChars",
    "get_character_config",
    "get_avatar_service_id",
    "get_voice_settings_for_character",
    "get_attire_id_for_character",
    "CharacterConfigError",
    "ThirdPartyAvatarAPIClient",
    "AvatarJobError"
]
