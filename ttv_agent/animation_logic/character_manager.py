import json
import os
import random
from typing import Dict, Optional, Any, List
from enum import Enum # For character names

# Define character names as an Enum for type safety and clarity
class InstructorChars(str, Enum):
    SUSAN = "susan"
    UNCLE_TREVOR = "uncle_trevor"

BASE_CHARACTER_ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "character_assets")
_character_configs_cache: Dict[str, Dict[str, Any]] = {}

class CharacterConfigError(Exception):
    """Custom exception for character configuration issues."""
    pass

def _load_character_config_from_file(instructor_character_name: str) -> Optional[Dict[str, Any]]:
    """Loads character configuration from its JSON file."""
    if not isinstance(instructor_character_name, str) or not instructor_character_name:
        raise CharacterConfigError("Instructor character name must be a non-empty string.")

    try:
        # Validate against Enum if using it strictly, or allow any string and handle missing files
        InstructorChars(instructor_character_name) # Raises ValueError if not in Enum
    except ValueError:
        print(f"Warning: '{instructor_character_name}' is not a pre-defined InstructorChar. Proceeding with loading attempt.")
        # Or raise CharacterConfigError(f"Invalid character name: {instructor_character_name}")

    config_filename = f"{instructor_character_name}_config.json"
    config_path = os.path.join(BASE_CHARACTER_ASSETS_PATH, instructor_character_name, config_filename)

    if not os.path.exists(config_path):
        raise CharacterConfigError(f"Character config file not found at {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        # Basic validation of loaded config (can be more extensive with Pydantic models)
        if not config_data.get("service_avatar_id") or not config_data.get("attires"):
            raise CharacterConfigError(f"Config for '{instructor_character_name}' is missing essential fields 'service_avatar_id' or 'attires'.")
        return config_data
    except json.JSONDecodeError as e:
        raise CharacterConfigError(f"Error decoding JSON from {config_path}: {e}")
    except Exception as e: # Catch other potential errors during file read or basic validation
        raise CharacterConfigError(f"Error loading or validating character config for '{instructor_character_name}': {e}")


def get_character_config(instructor_character_name: str) -> Dict[str, Any]:
    """
    Retrieves character configuration, using a cache.
    Raises CharacterConfigError if config is not found or invalid.
    """
    if instructor_character_name not in _character_configs_cache:
        config = _load_character_config_from_file(instructor_character_name)
        if config is None: # Should be handled by _load_character_config_from_file raising error
            raise CharacterConfigError(f"Failed to load configuration for '{instructor_character_name}'.")
        _character_configs_cache[instructor_character_name] = config
    return _character_configs_cache[instructor_character_name]


def get_avatar_service_id(instructor_character_name: str) -> str:
    """Gets the service_avatar_id for a character."""
    config = get_character_config(instructor_character_name)
    return config["service_avatar_id"] # Assumes field exists due to validation in load


def get_voice_settings(instructor_character_name: str, requested_language_code: Optional[str] = None) -> Dict[str, str]:
    """Gets default voice settings, potentially overridden by requested language."""
    config = get_character_config(instructor_character_name)
    default_settings = config.get("default_voice_settings", {})
    
    # If a specific language is requested and a mapping exists for it, use it
    if requested_language_code and requested_language_code in config.get("voice_id_map", {}):
        return {
            "service_tts_engine": default_settings.get("service_tts_engine", "unknown_tts_engine"), # Or from voice_id_map if specified there
            "service_voice_id": config["voice_id_map"][requested_language_code],
            "language_code": requested_language_code
        }
    return default_settings


def get_attire_id(
    instructor_character_name: str,
    preferred_attire_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    use_default_if_preferred_missing: bool = True
) -> Optional[str]:
    """
    Selects an attire ID for the character.
    1. Tries preferred_attire_name if provided.
    2. If not found or not provided, tries to find one matching tags.
    3. If still not found, falls back to default_attire_name from config.
    4. If default also not found, returns None or raises error.
    """
    config = get_character_config(instructor_character_name)
    attires_list = config.get("attires", [])
    if not attires_list:
        print(f"Warning: No attires defined for character '{instructor_character_name}'.")
        return None

    # 1. Try preferred_attire_name
    if preferred_attire_name:
        for attire in attires_list:
            if attire.get("name") == preferred_attire_name and attire.get("service_attire_id"):
                return attire["service_attire_id"]
        if not use_default_if_preferred_missing: # If preferred was given but not found, and we shouldn't fallback
            print(f"Warning: Preferred attire '{preferred_attire_name}' not found for '{instructor_character_name}'.")
            # Decide to return None or raise error based on strictness needed
            # return None 

    # 2. Try tags if no preferred_attire_name or preferred not found (and fallback allowed)
    if tags:
        tagged_attires = [
            attire for attire in attires_list if attire.get("service_attire_id") and any(tag in attire.get("tags", []) for tag in tags)
        ]
        if tagged_attires:
            selected_attire = random.choice(tagged_attires) # Choose randomly from matching tagged attires
            return selected_attire["service_attire_id"]
        else:
            print(f"Info: No attires found for character '{instructor_character_name}' with tags {tags}. Attempting default.")

    # 3. Fallback to default_attire_name from config
    default_attire_name = config.get("default_attire_name")
    if default_attire_name:
        for attire in attires_list:
            if attire.get("name") == default_attire_name and attire.get("service_attire_id"):
                return attire["service_attire_id"]
    
    # 4. If all else fails, pick any valid random attire as a last resort or return None
    valid_attires_ids = [attire.get("service_attire_id") for attire in attires_list if attire.get("service_attire_id")]
    if valid_attires_ids:
        print(f"Warning: Could not find preferred, tagged, or default attire for '{instructor_character_name}'. Picking a random valid one.")
        return random.choice(valid_attires_ids)

    raise CharacterConfigError(f"No suitable attire ID found for character '{instructor_character_name}' based on criteria.")
