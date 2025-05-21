import json
import os
import random
from typing import Dict, Optional, Any, List

BASE_CHARACTER_ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "character_assets")

_character_configs: Dict[str, Dict[str, Any]] = {} # In-memory cache for configs

def load_character_config(instructor_character_name: str) -> Optional[Dict[str, Any]]:
    """Loads character configuration from JSON file."""
    if instructor_character_name in _character_configs:
        return _character_configs[instructor_character_name]

    config_path = os.path.join(BASE_CHARACTER_ASSETS_PATH, instructor_character_name, f"{instructor_character_name}_config.json")
    if not os.path.exists(config_path):
        print(f"Error: Character config not found for '{instructor_character_name}' at {config_path}")
        return None
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            _character_configs[instructor_character_name] = config # Cache it
            return config
    except Exception as e:
        print(f"Error loading character config for '{instructor_character_name}': {e}")
        return None

def get_avatar_details(instructor_character_name: str) -> Optional[Dict[str, str]]:
    """Gets the service_avatar_id and default voice settings for a character."""
    config = load_character_config(instructor_character_name)
    if config:
        return {
            "service_avatar_id": config.get("service_avatar_id"),
            "default_voice_settings": config.get("default_voice_settings")
        }
    return None

def get_random_attire_id(instructor_character_name: str, tags: Optional[List[str]] = None) -> Optional[str]:
    """
    Selects a random attire ID for the character, optionally filtered by tags.
    Implements the "dress them differently each day" concept by random choice.
    More sophisticated logic (e.g., daily rotation) could be added.
    """
    config = load_character_config(instructor_character_name)
    if not config or not config.get("attires"):
        return None

    available_attires = config["attires"]
    
    if tags:
        tagged_attires = [
            attire for attire in available_attires if attire.get("service_attire_id") and any(tag in attire.get("tags", []) for tag in tags)
        ]
        if tagged_attires:
            selected_attire = random.choice(tagged_attires)
            return selected_attire.get("service_attire_id")
        else:
            print(f"Warning: No attires found for character '{instructor_character_name}' with tags {tags}. Falling back.")

    # Fallback to random choice from all available attires if no tags match or no tags provided
    valid_attires = [attire.get("service_attire_id") for attire in available_attires if attire.get("service_attire_id")]
    if valid_attires:
        return random.choice(valid_attires)
    
    print(f"Warning: No valid attires with service_attire_id found for {instructor_character_name}")
    return None

def get_default_attire_id(instructor_character_name: str) -> Optional[str]:
    config = load_character_config(instructor_character_name)
    if config:
        default_attire_name = config.get("default_attire_name")
        for attire_conf in config.get("attires", []):
            if attire_conf.get("name") == default_attire_name:
                return attire_conf.get("service_attire_id")
    return None
