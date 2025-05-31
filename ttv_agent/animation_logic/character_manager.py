# uplas-ai-agents/ttv_agent/animation_logic/character_manager.py
import json
import os
import random
from typing import Dict, Optional, Any, List
from enum import Enum
import logging # InnovateAI: Added for better logging

logger = logging.getLogger(__name__) # InnovateAI: Added logger

# Define character names as an Enum for type safety and clarity
class InstructorChars(str, Enum): #
    SUSAN = "susan"
    UNCLE_TREVOR = "uncle_trevor"

# InnovateAI: Path construction is good.
BASE_CHARACTER_ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "character_assets") #
_character_configs_cache: Dict[str, Dict[str, Any]] = {} #

class CharacterConfigError(Exception): #
    """Custom exception for character configuration issues."""
    pass

def _load_character_config_from_file(instructor_character_name: str) -> Dict[str, Any]: #
    """
    InnovateAI Enhanced: Loads character configuration from its JSON file.
    Raises CharacterConfigError if file not found, JSON is invalid, or essential fields are missing.
    """
    if not isinstance(instructor_character_name, str) or not instructor_character_name: #
        raise CharacterConfigError("Instructor character name must be a non-empty string.") #

    try: #
        # Validate against Enum if using it strictly
        InstructorChars(instructor_character_name) # Raises ValueError if not in Enum
    except ValueError: #
        # InnovateAI Note: Current behavior prints a warning. For stricter systems,
        # you might raise CharacterConfigError here if the name isn't in the Enum.
        logger.warning(f"InnovateAI Warning: Character name '{instructor_character_name}' is not a pre-defined InstructorChar. Proceeding with loading attempt.") #

    config_filename = f"{instructor_character_name}_config.json" #
    config_path = os.path.join(BASE_CHARACTER_ASSETS_PATH, instructor_character_name, config_filename) #

    if not os.path.exists(config_path): #
        logger.error(f"InnovateAI Error: Character config file not found at {config_path}") #
        raise CharacterConfigError(f"Character config file not found at {config_path}") #
    
    try: #
        with open(config_path, 'r', encoding='utf-8') as f: #
            config_data = json.load(f) #
        
        # InnovateAI Enhanced: More explicit validation of loaded config.
        # These 'service_avatar_id' and 'service_attire_id' inside attires MUST map to
        # actual IDs recognized by YOUR CHOSEN third-party avatar service.
        if not config_data.get("service_avatar_id"): #
            raise CharacterConfigError(f"Config for '{instructor_character_name}' is missing essential field 'service_avatar_id'.") #
        if "attires" not in config_data or not isinstance(config_data["attires"], list): #
             raise CharacterConfigError(f"Config for '{instructor_character_name}' must have an 'attires' list.")
        for i, attire in enumerate(config_data["attires"]):
            if not isinstance(attire, dict) or not attire.get("name") or not attire.get("service_attire_id"):
                raise CharacterConfigError(f"Attire at index {i} for '{instructor_character_name}' is malformed or missing 'name' or 'service_attire_id'.")

        logger.info(f"InnovateAI: Successfully loaded and validated config for character '{instructor_character_name}'.")
        return config_data #
    except json.JSONDecodeError as e: #
        logger.error(f"InnovateAI Error: Error decoding JSON from {config_path}: {e}", exc_info=True) #
        raise CharacterConfigError(f"Error decoding JSON from {config_path}: {e}") #
    except CharacterConfigError: # Re-raise specific config errors
        raise
    except Exception as e: # Catch other potential errors during file read or validation
        logger.error(f"InnovateAI Error: Error loading or validating character config for '{instructor_character_name}': {e}", exc_info=True) #
        raise CharacterConfigError(f"Error loading or validating character config for '{instructor_character_name}': {e}") #


def get_character_config(instructor_character_name: str) -> Dict[str, Any]: #
    """
    InnovateAI Enhanced: Retrieves character configuration, using a cache.
    Loads from file if not already cached.
    Raises CharacterConfigError if config is not found or is invalid.
    """
    if not isinstance(instructor_character_name, str) or not instructor_character_name:
        # InnovateAI: Adding check here as well for robustness, though _load should catch it.
        raise CharacterConfigError("Instructor character name must be a non-empty string for get_character_config.")

    # InnovateAI Suggestion: Use the Enum's value for cache key consistency if you validated against Enum.
    # char_key = InstructorChars(instructor_character_name).value
    char_key = instructor_character_name # Current approach allows string names not in Enum too, per warning

    if char_key not in _character_configs_cache: #
        logger.info(f"InnovateAI: Config for '{char_key}' not in cache. Loading from file.")
        config = _load_character_config_from_file(char_key) # _load raises on error
        _character_configs_cache[char_key] = config #
    return _character_configs_cache[char_key] #


def get_avatar_service_id(instructor_character_name: str) -> str: #
    """
    InnovateAI: Gets the service-specific avatar/character model ID for a given Uplas character.
    This ID MUST match an existing avatar model ID in your chosen third-party avatar service.
    """
    config = get_character_config(instructor_character_name) #
    # Validation in _load_character_config_from_file ensures "service_avatar_id" exists.
    return config["service_avatar_id"] #


def get_voice_settings(instructor_character_name: str, requested_language_code: Optional[str] = None) -> Dict[str, str]: #
    """
    InnovateAI Enhanced: Gets voice-related settings defined in the character's config.
    This is primarily useful if the third-party avatar service has its own internal TTS
    or requires specific voice hints for lip-sync even when using pre-generated audio.
    For Uplas, the main TTS is handled by the dedicated TTS Agent.

    Args:
        instructor_character_name: The Uplas internal name of the character.
        requested_language_code: Optional. BCP-47 language code. If provided, and if a
                                 mapping exists in `voice_id_map` in the config, it might
                                 return a service-specific voice ID for that language.

    Returns:
        A dictionary of voice settings, e.g., {"service_tts_engine": "...", "service_voice_id": "...", "language_code": "..."}.
        Returns default_voice_settings from config if no specific language override is found.
    """
    config = get_character_config(instructor_character_name) #
    default_settings = config.get("default_voice_settings", {}) #
    
    # If a specific language is requested and a mapping exists in the character config for it
    if requested_language_code and "voice_id_map" in config and requested_language_code in config["voice_id_map"]: #
        logger.info(f"InnovateAI: Found specific voice ID in config for character '{instructor_character_name}' and language '{requested_language_code}'.")
        return { #
            "service_tts_engine": default_settings.get("service_tts_engine", "unknown_tts_engine_from_service"), # Or from voice_id_map if specified there
            "service_voice_id": config["voice_id_map"][requested_language_code], #
            "language_code": requested_language_code #
        }
    logger.info(f"InnovateAI: Using default voice settings for character '{instructor_character_name}'. Lang requested: {requested_language_code}")
    return default_settings #


def get_attire_id( #
    instructor_character_name: str,
    preferred_attire_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    use_default_if_preferred_missing: bool = True #
) -> Optional[str]:
    """
    InnovateAI Enhanced: Selects a service-specific attire ID for the character based on preference, tags, or defaults.
    The selected ID MUST match an existing attire ID for the character in your chosen third-party avatar service.

    Order of preference for selection:
    1. `preferred_attire_name` if provided and found in the character's config.
    2. An attire matching one or more of the provided `tags` (randomly chosen if multiple match).
    3. The `default_attire_name` specified in the character's config.
    4. If all above fail, a random valid attire from the character's list.
    
    Raises CharacterConfigError if no suitable attire can be found.
    Returns None if no attires are defined at all (should not happen if config validation is good).
    """
    config = get_character_config(instructor_character_name) #
    attires_list = config.get("attires", []) #
    if not attires_list: #
        logger.warning(f"InnovateAI Warning: No attires defined for character '{instructor_character_name}'. Cannot select attire.") #
        return None #

    # 1. Try preferred_attire_name
    if preferred_attire_name: #
        for attire in attires_list: #
            if attire.get("name") == preferred_attire_name and attire.get("service_attire_id"): #
                logger.info(f"InnovateAI: Selected preferred attire '{preferred_attire_name}' for '{instructor_character_name}'.")
                return attire["service_attire_id"] #
        # If preferred was given but not found, and we shouldn't fallback to other methods
        if not use_default_if_preferred_missing: #
            logger.warning(f"InnovateAI Warning: Preferred attire '{preferred_attire_name}' not found for '{instructor_character_name}' and no fallback allowed.") #
            raise CharacterConfigError(f"Preferred attire '{preferred_attire_name}' not found for '{instructor_character_name}' and fallback is disabled.")

    # 2. Try tags if no preferred_attire_name was successfully matched (or not provided)
    if tags: #
        # InnovateAI: Ensure tags are lowercase for case-insensitive matching if desired,
        # or ensure config tags are consistently cased. Current logic is case-sensitive.
        processed_tags = [tag.lower() for tag in tags if isinstance(tag, str)] # Example processing
        tagged_attires = [ #
            attire for attire in attires_list 
            if attire.get("service_attire_id") and 
               any(tag_in_config.lower() in processed_tags for tag_in_config in attire.get("tags", []))
        ]
        if tagged_attires: #
            selected_attire = random.choice(tagged_attires) # Choose randomly from matching tagged attires
            logger.info(f"InnovateAI: Selected attire '{selected_attire.get('name')}' for '{instructor_character_name}' based on tags: {tags}.")
            return selected_attire["service_attire_id"] #
        else: #
            logger.info(f"InnovateAI Info: No attires found for character '{instructor_character_name}' with tags {tags}. Attempting default.") #

    # 3. Fallback to default_attire_name from config
    default_attire_name = config.get("default_attire_name") #
    if default_attire_name: #
        for attire in attires_list: #
            if attire.get("name") == default_attire_name and attire.get("service_attire_id"): #
                logger.info(f"InnovateAI: Selected default attire '{default_attire_name}' for '{instructor_character_name}'.")
                return attire["service_attire_id"] #
    
    # 4. If all else fails, pick any valid random attire as a last resort
    valid_attires_with_ids = [attire for attire in attires_list if attire.get("service_attire_id")] #
    if valid_attires_with_ids: #
        selected_random_attire = random.choice(valid_attires_with_ids) #
        logger.warning( #
            f"InnovateAI Warning: Could not find preferred, tagged, or default attire for '{instructor_character_name}'. "
            f"Picking a random valid one: '{selected_random_attire.get('name')}'."
        )
        return selected_random_attire["service_attire_id"] #

    # If we reach here, no valid attires with service_attire_id were found at all.
    logger.error(f"InnovateAI Critical: No suitable or fallback attire ID with a 'service_attire_id' found for character '{instructor_character_name}'.")
    raise CharacterConfigError(f"No suitable attire ID found for character '{instructor_character_name}' based on any criteria. Check config.") #

# InnovateAI Suggestion: You could add more functions here if needed, e.g.,
# - To get a list of all available attire names/tags for a character (for UI selection).
# - To get character-specific animation hints if your avatar service supports them (e.g., "idle_animation_style").

if __name__ == '__main__':
    # Example Usage and Testing
    print("InnovateAI: Testing CharacterManager functions...")
    try:
        susan_config = get_character_config(InstructorChars.SUSAN.value)
        print(f"\nSusan's Config (from cache on second call): {get_character_config(InstructorChars.SUSAN.value)}")
        print(f"Susan's Service Avatar ID: {get_avatar_service_id(InstructorChars.SUSAN.value)}")
        
        print(f"Susan's Default Voice Settings: {get_voice_settings(InstructorChars.SUSAN.value)}")
        # Assuming your susan_config.json has a voice_id_map for "fr-FR"
        # print(f"Susan's French Voice Settings: {get_voice_settings(InstructorChars.SUSAN.value, 'fr-FR')}")

        print(f"Susan's Attire (default logic): {get_attire_id(InstructorChars.SUSAN.value)}")
        print(f"Susan's Attire (preferred 'professional_blazer_blue'): {get_attire_id(InstructorChars.SUSAN.value, preferred_attire_name='professional_blazer_blue')}")
        print(f"Susan's Attire (tags ['formal']): {get_attire_id(InstructorChars.SUSAN.value, tags=['formal'])}")
        print(f"Susan's Attire (tags ['non_existent_tag']): {get_attire_id(InstructorChars.SUSAN.value, tags=['non_existent_tag'])}")

        trevor_config = get_character_config(InstructorChars.UNCLE_TREVOR.value)
        print(f"\nUncle Trevor's Service Avatar ID: {get_avatar_service_id(InstructorChars.UNCLE_TREVOR.value)}")
        print(f"Uncle Trevor's Attire (tags ['storytelling', 'friendly']): {get_attire_id(InstructorChars.UNCLE_TREVOR.value, tags=['storytelling', 'friendly'])}")

        # Test loading a non-enum character name (if warning is acceptable)
        # print(f"\nTesting non-enum name (will warn if strict enum validation is off in _load): {get_character_config('random_char_name_test')}")

    except CharacterConfigError as e:
        print(f"CharacterManager Test ERROR: {e}")
    except Exception as e_global:
        print(f"CharacterManager Test UNEXPECTED ERROR: {e_global}")
