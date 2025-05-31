# uplas-ai-agents/tts_agent/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union 
import os
import uuid
import time
import random # For mock audio content generation
from enum import Enum
import logging
from io import BytesIO 

# GCP Clients
from google.cloud import texttospeech_v1 as texttospeech # Explicit import for clarity
from google.cloud import storage
import google.auth

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TTS_AUDIO_GCS_BUCKET_NAME = os.getenv("TTS_AUDIO_GCS_BUCKET_NAME")

# Initialize GCP clients
if not GCP_PROJECT_ID:
    logging.warning("NovaSpark Warning: GCP_PROJECT_ID environment variable not set. TTS Agent may not function correctly.")
if not TTS_AUDIO_GCS_BUCKET_NAME:
    logging.warning("NovaSpark Warning: TTS_AUDIO_GCS_BUCKET_NAME environment variable not set. Audio storage will fail.")

try:
    tts_client = texttospeech.TextToSpeechAsyncClient()
    storage_client = storage.Client(project=GCP_PROJECT_ID if GCP_PROJECT_ID else None)
    logging.info(f"NovaSpark: TTS and Storage clients initialized. Project: {GCP_PROJECT_ID}, Bucket: {TTS_AUDIO_GCS_BUCKET_NAME}")
except Exception as e:
    logging.error(f"NovaSpark Critical: Error initializing GCP clients for TTS Agent: {e}", exc_info=True)
    tts_client = None
    storage_client = None

SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] 
DEFAULT_LANGUAGE = "en-US" 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

# --- Pydantic Models for API Contract ---
class VoiceSelectionParams(BaseModel): 
    voice_character_name: str = Field(..., examples=["susan_us_standard", "trevor_us_wavenet", "elodie_fr_standard", "priya_in_wavenet"])
    speaking_rate: Optional[float] = Field(1.0, ge=0.25, le=4.0, description="Speaking_rate/speed, 1.0 is normal.")
    pitch: Optional[float] = Field(0.0, ge=-20.0, le=20.0, description="Speaking pitch, 0.0 is normal.")
    effects_profile_id: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of strings representing the audio profile, e.g., ['wearable-class-device']."
    )

class AudioEncodingEnum(str, Enum): 
    MP3 = "MP3"
    LINEAR16 = "LINEAR16"
    OGG_OPUS = "OGG_OPUS"

class AudioConfig(BaseModel): 
    audio_encoding: AudioEncodingEnum = Field(AudioEncodingEnum.MP3)
    sample_rate_hertz: Optional[int] = Field(
        None,
        description="Optional. Synthesis sample rate (Hz). If not provided, service chooses default based on voice."
    )

class SynthesisInputType(str, Enum):
    TEXT = "text"
    SSML = "ssml"

class SynthesizeSpeechRequest(BaseModel): 
    input_type: SynthesisInputType = Field(SynthesisInputType.TEXT, description="Specify if the input content is plain text or SSML.")
    content_to_synthesize: str = Field(..., min_length=1, max_length=5000, examples=["Hello Uplas!", "<speak>Hello <emphasis>Uplas</emphasis>!</speak>"])
    language_code: Optional[str] = Field(
        None,
        examples=SUPPORTED_LANGUAGES,
        description="BCP-47 language tag. If None, voice character's default language takes precedence."
    )
    voice_params: VoiceSelectionParams
    audio_config: Optional[AudioConfig] = Field(default_factory=AudioConfig)
    prefer_premium_quality: Optional[bool] = Field(False, description="Attempt to use WaveNet/Studio voice if available for the character, subject to free tier/cost considerations.")
    context_course_id: Optional[str] = Field(None, description="Optional context for logging/analytics.") 
    context_topic_id: Optional[str] = Field(None, description="Optional context for logging/analytics.") 

    @validator('language_code', always=True)
    def validate_language_code(cls, v, values): 
        if v is None: # If None, it means use the character's default language.
            return v 
        if v not in SUPPORTED_LANGUAGES:
            # Log the warning but allow the request to proceed; voice selection logic will handle fallbacks.
            logger.warning(f"NovaSpark Warning: Unsupported language_code '{v}' in request. Voice selection will attempt best match or character's default.")
            # Returning None will force get_voice_config_from_character to use the character's inherent language or fallback.
            return None 
        return v

class SynthesizeSpeechResponse(BaseModel): 
    audio_url: str
    audio_duration_seconds: Optional[float] 
    voice_used_details: Dict[str, Any]
    text_character_count: int 
    processing_time_ms: Optional[float]

# --- NovaSpark Refined Uplas Voice Character Mapping ---
# This map defines Uplas-specific character names and links them to Google TTS voice configurations.
# 'quality_tier': 'standard', 'wavenet', 'studio'
# Justifications:
# - Standard voices are chosen for broad free-tier usage. Specific variants (A, B, C, D, E, F) are
#   selected based on common perception of clarity and gender typical for "Susan" (female) or "Trevor" (male) archetypes.
# - WaveNet/Studio voices are chosen as premium alternatives known for more natural-sounding speech.
#   Studio voices are generally preferred over WaveNet if available and of good quality for the language.
# The TTS Quality Audit (outlined separately) is CRITICAL to validate these choices.
UPLAS_VOICE_CHARACTER_MAP: Dict[str, Dict[str, Any]] = {
    # --- English (en-US) ---
    # Susan (Female, Professional, Clear)
    "susan_us_standard": {"google_voice_name": "en-US-Standard-C", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Clear, standard female voice."},
    "susan_us_wavenet":  {"google_voice_name": "en-US-Wavenet-F",  "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "Natural, high-quality female WaveNet voice."},
    "susan_us_studio":   {"google_voice_name": "en-US-Studio-O",   "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "studio", "notes": "Premium female Studio voice, often very expressive."},
    # Trevor (Male, Wise, Avuncular)
    "trevor_us_standard":{"google_voice_name": "en-US-Standard-D", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "notes": "Clear, standard male voice."},
    "trevor_us_wavenet": {"google_voice_name": "en-US-Wavenet-D",  "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "notes": "Natural, high-quality male WaveNet voice."},
    "trevor_us_studio":  {"google_voice_name": "en-US-Studio-M",   "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "studio", "notes": "Premium male Studio voice, often authoritative and warm."},
    # Generic/Neutral (if needed, Studio-M is quite versatile)
    "neutral_us_studio": {"google_voice_name": "en-US-Studio-M",   "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "studio", "notes": "Versatile premium voice, can sound fairly neutral."},


    # --- French (fr-FR) ---
    # Elodie (Female)
    "elodie_fr_standard": {"google_voice_name": "fr-FR-Standard-A", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Standard French female voice."},
    "elodie_fr_wavenet":  {"google_voice_name": "fr-FR-Wavenet-A",  "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "WaveNet French female voice."},
    "elodie_fr_studio":   {"google_voice_name": "fr-FR-Studio-A",   "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "studio", "notes": "Premium Studio French female voice (if available and superior to Wavenet)."},
    # Antoine (Male)
    "antoine_fr_standard":{"google_voice_name": "fr-FR-Standard-B", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "notes": "Standard French male voice."},
    "antoine_fr_wavenet": {"google_voice_name": "fr-FR-Wavenet-B",  "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "notes": "WaveNet French male voice."},
    "antoine_fr_studio":  {"google_voice_name": "fr-FR-Studio-D",   "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "studio", "notes": "Premium Studio French male voice."},

    # --- Spanish (es-ES) ---
    # Sofia (Female)
    "sofia_es_standard": {"google_voice_name": "es-ES-Standard-A", "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Standard Spanish (Spain) female voice."},
    "sofia_es_wavenet":  {"google_voice_name": "es-ES-Wavenet-A",  "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "WaveNet Spanish (Spain) female voice."},
    # Javier (Male) - New character example for male voice
    "javier_es_standard": {"google_voice_name": "es-ES-Standard-B", "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "notes": "Standard Spanish (Spain) male voice."},
    "javier_es_wavenet":  {"google_voice_name": "es-ES-Wavenet-B",  "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "notes": "WaveNet Spanish (Spain) male voice."},


    # --- German (de-DE) ---
    # Lena (Female)
    "lena_de_standard": {"google_voice_name": "de-DE-Standard-A", "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Standard German female voice (Standard-A/F variants often exist)."},
    "lena_de_wavenet":  {"google_voice_name": "de-DE-Wavenet-F",  "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "WaveNet German female voice (Wavenet-F is a common high-quality female)."},
    # Max (Male) - New character example
    "max_de_standard":  {"google_voice_name": "de-DE-Standard-B", "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "notes": "Standard German male voice."},
    "max_de_wavenet":   {"google_voice_name": "de-DE-Wavenet-B",  "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "notes": "WaveNet German male voice."},

    # --- Portuguese (pt-BR) ---
    # Isabela (Female)
    "isabela_br_standard": {"google_voice_name": "pt-BR-Standard-A", "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Standard Brazilian Portuguese female voice."},
    "isabela_br_wavenet":  {"google_voice_name": "pt-BR-Wavenet-A",  "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "WaveNet Brazilian Portuguese female voice."},
    # Rafael (Male) - New character example
     "rafael_br_standard": {"google_voice_name": "pt-BR-Standard-B", "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "notes": "Standard Brazilian Portuguese male voice."},
    "rafael_br_wavenet":  {"google_voice_name": "pt-BR-Wavenet-B",  "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "notes": "WaveNet Brazilian Portuguese male voice."},


    # --- Chinese (Simplified Mandarin - cmn-CN) ---
    # Meilin (Female)
    "meilin_cn_standard": {"google_voice_name": "cmn-CN-Standard-A", "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Standard Mandarin Chinese female voice."},
    "meilin_cn_wavenet":  {"google_voice_name": "cmn-CN-Wavenet-A",  "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "WaveNet Mandarin Chinese female voice."},
    # Jian (Male) - New character example
    "jian_cn_standard":{"google_voice_name": "cmn-CN-Standard-B", "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "notes": "Standard Mandarin Chinese male voice."},
    "jian_cn_wavenet": {"google_voice_name": "cmn-CN-Wavenet-B",  "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "notes": "WaveNet Mandarin Chinese male voice."},


    # --- Hindi (hi-IN) ---
    # Priya (Female)
    "priya_in_standard": {"google_voice_name": "hi-IN-Standard-A", "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Standard Hindi female voice."},
    "priya_in_wavenet":  {"google_voice_name": "hi-IN-Wavenet-A",  "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "WaveNet Hindi female voice."},
    # Rohan (Male) - New character example
    "rohan_in_standard":{"google_voice_name": "hi-IN-Standard-B", "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "notes": "Standard Hindi male voice."},
    "rohan_in_wavenet": {"google_voice_name": "hi-IN-Wavenet-B",  "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "notes": "WaveNet Hindi male voice."},
    

    # --- Default Fallback Voices (Ensure these exist and are generally acceptable) ---
    "default_en_us_standard": {"google_voice_name": "en-US-Standard-C", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "notes": "Default fallback English female voice."},
    "default_en_us_wavenet":  {"google_voice_name": "en-US-Wavenet-C",  "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "notes": "Default fallback English female WaveNet voice."},
}

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas Text-to-Speech (TTS) Agent - NovaSpark Multilingual Edition",
    description="Converts text or SSML to speech using Google Cloud TTS API, with dynamic voice & quality selection, GCS storage, and NovaSpark's multilingual enhancements.",
    version="0.4.0" # Incremented version
)

# --- Helper: Get Voice Configuration (NovaSpark Refined) ---
def get_voice_config_from_character(
    uplas_character_name_requested: str, 
    requested_language_code: Optional[str] = None, 
    prefer_premium: bool = False
) -> Dict[str, Any]:
    """
    NovaSpark Enhanced: Selects Google TTS voice config based on Uplas character name,
    language, and quality preference.
    The `uplas_character_name_requested` can be a base name (e.g., "susan_us") or a
    specific tiered name (e.g., "susan_us_wavenet").
    """
    logger.debug(f"get_voice_config called with: char_req='{uplas_character_name_requested}', lang_req='{requested_language_code}', premium='{prefer_premium}'")
    
    # Attempt direct match first (if user provided a fully qualified name like "susan_us_wavenet")
    direct_match_config = UPLAS_VOICE_CHARACTER_MAP.get(uplas_character_name_requested.lower())
    if direct_match_config:
        # If language is also requested, ensure it's compatible or ignore if character implies language
        if requested_language_code and direct_match_config["language_code"] != requested_language_code:
            logger.warning(f"NovaSpark: Direct match for '{uplas_character_name_requested}' found, but its lang '{direct_match_config['language_code']}' mismatches requested lang '{requested_language_code}'. Proceeding with direct match's language.")
            # Priority to character's inherent language if fully specified.
        logger.info(f"NovaSpark: Direct voice config match found for '{uplas_character_name_requested}': '{direct_match_config['google_voice_name']}' ({direct_match_config['quality_tier']})")
        return direct_match_config

    # If direct match fails or wasn't specific enough, try to build from base name + tier preference
    # Assuming character name might be "susan_us", "elodie_fr", etc. (base + lang_code)
    # Or it could be just "susan", "elodie" if language_code is reliably provided.
    # For simplicity, we'll assume `uplas_character_name_requested` is the base like "susan_us".
    
    base_char_name_lower = uplas_character_name_requested.lower()
    
    # Determine target language: either from request or infer from character base name (e.g., "_us", "_fr")
    target_language = requested_language_code 
    if not target_language: # Try to infer from character base if lang not in request
        for lang_suffix in [f"_{lc.split('-')[0].lower()}" for lc in SUPPORTED_LANGUAGES]: # e.g., _us, _fr
            if base_char_name_lower.endswith(lang_suffix):
                # Find the full BCP-47 code for this suffix
                for bcp47_lang in SUPPORTED_LANGUAGES:
                    if bcp47_lang.lower().startswith(base_char_name_lower.split('_')[-1]): # e.g. "us" in "en-US"
                        target_language = bcp47_lang
                        logger.debug(f"NovaSpark: Inferred target language '{target_language}' from character base '{base_char_name_lower}'")
                        break
                break
    
    if not target_language: # If still no language, use default
        target_language = DEFAULT_LANGUAGE
        logger.debug(f"NovaSpark: No specific language requested or inferred, defaulting to '{target_language}' for character '{base_char_name_lower}'.")


    # Construct search keys based on quality preference for the target language and base character type
    # This logic needs to be more robust if char names don't consistently include lang hints.
    # Example: "susan" could be "susan_us" or "susan_fr" - needs disambiguation or clear naming.
    # For now, we assume `base_char_name_lower` is somewhat unique for the character type (e.g. "susan_voice_profile")
    
    potential_configs: List[Dict[str, Any]] = []
    for key, config_val in UPLAS_VOICE_CHARACTER_MAP.items():
        # Check if config is for the target language AND if the character key contains the base name requested
        # This is a simple substring match for character type, might need refinement based on actual naming strategy
        if config_val["language_code"] == target_language and base_char_name_lower in key:
            potential_configs.append(config_val)
    
    if not potential_configs: # No configs match language and base character type
        logger.warning(f"NovaSpark: No voice configurations found for character base '{base_char_name_lower}' in language '{target_language}'. Falling back to language default.")
        # Fallback to first available voice for the target_language
        for key, config_val in UPLAS_VOICE_CHARACTER_MAP.items():
            if config_val["language_code"] == target_language:
                if prefer_premium and config_val["quality_tier"] in ["wavenet", "studio"]:
                    potential_configs.append(config_val)
                elif not prefer_premium and config_val["quality_tier"] == "standard":
                    potential_configs.append(config_val)
        # If still nothing, broaden further (e.g. any quality for the lang) - covered by final fallback.


    selected_config = None
    if potential_configs:
        # Sort by quality preference: studio > wavenet > standard
        quality_order = {"studio": 0, "wavenet": 1, "standard": 2}
        potential_configs.sort(key=lambda c: quality_order.get(c.get("quality_tier", "standard"), 99))

        if prefer_premium:
            selected_config = potential_configs[0] # Highest quality available for that char/lang
        else: # Prefer standard
            for conf in potential_configs:
                if conf.get("quality_tier") == "standard":
                    selected_config = conf
                    break
            if not selected_config: # If no standard, take best available (should be premium)
                selected_config = potential_configs[0]
    
    if selected_config:
        logger.info(f"NovaSpark: Voice config selected for '{base_char_name_lower}' (lang: {target_language}, premium: {prefer_premium}): '{selected_config['google_voice_name']}' ({selected_config['quality_tier']})")
        return selected_config
    else:
        # Ultimate fallback to a global default voice if absolutely nothing matches
        logger.error(f"NovaSpark Critical Fallback: No voice found for char '{base_char_name_lower}', lang '{target_language}'. Using global default '{DEFAULT_LANGUAGE}' standard voice.")
        default_key = f"default_{DEFAULT_LANGUAGE.lower().replace('-', '_')}_standard" #e.g. default_en_us_standard
        # Ensure this default key exists
        return UPLAS_VOICE_CHARACTER_MAP.get(default_key, UPLAS_VOICE_CHARACTER_MAP["default_en_us_standard"])


# --- API Endpoint (NovaSpark Enhanced) ---
@app.post("/v1/synthesize-speech", response_model=SynthesizeSpeechResponse, summary="Synthesize speech from text or SSML (NovaSpark Multilingual Edition)")
async def synthesize_speech_endpoint(request_data: SynthesizeSpeechRequest): 
    processing_start_time = time.perf_counter()

    if not tts_client or not storage_client:
        logger.error("NovaSpark Critical: TTS Client or Storage Client not initialized. Check GCP configuration.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTS service is not properly configured.")
    if not TTS_AUDIO_GCS_BUCKET_NAME:
        logger.error("NovaSpark Critical: TTS_AUDIO_GCS_BUCKET_NAME is not set. Cannot store audio.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Audio storage bucket not configured.")

    # 1. Determine Google TTS voice configuration
    voice_selection_details = get_voice_config_from_character(
        request_data.voice_params.voice_character_name,
        request_data.language_code, # This can be None
        request_data.prefer_premium_quality
    )
    # The actual language code used for synthesis is from the selected voice config
    synthesis_language_code = voice_selection_details["language_code"] 

    # 2. Prepare Synthesis Input
    if request_data.input_type == SynthesisInputType.SSML:
        synthesis_input = texttospeech.SynthesisInput(ssml=request_data.content_to_synthesize)
        logger.info(f"NovaSpark: Synthesizing speech from SSML input ({len(request_data.content_to_synthesize)} chars).")
    else: 
        synthesis_input = texttospeech.SynthesisInput(text=request_data.content_to_synthesize)
        logger.info(f"NovaSpark: Synthesizing speech from TEXT input ({len(request_data.content_to_synthesize)} chars).")

    # 3. Prepare Voice Selection Parameters
    voice_params_for_api = texttospeech.VoiceSelectionParams( 
        language_code=synthesis_language_code, # Use the language_code from the *selected voice*
        name=voice_selection_details["google_voice_name"],
        ssml_gender=voice_selection_details.get("ssml_gender", texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED)
    )

    # 4. Prepare Audio Configuration
    google_audio_encoding_map = { 
        AudioEncodingEnum.MP3: texttospeech.AudioEncoding.MP3,
        AudioEncodingEnum.LINEAR16: texttospeech.AudioEncoding.LINEAR16,
        AudioEncodingEnum.OGG_OPUS: texttospeech.AudioEncoding.OGG_OPUS,
    }
    audio_config_params_for_api = texttospeech.AudioConfig( 
        audio_encoding=google_audio_encoding_map[request_data.audio_config.audio_encoding],
        speaking_rate=request_data.voice_params.speaking_rate,
        pitch=request_data.voice_params.pitch,
        sample_rate_hertz=request_data.audio_config.sample_rate_hertz, # Optional
        effects_profile_id=request_data.voice_params.effects_profile_id or [] # Ensure it's a list
    )

    # 5. Call Google Cloud TTS API
    audio_content_bytes = None
    try: 
        logger.info(f"NovaSpark: Requesting synthesis. Voice: {voice_params_for_api.name} (Quality: {voice_selection_details['quality_tier']}), Lang: {voice_params_for_api.language_code}, Encoding: {request_data.audio_config.audio_encoding.value}")
        
        # --- NovaSpark: Actual call to tts_client.synthesize_speech ---
        # This replaces the placeholder from main (8).py
        api_response = await tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params_for_api,
            audio_config=audio_config_params_for_api
        )
        audio_content_bytes = api_response.audio_content
        # --- End actual call ---

        # # Mocked response for testing without API calls (REMOVE FOR PRODUCTION)
        # mock_audio_header = f"MockAudio_{voice_selection_details['quality_tier']}_{request_data.audio_config.audio_encoding.value}_"
        # audio_content_bytes = mock_audio_header.encode('utf-8') + os.urandom(1024 * random.randint(20,80)) 
        # logger.info(f"NovaSpark MOCK: Mock audio content generated ({len(audio_content_bytes)} bytes).")
        

        if not audio_content_bytes: 
            logger.error("NovaSpark Error: Speech synthesis produced no audio content bytes.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Speech synthesis produced no audio content.")
    except Exception as e: 
        logger.error(f"NovaSpark Error: Google Cloud TTS API call failed: {e}", exc_info=True)
        # Provide a more specific error if possible (e.g., from e.details() if grpc error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Speech synthesis service failed: {str(e)}")

    # 6. Store audio to GCS
    file_extension_map = { 
        AudioEncodingEnum.MP3: "mp3", AudioEncodingEnum.LINEAR16: "wav", AudioEncodingEnum.OGG_OPUS: "ogg"
    }
    file_extension = file_extension_map[request_data.audio_config.audio_encoding]
    # NovaSpark: Using a more structured blob name for better organization in GCS
    blob_name = f"tts_audio/{synthesis_language_code}/{request_data.voice_params.voice_character_name}/{uuid.uuid4()}.{file_extension}" 
    
    content_type_map = { 
        AudioEncodingEnum.MP3: "audio/mpeg", AudioEncodingEnum.LINEAR16: "audio/wav", AudioEncodingEnum.OGG_OPUS: "audio/ogg"
    }
    mime_type = content_type_map[request_data.audio_config.audio_encoding]
    
    gcs_public_url = ""
    try: 
        bucket = storage_client.bucket(TTS_AUDIO_GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        audio_file_like = BytesIO(audio_content_bytes) 
        
        # --- NovaSpark: Actual GCS upload ---
        # For async, consider blob.upload_from_file(..., client=storage_client) if library supports async context with it,
        # or use asyncio.to_thread for the blocking call.
        # Simple blocking call for now, assuming FastAPI runs workers that can handle this.
        # If performance becomes an issue, explore `aiohttp` with `google-auth` for fully async GCS.
        blob.upload_from_file(audio_file_like, content_type=mime_type)
        # For public access (adjust ACLs as needed, or use Signed URLs for better security)
        # blob.make_public() # Caution: Makes the object world-readable
        gcs_public_url = blob.public_url 
        # --- End actual GCS upload ---

        # # Mocked for testing without GCS calls (REMOVE FOR PRODUCTION)
        # gcs_public_url = f"https://storage.googleapis.com/{TTS_AUDIO_GCS_BUCKET_NAME}/{blob_name}" 
        # logger.info(f"NovaSpark MOCK: Mock audio upload to GCS. Assumed URL: {gcs_public_url}")


        logger.info(f"NovaSpark: Audio successfully uploaded to GCS: {gcs_public_url}")
    except Exception as e: 
        logger.error(f"NovaSpark Error: GCS upload failed for blob {blob_name}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to store synthesized audio: {str(e)}")

    # Estimate duration (rough estimate, can be improved if precise duration is critical)
    estimated_duration_seconds = 0.0
    if audio_content_bytes: # Check if there's actual content
        # Very rough estimate: assuming average bitrate. MP3 is variable. LINEAR16 is more predictable.
        # For LINEAR16: duration = num_bytes / (sample_rate_hertz * num_channels * bytes_per_sample)
        # For MP3: duration approx num_bytes / (bitrate_in_bytes_per_second)
        # Google TTS typically uses 24kHz for standard/wavenet MP3.
        sample_rate_for_estimation = request_data.audio_config.sample_rate_hertz or 24000
        if request_data.audio_config.audio_encoding == AudioEncodingEnum.LINEAR16:
            # Assuming 16-bit (2 bytes) per sample, mono (1 channel) for simplicity
            bytes_per_sample = 2 
            num_channels = 1 
            if sample_rate_for_estimation > 0: # Avoid division by zero
                 estimated_duration_seconds = len(audio_content_bytes) / (sample_rate_for_estimation * num_channels * bytes_per_sample)
        elif request_data.audio_config.audio_encoding == AudioEncodingEnum.MP3:
            # Assuming an average bitrate of 128 kbps (16 kBps) for estimation
            avg_bytes_per_second = 16000 
            estimated_duration_seconds = len(audio_content_bytes) / avg_bytes_per_second
        else: # OGG_OPUS or other
            estimated_duration_seconds = len(audio_content_bytes) / 16000 # Generic fallback estimator

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    return SynthesizeSpeechResponse( 
        audio_url=gcs_public_url, 
        audio_duration_seconds=round(estimated_duration_seconds, 2) if estimated_duration_seconds > 0 else None,
        voice_used_details={
            "uplas_voice_character_name_requested": request_data.voice_params.voice_character_name, 
            "google_voice_name_used": voice_params_for_api.name,
            "language_code_used": voice_params_for_api.language_code,
            "quality_tier_used": voice_selection_details.get("quality_tier", "unknown"), 
            "style_tags_available": voice_selection_details.get("notes", "") # Using notes as style tags for this example
        },
        text_character_count=len(request_data.content_to_synthesize),
        processing_time_ms=round(processing_time_ms, 2)
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK) 
async def health_check():
    # NovaSpark: More robust health check
    service_name = "TTS_Agent_NovaSpark_Multilingual"
    if not tts_client or not storage_client:
        return {"status": "unhealthy", "reason": "GCP clients not initialized properly.", "service": service_name, "innovate_ai_enhancements_active": True}
    if not GCP_PROJECT_ID or not TTS_AUDIO_GCS_BUCKET_NAME:
        return {"status": "unhealthy", "reason": "Essential GCP configuration (Project ID or GCS Bucket Name) is missing.", "service": service_name, "innovate_ai_enhancements_active": True}
    
    # Optional: Try a very small, non-costly check if clients are truly functional
    # e.g., list voices for a language (usually free metadata call)
    # try:
    #    await tts_client.list_voices(language_code="en-US") # Example check
    # except Exception as e_check:
    #    logger.error(f"NovaSpark Health Check: TTS client check failed: {e_check}")
    #    return {"status": "degraded", "reason": f"TTS client functional check failed: {e_check}", "service": service_name, "innovate_ai_enhancements_active": True}

    return {"status": "healthy", "service": service_name, "innovate_ai_enhancements_active": True}


if __name__ == "__main__": 
    import uvicorn
    logger.info("NovaSpark: Starting TTS Agent (NovaSpark Multilingual Edition) for local development...")
    if not GCP_PROJECT_ID: print("NovaSpark Warning: GCP_PROJECT_ID is not set.")
    if not TTS_AUDIO_GCS_BUCKET_NAME: print("NovaSpark Warning: TTS_AUDIO_GCS_BUCKET_NAME is not set.")
    
    port = int(os.getenv("PORT", 8002)) # Default from original file
    uvicorn.run(app, host="0.0.0.0", port=port)
