# uplas-ai-agents/tts_agent/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union # Added Union
import os
import uuid
import time
from enum import Enum
import logging
from io import BytesIO # For handling audio content in memory

# GCP Clients
from google.cloud import texttospeech # v1
from google.cloud import storage
import google.auth

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TTS_AUDIO_GCS_BUCKET_NAME = os.getenv("TTS_AUDIO_GCS_BUCKET_NAME")

# Initialize GCP clients
if not GCP_PROJECT_ID:
    logging.warning("InnovateAI Warning: GCP_PROJECT_ID environment variable not set. TTS Agent may not function correctly.")
if not TTS_AUDIO_GCS_BUCKET_NAME:
    logging.warning("InnovateAI Warning: TTS_AUDIO_GCS_BUCKET_NAME environment variable not set. Audio storage will fail.")

try:
    # InnovateAI Note: Using TextToSpeechAsyncClient for non-blocking operations, as in your original code.
    tts_client = texttospeech.TextToSpeechAsyncClient()
    storage_client = storage.Client(project=GCP_PROJECT_ID if GCP_PROJECT_ID else None)
except Exception as e:
    logging.error(f"InnovateAI Critical: Error initializing GCP clients for TTS Agent: {e}", exc_info=True)
    tts_client = None
    storage_client = None

# Supported languages - Align with shared_ai_libs if it becomes the source of truth
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] #
DEFAULT_LANGUAGE = "en-US" #

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO) #
logger = logging.getLogger(__name__) #

# --- Pydantic Models for API Contract (Enhanced by InnovateAI) ---

class VoiceSelectionParams(BaseModel): #
    voice_character_name: str = Field(..., examples=["susan_us_standard", "trevor_us_wavenet", "elodie_fr_standard"])
    speaking_rate: Optional[float] = Field(1.0, ge=0.25, le=4.0, description="Speaking_rate/speed, 1.0 is normal.")
    pitch: Optional[float] = Field(0.0, ge=-20.0, le=20.0, description="Speaking pitch, 0.0 is normal.")
    effects_profile_id: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of strings representing the audio profile, e.g., ['wearable-class-device']."
    )

class AudioEncodingEnum(str, Enum): #
    MP3 = "MP3"
    LINEAR16 = "LINEAR16"
    OGG_OPUS = "OGG_OPUS"

class AudioConfig(BaseModel): #
    audio_encoding: AudioEncodingEnum = Field(AudioEncodingEnum.MP3)
    sample_rate_hertz: Optional[int] = Field(
        None,
        description="Optional. Synthesis sample rate (Hz). If not provided, service chooses default based on voice."
    )

# InnovateAI Enhancement: Added input_type to distinguish TEXT and SSML
class SynthesisInputType(str, Enum):
    TEXT = "text"
    SSML = "ssml"

class SynthesizeSpeechRequest(BaseModel): # Enhanced from
    input_type: SynthesisInputType = Field(SynthesisInputType.TEXT, description="Specify if the input content is plain text or SSML.")
    content_to_synthesize: str = Field(..., min_length=1, max_length=5000, examples=["Hello Uplas!", "<speak>Hello <emphasis>Uplas</emphasis>!</speak>"])
    language_code: Optional[str] = Field(
        None,
        examples=SUPPORTED_LANGUAGES,
        description="BCP-47 language tag. If None or different from character's default, character's default takes precedence or best match is attempted."
    )
    voice_params: VoiceSelectionParams
    audio_config: Optional[AudioConfig] = Field(default_factory=AudioConfig)
    # InnovateAI Enhancement: Flag to explicitly request higher quality voice tier
    prefer_premium_quality: Optional[bool] = Field(False, description="Attempt to use WaveNet/Studio voice if available for the character, subject to free tier/cost considerations.")
    context_course_id: Optional[str] = Field(None, description="Optional context for logging/analytics.") #
    context_topic_id: Optional[str] = Field(None, description="Optional context for logging/analytics.") #


    @validator('language_code', always=True)
    def validate_language_code(cls, v, values): #
        if v is None:
            return v # Let voice selection logic handle it
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"InnovateAI Warning: Unsupported language_code '{v}'. Voice selection logic will attempt to find best match or use character's default.")
            return None # Mark as invalid, get_voice_config will handle fallback
        return v

class SynthesizeSpeechResponse(BaseModel): #
    audio_url: str
    audio_duration_seconds: Optional[float] # This remains an estimation
    voice_used_details: Dict[str, Any]
    text_character_count: int # For SSML, this is the length of the SSML string
    processing_time_ms: Optional[float]

# --- InnovateAI Refined Uplas Voice Character Mapping ---
# Emphasizing Standard voices for cost-effectiveness (first ~4M chars/month free).
# WaveNet/Studio voices (first ~1M chars/month free) for premium quality when `prefer_premium_quality` is true.
# 'quality_tier': 'standard', 'wavenet', 'studio'
UPLAS_VOICE_CHARACTER_MAP: Dict[str, Dict[str, Any]] = { # Based on structure, refined
    # English (US)
    "susan_us_standard": {"google_voice_name": "en-US-Standard-O", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["professional", "clear_standard"]},
    "susan_us_wavenet":  {"google_voice_name": "en-US-Wavenet-O",  "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["professional", "natural_wavenet"]},
    "susan_us_studio":   {"google_voice_name": "en-US-Studio-O",   "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "studio", "style_tags": ["professional", "premium_studio"]}, # From original map's susan_pro_voice
    "trevor_us_standard":{"google_voice_name": "en-US-Standard-D", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "style_tags": ["wise", "calm_standard"]},
    "trevor_us_wavenet": {"google_voice_name": "en-US-Wavenet-D",  "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "style_tags": ["wise", "avuncular_wavenet"]},
    "trevor_us_studio":  {"google_voice_name": "en-US-Studio-M",   "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "studio", "style_tags": ["wise", "avuncular_studio"]}, # from original map's trevor_wise_voice
    "alloy_us_studio":   {"google_voice_name": "en-US-Studio-M",   "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.NEUTRAL,"quality_tier": "studio", "style_tags": ["clear", "neutral_studio"]}, #

    # French (fr-FR)
    "elodie_fr_standard": {"google_voice_name": "fr-FR-Standard-A", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_fr", "clear"]}, # Neural2 in orig, mapping to standard
    "elodie_fr_wavenet":  {"google_voice_name": "fr-FR-Wavenet-A",  "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_fr"]},
    "antoine_fr_standard":{"google_voice_name": "fr-FR-Standard-D", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "style_tags": ["professional_fr"]}, # Studio in orig, mapping to standard
    "antoine_fr_wavenet": {"google_voice_name": "fr-FR-Wavenet-D",  "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "style_tags": ["professional_fr_wavenet"]},

    # Spanish (es-ES)
    "sofia_es_standard": {"google_voice_name": "es-ES-Standard-A", "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_es", "pleasant"]}, # Neural2 in orig
    "sofia_es_wavenet":  {"google_voice_name": "es-ES-Wavenet-A",  "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_es"]},

    # German (de-DE)
    "lena_de_standard": {"google_voice_name": "de-DE-Standard-F", "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_de", "friendly"]}, # Neural2 in orig
    "lena_de_wavenet":  {"google_voice_name": "de-DE-Wavenet-F",  "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_de"]},

    # Portuguese (pt-BR)
    "isabela_br_standard": {"google_voice_name": "pt-BR-Standard-A", "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_pt_br", "smooth"]}, # Neural2 in orig
    "isabela_br_wavenet":  {"google_voice_name": "pt-BR-Wavenet-A",  "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_pt_br"]},

    # Chinese (Simplified - cmn-CN)
    "meilin_cn_standard": {"google_voice_name": "cmn-CN-Standard-A", "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_zh_cn", "gentle"]}, # Neural2 in orig
    "meilin_cn_wavenet":  {"google_voice_name": "cmn-CN-Wavenet-A",  "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_zh_cn"]},

    # Hindi (hi-IN)
    "priya_in_standard": {"google_voice_name": "hi-IN-Standard-A", "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_hi_in", "clear"]}, # Neural2 in orig
    "priya_in_wavenet":  {"google_voice_name": "hi-IN-Wavenet-A",  "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_hi_in"]},

    # Default Fallback (Standard)
    "default_en_us_standard": {"google_voice_name": "en-US-Standard-C", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["default", "fallback"]},
}

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas Text-to-Speech (TTS) Agent - Enhanced by InnovateAI",
    description="Converts text or SSML to speech using Google Cloud TTS API, with dynamic voice & quality selection, and GCS storage.",
    version="0.3.0" # Incremented version
)

# --- Helper: Get Voice Configuration (InnovateAI Refined) ---
def get_voice_config_from_character(
    uplas_character_name_base: str, # e.g., "susan_us", "elodie_fr"
    requested_language_code: Optional[str] = None, # BCP-47
    prefer_premium: bool = False
) -> Dict[str, Any]:
    """
    InnovateAI Enhanced: Selects Google TTS voice config based on Uplas character base name,
    language, and quality preference. Prioritizes Standard voices for cost-effectiveness unless
    premium is preferred and available.
    """
    char_base_lower = uplas_character_name_base.lower()
    potential_names = []

    # Construct potential voice names based on preference
    if prefer_premium:
        # Try Studio first (if convention exists, e.g. charname_lang_studio)
        # For simplicity, we'll focus on wavenet as the primary premium tier after studio.
        potential_names.append(f"{char_base_lower}_studio") # Highest premium
        potential_names.append(f"{char_base_lower}_wavenet") # Next premium
    potential_names.append(f"{char_base_lower}_standard") # Standard tier
    potential_names.append(char_base_lower) # Direct match for base name (might be a standard or specific voice)

    selected_config = None
    for name_attempt in potential_names:
        config_candidate = UPLAS_VOICE_CHARACTER_MAP.get(name_attempt)
        if config_candidate:
            # If language is requested, ensure the candidate matches or is language-agnostic in selection
            if requested_language_code and config_candidate["language_code"] != requested_language_code:
                # This candidate is for a different language, skip if specific language requested
                # unless we want to allow cross-language character use (generally not for TTS)
                logger.debug(f"Candidate {name_attempt} lang {config_candidate['language_code']} != requested {requested_language_code}")
                continue
            selected_config = config_candidate
            break # Found a suitable candidate based on preference and name

    if selected_config:
        logger.info(f"InnovateAI: Voice config selected for '{uplas_character_name_base}' (premium: {prefer_premium}): '{selected_config['google_voice_name']}' ({selected_config['quality_tier']})")
        return selected_config
    else:
        # Fallback if specific character name (with or without tier) not found
        logger.warning(f"InnovateAI: Character '{uplas_character_name_base}' with preference not directly found. Attempting broader fallback.")
        target_lang = requested_language_code or DEFAULT_LANGUAGE
        # Prioritize standard voices for fallback
        fallback_voices = [
            config for name, config in UPLAS_VOICE_CHARACTER_MAP.items()
            if config["language_code"] == target_lang and config.get("quality_tier") == "standard"
        ]
        if fallback_voices:
            logger.warning(f"InnovateAI: Falling back to first standard voice for language '{target_lang}'.")
            return fallback_voices[0]
        
        logger.error(f"InnovateAI Critical Fallback: No standard voice for language '{target_lang}'. Using global default.")
        return UPLAS_VOICE_CHARACTER_MAP["default_en_us_standard"]


# --- API Endpoint (InnovateAI Enhanced) ---
@app.post("/v1/synthesize-speech", response_model=SynthesizeSpeechResponse, summary="Synthesize speech from text or SSML")
async def synthesize_speech_endpoint(request_data: SynthesizeSpeechRequest): # Based on
    processing_start_time = time.perf_counter()

    if not tts_client or not storage_client:
        logger.error("InnovateAI Critical: TTS Client or Storage Client not initialized. Check GCP configuration.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTS service is not properly configured.")
    if not TTS_AUDIO_GCS_BUCKET_NAME:
        logger.error("InnovateAI Critical: TTS_AUDIO_GCS_BUCKET_NAME is not set. Cannot store audio.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Audio storage bucket not configured.")

    # 1. Determine Google TTS voice configuration using InnovateAI's refined logic
    voice_selection_details = get_voice_config_from_character(
        request_data.voice_params.voice_character_name, # This should be the base name like "susan_us"
        request_data.language_code,
        request_data.prefer_premium_quality
    )
    synthesis_language_code = voice_selection_details["language_code"]

    # 2. Prepare Synthesis Input (Handles TEXT or SSML - InnovateAI Enhancement)
    if request_data.input_type == SynthesisInputType.SSML:
        synthesis_input = texttospeech.SynthesisInput(ssml=request_data.content_to_synthesize)
        logger.info("InnovateAI: Synthesizing speech from SSML input.")
    else: # Default to TEXT
        synthesis_input = texttospeech.SynthesisInput(text=request_data.content_to_synthesize)
        logger.info("InnovateAI: Synthesizing speech from TEXT input.")

    # 3. Prepare Voice Selection Parameters
    voice_params = texttospeech.VoiceSelectionParams( #
        language_code=synthesis_language_code,
        name=voice_selection_details["google_voice_name"],
        ssml_gender=voice_selection_details.get("ssml_gender", texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED)
    )

    # 4. Prepare Audio Configuration
    google_audio_encoding_map = { #
        AudioEncodingEnum.MP3: texttospeech.AudioEncoding.MP3,
        AudioEncodingEnum.LINEAR16: texttospeech.AudioEncoding.LINEAR16,
        AudioEncodingEnum.OGG_OPUS: texttospeech.AudioEncoding.OGG_OPUS,
    }
    audio_config_params = texttospeech.AudioConfig( #
        audio_encoding=google_audio_encoding_map[request_data.audio_config.audio_encoding],
        speaking_rate=request_data.voice_params.speaking_rate,
        pitch=request_data.voice_params.pitch,
        sample_rate_hertz=request_data.audio_config.sample_rate_hertz,
        effects_profile_id=request_data.voice_params.effects_profile_id or []
    )

    # 5. Call Google Cloud TTS API
    try: #
        logger.info(f"InnovateAI: Requesting synthesis with voice: {voice_params.name} (Quality: {voice_selection_details['quality_tier']}), lang: {voice_params.language_code}")
        # **** Placeholder for actual call to tts_client.synthesize_speech - Mugambi to implement ****
        # response = await tts_client.synthesize_speech(
        #     input=synthesis_input,
        #     voice=voice_params,
        #     audio_config=audio_config_params
        # )
        # audio_content_bytes = response.audio_content
        # Mocked response for InnovateAI framework:
        mock_audio_header = f"MockAudio_{voice_selection_details['quality_tier']}_{request_data.audio_config.audio_encoding.value}_"
        audio_content_bytes = mock_audio_header.encode('utf-8') + os.urandom(1024 * random.randint(20,80)) # Random audio data
        logger.info(f"InnovateAI Placeholder: Mock audio content generated ({len(audio_content_bytes)} bytes).")
        # **** END Placeholder ****

        if not audio_content_bytes: #
            logger.error("InnovateAI Error: Speech synthesis (mock or real) produced no audio content.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Speech synthesis produced no audio content.")
    except Exception as e: #
        logger.error(f"InnovateAI Error: Google Cloud TTS API call failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Speech synthesis service failed: {str(e)}")

    # 6. Store audio to GCS
    file_extension_map = { #
        AudioEncodingEnum.MP3: "mp3", AudioEncodingEnum.LINEAR16: "wav", AudioEncodingEnum.OGG_OPUS: "ogg"
    }
    file_extension = file_extension_map[request_data.audio_config.audio_encoding]
    blob_name = f"tts_audio/{uuid.uuid4()}.{file_extension}" #
    content_type_map = { #
        AudioEncodingEnum.MP3: "audio/mpeg", AudioEncodingEnum.LINEAR16: "audio/wav", AudioEncodingEnum.OGG_OPUS: "audio/ogg"
    }
    mime_type = content_type_map[request_data.audio_config.audio_encoding]
    public_audio_url_gs_path = f"gs://{TTS_AUDIO_GCS_BUCKET_NAME}/{blob_name}" # GCS path for reference

    try: #
        bucket = storage_client.bucket(TTS_AUDIO_GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        audio_file_like = BytesIO(audio_content_bytes) # Use BytesIO as in your original code
        
        # **** Placeholder for actual GCS upload - Mugambi to implement ****
        # For async upload, you might need to use asyncio.to_thread for the blocking GCS call
        # or use a library that supports async GCS uploads if available.
        # Example blocking call:
        # blob.upload_from_file(audio_file_like, content_type=mime_type)
        # public_audio_url = blob.public_url # Ensure public access or use signed URLs
        # Mocked for InnovateAI framework:
        public_audio_url = f"https://storage.googleapis.com/{TTS_AUDIO_GCS_BUCKET_NAME}/{blob_name}" # Construct assumed public URL
        logger.info(f"InnovateAI Placeholder: Mock audio upload to GCS. Assumed URL: {public_audio_url}")
        # **** END Placeholder ****

        logger.info(f"InnovateAI: Audio successfully prepared for GCS (mocked upload): {public_audio_url_gs_path}")
    except Exception as e: #
        logger.error(f"InnovateAI Error: GCS upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to store synthesized audio: {str(e)}")

    # Estimate duration (remains a rough estimate)
    estimated_duration_seconds = len(audio_content_bytes) / (24000 * 1.5) # Adjusted rough estimator
    if request_data.audio_config.audio_encoding == AudioEncodingEnum.MP3:
         estimated_duration_seconds = len(audio_content_bytes) / 16000 # Approx for 128kbps MP3
    
    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    return SynthesizeSpeechResponse( #
        audio_url=public_audio_url, # This would be the actual GCS public URL or signed URL
        audio_duration_seconds=round(estimated_duration_seconds, 2),
        voice_used_details={
            "uplas_voice_character_name": request_data.voice_params.voice_character_name, # Base name requested
            "google_voice_name": voice_params.name,
            "language_code": voice_params.language_code,
            "quality_tier_used": voice_selection_details.get("quality_tier", "unknown"), # InnovateAI addition
            "style_tags": voice_selection_details.get("style_tags", [])
        },
        text_character_count=len(request_data.content_to_synthesize),
        processing_time_ms=round(processing_time_ms, 2)
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK) #
async def health_check():
    if not tts_client or not storage_client:
        return {"status": "unhealthy", "reason": "GCP clients not initialized", "service": "TTS_Agent"}
    if not GCP_PROJECT_ID or not TTS_AUDIO_GCS_BUCKET_NAME:
        return {"status": "unhealthy", "reason": "Missing GCP_PROJECT_ID or TTS_AUDIO_GCS_BUCKET_NAME", "service": "TTS_Agent"}
    return {"status": "healthy", "service": "TTS_Agent", "innovate_ai_enhancements_active": True}


if __name__ == "__main__": #
    import uvicorn
    logger.info("InnovateAI: Starting TTS Agent for local development...")
    if not GCP_PROJECT_ID: print("InnovateAI Warning: GCP_PROJECT_ID is not set.")
    if not TTS_AUDIO_GCS_BUCKET_NAME: print("InnovateAI Warning: TTS_AUDIO_GCS_BUCKET_NAME is not set.")
    
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
