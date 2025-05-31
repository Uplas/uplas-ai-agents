# uplas-ai-agents/tts_agent/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union # Added Union
import os
import uuid
import time
from enum import Enum
import logging
from io import BytesIO # For handling audio content

# GCP Clients
from google.cloud import texttospeech # v1
from google.cloud import storage
import google.auth

# --- Configuration (as per existing) ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TTS_AUDIO_GCS_BUCKET_NAME = os.getenv("TTS_AUDIO_GCS_BUCKET_NAME")

# Initialize GCP clients (as per existing)
# ... (existing client initialization logic) ...
try:
    tts_client = texttospeech.TextToSpeechAsyncClient()
    storage_client = storage.Client(project=GCP_PROJECT_ID if GCP_PROJECT_ID else None)
except Exception as e:
    logging.error(f"Error initializing GCP clients: {e}")
    tts_client = None
    storage_client = None

SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] #
DEFAULT_LANGUAGE = "en-US" #

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Contract (ENHANCED) ---

class VoiceSelectionParams(BaseModel): # As per existing
    voice_character_name: str = Field(..., examples=["alloy_us_standard", "susan_pro_voice_wavenet"]) # Indicate quality tier in name
    speaking_rate: Optional[float] = Field(1.0, ge=0.25, le=4.0)
    pitch: Optional[float] = Field(0.0, ge=-20.0, le=20.0)
    effects_profile_id: Optional[List[str]] = Field(default_factory=list)

class AudioEncodingEnum(str, Enum): # As per existing
    MP3 = "MP3"
    LINEAR16 = "LINEAR16"
    OGG_OPUS = "OGG_OPUS"

class AudioConfig(BaseModel): # As per existing
    audio_encoding: AudioEncodingEnum = Field(AudioEncodingEnum.MP3)
    sample_rate_hertz: Optional[int] = None

# ENHANCED: SynthesizeSpeechRequest to support SSML
class SynthesisInputType(str, Enum):
    TEXT = "text"
    SSML = "ssml"

class SynthesizeSpeechRequest(BaseModel): # Modified from
    input_type: SynthesisInputType = Field(SynthesisInputType.TEXT, description="Specify if the input content is plain text or SSML.")
    # Content can be either plain text or SSML string
    content_to_synthesize: str = Field(..., min_length=1, max_length=5000, examples=["Hello, welcome to Uplas!", "<speak>Hello <emphasis>Uplas</emphasis> world!</speak>"])
    language_code: Optional[str] = Field(None, examples=SUPPORTED_LANGUAGES)
    voice_params: VoiceSelectionParams
    audio_config: Optional[AudioConfig] = Field(default_factory=AudioConfig)
    # Optional: flag to explicitly request higher quality voice if available and within limits
    prefer_wavenet_quality: Optional[bool] = Field(False, description="Attempt to use WaveNet voice if available for the character.")

    # Validator for language_code (as per existing)
    @validator('language_code', always=True)
    def validate_language_code(cls, v, values): #
        # ... (existing validation logic) ...
        if v is None: return v
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' received. Will attempt to derive from voice character or use default.")
            return None
        return v

class SynthesizeSpeechResponse(BaseModel): # As per existing
    audio_url: str
    audio_duration_seconds: Optional[float]
    voice_used_details: Dict[str, Any]
    text_character_count: int # For SSML, this would be char count of the SSML string
    processing_time_ms: Optional[float]

# --- Uplas Voice Character Mapping (REFINED for clarity and cost-effectiveness) ---
# We'll emphasize standard voices. WaveNet options can be suffixed with '_wavenet'.
# The 'google_voice_name' should differentiate between standard and WaveNet/Studio.
# (Existing map is good, this is more about usage strategy)
UPLAS_VOICE_CHARACTER_MAP: Dict[str, Dict[str, Any]] = {
    # English (US) - Standard tier for high volume, WaveNet for premium
    "susan_us_standard": {"google_voice_name": "en-US-Standard-O", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["professional", "clear"]},
    "susan_us_wavenet":  {"google_voice_name": "en-US-Wavenet-O",  "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["professional", "natural"]},
    "trevor_us_standard":{"google_voice_name": "en-US-Standard-D", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "style_tags": ["wise", "calm"]},
    "trevor_us_wavenet": {"google_voice_name": "en-US-Wavenet-D",  "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "wavenet", "style_tags": ["wise", "avuncular_natural"]},
    "alloy_us_studio":   {"google_voice_name": "en-US-Studio-M",   "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.NEUTRAL,"quality_tier": "studio", "style_tags": ["clear", "premium_neutral"]}, # Studio voice from existing map

    # French (fr-FR) - Standard Focus
    "elodie_fr_standard": {"google_voice_name": "fr-FR-Standard-A", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_fr", "clear"]},
    "antoine_fr_standard":{"google_voice_name": "fr-FR-Standard-D", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "quality_tier": "standard", "style_tags": ["professional_fr"]},
    "elodie_fr_wavenet":  {"google_voice_name": "fr-FR-Wavenet-A",  "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_fr"]},


    # Spanish (es-ES) - Standard Focus
    "sofia_es_standard": {"google_voice_name": "es-ES-Standard-A", "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_es", "pleasant"]},
    "sofia_es_wavenet":  {"google_voice_name": "es-ES-Wavenet-A",  "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_es"]},


    # German (de-DE) - Standard Focus
    "lena_de_standard": {"google_voice_name": "de-DE-Standard-F", "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_de", "friendly"]},
    "lena_de_wavenet":  {"google_voice_name": "de-DE-Wavenet-F",  "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_de"]},


    # Portuguese (pt-BR) - Standard Focus
    "isabela_br_standard": {"google_voice_name": "pt-BR-Standard-A", "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_pt_br", "smooth"]},
    "isabela_br_wavenet":  {"google_voice_name": "pt-BR-Wavenet-A",  "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_pt_br"]},


    # Chinese (Simplified - cmn-CN) - Standard Focus
    "meilin_cn_standard": {"google_voice_name": "cmn-CN-Standard-A", "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_zh_cn", "gentle"]},
    "meilin_cn_wavenet":  {"google_voice_name": "cmn-CN-Wavenet-A",  "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_zh_cn"]},


    # Hindi (hi-IN) - Standard Focus
    "priya_in_standard": {"google_voice_name": "hi-IN-Standard-A", "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["standard_hi_in", "clear"]},
    "priya_in_wavenet":  {"google_voice_name": "hi-IN-Wavenet-A",  "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "wavenet", "style_tags": ["natural_hi_in"]},


    # Default Fallback (Standard)
    "default_en_us_standard": {"google_voice_name": "en-US-Standard-C", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "quality_tier": "standard", "style_tags": ["default", "fallback"]},
}


# --- FastAPI Application (existing structure) ---
app = FastAPI(
    title="Uplas Text-to-Speech (TTS) Agent - Enhanced", #
    description="Converts text or SSML to speech using Google Cloud TTS API, stores in GCS, with dynamic voice selection.",
    version="0.3.0" # Incremented version
)

# --- Helper: Get Voice Configuration (REFINED) ---
def get_voice_config_from_character(
    uplas_character_name: str,
    requested_language_code: Optional[str] = None,
    prefer_wavenet: bool = False # New parameter
) -> Dict[str, Any]: # Modified from
    char_name_lower = uplas_character_name.lower()
    
    # Attempt to find direct match or quality-tiered match
    # e.g. if "susan_us" is requested and prefer_wavenet is true, try "susan_us_wavenet" first.
    # This logic can be made more sophisticated. For now, explicit names are in the map.
    selected_config = UPLAS_VOICE_CHARACTER_MAP.get(char_name_lower)

    # If a specific quality tier (e.g. "_wavenet") was part of the name, it's already selected.
    # If a generic name like "susan_us" was given, and prefer_wavenet is true,
    # we could try to find "susan_us_wavenet".
    # For simplicity in this iteration, assume `voice_character_name` includes the tier or is a standard voice.
    # A more advanced version could:
    # if prefer_wavenet and selected_config and selected_config.get("quality_tier") == "standard":
    #    wavenet_char_name = f"{char_name_lower}_wavenet" # or a mapping
    #    wavenet_config = UPLAS_VOICE_CHARACTER_MAP.get(wavenet_char_name)
    #    if wavenet_config: selected_config = wavenet_config

    if selected_config:
        if requested_language_code and requested_language_code != selected_config["language_code"]:
            logger.warning(
                f"Requested language '{requested_language_code}' differs from character '{uplas_character_name}'"
                f" default language '{selected_config['language_code']}'. Using character's default."
            )
        return selected_config
    else:
        # Fallback if character name not found
        logger.warning(f"Uplas character '{uplas_character_name}' not found. Attempting fallback based on language.")
        target_lang = requested_language_code or DEFAULT_LANGUAGE
        # Prioritize standard voices for fallback
        fallback_voices = [
            config for config in UPLAS_VOICE_CHARACTER_MAP.values()
            if config["language_code"] == target_lang and config.get("quality_tier") == "standard"
        ]
        if fallback_voices:
            return fallback_voices[0] # Pick the first standard voice for that language
        
        logger.warning(f"No standard fallback for language '{target_lang}'. Using global default.")
        return UPLAS_VOICE_CHARACTER_MAP["default_en_us_standard"]


# --- API Endpoint (MODIFIED for SSML and refined voice selection) ---
@app.post("/v1/synthesize-speech", response_model=SynthesizeSpeechResponse, summary="Synthesize speech from text or SSML")
async def synthesize_speech_endpoint(request_data: SynthesizeSpeechRequest): # Modified from
    processing_start_time = time.perf_counter()

    # Validations (as per existing)
    if not tts_client or not storage_client: #
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="TTS service is not properly configured.")
    if not TTS_AUDIO_GCS_BUCKET_NAME: #
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Audio storage bucket not configured.")

    # 1. Determine Google TTS voice configuration (using refined helper)
    voice_selection_details = get_voice_config_from_character(
        request_data.voice_params.voice_character_name,
        request_data.language_code,
        request_data.prefer_wavenet_quality # Pass preference
    )
    synthesis_language_code = voice_selection_details["language_code"]

    # 2. Prepare Synthesis Input (Handles TEXT or SSML)
    if request_data.input_type == SynthesisInputType.SSML:
        synthesis_input = texttospeech.SynthesisInput(ssml=request_data.content_to_synthesize)
    else: # Default to TEXT
        synthesis_input = texttospeech.SynthesisInput(text=request_data.content_to_synthesize)

    # 3. Prepare Voice Selection Parameters (as per existing)
    voice_params = texttospeech.VoiceSelectionParams( #
        language_code=synthesis_language_code,
        name=voice_selection_details["google_voice_name"],
        ssml_gender=voice_selection_details.get("ssml_gender", texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED)
    )

    # 4. Prepare Audio Configuration (as per existing)
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

    # 5. Call Google Cloud TTS API (as per existing)
    try: #
        logger.info(f"Synthesizing speech ({request_data.input_type.value}) with voice: {voice_params.name}, lang: {voice_params.language_code}")
        # **** Actual call to tts_client.synthesize_speech to be implemented by Mugambi HERE ****
        # response = await tts_client.synthesize_speech(
        #     input=synthesis_input,
        #     voice=voice_params,
        #     audio_config=audio_config_params
        # )
        # audio_content_bytes = response.audio_content
        # **** END Call ****
        
        # Mocked response for now
        audio_content_bytes = b"mock_audio_bytes_" + request_data.audio_config.audio_encoding.value.encode()
        if not audio_content_bytes: #
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Speech synthesis produced no audio content.")

    except Exception as e: #
        logger.error(f"Google Cloud TTS API error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Speech synthesis service failed: {str(e)}")

    # 6. Store audio to GCS (as per existing, slight modification for BytesIO)
    # ... (existing GCS upload logic using BytesIO, file_extension_map, mime_type_map) ...
    file_extension_map = {AudioEncodingEnum.MP3: "mp3", AudioEncodingEnum.LINEAR16: "wav", AudioEncodingEnum.OGG_OPUS: "ogg"}
    file_extension = file_extension_map[request_data.audio_config.audio_encoding]
    blob_name = f"tts_audio/{uuid.uuid4()}.{file_extension}"
    content_type_map = {AudioEncodingEnum.MP3: "audio/mpeg", AudioEncodingEnum.LINEAR16: "audio/wav", AudioEncodingEnum.OGG_OPUS: "audio/ogg"}
    mime_type = content_type_map[request_data.audio_config.audio_encoding]
    public_audio_url = f"gs://{TTS_AUDIO_GCS_BUCKET_NAME}/{blob_name}" # Placeholder URL

    try:
        bucket = storage_client.bucket(TTS_AUDIO_GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        audio_file_like = BytesIO(audio_content_bytes)
        # **** Actual call to blob.upload_from_file to be implemented by Mugambi HERE ****
        # blob.upload_from_file(audio_file_like, content_type=mime_type)
        # public_audio_url = blob.public_url # Use this if objects are public
        # **** END Call ****
        logger.info(f"Mock audio upload to GCS: {public_audio_url}")
    except Exception as e:
        logger.error(f"GCS upload error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to store synthesized audio: {str(e)}")


    # Estimate duration (existing logic)
    estimated_duration_seconds = len(audio_content_bytes) / (16000 * 0.5) # Rough estimate adjustment
    
    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    return SynthesizeSpeechResponse(
        audio_url=public_audio_url,
        audio_duration_seconds=round(estimated_duration_seconds, 2),
        voice_used_details={
            "uplas_voice_character_name": request_data.voice_params.voice_character_name,
            "google_voice_name": voice_params.name,
            "language_code": voice_params.language_code,
            "quality_tier_used": voice_selection_details.get("quality_tier", "unknown"),
            "style_tags": voice_selection_details.get("style_tags", [])
        },
        text_character_count=len(request_data.content_to_synthesize),
        processing_time_ms=round(processing_time_ms, 2)
    )

# --- Health Check Endpoint (as per existing) ---
@app.get("/health", status_code=status.HTTP_200_OK) #
async def health_check():
    # ... (existing health check logic) ...
    if not tts_client or not storage_client:
        return {"status": "unhealthy", "reason": "GCP clients not initialized", "service": "TTS_Agent"}
    if not GCP_PROJECT_ID or not TTS_AUDIO_GCS_BUCKET_NAME:
        return {"status": "unhealthy", "reason": "Missing GCP_PROJECT_ID or TTS_AUDIO_GCS_BUCKET_NAME", "service": "TTS_Agent"}
    return {"status": "healthy", "service": "TTS_Agent"}


if __name__ == "__main__": #
    import uvicorn
    # ... (existing main block) ...
    if not GCP_PROJECT_ID: print("Warning: GCP_PROJECT_ID is not set.")
    if not TTS_AUDIO_GCS_BUCKET_NAME: print("Warning: TTS_AUDIO_GCS_BUCKET_NAME is not set.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8002)))
