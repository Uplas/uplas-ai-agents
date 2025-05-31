# uplas-ai-agents/tts_agent/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import os
import uuid
import time
from enum import Enum
import logging

# GCP Clients
from google.cloud import texttospeech # v1
from google.cloud import storage
import google.auth

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
# Ensure this bucket exists and the service account has write permissions
TTS_AUDIO_GCS_BUCKET_NAME = os.getenv("TTS_AUDIO_GCS_BUCKET_NAME")

# Initialize GCP clients
if not GCP_PROJECT_ID:
    logging.warning("GCP_PROJECT_ID environment variable not set. TTS Agent may not function correctly.")
if not TTS_AUDIO_GCS_BUCKET_NAME:
    logging.warning("TTS_AUDIO_GCS_BUCKET_NAME environment variable not set. Audio storage will fail.")

# Attempt to initialize clients - they will use Application Default Credentials
try:
    tts_client = texttospeech.TextToSpeechAsyncClient() # Use async client
    storage_client = storage.Client(project=GCP_PROJECT_ID if GCP_PROJECT_ID else None)
except Exception as e:
    logging.error(f"Error initializing GCP clients: {e}")
    # Depending on policy, might raise an error or allow app to start with non-functional clients
    tts_client = None
    storage_client = None


# Supported languages (BCP-47 codes) - align with AI Tutor
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Contract ---

class VoiceSelectionParams(BaseModel):
    voice_character_name: str = Field(..., examples=["alloy_us", "susan_pro_voice", "elodie_fr_standard"])
    speaking_rate: Optional[float] = Field(1.0, ge=0.25, le=4.0, description="Speaking_rate/speed, 1.0 is normal.")
    pitch: Optional[float] = Field(0.0, ge=-20.0, le=20.0, description="Speaking pitch, 0.0 is normal.")
    effects_profile_id: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of strings representing the audio profile."
                    "For example, ['wearable-class-device'] or ['handset-class-device']"
    )

class AudioEncodingEnum(str, Enum):
    MP3 = "MP3"  # Good for web, widely compatible
    LINEAR16 = "LINEAR16"  # WAV, uncompressed, higher quality
    OGG_OPUS = "OGG_OPUS" # Good for streaming, efficient

class AudioConfig(BaseModel):
    audio_encoding: AudioEncodingEnum = Field(AudioEncodingEnum.MP3)
    sample_rate_hertz: Optional[int] = Field(
        None,
        description="Optional. The synthesis sample rate (in hertz) for this audio."
                    "If not provided, then the service will choose a default version based on the voice."
    )
    # volume_gain_db: Optional[float] = Field(None, ge=-96.0, le=16.0) # If needed

class SynthesizeSpeechRequest(BaseModel):
    text_to_speak: str = Field(..., min_length=1, max_length=5000, examples=["Hello, welcome to Uplas!"])
    language_code: Optional[str] = Field(
        None, # If None, will be inferred from voice_character_name
        examples=SUPPORTED_LANGUAGES,
        description="BCP-47 language tag. If provided, it can refine voice selection or override character's default."
    )
    voice_params: VoiceSelectionParams
    audio_config: Optional[AudioConfig] = Field(default_factory=AudioConfig)
    context_course_id: Optional[str] = None # For logging/analytics
    context_topic_id: Optional[str] = None  # For logging/analytics

    @validator('language_code', always=True) # always=True to run even if None
    def validate_language_code(cls, v, values):
        if v is None: # If not provided, it will be derived from voice_character_name later
            return v
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' received. Will attempt to derive from voice character or use default.")
            # Don't set to default here, let voice selection logic handle it.
            return None # Mark as invalid for now, or let voice selection try to handle
        return v


class SynthesizeSpeechResponse(BaseModel):
    audio_url: str = Field(..., examples=[f"https://storage.googleapis.com/{TTS_AUDIO_GCS_BUCKET_NAME or 'your-bucket'}/tts_audio/generated.mp3"])
    audio_duration_seconds: Optional[float] = Field(None, examples=[5.72], description="Estimated or actual duration of the audio.")
    voice_used_details: Dict[str, Any] = Field(..., examples=[{"uplas_voice_character_name": "alloy_us", "google_voice_name": "en-US-Studio-M", "language_code": "en-US"}])
    text_character_count: int
    processing_time_ms: Optional[float] = None

# --- Uplas Voice Character Mapping ---
# This map defines user-friendly names to specific Google Cloud TTS voice configurations.
# It should be expanded significantly with diverse voices for all 7 languages.
# 'google_voice_name' refers to the `name` field in Google's VoiceSelectionParams.
# 'language_code' is the BCP-47 code for that voice.
# 'ssml_gender' is an enum: SSML_VOICE_GENDER_UNSPECIFIED, MALE, FEMALE, NEUTRAL.
# 'natural_sample_rate_hertz' can be found from listing voices.
UPLAS_VOICE_CHARACTER_MAP: Dict[str, Dict[str, Any]] = {
    # English Voices (US, GB, AU)
    "alloy_us": {"google_voice_name": "en-US-Studio-M", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.NEUTRAL, "style_tags": ["clear", "professional"]},
    "susan_pro_voice": {"google_voice_name": "en-US-Studio-O", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["professional", "articulate"]},
    "trevor_wise_voice": {"google_voice_name": "en-US-Studio-M", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["wise", "avuncular"]}, # Can be same as alloy if persona fits
    "echo_us_male": {"google_voice_name": "en-US-Neural2-D", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["standard", "narration"]},
    "fable_gb_story": {"google_voice_name": "en-GB-Neural2-C", "language_code": "en-GB", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["storyteller", "warm"]},
    "onyx_au_deep": {"google_voice_name": "en-AU-Studio-B", "language_code": "en-AU", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["deep", "authoritative"]},
    "nova_us_female_bright": {"google_voice_name": "en-US-Neural2-F", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["bright", "energetic"]},

    # French (fr-FR)
    "elodie_fr_standard": {"google_voice_name": "fr-FR-Neural2-A", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["standard_fr", "clear"]},
    "antoine_fr_studio": {"google_voice_name": "fr-FR-Studio-D", "language_code": "fr-FR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["studio_fr", "professional"]},

    # Spanish (es-ES, es-US)
    "sofia_es_standard": {"google_voice_name": "es-ES-Neural2-A", "language_code": "es-ES", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["standard_es", "pleasant"]},
    "mateo_us_standard": {"google_voice_name": "es-US-Neural2-B", "language_code": "es-US", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["standard_us_es", "clear"]},

    # German (de-DE)
    "lena_de_standard": {"google_voice_name": "de-DE-Neural2-F", "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["standard_de", "friendly"]},
    "max_de_studio": {"google_voice_name": "de-DE-Studio-B", "language_code": "de-DE", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["studio_de", "professional"]},

    # Portuguese (pt-BR)
    "isabela_br_standard": {"google_voice_name": "pt-BR-Neural2-A", "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["standard_pt_br", "smooth"]},
    "rafael_br_neural": {"google_voice_name": "pt-BR-Neural2-B", "language_code": "pt-BR", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["neural_pt_br", "clear"]}, # Changed from Wavenet as Neural2 is common

    # Chinese (Simplified - cmn-CN)
    "meilin_cn_standard": {"google_voice_name": "cmn-CN-Neural2-A", "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["standard_zh_cn", "gentle"]},
    "jianyu_cn_neural": {"google_voice_name": "cmn-CN-Neural2-D", "language_code": "cmn-CN", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["neural_zh_cn", "standard"]},

    # Hindi (hi-IN)
    "priya_in_standard": {"google_voice_name": "hi-IN-Neural2-A", "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["standard_hi_in", "clear"]},
    "rohan_in_neural": {"google_voice_name": "hi-IN-Neural2-B", "language_code": "hi-IN", "ssml_gender": texttospeech.SsmlVoiceGender.MALE, "style_tags": ["neural_hi_in", "standard"]},

    # Default Fallback
    "default_en_us_female": {"google_voice_name": "en-US-Standard-C", "language_code": "en-US", "ssml_gender": texttospeech.SsmlVoiceGender.FEMALE, "style_tags": ["default", "fallback"]},
}

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas Text-to-Speech (TTS) Agent (Google Cloud Integrated)",
    description="Converts text to speech using Google Cloud Text-to-Speech API and stores it in GCS.",
    version="0.2.0"
)

# --- Helper: Get Voice Configuration ---
def get_voice_config_from_character(
    uplas_character_name: str,
    requested_language_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Selects the appropriate Google TTS voice configuration based on Uplas character name
    and an optional requested language code.
    """
    char_name_lower = uplas_character_name.lower()
    
    # Attempt to find a direct match for character name
    selected_config = UPLAS_VOICE_CHARACTER_MAP.get(char_name_lower)

    if selected_config:
        # If a language code is requested and it differs from the character's default,
        # we might need more sophisticated logic here, e.g., finding a variant of the character
        # for that language, or warning the user. For now, character's language_code takes precedence.
        if requested_language_code and requested_language_code != selected_config["language_code"]:
            logger.warning(
                f"Requested language '{requested_language_code}' differs from character '{uplas_character_name}'"
                f" default language '{selected_config['language_code']}'. Using character's default."
            )
        return selected_config
    else:
        # Fallback if character name not found: try to find a voice for the requested_language_code
        if requested_language_code:
            for config in UPLAS_VOICE_CHARACTER_MAP.values():
                if config["language_code"] == requested_language_code:
                    logger.warning(f"Character '{uplas_character_name}' not found. Falling back to a default voice for language '{requested_language_code}'.")
                    return config # Return the first match for the language
        
        # Ultimate fallback to the global default voice
        logger.warning(f"Character '{uplas_character_name}' not found, and no suitable voice for language '{requested_language_code}'. Using global default voice.")
        return UPLAS_VOICE_CHARACTER_MAP["default_en_us_female"]


# --- API Endpoint ---
@app.post("/v1/synthesize-speech", response_model=SynthesizeSpeechResponse, summary="Synthesize speech from text")
async def synthesize_speech_endpoint(request_data: SynthesizeSpeechRequest):
    processing_start_time = time.perf_counter()

    if not tts_client or not storage_client:
        logger.error("TTS Client or Storage Client not initialized. Check GCP configuration and credentials.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service is not properly configured or initialized."
        )
    if not TTS_AUDIO_GCS_BUCKET_NAME:
        logger.error("TTS_AUDIO_GCS_BUCKET_NAME is not set. Cannot store audio.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audio storage bucket is not configured."
        )

    # 1. Determine Google TTS voice configuration
    voice_selection_details = get_voice_config_from_character(
        request_data.voice_params.voice_character_name,
        request_data.language_code # Pass requested language to potentially influence choice
    )
    
    # The actual language code used for synthesis will be from the selected voice_selection_details
    synthesis_language_code = voice_selection_details["language_code"]

    # 2. Prepare Synthesis Input
    synthesis_input = texttospeech.SynthesisInput(text=request_data.text_to_speak)

    # 3. Prepare Voice Selection Parameters
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=synthesis_language_code,
        name=voice_selection_details["google_voice_name"],
        ssml_gender=voice_selection_details.get("ssml_gender", texttospeech.SsmlVoiceGender.SSML_VOICE_GENDER_UNSPECIFIED)
    )

    # 4. Prepare Audio Configuration
    google_audio_encoding_map = {
        AudioEncodingEnum.MP3: texttospeech.AudioEncoding.MP3,
        AudioEncodingEnum.LINEAR16: texttospeech.AudioEncoding.LINEAR16,
        AudioEncodingEnum.OGG_OPUS: texttospeech.AudioEncoding.OGG_OPUS,
    }
    audio_config_params = texttospeech.AudioConfig(
        audio_encoding=google_audio_encoding_map[request_data.audio_config.audio_encoding],
        speaking_rate=request_data.voice_params.speaking_rate,
        pitch=request_data.voice_params.pitch,
        sample_rate_hertz=request_data.audio_config.sample_rate_hertz, # Can be None
        effects_profile_id=request_data.voice_params.effects_profile_id or [] # Ensure it's a list
    )

    # 5. Call Google Cloud TTS API
    try:
        logger.info(f"Synthesizing speech with voice: {voice_params.name}, lang: {voice_params.language_code}")
        response = await tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config_params
        )
        audio_content_bytes = response.audio_content
    except Exception as e:
        logger.error(f"Google Cloud TTS API error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Speech synthesis service failed: {str(e)}")

    if not audio_content_bytes:
        logger.error("Speech synthesis produced no audio content.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Speech synthesis produced no audio content.")

    # 6. Store audio to GCS
    file_extension_map = {
        AudioEncodingEnum.MP3: "mp3",
        AudioEncodingEnum.LINEAR16: "wav", # LINEAR16 is typically WAV
        AudioEncodingEnum.OGG_OPUS: "ogg"
    }
    file_extension = file_extension_map[request_data.audio_config.audio_encoding]
    
    blob_name = f"tts_audio/{uuid.uuid4()}.{file_extension}"
    
    content_type_map = {
        AudioEncodingEnum.MP3: "audio/mpeg",
        AudioEncodingEnum.LINEAR16: "audio/wav",
        AudioEncodingEnum.OGG_OPUS: "audio/ogg"
    }
    mime_type = content_type_map[request_data.audio_config.audio_encoding]

    try:
        bucket = storage_client.bucket(TTS_AUDIO_GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        # For async upload with GCS, it's typically not directly in the client library's core methods.
        # You might run this in a thread pool executor if it becomes a bottleneck.
        # For simplicity here, using the synchronous upload method.
        # If truly async upload is needed, explore `aiohttp` with GCS signed URLs or `asyncio.to_thread`.
        # For now, we'll assume this is acceptable for the expected load.
        
        # Using a BytesIO buffer for upload_from_file might be slightly cleaner for in-memory bytes
        from io import BytesIO
        audio_file_like = BytesIO(audio_content_bytes)
        
        # Retry mechanism could be added here for robustness
        blob.upload_from_file(audio_file_like, content_type=mime_type)
        
        public_audio_url = blob.public_url # Ensure bucket/object has public access if using this directly
        # Alternatively, generate a signed URL if files are private:
        # public_audio_url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=15), method="GET")

        logger.info(f"Audio successfully uploaded to GCS: {public_audio_url}")

    except Exception as e:
        logger.error(f"GCS upload error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to store synthesized audio: {str(e)}")

    # Estimate duration (very rough, real TTS API might provide this or use a library)
    # This is a placeholder; a more accurate duration requires audio processing.
    # Google TTS API does not directly return duration.
    estimated_duration_seconds = len(audio_content_bytes) / (24000 * 2 * 0.1) # Highly approximate
    if request_data.audio_config.audio_encoding == AudioEncodingEnum.MP3:
         estimated_duration_seconds = len(audio_content_bytes) / 16000 # Approx 128kbps MP3

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    return SynthesizeSpeechResponse(
        audio_url=public_audio_url,
        audio_duration_seconds=round(estimated_duration_seconds, 2),
        voice_used_details={
            "uplas_voice_character_name": request_data.voice_params.voice_character_name,
            "google_voice_name": voice_params.name,
            "language_code": voice_params.language_code,
            "style_tags": voice_selection_details.get("style_tags", [])
        },
        text_character_count=len(request_data.text_to_speak),
        processing_time_ms=round(processing_time_ms, 2)
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    # Basic check for client initialization
    if not tts_client or not storage_client:
        return {"status": "unhealthy", "reason": "GCP clients not initialized", "service": "TTS_Agent"}
    if not GCP_PROJECT_ID or not TTS_AUDIO_GCS_BUCKET_NAME:
        return {"status": "unhealthy", "reason": "Missing GCP_PROJECT_ID or TTS_AUDIO_GCS_BUCKET_NAME", "service": "TTS_Agent"}
    return {"status": "healthy", "service": "TTS_Agent"}


if __name__ == "__main__":
    import uvicorn
    # Ensure environment variables are set for local testing
    # Example: GCP_PROJECT_ID, TTS_AUDIO_GCS_BUCKET_NAME
    # And ensure `gcloud auth application-default login` has been run
    if not GCP_PROJECT_ID:
        print("Warning: GCP_PROJECT_ID is not set. Please set this environment variable.")
    if not TTS_AUDIO_GCS_BUCKET_NAME:
        print("Warning: TTS_AUDIO_GCS_BUCKET_NAME is not set. Please set this environment variable.")

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8002)))

# To run this locally (ensure GCP auth is set up, e.g., `gcloud auth application-default login`):
# GCP_PROJECT_ID="your-gcp-project-id" TTS_AUDIO_GCS_BUCKET_NAME="your-gcs-bucket-name" python main.py
# or using uvicorn directly:
# GCP_PROJECT_ID="your-gcp-project-id" TTS_AUDIO_GCS_BUCKET_NAME="your-gcs-bucket-name" uvicorn main:app --reload --port 8002
