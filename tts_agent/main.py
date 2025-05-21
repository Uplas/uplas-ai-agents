from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import uuid
import time # For processing time
from enum import Enum
# from unittest.mock import MagicMock # We'll create a more direct mock

# --- Pydantic Models for API Contract ---

class VoiceSelectionParams(BaseModel):
    # User-friendly character name from frontend (e.g., "alloy", "echo")
    # This will be mapped to specific Google TTS voice configurations.
    voice_character_name: str = Field(..., examples=["alloy", "shimmer"])
    speaking_rate: Optional[float] = Field(1.0, ge=0.25, le=4.0, examples=[0.9, 1.1])
    pitch: Optional[float] = Field(0.0, ge=-20.0, le=20.0, examples=[-2.0, 1.5])
    # Optional: For effects like "headphone-class-device", "large-home-entertainment-class-device"
    effects_profile_id: Optional[List[str]] = Field(default_factory=list)


class AudioEncodingEnum(str, Enum):
    MP3 = "MP3"
    LINEAR16 = "LINEAR16" # WAV
    OGG_OPUS = "OGG_OPUS"


class AudioConfig(BaseModel):
    audio_encoding: AudioEncodingEnum = Field(AudioEncodingEnum.MP3)
    # sample_rate_hertz is often determined by the voice, but can be specified.
    sample_rate_hertz: Optional[int] = Field(None, examples=[24000, 48000])


class SynthesizeSpeechRequest(BaseModel):
    text_to_speak: str = Field(..., min_length=1, max_length=5000, examples=["Hello, welcome to Uplas!"]) # Max length for single request
    language_code: str = Field("en-US", examples=["en-US", "es-ES", "fr-FR"]) # BCP-47 language tag
    voice_params: VoiceSelectionParams
    audio_config: Optional[AudioConfig] = Field(default_factory=AudioConfig)
    # Optional context for logging/analytics, not directly used by TTS API
    context_course_id: Optional[str] = None
    context_topic_id: Optional[str] = None


class SynthesizeSpeechResponse(BaseModel):
    audio_url: str = Field(..., examples=["https://storage.googleapis.com/uplas-tts-audio/generated_audio.mp3"])
    audio_duration_seconds: Optional[float] = Field(None, examples=[5.72]) # Can be estimated or from file
    voice_used_details: Dict[str, str] = Field(..., examples=[{"google_voice_name": "en-US-Studio-M", "language_code": "en-US"}])
    text_character_count: int
    processing_time_ms: Optional[float] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas Text-to-Speech (TTS) Agent",
    description="Converts text to speech using a (mocked) cloud TTS service and stores it.",
    version="0.1.0"
)

# --- Mock Google Cloud TTS and Storage Clients ---

# This dictionary maps our Uplas voice character names to (mocked) Google TTS configurations.
# In a real scenario, these would be actual Google Cloud TTS voice names and language codes.
UPLAS_VOICE_CHARACTER_MAP: Dict[str, Dict[str, str]] = {
    "alloy": {"google_voice_name": "en-US-MockNeural2-A", "language_code": "en-US", "gender": "NEUTRAL"},
    "echo": {"google_voice_name": "en-US-MockNeural2-D", "language_code": "en-US", "gender": "MALE"},
    "fable": {"google_voice_name": "en-GB-MockWaveNet-C", "language_code": "en-GB", "gender": "MALE"}, # Storyteller
    "onyx": {"google_voice_name": "en-AU-MockStudio-B", "language_code": "en-AU", "gender": "MALE"}, # Deep
    "nova": {"google_voice_name": "fr-FR-MockWaveNet-E", "language_code": "fr-FR", "gender": "FEMALE"}, # Bright
    "shimmer": {"google_voice_name": "es-ES-MockStandard-A", "language_code": "es-ES", "gender": "FEMALE"}, # Smooth
    # Default/Fallback if character name not found
    "default": {"google_voice_name": "en-US-MockStandard-F", "language_code": "en-US", "gender": "FEMALE"},
}

MOCKED_GCS_BUCKET_NAME = os.getenv("MOCKED_TTS_AUDIO_BUCKET_NAME", "mocked-uplas-tts-audio")

class MockTextToSpeechClient:
    async def synthesize_speech(
        self,
        input_text: str,
        voice_config: Dict[str, str], # Contains language_code, name (google_voice_name)
        audio_config_params: AudioConfig,
        speaking_rate: float,
        pitch: float
    ) -> bytes:
        # Simulate audio content generation
        # In a real client, this calls the Google TTS API.
        print(f"MockTTS: Synthesizing '{input_text[:30]}...' with voice {voice_config.get('name')} "
              f"(Lang: {voice_config.get('language_code')}), Rate: {speaking_rate}, Pitch: {pitch}")
        
        # Simulate different audio lengths based on text length
        # This is a very rough simulation.
        mock_audio_header = f"TTS_AUDIO_FOR[{voice_config.get('name', 'default')}]_RATE[{speaking_rate:.1f}]_PITCH[{pitch:.0f}]"
        simulated_audio_data = f"{mock_audio_header}_CONTENT[{input_text}]_ENCODING[{audio_config_params.audio_encoding.value}]"
        
        # Return as bytes, like the real API would for audio_content
        return simulated_audio_data.encode('utf-8')

class MockStorageClient:
    def __init__(self):
        self.bucket_name = MOCKED_GCS_BUCKET_NAME
        self.blobs: Dict[str, bytes] = {} # Simulate GCS bucket storage

    async def upload_blob_from_string(self, blob_name: str, data: bytes, content_type: str) -> str:
        # Simulate uploading to GCS and returning a public URL
        print(f"MockGCS: Uploading '{blob_name}' to bucket '{self.bucket_name}' (Content-Type: {content_type}, Size: {len(data)} bytes)")
        self.blobs[blob_name] = data # Store in our mock "bucket"
        public_url = f"https://storage.googleapis.com/{self.bucket_name}/{blob_name}"
        return public_url

    async def get_blob_data(self, blob_name: str) -> Optional[bytes]: # For testing
        return self.blobs.get(blob_name)

# Initialize mock clients (these would be actual Google clients in production)
mock_tts_client = MockTextToSpeechClient()
mock_storage_client = MockStorageClient()


# --- API Endpoint ---
@app.post("/v1/synthesize-speech", response_model=SynthesizeSpeechResponse, summary="Synthesize speech from text")
async def synthesize_speech_endpoint(request_data: SynthesizeSpeechRequest):
    processing_start_time = time.perf_counter()

    # 1. Determine Google TTS voice configuration from Uplas voice character name
    uplas_char_name = request_data.voice_params.voice_character_name.lower()
    voice_map_entry = UPLAS_VOICE_CHARACTER_MAP.get(uplas_char_name, UPLAS_VOICE_CHARACTER_MAP["default"])
    
    # Override language_code if the mapping suggests a different one for the character,
    # otherwise use the request's language_code.
    # This allows characters to have inherent language/accent.
    effective_language_code = voice_map_entry.get("language_code", request_data.language_code)
    
    google_voice_config = {
        "language_code": effective_language_code,
        "name": voice_map_entry["google_voice_name"],
        # "ssml_gender": voice_map_entry.get("gender", "NEUTRAL").upper() # If needed by voice or API
    }

    # 2. Call Mocked TTS service
    try:
        audio_content_bytes = await mock_tts_client.synthesize_speech(
            input_text=request_data.text_to_speak,
            voice_config=google_voice_config,
            audio_config_params=request_data.audio_config,
            speaking_rate=request_data.voice_params.speaking_rate,
            pitch=request_data.voice_params.pitch
            # SSML could be constructed here if text_to_speak is intended as SSML
        )
    except Exception as e:
        print(f"Mock TTS synthesis error: {e}") # Log error
        raise HTTPException(status_code=503, detail="Speech synthesis service failed.")

    if not audio_content_bytes:
        raise HTTPException(status_code=500, detail="Speech synthesis produced no audio content.")

    # 3. Store audio to Mocked GCS
    file_extension = request_data.audio_config.audio_encoding.value.lower()
    blob_name = f"tts_audio/{uuid.uuid4()}.{file_extension}"
    content_type_map = {
        "MP3": "audio/mpeg",
        "LINEAR16": "audio/wav",
        "OGG_OPUS": "audio/ogg"
    }
    mime_type = content_type_map.get(request_data.audio_config.audio_encoding.value, "application/octet-stream")

    try:
        public_audio_url = await mock_storage_client.upload_blob_from_string(
            blob_name=blob_name,
            data=audio_content_bytes,
            content_type=mime_type
        )
    except Exception as e:
        print(f"Mock GCS upload error: {e}") # Log error
        raise HTTPException(status_code=500, detail="Failed to store synthesized audio.")

    # Estimate duration (very rough, real TTS API might provide this or use a library)
    # For MP3, avg 1 minute ~ 1MB at 128kbps.
    # For Linear16 (WAV), sample_rate * bits_per_sample/8 * num_channels * duration_seconds = file_size_bytes
    # This mock just uses text length.
    estimated_duration = len(request_data.text_to_speak) / 15.0 # Approx 15 chars/sec reading speed

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    return SynthesizeSpeechResponse(
        audio_url=public_audio_url,
        audio_duration_seconds=round(estimated_duration, 2),
        voice_used_details={
            "uplas_voice_character_name": uplas_char_name,
            "google_voice_name": google_voice_config["name"],
            "language_code": google_voice_config["language_code"]
        },
        text_character_count=len(request_data.text_to_speak),
        processing_time_ms=round(processing_time_ms, 2)
    )

# To run this FastAPI app locally (from within uplas-ai-agents/tts_agent/ directory):
# uvicorn main:app --reload --port 8002
