import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock # AsyncMock for async client methods
import uuid # For checking blob name format

# Import the FastAPI app and Pydantic models from main.py
from .main import (
    app,
    SynthesizeSpeechRequest,
    VoiceSelectionParams,
    AudioConfig,
    AudioEncodingEnum,
    UPLAS_VOICE_CHARACTER_MAP, # To access for testing mapping
    mock_storage_client # To inspect mock GCS after calls
)

client = TestClient(app)

# --- Fixtures ---
@pytest.fixture
def basic_tts_voice_params() -> VoiceSelectionParams:
    return VoiceSelectionParams(voice_character_name="alloy") # A known character from our map

@pytest.fixture
def basic_tts_request_payload(basic_tts_voice_params: VoiceSelectionParams) -> SynthesizeSpeechRequest:
    return SynthesizeSpeechRequest(
        text_to_speak="Hello Uplas world, this is a test.",
        language_code="en-US",
        voice_params=basic_tts_voice_params,
        audio_config=AudioConfig(audio_encoding=AudioEncodingEnum.MP3)
    )

# --- Test Cases ---

def test_synthesize_speech_success_default_params(basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test the main endpoint with a valid basic request and default MP3 encoding."""
    payload_dict = basic_tts_request_payload.model_dump()
    
    response = client.post("/v1/synthesize-speech", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    assert "audio_url" in data
    assert MOCKED_GCS_BUCKET_NAME in data["audio_url"] # Check if our mock bucket is in URL
    assert data["audio_url"].endswith(".mp3")
    assert "audio_duration_seconds" in data
    assert data["audio_duration_seconds"] > 0
    assert "voice_used_details" in data
    assert data["voice_used_details"]["uplas_voice_character_name"] == "alloy"
    assert data["voice_used_details"]["google_voice_name"] == UPLAS_VOICE_CHARACTER_MAP["alloy"]["google_voice_name"]
    assert data["text_character_count"] == len(basic_tts_request_payload.text_to_speak)
    assert "processing_time_ms" in data

    # Check if blob was "uploaded" to mock GCS
    blob_name_part = data["audio_url"].split('/')[-1]
    assert blob_name_part in mock_storage_client.blobs # Check by generated filename

def test_synthesize_speech_different_voice_character(basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test using a different voice character from the map."""
    payload_dict = basic_tts_request_payload.model_dump()
    payload_dict["voice_params"]["voice_character_name"] = "fable" # 'fable' uses en-GB
    
    response = client.post("/v1/synthesize-speech", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    assert data["voice_used_details"]["uplas_voice_character_name"] == "fable"
    assert data["voice_used_details"]["google_voice_name"] == UPLAS_VOICE_CHARACTER_MAP["fable"]["google_voice_name"]
    assert data["voice_used_details"]["language_code"] == UPLAS_VOICE_CHARACTER_MAP["fable"]["language_code"] # Should be en-GB

def test_synthesize_speech_unknown_voice_character_uses_default(basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test that an unknown voice character falls back to default."""
    payload_dict = basic_tts_request_payload.model_dump()
    payload_dict["voice_params"]["voice_character_name"] = "unknown_voice_for_test"
    
    response = client.post("/v1/synthesize-speech", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    assert data["voice_used_details"]["uplas_voice_character_name"] == "unknown_voice_for_test" # It should still reflect what was asked
    assert data["voice_used_details"]["google_voice_name"] == UPLAS_VOICE_CHARACTER_MAP["default"]["google_voice_name"]
    assert data["voice_used_details"]["language_code"] == UPLAS_VOICE_CHARACTER_MAP["default"]["language_code"]


def test_synthesize_speech_different_audio_encoding_wav(basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test LINEAR16 (WAV) audio encoding."""
    payload_dict = basic_tts_request_payload.model_dump()
    payload_dict["audio_config"]["audio_encoding"] = "LINEAR16" # WAV
    
    response = client.post("/v1/synthesize-speech", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    assert data["audio_url"].endswith(".linear16") # Check file extension reflects encoding

    # Verify content of mock audio data (optional, specific to mock implementation)
    blob_name_part = data["audio_url"].split('/')[-1]
    mock_audio_data = mock_storage_client.blobs.get(blob_name_part)
    assert mock_audio_data is not None
    assert b"ENCODING[LINEAR16]" in mock_audio_data


def test_synthesize_speech_custom_speaking_rate_and_pitch(basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test custom speaking rate and pitch parameters."""
    payload_dict = basic_tts_request_payload.model_dump()
    payload_dict["voice_params"]["speaking_rate"] = 1.2
    payload_dict["voice_params"]["pitch"] = -2.5
    
    # We need to verify that these params were "used" by the mock TTS client.
    # This might involve inspecting the print output from the mock or enhancing the mock
    # to store the parameters it was called with. For now, we trust it's passed.
    # A more robust test would patch the mock_tts_client.synthesize_speech itself.

    with patch('uplas-ai-agents.tts_agent.main.mock_tts_client.synthesize_speech', new_callable=AsyncMock) as mock_synthesize:
        # Make synthesize_speech return a valid bytes object
        mock_synthesize.return_value = b"mocked_audio_bytes_for_rate_pitch_test"
        
        response = client.post("/v1/synthesize-speech", json=payload_dict)
        assert response.status_code == 200
        
        # Assert that synthesize_speech was called with the correct rate and pitch
        mock_synthesize.assert_called_once()
        called_args = mock_synthesize.call_args[1] # Get keyword arguments
        assert called_args['speaking_rate'] == 1.2
        assert called_args['pitch'] == -2.5


def test_synthesize_speech_invalid_request_no_text(basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test request validation for missing text_to_speak."""
    payload_dict = basic_tts_request_payload.model_dump()
    payload_dict.pop("text_to_speak")
    
    response = client.post("/v1/synthesize-speech", json=payload_dict)
    assert response.status_code == 422 # FastAPI's Unprocessable Entity
    assert "text_to_speak" in response.json()["detail"][0]["loc"]

def test_synthesize_speech_invalid_request_empty_text(basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test request validation for empty text_to_speak."""
    payload_dict = basic_tts_request_payload.model_dump()
    payload_dict["text_to_speak"] = "" # Too short (min_length=1)
    
    response = client.post("/v1/synthesize-speech", json=payload_dict)
    assert response.status_code == 422
    assert "text_to_speak" in response.json()["detail"][0]["loc"]

@patch('uplas-ai-agents.tts_agent.main.mock_tts_client.synthesize_speech', new_callable=AsyncMock)
async def test_synthesize_speech_tts_service_failure(mock_synthesize_call, basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test error handling if the (mocked) TTS service call fails."""
    mock_synthesize_call.side_effect = Exception("Simulated TTS API Error")
    
    response = client.post("/v1/synthesize-speech", json=basic_tts_request_payload.model_dump())
    
    assert response.status_code == 503 # Our custom error for synthesis failure
    assert "speech synthesis service failed" in response.json()["detail"].lower()

@patch('uplas-ai-agents.tts_agent.main.mock_storage_client.upload_blob_from_string', new_callable=AsyncMock)
async def test_synthesize_speech_gcs_upload_failure(mock_gcs_upload, basic_tts_request_payload: SynthesizeSpeechRequest):
    """Test error handling if the (mocked) GCS upload fails."""
    # Mock TTS to succeed
    with patch('uplas-ai-agents.tts_agent.main.mock_tts_client.synthesize_speech', new_callable=AsyncMock, return_value=b"mock_audio"):
        mock_gcs_upload.side_effect = Exception("Simulated GCS Upload Error")
        
        response = client.post("/v1/synthesize-speech", json=basic_tts_request_payload.model_dump())
        
        assert response.status_code == 500
        assert "failed to store synthesized audio" in response.json()["detail"].lower()

# To run these tests:
# From within uplas-ai-agents/tts_agent/
# pytest
