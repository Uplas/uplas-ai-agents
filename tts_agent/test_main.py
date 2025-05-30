# uplas-ai-agents/tts_agent/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import os
import uuid

# Import the FastAPI app and Pydantic models from main.py
from .main import (
    app,
    SynthesizeSpeechRequest,
    VoiceSelectionParams,
    AudioConfig,
    AudioEncodingEnum,
    UPLAS_VOICE_CHARACTER_MAP,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE
    # tts_client and storage_client will be patched
)

# Set environment variables for testing
os.environ["GCP_PROJECT_ID"] = "test-gcp-project-id"
os.environ["TTS_AUDIO_GCS_BUCKET_NAME"] = "test-uplas-tts-bucket"

client = TestClient(app)

# --- Fixtures ---
@pytest.fixture
def basic_tts_request_payload() -> Dict:
    return {
        "text_to_speak": "Hello Uplas world, this is a test.",
        "language_code": "en-US", # Can be overridden by voice character's default
        "voice_params": {
            "voice_character_name": "alloy_us", # A known character from our map
            "speaking_rate": 1.0,
            "pitch": 0.0
        },
        "audio_config": {"audio_encoding": "MP3"}
    }

@pytest.fixture
def mock_google_tts_response():
    # Create a mock object that mimics the structure of google.cloud.texttospeech.SynthesizeSpeechResponse
    mock_response = MagicMock()
    mock_response.audio_content = b"mock_audio_bytes_mp3"
    return mock_response

@pytest.fixture
def mock_gcs_blob():
    mock_blob = MagicMock()
    mock_blob.public_url = f"https://storage.googleapis.com/{os.getenv('TTS_AUDIO_GCS_BUCKET_NAME')}/tts_audio/mock_audio_{uuid.uuid4()}.mp3"
    # Mock the upload_from_file method if needed, or it can be part of the bucket mock
    mock_blob.upload_from_file = MagicMock()
    return mock_blob

@pytest.fixture
def mock_gcs_bucket(mock_gcs_blob: MagicMock):
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_gcs_blob
    return mock_bucket

# --- Test Cases ---

@patch('uplas_ai_agents.tts_agent.main.storage_client')
@patch('uplas_ai_agents.tts_agent.main.tts_client', new_callable=AsyncMock) # tts_client is async
async def test_synthesize_speech_success_default_params(
    mock_tts_client_instance: AsyncMock, # Patched tts_client from main
    mock_storage_client_instance: MagicMock, # Patched storage_client from main
    basic_tts_request_payload: Dict,
    mock_google_tts_response: MagicMock,
    mock_gcs_bucket: MagicMock,
    mock_gcs_blob: MagicMock
):
    """Test the main endpoint with a valid basic request and default MP3 encoding."""
    mock_tts_client_instance.synthesize_speech.return_value = mock_google_tts_response
    mock_storage_client_instance.bucket.return_value = mock_gcs_bucket

    response = client.post("/v1/synthesize-speech", json=basic_tts_request_payload)

    assert response.status_code == 200
    data = response.json()
    assert "audio_url" in data
    assert os.getenv("TTS_AUDIO_GCS_BUCKET_NAME") in data["audio_url"]
    assert data["audio_url"].endswith(".mp3")
    assert "audio_duration_seconds" in data # This is an estimation in the current main.py
    assert "voice_used_details" in data
    assert data["voice_used_details"]["uplas_voice_character_name"] == "alloy_us"
    
    expected_google_voice_name = UPLAS_VOICE_CHARACTER_MAP["alloy_us"]["google_voice_name"]
    assert data["voice_used_details"]["google_voice_name"] == expected_google_voice_name
    assert data["voice_used_details"]["language_code"] == UPLAS_VOICE_CHARACTER_MAP["alloy_us"]["language_code"]
    assert data["text_character_count"] == len(basic_tts_request_payload["text_to_speak"])

    # Verify calls
    mock_tts_client_instance.synthesize_speech.assert_awaited_once()
    tts_call_args = mock_tts_client_instance.synthesize_speech.call_args[1] # kwargs
    assert tts_call_args['input'].text == basic_tts_request_payload["text_to_speak"]
    assert tts_call_args['voice'].name == expected_google_voice_name
    assert tts_call_args['voice'].language_code == UPLAS_VOICE_CHARACTER_MAP["alloy_us"]["language_code"]
    assert tts_call_args['audio_config'].audio_encoding.name == "MP3" # Check enum name

    mock_storage_client_instance.bucket.assert_called_once_with(os.getenv("TTS_AUDIO_GCS_BUCKET_NAME"))
    mock_gcs_bucket.blob.assert_called_once() # Check that a blob was created
    blob_name_arg = mock_gcs_bucket.blob.call_args[0][0]
    assert blob_name_arg.startswith("tts_audio/")
    assert blob_name_arg.endswith(".mp3")
    mock_gcs_blob.upload_from_file.assert_called_once()


@patch('uplas_ai_agents.tts_agent.main.storage_client')
@patch('uplas_ai_agents.tts_agent.main.tts_client', new_callable=AsyncMock)
async def test_synthesize_speech_different_voice_and_language(
    mock_tts_client_instance: AsyncMock,
    mock_storage_client_instance: MagicMock,
    basic_tts_request_payload: Dict,
    mock_google_tts_response: MagicMock,
    mock_gcs_bucket: MagicMock
):
    """Test using a different voice character that implies a different language."""
    mock_tts_client_instance.synthesize_speech.return_value = mock_google_tts_response
    mock_storage_client_instance.bucket.return_value = mock_gcs_bucket

    payload = basic_tts_request_payload.copy()
    payload["voice_params"]["voice_character_name"] = "elodie_fr_standard"
    # language_code in request can be None or fr-FR, voice character map should take precedence if specific
    payload["language_code"] = "fr-FR" # Explicitly setting, but could be None

    response = client.post("/v1/synthesize-speech", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["voice_used_details"]["uplas_voice_character_name"] == "elodie_fr_standard"
    expected_voice_details = UPLAS_VOICE_CHARACTER_MAP["elodie_fr_standard"]
    assert data["voice_used_details"]["google_voice_name"] == expected_voice_details["google_voice_name"]
    assert data["voice_used_details"]["language_code"] == expected_voice_details["language_code"] # Should be fr-FR

    # Verify TTS call used the French voice
    tts_call_args = mock_tts_client_instance.synthesize_speech.call_args[1]
    assert tts_call_args['voice'].name == expected_voice_details["google_voice_name"]
    assert tts_call_args['voice'].language_code == expected_voice_details["language_code"]


@patch('uplas_ai_agents.tts_agent.main.storage_client')
@patch('uplas_ai_agents.tts_agent.main.tts_client', new_callable=AsyncMock)
async def test_synthesize_speech_unknown_voice_character_uses_fallback(
    mock_tts_client_instance: AsyncMock,
    mock_storage_client_instance: MagicMock,
    basic_tts_request_payload: Dict,
    mock_google_tts_response: MagicMock,
    mock_gcs_bucket: MagicMock
):
    """Test that an unknown voice character falls back to a default voice (for the requested lang or global default)."""
    mock_tts_client_instance.synthesize_speech.return_value = mock_google_tts_response
    mock_storage_client_instance.bucket.return_value = mock_gcs_bucket

    payload = basic_tts_request_payload.copy()
    payload["voice_params"]["voice_character_name"] = "unknown_voice_for_test"
    payload["language_code"] = "es-ES" # Request Spanish

    response = client.post("/v1/synthesize-speech", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    # The voice used should be a Spanish voice from the map, as "unknown_voice_for_test" isn't found
    # and language_code "es-ES" is requested.
    # The `get_voice_config_from_character` logic will pick the first Spanish voice.
    expected_voice_details = UPLAS_VOICE_CHARACTER_MAP["sofia_es_standard"] # Assuming this is the first es-ES
    assert data["voice_used_details"]["google_voice_name"] == expected_voice_details["google_voice_name"]
    assert data["voice_used_details"]["language_code"] == "es-ES"
    
    # Verify TTS call used the fallback Spanish voice
    tts_call_args = mock_tts_client_instance.synthesize_speech.call_args[1]
    assert tts_call_args['voice'].name == expected_voice_details["google_voice_name"]
    assert tts_call_args['voice'].language_code == "es-ES"


@patch('uplas_ai_agents.tts_agent.main.storage_client')
@patch('uplas_ai_agents.tts_agent.main.tts_client', new_callable=AsyncMock)
async def test_synthesize_speech_linear16_encoding(
    mock_tts_client_instance: AsyncMock,
    mock_storage_client_instance: MagicMock,
    basic_tts_request_payload: Dict,
    mock_google_tts_response: MagicMock, # Re-use, content doesn't matter for this test
    mock_gcs_bucket: MagicMock,
    mock_gcs_blob: MagicMock
):
    """Test LINEAR16 (WAV) audio encoding."""
    mock_tts_client_instance.synthesize_speech.return_value = mock_google_tts_response
    mock_storage_client_instance.bucket.return_value = mock_gcs_bucket
    # Update public_url for the blob to reflect .wav extension for this test
    mock_gcs_blob.public_url = f"https://storage.googleapis.com/{os.getenv('TTS_AUDIO_GCS_BUCKET_NAME')}/tts_audio/mock_audio_{uuid.uuid4()}.wav"


    payload = basic_tts_request_payload.copy()
    payload["audio_config"]["audio_encoding"] = "LINEAR16"

    response = client.post("/v1/synthesize-speech", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["audio_url"].endswith(".wav")

    # Verify audio_config in TTS call
    tts_call_args = mock_tts_client_instance.synthesize_speech.call_args[1]
    from google.cloud import texttospeech as tts_gapic # For enum access
    assert tts_call_args['audio_config'].audio_encoding == tts_gapic.AudioEncoding.LINEAR16
    
    # Verify GCS blob name and content type
    mock_gcs_bucket.blob.assert_called_once()
    blob_name_arg = mock_gcs_bucket.blob.call_args[0][0]
    assert blob_name_arg.endswith(".wav")
    
    mock_gcs_blob.upload_from_file.assert_called_once()
    upload_call_kwargs = mock_gcs_blob.upload_from_file.call_args.kwargs
    assert upload_call_kwargs['content_type'] == "audio/wav"


def test_synthesize_speech_invalid_request_no_text(basic_tts_request_payload: Dict):
    """Test request validation for missing text_to_speak."""
    payload = basic_tts_request_payload.copy()
    del payload["text_to_speak"]
    
    response = client.post("/v1/synthesize-speech", json=payload)
    assert response.status_code == 422
    assert any(err["loc"] == ["body", "text_to_speak"] for err in response.json()["detail"])


@patch('uplas_ai_agents.tts_agent.main.tts_client', new_callable=AsyncMock)
async def test_synthesize_speech_tts_api_failure(
    mock_tts_client_instance: AsyncMock,
    basic_tts_request_payload: Dict
):
    """Test error handling if the Google TTS API call fails."""
    mock_tts_client_instance.synthesize_speech.side_effect = Exception("Simulated Google TTS API Error")
    
    response = client.post("/v1/synthesize-speech", json=basic_tts_request_payload)
    
    assert response.status_code == 503
    assert "speech synthesis service failed" in response.json()["detail"].lower()
    assert "simulated google tts api error" in response.json()["detail"].lower()


@patch('uplas_ai_agents.tts_agent.main.storage_client')
@patch('uplas_ai_agents.tts_agent.main.tts_client', new_callable=AsyncMock)
async def test_synthesize_speech_gcs_upload_failure(
    mock_tts_client_instance: AsyncMock,
    mock_storage_client_instance: MagicMock,
    basic_tts_request_payload: Dict,
    mock_google_tts_response: MagicMock,
    mock_gcs_bucket: MagicMock, # Bucket mock is still needed
    mock_gcs_blob: MagicMock   # Blob mock for upload failure
):
    """Test error handling if the GCS upload fails."""
    mock_tts_client_instance.synthesize_speech.return_value = mock_google_tts_response
    mock_storage_client_instance.bucket.return_value = mock_gcs_bucket
    mock_gcs_blob.upload_from_file.side_effect = Exception("Simulated GCS Upload Error")
        
    response = client.post("/v1/synthesize-speech", json=basic_tts_request_payload)
    
    assert response.status_code == 500
    assert "failed to store synthesized audio" in response.json()["detail"].lower()
    assert "simulated gcs upload error" in response.json()["detail"].lower()


def test_health_check_endpoint_healthy():
    # Ensure env vars are set for a healthy response
    original_project_id = os.environ.get("GCP_PROJECT_ID")
    original_bucket_name = os.environ.get("TTS_AUDIO_GCS_BUCKET_NAME")
    os.environ["GCP_PROJECT_ID"] = "test-project"
    os.environ["TTS_AUDIO_GCS_BUCKET_NAME"] = "test-bucket"

    # We also need to ensure tts_client and storage_client appear initialized
    # This is tricky as they are global. For a robust test, you might re-import main
    # or have a way to reset them. Here, we assume they were initialized.
    # A better way would be to patch them at the module level for this specific test.
    with patch('uplas_ai_agents.tts_agent.main.tts_client', new=MagicMock()), \
         patch('uplas_ai_agents.tts_agent.main.storage_client', new=MagicMock()):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy", "service": "TTS_Agent"}
    
    # Restore original env vars if they existed
    if original_project_id: os.environ["GCP_PROJECT_ID"] = original_project_id
    else: del os.environ["GCP_PROJECT_ID"]
    if original_bucket_name: os.environ["TTS_AUDIO_GCS_BUCKET_NAME"] = original_bucket_name
    else: del os.environ["TTS_AUDIO_GCS_BUCKET_NAME"]


def test_health_check_endpoint_unhealthy_missing_config():
    original_project_id = os.environ.get("GCP_PROJECT_ID")
    # Simulate missing GCP_PROJECT_ID
    if "GCP_PROJECT_ID" in os.environ: del os.environ["GCP_PROJECT_ID"]

    with patch('uplas_ai_agents.tts_agent.main.tts_client', new=MagicMock()), \
         patch('uplas_ai_agents.tts_agent.main.storage_client', new=MagicMock()):
        response = client.get("/health")
        assert response.status_code == 200 # Health check itself doesn't fail with 500
        assert response.json()["status"] == "unhealthy"
        assert "Missing GCP_PROJECT_ID" in response.json()["reason"]
    
    if original_project_id: os.environ["GCP_PROJECT_ID"] = original_project_id

# To run these tests:
# From within uplas-ai-agents/tts_agent/
# pytest

