
# uplas-ai-agents/tts_agent/test_main.py
import pytest
import pytest_asyncio # For async fixtures
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import os
import uuid
import json # Though not heavily used here, good to have for complex mocks if needed
from io import BytesIO

# Import the FastAPI app and Pydantic models from our refined main.py
from .main import (
    app,
    SynthesizeSpeechRequest,
    VoiceSelectionParams,
    AudioConfig,
    AudioEncodingEnum,
    SynthesisInputType, # InnovateAI: Added for SSML testing
    UPLAS_VOICE_CHARACTER_MAP, # For verifying voice selection
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    get_voice_config_from_character # We can test this helper directly
)
# For accessing Google's enum texttospeech.SsmlVoiceGender etc.
from google.cloud import texttospeech_v1 as texttospeech


# Set environment variables for testing (as in original)
os.environ["GCP_PROJECT_ID"] = "test-gcp-project-id-tts"
os.environ["TTS_AUDIO_GCS_BUCKET_NAME"] = "test-uplas-tts-bucket-innovate"

# Use TestClient for synchronous testing of async FastAPI app
client = TestClient(app)


# --- Fixtures (InnovateAI Refined & New) ---

@pytest.fixture
def basic_tts_request_dict() -> dict: # Renamed from payload for clarity
    """Provides a basic, valid request payload as a dictionary."""
    return {
        "input_type": "text", # Default to text
        "content_to_synthesize": "Hello Uplas world, this is an InnovateAI test.",
        "language_code": "en-US",
        "voice_params": {
            "voice_character_name": "susan_us_standard", # Using a refined name that implies tier
            "speaking_rate": 1.05,
            "pitch": -0.5
        },
        "audio_config": {"audio_encoding": "MP3"},
        "prefer_premium_quality": False # Default to standard for cost
    }

@pytest.fixture
def ssml_tts_request_dict(basic_tts_request_dict: dict) -> dict:
    """InnovateAI: Provides a request payload dictionary for SSML input."""
    payload = basic_tts_request_dict.copy()
    payload["input_type"] = SynthesisInputType.SSML.value
    payload["content_to_synthesize"] = "<speak>Hello <emphasis level='strong'>Uplas</emphasis> SSML world! This is <break time='300ms'/> an InnovateAI test.</speak>"
    payload["voice_params"]["voice_character_name"] = "trevor_us_standard"
    return payload

@pytest.fixture
def premium_quality_request_dict(basic_tts_request_dict: dict) -> dict:
    """InnovateAI: Provides a request payload dictionary preferring premium quality."""
    payload = basic_tts_request_dict.copy()
    payload["voice_params"]["voice_character_name"] = "susan_us" # Use base name
    payload["prefer_premium_quality"] = True
    return payload


@pytest.fixture
def mock_google_tts_api_response(): #
    """Mocks the response from Google's tts_client.synthesize_speech."""
    mock_response = MagicMock(spec=texttospeech.SynthesizeSpeechResponse)
    mock_response.audio_content = b"mock_innovateai_audio_bytes_for_tts_mp3"
    return mock_response

@pytest.fixture
def mock_gcs_blob_instance(): # Renamed from mock_gcs_blob for clarity
    """Mocks a GCS Blob object."""
    mock_blob = MagicMock(spec=storage.Blob)
    mock_blob.name = f"tts_audio/mock_audio_{uuid.uuid4()}.mp3" # Simulate a name
    mock_blob.public_url = f"https://storage.googleapis.com/{os.getenv('TTS_AUDIO_GCS_BUCKET_NAME')}/{mock_blob.name}" #
    mock_blob.upload_from_file = MagicMock() #
    return mock_blob

@pytest.fixture
def mock_gcs_bucket_instance(mock_gcs_blob_instance: MagicMock): # Renamed
    """Mocks a GCS Bucket object."""
    mock_bucket = MagicMock(spec=storage.Bucket)
    mock_bucket.blob.return_value = mock_gcs_blob_instance #
    return mock_bucket

# --- Centralized Mocking for GCP Clients (InnovateAI Enhancement) ---
@pytest_asyncio.fixture(autouse=True) # Autouse to apply to all tests in this file
async def mock_gcp_clients(monkeypatch):
    """
    InnovateAI: Automatically mocks tts_client and storage_client for all tests.
    Returns a dictionary of the mock instances for finer-grained control in tests if needed.
    """
    mock_tts_client = AsyncMock(spec=texttospeech.TextToSpeechAsyncClient)
    mock_storage_client = MagicMock(spec=storage.Client)
    
    monkeypatch.setattr("tts_agent.main.tts_client", mock_tts_client)
    monkeypatch.setattr("tts_agent.main.storage_client", mock_storage_client)
    
    # Also patch the global instances if they are used directly after import (though main.py uses module-level)
    # This ensures that if main.py's global tts_client/storage_client were accessed, they're patched.
    # However, our main.py uses module-level clients that are set at import time.
    # The setattr above should correctly patch those instances referenced by the endpoint.

    return {"tts_client": mock_tts_client, "storage_client": mock_storage_client}


# --- Test Cases (InnovateAI Refined & New) ---

async def test_synthesize_speech_success_text_input(
    mock_gcp_clients: Dict[str, MagicMock],
    basic_tts_request_dict: dict,
    mock_google_tts_api_response: MagicMock,
    mock_gcs_bucket_instance: MagicMock,
    mock_gcs_blob_instance: MagicMock
): # Based on
    """InnovateAI Test: Successful synthesis with TEXT input and default MP3 encoding."""
    mock_gcp_clients["tts_client"].synthesize_speech.return_value = mock_google_tts_api_response #
    mock_gcp_clients["storage_client"].bucket.return_value = mock_gcs_bucket_instance #

    response = client.post("/v1/synthesize-speech", json=basic_tts_request_dict)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "audio_url" in data
    assert os.getenv("TTS_AUDIO_GCS_BUCKET_NAME") in data["audio_url"]
    assert data["audio_url"].endswith(".mp3")
    assert "voice_used_details" in data
    assert data["voice_used_details"]["uplas_voice_character_name"] == "susan_us_standard"
    
    expected_voice_config = UPLAS_VOICE_CHARACTER_MAP["susan_us_standard"] #
    assert data["voice_used_details"]["google_voice_name"] == expected_voice_config["google_voice_name"] #
    assert data["voice_used_details"]["language_code"] == expected_voice_config["language_code"] #
    assert data["voice_used_details"]["quality_tier_used"] == "standard" # InnovateAI Check
    assert data["text_character_count"] == len(basic_tts_request_dict["content_to_synthesize"]) #

    # Verify calls
    mock_gcp_clients["tts_client"].synthesize_speech.assert_awaited_once() #
    tts_call_args = mock_gcp_clients["tts_client"].synthesize_speech.call_args[1] # kwargs
    assert tts_call_args['input'].text == basic_tts_request_dict["content_to_synthesize"] #
    assert tts_call_args['voice'].name == expected_voice_config["google_voice_name"] #
    assert tts_call_args['audio_config'].audio_encoding == texttospeech.AudioEncoding.MP3 # Check actual enum value

    mock_gcp_clients["storage_client"].bucket.assert_called_once_with(os.getenv("TTS_AUDIO_GCS_BUCKET_NAME")) #
    mock_gcs_bucket_instance.blob.assert_called_once() #
    blob_name_arg = mock_gcs_bucket_instance.blob.call_args[0][0] #
    assert blob_name_arg.startswith("tts_audio/") and blob_name_arg.endswith(".mp3") #
    mock_gcs_blob_instance.upload_from_file.assert_called_once() #
    # Check that upload_from_file was called with a BytesIO object
    upload_arg = mock_gcs_blob_instance.upload_from_file.call_args[0][0]
    assert isinstance(upload_arg, BytesIO)


async def test_synthesize_speech_success_ssml_input(
    mock_gcp_clients: Dict[str, MagicMock],
    ssml_tts_request_dict: dict, # Using SSML fixture
    mock_google_tts_api_response: MagicMock,
    mock_gcs_bucket_instance: MagicMock
):
    """InnovateAI Test: Successful synthesis with SSML input."""
    mock_gcp_clients["tts_client"].synthesize_speech.return_value = mock_google_tts_api_response
    mock_gcp_clients["storage_client"].bucket.return_value = mock_gcs_bucket_instance

    response = client.post("/v1/synthesize-speech", json=ssml_tts_request_dict)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["voice_used_details"]["uplas_voice_character_name"] == "trevor_us_standard"

    # Verify that SSML input was used in the TTS call
    mock_gcp_clients["tts_client"].synthesize_speech.assert_awaited_once()
    tts_call_args = mock_gcp_clients["tts_client"].synthesize_speech.call_args[1]
    assert tts_call_args['input'].ssml == ssml_tts_request_dict["content_to_synthesize"]
    assert not tts_call_args['input'].text # Ensure text field is not populated for SSML


async def test_synthesize_speech_prefer_premium_quality_selects_wavenet(
    mock_gcp_clients: Dict[str, MagicMock],
    premium_quality_request_dict: dict, # Uses "susan_us" base name and prefer_premium_quality=True
    mock_google_tts_api_response: MagicMock,
    mock_gcs_bucket_instance: MagicMock
):
    """InnovateAI Test: `prefer_premium_quality` flag leads to selection of WaveNet voice."""
    mock_gcp_clients["tts_client"].synthesize_speech.return_value = mock_google_tts_api_response
    mock_gcp_clients["storage_client"].bucket.return_value = mock_gcs_bucket_instance

    response = client.post("/v1/synthesize-speech", json=premium_quality_request_dict)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # get_voice_config_from_character should have picked "susan_us_wavenet"
    expected_voice_config = UPLAS_VOICE_CHARACTER_MAP["susan_us_wavenet"]
    assert data["voice_used_details"]["google_voice_name"] == expected_voice_config["google_voice_name"]
    assert data["voice_used_details"]["quality_tier_used"] == "wavenet"
    
    tts_call_args = mock_gcp_clients["tts_client"].synthesize_speech.call_args[1]
    assert tts_call_args['voice'].name == expected_voice_config["google_voice_name"]


async def test_synthesize_speech_default_quality_selects_standard(
    mock_gcp_clients: Dict[str, MagicMock],
    basic_tts_request_dict: dict, # This uses "susan_us_standard" explicitly, let's make it use base name
    mock_google_tts_api_response: MagicMock,
    mock_gcs_bucket_instance: MagicMock
):
    """InnovateAI Test: Default `prefer_premium_quality=False` selects Standard voice."""
    payload = basic_tts_request_dict.copy()
    payload["voice_params"]["voice_character_name"] = "susan_us" # Use base name
    payload["prefer_premium_quality"] = False # Explicitly false or omitted

    mock_gcp_clients["tts_client"].synthesize_speech.return_value = mock_google_tts_api_response
    mock_gcp_clients["storage_client"].bucket.return_value = mock_gcs_bucket_instance

    response = client.post("/v1/synthesize-speech", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # get_voice_config_from_character should have picked "susan_us_standard"
    expected_voice_config = UPLAS_VOICE_CHARACTER_MAP["susan_us_standard"]
    assert data["voice_used_details"]["google_voice_name"] == expected_voice_config["google_voice_name"]
    assert data["voice_used_details"]["quality_tier_used"] == "standard"


async def test_synthesize_speech_linear16_encoding(
    mock_gcp_clients: Dict[str, MagicMock],
    basic_tts_request_dict: dict,
    mock_google_tts_api_response: MagicMock,
    mock_gcs_bucket_instance: MagicMock,
    mock_gcs_blob_instance: MagicMock # Needed to modify its public_url
): # Based on
    """InnovateAI Test: LINEAR16 (WAV) audio encoding works as expected."""
    # Update public_url for the blob to reflect .wav extension for this test
    mock_gcs_blob_instance.public_url = f"https://storage.googleapis.com/{os.getenv('TTS_AUDIO_GCS_BUCKET_NAME')}/tts_audio/mock_audio_innovate.wav" #
    mock_gcs_blob_instance.name = "tts_audio/mock_audio_innovate.wav" # Ensure name also updated if used

    mock_gcp_clients["tts_client"].synthesize_speech.return_value = mock_google_tts_api_response #
    mock_gcp_clients["storage_client"].bucket.return_value = mock_gcs_bucket_instance #
    
    payload = basic_tts_request_dict.copy()
    payload["audio_config"]["audio_encoding"] = "LINEAR16" #

    response = client.post("/v1/synthesize-speech", json=payload) #
    
    assert response.status_code == status.HTTP_200_OK #
    data = response.json()
    assert data["audio_url"].endswith(".wav") #

    tts_call_args = mock_gcp_clients["tts_client"].synthesize_speech.call_args[1] #
    assert tts_call_args['audio_config'].audio_encoding == texttospeech.AudioEncoding.LINEAR16 #
    
    upload_call_kwargs = mock_gcs_blob_instance.upload_from_file.call_args.kwargs #
    assert upload_call_kwargs['content_type'] == "audio/wav" #


def test_synthesize_speech_invalid_request_no_content(basic_tts_request_dict: dict): # Changed from no_text
    """InnovateAI Test: Request validation for missing content_to_synthesize."""
    payload = basic_tts_request_dict.copy()
    del payload["content_to_synthesize"] # Renamed from text_to_speak
    
    response = client.post("/v1/synthesize-speech", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY #
    assert any(err["loc"] == ["body", "content_to_synthesize"] for err in response.json()["detail"]) #


async def test_synthesize_speech_tts_api_failure(
    mock_gcp_clients: Dict[str, MagicMock], basic_tts_request_dict: dict
): #
    """InnovateAI Test: Error handling if the Google TTS API call fails."""
    mock_gcp_clients["tts_client"].synthesize_speech.side_effect = Exception("Simulated InnovateAI Google TTS API Error") #
    
    response = client.post("/v1/synthesize-speech", json=basic_tts_request_dict) #
    
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE #
    assert "speech synthesis service failed" in response.json()["detail"].lower() #
    assert "simulated innovateai google tts api error" in response.json()["detail"].lower() #


async def test_synthesize_speech_gcs_upload_failure(
    mock_gcp_clients: Dict[str, MagicMock],
    basic_tts_request_dict: dict,
    mock_google_tts_api_response: MagicMock,
    mock_gcs_bucket_instance: MagicMock, # Bucket mock is still needed
    mock_gcs_blob_instance: MagicMock   # Blob mock for upload failure
): #
    """InnovateAI Test: Error handling if the GCS upload fails."""
    mock_gcp_clients["tts_client"].synthesize_speech.return_value = mock_google_tts_api_response #
    mock_gcp_clients["storage_client"].bucket.return_value = mock_gcs_bucket_instance #
    mock_gcs_blob_instance.upload_from_file.side_effect = Exception("Simulated InnovateAI GCS Upload Error") #
        
    response = client.post("/v1/synthesize-speech", json=basic_tts_request_dict) #
    
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR #
    assert "failed to store synthesized audio" in response.json()["detail"].lower() #
    assert "simulated innovateai gcs upload error" in response.json()["detail"].lower() #


# --- InnovateAI Tests for get_voice_config_from_character helper ---

@pytest.mark.parametrize("char_base, req_lang, prefer_premium, expected_google_voice, expected_tier", [
    ("susan_us", "en-US", False, UPLAS_VOICE_CHARACTER_MAP["susan_us_standard"]["google_voice_name"], "standard"),
    ("susan_us", "en-US", True, UPLAS_VOICE_CHARACTER_MAP["susan_us_wavenet"]["google_voice_name"], "wavenet"),
    ("susan_us", "en-US", True, UPLAS_VOICE_CHARACTER_MAP["susan_us_wavenet"]["google_voice_name"], "wavenet"), # Test studio if susan_us_studio existed and was preferred
    ("elodie_fr", "fr-FR", False, UPLAS_VOICE_CHARACTER_MAP["elodie_fr_standard"]["google_voice_name"], "standard"),
    ("elodie_fr", "fr-FR", True, UPLAS_VOICE_CHARACTER_MAP["elodie_fr_wavenet"]["google_voice_name"], "wavenet"),
    ("unknown_char", "en-US", False, UPLAS_VOICE_CHARACTER_MAP["default_en_us_standard"]["google_voice_name"], "standard"), # Fallback to default for lang
    ("unknown_char", "xx-YY", False, UPLAS_VOICE_CHARACTER_MAP["default_en_us_standard"]["google_voice_name"], "standard"), # Fallback to global default
    ("priya_in", "hi-IN", True, UPLAS_VOICE_CHARACTER_MAP["priya_in_wavenet"]["google_voice_name"], "wavenet"),
    # Test case where premium is preferred but only standard exists for that character base (e.g. if "charx_lang_wavenet" is missing)
    # This depends on how UPLAS_VOICE_CHARACTER_MAP is populated. Assume direct tiered names for now.
])
def test_get_voice_config_selection_logic(char_base, req_lang, prefer_premium, expected_google_voice, expected_tier):
    """InnovateAI Test: Verifies the voice selection logic of get_voice_config_from_character."""
    config = get_voice_config_from_character(char_base, req_lang, prefer_premium)
    assert config["google_voice_name"] == expected_google_voice
    assert config["quality_tier"] == expected_tier
    if req_lang and req_lang in SUPPORTED_LANGUAGES: # If valid requested lang, it should try to match or fallback gracefully
        if "unknown_char" not in char_base : # For known chars, lang should match if specific variant exists
             if any(char_base in k and req_lang.lower() in k.lower() for k in UPLAS_VOICE_CHARACTER_MAP.keys()):
                assert config["language_code"] == req_lang


def test_health_check_endpoint_healthy(monkeypatch): # Modified to use monkeypatch for env vars
    """InnovateAI Test: Health check returns healthy when configured.""" #
    # Using monkeypatch from pytest to set env vars for the test's scope
    monkeypatch.setenv("GCP_PROJECT_ID", "test-project-tts-health") #
    monkeypatch.setenv("TTS_AUDIO_GCS_BUCKET_NAME", "test-bucket-tts-health") #

    # Patch the global clients used by the health check if they are module-level
    with patch('tts_agent.main.tts_client', new=MagicMock(spec=texttospeech.TextToSpeechAsyncClient)), \
         patch('tts_agent.main.storage_client', new=MagicMock(spec=storage.Client)): #
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK #
        assert response.json() == {"status": "healthy", "service": "TTS_Agent", "innovate_ai_enhancements_active": True} # InnovateAI addition


def test_health_check_endpoint_unhealthy_missing_config(monkeypatch): #
    """InnovateAI Test: Health check for missing GCP_PROJECT_ID."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False) #
    # Ensure TTS_AUDIO_GCS_BUCKET_NAME is set for this specific check
    monkeypatch.setenv("TTS_AUDIO_GCS_BUCKET_NAME", "test-bucket-tts-health-unhealthy")

    with patch('tts_agent.main.tts_client', new=MagicMock()), \
         patch('tts_agent.main.storage_client', new=MagicMock()): #
        response = client.get("/health") #
        assert response.status_code == status.HTTP_200_OK # Health check itself is 200
        data = response.json()
        assert data["status"] == "unhealthy" #
        assert "GCP_PROJECT_ID" in data["reason"] #
