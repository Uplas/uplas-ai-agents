# uplas-ai-agents/ttv_agent/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import os
import uuid
import httpx # For mocking its client

# Import the FastAPI app and Pydantic models from main.py
from .main import (
    app,
    GenerateVideoRequest,
    UserProfileSnapshotForTTV,
    ContentSource,
    InstructorChars,
    VideoGenerationJobStatus,
    video_jobs # Import the global job store for inspection/manipulation in tests
)

# Set environment variables for testing
# These are crucial for the main TTV agent's initialization and operation logic
os.environ["GCP_PROJECT_ID"] = "test-gcp-project"
os.environ["TTV_GCS_BUCKET_NAME"] = "test-ttv-bucket"
os.environ["DJANGO_TTV_CALLBACK_URL"] = "http://mock-django/api/internal/ttv-callback/"
os.environ["AI_TUTOR_AGENT_URL"] = "http://mock-ai-tutor-agent"
os.environ["TTS_AGENT_URL"] = "http://mock-tts-agent"
os.environ["THIRD_PARTY_AVATAR_API_KEY"] = "mock_avatar_api_key"
os.environ["THIRD_PARTY_AVATAR_BASE_URL"] = "http://mock-avatar-service.com/api"


client = TestClient(app)

# --- Fixtures ---
@pytest.fixture
def basic_user_profile_ttv() -> UserProfileSnapshotForTTV:
    return UserProfileSnapshotForTTV(
        industry="Tech",
        profession="Developer",
        country="Testland",
        city="Testville",
        career_interest="AI",
        learning_goals="Build cool TTV stuff",
        preferred_tutor_persona="Witty"
    )

@pytest.fixture
def basic_ttv_request_payload_dict(basic_user_profile_ttv: UserProfileSnapshotForTTV) -> Dict:
    return {
        "user_id": "test_ttv_user_001",
        "content_source": {"raw_text_content": "Explain the concept of asynchronous programming simply."},
        "instructor_character": InstructorChars.SUSAN.value, # Use enum value
        "user_profile_snapshot": basic_user_profile_ttv.model_dump(),
        "language_code": "en-US",
        "preferred_video_length_minutes": "1-2" # Shorter for testing
    }

@pytest.fixture(autouse=True)
def clear_video_jobs_store():
    """Clears the in-memory video_jobs store before each test."""
    original_avatar_client = app.state.avatar_service_client if hasattr(app.state, 'avatar_service_client') else None
    video_jobs.clear()
    yield # Test runs here
    video_jobs.clear()
    if original_avatar_client: # Restore if it was changed by a test
        app.state.avatar_service_client = original_avatar_client


# --- Mock Responses for External Services ---
@pytest.fixture
def mock_ai_tutor_response() -> Dict:
    return {"answer_text": "[SCENE] Mocked Script: Asynchronous programming is like ordering coffee..."}

@pytest.fixture
def mock_tts_agent_response() -> Dict:
    return {
        "audio_url": "gs://test-uplas-tts-bucket/mock_audio.mp3",
        "audio_duration_seconds": 60.5
    }

@pytest.fixture
def mock_avatar_submit_response() -> Dict:
    return {"service_job_id": f"avatar_svc_job_{uuid.uuid4()}", "initial_status": "queued"}

@pytest.fixture
def mock_avatar_poll_processing_response() -> Dict:
    return {"status": "processing", "progressPercent": 50}

@pytest.fixture
def mock_avatar_poll_completed_response() -> Dict:
    return {
        "status": "completed",
        "videoUrl": "https://mock-avatar-service.com/videos/completed_video.mp4",
        "thumbnailUrl": "https://mock-avatar-service.com/videos/completed_thumb.jpg",
        "durationSeconds": 58.0
    }

@pytest.fixture
def mock_avatar_poll_failed_response() -> Dict:
    return {"status": "failed", "errorMessage": "Simulated avatar rendering failure."}


# --- Test Cases for Endpoints ---

def test_generate_video_endpoint_success(basic_ttv_request_payload_dict: Dict):
    """Test the /v1/generate-video endpoint successfully queues a job."""
    # Patch the background task so it doesn't run immediately during this specific test
    with patch('uplas_ai_agents.ttv_agent.main.process_video_generation_task', new_callable=AsyncMock) as mock_process_task:
        response = client.post("/v1/generate-video", json=basic_ttv_request_payload_dict)

        assert response.status_code == 202 # Accepted
        data = response.json()
        assert "job_id" in data
        job_id = data["job_id"]
        assert data["status"] == VideoGenerationJobStatus.PENDING.value
        assert "Video generation task accepted" in data["message"]

        assert job_id in video_jobs # Check if job was added to our mock store
        assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.PENDING
        assert video_jobs[job_id]["user_id"] == basic_ttv_request_payload_dict["user_id"]

        # Check if background task was called with correct arguments
        mock_process_task.assert_awaited_once()
        # The first arg to add_task's target is job_id, then request_data (as Pydantic model)
        called_job_id, called_request_data_model = mock_process_task.call_args[0]
        assert called_job_id == job_id
        assert isinstance(called_request_data_model, GenerateVideoRequest)
        assert called_request_data_model.user_id == basic_ttv_request_payload_dict["user_id"]
        assert called_request_data_model.content_source.raw_text_content == basic_ttv_request_payload_dict["content_source"]["raw_text_content"]


def test_get_video_status_job_exists_and_completed(basic_ttv_request_payload_dict: Dict):
    """Test retrieving status for an existing and completed job."""
    # First, create a job (mocking the background task)
    with patch('uplas_ai_agents.ttv_agent.main.process_video_generation_task', new_callable=AsyncMock):
        initial_response = client.post("/v1/generate-video", json=basic_ttv_request_payload_dict)
    job_id = initial_response.json()["job_id"]

    # Manually update job status in our mock store for testing this endpoint
    mock_video_url = "https://mocked-uplas-ttv-videos/final_videos/video_susan_mock.mp4"
    video_jobs[job_id].update({
        "status": VideoGenerationJobStatus.COMPLETED,
        "video_url": mock_video_url,
        "thumbnail_url": "https://mocked-uplas-ttv-videos/final_videos/thumb_susan_mock.jpg",
        "video_duration_seconds": 185.5,
        "script_generated_preview": "Once upon a time...",
        "character_used": InstructorChars.SUSAN.value,
        "attire_used": "default_susan_attire_id"
    })

    status_response = client.get(f"/v1/video-status/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == VideoGenerationJobStatus.COMPLETED.value
    assert data["video_url"] == mock_video_url
    assert data["video_duration_seconds"] == 185.5
    assert data["script_generated_preview"] == "Once upon a time..."
    assert data["character_used"] == InstructorChars.SUSAN.value


def test_get_video_status_job_not_found():
    """Test retrieving status for a non-existent job ID."""
    non_existent_job_id = f"ttvjob_{uuid.uuid4()}"
    response = client.get(f"/v1/video-status/{non_existent_job_id}")
    assert response.status_code == 404
    assert "Job ID not found" in response.json()["detail"]


# --- Test Cases for the Background Task (process_video_generation_task) ---
# These are more like integration tests for the mocked pipeline.

@patch('uplas_ai_agents.ttv_agent.main.asyncio.sleep', new_callable=AsyncMock) # Patch asyncio.sleep
@patch('uplas_ai_agents.ttv_agent.main.avatar_service_client', spec=True) # Spec the actual client class
@patch('uplas_ai_agents.ttv_agent.main.httpx.AsyncClient') # Mock the httpx client used for inter-agent calls
@patch('uplas_ai_agents.ttv_agent.main.get_character_avatar_id_from_service')
@patch('uplas_ai_agents.ttv_agent.main.get_character_attire_id')
async def test_process_video_generation_task_successful_pipeline(
    mock_get_attire_id: MagicMock,
    mock_get_avatar_id: MagicMock,
    mock_httpx_client_constructor: MagicMock, # Constructor for httpx.AsyncClient
    mock_avatar_service_client_instance: MagicMock, # Instance of the mocked avatar_service_client
    mock_asyncio_sleep: AsyncMock,
    basic_ttv_request_payload_dict: Dict,
    mock_ai_tutor_response: Dict,
    mock_tts_agent_response: Dict,
    mock_avatar_submit_response: Dict,
    mock_avatar_poll_completed_response: Dict
):
    """Test the full (mocked) video generation pipeline successfully completes."""
    # Setup mock return values for character manager functions
    mock_get_avatar_id.return_value = "mock_susan_avatar_id_from_service"
    mock_get_attire_id.return_value = "mock_susan_attire_id_daily_prof"

    # Setup mock httpx client behavior
    mock_http_client = AsyncMock() # This is the instance of httpx.AsyncClient
    mock_httpx_client_constructor.return_value.__aenter__.return_value = mock_http_client # Handle async context manager

    # Configure responses for AI Tutor, TTS, and Django callback
    # Use a list of side_effects if post is called multiple times with different URLs
    ai_tutor_call_response = AsyncMock()
    ai_tutor_call_response.json.return_value = mock_ai_tutor_response
    ai_tutor_call_response.raise_for_status = MagicMock()

    tts_agent_call_response = AsyncMock()
    tts_agent_call_response.json.return_value = mock_tts_agent_response
    tts_agent_call_response.raise_for_status = MagicMock()
    
    django_callback_response = AsyncMock()
    django_callback_response.raise_for_status = MagicMock() # For successful callback

    # Order of calls: AI Tutor, TTS, Django callback
    mock_http_client.post.side_effect = [
        ai_tutor_call_response,
        tts_agent_call_response,
        django_callback_response
    ]

    # Setup mock avatar service client behavior
    mock_avatar_service_client_instance.submit_video_creation_job.return_value = mock_avatar_submit_response
    mock_avatar_service_client_instance.poll_video_job_status.return_value = mock_avatar_poll_completed_response
    
    # Patch asyncio.sleep to avoid actual sleeping during tests
    mock_asyncio_sleep.return_value = None


    # --- Execute the task ---
    job_id = f"ttvjob_test_pipeline_{uuid.uuid4()}"
    # Initialize job in the store as the endpoint would
    video_jobs[job_id] = {"status": VideoGenerationJobStatus.PENDING, "user_id": basic_ttv_request_payload_dict["user_id"]}
    
    request_model = GenerateVideoRequest(**basic_ttv_request_payload_dict) # Convert dict to Pydantic model

    # Import the task directly for testing
    from uplas_ai_agents.ttv_agent.main import process_video_generation_task
    await process_video_generation_task(job_id, request_model)

    # --- Assertions ---
    # Job store assertions
    assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.COMPLETED
    assert video_jobs[job_id]["video_url"] == mock_avatar_poll_completed_response["videoUrl"]
    assert video_jobs[job_id]["thumbnail_url"] == mock_avatar_poll_completed_response["thumbnailUrl"]
    assert video_jobs[job_id]["video_duration_seconds"] == mock_avatar_poll_completed_response["durationSeconds"]
    assert video_jobs[job_id]["error_message"] is None
    assert video_jobs[job_id]["script_generated_preview"] == mock_ai_tutor_response["answer_text"][:500]
    assert video_jobs[job_id]["character_used"] == request_model.instructor_character.value
    assert video_jobs[job_id]["attire_used"] == "mock_susan_attire_id_daily_prof" # From mock_get_attire_id

    # Assert calls to mocked CharacterManager functions
    mock_get_avatar_id.assert_called_once_with(request_model.instructor_character.value)
    mock_get_attire_id.assert_called_once_with(
        instructor_character_name=request_model.instructor_character.value,
        preferred_attire_name=request_model.preferred_attire_name, # Which is None in basic payload
        tags=["daily_professional"] # Default tags from main.py logic
    )

    # Assert calls to AI Tutor Agent
    ai_tutor_url = os.getenv("AI_TUTOR_AGENT_URL") + "/v1/ask-tutor"
    # Check the first call to http_client.post (AI Tutor)
    assert mock_http_client.post.call_args_list[0][0][0] == ai_tutor_url # URL
    ai_tutor_payload_sent = mock_http_client.post.call_args_list[0][1]['json'] # Payload
    assert ai_tutor_payload_sent["query_text"].startswith("Generate a 1-2 minute video script")
    assert ai_tutor_payload_sent["user_profile_snapshot"]["profession"] == request_model.user_profile_snapshot.profession

    # Assert calls to TTS Agent
    tts_agent_url = os.getenv("TTS_AGENT_URL") + "/v1/synthesize-speech"
    # Check the second call to http_client.post (TTS Agent)
    assert mock_http_client.post.call_args_list[1][0][0] == tts_agent_url
    tts_payload_sent = mock_http_client.post.call_args_list[1][1]['json']
    assert tts_payload_sent["text_to_speak"] == mock_ai_tutor_response["answer_text"]
    assert tts_payload_sent["voice_params"]["voice_character_name"] == "susan_pro_voice" # Mapped from SUSAN

    # Assert calls to Avatar Service Client
    mock_avatar_service_client_instance.submit_video_creation_job.assert_awaited_once_with(
        service_avatar_id="mock_susan_avatar_id_from_service",
        audio_file_gcs_url=mock_tts_agent_response["audio_url"],
        service_attire_id="mock_susan_attire_id_daily_prof",
        # language_code=request_model.language_code, # Check if your client sends this
        # background_settings=None,
        # output_webhook_url=None
    )
    mock_avatar_service_client_instance.poll_video_job_status.assert_awaited_once_with(
        mock_avatar_submit_response["service_job_id"]
    )
    
    # Assert asyncio.sleep was called (for polling loop)
    mock_asyncio_sleep.assert_awaited_once_with(ANY) # Check it was called at least once

    # Assert callback to Django
    django_callback_url = os.getenv("DJANGO_TTV_CALLBACK_URL")
    # Check the third call to http_client.post (Django callback)
    assert mock_http_client.post.call_args_list[2][0][0] == django_callback_url
    django_payload_sent = mock_http_client.post.call_args_list[2][1]['json']
    assert django_payload_sent["job_id"] == job_id
    assert django_payload_sent["status"] == VideoGenerationJobStatus.COMPLETED.value
    assert django_payload_sent["video_url"] == mock_avatar_poll_completed_response["videoUrl"]


@patch('uplas_ai_agents.ttv_agent.main.httpx.AsyncClient')
async def test_process_video_generation_task_ai_tutor_failure(
    mock_httpx_client_constructor: MagicMock,
    basic_ttv_request_payload_dict: Dict
):
    """Test pipeline failure at the script generation (AI Tutor) stage."""
    mock_http_client = AsyncMock()
    mock_httpx_client_constructor.return_value.__aenter__.return_value = mock_http_client
    # Simulate AI Tutor call failure
    mock_http_client.post.side_effect = httpx.HTTPStatusError("AI Tutor Error", request=MagicMock(), response=MagicMock(status_code=500))

    job_id = f"ttvjob_test_tutor_fail_{uuid.uuid4()}"
    video_jobs[job_id] = {"status": VideoGenerationJobStatus.PENDING, "user_id": basic_ttv_request_payload_dict["user_id"]}
    request_model = GenerateVideoRequest(**basic_ttv_request_payload_dict)

    from uplas_ai_agents.ttv_agent.main import process_video_generation_task
    await process_video_generation_task(job_id, request_model)

    assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.FAILED
    assert "AI Tutor Error" in video_jobs[job_id]["error_message"]
    
    # Ensure Django callback was still made with failure status
    # The last call to post (index 1 if AI Tutor was first and failed, then Django)
    # If AI Tutor fails, it should go straight to Django callback. So only 2 calls.
    assert mock_http_client.post.call_count == 2 # AI Tutor (failed) + Django Callback
    django_payload_sent = mock_http_client.post.call_args_list[1][1]['json']
    assert django_payload_sent["status"] == VideoGenerationJobStatus.FAILED.value
    assert "AI Tutor Error" in django_payload_sent["error_message"]

# Add more tests for failures at TTS stage, Avatar submission, Avatar polling, Django callback failure etc.

def test_health_check_endpoint_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_check_endpoint_unhealthy_missing_config():
    original_tutor_url = os.environ.pop("AI_TUTOR_AGENT_URL", None) # Remove a critical config
    response = client.get("/health")
    assert response.status_code == 200 # Health check itself doesn't fail with 500
    assert response.json()["status"] == "unhealthy"
    assert "Dependent agent URLs not configured" in response.json()["reason"]
    if original_tutor_url: os.environ["AI_TUTOR_AGENT_URL"] = original_tutor_url # Restore

# To run these tests:
# From within uplas-ai-agents/ttv_agent/
# pytest
# (Ensure pytest-asyncio is installed if not already a dependency of FastAPI's test utils)
