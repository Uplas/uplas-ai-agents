import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, ANY # ANY for some arguments in assertions
import uuid # For job_id checks

# Import the FastAPI app and Pydantic models from main.py
from .main import (
    app,
    GenerateVideoRequest,
    ContentSource,
    InstructorCharacterEnum,
    VideoGenerationJobStatus,
    video_jobs # Import the global job store for inspection/manipulation in tests
)

client = TestClient(app)

# --- Fixtures ---
@pytest.fixture
def basic_ttv_request_payload() -> GenerateVideoRequest:
    return GenerateVideoRequest(
        user_id="test_ttv_user_001",
        content_source=ContentSource(raw_text_content="Explain the concept of photosynthesis simply."),
        instructor_character=InstructorCharacterEnum.SUSAN,
        personalization_hints={"industry": "Education", "profession": "Student"},
        language_code="en-US"
    )

@pytest.fixture(autouse=True)
def clear_video_jobs_store():
    """Clears the in-memory video_jobs store before each test."""
    video_jobs.clear()
    yield # Test runs here
    video_jobs.clear()


# --- Test Cases for Endpoints ---

def test_generate_video_endpoint_success(basic_ttv_request_payload: GenerateVideoRequest):
    """Test the /v1/generate-video endpoint successfully queues a job."""
    payload_dict = basic_tts_request_payload.model_dump() # Corrected: should be basic_ttv_request_payload
    
    # Patch the background task so it doesn't run immediately during this specific test
    # We'll test process_video_generation_task separately
    with patch('uplas-ai-agents.ttv_agent.main.process_video_generation_task', new_callable=AsyncMock) as mock_process_task:
        response = client.post("/v1/generate-video", json=payload_dict)
        
        assert response.status_code == 202 # Accepted
        data = response.json()
        assert "job_id" in data
        assert data["status"] == VideoGenerationJobStatus.PENDING.value
        assert "Video generation task accepted" in data["message"]
        
        job_id = data["job_id"]
        assert job_id in video_jobs # Check if job was added to our mock store
        assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.PENDING
        assert video_jobs[job_id]["user_id"] == basic_ttv_request_payload.user_id

        # Check if background task was called with correct arguments
        mock_process_task.assert_called_once()
        # The first arg to add_task's target is job_id, then request_data
        called_job_id, called_request_data = mock_process_task.call_args[0]
        assert called_job_id == job_id
        assert called_request_data.user_id == basic_ttv_request_payload.user_id
        assert called_request_data.content_source.raw_text_content == basic_ttv_request_payload.content_source.raw_text_content


def test_generate_video_endpoint_invalid_payload_no_content(basic_ttv_request_payload: GenerateVideoRequest):
    """Test validation when no content source is provided."""
    payload_dict = basic_ttv_request_payload.model_dump()
    payload_dict["content_source"] = {} # Empty content_source
    
    response = client.post("/v1/generate-video", json=payload_dict)
    assert response.status_code == 422 # Unprocessable Entity by Pydantic
    assert "content_source" in response.text # Check if error message relates to content_source


def test_get_video_status_job_exists(basic_ttv_request_payload: GenerateVideoRequest):
    """Test retrieving status for an existing job."""
    # First, create a job
    with patch('uplas-ai-agents.ttv_agent.main.process_video_generation_task', new_callable=AsyncMock):
        initial_response = client.post("/v1/generate-video", json=basic_ttv_request_payload.model_dump())
    job_id = initial_response.json()["job_id"]

    # Update job status manually in our mock store for testing this endpoint
    video_jobs[job_id]["status"] = VideoGenerationJobStatus.PROCESSING_AUDIO
    video_jobs[job_id]["video_url"] = None # Still processing

    status_response = client.get(f"/v1/video-status/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == VideoGenerationJobStatus.PROCESSING_AUDIO.value
    assert data["video_url"] is None

def test_get_video_status_job_completed(basic_ttv_request_payload: GenerateVideoRequest):
    """Test retrieving status for a completed job."""
    with patch('uplas-ai-agents.ttv_agent.main.process_video_generation_task', new_callable=AsyncMock):
        initial_response = client.post("/v1/generate-video", json=basic_ttv_request_payload.model_dump())
    job_id = initial_response.json()["job_id"]

    mock_video_url = "https://storage.googleapis.com/mocked-uplas-ttv-videos/final_videos/video_susan_mock.mp4"
    video_jobs[job_id].update({
        "status": VideoGenerationJobStatus.COMPLETED,
        "video_url": mock_video_url,
        "thumbnail_url": "https://storage.googleapis.com/mocked-uplas-ttv-videos/final_videos/thumb_susan_mock.jpg",
        "video_duration_seconds": 185.5,
        "error_message": None
    })

    status_response = client.get(f"/v1/video-status/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["status"] == VideoGenerationJobStatus.COMPLETED.value
    assert data["video_url"] == mock_video_url
    assert data["video_duration_seconds"] == 185.5

def test_get_video_status_job_not_found():
    """Test retrieving status for a non-existent job ID."""
    non_existent_job_id = f"ttvjob_{uuid.uuid4()}"
    response = client.get(f"/v1/video-status/{non_existent_job_id}")
    assert response.status_code == 404
    assert "Job ID not found" in response.json()["detail"]


# --- Test Cases for the Background Task (process_video_generation_task) ---
# These are more like integration tests for the mocked pipeline.
# We need to use pytest-asyncio for async test functions.

# Import the task and mock clients directly for more granular testing
from .main import process_video_generation_task, mock_script_llm, mock_tts_video, mock_avatar_client

@pytest.mark.asyncio # Mark test as asynchronous
async def test_process_video_generation_task_successful_pipeline(basic_ttv_request_payload: GenerateVideoRequest):
    """Test the full (mocked) video generation pipeline successfully completes."""
    job_id = f"ttvjob_test_pipeline_{uuid.uuid4()}"
    video_jobs[job_id] = { # Initial job entry
        "status": VideoGenerationJobStatus.PENDING,
        "user_id": basic_ttv_request_payload.user_id
    }

    # Mock the external calls within the pipeline
    with patch.object(mock_script_llm, 'generate_script', new_callable=AsyncMock, return_value="[SCENE] Mocked Script Content") as mock_gen_script, \
         patch.object(mock_tts_video, 'generate_audio_from_script', new_callable=AsyncMock, return_value={"gcs_audio_url": "gs://mock/audio.mp3", "duration_seconds": 120.0}) as mock_gen_audio, \
         patch.object(mock_avatar_client, 'generate_video_from_audio_and_avatar', new_callable=AsyncMock, return_value={"gcs_video_url": "https://mock/video.mp4", "gcs_thumbnail_url": "https://mock/thumb.jpg"}) as mock_gen_avatar, \
         patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_django_callback: # Mock the final callback

        await process_video_generation_task(job_id, basic_ttv_request_payload)

        # Assertions on job store
        assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.COMPLETED
        assert video_jobs[job_id]["video_url"] == "https://mock/video.mp4"
        assert video_jobs[job_id]["thumbnail_url"] == "https://mock/thumb.jpg"
        assert video_jobs[job_id]["video_duration_seconds"] == 120.0
        assert video_jobs[job_id]["error_message"] is None

        # Assertions on mock calls
        mock_gen_script.assert_awaited_once()
        mock_gen_audio.assert_awaited_once_with(script="[SCENE] Mocked Script Content", instructor=ANY, lang=ANY)
        mock_gen_avatar.assert_awaited_once_with(audio_url="gs://mock/audio.mp3", instructor=ANY, instructor_attire=ANY)
        
        # Assert callback to Django
        mock_django_callback.assert_awaited_once()
        callback_args, callback_kwargs = mock_django_callback.call_args
        # callback_url = callback_args[0] # First positional argument is the URL
        callback_payload_json = callback_kwargs['json'] # Payload is passed as json kwarg
        
        assert callback_payload_json["job_id"] == job_id
        assert callback_payload_json["status"] == VideoGenerationJobStatus.COMPLETED.value
        assert callback_payload_json["video_url"] == "https://mock/video.mp4"


@pytest.mark.asyncio
async def test_process_video_generation_task_script_failure(basic_ttv_request_payload: GenerateVideoRequest):
    """Test pipeline failure at the script generation stage."""
    job_id = f"ttvjob_test_script_fail_{uuid.uuid4()}"
    video_jobs[job_id] = {"status": VideoGenerationJobStatus.PENDING, "user_id": basic_ttv_request_payload.user_id}

    with patch.object(mock_script_llm, 'generate_script', new_callable=AsyncMock, side_effect=Exception("LLM Script Error")) as mock_gen_script, \
         patch('httpx.AsyncClient.post', new_callable=AsyncMock) as mock_django_callback:

        await process_video_generation_task(job_id, basic_ttv_request_payload)

        assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.FAILED
        assert "LLM Script Error" in video_jobs[job_id]["error_message"]
        assert video_jobs[job_id]["video_url"] is None
        
        mock_gen_script.assert_awaited_once()
        mock_django_callback.assert_awaited_once() # Callback should still be made with failure status
        callback_payload_json = mock_django_callback.call_args.kwargs['json']
        assert callback_payload_json["status"] == VideoGenerationJobStatus.FAILED.value
        assert "LLM Script Error" in callback_payload_json["error_message"]

# Add more tests for failures at TTS stage, Avatar stage, Django callback failure etc.

# To run these tests (requires pytest and pytest-asyncio):
# From within uplas-ai-agents/ttv_agent/
# pytest
