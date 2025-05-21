import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, ANY # ANY for matching some arguments
import uuid
import time

from .main import (
    app,
    TTVGenerationRequest,
    UserProfileSnapshotForTTV,
    InstructorCharacter,
    TTVGenerationJob,
    JOB_STORE, # To inspect directly for testing background tasks
    # Import mock clients if we need to assert calls on them specifically
    mock_script_client,
    mock_tts_video_client,
    mock_avatar_client
)

client = TestClient(app)

# --- Fixtures ---
@pytest.fixture
def basic_user_profile_ttv() -> UserProfileSnapshotForTTV:
    return UserProfileSnapshotForTTV(
        industry="Creative Arts",
        profession="Animator",
        country="Kenya",
        preferred_ttv_instructor=InstructorCharacter.SUSAN.value 
    )

@pytest.fixture
def basic_ttv_request_payload(basic_user_profile_ttv: UserProfileSnapshotForTTV) -> TTVGenerationRequest:
    return TTVGenerationRequest(
        user_id="ttv_user_001",
        topic_id="topic_animation_basics",
        instructor_character=InstructorCharacter.SUSAN,
        user_profile_snapshot=basic_user_profile_ttv
    )

@pytest.fixture(autouse=True)
def clear_job_store_and_reset_mocks():
    """Clears the job store before each test and resets mocks."""
    JOB_STORE.clear()
    # If mock clients have state or call counts, reset them here.
    # For MagicMocks, you might use mock_name.reset_mock()
    # For our simple class mocks, they are stateless or re-initialized per test effectively.
    yield # Test runs here

# --- Test Cases ---

def test_request_video_generation_success(basic_ttv_request_payload: TTVGenerationRequest):
    """Test the /v1/generate-video endpoint successfully accepts a request."""
    # Mock the background task execution for this specific test of the endpoint
    with patch("uplas-ai-agents.ttv_agent.main.process_video_generation_task", new_callable=AsyncMock) as mock_process_task:
        payload_dict = basic_ttv_request_payload.model_dump()
        response = client.post("/v1/generate-video", json=payload_dict)

        assert response.status_code == 202 # Accepted
        data = response.json()
        assert "job_id" in data
        job_id = data["job_id"]
        assert data["status"] == "pending"
        assert data["uplas_topic_id"] == basic_ttv_request_payload.topic_id
        assert data["instructor_character"] == basic_ttv_request_payload.instructor_character.value

        # Check if job is in our mock store
        assert job_id in JOB_STORE
        assert JOB_STORE[job_id].status == "pending"

        # Check if background task was called
        mock_process_task.assert_called_once_with(job_id)


def test_get_job_status_found(basic_ttv_request_payload: TTVGenerationRequest):
    """Test retrieving status for an existing job."""
    # First, create a job by calling the generation endpoint
    with patch("uplas-ai-agents.ttv_agent.main.process_video_generation_task", new_callable=AsyncMock):
        initial_response = client.post("/v1/generate-video", json=basic_ttv_request_payload.model_dump())
        job_id = initial_response.json()["job_id"]

    # Now get its status
    status_response = client.get(f"/v1/jobs/{job_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["job_id"] == job_id
    assert data["status"] == "pending" # Background task is mocked, so it won't have progressed

def test_get_job_status_not_found():
    """Test retrieving status for a non-existent job."""
    non_existent_job_id = str(uuid.uuid4())
    response = client.get(f"/v1/jobs/{non_existent_job_id}")
    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]

# --- Testing the background task orchestration ---
# We need to test process_video_generation_task more directly.
# Since it's an async function called by background_tasks, we can call it with await in an async test.
# Or, we can patch its dependencies (LLM, TTS, Avatar clients, Django fetch/notify)

@pytest.mark.asyncio # Requires pytest-asyncio
async def test_process_video_generation_task_full_success_flow(basic_ttv_request_payload: TTVGenerationRequest):
    """
    Test the e2e logic of the background task 'process_video_generation_task'
    by mocking its dependent services.
    """
    from .main import process_video_generation_task, TTVGenerationJob # Import here for test scope

    # 1. Create the initial job and put it in the store
    job = TTVGenerationJob(
        uplas_user_id=basic_ttv_request_payload.user_id,
        uplas_topic_id=basic_ttv_request_payload.topic_id,
        instructor_character=basic_ttv_request_payload.instructor_character,
        user_profile_snapshot=basic_ttv_request_payload.user_profile_snapshot
    )
    JOB_STORE[job.job_id] = job

    # 2. Mock all external dependencies of process_video_generation_task
    with patch("uplas-ai-agents.ttv_agent.main.fetch_topic_details_from_django", new_callable=AsyncMock) as mock_fetch_topic, \
         patch.object(mock_script_client, "generate_video_script", new_callable=AsyncMock) as mock_gen_script, \
         patch.object(mock_tts_video_client, "synthesize_segments_for_video", new_callable=AsyncMock) as mock_gen_audio, \
         patch.object(mock_avatar_client, "render_video_from_audio_segments", new_callable=AsyncMock) as mock_render_video, \
         patch("uplas-ai-agents.ttv_agent.main.notify_django_of_video_status", new_callable=AsyncMock) as mock_notify_django:

        # Define return values for mocks
        mock_fetch_topic.return_value = {"id": job.uplas_topic_id, "title": "Mocked Topic Title", "content_summary": "Summary for TTV."}
        
        mock_script_return = MagicMock() # Simulate GeneratedVideoScript object
        mock_script_return.title = "Mocked Script Title"
        mock_script_return.segments = [MagicMock(dialogue="Segment 1 dialogue", estimated_duration_seconds=30)] # List of GeneratedScriptSegment mocks
        mock_script_return.total_estimated_duration_seconds = 30
        mock_gen_script.return_value = mock_script_return
        
        mock_gen_audio.return_value = {0: "gcs://mock-audio/seg0.mp3"}
        mock_render_video.return_value = "gcs://mock-final-video/final.mp4"

        # 3. Execute the task
        await process_video_generation_task(job.job_id)

        # 4. Assertions
        # Check final job status and URL
        updated_job = JOB_STORE[job.job_id]
        assert updated_job.status == "completed"
        assert updated_job.video_url == "gcs://mock-final-video/final.mp4"
        assert updated_job.error_message is None

        # Check if dependencies were called
        mock_fetch_topic.assert_called_once_with(job.uplas_topic_id)
        mock_gen_script.assert_called_once()
        mock_gen_audio.assert_called_once()
        mock_render_video.assert_called_once()
        
        # Check Django notifications
        # It's called multiple times for status updates + final
        assert mock_notify_django.call_count >= 4 # fetching_details, script_generating, audio_generating, video_rendering, completed/failed
        
        # Check the final notification payload
        final_notify_call_args = mock_notify_django.call_args_list[-1][0][0] # Get the job object from the last call
        assert final_notify_call_args.status == "completed"
        assert final_notify_call_args.video_url == "gcs://mock-final-video/final.mp4"

@pytest.mark.asyncio
async def test_process_video_generation_task_script_failure(basic_ttv_request_payload: TTVGenerationRequest):
    """Test background task when script generation fails."""
    from .main import process_video_generation_task, TTVGenerationJob

    job = TTVGenerationJob(
        uplas_user_id=basic_ttv_request_payload.user_id,
        uplas_topic_id=basic_ttv_request_payload.topic_id,
        instructor_character=basic_ttv_request_payload.instructor_character,
        user_profile_snapshot=basic_ttv_request_payload.user_profile_snapshot
    )
    JOB_STORE[job.job_id] = job

    with patch("uplas-ai-agents.ttv_agent.main.fetch_topic_details_from_django", new_callable=AsyncMock, return_value={"title": "T", "content_summary": "S"}), \
         patch.object(mock_script_client, "generate_video_script", new_callable=AsyncMock, side_effect=Exception("LLM script error")), \
         patch("uplas-ai-agents.ttv_agent.main.notify_django_of_video_status", new_callable=AsyncMock) as mock_notify_django:

        await process_video_generation_task(job.job_id)

        updated_job = JOB_STORE[job.job_id]
        assert updated_job.status == "failed"
        assert "LLM script error" in updated_job.error_message
        assert updated_job.video_url is None
        
        final_notify_call_args = mock_notify_django.call_args_list[-1][0][0]
        assert final_notify_call_args.status == "failed"

# To run these tests (requires pytest and pytest-asyncio):
# From within uplas-ai-agents/ttv_agent/
# pytest
