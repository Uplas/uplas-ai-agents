# uplas-ai-agents/ttv_agent/test_main.py
import pytest
import pytest_asyncio # For async fixtures
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock, ANY, call
import os
import uuid
import httpx # For mocking its client
import json # For payloads
import re # For some string checks

# Import the FastAPI app and Pydantic models from our refined main.py
from .main import (
    app,
    GenerateVideoRequest,
    UserProfileSnapshotForTTV,
    ContentSource,
    InstructorChars,
    VideoGenerationJobStatus,
    VideoCallbackPayload, # For asserting callback structure
    TtsSynthesisInputType, # For verifying TTS call
    video_jobs, # Import the global job store for inspection/manipulation
    # Helper functions we'll test directly or want to assert their effects
    convert_script_to_ssml,
    extract_visual_aid_cues,
    determine_attire_tags,
    determine_background_settings
)
# Import from animation_logic as they are used by main.py
from .animation_logic.character_manager import (
    get_character_avatar_id_from_service,
    get_character_attire_id,
    CharacterConfigError
)
from .animation_logic.avatar_api_client import ThirdPartyAvatarAPIClient, AvatarJobError


# Set environment variables for testing (as in original test file)
#
os.environ["GCP_PROJECT_ID"] = "test-gcp-project-ttv"
os.environ["TTV_GCS_BUCKET_NAME"] = "test-ttv-bucket-innovate" # For potential video storage
os.environ["DJANGO_TTV_CALLBACK_URL"] = "http://mock-django-innovate/api/internal/ttv-callback/"
os.environ["AI_TUTOR_AGENT_URL"] = "http://mock-ai-tutor-agent-innovate"
os.environ["TTS_AGENT_URL"] = "http://mock-tts-agent-innovate"
os.environ["THIRD_PARTY_AVATAR_API_KEY"] = "mock_avatar_api_key_innovate"
os.environ["THIRD_PARTY_AVATAR_BASE_URL"] = "http://mock-avatar-service-innovate.com/api"

# Use TestClient for synchronous testing of async FastAPI app
client = TestClient(app)

# --- Fixtures (InnovateAI Refined & New) ---

@pytest.fixture
def basic_user_profile_ttv() -> UserProfileSnapshotForTTV: #
    return UserProfileSnapshotForTTV(
        industry="Creative Arts",
        profession="Animator",
        country="Testland", city="InnovateCity",
        career_interest="AI Storytelling",
        learning_goals="Create engaging educational videos.",
        preferred_tutor_persona="Enthusiastic and inspiring",
        learning_pace_preference="normal" # InnovateAI: Added
    )

@pytest.fixture
def basic_ttv_request_dict(basic_user_profile_ttv: UserProfileSnapshotForTTV) -> dict: #
    return {
        "user_id": "test_ttv_user_innovate_007",
        "content_source": {"raw_text_content": "Explain the principles of animation with <emphasis level='strong'>impact</emphasis> and a <pause strength='medium' />. <visual_aid_suggestion type='diagram_needed' description='12 principles chart'/>"},
        "instructor_character": InstructorChars.UNCLE_TREVOR.value,
        "user_profile_snapshot": basic_user_profile_ttv.model_dump(exclude_none=True),
        "language_code": "en-US",
        "preferred_video_length_minutes": "2-3", # Shorter for testing
        "background_theme_preference": "dynamic_abstract" # InnovateAI: Added
    }

@pytest.fixture(autouse=True) #
def clear_video_jobs_store_and_mock_avatar_client(monkeypatch):
    """Clears job store & ensures avatar_service_client is freshly mocked for each test."""
    video_jobs.clear()
    
    # Create a fresh mock instance for the avatar_service_client for each test
    mock_avatar_client_instance = AsyncMock(spec=ThirdPartyAvatarAPIClient)
    # Configure its methods as AsyncMock if they are async
    mock_avatar_client_instance.submit_video_creation_job = AsyncMock()
    mock_avatar_client_instance.poll_video_job_status = AsyncMock()
    
    # Patch the global instance in main.py that is used by process_video_generation_task
    monkeypatch.setattr("ttv_agent.main.avatar_service_client", mock_avatar_client_instance)
    
    yield mock_avatar_client_instance # Provide the mock to tests that need it directly
    
    video_jobs.clear()


# --- Mock Responses for External Services (as in original, with InnovateAI considerations) ---
@pytest.fixture #
def mock_ai_tutor_response_with_tags() -> Dict:
    # InnovateAI: Script from Tutor should contain semantic tags for SSML/Visuals
    return {"answer_text": "Hello Animator! Let's talk animation. <pause strength='long' /> Key is <emphasis level='strong'>timing</emphasis>. <visual_aid_suggestion type='animation_example' description='Bouncing ball with different timings'/>. Another is <topic>squash and stretch</topic>. <analogy type='real_world_physics'/> This is <difficulty type='foundational_info'>fundamental</difficulty>."}

@pytest.fixture #
def mock_tts_agent_response() -> Dict:
    return {
        "audio_url": "gs://test-uplas-tts-bucket-innovate/mock_audio_ssml.mp3",
        "audio_duration_seconds": 75.2,
        "voice_used_details": {"quality_tier_used": "standard"}, # InnovateAI: Added
        "text_character_count": 150 # For the SSML string
    }

@pytest.fixture #
def mock_avatar_submit_response() -> Dict:
    return {"service_job_id": f"avatar_svc_job_innovate_{uuid.uuid4()}", "initial_status": "queued"}

@pytest.fixture #
def mock_avatar_poll_completed_response() -> Dict:
    return {
        "status": "completed",
        "videoUrl": "https://mock-avatar-service-innovate.com/videos/innovate_video.mp4",
        "thumbnailUrl": "https://mock-avatar-service-innovate.com/videos/innovate_thumb.jpg",
        "durationSeconds": 73.0, "progressPercent": 100
    }

@pytest.fixture #
def mock_avatar_poll_failed_response() -> Dict:
    return {"status": "failed", "errorMessage": "InnovateAI Simulated avatar rendering failure.", "progressPercent": 40}

# --- Centralized Mocking for httpx.AsyncClient used for inter-agent calls ---
@pytest_asyncio.fixture
async def mock_httpx_client():
    """Mocks httpx.AsyncClient for AI Tutor, TTS, and Django calls."""
    with patch("ttv_agent.main.httpx.AsyncClient", new_callable=AsyncMock) as MockAsyncClient:
        # This mock_client_instance is what `async with httpx.AsyncClient() as client:` will yield.
        mock_client_instance = AsyncMock()
        MockAsyncClient.return_value.__aenter__.return_value = mock_client_instance
        yield mock_client_instance


# --- Test Cases for Helper Functions (InnovateAI New Tests) ---

def test_convert_script_to_ssml_basic_tags_and_pace():
    """InnovateAI Test: SSML conversion of custom tags and pace adjustment."""
    raw_script = "Intro. <emphasis level='strong'>Important</emphasis>. Then <pause strength='medium' />. <visual_aid_suggestion type='chart' description='A pie chart'/> Outro."
    ssml_normal_pace = convert_script_to_ssml(raw_script, "en-US", "normal")
    ssml_slow_pace = convert_script_to_ssml(raw_script, "en-US", "slow")

    assert "<speak>Intro. <emphasis level=\"strong\">Important</emphasis>. Then <break time=\"400ms\"/>.  Outro.</speak>" in ssml_normal_pace
    assert "<visual_aid_suggestion" not in ssml_normal_pace # Should be stripped
    
    assert "<break time=\"700ms\"/>" in ssml_slow_pace # Slow pace = longer break

def test_extract_visual_aid_cues_multiple():
    """InnovateAI Test: Extraction of visual aid suggestions."""
    script = "See <visual_aid_suggestion type='diagram' description='Flow A'/> and <visual_aid_suggestion type='image' description='Result B'/>."
    cues = extract_visual_aid_cues(script)
    assert len(cues) == 2
    assert {"type": "diagram", "description": "Flow A"} in cues
    assert {"type": "image", "description": "Result B"} in cues

def test_determine_attire_tags_from_instructions(basic_ttv_request_dict):
    """InnovateAI Test: Attire tag determination based on additional_instructions."""
    request_data_model = GenerateVideoRequest(**basic_ttv_request_dict)
    request_data_model.additional_instructions = "Make it a formal presentation."
    tags = determine_attire_tags(request_data_model)
    assert "formal" in tags
    assert "presentation" in tags

    request_data_model.additional_instructions = "A casual tutorial please."
    tags_casual = determine_attire_tags(request_data_model)
    assert "casual" in tags_casual
    assert "tutorial" in tags_casual

def test_determine_background_settings_from_preference(basic_ttv_request_dict):
    """InnovateAI Test: Background settings determination."""
    request_data_model = GenerateVideoRequest(**basic_ttv_request_dict) # Has dynamic_abstract
    settings = determine_background_settings(request_data_model, [])
    assert settings["type"] == "animated_loop_id" # Based on mock logic in main.py
    assert settings["id"] == "abstract_loop_blue_003"

    request_data_model.background_theme_preference = "color:#FF0000"
    settings_color = determine_background_settings(request_data_model, [])
    assert settings_color["type"] == "solid_color"
    assert settings_color["hex"] == "#FF0000"


# --- Test Cases for Endpoints (Existing structure, InnovateAI refined) ---

def test_generate_video_endpoint_success(basic_ttv_request_payload_dict: Dict): #
    """InnovateAI Test: /v1/generate-video successfully queues a job (background task mocked)."""
    with patch('ttv_agent.main.process_video_generation_task', new_callable=AsyncMock) as mock_process_task:
        response = client.post("/v1/generate-video", json=basic_ttv_request_payload_dict)

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        job_id = data["job_id"]
        assert data["status"] == VideoGenerationJobStatus.PENDING.value
        
        assert job_id in video_jobs
        mock_process_task.assert_awaited_once()
        # Further assertions on called_request_data_model as in original test


def test_get_video_status_job_completed(basic_ttv_request_payload_dict: Dict): # Based on
    """InnovateAI Test: Retrieving status for a completed job."""
    with patch('ttv_agent.main.process_video_generation_task', new_callable=AsyncMock): # Prevent actual run
        initial_response = client.post("/v1/generate-video", json=basic_ttv_request_payload_dict)
    job_id = initial_response.json()["job_id"]

    # Manually update job state for test
    video_jobs[job_id].update({
        "status": VideoGenerationJobStatus.COMPLETED,
        "video_url": "https://innovate.uplas.com/video.mp4",
        "attire_used": "innovate_formal_blue", "character_used": "UNCLE_TREVOR",
        "script_generated_preview": "InnovateAI script..."
    })
    response = client.get(f"/v1/video-status/{job_id}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == VideoGenerationJobStatus.COMPLETED.value
    assert data["video_url"] == "https://innovate.uplas.com/video.mp4"


def test_get_video_status_job_not_found(): #
    response = client.get(f"/v1/video-status/non_existent_job_innovate")
    assert response.status_code == status.HTTP_404_NOT_FOUND


# --- Test Cases for the Background Task `process_video_generation_task` (InnovateAI Enhanced) ---

@patch('ttv_agent.main.asyncio.sleep', new_callable=AsyncMock) # Patch asyncio.sleep
@patch('ttv_agent.main.get_character_avatar_id_from_service') #
@patch('ttv_agent.main.get_character_attire_id') #
async def test_process_video_task_successful_innovate_pipeline(
    mock_get_attire_id: MagicMock, # Patched CharacterManager functions
    mock_get_avatar_id: MagicMock,
    mock_asyncio_sleep: AsyncMock, # For polling
    mock_httpx_client: AsyncMock, # Centralized mock for httpx calls
    clear_video_jobs_store_and_mock_avatar_client: AsyncMock, # Provides mocked avatar_service_client
    basic_ttv_request_payload_dict: dict,
    mock_ai_tutor_response_with_tags: Dict, # Using script with tags
    mock_tts_agent_response: Dict,
    mock_avatar_submit_response: Dict,
    mock_avatar_poll_completed_response: Dict
): # Based on
    """InnovateAI Test: Full (mocked) TTV pipeline with SSML preprocessing and dynamic elements."""
    mock_avatar_service_client = clear_video_jobs_store_and_mock_avatar_client # Get the instance
    
    mock_get_avatar_id.return_value = "innovate_trevor_avatar_id"
    mock_get_attire_id.return_value = "innovate_trevor_attire_dynamic_abstract" # Should be selected

    ai_tutor_call_response = AsyncMock()
    ai_tutor_call_response.json.return_value = mock_ai_tutor_response_with_tags
    ai_tutor_call_response.raise_for_status = MagicMock()

    tts_agent_call_response = AsyncMock()
    tts_agent_call_response.json.return_value = mock_tts_agent_response
    tts_agent_call_response.raise_for_status = MagicMock()
    
    django_callback_response = AsyncMock()
    django_callback_response.raise_for_status = MagicMock()
    
    mock_httpx_client.post.side_effect = [ai_tutor_call_response, tts_agent_call_response, django_callback_response]

    mock_avatar_service_client.submit_video_creation_job.return_value = mock_avatar_submit_response
    mock_avatar_service_client.poll_video_job_status.return_value = mock_avatar_poll_completed_response
    mock_asyncio_sleep.return_value = None

    job_id = f"ttvjob_innovate_pipeline_{uuid.uuid4()}"
    video_jobs[job_id] = {"status": VideoGenerationJobStatus.PENDING, "user_id": basic_ttv_request_payload_dict["user_id"]}
    request_model = GenerateVideoRequest(**basic_ttv_request_payload_dict)

    from ttv_agent.main import process_video_generation_task # Import task
    await process_video_generation_task(job_id, request_model)

    assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.COMPLETED
    assert video_jobs[job_id]["video_url"] == mock_avatar_poll_completed_response["videoUrl"]
    
    # Assert AI Tutor call
    mock_httpx_client.post.assert_any_call(f"{os.getenv('AI_TUTOR_AGENT_URL')}/v1/ask-tutor", json=ANY)
    
    # Assert TTS Agent call - check for SSML input
    tts_call_payload = mock_httpx_client.post.call_args_list[1][1]['json'] # Second call is TTS
    assert tts_call_payload["input_type"] == TtsSynthesisInputType.SSML.value
    assert "<speak>" in tts_call_payload["content_to_synthesize"]
    assert "<emphasis level='strong'>timing</emphasis>" in tts_call_payload["content_to_synthesize"] # From mock script
    assert "<break time=\"700ms\"/>" in tts_call_payload["content_to_synthesize"] # Check for pace-adjusted pause (long = 700ms for normal pace)
    assert "<visual_aid_suggestion" not in tts_call_payload["content_to_synthesize"] # Should be stripped for TTS

    # Assert Avatar Service call - check background_settings
    mock_avatar_service_client.submit_video_creation_job.assert_awaited_once()
    avatar_call_kwargs = mock_avatar_service_client.submit_video_creation_job.call_args.kwargs
    assert avatar_call_kwargs["service_avatar_id"] == "innovate_trevor_avatar_id"
    assert avatar_call_kwargs["service_attire_id"] == "innovate_trevor_attire_dynamic_abstract"
    # Check that background_settings were passed (based on 'dynamic_abstract' preference)
    expected_bg_settings = {"type": "animated_loop_id", "id": "abstract_loop_blue_003"} # From main.py's mock logic
    assert avatar_call_kwargs["background_settings"] == expected_bg_settings
    
    # Assert visual cues were extracted and stored (conceptually)
    assert "visual_cues_identified" in video_jobs[job_id]
    assert len(video_jobs[job_id]["visual_cues_identified"]) == 1
    assert video_jobs[job_id]["visual_cues_identified"][0]["type"] == "animation_example"
    assert video_jobs[job_id]["visual_cues_identified"][0]["description"] == "Bouncing ball with different timings"

    mock_httpx_client.post.assert_any_call(os.getenv("DJANGO_TTV_CALLBACK_URL"), json=ANY) # Django callback


async def test_process_video_task_tts_failure(
    mock_httpx_client: AsyncMock,
    clear_video_jobs_store_and_mock_avatar_client: AsyncMock, # Ensure avatar client is mocked but not called
    basic_ttv_request_payload_dict: dict,
    mock_ai_tutor_response_with_tags: dict
):
    """InnovateAI Test: Pipeline failure at the TTS Agent stage."""
    ai_tutor_call_response = AsyncMock()
    ai_tutor_call_response.json.return_value = mock_ai_tutor_response_with_tags
    ai_tutor_call_response.raise_for_status = MagicMock()

    # Simulate TTS call failure
    mock_httpx_client.post.side_effect = [
        ai_tutor_call_response, # AI Tutor succeeds
        httpx.HTTPStatusError("TTS Agent Error", request=MagicMock(), response=MagicMock(status_code=500)), # TTS fails
        AsyncMock() # For Django callback
    ]

    job_id = f"ttvjob_tts_fail_{uuid.uuid4()}"
    video_jobs[job_id] = {"status": VideoGenerationJobStatus.PENDING, "user_id": basic_ttv_request_payload_dict["user_id"]}
    request_model = GenerateVideoRequest(**basic_ttv_request_payload_dict)

    from ttv_agent.main import process_video_generation_task
    await process_video_generation_task(job_id, request_model)

    assert video_jobs[job_id]["status"] == VideoGenerationJobStatus.FAILED
    assert "TTS Agent Error" in video_jobs[job_id]["error_message"]
    clear_video_jobs_store_and_mock_avatar_client.submit_video_creation_job.assert_not_awaited() # Avatar service should not be called
    
    # Ensure Django callback was made with failure status
    django_callback_call = mock_httpx_client.post.call_args_list[-1] # Last call should be Django
    assert os.getenv("DJANGO_TTV_CALLBACK_URL") in django_callback_call[0][0]
    django_payload_sent = django_callback_call[1]['json']
    assert django_payload_sent["status"] == VideoGenerationJobStatus.FAILED.value


def test_health_check_endpoint_healthy_innovate(monkeypatch): # Based on
    """InnovateAI Test: Health check ensuring all configs are present."""
    # All env vars are set globally for tests in this file
    # If avatar_service_client was initialized as mock due to missing keys, this would fail
    # So, ensure the mock client considers itself "configured" for this test.
    with patch('ttv_agent.main.avatar_service_client.is_mock_mode', False): # Simulate non-mock, fully configured
         response = client.get("/health")
         assert response.status_code == status.HTTP_200_OK
         assert response.json()["status"] == "healthy"
         assert response.json()["service"] == "TTV_Agent_InnovateAI" # Check refined name

# (Keep existing test_health_check_endpoint_unhealthy_missing_config, adapt if needed)
