# uplas-ai-agents/nlp_content_agent/test_main.py
import pytest
import pytest_asyncio # For async fixtures
from httpx import AsyncClient # For testing FastAPI endpoints
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock # For mocking
import json
import os

# Import the FastAPI app and Pydantic models from your nlp_content_agent
# Assuming your main.py is in the same directory or test discovery handles it.
# If running tests from root, you might need to adjust path or use conftest.py
try:
    from .main import app, ProcessContentRequest, ProcessedModule, NlpLesson, NlpTopic, VertexAILLMClientForNLP
except ImportError: # Fallback for different execution contexts (e.g. running pytest from root)
    from nlp_content_agent.main import app, ProcessContentRequest, ProcessedModule, NlpLesson, NlpTopic, VertexAILLMClientForNLP


# --- Fixtures ---

@pytest_asyncio.fixture
async def async_test_client():
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_nlp_llm_client():
    """Fixture to mock the VertexAILLMClientForNLP instance and its methods."""
    with patch('nlp_content_agent.main.nlp_llm_client', autospec=True) as mock_client_instance:
        # Configure the methods of the mock client instance to be AsyncMock
        mock_client_instance.macro_segment_module = AsyncMock()
        mock_client_instance.micro_segment_and_enrich_lesson = AsyncMock()
        # Mock the internal _call_gemini_api if you were testing the client's methods directly
        # For endpoint tests, mocking the higher-level methods is usually sufficient.
        yield mock_client_instance

@pytest.fixture(autouse=True)
def set_mock_env_vars(monkeypatch):
    """Set essential environment variables for tests."""
    monkeypatch.setenv("GCP_PROJECT_ID", "test-gcp-project-id")
    monkeypatch.setenv("NLP_LLM_MODEL_NAME", "gemini-1.5-flash-test")
    # Ensure Vertex AI init in client doesn't fail if it relies on ADC or specific files
    # by potentially patching aiplatform.init if it's called globally in main.py
    # For now, assuming the client init within main might be robust enough or patched.
    with patch('google.cloud.aiplatform.init', return_value=None) as _: # Patch to prevent actual SDK init
        yield


# --- Test Cases for /v1/process-course-content Endpoint ---

@pytest.mark.asyncio
async def test_process_content_success(async_test_client: AsyncClient, mock_nlp_llm_client: MagicMock):
    """InnovateAI Test: Successful content processing with valid input and mocked LLM responses."""
    # Mock responses from the LLM client methods
    mock_nlp_llm_client.macro_segment_module.return_value = [
        {"lesson_title": "Lesson 1: Mock Intro", "text_segment_start_index": 0, "text_segment_end_index": 50},
        {"lesson_title": "Lesson 2: Mock Details", "text_segment_start_index": 50, "text_segment_end_index": 100}
    ]
    mock_nlp_llm_client.micro_segment_and_enrich_lesson.side_effect = [
        { # For "Lesson 1"
            "lesson_title": "Lesson 1: Mock Intro (Enriched)",
            "lesson_summary": "This is a mock intro summary.",
            "topics": [{
                "topic_id": "t1_intro", "topic_title": "Intro Topic", "key_concepts": ["KC1"],
                "content_with_tags": "Content for intro <difficulty type=\"foundational_info\"/>",
                "estimated_complexity_score": 0.2
            }]
        },
        { # For "Lesson 2"
            "lesson_title": "Lesson 2: Mock Details (Enriched)",
            "lesson_summary": "This is a mock details summary.",
            "topics": [{
                "topic_id": "t2_details", "topic_title": "Details Topic", "key_concepts": ["KC2", "KC3"],
                "content_with_tags": "Content for details <analogy type=\"general_analogy_needed\"/>",
                "estimated_complexity_score": 0.5, "suggested_prerequisites": ["Intro Topic"]
            }]
        }
    ]

    request_payload = ProcessContentRequest(
        module_id="test_module_001_raw",
        raw_text_content="This is the first part of the test content. This is the second part of the test content for our module.",
        language_code="en-US",
        module_title="Test Module InnovateAI"
    ).model_dump()

    response = await async_test_client.post("/v1/process-course-content", json=request_payload)

    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    
    # Validate using the Pydantic model
    processed_module = ProcessedModule(**response_data)
    assert processed_module.source_module_id == "test_module_001_raw"
    assert processed_module.module_title == "Test Module InnovateAI"
    assert len(processed_module.lessons) == 2
    assert processed_module.lessons[0].lesson_title == "Lesson 1: Mock Intro (Enriched)"
    assert processed_module.lessons[0].lesson_summary == "This is a mock intro summary."
    assert len(processed_module.lessons[0].topics) == 1
    assert processed_module.lessons[0].topics[0].topic_title == "Intro Topic"
    assert "KC1" in processed_module.lessons[0].topics[0].key_concepts
    assert "<difficulty type=\"foundational_info\"/>" in processed_module.lessons[0].topics[0].content_with_tags

    assert processed_module.lessons[1].lesson_title == "Lesson 2: Mock Details (Enriched)"
    assert len(processed_module.lessons[1].topics) == 1
    assert "<analogy type=\"general_analogy_needed\"/>" in processed_module.lessons[1].topics[0].content_with_tags
    assert "Intro Topic" in processed_module.lessons[1].topics[0].suggested_prerequisites

    # Check that client methods were called as expected
    mock_nlp_llm_client.macro_segment_module.assert_awaited_once()
    assert mock_nlp_llm_client.micro_segment_and_enrich_lesson.await_count == 2


@pytest.mark.asyncio
async def test_process_content_invalid_input_short_text(async_test_client: AsyncClient):
    """InnovateAI Test: Request with raw_text_content shorter than min_length."""
    request_payload = {
        "module_id": "test_short_text_module",
        "raw_text_content": "Too short.", # Default min_length is 50 in refined main.py
        "language_code": "en-US",
        "module_title": "Short Text Test"
    }
    response = await async_test_client.post("/v1/process-course-content", json=request_payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_process_content_unsupported_language_fallback(async_test_client: AsyncClient, mock_nlp_llm_client: MagicMock):
    """InnovateAI Test: Request with unsupported language, should fallback to default and process."""
    # Mock responses as in success test, but check language propagation
    mock_nlp_llm_client.macro_segment_module.return_value = [
        {"lesson_title": "Lesson Default Lang", "text_segment_start_index": 0, "text_segment_end_index": 50}
    ]
    mock_nlp_llm_client.micro_segment_and_enrich_lesson.return_value = {
        "lesson_title": "Lesson Default Lang (Enriched)", "lesson_summary": "Summary in default.",
        "topics": [{"topic_id": "t_def", "topic_title": "Topic Default", "key_concepts": [], "content_with_tags": "Default content"}]
    }

    request_payload = ProcessContentRequest(
        module_id="test_unsupported_lang_module",
        raw_text_content="This is some test content that is long enough for processing this time.",
        language_code="xx-XX", # Unsupported language
        module_title="Unsupported Language Test"
    ).model_dump()

    response = await async_test_client.post("/v1/process-course-content", json=request_payload)
    
    assert response.status_code == status.HTTP_200_OK # Should process with default
    response_data = response.json()
    assert response_data["language_code"] == "en-US" # Check if it fell back to default

    # Verify macro_segment_module was called with the default language
    mock_nlp_llm_client.macro_segment_module.assert_awaited_with(
        request_payload["raw_text_content"],
        "en-US", # Expected default language
        request_payload["module_title"]
    )
    # Verify micro_segment_and_enrich_lesson was called with the default language
    mock_nlp_llm_client.micro_segment_and_enrich_lesson.assert_awaited_with(
        request_payload["raw_text_content"][:50], # Assuming the mock segment indices
        "Lesson Default Lang",
        "en-US" # Expected default language
    )

@pytest.mark.asyncio
async def test_process_content_llm_macro_segment_fails(async_test_client: AsyncClient, mock_nlp_llm_client: MagicMock):
    """InnovateAI Test: LLM call for macro-segmentation fails."""
    mock_nlp_llm_client.macro_segment_module.side_effect = ValueError("Mocked LLM API Error for macro_segment")

    request_payload = ProcessContentRequest(
        module_id="test_macro_fail_module",
        raw_text_content="Valid content for a module that will experience a failure.",
        language_code="en-US",
        module_title="Macro Fail Test"
    ).model_dump()

    response = await async_test_client.post("/v1/process-course-content", json=request_payload)
    
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Error processing content: Mocked LLM API Error for macro_segment" in response.json()["detail"]


@pytest.mark.asyncio
async def test_process_content_llm_micro_segment_fails(async_test_client: AsyncClient, mock_nlp_llm_client: MagicMock):
    """InnovateAI Test: LLM call for micro-segmentation fails for one of the segments."""
    mock_nlp_llm_client.macro_segment_module.return_value = [
        {"lesson_title": "Lesson OK", "text_segment_start_index": 0, "text_segment_end_index": 20}
    ]
    mock_nlp_llm_client.micro_segment_and_enrich_lesson.side_effect = ValueError("Mocked LLM API Error for micro_segment")

    request_payload = ProcessContentRequest(
        module_id="test_micro_fail_module",
        raw_text_content="This content is perfectly fine for processing.",
        language_code="en-US",
        module_title="Micro Fail Test"
    ).model_dump()

    response = await async_test_client.post("/v1/process-course-content", json=request_payload)
    
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Error processing content: Mocked LLM API Error for micro_segment" in response.json()["detail"]


@pytest.mark.asyncio
async def test_process_content_empty_lessons_from_macro(async_test_client: AsyncClient, mock_nlp_llm_client: MagicMock):
    """InnovateAI Test: Macro segmentation returns no lessons."""
    mock_nlp_llm_client.macro_segment_module.return_value = [] # Empty list of lessons

    request_payload = ProcessContentRequest(
        module_id="test_no_lessons_module",
        raw_text_content="Content that somehow results in no lessons being segmented.",
        language_code="en-US",
        module_title="No Lessons Test"
    ).model_dump()

    response = await async_test_client.post("/v1/process-course-content", json=request_payload)
    
    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    processed_module = ProcessedModule(**response_data)
    assert len(processed_module.lessons) == 0
    # Ensure micro_segment_and_enrich_lesson was not called
    mock_nlp_llm_client.micro_segment_and_enrich_lesson.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_content_skip_invalid_segment_from_macro(async_test_client: AsyncClient, mock_nlp_llm_client: MagicMock):
    """InnovateAI Test: Macro segmentation returns an invalid segment (e.g., bad indices)."""
    mock_nlp_llm_client.macro_segment_module.return_value = [
        {"lesson_title": "Valid Lesson", "text_segment_start_index": 0, "text_segment_end_index": 10},
        {"lesson_title": "Invalid Segment - Bad Indices", "text_segment_start_index": 20, "text_segment_end_index": 15}, # Invalid
        {"lesson_title": "Another Valid Lesson", "text_segment_start_index": 10, "text_segment_end_index": 20}
    ]
    # Mock micro_segment to return valid data for the valid segments
    mock_nlp_llm_client.micro_segment_and_enrich_lesson.side_effect = [
        { "lesson_title": "Valid Lesson (Enriched)", "lesson_summary": "s1", "topics": [{"topic_id":"t1", "topic_title": "T1", "key_concepts": [], "content_with_tags": "c1"}]},
        { "lesson_title": "Another Valid Lesson (Enriched)", "lesson_summary": "s2", "topics": [{"topic_id":"t2", "topic_title": "T2", "key_concepts": [], "content_with_tags": "c2"}]}
    ]

    request_payload = ProcessContentRequest(
        module_id="test_invalid_segment_module",
        raw_text_content="First part. Second part for skipping.",
        language_code="en-US",
        module_title="Invalid Segment Test"
    ).model_dump()

    response = await async_test_client.post("/v1/process-course-content", json=request_payload)
    assert response.status_code == status.HTTP_200_OK
    response_data = response.json()
    processed_module = ProcessedModule(**response_data)
    
    assert len(processed_module.lessons) == 2 # Only two valid lessons should be processed
    assert processed_module.lessons[0].lesson_title == "Valid Lesson (Enriched)"
    assert processed_module.lessons[1].lesson_title == "Another Valid Lesson (Enriched)"
    # micro_segment_and_enrich_lesson should have been called only twice
    assert mock_nlp_llm_client.micro_segment_and__enrich_lesson.await_count == 2


# --- Test Cases for VertexAILLMClientForNLP Methods (Illustrative - more detailed mocking needed) ---
# These tests would typically patch the `_call_gemini_api` method of the client.

@pytest.mark.asyncio
async def test_nlp_client_macro_segment_constructs_prompt_correctly():
    """InnovateAI Test: Verify the system prompt construction for macro_segment_module."""
    client = VertexAILLMClientForNLP(model_name="test-model")
    
    # Patch the internal _call_gemini_api method for this specific test
    with patch.object(client, '_call_gemini_api', new_callable=AsyncMock) as mock_api_call:
        mock_api_call.return_value = json.dumps([{"lesson_title": "Test", "text_segment_start_index": 0, "text_segment_end_index": 10}]) # Valid JSON
        
        await client.macro_segment_module("Test content", "en-US", "Test Module Title")
        
        mock_api_call.assert_awaited_once()
        args, kwargs = mock_api_call.call_args
        system_prompt_sent = args[0] # First positional argument is system_prompt
        
        assert "You are an expert instructional designer" in system_prompt_sent
        assert "segment it into distinct, coherent, high-level lessons" in system_prompt_sent
        assert "language: [en-US]" in system_prompt_sent # Ensure language code is in prompt
        assert "module is titled: 'Test Module Title'" in system_prompt_sent
        assert "Respond ONLY with a valid JSON list of objects" in system_prompt_sent


@pytest.mark.asyncio
async def test_nlp_client_micro_segment_constructs_prompt_correctly():
    """InnovateAI Test: Verify system prompt for micro_segment_and_enrich_lesson."""
    client = VertexAILLMClientForNLP(model_name="test-model")
    
    with patch.object(client, '_call_gemini_api', new_callable=AsyncMock) as mock_api_call:
        mock_api_call.return_value = json.dumps({
            "lesson_title": "Original Lesson Title", "lesson_summary": "Summary.",
            "topics": [{"topic_id": "t1", "topic_title": "TT1", "key_concepts": [], "content_with_tags": "Content"}]
        })
        
        await client.micro_segment_and_enrich_lesson("Lesson text here.", "Original Lesson Title", "fr-FR")
        
        mock_api_call.assert_awaited_once()
        args, kwargs = mock_api_call.call_args
        system_prompt_sent = args[0]
        
        assert "You are an AI pedagogical content specialist" in system_prompt_sent
        assert "lesson titled 'Original Lesson Title'" in system_prompt_sent
        assert "language: [fr-FR]" in system_prompt_sent
        assert "Generate a concise 'lesson_summary'" in system_prompt_sent
        assert "<analogy type=" in system_prompt_sent # Check for enrichment tags instructions
        assert "<difficulty type=" in system_prompt_sent
        assert "Respond ONLY with a single, valid JSON object" in system_prompt_sent


@pytest.mark.asyncio
async def test_nlp_client_macro_segment_handles_invalid_llm_json(monkeypatch):
    """InnovateAI Test: macro_segment_module handles invalid JSON from LLM."""
    client = VertexAILLMClientForNLP(model_name="test-model")
    
    with patch.object(client, '_call_gemini_api', new_callable=AsyncMock) as mock_api_call:
        mock_api_call.return_value = "This is not JSON" # Invalid JSON
        
        with pytest.raises(ValueError) as excinfo:
            await client.macro_segment_module("Test content", "en-US", "Test Title")
        assert "Failed to parse LLM response for macro-segmentation" in str(excinfo.value)

# --- Health Check Tests ---

@pytest.mark.asyncio
async def test_health_check_healthy(async_test_client: AsyncClient):
    """InnovateAI Test: Health check returns healthy status when configured."""
    # Environment variables are set by set_mock_env_vars fixture
    response = await async_test_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "status": "healthy",
        "service": "NLP_Content_Agent_InnovateAI",
        "innovate_ai_enhancements_active": True
    }

@pytest.mark.asyncio
async def test_health_check_unhealthy_missing_gcp_project(async_test_client: AsyncClient, monkeypatch):
    """InnovateAI Test: Health check returns unhealthy if GCP_PROJECT_ID is missing."""
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    # We need to re-initialize the app or client for this to take effect if checked at startup
    # For simplicity, assuming health check re-evaluates or main.py logic handles it
    # This test might require more intricate setup depending on how config is loaded and checked by health endpoint
    
    # To properly test this, we'd ideally reload the app or directly test the health_check function
    # with the environment variable modified. FastAPI TestClient uses the app instance as is.
    # A simple way for this test structure:
    with patch('nlp_content_agent.main.GCP_PROJECT_ID', None): # Temporarily override module-level var
        response = await async_test_client.get("/health")
        assert response.status_code == status.HTTP_200_OK # Health check itself might be 200
        assert response.json()["status"] == "unhealthy"
        assert "GCP_PROJECT_ID" in response.json()["reason"]
