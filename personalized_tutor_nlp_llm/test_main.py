# uplas-ai-agents/personalized_tutor_nlp_llm/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, ANY
import os

# Import the FastAPI app and Pydantic models from main.py
# Assuming main.py is in the same directory or Python path is configured
from .main import (
    app,
    AiTutorQueryRequest,
    UserProfileSnapshot,
    TutorRequestContext,
    ConversationTurn,
    ConversationRole,
    SUPPORTED_LANGUAGES, # Import for testing language validation
    DEFAULT_LANGUAGE
    # We will patch llm_client directly where it's used in the endpoint
)

# Set environment variables for testing if not already set globally for tests
# These are needed because main.py might try to init VertexAI platform
os.environ["GCP_PROJECT_ID"] = "test-gcp-project-id"
os.environ["GCP_LOCATION"] = "test-gcp-location"
os.environ["LLM_MODEL_NAME"] = "test-gemini-model"

client = TestClient(app) # TestClient for making HTTP requests to the FastAPI app

# --- Fixtures ---
@pytest.fixture
def basic_user_profile() -> UserProfileSnapshot:
    return UserProfileSnapshot(
        industry="Education",
        profession="Teacher",
        country="Kenya",
        city="Nairobi",
        preferred_tutor_persona="Supportive and clear"
    )

@pytest.fixture
def basic_tutor_request_payload(basic_user_profile: UserProfileSnapshot) -> Dict:
    return {
        "user_id": "test_user_001",
        "query_text": "What are Python functions?",
        "user_profile_snapshot": basic_user_profile.model_dump(),
        "language_code": "en-US"
    }

@pytest.fixture
def mock_llm_successful_response() -> Dict:
    return {
        "answer_text": "This is a detailed, personalized answer about Python functions, considering your Teacher profile in Education from Kenya.",
        "generated_analogies": [{"analogy": "Think of functions like lesson plans."}],
        "suggested_follow_up_questions": ["How do I define a function?", "What are arguments?"],
        "prompt_token_count": 150,
        "response_token_count": 200
    }

@pytest.fixture
def mock_llm_config_error_response() -> Dict:
    # This fixture isn't directly used as a return value for the mock,
    # but represents a scenario where the LLM client itself raises an EnvironmentError
    return {"detail": "AI service configuration error: GCP_PROJECT_ID is not configured."}


# --- Test Cases ---

@patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.llm_client.generate_response', new_callable=AsyncMock)
async def test_ask_tutor_endpoint_success_basic_query(
    mock_generate_llm_response: AsyncMock,
    basic_tutor_request_payload: Dict,
    mock_llm_successful_response: Dict
):
    """Test the main endpoint with a valid basic request and successful LLM call."""
    mock_generate_llm_response.return_value = mock_llm_successful_response

    response = client.post("/v1/ask-tutor", json=basic_tutor_request_payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer_text" in data
    assert "Python functions" in data["answer_text"] # Check if it's related
    assert "Teacher" in data["answer_text"] # Check for personalization cue
    assert data["suggested_follow_up_questions"] == mock_llm_successful_response["suggested_follow_up_questions"]
    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == os.getenv("LLM_MODEL_NAME")
    assert data["debug_info"]["language_used"] == basic_tutor_request_payload["language_code"]

    # Verify llm_client.generate_response was called
    mock_generate_llm_response.assert_awaited_once()
    call_args = mock_generate_llm_response.call_args[1] # Keyword arguments
    assert "system_prompt" in call_args
    assert "conversation_turns" in call_args
    assert len(call_args["conversation_turns"]) == 1 # Only the current user query
    assert call_args["conversation_turns"][0]["content"] == basic_tutor_request_payload["query_text"]
    assert basic_tutor_request_payload["language_code"] in call_args["system_prompt"]


@patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.llm_client.generate_response', new_callable=AsyncMock)
async def test_ask_tutor_with_context_and_history(
    mock_generate_llm_response: AsyncMock,
    basic_user_profile: UserProfileSnapshot,
    mock_llm_successful_response: Dict
):
    """Test when context and conversation history are provided."""
    mock_generate_llm_response.return_value = mock_llm_successful_response

    payload = {
        "user_id": "test_user_002",
        "query_text": "Explain this more simply.",
        "user_profile_snapshot": basic_user_profile.model_dump(),
        "language_code": "fr-FR",
        "context": {
            "topic_id": "topic_python_lists_id",
            "current_topic_title": "Python Lists Fundamentals",
            "project_assessment_feedback": "User needs to focus on list comprehensions."
        },
        "conversation_history": [
            {"role": "user", "content": "What are lists?"},
            {"role": "assistant", "content": "Lists are ordered collections."}
        ]
    }
    # Mock the Django content fetcher
    with patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.fetch_course_content_from_django', new_callable=AsyncMock) as mock_fetch_content:
        mock_fetch_content.return_value = "Mocked course content about Python lists."
        
        response = client.post("/v1/ask-tutor", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["debug_info"]["language_used"] == "fr-FR"

    mock_fetch_content.assert_awaited_once_with("topic_python_lists_id", None, "fr-FR")
    
    mock_generate_llm_response.assert_awaited_once()
    call_args = mock_generate_llm_response.call_args[1]
    assert "fr-FR" in call_args["system_prompt"] # Check language in system prompt
    assert "Mocked course content about Python lists." in call_args["system_prompt"]
    assert "User needs to focus on list comprehensions." in call_args["system_prompt"] # Check assessment feedback in prompt
    assert len(call_args["conversation_turns"]) == 3 # 2 from history + 1 current query
    assert call_args["conversation_turns"][0]["content"] == "What are lists?"
    assert call_args["conversation_turns"][1]["content"] == "Lists are ordered collections."
    assert call_args["conversation_turns"][2]["content"] == "Explain this more simply."


def test_ask_tutor_invalid_language_code(basic_tutor_request_payload: Dict):
    """Test with an unsupported language code, expecting fallback to default."""
    payload = basic_tutor_request_payload.copy()
    payload["language_code"] = "xx-XX" # Invalid language

    # We don't need to mock the LLM call itself for this validation test,
    # but if it proceeds far enough to call it, we should ensure it uses the default.
    with patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.llm_client.generate_response', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = {"answer_text": "Fallback response"} # Dummy response
        response = client.post("/v1/ask-tutor", json=payload)

    assert response.status_code == 200 # The request itself is valid after Pydantic coercion
    data = response.json()
    assert data["debug_info"]["language_used"] == DEFAULT_LANGUAGE # Should have fallen back

    # Check that the system prompt passed to LLM (if called) used the default language
    mock_llm_call.assert_awaited_once()
    call_args = mock_llm_call.call_args[1]
    assert DEFAULT_LANGUAGE in call_args["system_prompt"]


def test_ask_tutor_invalid_request_missing_query(basic_user_profile: UserProfileSnapshot):
    """Test request validation for missing query_text."""
    invalid_payload = {
        "user_id": "test_user_003",
        "user_profile_snapshot": basic_user_profile.model_dump()
        # query_text is missing
    }
    response = client.post("/v1/ask-tutor", json=invalid_payload)
    assert response.status_code == 422 # FastAPI's Unprocessable Entity
    assert any(err["loc"] == ["body", "query_text"] for err in response.json()["detail"])


@patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.llm_client.generate_response', new_callable=AsyncMock)
async def test_ask_tutor_llm_call_failure(
    mock_generate_llm_response: AsyncMock,
    basic_tutor_request_payload: Dict
):
    """Test error handling if the LLM call fails with a generic exception."""
    mock_generate_llm_response.side_effect = Exception("Simulated LLM API Error")

    response = client.post("/v1/ask-tutor", json=basic_tutor_request_payload)
    
    assert response.status_code == 503
    assert "issue communicating with the AI assistance service" in response.json()["detail"].lower()


@patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.llm_client.generate_response', new_callable=AsyncMock)
async def test_ask_tutor_llm_config_env_error(
    mock_generate_llm_response: AsyncMock,
    basic_tutor_request_payload: Dict
):
    """Test error handling if LLM client raises EnvironmentError (e.g., GCP_PROJECT_ID missing)."""
    mock_generate_llm_response.side_effect = EnvironmentError("GCP_PROJECT_ID is not configured.")

    response = client.post("/v1/ask-tutor", json=basic_tutor_request_payload)
    
    assert response.status_code == 503 # Service Unavailable
    assert "ai service configuration error" in response.json()["detail"].lower()
    assert "gcp_project_id is not configured" in response.json()["detail"].lower()


@patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.fetch_course_content_from_django', new_callable=AsyncMock)
@patch('uplas_ai_agents.personalized_tutor_nlp_llm.main.llm_client.generate_response', new_callable=AsyncMock)
async def test_ask_tutor_django_fetch_failure(
    mock_llm_call: AsyncMock,
    mock_django_fetch: AsyncMock,
    basic_tutor_request_payload: Dict,
    mock_llm_successful_response: Dict
):
    """Test scenario where fetching course content from Django fails, but agent still proceeds."""
    mock_django_fetch.side_effect = Exception("Django connection error")
    mock_llm_call.return_value = mock_llm_successful_response # LLM should still be called

    payload_with_context = basic_tutor_request_payload.copy()
    payload_with_context["context"] = {"topic_id": "some_topic_id"}

    response = client.post("/v1/ask-tutor", json=payload_with_context)

    assert response.status_code == 200 # Agent should handle this gracefully and proceed
    mock_django_fetch.assert_awaited_once()
    mock_llm_call.assert_awaited_once() # Ensure LLM was still called
    
    # Check that the system prompt does not contain the course content placeholder if fetch failed
    # (or contains a message indicating content couldn't be fetched)
    call_args = mock_llm_call.call_args[1]
    assert "--- Relevant Information/Course Material Snippet ---" not in call_args["system_prompt"]


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "PersonalizedTutorNLP_LLM_Agent"}

# To run these tests:
# From within uplas-ai-agents/personalized_tutor_nlp_llm/
# pytest

