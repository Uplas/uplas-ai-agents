# uplas-ai-agents/personalized_tutor_nlp_llm/test_main.py
import pytest
import pytest_asyncio # For async fixtures
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, ANY, MagicMock # ANY and MagicMock are useful
import os
import json # For creating JSON string mocks

# Import the FastAPI app and Pydantic models from our refined main.py
from .main import (
    app,
    AiTutorQueryRequest,
    UserProfileSnapshot,
    TutorRequestContext,
    ConversationTurn,
    ConversationRole,
    # InnovateAI Enhanced/New Models:
    NlpModuleProcessed, # Expected structure from NLP content fetcher
    NlpLessonProcessed,
    NlpTopicProcessed,
    StructuredLLMOutput, # Expected JSON structure from LLM
    AiTutorQueryResponse,
    GeneratedAnalogy,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE
)

# Set environment variables for testing (as in original test file)
# These prevent actual VertexAI initialization during tests.
os.environ["GCP_PROJECT_ID"] = "test-gcp-project-id"
os.environ["GCP_LOCATION"] = "test-gcp-location"
os.environ["LLM_MODEL_NAME"] = "test-gemini-model-tutor"
# For fetching NLP content
os.environ["NLP_CONTENT_SERVICE_URL"] = "http://mock-nlp-content-service"


# Use TestClient for synchronous testing of async FastAPI app
client = TestClient(app) #

# --- Fixtures (InnovateAI Refined) ---

@pytest.fixture
def basic_user_profile() -> UserProfileSnapshot: #
    return UserProfileSnapshot(
        industry="Technology",
        profession="Software Developer",
        country="Kenya",
        city="Nairobi",
        preferred_tutor_persona="Direct and technical",
        career_interest="AI Ethics",
        learning_goals="Understand advanced NLP techniques."
    )

@pytest.fixture
def basic_tutor_request_dict(basic_user_profile: UserProfileSnapshot) -> dict: # Changed from payload to dict for easier modification
    """Provides a basic, valid request payload as a dictionary."""
    return {
        "user_id": "test_user_innovate_001",
        "query_text": "Explain attention mechanisms in transformers.",
        "user_profile_snapshot": basic_user_profile.model_dump(exclude_none=True),
        "language_code": "en-US",
        "max_tokens_response": 1024
    }

@pytest.fixture
def mock_structured_llm_output_success() -> StructuredLLMOutput:
    """InnovateAI: Provides a mock successful StructuredLLMOutput object."""
    return StructuredLLMOutput(
        main_answer_text="Attention mechanisms allow models to weigh the importance of different parts of the input sequence. As a Software Developer in Technology from Kenya, think of it like focusing on specific lines of code during a debug session.",
        suggested_follow_ups=["How is self-attention different?", "What are some applications of transformers?"],
        generated_analogies_for_answer=["Like a spotlight in a dark room.", "Similar to a dictionary lookup where keys have different relevance."],
        answer_confidence_score=0.92
    )

@pytest.fixture
def mock_nlp_module_processed_data() -> NlpModuleProcessed:
    """InnovateAI: Provides mock NLP-processed content for a module."""
    return NlpModuleProcessed(
        module_id="nlp_module_transformers_01_processed",
        module_title="Advanced Transformer Architectures",
        language_code="en-US",
        lessons=[
            NlpLessonProcessed(
                lesson_id="L1_attention_proc",
                lesson_title="Understanding Attention Mechanisms",
                lesson_summary="This lesson details the core concepts of attention in neural networks.",
                topics=[
                    NlpTopicProcessed(
                        topic_id="T1_self_attention_proc",
                        topic_title="Self-Attention Deep Dive",
                        key_concepts=["Query, Key, Value", "Scaled Dot-Product Attention"],
                        content_with_tags="Self-attention allows inputs to interact with each other. <analogy type=\"technical_analogy_needed\" /> It's crucial for understanding context. <difficulty type=\"advanced_detail\" /> Consider this example: <example domain=\"nlp_translation_example_needed\" />. Any questions? <interactive_question_opportunity text_suggestion=\"How does self-attention help in long sequences?\" />"
                    ),
                    NlpTopicProcessed(
                        topic_id="T2_multi_head_proc",
                        topic_title="Multi-Head Attention",
                        key_concepts=["Parallel attention layers", "Diverse representations"],
                        content_with_tags="Multi-head attention runs multiple attention mechanisms in parallel. <visual_aid_suggestion type=\"diagram_needed\" description=\"Diagram showing multiple attention heads\" /> This helps capture different aspects. <difficulty type=\"advanced_detail\" />"
                    )
                ]
            )
        ]
    )

@pytest_asyncio.fixture
async def mock_dependencies(monkeypatch):
    """
    InnovateAI: Central fixture to mock key external dependencies:
    - llm_client.generate_structured_response
    - fetch_processed_nlp_content
    """
    mock_llm_generate = AsyncMock(name="mock_llm_generate_structured_response")
    mock_fetch_nlp = AsyncMock(name="mock_fetch_processed_nlp_content")

    # Patch where these are instantiated or used in main.py
    monkeypatch.setattr("personalized_tutor_nlp_llm.main.llm_client.generate_structured_response", mock_llm_generate)
    monkeypatch.setattr("personalized_tutor_nlp_llm.main.fetch_processed_nlp_content", mock_fetch_nlp)
    
    return {
        "llm_generate": mock_llm_generate,
        "fetch_nlp": mock_fetch_nlp
    }


# --- Test Cases (InnovateAI Refined & New) ---

async def test_ask_tutor_success_with_nlp_content(
    mock_dependencies: Dict[str, AsyncMock], # Use the central mock fixture
    basic_tutor_request_dict: dict,
    mock_structured_llm_output_success: StructuredLLMOutput,
    mock_nlp_module_processed_data: NlpModuleProcessed
):
    """InnovateAI Test: Successful query, utilizes fetched NLP content, and parses structured LLM response."""
    mock_dependencies["fetch_nlp"].return_value = mock_nlp_module_processed_data
    mock_dependencies["llm_generate"].return_value = { # LLM client returns this dict
        "raw_response_text": mock_structured_llm_output_success.model_dump_json(),
        "prompt_token_count": 180,
        "response_token_count": 220
    }

    payload = basic_tutor_request_dict.copy()
    payload["context"] = {
        "module_id": "nlp_module_transformers_01_processed", # To trigger NLP fetch
        "topic_id": "T1_self_attention_proc" # To focus on a specific topic
    }

    response = client.post("/v1/ask-tutor", json=payload)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    # Validate response structure based on AiTutorQueryResponse and StructuredLLMOutput
    assert data["answer_text"] == mock_structured_llm_output_success.main_answer_text
    assert data["suggested_follow_up_questions"] == mock_structured_llm_output_success.suggested_follow_ups
    assert len(data["generated_analogies"]) == len(mock_structured_llm_output_success.generated_analogies_for_answer)
    assert data["generated_analogies"][0]["analogy"] == mock_structured_llm_output_success.generated_analogies_for_answer[0]
    assert data["confidence_score"] == mock_structured_llm_output_success.answer_confidence_score
    assert "debug_info" in data
    assert data["debug_info"]["language_used"] == payload["language_code"]

    # Verify mocks were called
    mock_dependencies["fetch_nlp"].assert_awaited_once_with(
        payload["context"]["module_id"],
        payload["context"]["topic_id"],
        payload["language_code"]
    )
    mock_dependencies["llm_generate"].assert_awaited_once()
    
    # Verify system prompt content (deeper inspection)
    call_args = mock_dependencies["llm_generate"].call_args[1] # Keyword arguments
    system_prompt_sent = call_args["system_prompt"]
    assert "InnovateAI Content Interaction Instructions" in system_prompt_sent
    assert mock_nlp_module_processed_data.lessons[0].topics[0].content_with_tags in system_prompt_sent # Check if NLP content snippet is there
    assert "Your entire response MUST be a single, valid JSON object" in system_prompt_sent
    assert StructuredLLMOutput.model_json_schema(indent=2) in system_prompt_sent # Check schema in prompt


async def test_ask_tutor_empathetic_project_remediation(
    mock_dependencies: Dict[str, AsyncMock],
    basic_user_profile: UserProfileSnapshot,
    mock_structured_llm_output_success: StructuredLLMOutput,
    mock_nlp_module_processed_data: NlpModuleProcessed # Assume some relevant course context
):
    """InnovateAI Test: Scenario where tutor provides empathetic feedback for a failed project."""
    mock_dependencies["fetch_nlp"].return_value = mock_nlp_module_processed_data
    mock_dependencies["llm_generate"].return_value = {
        "raw_response_text": mock_structured_llm_output_success.model_dump_json(), # LLM responds with guidance
        "prompt_token_count": 250, "response_token_count": 300
    }

    payload = {
        "user_id": "test_user_proj_fail_007",
        "query_text": "I failed my project on 'Transformer Performance Analysis'. Can you help me understand why?",
        "user_profile_snapshot": basic_user_profile.model_dump(exclude_none=True),
        "language_code": "en-US",
        "context": TutorRequestContext(
            module_id=mock_nlp_module_processed_data.module_id, # Link to relevant module
            current_project_title="Transformer Performance Analysis",
            project_assessment_feedback="The analysis lacked depth in evaluating different optimizers, and the report structure was unclear."
        ).model_dump(exclude_none=True)
    }

    response = client.post("/v1/ask-tutor", json=payload)
    assert response.status_code == status.HTTP_200_OK

    mock_dependencies["llm_generate"].assert_awaited_once()
    system_prompt_sent = mock_dependencies["llm_generate"].call_args[1]["system_prompt"]
    assert "InnovateAI Guidance for Project Remediation" in system_prompt_sent
    assert "Adopt an **Empathetic Mentor** role" in system_prompt_sent
    assert payload["context"]["project_assessment_feedback"] in system_prompt_sent


async def test_ask_tutor_llm_returns_invalid_json(
    mock_dependencies: Dict[str, AsyncMock],
    basic_tutor_request_dict: dict,
    mock_nlp_module_processed_data: NlpModuleProcessed
):
    """InnovateAI Test: LLM returns a non-JSON string when JSON was expected."""
    mock_dependencies["fetch_nlp"].return_value = mock_nlp_module_processed_data
    mock_dependencies["llm_generate"].return_value = {
        "raw_response_text": "This is not JSON, just plain text.", # Invalid JSON
        "prompt_token_count": 100, "response_token_count": 10
    }
    payload = basic_tutor_request_dict.copy()
    payload["context"] = {"module_id": mock_nlp_module_processed_data.module_id}


    response = client.post("/v1/ask-tutor", json=payload)
    assert response.status_code == status.HTTP_200_OK # Endpoint should handle gracefully
    data = response.json()
    # Check for fallback behavior defined in main.py
    assert "InnovateAI Apology: I couldn't structure my thoughts perfectly" in data["answer_text"]
    assert data["answer_text"].endswith("This is not JSON, just plain text.")


async def test_ask_tutor_llm_returns_json_not_matching_schema(
    mock_dependencies: Dict[str, AsyncMock],
    basic_tutor_request_dict: dict,
    mock_nlp_module_processed_data: NlpModuleProcessed
):
    """InnovateAI Test: LLM returns valid JSON, but it doesn't match StructuredLLMOutput schema."""
    mock_dependencies["fetch_nlp"].return_value = mock_nlp_module_processed_data
    invalid_schema_json_str = json.dumps({"unexpected_field": "some value", "main_answer_text_typo": "Answer here"})
    mock_dependencies["llm_generate"].return_value = {
        "raw_response_text": invalid_schema_json_str,
        "prompt_token_count": 100, "response_token_count": 10
    }
    payload = basic_tutor_request_dict.copy()
    payload["context"] = {"module_id": mock_nlp_module_processed_data.module_id}

    response = client.post("/v1/ask-tutor", json=payload)
    assert response.status_code == status.HTTP_200_OK # Graceful handling
    data = response.json()
    assert "InnovateAI Alert: My response structure was a bit off" in data["answer_text"]
    assert data["answer_text"].endswith(invalid_schema_json_str)


async def test_ask_tutor_nlp_content_fetch_fails(
    mock_dependencies: Dict[str, AsyncMock],
    basic_tutor_request_dict: dict,
    mock_structured_llm_output_success: StructuredLLMOutput
):
    """InnovateAI Test: fetch_processed_nlp_content returns None (e.g., content not found)."""
    mock_dependencies["fetch_nlp"].return_value = None # Simulate NLP content not found
    mock_dependencies["llm_generate"].return_value = {
        "raw_response_text": mock_structured_llm_output_success.model_dump_json(),
        "prompt_token_count": 90, "response_token_count": 150
    }
    
    payload = basic_tutor_request_dict.copy()
    payload["context"] = {"module_id": "non_existent_module_id"}

    response = client.post("/v1/ask-tutor", json=payload)
    assert response.status_code == status.HTTP_200_OK # Agent should still try to answer

    mock_dependencies["fetch_nlp"].assert_awaited_once_with(
        payload["context"]["module_id"], ANY, payload["language_code"]
    )
    mock_dependencies["llm_generate"].assert_awaited_once()
    system_prompt_sent = mock_dependencies["llm_generate"].call_args[1]["system_prompt"]
    # Verify that the prompt indicates no specific course material was found
    assert "No specific course material was found for this query" in system_prompt_sent
    assert "InnovateAI Content Interaction Instructions" not in system_prompt_sent # As no NLP content with tags was injected


async def test_ask_tutor_llm_call_general_exception(
    mock_dependencies: Dict[str, AsyncMock],
    basic_tutor_request_dict: dict
):
    """InnovateAI Test: LLM client's generate_structured_response raises a generic Exception."""
    mock_dependencies["fetch_nlp"].return_value = None # NLP fetch outcome doesn't matter here
    mock_dependencies["llm_generate"].side_effect = Exception("Simulated unexpected LLM API error")

    response = client.post("/v1/ask-tutor", json=basic_tutor_request_dict)
    
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "issue communicating with the AI assistance service" in response.json()["detail"].lower()


def test_health_check_endpoint_healthy(monkeypatch): # Modified from original to use monkeypatch for env var
    """InnovateAI Test: Health check returns healthy when configured."""
    # Env vars are set by global os.environ or could be managed by a fixture
    # Ensuring NLP_CONTENT_SERVICE_URL is not the default for a fully healthy check
    monkeypatch.setenv("NLP_CONTENT_SERVICE_URL", "http://actual-nlp-service")
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "status": "healthy",
        "service": "PersonalizedTutorNLP_LLM_Agent_InnovateAI",
        "innovate_ai_enhancements_active": True
    }

def test_health_check_unhealthy_missing_gcp_id(monkeypatch):
    """InnovateAI Test: Health check for missing GCP_PROJECT_ID."""
    with patch('personalized_tutor_nlp_llm.main.GCP_PROJECT_ID', None): # Temporarily override
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK # Health check endpoint itself works
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "GCP_PROJECT_ID not configured" in data["reason"]


def test_health_check_warning_nlp_service_url(monkeypatch):
    """InnovateAI Test: Health check logs warning if NLP_CONTENT_SERVICE_URL is default/missing but still returns healthy."""
    # Using default "your-backend-service" URL from main.py
    monkeypatch.setenv("NLP_CONTENT_SERVICE_URL", "http://your-backend-service/api/internal/get-processed-nlp-content")
    with patch.object(logging.getLogger("personalized_tutor_nlp_llm.main"), 'warning') as mock_log_warning:
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy" # Still healthy, but a warning should be logged
        mock_log_warning.assert_any_call(
            "InnovateAI Health Warning: NLP_CONTENT_SERVICE_URL is not configured or using default. NLP content fetching will be mocked/fail."
        )


# Remember to run tests with: pytest
# Ensure pytest-asyncio is installed: pip install pytest-asyncio
