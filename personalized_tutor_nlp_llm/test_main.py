import pytest # Using pytest for more expressive tests and fixtures
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock # AsyncMock for async functions

# Import the FastAPI app and Pydantic models from main.py
# Assuming main.py is in the same directory or Python path is configured
from .main import (
    app,
    AiTutorQueryRequest,
    UserProfileSnapshot,
    TutorRequestContext,
    ConversationTurn,
    ConversationRole,
    mock_llm_client # We might want to patch this for some tests
)

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
def basic_tutor_request(basic_user_profile: UserProfileSnapshot) -> AiTutorQueryRequest:
    return AiTutorQueryRequest(
        user_id="test_user_001",
        query_text="What are Python functions?",
        user_profile_snapshot=basic_user_profile
    )

# --- Test Cases ---

def test_ask_tutor_endpoint_success_basic_query(basic_tutor_request: AiTutorQueryRequest):
    """Test the main endpoint with a valid basic request."""
    response = client.post("/v1/ask-tutor", json=basic_tutor_request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert "answer_text" in data
    assert "mocked, personalized answer" in data["answer_text"].lower() # Check for mock indicator
    assert basic_tutor_request.user_profile_snapshot.profession.lower() in data["answer_text"].lower()
    assert "suggested_follow_up_questions" in data
    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == mock_llm_client.model_name

def test_ask_tutor_with_specific_context(basic_user_profile: UserProfileSnapshot):
    """Test when context (e.g., topic title) is provided."""
    request_data = AiTutorQueryRequest(
        user_id="test_user_002",
        query_text="Explain this more simply.",
        user_profile_snapshot=basic_user_profile,
        context=TutorRequestContext(
            topic_id="topic_python_lists_id", # This ID will trigger specific mock RAG content
            current_topic_title="Python Lists Fundamentals"
        )
    )
    response = client.post("/v1/ask-tutor", json=request_data.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert "answer_text" in data
    assert "python lists fundamentals" in data["answer_text"].lower() # Mock LLM should reference this
    assert "python lists are versatile" in data["answer_text"].lower() # From mocked RAG content

def test_ask_tutor_with_chef_profile_for_lists(basic_user_profile: UserProfileSnapshot):
    """Test specific mock logic for a Chef asking about Python lists."""
    chef_profile = basic_user_profile.model_copy(update={"profession": "Chef", "industry": "Culinary"})
    request_data = AiTutorQueryRequest(
        user_id="chef_user_001",
        query_text="Explain Python lists to me.",
        user_profile_snapshot=chef_profile,
        context=TutorRequestContext(topic_id="topic_python_lists_id")
    )
    response = client.post("/v1/ask-tutor", json=request_data.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert "recipe ingredient list" in data["answer_text"].lower() # Specific analogy for chef
    assert len(data["generated_analogies"]) > 0
    assert "recipe ingredient list" in data["generated_analogies"][0]["analogy"].lower()

def test_ask_tutor_with_conversation_history(basic_tutor_request: AiTutorQueryRequest):
    """Test that conversation history is passed and potentially used (in prompt construction)."""
    request_data_dict = basic_tutor_request.model_dump()
    request_data_dict["conversation_history"] = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."}
    ]
    # We can't easily assert history *use* by the mock, but we can assert the endpoint accepts it
    # and the prompt constructor includes it.
    with patch('uplas-ai-agents.personalized_tutor_nlp_llm.main.construct_llm_prompt', return_value="Mocked Prompt") as mock_construct_prompt:
        response = client.post("/v1/ask-tutor", json=request_data_dict)
        assert response.status_code == 200
        
        # Check if construct_llm_prompt was called with history
        call_args = mock_construct_prompt.call_args[1] # Get keyword arguments
        assert len(call_args['conversation_history']) == 2
        assert call_args['conversation_history'][0].content == "What is Python?"


def test_ask_tutor_invalid_request_missing_query(basic_user_profile: UserProfileSnapshot):
    """Test request validation for missing query_text."""
    invalid_payload = {
        "user_id": "test_user_003",
        "user_profile_snapshot": basic_user_profile.model_dump()
        # query_text is missing
    }
    response = client.post("/v1/ask-tutor", json=invalid_payload)
    assert response.status_code == 422 # FastAPI's Unprocessable Entity
    assert "query_text" in response.json()["detail"][0]["loc"]

def test_ask_tutor_invalid_request_short_query(basic_tutor_request: AiTutorQueryRequest):
    """Test request validation for too short query_text."""
    request_data_dict = basic_tutor_request.model_dump()
    request_data_dict["query_text"] = "Hi" # Too short (min_length=3)
    response = client.post("/v1/ask-tutor", json=request_data_dict)
    assert response.status_code == 422
    assert "query_text" in response.json()["detail"][0]["loc"]


# Test the prompt construction logic directly
from .main import construct_llm_prompt # Assuming test_main.py is in the same dir as main.py

def test_construct_llm_prompt_basic(basic_user_profile: UserProfileSnapshot):
    prompt = construct_llm_prompt(
        query="Explain loops.",
        user_profile=basic_user_profile
    )
    assert "You are Uplas AI Tutor" in prompt # System message
    assert "Profession: Teacher" in prompt
    assert "Industry: Education" in prompt
    assert "User's Current Question" in prompt
    assert "USER: Explain loops." in prompt
    assert "ASSISTANT:" in prompt # LLM should start its response after this

def test_construct_llm_prompt_with_all_context(basic_user_profile: UserProfileSnapshot):
    profile = basic_user_profile.model_copy(update={
        "learning_style_preference": {"visual": 0.8, "reading_writing": 0.6},
        "current_knowledge_level": {"loops": "Beginner"}
    })
    context = TutorRequestContext(
        course_id="course1",
        topic_id="topic_loops_advanced",
        current_course_title="Advanced Python",
        current_topic_title="Advanced Loop Constructs",
        previous_assessment_feedback="User struggled with nested loops."
    )
    history = [
        ConversationTurn(role=ConversationRole.USER, content="What are loops?"),
        ConversationTurn(role=ConversationRole.ASSISTANT, content="Loops repeat code.")
    ]
    course_material = "Advanced loops include nested loops and list comprehensions."

    prompt = construct_llm_prompt(
        query="Tell me more about nested loops, considering my feedback.",
        user_profile=profile,
        course_content=course_material,
        context_data=context,
        conversation_history=history
    )
    # Check for key sections
    assert "SYSTEM:" in prompt
    assert "--- User Profile Summary for Personalization ---" in prompt
    assert "Learning Styles: visual (80%), reading_writing (60%)" in prompt # Check formatting
    assert "Knowledge: loops: Beginner" in prompt
    assert "--- Current Learning Context ---" in prompt
    assert "Course: Advanced Python" in prompt
    assert "Feedback from previous project attempt: User struggled with nested loops." in prompt
    assert "--- Relevant Information/Course Material Snippet ---" in prompt
    assert "Advanced loops include nested loops" in prompt
    assert "--- Recent Conversation History ---" in prompt
    assert "USER: What are loops?" in prompt
    assert "ASSISTANT: Loops repeat code." in prompt
    assert "--- User's Current Question ---" in prompt
    assert "USER: Tell me more about nested loops, considering my feedback." in prompt

@patch('uplas-ai-agents.personalized_tutor_nlp_llm.main.mock_llm_client.generate_response', new_callable=AsyncMock)
async def test_ask_tutor_llm_call_failure(mock_generate_llm_response, basic_tutor_request: AiTutorQueryRequest, event_loop):
    """Test error handling if the (mocked) LLM call fails."""
    # We need event_loop fixture when using AsyncMock with TestClient if the endpoint is async
    mock_generate_llm_response.side_effect = Exception("Simulated LLM API Error")

    # Need to use async client for async endpoint testing with pytest-asyncio if not using TestClient directly
    # For TestClient, it handles the event loop for sync calls to async endpoints.
    # If the endpoint itself was `async def` and we were calling it directly in an async test,
    # then pytest-asyncio and event_loop would be more directly involved.
    # TestClient bridges this.

    response = client.post("/v1/ask-tutor", json=basic_tutor_request.model_dump())
    
    assert response.status_code == 503 # Our custom error for LLM communication issues
    assert "issue communicating with the AI assistance service" in response.json()["detail"].lower()


# To run these tests (assuming pytest is installed and main.py is in the same directory as test_main.py,
# or appropriatePYTHONPATH is set):
# From within uplas-ai-agents/personalized_tutor_nlp_llm/
# pytest
