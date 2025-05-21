import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import uuid

# Import the FastAPI app and Pydantic models from main.py
from .main import (
    app,
    ProjectIdeaGenerationRequest,
    UserProfileSnapshotForProjects,
    ProjectPreferences,
    GeneratedProjectIdea, # For checking response structure
    mock_project_llm # To potentially patch its methods
)

client = TestClient(app)

# --- Fixtures ---
@pytest.fixture
def basic_user_profile_for_projects() -> UserProfileSnapshotForProjects:
    return UserProfileSnapshotForProjects(
        user_id=f"user_{uuid.uuid4().hex[:6]}",
        industry="Software Development",
        profession="Junior Developer",
        career_interest="Full-Stack Development",
        current_knowledge_level={"Python": "Beginner", "JavaScript": "Novice"},
        areas_of_interest=["Web Applications", "API Design"],
        learning_goals="Build a complete web application."
    )

@pytest.fixture
def basic_project_preferences() -> ProjectPreferences:
    return ProjectPreferences(
        difficulty_level="beginner",
        preferred_technologies=["Python", "FastAPI"],
        project_type_focus="Portfolio Piece",
        time_commitment_hours_estimate=20
    )

@pytest.fixture
def project_gen_request_payload(
    basic_user_profile_for_projects: UserProfileSnapshotForProjects,
    basic_project_preferences: ProjectPreferences
) -> ProjectIdeaGenerationRequest:
    return ProjectIdeaGenerationRequest(
        user_profile_snapshot=basic_user_profile_for_projects,
        preferences=basic_project_preferences,
        number_of_ideas=2
    )

# --- Test Cases ---

def test_generate_project_ideas_success(project_gen_request_payload: ProjectIdeaGenerationRequest):
    """Test the main endpoint with a valid request, expecting successful idea generation."""
    payload_dict = project_gen_request_payload.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "generated_ideas" in data
    assert len(data["generated_ideas"]) == project_gen_request_payload.number_of_ideas
    
    for idea in data["generated_ideas"]:
        # Validate structure against Pydantic model by trying to parse it (or check key fields)
        parsed_idea = GeneratedProjectIdea(**idea) # This will raise error if fields mismatch
        assert parsed_idea.title is not None
        assert project_gen_request_payload.user_profile_snapshot.profession.lower() in parsed_idea.title.lower() \
            or project_gen_request_payload.user_profile_snapshot.industry.lower() in parsed_idea.title.lower() # Check mock personalization
        assert parsed_idea.difficulty_level == project_gen_request_payload.preferences.difficulty_level
        assert len(parsed_idea.learning_objectives_html) > 0
        assert len(parsed_idea.key_tasks) > 0
        assert len(parsed_idea.suggested_technologies) > 0

    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == "mocked-gemini-pro-project-gen"


def test_generate_project_ideas_minimal_input(basic_user_profile_for_projects: UserProfileSnapshotForProjects):
    """Test with minimal preferences (relying on defaults)."""
    minimal_request = ProjectIdeaGenerationRequest(
        user_profile_snapshot=basic_user_profile_for_projects
        # preferences and number_of_ideas will use defaults
    )
    payload_dict = minimal_request.model_dump()
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    assert response.status_code == 200
    data = response.json()
    assert len(data["generated_ideas"]) == 1 # Default number_of_ideas
    # Check if default difficulty was used by mock
    assert data["generated_ideas"][0]["difficulty_level"] == ProjectPreferences().difficulty_level


@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_returns_empty_list(
    mock_llm_generate, 
    project_gen_request_payload: ProjectIdeaGenerationRequest,
    event_loop # For async test with TestClient if not handled automatically
):
    """Test scenario where the (mocked) LLM returns no ideas."""
    mock_llm_generate.return_value = [] # LLM found nothing suitable
    payload_dict = project_gen_request_payload.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 404 # As per our endpoint logic
    assert "Could not generate suitable project ideas" in response.json()["detail"]


@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_returns_invalid_idea_structure(
    mock_llm_generate, 
    project_gen_request_payload: ProjectIdeaGenerationRequest
):
    """Test when LLM returns data that doesn't match GeneratedProjectIdea Pydantic model."""
    mock_llm_generate.return_value = [
        {"title": "Valid Idea 1", "description_html": "...", "difficulty_level": "easy", "learning_objectives_html": [], "key_tasks": [], "suggested_technologies": []}, # Missing required fields
        {"completely_wrong_field": "some_value"} 
    ]
    payload_dict = project_gen_request_payload.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    # If at least one idea was valid, it might still return 200 with only the valid ones.
    # If ALL ideas from LLM fail Pydantic validation within the loop, the endpoint raises 500 or 404.
    # Our current logic: if validated_ideas is empty BUT raw_ideas_from_llm was not, it raises 500.
    # If raw_ideas_from_llm itself was empty, it raises 404.
    # If mock_llm_generate.return_value contains items that individually fail Pydantic's GeneratedProjectIdea(**raw_idea),
    # the print warning in the endpoint will trigger, and validated_ideas will be empty.
    assert response.status_code == 500 # Because raw ideas existed but all failed validation
    assert "AI service returned ideas in an unexpected format" in response.json()["detail"]


def test_generate_project_ideas_invalid_request_payload_bad_num_ideas(
    project_gen_request_payload: ProjectIdeaGenerationRequest
):
    """Test with invalid number_of_ideas (e.g., too large)."""
    payload_dict = project_gen_request_payload.model_dump()
    payload_dict["number_of_ideas"] = 10 # Assuming our Pydantic model caps it at 5

    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    assert response.status_code == 422 # Pydantic validation error
    error_detail = response.json()["detail"]
    assert any("number_of_ideas" in e["loc"] for e in error_detail if "loc" in e)


@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_call_exception(
    mock_llm_generate, 
    project_gen_request_payload: ProjectIdeaGenerationRequest
):
    """Test handling of an unexpected exception from the LLM client."""
    mock_llm_generate.side_effect = Exception("Simulated LLM Service Outage")
    payload_dict = project_gen_request_payload.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 503 # Error communicating with the AI project generation service
    assert "Error communicating with the AI project generation service" in response.json()["detail"]


# Optional: Direct test for prompt construction if it were complex and used by a real LLM
# from .main import construct_project_gen_prompt
# def test_construct_project_gen_prompt_structure(
#     basic_user_profile_for_projects: UserProfileSnapshotForProjects,
#     basic_project_preferences: ProjectPreferences
# ):
#     prompt = construct_project_gen_prompt(
#         user_profile=basic_user_profile_for_projects,
#         preferences=basic_project_preferences,
#         num_ideas=1
#     )
#     assert "Generate 1 personalized real-world project idea(s)" in prompt
#     assert f"- Industry: {basic_user_profile_for_projects.industry}" in prompt
#     assert f"- Difficulty: {basic_project_preferences.difficulty_level}" in prompt
#     assert "- title (string, catchy and descriptive)" in prompt # Check for output structure guidance

# To run these tests (requires pytest and pytest-asyncio):
# From within uplas-ai-agents/project_generator_agent/
# pytest
