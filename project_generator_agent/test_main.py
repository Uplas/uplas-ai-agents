import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import uuid
import json # For more complex payload assertions

# Import the FastAPI app and Pydantic models from main.py
from .main import (
    app,
    ProjectIdeaGenerationRequest,
    UserProfileSnapshotForProjects,
    ProjectPreferences,
    GeneratedProjectIdea, # For checking response structure
    GeneratedProjectTask, # For detailed checking
    MOCKED_PROJECT_LLM_NAME # Import to check in debug info
    # mock_project_llm # To potentially patch its methods if needed, but usually patch the method directly
)

client = TestClient(app)

# --- Fixtures ---
@pytest.fixture
def user_profile_data_analyst() -> UserProfileSnapshotForProjects:
    return UserProfileSnapshotForProjects(
        user_id=f"user_analyst_{uuid.uuid4().hex[:6]}",
        industry="Finance",
        profession="Data Analyst",
        career_interest="Quantitative Finance",
        current_knowledge_level={"Python": "Intermediate", "SQL": "Advanced", "Pandas": "Intermediate"},
        areas_of_interest=["Algorithmic Trading", "Risk Management", "Data Visualization"],
        learning_goals="Build a project demonstrating financial data analysis."
    )

@pytest.fixture
def user_profile_web_dev_student() -> UserProfileSnapshotForProjects:
    return UserProfileSnapshotForProjects(
        user_id=f"user_webdev_{uuid.uuid4().hex[:6]}",
        industry="Technology", # General tech
        profession="Student",
        career_interest="Full-Stack Web Developer",
        current_knowledge_level={"JavaScript": "Beginner", "HTML/CSS": "Intermediate"},
        areas_of_interest=["E-commerce Platforms", "Social Networking Apps", "API Development"],
        learning_goals="Create a full-stack web application with user authentication."
    )

@pytest.fixture
def project_prefs_advanced_python() -> ProjectPreferences:
    return ProjectPreferences(
        difficulty_level="advanced",
        preferred_technologies=["Python", "FastAPI", "PostgreSQL"],
        project_type_focus="Scalable Backend Service",
        time_commitment_hours_estimate=40
    )

@pytest.fixture
def project_prefs_beginner_frontend() -> ProjectPreferences:
    return ProjectPreferences(
        difficulty_level="beginner",
        preferred_technologies=["HTML", "CSS", "JavaScript (Vanilla)"],
        project_type_focus="Interactive Frontend Component",
        time_commitment_hours_estimate=15
    )

@pytest.fixture
def project_gen_request_data_analyst( # More specific fixture name
    user_profile_data_analyst: UserProfileSnapshotForProjects,
    project_prefs_advanced_python: ProjectPreferences
) -> ProjectIdeaGenerationRequest:
    return ProjectIdeaGenerationRequest(
        user_profile_snapshot=user_profile_data_analyst,
        preferences=project_prefs_advanced_python,
        number_of_ideas=2
    )

# --- Test Cases for the Endpoint ---

def test_generate_project_ideas_success_data_analyst_profile(
    project_gen_request_data_analyst: ProjectIdeaGenerationRequest
):
    """Test successful idea generation for a Data Analyst profile with specific preferences."""
    payload_dict = project_gen_request_data_analyst.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "generated_ideas" in data
    assert len(data["generated_ideas"]) == project_gen_request_data_analyst.number_of_ideas
    
    for idea_dict in data["generated_ideas"]:
        idea = GeneratedProjectIdea(**idea_dict) # Validate structure by parsing
        assert idea.title is not None
        # Check for personalization cues based on the refined mock LLM
        assert "Finance-Focused" in idea.title or "Data Analyst" in idea.title or \
               any(interest.lower() in idea.title.lower() for interest in project_gen_request_data_analyst.user_profile_snapshot.areas_of_interest)
        assert idea.difficulty_level == project_gen_request_data_analyst.preferences.difficulty_level
        assert len(idea.learning_objectives_html) >= 3 # Refined mock generates more
        assert len(idea.key_tasks) >= 5 # Refined mock generates more tasks
        assert any(tech in idea.suggested_technologies for tech in project_gen_request_data_analyst.preferences.preferred_technologies), "Expected preferred tech not found"
        assert idea.personalization_rationale is not None and len(idea.personalization_rationale) > 10
        assert "Finance" in idea.personalization_rationale or "Data Analyst" in idea.personalization_rationale or \
               any(interest.lower() in idea.personalization_rationale.lower() for interest in project_gen_request_data_analyst.user_profile_snapshot.areas_of_interest)

    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == MOCKED_PROJECT_LLM_NAME
    assert data["debug_info"]["num_ideas_generated_valid"] == project_gen_request_data_analyst.number_of_ideas


def test_generate_project_ideas_success_web_dev_student_profile(
    user_profile_web_dev_student: UserProfileSnapshotForProjects,
    project_prefs_beginner_frontend: ProjectPreferences
):
    """Test successful idea generation for a Web Dev Student profile with beginner preferences."""
    request_payload = ProjectIdeaGenerationRequest(
        user_profile_snapshot=user_profile_web_dev_student,
        preferences=project_prefs_beginner_frontend,
        number_of_ideas=1
    )
    payload_dict = request_payload.model_dump()
    response = client.post("/v1/generate-project-ideas", json=payload_dict)

    assert response.status_code == 200
    data = response.json()
    assert len(data["generated_ideas"]) == 1
    idea = GeneratedProjectIdea(**data["generated_ideas"][0])
    assert "Student" in idea.title or "Web Dev" in idea.title or \
           any(interest.lower() in idea.title.lower() for interest in user_profile_web_dev_student.areas_of_interest)
    assert idea.difficulty_level == "beginner"
    assert any(tech.lower() in map(str.lower, idea.suggested_technologies) for tech in project_prefs_beginner_frontend.preferred_technologies), "Expected preferred frontend tech not found"
    assert "portfolio piece" in idea.personalization_rationale.lower() or "full-stack" in idea.personalization_rationale.lower() or \
           any(interest.lower() in idea.personalization_rationale.lower() for interest in user_profile_web_dev_student.areas_of_interest)

def test_generate_project_ideas_respects_number_of_ideas_param(
    user_profile_data_analyst: UserProfileSnapshotForProjects
):
    """Test that the number_of_ideas parameter is respected by the mock (up to its template limit)."""
    max_mock_ideas = 3 # Our mock LLM template list size might be the actual cap if num_ideas is larger
    
    for num_req in [1, 2, 3]:
        request_payload = ProjectIdeaGenerationRequest(
            user_profile_snapshot=user_profile_data_analyst,
            number_of_ideas=num_req
        )
        payload_dict = request_payload.model_dump()
        response = client.post("/v1/generate-project-ideas", json=payload_dict)
        assert response.status_code == 200
        data = response.json()
        assert len(data["generated_ideas"]) == min(num_req, max_mock_ideas) # Mock might have fewer templates than requested if num_req > available_templates

@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_returns_empty_list( # Corrected test name
    mock_llm_generate_ideas, # Renamed mock object for clarity
    project_gen_request_data_analyst: ProjectIdeaGenerationRequest, # Use fixture
    event_loop 
):
    """Test scenario where the (mocked) LLM returns no ideas."""
    mock_llm_generate_ideas.return_value = [] # LLM found nothing suitable
    payload_dict = project_gen_request_data_analyst.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 404 # As per our refined endpoint logic
    assert "Could not generate suitable project ideas" in response.json()["detail"]
    mock_llm_generate_ideas.assert_awaited_once()

@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_returns_invalid_idea_structure(
    mock_llm_generate_ideas, 
    project_gen_request_data_analyst: ProjectIdeaGenerationRequest
):
    """Test when LLM returns data that doesn't match GeneratedProjectIdea Pydantic model."""
    mock_llm_generate_ideas.return_value = [
        {"title": "Only Title Field", "difficulty_level": "easy"} # Missing many required fields like description_html, learning_objectives_html etc.
    ]
    payload_dict = project_gen_request_data_analyst.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 500 # Because raw ideas existed but all failed Pydantic validation
    assert "AI service returned project ideas, but they were in an unexpected or incomplete format." in response.json()["detail"]
    assert "1 ideas had validation issues" in response.json()["detail"] # Check the count in the message
    mock_llm_generate_ideas.assert_awaited_once()


def test_generate_project_ideas_invalid_request_payload_bad_num_ideas(
    project_gen_request_data_analyst: ProjectIdeaGenerationRequest # Use fixture
):
    """Test with invalid number_of_ideas (e.g., too large or zero)."""
    payload_dict_too_many = project_gen_request_data_analyst.model_dump()
    payload_dict_too_many["number_of_ideas"] = 10 # Pydantic model caps at 3 (le=3)

    response_too_many = client.post("/v1/generate-project-ideas", json=payload_dict_too_many)
    assert response_too_many.status_code == 422 # Pydantic validation error
    error_detail_too_many = response_too_many.json()["detail"]
    assert any("number_of_ideas" in e["loc"] and "ensure this value is less than or equal to 3" in e["msg"].lower() for e in error_detail_too_many if "loc" in e and "msg" in e)

    payload_dict_zero = project_gen_request_data_analyst.model_dump()
    payload_dict_zero["number_of_ideas"] = 0 # Pydantic model requires ge=1
    response_zero = client.post("/v1/generate-project-ideas", json=payload_dict_zero)
    assert response_zero.status_code == 422 # Pydantic validation error
    error_detail_zero = response_zero.json()["detail"]
    assert any("number_of_ideas" in e["loc"] and "ensure this value is greater than or equal to 1" in e["msg"].lower() for e in error_detail_zero if "loc" in e and "msg" in e)


@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_call_exception(
    mock_llm_generate_ideas, 
    project_gen_request_data_analyst: ProjectIdeaGenerationRequest # Use fixture
):
    """Test handling of an unexpected exception from the LLM client call."""
    mock_llm_generate_ideas.side_effect = Exception("Simulated LLM Service Network Outage")
    payload_dict = project_gen_request_data_analyst.model_dump()
    
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 503
    assert "Error communicating with the AI project generation service" in response.json()["detail"]
    mock_llm_generate_ideas.assert_awaited_once()


# --- Direct tests for construct_project_gen_prompt (Conceptual LLM Prompting) ---
from .main import construct_project_gen_prompt # Assuming test_main.py is in the same dir as main.py

def test_construct_project_gen_prompt_includes_all_sections_and_personalization_cues(
    user_profile_data_analyst: UserProfileSnapshotForProjects,
    project_prefs_advanced_python: ProjectPreferences
):
    """Test the detailed prompt construction for a real LLM."""
    prompt = construct_project_gen_prompt(
        user_profile=user_profile_data_analyst,
        preferences=project_prefs_advanced_python,
        num_ideas=1
    )
    # Check for key instructional phrases
    assert "You are an AI assistant specialized in generating personalized, real-world project ideas" in prompt
    assert "USER PROFILE:" in prompt
    assert f"- Current or Target Industry: {user_profile_data_analyst.industry}" in prompt
    assert f"- Self-Assessed Knowledge Levels: Python: Intermediate, SQL: Advanced, Pandas: Intermediate" in prompt
    assert f"- Specific Areas of Interest: Algorithmic Trading, Risk Management, Data Visualization" in prompt
    
    assert "USER'S PROJECT PREFERENCES:" in prompt
    assert f"- Desired Difficulty Level: {project_prefs_advanced_python.difficulty_level}" in prompt
    assert f"- Preferred Technologies: Python, FastAPI, PostgreSQL" in prompt
    
    assert "TASK: Generate 1 distinct project idea(s)" in prompt
    assert '"title": "string (catchy, descriptive, and highly personalized project title' in prompt
    assert '"personalization_rationale": "string (CRUCIAL: 2-3 sentences explaining *precisely* why this project' in prompt
    assert "IMPORTANT: Respond with a valid JSON list" in prompt

def test_construct_project_gen_prompt_handles_minimal_profile_and_default_prefs():
    """Test prompt construction with minimal user input, relying on defaults for prefs."""
    minimal_profile = UserProfileSnapshotForProjects(user_id="user_minimal_test_123")
    # Preferences will use their Pydantic model defaults (e.g., difficulty='intermediate', empty preferred_technologies)
    default_prefs = ProjectPreferences() 
    
    prompt = construct_project_gen_prompt(
        user_profile=minimal_profile,
        preferences=default_prefs, # Pass default preferences explicitly
        num_ideas=1
    )
    assert "USER PROFILE:" in prompt
    assert f"- User ID (for reference only): {minimal_profile.user_id}" in prompt
    assert "- Current or Target Industry:" not in prompt # Should not appear if field is None in profile
    assert "- Self-Assessed Knowledge Levels:" not in prompt # Should not appear if field is empty dict

    assert "USER'S PROJECT PREFERENCES:" in prompt
    assert f"- Desired Difficulty Level: {default_prefs.difficulty_level}" in prompt # Default 'intermediate'
    assert "- Preferred Technologies: User is open to suggestions" in prompt # Specific text for empty list
    assert "TASK: Generate 1 distinct project idea(s)" in prompt


# To run these tests (requires pytest and pytest-asyncio for async patched methods):
# From within the directory containing main.py and test_main.py (e.g., uplas-ai-agents/project_generator_agent/):
# pytest
