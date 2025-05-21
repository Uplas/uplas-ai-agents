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
    GeneratedProjectIdea,
    GeneratedProjectTask, # For detailed checking
    mock_project_llm # To potentially patch its methods
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

# --- Test Cases for the Endpoint ---

def test_generate_project_ideas_success_data_analyst_profile(
    user_profile_data_analyst: UserProfileSnapshotForProjects,
    project_prefs_advanced_python: ProjectPreferences # Mix and match profile and prefs
):
    """Test successful idea generation for a Data Analyst profile."""
    request_payload = ProjectIdeaGenerationRequest(
        user_profile_snapshot=user_profile_data_analyst,
        preferences=project_prefs_advanced_python, # Using advanced python prefs for a data analyst
        number_of_ideas=2
    )
    payload_dict = request_payload.model_dump()
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    assert "generated_ideas" in data
    assert len(data["generated_ideas"]) == 2
    
    for idea_dict in data["generated_ideas"]:
        idea = GeneratedProjectIdea(**idea_dict) # Validate structure
        assert "Finance-Focused" in idea.title or "Data Analyst" in idea.title or "Algorithmic Trading" in idea.title # Check for personalization from mock
        assert idea.difficulty_level == "advanced" # From preferences
        assert any(tech in idea.suggested_technologies for tech in ["Python", "FastAPI", "Pandas"]), "Expected relevant tech not found"
        assert len(idea.key_tasks) > 0
        assert idea.personalization_rationale is not None
        assert "Finance" in idea.personalization_rationale or "Data Analyst" in idea.personalization_rationale

def test_generate_project_ideas_success_web_dev_student_profile(
    user_profile_web_dev_student: UserProfileSnapshotForProjects,
    project_prefs_beginner_frontend: ProjectPreferences
):
    """Test successful idea generation for a Web Dev Student profile."""
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
    assert "Student" in idea.title or "Web Dev" in idea.title or "E-commerce" in idea.title
    assert idea.difficulty_level == "beginner"
    assert "HTML" in idea.suggested_technologies or "JavaScript" in idea.suggested_technologies
    assert "portfolio piece" in idea.personalization_rationale.lower() or "full-stack" in idea.personalization_rationale.lower()

def test_generate_project_ideas_respects_number_of_ideas(
    user_profile_data_analyst: UserProfileSnapshotForProjects
):
    """Test that the number_of_ideas parameter is respected."""
    request_payload = ProjectIdeaGenerationRequest(
        user_profile_snapshot=user_profile_data_analyst,
        number_of_ideas=3
    )
    payload_dict = request_payload.model_dump()
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    assert response.status_code == 200
    data = response.json()
    assert len(data["generated_ideas"]) == 3

    request_payload.number_of_ideas = 1
    payload_dict = request_payload.model_dump()
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    assert response.status_code == 200
    data = response.json()
    assert len(data["generated_ideas"]) == 1

@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_returns_fewer_than_requested(
    mock_llm_generate, 
    user_profile_data_analyst: UserProfileSnapshotForProjects
):
    """Test scenario where LLM mock returns fewer ideas than requested but some are valid."""
    # Mock LLM to return only 1 idea even if more are requested
    mock_llm_generate.return_value = [ # A single valid idea structure
        {
            "request_id": "req_test", "project_idea_id": "idea_single",
            "title": "Single Valid Idea from Mock", "subtitle": "A subtitle",
            "description_html": "<p>Valid description.</p>", "difficulty_level": "intermediate",
            "learning_objectives_html": ["<li>Learn A</li>"], "key_tasks": [{"task_id":1, "description":"Do X"}],
            "suggested_technologies": ["Python"],
            "personalization_rationale": "Good for you."
        }
    ]
    request_payload = ProjectIdeaGenerationRequest(
        user_profile_snapshot=user_profile_data_analyst,
        number_of_ideas=3 # Request 3
    )
    payload_dict = request_payload.model_dump()
    response = client.post("/v1/generate-project-ideas", json=payload_dict)
    
    assert response.status_code == 200
    data = response.json()
    assert len(data["generated_ideas"]) == 1 # But only 1 was returned and validated
    assert data["generated_ideas"][0]["title"] == "Single Valid Idea from Mock"
    mock_llm_generate.assert_awaited_once()


@patch('uplas-ai-agents.project_generator_agent.main.mock_project_llm.generate_ideas', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_returns_mix_of_valid_and_invalid_ideas(
    mock_llm_generate, 
    user_profile_data_analyst: UserProfileSnapshotForProjects
):
    """Test when LLM returns a mix, only valid ones are passed through."""
    mock_llm_generate.return_value = [
        { # Valid Idea
            "request_id": "req_valid", "project_idea_id": "idea_v",
            "title": "Valid Idea Structure", "subtitle": "Sub",
            "description_html": "<p>Desc.</p>", "difficulty_level": "beginner",
            "learning_objectives_html": ["<li>Obj</li>"], "key_tasks": [{"task_id":1, "description":"T"}],
            "suggested_technologies": ["TechX"]
        },
        {"title": "Idea Missing Description HTML", "difficulty_level": "easy"}, # Invalid (missing description_html)
        { # Valid Idea 2
            "request_id": "req_valid2", "project_idea_id": "idea_v2",
            "title": "Another Valid Idea", "subtitle": "Sub2",
            "description_html": "<p>Desc2.</p>", "difficulty_level": "intermediate",
            "learning_objectives_html": ["<li>Obj2</li>"], "key_tasks": [{"task_id":1, "description":"T2"}],
            "suggested_technologies": ["TechY"]
        }
    ]
    request_payload = ProjectIdeaGenerationRequest(
        user_profile_snapshot=user_profile_data_analyst,
        number_of_ideas=3
    )
    payload_dict = request_payload.model_dump()
    response = client.post("/v1/generate-project-ideas", json=payload_dict)

    assert response.status_code == 200 # Should still succeed if at least one is valid
    data = response.json()
    assert len(data["generated_ideas"]) == 2 # Only the two valid ideas
    assert data["generated_ideas"][0]["title"] == "Valid Idea Structure"
    assert data["generated_ideas"][1]["title"] == "Another Valid Idea"
    assert data["debug_info"]["ideas_validation_failures"] == 1


# --- Direct tests for construct_project_gen_prompt ---
from .main import construct_project_gen_prompt

def test_construct_project_gen_prompt_includes_all_sections(
    user_profile_data_analyst: UserProfileSnapshotForProjects,
    project_prefs_advanced_python: ProjectPreferences
):
    prompt = construct_project_gen_prompt(
        user_profile=user_profile_data_analyst,
        preferences=project_prefs_advanced_python,
        num_ideas=1
    )
    assert "USER PROFILE:" in prompt
    assert f"- Current or Target Industry: {user_profile_data_analyst.industry}" in prompt
    assert f"- Self-Assessed Knowledge Levels: Python: Intermediate, SQL: Advanced, Pandas: Intermediate" in prompt # Check formatting
    assert "USER'S PROJECT PREFERENCES:" in prompt
    assert f"- Desired Difficulty Level: {project_prefs_advanced_python.difficulty_level}" in prompt
    assert f"- Preferred Technologies: Python, FastAPI, PostgreSQL" in prompt
    assert "TASK: Generate 1 distinct project idea(s)" in prompt
    assert '"title": "string (catchy, descriptive, and highly personalized project title' in prompt # Check for JSON structure guidance
    assert "personalization_rationale" in prompt # Key field for LLM
    assert "IMPORTANT: Respond with a valid JSON list" in prompt

def test_construct_project_gen_prompt_handles_minimal_profile_and_prefs():
    minimal_profile = UserProfileSnapshotForProjects(user_id="user_min_123")
    minimal_prefs = ProjectPreferences() # All defaults
    
    prompt = construct_project_gen_prompt(
        user_profile=minimal_profile,
        preferences=minimal_prefs,
        num_ideas=1
    )
    assert "USER PROFILE:" in prompt
    assert "- Current or Target Industry:" not in prompt # Should not appear if None
    assert "- Self-Assessed Knowledge Levels:" not in prompt # Should not appear if empty dict
    assert "USER'S PROJECT PREFERENCES:" in prompt
    assert f"- Desired Difficulty Level: {minimal_prefs.difficulty_level}" in prompt # Default intermediate
    assert "- Preferred Technologies: User is open to suggestions" in prompt # Specific text for empty list
    assert "TASK: Generate 1 distinct project idea(s)" in prompt


# To run these tests (requires pytest and pytest-asyncio for async patched methods):
# From within uplas-ai-agents/project_generator_agent/
# pytest
