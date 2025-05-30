# uplas-ai-agents/project_generator_agent/test_main.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import os
import uuid
import json
import httpx # For mocking calls to AI Tutor

# Import the FastAPI app and Pydantic models from main.py
from .main import (
    app,
    ProjectIdeaGenerationRequest,
    UserProfileSnapshotForProjects,
    ProjectPreferences,
    GeneratedProjectIdea, # For constructing mock LLM responses
    GeneratedProjectTask,
    ProjectAssessmentRequest,
    ProjectSubmissionDetails,
    ProjectAssessmentResult,
    ASSESSMENT_PASS_THRESHOLD,
    DEFAULT_LANGUAGE
    # project_idea_llm and assessment_llm will be patched
)

# Set environment variables for testing
os.environ["GCP_PROJECT_ID"] = "test-gcp-project-id"
os.environ["GCP_LOCATION"] = "test-gcp-location"
os.environ["PROJECT_LLM_MODEL_NAME"] = "test-project-gemini-model"
os.environ["ASSESSMENT_LLM_MODEL_NAME"] = "test-assessment-gemini-model"
os.environ["AI_TUTOR_AGENT_URL"] = "http://mock-ai-tutor-agent.com"

client = TestClient(app)

# --- Fixtures for Project Idea Generation ---
@pytest.fixture
def user_profile_data_analyst() -> UserProfileSnapshotForProjects:
    return UserProfileSnapshotForProjects(
        user_id=f"user_analyst_{uuid.uuid4().hex[:6]}",
        industry="Finance",
        profession="Data Analyst",
        career_interest="Quantitative Finance",
        current_knowledge_level={"Python": "Intermediate", "SQL": "Advanced"},
        areas_of_interest=["Algorithmic Trading", "Risk Management"],
        learning_goals="Build a project demonstrating financial data analysis.",
        preferred_tutor_persona="Direct and informative"
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
def project_gen_request_payload_dict(
    user_profile_data_analyst: UserProfileSnapshotForProjects,
    project_prefs_advanced_python: ProjectPreferences
) -> Dict:
    return {
        "user_profile_snapshot": user_profile_data_analyst.model_dump(),
        "preferences": project_prefs_advanced_python.model_dump(),
        "number_of_ideas": 1,
        "language_code": "en-US"
    }

@pytest.fixture
def mock_successful_project_idea_llm_response_text() -> str:
    # LLM is expected to return a JSON string which is a list of project ideas
    ideas = [
        GeneratedProjectIdea(
            title="Personalized Stock Portfolio Analyzer",
            description_html="<p>Analyze stock performance and build a personalized dashboard.</p>",
            difficulty_level="advanced",
            learning_objectives_html=["<li>Master FastAPI</li>", "<li>Understand financial APIs</li>"],
            key_tasks=[GeneratedProjectTask(task_id=1, description="Setup FastAPI project.")],
            suggested_technologies=["Python", "FastAPI", "AlphaVantage API"],
            assessment_rubric_preview_html=["<li>API Integration: 50%</li>"],
            language_code="en-US"
        ).model_dump() # Convert Pydantic model to dict for JSON serialization
    ]
    return json.dumps(ideas)


# --- Fixtures for Project Assessment ---
@pytest.fixture
def sample_generated_project_idea() -> GeneratedProjectIdea:
    return GeneratedProjectIdea(
        project_idea_id="proj_idea_sample_123",
        title="Eco-Friendly Route Planner API",
        description_html="<p>Develop an API that suggests the most eco-friendly routes.</p>",
        difficulty_level="intermediate",
        learning_objectives_html=["<li>Learn API design.</li>", "<li>Work with geospatial data.</li>"],
        key_tasks=[
            GeneratedProjectTask(task_id=1, description="Design API endpoints."),
            GeneratedProjectTask(task_id=2, description="Implement routing algorithm.")
        ],
        suggested_technologies=["Python", "FastAPI", "GeoPy"],
        assessment_rubric_preview_html=["<li>Functionality: 60%</li>", "<li>Code Quality: 20%</li>", "<li>API Design: 20%</li>"],
        language_code="en-US"
    )

@pytest.fixture
def project_submission_details_pass() -> ProjectSubmissionDetails:
    return ProjectSubmissionDetails(
        github_url="https://github.com/testuser/eco-planner",
        textual_summary="Implemented all core features and added unit tests."
    )

@pytest.fixture
def project_submission_details_fail() -> ProjectSubmissionDetails:
    return ProjectSubmissionDetails(
        textual_summary="Only implemented the basic API design. Struggled with the routing algorithm."
    )

@pytest.fixture
def project_assessment_request_payload_pass_dict(
    user_profile_data_analyst: UserProfileSnapshotForProjects, # Re-use for user_id
    sample_generated_project_idea: GeneratedProjectIdea,
    project_submission_details_pass: ProjectSubmissionDetails
) -> Dict:
    return {
        "user_id": user_profile_data_analyst.user_id,
        "project_idea": sample_generated_project_idea.model_dump(),
        "submission": project_submission_details_pass.model_dump(),
        "language_code": "en-US"
    }

@pytest.fixture
def project_assessment_request_payload_fail_dict(
    user_profile_data_analyst: UserProfileSnapshotForProjects,
    sample_generated_project_idea: GeneratedProjectIdea,
    project_submission_details_fail: ProjectSubmissionDetails
) -> Dict:
    return {
        "user_id": user_profile_data_analyst.user_id,
        "project_idea": sample_generated_project_idea.model_dump(),
        "submission": project_submission_details_fail.model_dump(),
        "language_code": "en-US"
    }

@pytest.fixture
def mock_successful_assessment_llm_response_text_pass() -> str:
    result = ProjectAssessmentResult(
        project_idea_id="proj_idea_sample_123",
        user_id="user_analyst_test",
        score=85.0,
        is_passed=True,
        feedback_summary_html="<p>Great job! Excellent implementation of the core features.</p>",
        detailed_feedback_points_html=["<li>API design is solid.</li>", "<li>Routing logic is efficient.</li>"],
        areas_for_improvement_html=["<li>Consider adding more edge case handling for geo-queries.</li>"],
        positive_points_html=["<li>Clean code and good use of FastAPI.</li>"],
        language_code="en-US"
    ).model_dump()
    return json.dumps(result)

@pytest.fixture
def mock_successful_assessment_llm_response_text_fail() -> str:
    result = ProjectAssessmentResult(
        project_idea_id="proj_idea_sample_123",
        user_id="user_analyst_test",
        score=60.0,
        is_passed=False, # Based on score < ASSESSMENT_PASS_THRESHOLD
        feedback_summary_html="<p>Good start, but the core routing algorithm needs more work.</p>",
        detailed_feedback_points_html=["<li>API endpoints are well-defined.</li>"],
        areas_for_improvement_html=["<li>The routing logic is incomplete.</li>", "<li>Error handling is missing.</li>"],
        positive_points_html=["<li>Good understanding of the project requirements shown in the API design.</li>"],
        language_code="en-US"
    ).model_dump()
    return json.dumps(result)

# --- Test Cases for Project Idea Generation ---

@patch('uplas_ai_agents.project_generator_agent.main.project_idea_llm.generate_structured_response', new_callable=AsyncMock)
async def test_generate_project_ideas_success(
    mock_llm_gen_ideas: AsyncMock,
    project_gen_request_payload_dict: Dict,
    mock_successful_project_idea_llm_response_text: str
):
    """Test successful project idea generation."""
    mock_llm_gen_ideas.return_value = {
        "response_text": mock_successful_project_idea_llm_response_text,
        "prompt_token_count": 100,
        "response_token_count": 300
    }

    response = client.post("/v1/generate-project-ideas", json=project_gen_request_payload_dict)

    assert response.status_code == 200
    data = response.json()
    assert "generated_ideas" in data
    assert len(data["generated_ideas"]) == 1
    idea = data["generated_ideas"][0]
    assert idea["title"] == "Personalized Stock Portfolio Analyzer"
    assert idea["language_code"] == project_gen_request_payload_dict["language_code"]
    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == os.getenv("PROJECT_LLM_MODEL_NAME")

    mock_llm_gen_ideas.assert_awaited_once()
    call_args = mock_llm_gen_ideas.call_args[1]
    assert "system_prompt" in call_args
    assert project_gen_request_payload_dict["language_code"] in call_args["system_prompt"]
    assert "user_query_or_context" in call_args
    assert project_gen_request_payload_dict["user_profile_snapshot"]["industry"] in call_args["user_query_or_context"]
    assert "response_schema" in call_args # Check that schema is being passed for JSON mode
    assert call_args["response_schema"]["type"] == "array"
    assert call_args["response_schema"]["items"]["title"] == GeneratedProjectIdea.model_json_schema()["title"]


@patch('uplas_ai_agents.project_generator_agent.main.project_idea_llm.generate_structured_response', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_returns_invalid_json(
    mock_llm_gen_ideas: AsyncMock,
    project_gen_request_payload_dict: Dict
):
    """Test when LLM returns a non-JSON string or malformed JSON for ideas."""
    mock_llm_gen_ideas.return_value = {"response_text": "This is not JSON, just plain text."}
    response = client.post("/v1/generate-project-ideas", json=project_gen_request_payload_dict)
    assert response.status_code == 500
    assert "llm returned invalid json for project ideas" in response.json()["detail"].lower()

    mock_llm_gen_ideas.return_value = {"response_text": "[{\"title\": \"Missing fields...\"}]"} # Malformed based on Pydantic
    response = client.post("/v1/generate-project-ideas", json=project_gen_request_payload_dict)
    assert response.status_code == 404 # Because validation will fail for all ideas
    assert "could not generate suitable project ideas after validation" in response.json()["detail"].lower()


@patch('uplas_ai_agents.project_generator_agent.main.project_idea_llm.generate_structured_response', new_callable=AsyncMock)
async def test_generate_project_ideas_llm_call_exception(
    mock_llm_gen_ideas: AsyncMock,
    project_gen_request_payload_dict: Dict
):
    """Test handling of an unexpected exception from the LLM client call for ideas."""
    mock_llm_gen_ideas.side_effect = Exception("Simulated LLM Network Outage")
    response = client.post("/v1/generate-project-ideas", json=project_gen_request_payload_dict)
    assert response.status_code == 503
    assert "error generating project ideas" in response.json()["detail"].lower()

# --- Test Cases for Project Assessment ---

@patch('uplas_ai_agents.project_generator_agent.main.assessment_llm.generate_structured_response', new_callable=AsyncMock)
@patch('uplas_ai_agents.project_generator_agent.main.trigger_ai_tutor_for_failed_assessment', new_callable=AsyncMock) # Mock the trigger function
async def test_assess_project_success_pass(
    mock_trigger_tutor: AsyncMock,
    mock_llm_assess: AsyncMock,
    project_assessment_request_payload_pass_dict: Dict,
    mock_successful_assessment_llm_response_text_pass: str
):
    """Test successful project assessment where user passes."""
    mock_llm_assess.return_value = {
        "response_text": mock_successful_assessment_llm_response_text_pass,
        "prompt_token_count": 200,
        "response_token_count": 150
    }

    response = client.post("/v1/assess-project", json=project_assessment_request_payload_pass_dict)

    assert response.status_code == 200
    data = response.json()
    assert "assessment_result" in data
    result = data["assessment_result"]
    assert result["score"] == 85.0
    assert result["is_passed"] is True
    assert result["language_code"] == project_assessment_request_payload_pass_dict["language_code"]
    assert result["tutor_session_triggered"] is False # Should not be triggered for a pass
    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == os.getenv("ASSESSMENT_LLM_MODEL_NAME")

    mock_llm_assess.assert_awaited_once()
    call_args = mock_llm_assess.call_args[1]
    assert "system_prompt" in call_args
    assert project_assessment_request_payload_pass_dict["language_code"] in call_args["system_prompt"]
    assert "user_query_or_context" in call_args
    assert project_assessment_request_payload_pass_dict["project_idea"]["title"] in call_args["user_query_or_context"]
    assert project_assessment_request_payload_pass_dict["submission"]["github_url"] in call_args["user_query_or_context"]
    assert "response_schema" in call_args
    assert call_args["response_schema"]["title"] == ProjectAssessmentResult.model_json_schema()["title"]
    
    mock_trigger_tutor.assert_not_awaited() # Ensure tutor was NOT called


@patch('uplas_ai_agents.project_generator_agent.main.assessment_llm.generate_structured_response', new_callable=AsyncMock)
@patch('uplas_ai_agents.project_generator_agent.main.trigger_ai_tutor_for_failed_assessment', new_callable=AsyncMock)
async def test_assess_project_success_fail_and_trigger_tutor(
    mock_trigger_tutor: AsyncMock,
    mock_llm_assess: AsyncMock,
    project_assessment_request_payload_fail_dict: Dict, # Using the fail payload
    mock_successful_assessment_llm_response_text_fail: str,
    user_profile_data_analyst: UserProfileSnapshotForProjects # To verify tutor call payload
):
    """Test successful project assessment where user fails, and AI Tutor is triggered."""
    mock_llm_assess.return_value = {
        "response_text": mock_successful_assessment_llm_response_text_fail
    }
    mock_trigger_tutor.return_value = True # Simulate successful tutor trigger

    response = client.post("/v1/assess-project", json=project_assessment_request_payload_fail_dict)

    assert response.status_code == 200
    data = response.json()
    result = data["assessment_result"]
    assert result["score"] == 60.0
    # is_passed is determined by the LLM's output in this mock, but should align with score
    # In main.py, is_passed is directly from LLM. Let's assume LLM sets it correctly.
    assert result["is_passed"] is False
    assert result["tutor_session_triggered"] is True

    mock_llm_assess.assert_awaited_once()
    mock_trigger_tutor.assert_awaited_once()
    
    # Verify arguments passed to trigger_ai_tutor
    tutor_call_args = mock_trigger_tutor.call_args[1] # kwargs
    assert tutor_call_args["user_id"] == project_assessment_request_payload_fail_dict["user_id"]
    assert tutor_call_args["project_title"] == project_assessment_request_payload_fail_dict["project_idea"]["title"]
    assert "Good start, but the core routing algorithm needs more work." in tutor_call_args["assessment_feedback"]
    assert tutor_call_args["language_code"] == project_assessment_request_payload_fail_dict["language_code"]
    # Check that a UserProfileSnapshotForProjects was passed (even if minimal from test setup)
    assert isinstance(tutor_call_args["user_profile"], UserProfileSnapshotForProjects)


@patch('uplas_ai_agents.project_generator_agent.main.assessment_llm.generate_structured_response', new_callable=AsyncMock)
async def test_assess_project_llm_returns_invalid_assessment_json(
    mock_llm_assess: AsyncMock,
    project_assessment_request_payload_pass_dict: Dict
):
    """Test when LLM returns malformed JSON for assessment."""
    mock_llm_assess.return_value = {"response_text": "Not a valid JSON assessment."}
    response = client.post("/v1/assess-project", json=project_assessment_request_payload_pass_dict)
    assert response.status_code == 500
    assert "llm returned invalid json for project assessment" in response.json()["detail"].lower()


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "ProjectGeneratorAgent"}

# To run these tests:
# From within uplas-ai-agents/project_generator_agent/
# pytest
