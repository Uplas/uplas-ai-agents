# uplas-ai-agents/project_generator_agent/test_main.py
import pytest
import pytest_asyncio # For async fixtures
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import os
import uuid
import json
import httpx # For mocking calls to AI Tutor (though we mock the helper function directly)

# Import the FastAPI app and Pydantic models from our rewritten main.py
from .main import (
    app,
    # Generation Models
    ProjectIdeaGenerationRequest, UserProfileSnapshotForProjects, ProjectPreferences,
    GeneratedProjectIdea, GeneratedProjectTask, ProjectGenerationResponse,
    # Assessment Models
    ProjectAssessmentRequest, ProjectSubmissionContentItem, ProjectSubmissionContentType,
    ProjectAssessmentResponse, ProjectAssessmentResult, AssessmentFeedbackPoint, # Renamed from main (5)
    # Config
    COMPETENCY_THRESHOLD, DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES,
    PROJECT_LLM_MODEL_NAME, ASSESSMENT_LLM_MODEL_NAME, AI_TUTOR_AGENT_URL
    # LLM clients (project_idea_llm, assessment_llm) will be patched where used.
)

# Set environment variables for testing (as in original)
#
os.environ["GCP_PROJECT_ID"] = "test-gcp-project-id-pg"
os.environ["GCP_LOCATION"] = "test-gcp-location-pg"
os.environ["PROJECT_LLM_MODEL_NAME"] = "test-project-gemini-model-pg"
os.environ["ASSESSMENT_LLM_MODEL_NAME"] = "test-assessment-gemini-model-pg"
os.environ["AI_TUTOR_AGENT_URL"] = "http://mock-ai-tutor-agent-innovate.com"
os.environ["PROJECT_COMPETENCY_THRESHOLD"] = "0.75" # Ensure consistency

client = TestClient(app)

# --- Fixtures for Project Idea Generation (InnovateAI Refined) ---
@pytest.fixture #
def user_profile_innovate() -> UserProfileSnapshotForProjects:
    return UserProfileSnapshotForProjects(
        user_id=f"user_innovate_{uuid.uuid4().hex[:6]}", # From rewritten main.py model
        industry="Sustainable Energy",
        profession="Environmental Engineer",
        career_interest="AI for Green Tech",
        current_knowledge_level={"Python": "Advanced", "Data Analysis": "Intermediate", "Cloud Deployment": "Beginner"},
        areas_of_interest=["renewable_energy_optimization", "smart_grid_ai"],
        learning_goals="Develop a capstone project for my AI in Sustainability portfolio.",
        preferred_tutor_persona="Insightful and challenging"
    )

@pytest.fixture #
def project_prefs_innovate() -> ProjectPreferences:
    return ProjectPreferences(
        difficulty_level="advanced",
        preferred_technologies=["Python", "TensorFlow", "GCP Vertex AI Pipelines"],
        project_type_focus="End-to-end AI solution with deployment considerations",
        time_commitment_hours_estimate=60
    )

@pytest.fixture #
def project_gen_request_dict(
    user_profile_innovate: UserProfileSnapshotForProjects,
    project_prefs_innovate: ProjectPreferences
) -> Dict:
    return {
        "user_profile_snapshot": user_profile_innovate.model_dump(),
        "preferences": project_prefs_innovate.model_dump(), # From main (5)
        "course_context_summary": "Advanced AI applications in environmental science.",
        "topic_focus_keywords": ["predictive_modeling", "iot_sensor_data"],
        "number_of_ideas": 1,
        "language_code": "en-US"
    }

@pytest.fixture #
def mock_successful_generated_project_idea_llm_response() -> str:
    # InnovateAI: LLM returns a dict with a "generated_projects" list (as per prompt in rewritten main.py)
    project_idea = GeneratedProjectIdea(
        project_id="proj_innovate_gen_1", # Will be auto-generated in real scenario
        title="AI-Optimized Solar Panel Array Placement",
        subtitle="Using geospatial data and predictive modeling to maximize energy yield.",
        description_html="<p>Develop a system that analyzes terrain, weather patterns, and insolation data to recommend optimal placement for solar panel arrays in a given region.</p>",
        difficulty_level="Advanced",
        estimated_duration_hours=60,
        learning_objectives_html=["<li>Apply machine learning for predictive tasks.</li>", "<li>Integrate external geospatial APIs.</li>", "<li>Develop a clear technical proposal.</li>"],
        expected_deliverables_html=["<li>A Python script implementing the predictive model.</li>", "<li>A documented Jupyter Notebook showcasing data analysis.</li>", "<li>A 5-page PDF proposal detailing the methodology and potential impact.</li>"],
        key_tasks=[
            GeneratedProjectTask(task_id="t1_innovate", title="Data Acquisition & Preprocessing", description="Collect and clean relevant geospatial and weather data."),
            GeneratedProjectTask(task_id="t2_innovate", title="Model Development", description="Train a predictive model for solar energy yield."),
            GeneratedProjectTask(task_id="t3_innovate", title="Proposal Writing", description="Document findings and recommendations.")
        ],
        suggested_tools_technologies=["Python (GeoPandas, Scikit-learn, TensorFlow/Keras)", "GCP (Vertex AI, Google Earth Engine API)"],
        assessment_rubric_html=[
            "<li>Accuracy and innovativeness of the predictive model (40%)</li>",
            "<li>Quality of data analysis and visualization (25%)</li>",
            "<li>Clarity, feasibility, and impact outlined in the proposal (25%)</li>",
            "<li>Code quality and documentation (10%)</li>"
        ],
        personalization_rationale="Aligns with interest in AI for Green Tech and Python/Data Analysis skills.",
        real_world_application_examples_html=["<p>Used by energy companies for site selection.</p>"],
        language_code="en-US"
    ).model_dump()
    return json.dumps({"generated_projects": [project_idea]})


# --- Fixtures for Project Assessment (InnovateAI Refined) ---
@pytest.fixture #
def sample_original_project_for_assessment() -> GeneratedProjectIdea:
    # This should match the structure of what's stored after generation
    return GeneratedProjectIdea(
        project_id="proj_assess_innovate_001",
        title="AI-Powered Smart Irrigation System Design",
        subtitle="Conceptual design for an IoT and AI based irrigation system.",
        description_html="<p>Design a system using soil moisture sensors, weather forecasts, and AI to optimize water usage for agriculture.</p>",
        difficulty_level="Intermediate",
        learning_objectives_html=["<li>Understand IoT data integration.</li>", "<li>Design a basic predictive control algorithm.</li>", "<li>Document system architecture.</li>"],
        expected_deliverables_html=["<li>System architecture diagram (PDF).</li>", "<li>Pseudo-code for the control algorithm.</li>", "<li>A 3-page report detailing sensor choices, data flow, and AI model rationale.</li>"],
        key_tasks=[GeneratedProjectTask(task_id="task_assess_1", title="Sensor Research", description="...")], # Simplified
        assessment_rubric_html=[ # Crucial for assessment prompt
            "<li>Clarity and completeness of architecture diagram (30%)</li>",
            "<li>Logic and feasibility of the control algorithm pseudo-code (40%)</li>",
            "<li>Rationale and depth of the report (30%)</li>"
        ],
        language_code="en-US"
    )

@pytest.fixture # Based on (ProjectSubmissionDetails -> List[ProjectSubmissionContentItem])
def project_submission_items_pass() -> List[Dict]: # Returns dicts to match request model
    return [
        ProjectSubmissionContentItem(content_type=ProjectSubmissionContentType.GCS_URL_PDF, value="gs://mybucket/arch_diagram_final.pdf", filename="architecture_v2.pdf").model_dump(),
        ProjectSubmissionContentItem(content_type=ProjectSubmissionContentType.TEXT_REPORT, value="Control Algorithm Pseudo-code:\nIF soil_moisture < LOW_THRESHOLD AND no_rain_forecasted THEN\n  ACTIVATE_IRRIGATION_ZONE_A\nENDIF\nReport details attached...", filename="report_and_pseudo.txt").model_dump()
    ]

@pytest.fixture # Similar to above
def project_submission_items_fail() -> List[Dict]:
    return [
        ProjectSubmissionContentItem(content_type=ProjectSubmissionContentType.TEXT_REPORT, value="Only did the architecture diagram description. Ran out of time for algorithm and report. Sensor choices were hard.", filename="submission_partial.txt").model_dump()
    ]

@pytest.fixture # (adapted to new request model)
def project_assessment_request_dict_pass(
    user_profile_innovate: UserProfileSnapshotForProjects,
    project_submission_items_pass: List[Dict]
) -> Dict:
    return {
        "user_id": user_profile_innovate.user_id,
        "project_id": "proj_assess_innovate_001", # Matches sample_original_project_for_assessment
        "submission_items": project_submission_items_pass,
        "user_profile_snapshot": user_profile_innovate.model_dump(),
        "language_code": "en-US"
    }

@pytest.fixture # Similar to above
def project_assessment_request_dict_fail(
    user_profile_innovate: UserProfileSnapshotForProjects,
    project_submission_items_fail: List[Dict]
) -> Dict:
    return {
        "user_id": user_profile_innovate.user_id,
        "project_id": "proj_assess_innovate_001",
        "submission_items": project_submission_items_fail,
        "user_profile_snapshot": user_profile_innovate.model_dump(),
        "language_code": "en-US"
    }

@pytest.fixture # (adapted to new LLMAssessmentRoot from main.py)
def mock_llm_assessment_response_pass_json_str() -> str:
    # This is what the LLM is prompted to return (subset of ProjectAssessmentResult)
    data = {
        "overall_competency_score": 0.85, # Above COMPETENCY_THRESHOLD
        "feedback_summary_html": "<p>Excellent submission! The architecture is clear and the algorithm logic is sound.</p>",
        "detailed_feedback_points": [
            AssessmentFeedbackPoint(aspect_evaluated="Architecture Diagram Clarity", score_achieved=0.9, observation_feedback_html="<p>Diagram is very clear and well-labeled.</p>", is_strength=True).model_dump(),
            AssessmentFeedbackPoint(aspect_evaluated="Control Algorithm Logic", score_achieved=0.8, observation_feedback_html="<p>Algorithm is logical and covers main cases. Consider adding drought-mode.</p>", is_strength=True).model_dump()
        ],
        "skills_demonstrated": ["System Design", "Algorithmic Thinking", "Technical Documentation"],
        "critical_areas_for_improvement_html": ["<li>Consider scalability for larger farms.</li>"],
        "positive_points_highlighted_html": ["<li>Thorough sensor rationale in report.</li>", "<li>Well-structured pseudo-code.</li>"]
    }
    return json.dumps(data)

@pytest.fixture # Similar to above
def mock_llm_assessment_response_fail_json_str() -> str:
    data = {
        "overall_competency_score": 0.50, # Below COMPETENCY_THRESHOLD
        "feedback_summary_html": "<p>A good starting point, but key components like the control algorithm and detailed report are missing or incomplete.</p>",
        "detailed_feedback_points": [
            AssessmentFeedbackPoint(aspect_evaluated="Architecture Diagram Clarity", score_achieved=0.7, observation_feedback_html="<p>Diagram description is okay, but an actual diagram was expected.</p>", is_strength=False).model_dump(),
            AssessmentFeedbackPoint(aspect_evaluated="Control Algorithm Logic", score_achieved=0.3, observation_feedback_html="<p>Algorithm is not provided or is very rudimentary. This was a core part.</p>", is_strength=False).model_dump()
        ],
        "skills_demonstrated": ["Basic requirements understanding"],
        "critical_areas_for_improvement_html": ["<li>Complete the control algorithm design.</li>", "<li>Write the detailed technical report as per deliverables.</li>", "<li>Ensure all expected deliverables are submitted.</li>"],
        "positive_points_highlighted_html": ["<li>Good initial thoughts on sensor types.</li>"]
    }
    return json.dumps(data)


# --- Centralized Mocking for LLM clients (InnovateAI Enhancement) ---
@pytest_asyncio.fixture
async def mock_llm_clients(monkeypatch):
    """Mocks both project_idea_llm and assessment_llm clients and their methods."""
    mock_project_idea_llm_client = AsyncMock(spec=VertexAILLMClient)
    mock_project_idea_llm_client.generate_structured_response = AsyncMock()
    
    mock_assessment_llm_client = AsyncMock(spec=VertexAILLMClient)
    mock_assessment_llm_client.generate_structured_response = AsyncMock()

    monkeypatch.setattr("project_generator_agent.main.project_idea_llm", mock_project_idea_llm_client)
    monkeypatch.setattr("project_generator_agent.main.assessment_llm", mock_assessment_llm_client)
    
    return {
        "project_idea_llm": mock_project_idea_llm_client,
        "assessment_llm": mock_assessment_llm_client
    }

# --- Test Cases for Project Idea Generation (InnovateAI Refined) ---

async def test_generate_project_ideas_success( # Based on
    mock_llm_clients: Dict[str, AsyncMock],
    project_gen_request_dict: Dict,
    mock_successful_generated_project_idea_llm_response: str # This is now a JSON string
):
    """InnovateAI Test: Successful project idea generation with structured LLM response."""
    mock_llm_clients["project_idea_llm"].generate_structured_response.return_value = {
        "raw_json_response": mock_successful_generated_project_idea_llm_response, # LLM client returns this structure
        "prompt_token_count": 150, "response_token_count": 400
    }

    response = client.post("/v1/generate-project-ideas", json=project_gen_request_dict)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "generated_projects" in data
    assert len(data["generated_projects"]) == 1
    project = data["generated_projects"][0]
    assert project["title"] == "AI-Optimized Solar Panel Array Placement" # From mock
    assert project["language_code"] == project_gen_request_dict["language_code"]
    assert "assessment_rubric_html" in project # Crucial field
    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == PROJECT_LLM_MODEL_NAME

    mock_llm_clients["project_idea_llm"].generate_structured_response.assert_awaited_once()
    call_args = mock_llm_clients["project_idea_llm"].generate_structured_response.call_args[1] # kwargs
    assert "system_prompt" in call_args
    assert "user_query_or_context" in call_args
    assert "pydantic_model_for_schema" in call_args # Ensure schema for JSON mode is passed
    assert call_args["pydantic_model_for_schema"].__name__ == "LLMProjectGenRoot" # Check correct Pydantic model for schema


async def test_generate_project_ideas_llm_returns_invalid_json_structure( # Based on
    mock_llm_clients: Dict[str, AsyncMock],
    project_gen_request_dict: Dict
):
    """InnovateAI Test: LLM returns JSON that doesn't match expected Pydantic schema for project ideas."""
    mock_llm_clients["project_idea_llm"].generate_structured_response.return_value = {
        "raw_json_response": json.dumps({"generated_projects": [{"title": "Only title, missing other fields"}]})
    }
    response = client.post("/v1/generate-project-ideas", json=project_gen_request_dict)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR # Due to Pydantic validation error
    assert "Failed to generate or parse project ideas" in response.json()["detail"]


async def test_generate_project_ideas_llm_call_exception( # Based on
    mock_llm_clients: Dict[str, AsyncMock],
    project_gen_request_dict: Dict
):
    """InnovateAI Test: Handling of an unexpected exception from the LLM client for project generation."""
    mock_llm_clients["project_idea_llm"].generate_structured_response.side_effect = Exception("Simulated InnovateAI LLM Network Outage")
    response = client.post("/v1/generate-project-ideas", json=project_gen_request_dict)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR # Changed from 503 to align with current generic catch
    assert "Project idea generation failed unexpectedly" in response.json()["detail"]


# --- Test Cases for Project Assessment (InnovateAI New & Refined) ---

@patch('project_generator_agent.main.trigger_ai_tutor_for_failed_assessment', new_callable=AsyncMock) # Mock the trigger helper
async def test_assess_project_submission_pass( # Based on
    mock_trigger_tutor: AsyncMock,
    mock_llm_clients: Dict[str, AsyncMock],
    project_assessment_request_dict_pass: dict,
    sample_original_project_for_assessment: GeneratedProjectIdea, # Used for mocking DB fetch
    mock_llm_assessment_response_pass_json_str: str
):
    """InnovateAI Test: Successful project assessment where user passes."""
    mock_llm_clients["assessment_llm"].generate_structured_response.return_value = {
        "raw_json_response": mock_llm_assessment_response_pass_json_str,
        "prompt_token_count": 300, "response_token_count": 250
    }
    # Mock the conceptual DB fetch of original project details
    with patch('project_generator_agent.main.fetch_project_details_from_db', new_callable=AsyncMock) as mock_fetch_db: # Assuming you'd create this helper
        mock_fetch_db.return_value = sample_original_project_for_assessment
        
        response = client.post("/v1/assess-project-submission", json=project_assessment_request_dict_pass)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "assessment_result" in data
    result = data["assessment_result"]
    
    assert result["overall_competency_score"] == 0.85
    assert result["is_passed"] is True # Based on COMPETENCY_THRESHOLD = 0.75
    assert result["language_code"] == project_assessment_request_dict_pass["language_code"]
    assert result["tutor_session_triggered"] is False
    assert "Excellent submission!" in result["feedback_summary_html"]
    assert len(result["detailed_feedback_points"]) > 0
    assert "System Design" in result["skills_demonstrated"]
    assert "debug_info" in data
    assert data["debug_info"]["llm_model_name_used"] == ASSESSMENT_LLM_MODEL_NAME

    mock_llm_clients["assessment_llm"].generate_structured_response.assert_awaited_once()
    call_args = mock_llm_clients["assessment_llm"].generate_structured_response.call_args[1]
    assert "system_prompt" in call_args
    assert "user_query_or_context" in call_args
    assert sample_original_project_for_assessment.title in call_args["user_query_or_context"] # Original project details in prompt
    assert project_assessment_request_dict_pass["submission_items"][0]["value"] in call_args["user_query_or_context"] # Submission details in prompt
    assert "pydantic_model_for_schema" in call_args
    assert call_args["pydantic_model_for_schema"].__name__ == "LLMAssessmentRoot" # Check correct Pydantic model

    mock_trigger_tutor.assert_not_awaited()


@patch('project_generator_agent.main.trigger_ai_tutor_for_failed_assessment', new_callable=AsyncMock)
async def test_assess_project_submission_fail_and_trigger_tutor( # Based on
    mock_trigger_tutor: AsyncMock,
    mock_llm_clients: Dict[str, AsyncMock],
    project_assessment_request_dict_fail: dict,
    sample_original_project_for_assessment: GeneratedProjectIdea,
    mock_llm_assessment_response_fail_json_str: str,
    user_profile_innovate: UserProfileSnapshotForProjects # To verify tutor call payload
):
    """InnovateAI Test: Project assessment where user fails, and AI Tutor IS triggered."""
    mock_llm_clients["assessment_llm"].generate_structured_response.return_value = {
        "raw_json_response": mock_llm_assessment_response_fail_json_str
    }
    mock_trigger_tutor.return_value = True # Simulate successful tutor trigger

    with patch('project_generator_agent.main.fetch_project_details_from_db', new_callable=AsyncMock) as mock_fetch_db:
        mock_fetch_db.return_value = sample_original_project_for_assessment
        response = client.post("/v1/assess-project-submission", json=project_assessment_request_dict_fail)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    result = data["assessment_result"]
    assert result["overall_competency_score"] == 0.50
    assert result["is_passed"] is False
    assert result["tutor_session_triggered"] is True # Should be triggered

    mock_llm_clients["assessment_llm"].generate_structured_response.assert_awaited_once()
    mock_trigger_tutor.assert_awaited_once()
    
    # Verify arguments passed to trigger_ai_tutor_for_failed_assessment
    tutor_call_args = mock_trigger_tutor.call_args.kwargs # Direct access to kwargs
    assert tutor_call_args["user_profile"].user_id == project_assessment_request_dict_fail["user_profile_snapshot"]["user_id"]
    assert tutor_call_args["project_title"] == sample_original_project_for_assessment.title
    assert "Good start, but key components" in tutor_call_args["assessment_summary_html"]
    assert "Complete the control algorithm design." in tutor_call_args["areas_for_improvement_html"][0]
    assert tutor_call_args["language_code"] == project_assessment_request_dict_fail["language_code"]


async def test_assess_project_original_project_not_found(
    mock_llm_clients: Dict[str, AsyncMock], # LLM not called
    project_assessment_request_dict_pass: dict
):
    """InnovateAI Test: Assessment fails if original project definition cannot be fetched."""
    with patch('project_generator_agent.main.fetch_project_details_from_db', new_callable=AsyncMock) as mock_fetch_db:
        mock_fetch_db.return_value = None # Simulate project not found
        response = client.post("/v1/assess-project-submission", json=project_assessment_request_dict_pass)
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Original project definition not found" in response.json()["detail"]
    mock_llm_clients["assessment_llm"].generate_structured_response.assert_not_awaited()


@patch('project_generator_agent.main.fetch_project_details_from_db', new_callable=AsyncMock) # Mock DB fetch
async def test_assess_project_llm_returns_invalid_assessment_json( # Based on
    mock_fetch_db: AsyncMock,
    mock_llm_clients: Dict[str, AsyncMock],
    project_assessment_request_dict_pass: dict,
    sample_original_project_for_assessment: GeneratedProjectIdea
):
    """InnovateAI Test: Assessment LLM returns malformed JSON."""
    mock_fetch_db.return_value = sample_original_project_for_assessment
    mock_llm_clients["assessment_llm"].generate_structured_response.return_value = {
        "raw_json_response": "This is definitely not the JSON I was expecting for assessment."
    }
    response = client.post("/v1/assess-project-submission", json=project_assessment_request_dict_pass)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Failed to parse or validate assessment data from AI" in response.json()["detail"]


def test_health_check_endpoint_project_agent(): #
    """InnovateAI Test: Health check for the combined Project Generation & Assessment Agent."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "status": "healthy",
        "service": "ProjectGenerationAssessmentAgent_InnovateAI", # Check updated service name
        "innovate_ai_enhancements_active": True
    }

# InnovateAI Note: Remember to add tests for:
# - Different types of ProjectSubmissionContentItem (e.g., code strings, GCS URLs).
# - Edge cases in LLM responses for assessment (e.g., empty feedback points).
# - Failure of the trigger_ai_tutor_for_failed_assessment helper itself.
