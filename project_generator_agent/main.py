# uplas-ai-agents/project_generator_agent/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator, HttpUrl
from typing import List, Dict, Optional, Any, Union
import os
import uuid
import time
import httpx # For calling AI Tutor Agent
import logging
import json # For parsing LLM JSON outputs

# GCP Clients
from google.cloud import aiplatform
import google.auth
import google.auth.transport.requests

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
PROJECT_LLM_MODEL_NAME = os.getenv("PROJECT_LLM_MODEL_NAME", "gemini-1.5-flash-001") # For project ideas
ASSESSMENT_LLM_MODEL_NAME = os.getenv("ASSESSMENT_LLM_MODEL_NAME", "gemini-1.5-flash-001") # For project assessment

AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL") # e.g., http://ai-tutor-agent-service/ or http://localhost:8001

# Initialize Vertex AI
if not GCP_PROJECT_ID:
    logging.warning("GCP_PROJECT_ID environment variable not set. LLM calls may fail.")
else:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

# Supported languages - align with other agents
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

ASSESSMENT_PASS_THRESHOLD = 75 # Percentage

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Contract (Updated) ---

class UserProfileSnapshotForProjects(BaseModel):
    user_id: str = Field(..., examples=["user_uuid_abc123"])
    industry: Optional[str] = Field(None, examples=["Finance"])
    profession: Optional[str] = Field(None, examples=["Data Analyst"])
    career_interest: Optional[str] = Field(None, examples=["Machine Learning Engineer"])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate"}])
    areas_of_interest: Optional[List[str]] = Field(default_factory=list, examples=[["NLP"]])
    learning_goals: Optional[str] = Field(None, examples=["Build a portfolio piece."])
    preferred_tutor_persona: Optional[str] = Field("Friendly and encouraging") # For AI Tutor call

class ProjectPreferences(BaseModel):
    difficulty_level: Optional[str] = Field("intermediate", examples=["beginner", "intermediate", "advanced"])
    preferred_technologies: Optional[List[str]] = Field(default_factory=list, examples=[["Python", "FastAPI"]])
    project_type_focus: Optional[str] = Field(None, examples=["Portfolio Piece"])
    time_commitment_hours_estimate: Optional[int] = Field(None, examples=[20], ge=5, le=100)

class ProjectIdeaGenerationRequest(BaseModel):
    user_profile_snapshot: UserProfileSnapshotForProjects
    preferences: Optional[ProjectPreferences] = Field(default_factory=ProjectPreferences)
    number_of_ideas: Optional[int] = Field(1, ge=1, le=3)
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)

    @validator('language_code')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' in ProjectIdeaGenerationRequest. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class GeneratedProjectTask(BaseModel):
    task_id: int = Field(..., ge=1)
    description: str = Field(..., min_length=10)
    estimated_sub_duration: Optional[str] = Field(None, examples=["2-4 hours"])

class GeneratedProjectIdea(BaseModel):
    project_idea_id: str = Field(default_factory=lambda: f"proj_idea_{uuid.uuid4().hex[:12]}")
    title: str
    subtitle: Optional[str] = None
    description_html: str
    difficulty_level: str
    estimated_duration: Optional[str] = None
    learning_objectives_html: List[str] = Field(min_items=1)
    requirements_html: Optional[List[str]] = Field(min_items=1)
    target_audience_html: Optional[List[str]] = None
    key_tasks: List[GeneratedProjectTask] = Field(min_items=1)
    suggested_technologies: List[str] = Field(min_items=1)
    personalization_rationale: Optional[str] = None
    potential_challenges: Optional[List[str]] = None
    # This is what the assessment LLM will use as a base.
    assessment_rubric_preview_html: List[str] = Field(min_items=1, description="Key criteria for successful completion.")
    real_world_application_examples: Optional[List[str]] = None
    language_code: str # Store the language this idea was generated in

class ProjectIdeaGenerationResponse(BaseModel):
    generated_ideas: List[GeneratedProjectIdea]
    debug_info: Optional[Dict[str, Any]] = None

# --- Models for Project Assessment ---
class ProjectSubmissionDetails(BaseModel):
    github_url: Optional[HttpUrl] = Field(None, description="URL to the user's project repository.")
    # Allow for direct code submission if GitHub URL is not available or for smaller snippets
    # For large codebases, GitHub URL is preferred.
    code_files: Optional[Dict[str, str]] = Field(None, description="Dictionary of filename: code_content.")
    textual_summary: Optional[str] = Field(None, description="User's summary of their approach and solution.")
    # Add any other fields that represent the user's submitted work

    @validator('textual_summary', always=True)
    def check_submission_content(cls, v, values):
        if not values.get('github_url') and not values.get('code_files') and not v:
            raise ValueError("At least one of github_url, code_files, or textual_summary must be provided for assessment.")
        return v

class ProjectAssessmentRequest(BaseModel):
    user_id: str
    project_idea: GeneratedProjectIdea # The original project idea details
    submission: ProjectSubmissionDetails
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    # user_profile_snapshot: UserProfileSnapshotForProjects # Needed for AI Tutor call if triggered

    @validator('language_code')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' in ProjectAssessmentRequest. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v
    
class ProjectAssessmentResult(BaseModel):
    assessment_id: str = Field(default_factory=lambda: f"assess_{uuid.uuid4().hex[:12]}")
    project_idea_id: str
    user_id: str
    score: float = Field(..., ge=0.0, le=100.0, description="Overall score from 0 to 100.")
    is_passed: bool
    feedback_summary_html: str
    detailed_feedback_points_html: List[str] = Field(default_factory=list)
    areas_for_improvement_html: List[str] = Field(default_factory=list)
    positive_points_html: List[str] = Field(default_factory=list)
    tutor_session_triggered: bool = False
    language_code: str # Language of the feedback

class ProjectAssessmentResponse(BaseModel):
    assessment_result: ProjectAssessmentResult
    debug_info: Optional[Dict[str, Any]] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Project Generator & Assessor (Vertex AI)",
    description="Generates personalized project ideas and assesses submissions using Google Cloud Vertex AI.",
    version="0.2.0"
)

# --- Vertex AI LLM Client Logic (Similar to AI Tutor's) ---
class VertexAILLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate_structured_response(
        self,
        system_prompt: str,
        user_query_or_context: str, # For project ideas, this is the main prompt; for assessment, it's the combined context
        max_output_tokens: int,
        temperature: float = 0.6, # Slightly lower for more factual/structured output
        top_p: float = 0.9,
        top_k: int = 30,
        response_schema: Optional[Dict[str, Any]] = None # For Gemini JSON mode
    ) -> Dict[str, Any]:
        if not GCP_PROJECT_ID:
            raise EnvironmentError("GCP_PROJECT_ID is not configured.")

        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

        model = GenerativeModel(
            self.model_name,
            system_instruction=[Part.from_text(system_prompt)] if system_prompt else None
        )
        
        generation_config = GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if response_schema:
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = response_schema


        llm_response = await model.generate_content_async(
            [Part.from_text(user_query_or_context)],
            generation_config=generation_config
            # Add safety_settings if needed
        )

        response_text = ""
        prompt_tokens = 0
        response_tokens = 0

        if llm_response.candidates:
            candidate = llm_response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = "".join([part.text for part in candidate.content.parts if part.text])
            
            if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
                prompt_tokens = llm_response.usage_metadata.prompt_token_count
                response_tokens = llm_response.usage_metadata.candidates_token_count
        
        # If JSON mode was used, response_text should be a JSON string.
        # The caller will be responsible for parsing it.
        return {
            "response_text": response_text,
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

project_idea_llm = VertexAILLMClient(model_name=PROJECT_LLM_MODEL_NAME)
assessment_llm = VertexAILLMClient(model_name=ASSESSMENT_LLM_MODEL_NAME)


# --- Prompt Construction ---
def construct_project_gen_prompt_system(language_code: str) -> str:
    return (
        "You are an AI expert in crafting personalized, real-world project ideas for learners of various skill levels. "
        "Your goal is to generate engaging and practical projects that help users build their portfolio and enhance specific skills. "
        f"IMPORTANT: All textual output (titles, descriptions, tasks, etc.) MUST be in the language: {language_code}. "
        "You will be provided with user profile details, project preferences, and the number of ideas to generate. "
        "Respond ONLY with a valid JSON list, where each item in the list is a JSON object strictly adhering to the provided schema for a project idea. "
        "Do not include any introductory or concluding text outside this JSON list."
    )

def construct_project_gen_prompt_user(
    user_profile: UserProfileSnapshotForProjects,
    preferences: ProjectPreferences,
    num_ideas: int,
    language_code: str # For placeholders in the schema example
) -> str:
    # Define the Pydantic model as a JSON schema for the LLM
    # This helps Gemini understand the desired output structure for JSON mode
    project_idea_schema = GeneratedProjectIdea.model_json_schema()

    # Simplified user prompt, focusing on the data and the request for JSON.
    # The detailed instructions about fields are now part of the schema.
    user_prompt_parts = [
        f"Please generate {num_ideas} distinct project idea(s) in {language_code} based on the following user profile and preferences.",
        "--- User Profile ---",
        json.dumps(user_profile.model_dump(exclude_none=True), indent=2),
        "--- Project Preferences ---",
        json.dumps(preferences.model_dump(exclude_none=True), indent=2),
        "\nEnsure each generated project idea strictly follows this JSON schema:",
        # json.dumps(project_idea_schema, indent=2), # The schema is passed to GenerationConfig
        "The 'language_code' field in each generated idea should be set to the target language of generation, which is: " + language_code
    ]
    return "\n".join(user_prompt_parts)


def construct_project_assessment_prompt_system(language_code: str) -> str:
    return (
        "You are an AI expert project assessor. Your task is to evaluate a user's project submission against the original project idea's requirements and rubric. "
        "Provide a score (0-100), determine if it passes (>=75%), and offer constructive, detailed feedback. "
        f"IMPORTANT: All your feedback (summary, detailed points, etc.) MUST be in the language: {language_code}. "
        "Respond ONLY with a single, valid JSON object strictly adhering to the provided schema for an assessment result. "
        "Do not include any introductory or concluding text outside this JSON object."
    )

def construct_project_assessment_prompt_user(
    original_project: GeneratedProjectIdea,
    submission: ProjectSubmissionDetails,
    language_code: str # For placeholders in schema
) -> str:
    # assessment_result_schema = ProjectAssessmentResult.model_json_schema() # Passed to GenerationConfig

    user_prompt_parts = [
        f"Please assess the following user project submission in {language_code}.",
        "--- Original Project Idea Description & Rubric ---",
        f"Title: {original_project.title}",
        f"Description: {original_project.description_html}",
        f"Key Tasks Expected: {json.dumps([task.description for task in original_project.key_tasks])}",
        f"Assessment Rubric Preview: {json.dumps(original_project.assessment_rubric_preview_html)}",
        "--- User's Project Submission ---",
        f"GitHub URL: {submission.github_url or 'Not provided'}",
        f"Submitted Code Files: {json.dumps(submission.code_files, indent=2) if submission.code_files else 'Not provided'}",
        f"User's Summary: {submission.textual_summary or 'Not provided'}",
        "\nBased on the above, provide your assessment. Consider completeness, correctness, code quality (if applicable), and adherence to the project's goals.",
        "The 'language_code' field in your assessment result JSON should be set to: " + language_code
    ]
    return "\n".join(user_prompt_parts)

# --- Helper to call AI Tutor ---
async def trigger_ai_tutor_for_failed_assessment(
    user_id: str,
    user_profile: UserProfileSnapshotForProjects, # Get this from the original assessment request or fetch
    project_title: str,
    assessment_feedback: str,
    language_code: str
) -> bool:
    if not AI_TUTOR_AGENT_URL:
        logger.warning("AI_TUTOR_AGENT_URL not set. Cannot trigger AI Tutor for failed assessment.")
        return False

    tutor_query = (
        f"I need help with my project '{project_title}'. I didn't pass the assessment. "
        f"The feedback I received was: '{assessment_feedback}'. Can you help me understand where I went wrong and how to improve?"
    )
    tutor_payload = {
        "user_id": user_id,
        "query_text": tutor_query,
        "user_profile_snapshot": { # Construct the snapshot for the tutor
            "industry": user_profile.industry,
            "profession": user_profile.profession,
            "country": None, # Add if available in UserProfileSnapshotForProjects
            "city": None, # Add if available
            "preferred_tutor_persona": user_profile.preferred_tutor_persona,
            "current_knowledge_level": user_profile.current_knowledge_level,
            "career_interest": user_profile.career_interest,
            "learning_goals": user_profile.learning_goals
        },
        "language_code": language_code,
        "context": {
            "current_project_title": project_title,
            "project_assessment_feedback": assessment_feedback # Pass the feedback directly
        }
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{AI_TUTOR_AGENT_URL}/v1/ask-tutor", json=tutor_payload)
            response.raise_for_status() # Will raise an exception for 4XX/5XX responses
            logger.info(f"Successfully triggered AI Tutor for user {user_id}, project '{project_title}'. Tutor response status: {response.status_code}")
            return True
    except Exception as e:
        logger.error(f"Failed to trigger AI Tutor for user {user_id}, project '{project_title}'. Error: {e}", exc_info=True)
        return False

# --- API Endpoints ---
@app.post("/v1/generate-project-ideas", response_model=ProjectIdeaGenerationResponse, summary="Generate Personalized Project Ideas")
async def generate_project_ideas_endpoint(request_data: ProjectIdeaGenerationRequest):
    processing_start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE

    system_prompt = construct_project_gen_prompt_system(effective_language_code)
    user_prompt = construct_project_gen_prompt_user(
        user_profile=request_data.user_profile_snapshot,
        preferences=request_data.preferences,
        num_ideas=request_data.number_of_ideas,
        language_code=effective_language_code
    )
    
    # Define the response schema for the LLM (for JSON mode)
    # This tells Gemini to structure its output as a list of GeneratedProjectIdea objects
    # We need to get the schema for the *items* in the list.
    list_of_ideas_schema = {
        "type": "array",
        "items": GeneratedProjectIdea.model_json_schema()
    }

    try:
        llm_response_data = await project_idea_llm.generate_structured_response(
            system_prompt=system_prompt,
            user_query_or_context=user_prompt,
            max_output_tokens=4096, # Allow more tokens for multiple ideas
            response_schema=list_of_ideas_schema
        )
        
        raw_ideas_json_str = llm_response_data.get("response_text")
        if not raw_ideas_json_str:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM returned no content for project ideas.")

        # Parse the JSON string from LLM
        try:
            raw_ideas_list = json.loads(raw_ideas_json_str)
            if not isinstance(raw_ideas_list, list):
                 raise ValueError("LLM response for project ideas is not a list.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM for project ideas: {e}. Response: {raw_ideas_json_str[:500]}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM returned invalid JSON for project ideas.")

        validated_ideas: List[GeneratedProjectIdea] = []
        for raw_idea_dict in raw_ideas_list:
            try:
                # Ensure language_code is set correctly from the generation context
                raw_idea_dict["language_code"] = effective_language_code
                validated_ideas.append(GeneratedProjectIdea(**raw_idea_dict))
            except Exception as e_val:
                logger.warning(f"Pydantic validation failed for a raw project idea: {raw_idea_dict}. Error: {e_val}")
                # Optionally skip this idea or collect errors

        if not validated_ideas:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Could not generate suitable project ideas after validation.")

    except EnvironmentError as e:
        logger.error(f"Configuration error for LLM: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI service configuration error: {e}")
    except Exception as e:
        logger.error(f"Error during project idea generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Error generating project ideas.")

    processing_end_time = time.perf_counter()
    debug_info = {
        "llm_model_name_used": project_idea_llm.model_name,
        "processing_time_ms": round((processing_end_time - processing_start_time) * 1000, 2),
        "prompt_token_count": llm_response_data.get("prompt_token_count"),
        "response_token_count": llm_response_data.get("response_token_count"),
        "language_generated": effective_language_code,
        "num_ideas_requested": request_data.number_of_ideas,
        "num_ideas_generated_valid": len(validated_ideas)
    }
    return ProjectIdeaGenerationResponse(generated_ideas=validated_ideas, debug_info=debug_info)


@app.post("/v1/assess-project", response_model=ProjectAssessmentResponse, summary="Assess a User's Project Submission")
async def assess_project_endpoint(request_data: ProjectAssessmentRequest):
    processing_start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE

    system_prompt = construct_project_assessment_prompt_system(effective_language_code)
    user_prompt = construct_project_assessment_prompt_user(
        original_project=request_data.project_idea,
        submission=request_data.submission,
        language_code=effective_language_code
    )
    
    assessment_result_schema = ProjectAssessmentResult.model_json_schema()

    try:
        llm_response_data = await assessment_llm.generate_structured_response(
            system_prompt=system_prompt,
            user_query_or_context=user_prompt,
            max_output_tokens=2048,
            response_schema=assessment_result_schema
        )

        raw_assessment_json_str = llm_response_data.get("response_text")
        if not raw_assessment_json_str:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM returned no content for project assessment.")

        try:
            assessment_data_dict = json.loads(raw_assessment_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM for assessment: {e}. Response: {raw_assessment_json_str[:500]}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LLM returned invalid JSON for project assessment.")
        
        # Ensure language_code is correctly set
        assessment_data_dict["language_code"] = effective_language_code
        assessment_result = ProjectAssessmentResult(**assessment_data_dict)

    except EnvironmentError as e:
        logger.error(f"Configuration error for LLM: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI service configuration error: {e}")
    except Exception as e:
        logger.error(f"Error during project assessment: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Error assessing project.")

    # Trigger AI Tutor if assessment score is below threshold
    tutor_triggered = False
    if not assessment_result.is_passed or assessment_result.score < ASSESSMENT_PASS_THRESHOLD:
        # Fetch user profile for tutor call - this is a simplification.
        # In a real app, you'd fetch the full UserProfileSnapshot based on request_data.user_id
        # For now, creating a minimal one.
        user_profile_for_tutor = UserProfileSnapshotForProjects(
            user_id=request_data.user_id,
            # Try to get more details if they were part of an initial request or stored with the project
            # For this example, we'll assume we don't have them readily available for the tutor call
            # and the tutor will have to work with what's in the assessment feedback.
            # Ideally, the original UserProfileSnapshotForProjects used for idea generation
            # would be passed along or fetched.
        )
        
        # Ensure AI_TUTOR_AGENT_URL is configured
        if AI_TUTOR_AGENT_URL:
            tutor_triggered = await trigger_ai_tutor_for_failed_assessment(
                user_id=request_data.user_id,
                user_profile=user_profile_for_tutor, # Pass the fetched/constructed profile
                project_title=request_data.project_idea.title,
                assessment_feedback=assessment_result.feedback_summary_html + " Areas for improvement: " + "; ".join(assessment_result.areas_for_improvement_html),
                language_code=effective_language_code
            )
            assessment_result.tutor_session_triggered = tutor_triggered
        else:
            logger.warning("AI_TUTOR_AGENT_URL not configured. Cannot trigger tutor for failed assessment.")


    processing_end_time = time.perf_counter()
    debug_info = {
        "llm_model_name_used": assessment_llm.model_name,
        "processing_time_ms": round((processing_end_time - processing_start_time) * 1000, 2),
        "prompt_token_count": llm_response_data.get("prompt_token_count"),
        "response_token_count": llm_response_data.get("response_token_count"),
        "language_assessed_in": effective_language_code
    }
    return ProjectAssessmentResponse(assessment_result=assessment_result, debug_info=debug_info)

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if not GCP_PROJECT_ID:
        return {"status": "unhealthy", "reason": "GCP_PROJECT_ID not configured.", "service": "ProjectGeneratorAgent"}
    # Add check for AI_TUTOR_AGENT_URL if assessment triggering is critical
    return {"status": "healthy", "service": "ProjectGeneratorAgent"}

if __name__ == "__main__":
    import uvicorn
    if not GCP_PROJECT_ID:
        print("Warning: GCP_PROJECT_ID is not set. Please set this environment variable.")
    if not AI_TUTOR_AGENT_URL:
        print("Warning: AI_TUTOR_AGENT_URL is not set. Tutor triggering for failed assessments will not work.")
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8004)))

# To run locally:
# GCP_PROJECT_ID="your-gcp-project" AI_TUTOR_AGENT_URL="http://localhost:8001" uvicorn main:app --reload --port 8004

