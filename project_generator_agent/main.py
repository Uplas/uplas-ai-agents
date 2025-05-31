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
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") #
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") #
PROJECT_LLM_MODEL_NAME = os.getenv("PROJECT_LLM_MODEL_NAME", "gemini-1.5-flash-001") # For project ideas
ASSESSMENT_LLM_MODEL_NAME = os.getenv("ASSESSMENT_LLM_MODEL_NAME", "gemini-1.5-pro-001") # Potentially a more robust model for assessment

AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL") # e.g., http://ai-tutor-agent-service/ or http://localhost:8001

# InnovateAI Enhancement: Define a threshold for competency (0.0 to 1.0 scale)
COMPETENCY_THRESHOLD = float(os.getenv("PROJECT_COMPETENCY_THRESHOLD", "0.75"))

# Initialize Vertex AI
if not GCP_PROJECT_ID: #
    logging.warning("InnovateAI Warning: GCP_PROJECT_ID environment variable not set for Project Agent. LLM calls may fail.")
else:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION) #

SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] #
DEFAULT_LANGUAGE = "en-US" #

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO) #
logger = logging.getLogger(__name__) #

# --- Pydantic Models (InnovateAI Refined & Expanded from main (5).py) ---

class UserProfileSnapshotForProjects(BaseModel): # Based on
    user_id: str = Field(..., examples=["user_uuid_abc123"])
    industry: Optional[str] = Field(None, examples=["Fintech", "Healthcare", "Education Technology"])
    profession: Optional[str] = Field(None, examples=["Data Analyst", "Software Developer", "UX Designer"])
    career_interest: Optional[str] = Field(None, examples=["Machine Learning Engineer", "Cloud Solutions Architect", "Cybersecurity Analyst"])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "SQL": "Beginner", "Cloud Functions": "Novice"}])
    areas_of_interest: Optional[List[str]] = Field(default_factory=list, examples=[["natural_language_processing", "sustainable_technology_solutions"]])
    learning_goals: Optional[str] = Field(None, examples=["Develop a strong portfolio project for job applications.", "Master practical AI implementation."])
    # InnovateAI Addition: For personalizing feedback and AI Tutor interaction style
    preferred_tutor_persona: Optional[str] = Field("Encouraging and constructive", examples=["Direct and technical", "Socratic and guiding"])


class ProjectPreferences(BaseModel): # Based on
    difficulty_level: Optional[str] = Field("intermediate", examples=["beginner", "intermediate", "advanced"])
    preferred_technologies: Optional[List[str]] = Field(default_factory=list, examples=[["Python", "FastAPI", "Google Cloud Vision API"]])
    project_type_focus: Optional[str] = Field(None, examples=["Portfolio Piece demonstrating full-stack skills", "Research-oriented analysis", "Creative design challenge"])
    time_commitment_hours_estimate: Optional[int] = Field(None, examples=[20, 40], ge=5, le=150)


class ProjectIdeaGenerationRequest(BaseModel): # Based on
    user_profile_snapshot: UserProfileSnapshotForProjects
    preferences: Optional[ProjectPreferences] = Field(default_factory=ProjectPreferences)
    course_context_summary: Optional[str] = Field(None, description="Brief summary of the course or module for which the project is being generated.", examples=["A course on building AI-powered web applications."])
    topic_focus_keywords: Optional[List[str]] = Field(default_factory=list, description="Specific keywords or concepts from the course to focus the project on.", examples=["image_recognition", "user_authentication"])
    number_of_ideas: Optional[int] = Field(1, ge=1, le=3, description="Number of distinct project ideas to generate.")
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)

    @validator('language_code') #
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"InnovateAI: Unsupported language_code '{v}' in ProjectIdeaGenerationRequest. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class GeneratedProjectTask(BaseModel): # Based on
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:6]}" )
    title: str = Field(..., min_length=5, description="Short title for the task.")
    description: str = Field(..., min_length=20, description="Detailed description of what the task involves.")
    estimated_sub_duration_hours: Optional[float] = Field(None, examples=[2.5, 4.0], ge=0.5)


# InnovateAI Refined: GeneratedProjectIdea (this is what one generated project looks like)
class GeneratedProjectIdea(BaseModel): # Refined from
    project_id: str = Field(default_factory=lambda: f"proj_{uuid.uuid4().hex[:12]}", description="Unique ID for this generated project.")
    title: str = Field(..., examples=["AI-Powered Customer Feedback Analyzer for E-commerce"])
    subtitle: Optional[str] = Field(None, examples=["Leveraging NLP to categorize and prioritize user reviews."])
    description_html: str = Field(..., description="Engaging real-world scenario or problem statement (can include simple HTML for formatting).")
    difficulty_level: str = Field(..., examples=["Beginner", "Intermediate", "Advanced"])
    estimated_duration_hours: Optional[int] = Field(None, examples=[25, 50])
    learning_objectives_html: List[str] = Field(..., min_length=2, description="What the user will learn/demonstrate (can include simple HTML).")
    # InnovateAI Addition: Explicitly list expected deliverables
    expected_deliverables_html: List[str] = Field(..., min_length=1, description="Specific items the user must produce (e.g., 'A Python script performing sentiment analysis', 'A summary report in PDF format').")
    key_tasks: List[GeneratedProjectTask] = Field(..., min_length=2)
    suggested_tools_technologies: List[str] = Field(default_factory=list, examples=["Python (NLTK, Scikit-learn)", "Flask/FastAPI", "GCP Natural Language API"])
    # This is crucial for the assessment phase
    assessment_rubric_html: List[str] = Field(..., min_length=2, description="Key criteria for successful completion and how it will be judged (can include simple HTML). E.g., 'Functionality of the NLP model (40%)', 'Clarity of the report (30%)'.")
    personalization_rationale: Optional[str] = Field(None, description="Brief explanation of why this project is suitable for the user's profile.")
    potential_challenges_html: Optional[List[str]] = Field(default_factory=list, description="Possible hurdles or complexities the user might encounter.")
    real_world_application_examples_html: Optional[List[str]] = Field(default_factory=list, description="Examples of how this project type is used in industry.")
    language_code: str # Store the language this idea was generated in


class ProjectIdeaGenerationResponse(BaseModel): # Based on
    generated_projects: List[GeneratedProjectIdea] # Changed from generated_ideas
    debug_info: Optional[Dict[str, Any]] = None

# --- InnovateAI New Pydantic Models for Project Assessment ---

class ProjectSubmissionContentType(str, Enum): # Same as previous definition
    TEXT_REPORT = "text_report"
    PYTHON_CODE_STRING = "python_code_string"
    MARKDOWN_DOCUMENT = "markdown_document"
    GCS_URL_ZIP = "gcs_url_zip_file"
    GCS_URL_PDF = "gcs_url_pdf_document"
    GITHUB_URL = "github_url" # More specific than OTHER_URL
    OTHER_URL_GENERAL = "other_url_general"

class ProjectSubmissionContentItem(BaseModel): # Same as previous definition
    content_type: ProjectSubmissionContentType
    value: str = Field(..., description="The actual content (e.g. code string, text report) or a fully qualified URL.")
    filename: Optional[str] = Field(None, description="Original filename if applicable (e.g., for GCS uploads or when content_type is code_string).")
    notes: Optional[str] = Field(None, description="Optional user notes about this specific submission item.")


class ProjectAssessmentRequest(BaseModel): # Refined from
    user_id: str = Field(..., examples=["user_assess_xyz"])
    project_id: str = Field(..., description="The ID of the Uplas-generated project being submitted against.") # Renamed from project_idea_id for clarity
    submission_items: List[ProjectSubmissionContentItem] = Field(..., min_length=1, description="List of submitted content items.")
    # InnovateAI: User profile is now part of the request for personalizing feedback
    user_profile_snapshot: UserProfileSnapshotForProjects
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)

    @validator('language_code') #
    def validate_language_code(cls, v):
        # ... (existing validation logic)
        if v not in SUPPORTED_LANGUAGES: return DEFAULT_LANGUAGE
        return v

class AssessmentFeedbackPoint(BaseModel): # Same as previous definition
    aspect_evaluated: str = Field(..., examples=["Algorithm Correctness", "Report Clarity", "Code Readability"])
    score_achieved: float = Field(..., ge=0.0, le=1.0, description="Normalized score for this aspect (0.0-1.0).")
    observation_feedback_html: str = Field(..., description="Specific feedback (strength or area for improvement), can use simple HTML.")
    is_strength: Optional[bool] = Field(None, description="True if this point is primarily a positive observation.")

class ProjectAssessmentResult(BaseModel): # Refined from
    assessment_id: str = Field(default_factory=lambda: f"assess_{uuid.uuid4().hex[:12]}")
    project_id: str # Links back to the GeneratedProjectIdea.project_id
    user_id: str
    overall_competency_score: float = Field(..., ge=0.0, le=1.0, description="Overall weighted score from 0.0 to 1.0.")
    is_passed: bool # Derived from overall_competency_score >= COMPETENCY_THRESHOLD
    feedback_summary_html: str # General summary, can use simple HTML
    detailed_feedback_points: List[AssessmentFeedbackPoint] = Field(default_factory=list) # Changed from _html
    skills_demonstrated: List[str] = Field(default_factory=list, description="Skills identified from the submission, for gamification/badges.")
    critical_areas_for_improvement_html: List[str] = Field(default_factory=list) # Can use simple HTML
    positive_points_highlighted_html: List[str] = Field(default_factory=list) # Can use simple HTML, changed from positive_points_html
    tutor_session_triggered: bool = False #
    language_code: str # Language of the feedback

class ProjectAssessmentResponse(BaseModel): # Based on
    assessment_result: ProjectAssessmentResult
    debug_info: Optional[Dict[str, Any]] = None


# --- FastAPI Application ---
app = FastAPI( #
    title="Uplas Personalized Project Generation & Assessment Agent - InnovateAI Enhanced",
    description="Generates tailored real-world projects and assesses submissions using Vertex AI, with InnovateAI's strategic enhancements.",
    version="1.0.0"
)

# --- Vertex AI LLM Client Logic (Reused from existing, suitable for both tasks) ---
class VertexAILLMClient: #
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate_structured_response( #
        self, system_prompt: str, user_query_or_context: str, max_output_tokens: int,
        temperature: float = 0.5, top_p: float = 0.9, top_k: int = 35, # Adjusted defaults
        pydantic_model_for_schema: Optional[Any] = None # InnovateAI: Pass Pydantic model for schema
    ) -> Dict[str, Any]: # Returns dict with "raw_json_response", "prompt_token_count", "response_token_count"
        if not GCP_PROJECT_ID: #
            raise EnvironmentError("InnovateAI Critical: GCP_PROJECT_ID is not configured for Project Agent LLM Client.")

        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold #

        model = GenerativeModel( #
            self.model_name,
            system_instruction=[Part.from_text(system_prompt)] if system_prompt else None
        )
        
        generation_config_params = { #
            "max_output_tokens": max_output_tokens, "temperature": temperature,
            "top_p": top_p, "top_k": top_k,
        }
        if pydantic_model_for_schema: #
            try:
                generation_config_params["response_mime_type"] = "application/json"
                generation_config_params["response_schema"] = pydantic_model_for_schema.model_json_schema()
                logger.info(f"InnovateAI: Configured Gemini for JSON output using schema for {pydantic_model_for_schema.__name__}")
            except Exception as e_schema:
                 logger.error(f"InnovateAI Error: Failed to set JSON schema for model {pydantic_model_for_schema.__name__}. Error: {e_schema}", exc_info=True)
                 # Fallback to text or raise an error depending on how critical JSON is
                 generation_config_params["response_mime_type"] = "text/plain"
        
        generation_config = GenerationConfig(**generation_config_params)

        # InnovateAI: Define safety settings consistently
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        llm_response = await model.generate_content_async( #
            [Part.from_text(user_query_or_context)],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        raw_json_response = "" #
        prompt_tokens = 0 #
        response_tokens = 0 #

        if llm_response.candidates: #
            # ... (existing logic for extracting response_text and token counts)
            candidate = llm_response.candidates[0]
            if candidate.content and candidate.content.parts:
                raw_json_response = "".join([part.text for part in candidate.content.parts if part.text])
            if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
                prompt_tokens = llm_response.usage_metadata.prompt_token_count
                response_tokens = llm_response.usage_metadata.candidates_token_count
        
        return { #
            "raw_json_response": raw_json_response, # The calling endpoint will parse this
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

project_idea_llm = VertexAILLMClient(model_name=PROJECT_LLM_MODEL_NAME) #
assessment_llm = VertexAILLMClient(model_name=ASSESSMENT_LLM_MODEL_NAME) #


# --- Prompt Construction (InnovateAI Refined & New) ---

def construct_project_gen_system_prompt(language_code: str) -> str: # Based on
    # InnovateAI: System prompt sets the stage and desired output format.
    return (
        "You are an Uplas AI Curriculum Architect, an expert in devising innovative, personalized, and real-world applicable project assignments for learners across diverse technical fields. "
        f"Your primary objective is to generate engaging project ideas that directly map to user profiles, learning goals, and course contexts. Adhere strictly to the JSON output format specified by the user query, using the Pydantic model 'GeneratedProjectIdea' as the schema for each project idea. "
        f"All textual content (titles, descriptions, objectives, tasks, criteria, etc.) MUST be in the language: [{language_code}]. Do not include any markdown unless the field name explicitly ends with '_html' (in which case simple HTML for paragraphs, lists, bold, italics is acceptable). Do not add any introductory or concluding text outside the main JSON structure requested."
    )

def construct_project_gen_user_query( # Based on
    user_profile: UserProfileSnapshotForProjects,
    preferences: ProjectPreferences,
    course_summary: Optional[str],
    topic_keywords: Optional[List[str]],
    num_ideas: int,
    language_code: str
) -> str:
    # InnovateAI: User query provides the specifics and reiterates JSON output.
    # The Pydantic model schema will be passed via GenerationConfig, so no need to put full schema in prompt.
    user_query_parts = [
        f"Generate {num_ideas} distinct project idea(s) in [{language_code}] tailored to the following context.",
        "--- User Profile ---",
        json.dumps(user_profile.model_dump(exclude_none=True, indent=2)),
        "--- Project Preferences ---",
        json.dumps(preferences.model_dump(exclude_none=True, indent=2)),
    ]
    if course_summary:
        user_query_parts.append(f"--- Course Context Summary ---\n{course_summary}")
    if topic_keywords:
        user_query_parts.append(f"--- Key Topic Focus Keywords ---\n{', '.join(topic_keywords)}")
    
    user_query_parts.append(
        "\nReturn a JSON object with a single root key 'generated_projects'. The value of this key MUST be a list, where each item is a complete project idea object strictly conforming to the 'GeneratedProjectIdea' Pydantic model schema. "
        "Pay close attention to generating meaningful 'learning_objectives_html', 'expected_deliverables_html', and especially actionable 'assessment_rubric_html' for each project."
    )
    return "\n".join(user_query_parts)


def construct_project_assessment_system_prompt(language_code: str) -> str: # Based on
    # InnovateAI: System prompt for assessment emphasizes objectivity, constructiveness, and JSON format.
    return (
        "You are an Uplas AI Lead Project Evaluator, renowned for your insightful, fair, and constructive assessment of technical projects. "
        "Your task is to meticulously evaluate a user's project submission against the original project's defined objectives, deliverables, and assessment rubric. "
        f"Provide a comprehensive assessment, including an overall competency score (0.0-1.0), specific feedback points with scores, identified skills, and areas for improvement. All textual feedback MUST be in the language: [{language_code}]. "
        "Your entire response must be a single, valid JSON object strictly adhering to the Pydantic model 'LLMAssessmentRoot' (which forms part of 'ProjectAssessmentResponse'). No introductory or concluding text."
    )

def construct_project_assessment_user_query( # Based on
    original_project: GeneratedProjectIdea,
    submission_items: List[ProjectSubmissionContentItem],
    user_profile: UserProfileSnapshotForProjects, # For personalizing feedback tone/style
    language_code: str
) -> str:
    # InnovateAI: User query for assessment provides all necessary context.
    submission_details_str = ""
    for idx, item in enumerate(submission_items):
        content_val = item.value
        if item.content_type in [ProjectSubmissionContentType.GCS_URL_ZIP, ProjectSubmissionContentType.GCS_URL_PDF, ProjectSubmissionContentType.GITHUB_URL, ProjectSubmissionContentType.OTHER_URL_GENERAL]:
            content_preview = f"URL: {item.value}"
        elif item.content_type in [ProjectSubmissionContentType.PYTHON_CODE_STRING, ProjectSubmissionContentType.TEXT_REPORT, ProjectSubmissionContentType.MARKDOWN_DOCUMENT]:
            content_preview = item.value[:400] + ("..." if len(item.value) > 400 else "") # Preview for large text/code
        else:
            content_preview = "Content not directly previewable in prompt."
        submission_details_str += f"\nSubmission Item {idx+1}:\n- Type: {item.content_type.value}\n- Filename: {item.filename or 'N/A'}\n- Notes: {item.notes or 'N/A'}\n- Content/URL: {content_preview}\n"

    user_query_parts = [
        f"Assess the following user project submission in [{language_code}] based on the provided original project details and user profile context.",
        "\n--- Original Project Definition ---",
        f"Title: {original_project.title}",
        f"Learning Objectives: {json.dumps(original_project.learning_objectives_html)}",
        f"Expected Deliverables: {json.dumps(original_project.expected_deliverables_html)}",
        f"Assessment Rubric: {json.dumps(original_project.assessment_rubric_html)}",
        
        "\n--- User's Submission ---",
        submission_details_str,
        
        "\n--- User Profile (for feedback personalization) ---",
        f"User ID: {user_profile.user_id}",
        f"Preferred Persona for Feedback: {user_profile.preferred_tutor_persona}",

        "\n--- Assessment Instructions ---",
        "Carefully review the submission against each point in the 'Assessment Rubric' and the 'Learning Objectives'.",
        "For each distinct aspect you evaluate from the rubric or objectives, create a feedback point under 'detailed_feedback_points'. Assign a 'score_achieved' (0.0 to 1.0) and provide 'observation_feedback_html'.",
        "Calculate 'overall_competency_score' as a weighted average if rubric implies weights, or a simple average otherwise.",
        "Identify specific 'skills_demonstrated' and list 'critical_areas_for_improvement_html'.",
        "If assessing code (provided as string or via URL), perform a static review. Focus on correctness of logic, structure, readability, use of relevant technologies/libraries, and fulfillment of project tasks. Do NOT attempt to execute any code.",
        "Provide feedback in a '{user_profile.preferred_tutor_persona}' tone.",
        "Return a single JSON object as per the 'LLMAssessmentRoot' schema."
    ]
    return "\n".join(user_query_parts)

# --- Helper to call AI Tutor (InnovateAI Enhanced) ---
async def trigger_ai_tutor_for_failed_assessment( # Based on
    user_profile: UserProfileSnapshotForProjects, # Now using the richer profile
    project_title: str,
    assessment_summary_html: str, # Pass richer feedback
    areas_for_improvement_html: List[str],
    language_code: str
) -> bool:
    if not AI_TUTOR_AGENT_URL: #
        logger.warning("InnovateAI: AI_TUTOR_AGENT_URL not set. Cannot trigger AI Tutor.")
        return False

    # InnovateAI: Craft a more contextual query for the AI Tutor
    tutor_trigger_query = (
        f"I've received feedback on my project titled '{project_title}' and it seems I need some help. "
        f"The summary said: \"{assessment_summary_html}\". "
        f"Key areas I need to improve are: {'; '.join(areas_for_improvement_html)}. "
        "Can you help me understand these points better and guide me on how to improve my project?"
    )
    # Map UserProfileSnapshotForProjects to the UserProfileSnapshot expected by AI Tutor
    # This assumes AI Tutor's UserProfileSnapshot matches or is a subset.
    tutor_user_profile_payload = { #
        "industry": user_profile.industry, "profession": user_profile.profession,
        "career_interest": user_profile.career_interest,
        "current_knowledge_level": user_profile.current_knowledge_level,
        "preferred_tutor_persona": user_profile.preferred_tutor_persona,
        # Add country, city, learning_goals if AI Tutor's model includes them and they are in UserProfileSnapshotForProjects
    }

    tutor_payload = { #
        "user_id": user_profile.user_id,
        "query_text": tutor_trigger_query,
        "user_profile_snapshot": tutor_user_profile_payload,
        "language_code": language_code,
        "context": {
            "current_project_title": project_title,
            # Pass the full structured feedback if AI Tutor can process it, or a detailed summary.
            "project_assessment_feedback": f"Summary: {assessment_summary_html}. Areas to focus on: {'; '.join(areas_for_improvement_html)}"
        }
    }
    try: #
        # **** Placeholder for actual HTTP call to AI Tutor - Mugambi to implement ****
        logger.info(f"InnovateAI Placeholder: Would call AI Tutor for user {user_profile.user_id}, project '{project_title}'.")
        # async with httpx.AsyncClient(timeout=60.0) as client:
        #     response = await client.post(f"{AI_TUTOR_AGENT_URL}/v1/ask-tutor", json=tutor_payload)
        #     response.raise_for_status()
        #     logger.info(f"InnovateAI: Successfully triggered AI Tutor for user {user_profile.user_id}. Tutor response status: {response.status_code}")
        #     return True
        return True # Mock success
        # **** END Placeholder ****
    except Exception as e: #
        logger.error(f"InnovateAI Error: Failed to trigger AI Tutor for user {user_profile.user_id}. Error: {e}", exc_info=True)
        return False

# --- API Endpoints (InnovateAI Fully Rewritten) ---

@app.post("/v1/generate-project-ideas", response_model=ProjectIdeaGenerationResponse, summary="Generate Personalized Project Ideas (InnovateAI Enhanced)")
async def generate_project_ideas_endpoint(request_data: ProjectIdeaGenerationRequest): # Based on
    start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE
    logger.info(f"InnovateAI: Received project generation request for user {request_data.user_profile_snapshot.user_id} in {effective_language_code}.")

    if not GCP_PROJECT_ID: #
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Project Generation service not configured (Missing GCP Project ID).")

    system_prompt = construct_project_gen_system_prompt(effective_language_code)
    user_query = construct_project_gen_user_query(
        user_profile=request_data.user_profile_snapshot,
        preferences=request_data.preferences,
        course_summary=request_data.course_context_summary,
        topic_keywords=request_data.topic_focus_keywords,
        num_ideas=request_data.number_of_ideas,
        language_code=effective_language_code
    )
    
    # Define the Pydantic model for the root of the LLM's JSON response for generation
    class LLMProjectGenRoot(BaseModel):
        generated_projects: List[GeneratedProjectIdea]

    raw_json_string = "{}" # Default in case of issues
    try:
        llm_response_metrics = await project_idea_llm.generate_structured_response( #
            system_prompt=system_prompt, user_query_or_context=user_query,
            max_output_tokens=8192, # Generous for potentially multiple detailed ideas
            pydantic_model_for_schema=LLMProjectGenRoot
        )
        raw_json_string = llm_response_metrics.get("raw_json_response", "{}")
        if not raw_json_string.strip() or raw_json_string == "{}":
             raise ValueError("LLM returned empty content for project ideas.")
        
        parsed_llm_root = LLMProjectGenRoot(**json.loads(raw_json_string))
        generated_projects_list = parsed_llm_root.generated_projects

        if not generated_projects_list:
             raise ValueError("LLM response parsed, but no project ideas were found in the 'generated_projects' list.")

        # InnovateAI: Ensure language_code is correctly set in each generated project
        for project in generated_projects_list:
            project.language_code = effective_language_code
        
        # InnovateAI Note: Here, you would persist each project in `generated_projects_list` to your database.
        # The `project_id` is auto-generated. This stored data is vital for the assessment phase.
        # Example: for project_idea in generated_projects_list: db.save_project_definition(project_idea)
        logger.info(f"InnovateAI: Successfully generated and validated {len(generated_projects_list)} project ideas for user {request_data.user_profile_snapshot.user_id}.")

    except (json.JSONDecodeError, Exception) as e: # Includes Pydantic validation by LLMProjectGenRoot
        logger.error(f"InnovateAI Error: Parsing/validating LLM response for project generation failed. Error: {e}. Response: {raw_json_string[:500]}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate or parse project ideas due to AI service error: {str(e)}")
    except Exception as e: #
        logger.error(f"InnovateAI Unexpected error during project idea generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Project idea generation failed unexpectedly: {str(e)}")

    processing_time_ms = (time.perf_counter() - start_time) * 1000
    debug_info = { #
        "llm_model_name_used": project_idea_llm.model_name,
        "processing_time_ms": round(processing_time_ms, 2),
        "prompt_token_count": llm_response_metrics.get("prompt_token_count"),
        "response_token_count": llm_response_metrics.get("response_token_count"),
        "language_generated": effective_language_code,
        "num_ideas_requested": request_data.number_of_ideas,
        "num_ideas_generated_valid": len(generated_projects_list)
    }
    return ProjectIdeaGenerationResponse(generated_projects=generated_projects_list, debug_info=debug_info)


@app.post("/v1/assess-project-submission", response_model=ProjectAssessmentResponse, summary="Assess a User's Project Submission (InnovateAI Enhanced)")
async def assess_project_submission_endpoint(request_data: ProjectAssessmentRequest): # Based on assess_project_endpoint from
    start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE
    logger.info(f"InnovateAI: Received project assessment request for user {request_data.user_id}, project {request_data.project_id}.")

    if not GCP_PROJECT_ID: #
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Project Assessment service not configured (Missing GCP Project ID).")

    # --- InnovateAI Step 1: Fetch original project details (objectives, criteria) ---
    # This would typically come from your database using request_data.project_id
    original_project_details: Optional[GeneratedProjectIdea] = None
    # **** InnovateAI Placeholder for DB call: original_project_details = await db.get_project_definition(request_data.project_id) ****
    if request_data.project_id == "proj_mock_reco_sys_123": # Example known ID for mocking
        logger.info(f"InnovateAI Placeholder: Using mock original project details for assessment of {request_data.project_id}")
        original_project_details = GeneratedProjectIdea( # Ensure this mock has all fields, esp. assessment_rubric_html
            project_id="proj_mock_reco_sys_123", title="Mock Recommendation System Design",
            description_html="<p>Design a basic recommendation engine.</p>", difficulty_level="Intermediate",
            learning_objectives_html=["Understand collaborative filtering.", "Outline data needs."],
            expected_deliverables_html=["A 2-page PDF report."],
            key_tasks=[GeneratedProjectTask(task_id="t1", title="Research", description="Research techniques.")],
            assessment_rubric_html=["Clarity of explanation (50%)", "Feasibility of design (50%)"],
            language_code=effective_language_code # Ensure it matches
        )
    # **** END Placeholder ****
    if not original_project_details:
        logger.error(f"InnovateAI Error: Original project details not found for project_id: {request_data.project_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Original project definition not found for project ID {request_data.project_id}.")

    # --- InnovateAI Step 2: Construct Assessment Prompt ---
    system_prompt = construct_project_assessment_system_prompt(effective_language_code)
    user_query = construct_project_assessment_user_query(
        original_project=original_project_details,
        submission_items=request_data.submission_items,
        user_profile=request_data.user_profile_snapshot,
        language_code=effective_language_code
    )
    
    # Define the Pydantic model for the root of the LLM's assessment JSON response
    class LLMAssessmentRoot(BaseModel): # This matches the fields requested in the assessment prompt
        overall_competency_score: float
        feedback_summary_html: str # Changed from feedback_summary
        detailed_feedback_points: List[AssessmentFeedbackPoint] # Changed from _html
        skills_demonstrated: List[str]
        critical_areas_for_improvement_html: List[str] # Changed from _html
        positive_points_highlighted_html: List[str] # InnovateAI: new field for explicitly positive points

    assessment_result_dict: Optional[Dict] = None
    raw_json_string = "{}"
    llm_response_metrics = {}
    try:
        llm_response_metrics = await assessment_llm.generate_structured_response( #
            system_prompt=system_prompt, user_query_or_context=user_query,
            max_output_tokens=4096, # Allow ample space for detailed feedback
            pydantic_model_for_schema=LLMAssessmentRoot
        )
        raw_json_string = llm_response_metrics.get("raw_json_response", "{}")
        if not raw_json_string.strip() or raw_json_string == "{}":
            raise ValueError("LLM returned empty content for project assessment.")

        assessment_data_from_llm = json.loads(raw_json_string)
        # Validate with LLMAssessmentRoot
        parsed_llm_assessment = LLMAssessmentRoot(**assessment_data_from_llm)

    except (json.JSONDecodeError, Exception) as e: # Includes Pydantic validation
        logger.error(f"InnovateAI Error: Parsing/validating LLM response for project assessment: {e}. Response: {raw_json_string[:500]}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to parse or validate assessment data from AI: {str(e)}")
    except Exception as e: #
        logger.error(f"InnovateAI Unexpected error during project assessment LLM call: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Project assessment failed due to AI service error: {str(e)}")

    # --- InnovateAI Step 3: Determine Competency & Tutor Trigger ---
    overall_score_from_llm = parsed_llm_assessment.overall_competency_score
    is_user_competent = overall_score_from_llm >= COMPETENCY_THRESHOLD
    trigger_tutor = not is_user_competent

    assessment_result_obj = ProjectAssessmentResult(
        project_id=request_data.project_id,
        user_id=request_data.user_id,
        overall_competency_score=overall_score_from_llm,
        is_passed=is_user_competent, # Renamed from is_competent to match original file better
        feedback_summary_html=parsed_llm_assessment.feedback_summary_html,
        detailed_feedback_points=parsed_llm_assessment.detailed_feedback_points,
        skills_demonstrated=parsed_llm_assessment.skills_demonstrated,
        critical_areas_for_improvement_html=parsed_llm_assessment.critical_areas_for_improvement_html,
        positive_points_highlighted_html=parsed_llm_assessment.positive_points_highlighted_html, # New
        tutor_session_triggered=trigger_tutor, # Set this based on logic
        language_code=effective_language_code
    )

    if trigger_tutor: #
        logger.info(f"InnovateAI: User {request_data.user_id} for project {request_data.project_id} scored {overall_score_from_llm:.2f}. Triggering AI Tutor assistance.")
        # The UserProfileSnapshotForProjects is directly available in request_data
        tutor_triggered_successfully = await trigger_ai_tutor_for_failed_assessment(
            user_profile=request_data.user_profile_snapshot,
            project_title=original_project_details.title,
            assessment_summary_html=assessment_result_obj.feedback_summary_html,
            areas_for_improvement_html=assessment_result_obj.critical_areas_for_improvement_html,
            language_code=effective_language_code
        )
        assessment_result_obj.tutor_session_triggered = tutor_triggered_successfully # Update based on actual trigger attempt
    
    # InnovateAI Note: Persist `assessment_result_obj` to your database.
    # Example: await db.save_assessment_result(assessment_result_obj)

    processing_time_ms = (time.perf_counter() - start_time) * 1000
    debug_info = { #
        "llm_model_name_used": assessment_llm.model_name,
        "processing_time_ms": round(processing_time_ms, 2),
        "prompt_token_count": llm_response_metrics.get("prompt_token_count"),
        "response_token_count": llm_response_metrics.get("response_token_count"),
        "language_assessed_in": effective_language_code
    }
    return ProjectAssessmentResponse(assessment_result=assessment_result_obj, debug_info=debug_info)


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK) #
async def health_check():
    if not GCP_PROJECT_ID: #
        return {"status": "unhealthy", "reason": "GCP_PROJECT_ID not configured.", "service": "ProjectGenerationAssessmentAgent_InnovateAI"}
    if not AI_TUTOR_AGENT_URL:
        logger.warning("InnovateAI Health Warning: AI_TUTOR_AGENT_URL not set. Tutor triggering on failed assessment will not function.")
        # Not marking as unhealthy for this, but it's a partial degradation.
    return {"status": "healthy", "service": "ProjectGenerationAssessmentAgent_InnovateAI", "innovate_ai_enhancements_active": True}


if __name__ == "__main__": #
    import uvicorn
    logger.info("InnovateAI: Starting Personalized Project Generation & Assessment Agent for local development...")
    if not GCP_PROJECT_ID: #
        print("InnovateAI Warning: GCP_PROJECT_ID is not set for Project Agent. LLM calls will fail.")
    if not AI_TUTOR_AGENT_URL: #
        print("InnovateAI Warning: AI_TUTOR_AGENT_URL is not set. Tutor triggering for failed assessments will not work.")
    
    port = int(os.getenv("PORT", 8004)) # Default port from original file
    uvicorn.run(app, host="0.0.0.0", port=port)
