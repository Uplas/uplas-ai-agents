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
import random # For selecting persona snippets

# GCP Clients
from google.cloud import aiplatform
import google.auth
import google.auth.transport.requests

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") 
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") 
PROJECT_LLM_MODEL_NAME = os.getenv("PROJECT_LLM_MODEL_NAME", "gemini-1.5-flash-001") 
ASSESSMENT_LLM_MODEL_NAME = os.getenv("ASSESSMENT_LLM_MODEL_NAME", "gemini-1.5-pro-001")
AI_TUTOR_AGENT_URL = os.getenv("AI_TUTOR_AGENT_URL") 
COMPETENCY_THRESHOLD = float(os.getenv("PROJECT_COMPETENCY_THRESHOLD", "0.75"))

if not GCP_PROJECT_ID: 
    logging.warning("NovaSpark Warning: GCP_PROJECT_ID environment variable not set for Project Agent. LLM calls may fail.")
else:
    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION) 
        logging.info(f"NovaSpark: Vertex AI SDK initialized for Project Agent. Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}")
    except Exception as e_init:
        logging.error(f"NovaSpark Critical: Failed to initialize Vertex AI SDK for Project Agent. Error: {e_init}", exc_info=True)


SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] 
DEFAULT_LANGUAGE = "en-US" 

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

# --- NovaSpark: Persona Snippet Library for Assessment Feedback (Task 4) ---
ASSESSMENT_PERSONA_LIBRARY = {
    "empathetic_mentor": {
        "description": "Provides feedback that is understanding, supportive, patient, focuses on building confidence, and gently diagnoses underlying issues.",
        "snippets": [
            "I can see you've put a good amount of effort into this, and tackling [project_aspect] can indeed be challenging. Let's look at how we can build on this foundation.",
            "This is a solid start! One area we can refine together to make it even stronger is [specific_point]. Think of it as polishing a gem to really make it shine.",
            "It's perfectly okay if some parts didn't click immediately – that's a natural part of the learning process. The key is to identify those areas and work through them, which is exactly what we're doing now."
        ]
    },
    "constructive_critic": { # Example of an alternative persona
        "description": "Provides direct, clear, and actionable feedback focusing on technical accuracy and areas for concrete improvement, while maintaining a professional and respectful tone.",
        "snippets": [
            "The submission demonstrates a foundational understanding of [concept_A]. However, to meet the project's objectives for [objective_B], consider refining [specific_area_1] by implementing [suggestion_1], and re-evaluating [specific_area_2] against the rubric point concerning [rubric_point_X].",
            "A key requirement was [requirement_Y]. While your approach to [related_part] is noted, it doesn't fully address this. A more effective strategy would involve [alternative_approach_Z].",
            "To elevate this work, focus on these three areas: 1. [Improvement_1], 2. [Improvement_2], 3. [Improvement_3]. Addressing these will significantly enhance the project's quality and alignment with the learning objectives."
        ]
    },
    "encouraging_reviewer": {
         "description": "Highlights positives, frames areas for improvement as growth opportunities, and maintains an upbeat and motivating tone.",
         "snippets": [
            "Great job on getting this far and submitting your work! I was particularly impressed with [positive_aspect_1] and [positive_aspect_2]. That shows real promise!",
            "You're building some excellent skills here! To take it to the next level, let's focus a bit on [area_for_growth]. Mastering this will really round out your understanding.",
            "This is a fantastic learning opportunity! The way you tackled [specific_challenge] was innovative. With a few adjustments to [another_area], this project will be even more impactful."
         ]
    }
}
# --- Pydantic Models ---
# UserProfileSnapshotForProjects, ProjectPreferences, ProjectIdeaGenerationRequest, 
# GeneratedProjectTask, GeneratedProjectIdea, ProjectIdeaGenerationResponse,
# ProjectSubmissionContentType, ProjectSubmissionContentItem, ProjectAssessmentRequest,
# AssessmentFeedbackPoint, ProjectAssessmentResult, ProjectAssessmentResponse
# (These models are taken directly from main (13).py and seem robust for the tasks)

class UserProfileSnapshotForProjects(BaseModel):
    user_id: str = Field(..., examples=["user_uuid_abc123"])
    industry: Optional[str] = Field(None, examples=["Fintech", "Healthcare", "Education Technology"])
    profession: Optional[str] = Field(None, examples=["Data Analyst", "Software Developer", "UX Designer"])
    career_interest: Optional[str] = Field(None, examples=["Machine Learning Engineer", "Cloud Solutions Architect", "Cybersecurity Analyst"])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "SQL": "Beginner", "Cloud Functions": "Novice"}])
    areas_of_interest: Optional[List[str]] = Field(default_factory=list, examples=[["natural_language_processing", "sustainable_technology_solutions"]])
    learning_goals: Optional[str] = Field(None, examples=["Develop a strong portfolio project for job applications.", "Master practical AI implementation."])
    preferred_tutor_persona: Optional[str] = Field("encouraging_reviewer", examples=["empathetic_mentor", "constructive_critic", "encouraging_reviewer"]) # Task 4: Updated examples


class ProjectPreferences(BaseModel):
    difficulty_level: Optional[str] = Field("intermediate", examples=["beginner", "intermediate", "advanced"])
    preferred_technologies: Optional[List[str]] = Field(default_factory=list, examples=[["Python", "FastAPI", "Google Cloud Vision API"]])
    project_type_focus: Optional[str] = Field(None, examples=["Portfolio Piece demonstrating full-stack skills", "Research-oriented analysis", "Creative design challenge"])
    time_commitment_hours_estimate: Optional[int] = Field(None, examples=[20, 40], ge=5, le=150)


class ProjectIdeaGenerationRequest(BaseModel):
    user_profile_snapshot: UserProfileSnapshotForProjects
    preferences: Optional[ProjectPreferences] = Field(default_factory=ProjectPreferences)
    course_context_summary: Optional[str] = Field(None, description="Brief summary of the course or module for which the project is being generated.", examples=["A course on building AI-powered web applications."])
    topic_focus_keywords: Optional[List[str]] = Field(default_factory=list, description="Specific keywords or concepts from the course to focus the project on.", examples=["image_recognition", "user_authentication"])
    number_of_ideas: Optional[int] = Field(1, ge=1, le=3, description="Number of distinct project ideas to generate.")
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)

    @validator('language_code')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"NovaSpark: Unsupported language_code '{v}' in ProjectIdeaGenerationRequest. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class GeneratedProjectTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:6]}" )
    title: str = Field(..., min_length=5, description="Short title for the task.")
    description: str = Field(..., min_length=20, description="Detailed description of what the task involves.")
    estimated_sub_duration_hours: Optional[float] = Field(None, examples=[2.5, 4.0], ge=0.5)

class GeneratedProjectIdea(BaseModel):
    project_id: str = Field(default_factory=lambda: f"proj_{uuid.uuid4().hex[:12]}", description="Unique ID for this generated project.")
    title: str = Field(..., examples=["AI-Powered Customer Feedback Analyzer for E-commerce"])
    subtitle: Optional[str] = Field(None, examples=["Leveraging NLP to categorize and prioritize user reviews."])
    description_html: str = Field(..., description="Engaging real-world scenario or problem statement (can include simple HTML for formatting).")
    difficulty_level: str = Field(..., examples=["Beginner", "Intermediate", "Advanced"])
    estimated_duration_hours: Optional[int] = Field(None, examples=[25, 50])
    learning_objectives_html: List[str] = Field(..., min_length=2, description="What the user will learn/demonstrate (can include simple HTML).")
    expected_deliverables_html: List[str] = Field(..., min_length=1, description="Specific items the user must produce (e.g., 'A Python script performing sentiment analysis', 'A summary report in PDF format').")
    key_tasks: List[GeneratedProjectTask] = Field(..., min_length=2)
    suggested_tools_technologies: List[str] = Field(default_factory=list, examples=["Python (NLTK, Scikit-learn)", "Flask/FastAPI", "GCP Natural Language API"])
    assessment_rubric_html: List[str] = Field(..., min_length=2, description="Key criteria for successful completion and how it will be judged (can include simple HTML). E.g., 'Functionality of the NLP model (40%)', 'Clarity of the report (30%)'.")
    personalization_rationale: Optional[str] = Field(None, description="Brief explanation of why this project is suitable for the user's profile.")
    potential_challenges_html: Optional[List[str]] = Field(default_factory=list, description="Possible hurdles or complexities the user might encounter.")
    real_world_application_examples_html: Optional[List[str]] = Field(default_factory=list, description="Examples of how this project type is used in industry.")
    language_code: str 

class ProjectIdeaGenerationResponse(BaseModel):
    generated_projects: List[GeneratedProjectIdea] 
    debug_info: Optional[Dict[str, Any]] = None

class ProjectSubmissionContentType(str, Enum):
    TEXT_REPORT = "text_report"
    PYTHON_CODE_STRING = "python_code_string"
    MARKDOWN_DOCUMENT = "markdown_document"
    GCS_URL_ZIP = "gcs_url_zip_file"
    GCS_URL_PDF = "gcs_url_pdf_document"
    GITHUB_URL = "github_url" 
    OTHER_URL_GENERAL = "other_url_general"

class ProjectSubmissionContentItem(BaseModel):
    content_type: ProjectSubmissionContentType
    value: str = Field(..., description="The actual content (e.g. code string, text report) or a fully qualified URL.")
    filename: Optional[str] = Field(None, description="Original filename if applicable (e.g., for GCS uploads or when content_type is code_string).")
    notes: Optional[str] = Field(None, description="Optional user notes about this specific submission item.")

class ProjectAssessmentRequest(BaseModel):
    user_id: str = Field(..., examples=["user_assess_xyz"])
    project_id: str = Field(..., description="The ID of the Uplas-generated project being submitted against.") 
    submission_items: List[ProjectSubmissionContentItem] = Field(..., min_length=1, description="List of submitted content items.")
    user_profile_snapshot: UserProfileSnapshotForProjects # For personalizing feedback
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)

    @validator('language_code')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES: 
            logger.warning(f"NovaSpark: Unsupported language_code '{v}' in ProjectAssessmentRequest. Defaulting to '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class AssessmentFeedbackPoint(BaseModel):
    aspect_evaluated: str = Field(..., examples=["Algorithm Correctness", "Report Clarity", "Code Readability"])
    score_achieved: float = Field(..., ge=0.0, le=1.0, description="Normalized score for this aspect (0.0-1.0).")
    observation_feedback_html: str = Field(..., description="Specific feedback (strength or area for improvement), can use simple HTML.")
    is_strength: Optional[bool] = Field(None, description="True if this point is primarily a positive observation.")

class ProjectAssessmentResult(BaseModel):
    assessment_id: str = Field(default_factory=lambda: f"assess_{uuid.uuid4().hex[:12]}")
    project_id: str 
    user_id: str
    overall_competency_score: float = Field(..., ge=0.0, le=1.0)
    is_passed: bool 
    feedback_summary_html: str 
    detailed_feedback_points: List[AssessmentFeedbackPoint] = Field(default_factory=list) 
    skills_demonstrated: List[str] = Field(default_factory=list)
    critical_areas_for_improvement_html: List[str] = Field(default_factory=list) 
    positive_points_highlighted_html: List[str] = Field(default_factory=list) 
    tutor_session_triggered: bool = False 
    language_code: str 

class ProjectAssessmentResponse(BaseModel):
    assessment_result: ProjectAssessmentResult
    debug_info: Optional[Dict[str, Any]] = None

app = FastAPI( 
    title="Uplas Personalized Project Generation & Assessment Agent - NovaSpark Enhanced",
    description="Generates tailored real-world projects and assesses submissions using Vertex AI, with NovaSpark's strategic enhancements for personalization, multilingual support, and human-like feedback.",
    version="1.1.0" # Incremented for Tasks 2, 3, 4
)

# --- Vertex AI LLM Client Logic ---
class VertexAILLMClient: 
    # ... (LLM Client from main (13).py - no changes needed for these tasks, it's generic enough)
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_initialized = False
        if not GCP_PROJECT_ID:
            logger.error("NovaSpark Critical: GCP_PROJECT_ID not set for Project Agent LLM Client.")
        else:
            try:
                aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
                self.is_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Vertex AI for Project Agent: {e}", exc_info=True)


    async def generate_structured_response( 
        self, system_prompt: str, user_query_or_context: str, max_output_tokens: int,
        temperature: float = 0.5, top_p: float = 0.9, top_k: int = 35, 
        pydantic_model_for_schema: Optional[Any] = None 
    ) -> Dict[str, Any]: 
        if not self.is_initialized:
            logger.error("NovaSpark Error: Project Agent LLM client not initialized.")
            return {"raw_json_response": json.dumps({"error": "LLM client not initialized"}), "prompt_token_count": 0, "response_token_count": 0}

        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold 
        model = GenerativeModel(self.model_name, system_instruction=[Part.from_text(system_prompt)] if system_prompt else None)
        
        generation_config_params = {"max_output_tokens": max_output_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}
        if pydantic_model_for_schema: 
            try:
                generation_config_params["response_mime_type"] = "application/json"
                generation_config_params["response_schema"] = pydantic_model_for_schema.model_json_schema()
                logger.info(f"NovaSpark: Configured Gemini for JSON output using schema for {pydantic_model_for_schema.__name__}")
            except Exception as e_schema:
                 logger.error(f"NovaSpark Error: Failed to set JSON schema for model {pydantic_model_for_schema.__name__}. Error: {e_schema}", exc_info=True)
                 generation_config_params["response_mime_type"] = "text/plain"
        generation_config = GenerationConfig(**generation_config_params)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        raw_json_response = "" 
        prompt_tokens = 0 
        response_tokens = 0 
        candidate = None
        try:
            llm_response = await model.generate_content_async(
                [Part.from_text(user_query_or_context)],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            if llm_response.candidates: 
                candidate = llm_response.candidates[0]
                if candidate.finish_reason not in [candidate.FinishReason.STOP, candidate.FinishReason.MAX_TOKENS]:
                    logger.warning(f"NovaSpark LLM Warning: Project Agent response generation finished with reason: {candidate.finish_reason.name if candidate else 'N/A'}.")
                    if candidate.finish_reason == candidate.FinishReason.SAFETY:
                         # Provide a valid JSON error structure if JSON output was expected
                         error_payload = {"error": "Content generation stopped due to safety reasons."}
                         if pydantic_model_for_schema: # Try to make it somewhat schema compliant if possible
                             # This is a simple attempt; more complex schemas would need more sophisticated error objects.
                             # For LLMProjectGenRoot, it expects 'generated_projects': []
                             if pydantic_model_for_schema.__name__ == "LLMProjectGenRoot":
                                 error_payload = {"generated_projects": [{"project_id": "error_safety", "title": "Error", "description_html": error_payload["error"], "difficulty_level":"N/A", "learning_objectives_html":[], "expected_deliverables_html":[], "key_tasks":[], "assessment_rubric_html":[], "language_code": DEFAULT_LANGUAGE}]}
                             # For LLMAssessmentRoot
                             elif pydantic_model_for_schema.__name__ == "LLMAssessmentRoot":
                                  error_payload = {"overall_competency_score": 0.0, "feedback_summary_html": error_payload["error"], "detailed_feedback_points": [], "skills_demonstrated": [], "critical_areas_for_improvement_html": [], "positive_points_highlighted_html": []}

                         raw_json_response = json.dumps(error_payload)


                if not raw_json_response and candidate.content and candidate.content.parts:
                    raw_json_response = "".join([part.text for part in candidate.content.parts if part.text])
                
                if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
                    prompt_tokens = llm_response.usage_metadata.prompt_token_count
                    response_tokens = llm_response.usage_metadata.candidates_token_count
            
            if not raw_json_response and candidate and candidate.finish_reason == candidate.FinishReason.STOP:
                 raw_json_response = "{}" 
                 logger.warning("NovaSpark LLM Warning: Project Agent received successful stop but no text content. Assuming empty JSON.")
        except Exception as e_sdk:
            logger.error(f"NovaSpark SDK Error for Project Agent LLM: {e_sdk}", exc_info=True)
            raise # Let endpoint handle
            
        return { 
            "raw_json_response": raw_json_response, 
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

project_idea_llm = VertexAILLMClient(model_name=PROJECT_LLM_MODEL_NAME) 
assessment_llm = VertexAILLMClient(model_name=ASSESSMENT_LLM_MODEL_NAME) 


# --- Prompt Construction ---

# NovaSpark Enhanced: Project Generation System Prompt (Tasks 2 & 3)
def construct_project_gen_system_prompt(language_code: str) -> str: 
    return (
        f"You are an Uplas AI Curriculum Architect, an expert in devising innovative, personalized, and real-world applicable project assignments. Your response MUST be entirely in the specified language: [{language_code}].\n"
        "Your primary objective is to generate engaging project ideas that are DEEPLY TAILORED to the provided user profile, preferences, course context, and topic keywords. "
        "Do not generate generic templates; each project should feel uniquely suited to the individual learner.\n"
        "Key Personalization Instructions:\n"
        "1. Analyze all aspects of the 'User Profile': industry, profession, career interests, current knowledge levels (especially for suggesting appropriate difficulty and technologies), and areas of interest.\n"
        "2. Analyze 'Project Preferences': difficulty, preferred technologies, project type focus, and time commitment.\n"
        "3. Generate a 'title', 'description_html', 'learning_objectives_html', 'key_tasks', and 'suggested_tools_technologies' that directly reflect these user inputs.\n"
        "4. CRITICAL: The 'personalization_rationale' field MUST clearly and specifically explain HOW and WHY the generated project (its theme, tools, complexity, or deliverables) is a good fit for THIS user, referencing specific details from their profile and preferences. Example: 'This project on visualizing financial data with Python/Plotly aligns with your fintech industry background, data analyst profession, and intermediate Python knowledge. It will help you achieve your learning goal of mastering data visualization for portfolio development.'\n"
        "5. Ensure the 'assessment_rubric_html' is clear, actionable, and appropriate for the project's complexity and learning objectives.\n"
        f"Strictly adhere to the JSON output format specified (Pydantic model 'GeneratedProjectIdea' for each project). All textual content MUST be in [{language_code}]. "
        "Do not include any markdown unless the field name explicitly ends with '_html'. No introductory or concluding text outside the main JSON structure requested."
    )

# NovaSpark Enhanced: Project Generation User Query (Task 2)
def construct_project_gen_user_query( 
    user_profile: UserProfileSnapshotForProjects, preferences: ProjectPreferences,
    course_summary: Optional[str], topic_keywords: Optional[List[str]],
    num_ideas: int, language_code: str
) -> str:
    user_query_parts = [
        f"Generate {num_ideas} distinct project idea(s) IN LANGUAGE [{language_code}] tailored to the following context. Ensure each project idea is deeply personalized and not a generic template, actively leveraging all provided user details in its design, content, and especially the 'personalization_rationale'.",
        "--- User Profile ---", json.dumps(user_profile.model_dump(exclude_none=True, indent=2)),
        "--- Project Preferences ---", json.dumps(preferences.model_dump(exclude_none=True, indent=2)),
    ]
    if course_summary: user_query_parts.append(f"--- Course Context Summary ---\n{course_summary}")
    if topic_keywords: user_query_parts.append(f"--- Key Topic Focus Keywords ---\n{', '.join(topic_keywords)}")
    
    user_query_parts.append(
        "\nReturn a JSON object with a single root key 'generated_projects'. The value of this key MUST be a list, where each item is a complete project idea object strictly conforming to the 'GeneratedProjectIdea' Pydantic model schema. "
        "Pay close attention to generating meaningful 'learning_objectives_html', 'expected_deliverables_html', actionable 'assessment_rubric_html', and a compelling, specific 'personalization_rationale' for each project. "
        f"ALL TEXTUAL CONTENT in the response, including all project details, MUST be in [{language_code}]."
    )
    return "\n".join(user_query_parts)

# NovaSpark Enhanced: Project Assessment System Prompt (Tasks 3 & 4)
def construct_project_assessment_system_prompt(language_code: str, preferred_persona: Optional[str]) -> str: 
    persona_instructions = ""
    # Task 4: Incorporate Persona Snippets
    normalized_persona_key = (preferred_persona or "encouraging_reviewer").lower().replace(" ", "_").replace("-", "_")
    selected_persona_data = ASSESSMENT_PERSONA_LIBRARY.get(normalized_persona_key, ASSESSMENT_PERSONA_LIBRARY["encouraging_reviewer"]) # Fallback
    
    persona_instructions = f"\n--- Persona Guidance for Feedback ---\n"
    persona_instructions += f"Adopt the persona of an '{selected_persona_data['description']}'. Your feedback tone should reflect this.\n"
    if selected_persona_data['snippets']:
        persona_instructions += "Consider these stylistic examples (adapt, don't copy verbatim):\n"
        for snippet in random.sample(selected_persona_data['snippets'], min(len(selected_persona_data['snippets']), 1)): # Pick one snippet
            persona_instructions += f"- \"{snippet}\"\n"

    return (
        f"You are an Uplas AI Lead Project Evaluator. Your response MUST be entirely in the specified language: [{language_code}].\n"
        "Your task is to meticulously evaluate a user's project submission against the original project's defined objectives, deliverables, and assessment rubric. "
        f"{persona_instructions}" # Task 4: Persona instructions injected
        "Provide a comprehensive assessment, including an overall competency score (0.0-1.0), specific feedback points with scores, identified skills, and areas for improvement. "
        "Be constructive, fair, and clear. If the persona calls for empathy, ensure your language reflects understanding and support, especially if the score is low.\n"
        # Task 4: Few-shot example of feedback style
        "--- Example of Constructive & Empathetic Feedback Point Structure ---\n"
        "If evaluating 'Code Quality' and it needs improvement (score 0.5):\n"
        "\"Aspect Evaluated: Code Quality\n"
        "Score Achieved: 0.5\n"
        "Observation/Feedback (HTML): '<p>I see you've focused on getting the core functionality working, which is a great first step! To enhance the code further, consider applying consistent formatting and adding comments to explain complex logic sections. This will make it much easier for others (and your future self!) to understand and maintain. For instance, tools like Black for Python can automate formatting. Keep up the great work on tackling the main problem!</p>'\n"
        "Is Strength: false\"\n"
        f"Your entire response must be a single, valid JSON object strictly adhering to the Pydantic model 'LLMAssessmentRoot'. ALL textual feedback MUST be in [{language_code}]. No introductory or concluding text."
    )

# NovaSpark Enhanced: Project Assessment User Query (Tasks 3 & 4)
def construct_project_assessment_user_query( 
    original_project: GeneratedProjectIdea, submission_items: List[ProjectSubmissionContentItem],
    user_profile: UserProfileSnapshotForProjects, language_code: str
) -> str:
    submission_details_str = "" # ... (submission_details_str formatting from main (13).py - assumed correct)
    for idx, item in enumerate(submission_items):
        content_val = item.value
        preview = f"URL: {item.value}" if "url" in item.content_type.value.lower() or "github" in item.content_type.value.lower() else item.value[:400] + ("..." if len(item.value) > 400 else "")
        submission_details_str += f"\nSubmission Item {idx+1}:\n- Type: {item.content_type.value}\n- Filename: {item.filename or 'N/A'}\n- Notes: {item.notes or 'N/A'}\n- Content/URL Preview: {preview}\n"


    user_query_parts = [
        f"Assess the following user project submission IN LANGUAGE [{language_code}] based on the provided original project details and user profile context. Ensure all your feedback text is in [{language_code}].",
        "\n--- Original Project Definition ---",
        f"Title: {original_project.title}",
        f"Language of Project Definition: [{original_project.language_code}]", # For LLM context
        f"Learning Objectives: {json.dumps(original_project.learning_objectives_html)}",
        f"Expected Deliverables: {json.dumps(original_project.expected_deliverables_html)}",
        f"Assessment Rubric: {json.dumps(original_project.assessment_rubric_html)}",
        "\n--- User's Submission ---", submission_details_str,
        "\n--- User Profile (for feedback tone personalization) ---",
        f"User ID: {user_profile.user_id}",
        f"Preferred Persona for Feedback: {user_profile.preferred_tutor_persona}", # This informs system prompt persona choice
        "\n--- Assessment Instructions ---",
        "Carefully review the submission against each point in the 'Assessment Rubric' and 'Learning Objectives'.",
        "For each distinct aspect, create a feedback point under 'detailed_feedback_points' with a 'score_achieved' (0.0-1.0) and 'observation_feedback_html'.",
        "Calculate 'overall_competency_score' based on the rubric.",
        "Identify 'skills_demonstrated' and list 'critical_areas_for_improvement_html' and 'positive_points_highlighted_html'.",
        "If assessing code, perform a static review. Do NOT attempt to execute any code.",
        f"Adhere to the requested feedback persona and ensure ALL generated text is in [{language_code}]. Return a single JSON object as per 'LLMAssessmentRoot' schema."
    ]
    return "\n".join(user_query_parts)

async def trigger_ai_tutor_for_failed_assessment( 
    user_profile: UserProfileSnapshotForProjects, project_title: str,
    assessment_summary_html: str, areas_for_improvement_html: List[str],
    language_code: str
) -> bool:
    # ... (trigger_ai_tutor_for_failed_assessment from main (13).py - no changes needed for these tasks)
    if not AI_TUTOR_AGENT_URL: 
        logger.warning("NovaSpark: AI_TUTOR_AGENT_URL not set. Cannot trigger AI Tutor.")
        return False
    tutor_trigger_query = (
        f"I've received feedback on my project titled '{project_title}' and it seems I need some help. "
        f"The summary said: \"{assessment_summary_html}\". Key areas I need to improve are: {'; '.join(areas_for_improvement_html)}. "
        "Can you help me understand these points better and guide me on how to improve my project?"
    )
    tutor_user_profile_payload = { 
        "industry": user_profile.industry, "profession": user_profile.profession,
        "career_interest": user_profile.career_interest,
        "current_knowledge_level": user_profile.current_knowledge_level,
        "preferred_tutor_persona": user_profile.preferred_tutor_persona,
        "hobbies_and_interests": user_profile.hobbies_and_interests # Pass new field if AI Tutor can use it
    }
    tutor_payload = { 
        "user_id": user_profile.user_id, "query_text": tutor_trigger_query,
        "user_profile_snapshot": tutor_user_profile_payload, "language_code": language_code,
        "context": {"current_project_title": project_title, "project_assessment_feedback": f"Summary: {assessment_summary_html}. Areas to focus on: {'; '.join(areas_for_improvement_html)}"}
    }
    try: 
        # NovaSpark: Adding timeout and retry to this HTTP call
        async with httpx.AsyncClient(timeout=30.0, transport=httpx.AsyncHTTPTransport(retries=2)) as client:
            response = await client.post(f"{AI_TUTOR_AGENT_URL}/v1/ask-tutor", json=tutor_payload)
            response.raise_for_status()
            logger.info(f"NovaSpark: Successfully triggered AI Tutor for user {user_profile.user_id}. Tutor response status: {response.status_code}")
            return True
    except Exception as e: 
        logger.error(f"NovaSpark Error: Failed to trigger AI Tutor for user {user_profile.user_id}. Error: {e}", exc_info=True)
        return False

# --- API Endpoints ---
# (generate_project_ideas_endpoint and assess_project_submission_endpoint remain largely the same
# as in main (13).py, as the core logic changes are in the prompt construction functions.
# Minor updates for logging and ensuring new prompt functions are called.)

@app.post("/v1/generate-project-ideas", response_model=ProjectIdeaGenerationResponse, summary="Generate Personalized Project Ideas (NovaSpark Enhanced)")
async def generate_project_ideas_endpoint(request_data: ProjectIdeaGenerationRequest): 
    start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE
    logger.info(f"NovaSpark ProjectGen: Received request for user {request_data.user_profile_snapshot.user_id} in {effective_language_code}.")

    if not GCP_PROJECT_ID or not project_idea_llm.is_initialized: 
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Project Generation service not configured or LLM client failed to initialize.")

    # Task 2 & 3: Call NovaSpark enhanced prompt functions
    system_prompt = construct_project_gen_system_prompt(effective_language_code)
    user_query = construct_project_gen_user_query(
        user_profile=request_data.user_profile_snapshot, preferences=request_data.preferences,
        course_summary=request_data.course_context_summary, topic_keywords=request_data.topic_focus_keywords,
        num_ideas=request_data.number_of_ideas, language_code=effective_language_code
    )
    
    class LLMProjectGenRoot(BaseModel): generated_projects: List[GeneratedProjectIdea]
    raw_json_string = "{}"
    generated_projects_list = []
    llm_response_metrics = {}

    try:
        llm_response_metrics = await project_idea_llm.generate_structured_response( 
            system_prompt=system_prompt, user_query_or_context=user_query,
            max_output_tokens=8192, pydantic_model_for_schema=LLMProjectGenRoot)
        raw_json_string = llm_response_metrics.get("raw_json_response", "{}")
        if not raw_json_string.strip() or raw_json_string == "{}":
             raise ValueError("LLM returned empty content for project ideas.")
        
        parsed_llm_root = LLMProjectGenRoot(**json.loads(raw_json_string))
        generated_projects_list = parsed_llm_root.generated_projects

        if not generated_projects_list:
             raise ValueError("LLM response parsed, but no project ideas found in 'generated_projects'.")

        for project in generated_projects_list: project.language_code = effective_language_code # Task 3
        
        logger.info(f"NovaSpark ProjectGen: Generated {len(generated_projects_list)} project ideas for user {request_data.user_profile_snapshot.user_id}.")
    except (json.JSONDecodeError, Exception) as e: 
        logger.error(f"NovaSpark ProjectGen Error: Parsing/validating LLM response failed. Error: {e}. Response: {raw_json_string[:500]}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate/parse project ideas (AI service error): {str(e)}")
    except Exception as e: 
        logger.error(f"NovaSpark ProjectGen Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Project idea generation failed unexpectedly: {str(e)}")

    processing_time_ms = (time.perf_counter() - start_time) * 1000
    debug_info = { 
        "llm_model_name_used": project_idea_llm.model_name, "processing_time_ms": round(processing_time_ms, 2),
        "prompt_token_count": llm_response_metrics.get("prompt_token_count"), "response_token_count": llm_response_metrics.get("response_token_count"),
        "language_generated": effective_language_code, "num_ideas_requested": request_data.number_of_ideas,
        "num_ideas_generated_valid": len(generated_projects_list)
    }
    return ProjectIdeaGenerationResponse(generated_projects=generated_projects_list, debug_info=debug_info)


@app.post("/v1/assess-project-submission", response_model=ProjectAssessmentResponse, summary="Assess User Project Submission (NovaSpark Enhanced)")
async def assess_project_submission_endpoint(request_data: ProjectAssessmentRequest): 
    start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE
    logger.info(f"NovaSpark Assessment: Received request for user {request_data.user_id}, project {request_data.project_id}.")

    if not GCP_PROJECT_ID or not assessment_llm.is_initialized: 
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Project Assessment service not configured or LLM client failed to initialize.")

    original_project_details: Optional[GeneratedProjectIdea] = None
    # NovaSpark Note: DB call to fetch original_project_details is crucial here. Using mock for now.
    # In a real system: original_project_details = await db_service.get_project_by_id(request_data.project_id)
    if request_data.project_id == "proj_mock_reco_sys_123": 
        logger.info(f"NovaSpark Assessment: Using mock original project details for {request_data.project_id}")
        original_project_details = GeneratedProjectIdea( 
            project_id="proj_mock_reco_sys_123", title="Mock Recommendation System Design",
            description_html="<p>Design a basic recommendation engine.</p>", difficulty_level="Intermediate",
            learning_objectives_html=["Understand collaborative filtering.", "Outline data needs."],
            expected_deliverables_html=["A 2-page PDF report detailing the system design and algorithm choice.", "Pseudo-code for the core recommendation logic."],
            key_tasks=[GeneratedProjectTask(task_id="t1", title="Research", description="Research techniques.")],
            assessment_rubric_html=["Clarity of system design explanation (40%)", "Feasibility and correctness of pseudo-code (40%)", "Overall report quality and adherence to deliverables (20%)"],
            language_code=effective_language_code 
        )
    if not original_project_details:
        logger.error(f"NovaSpark Assessment Error: Original project details not found for project_id: {request_data.project_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Original project definition not found for project ID {request_data.project_id}.")

    # Task 3 & 4: Call NovaSpark enhanced prompt functions
    system_prompt = construct_project_assessment_system_prompt(effective_language_code, request_data.user_profile_snapshot.preferred_tutor_persona)
    user_query = construct_project_assessment_user_query(
        original_project=original_project_details, submission_items=request_data.submission_items,
        user_profile=request_data.user_profile_snapshot, language_code=effective_language_code
    )
    
    class LLMAssessmentRoot(BaseModel): 
        overall_competency_score: float; feedback_summary_html: str
        detailed_feedback_points: List[AssessmentFeedbackPoint]; skills_demonstrated: List[str]
        critical_areas_for_improvement_html: List[str]; positive_points_highlighted_html: List[str]

    raw_json_string = "{}"
    llm_response_metrics = {}
    parsed_llm_assessment = None
    try:
        llm_response_metrics = await assessment_llm.generate_structured_response( 
            system_prompt=system_prompt, user_query_or_context=user_query,
            max_output_tokens=4096, pydantic_model_for_schema=LLMAssessmentRoot)
        raw_json_string = llm_response_metrics.get("raw_json_response", "{}")
        if not raw_json_string.strip() or raw_json_string == "{}":
            raise ValueError("LLM returned empty content for project assessment.")
        
        assessment_data_from_llm = json.loads(raw_json_string)
        parsed_llm_assessment = LLMAssessmentRoot(**assessment_data_from_llm) # Validation
    except (json.JSONDecodeError, Exception) as e: 
        logger.error(f"NovaSpark Assessment Error: Parsing/validating LLM response: {e}. Response: {raw_json_string[:500]}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to parse/validate assessment data from AI: {str(e)}")
    except Exception as e: 
        logger.error(f"NovaSpark Assessment Unexpected LLM error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Project assessment failed (AI service error): {str(e)}")

    overall_score_from_llm = parsed_llm_assessment.overall_competency_score
    is_user_competent = overall_score_from_llm >= COMPETENCY_THRESHOLD
    trigger_tutor_flag = not is_user_competent # Renamed for clarity

    assessment_result_obj = ProjectAssessmentResult(
        project_id=request_data.project_id, user_id=request_data.user_id,
        overall_competency_score=overall_score_from_llm, is_passed=is_user_competent,
        feedback_summary_html=parsed_llm_assessment.feedback_summary_html,
        detailed_feedback_points=parsed_llm_assessment.detailed_feedback_points,
        skills_demonstrated=parsed_llm_assessment.skills_demonstrated,
        critical_areas_for_improvement_html=parsed_llm_assessment.critical_areas_for_improvement_html,
        positive_points_highlighted_html=parsed_llm_assessment.positive_points_highlighted_html,
        tutor_session_triggered=False, # Initialize, will be updated
        language_code=effective_language_code # Task 3
    )

    if trigger_tutor_flag: 
        logger.info(f"NovaSpark Assessment: User {request_data.user_id} for project {request_data.project_id} scored {overall_score_from_llm:.2f}. Triggering AI Tutor.")
        tutor_triggered_successfully = await trigger_ai_tutor_for_failed_assessment(
            user_profile=request_data.user_profile_snapshot, project_title=original_project_details.title,
            assessment_summary_html=assessment_result_obj.feedback_summary_html,
            areas_for_improvement_html=assessment_result_obj.critical_areas_for_improvement_html,
            language_code=effective_language_code)
        assessment_result_obj.tutor_session_triggered = tutor_triggered_successfully 
    
    # NovaSpark Note: Persist assessment_result_obj to DB here.

    processing_time_ms = (time.perf_counter() - start_time) * 1000
    debug_info = { 
        "llm_model_name_used": assessment_llm.model_name, "processing_time_ms": round(processing_time_ms, 2),
        "prompt_token_count": llm_response_metrics.get("prompt_token_count"), "response_token_count": llm_response_metrics.get("response_token_count"),
        "language_assessed_in": effective_language_code
    }
    return ProjectAssessmentResponse(assessment_result=assessment_result_obj, debug_info=debug_info)

@app.get("/health", status_code=status.HTTP_200_OK) 
async def health_check():
    # ... (health check from main (13).py - no changes needed for these tasks)
    service_name = "ProjectGenerationAssessmentAgent_NovaSpark_Enhanced" # Task 4: Updated name
    if not GCP_PROJECT_ID: 
        return {"status": "unhealthy", "reason": "GCP_PROJECT_ID not configured.", "service": service_name}
    if not AI_TUTOR_AGENT_URL:
        logger.warning("NovaSpark Health Warning: AI_TUTOR_AGENT_URL not set. Tutor triggering will not function.")
    # Check LLM client initializations
    if not project_idea_llm.is_initialized or not assessment_llm.is_initialized:
        return {"status": "unhealthy", "reason": "One or more LLM clients failed to initialize.", "service": service_name}
    return {"status": "healthy", "service": service_name, "innovate_ai_enhancements_active": True}

if __name__ == "__main__": 
    import uvicorn
    logger.info(f"NovaSpark: Starting {app.title} v{app.version} for local development...")
    if not GCP_PROJECT_ID: 
        print("NovaSpark Warning: GCP_PROJECT_ID is not set for Project Agent. LLM calls will fail.")
    if not AI_TUTOR_AGENT_URL: 
        print("NovaSpark Warning: AI_TUTOR_AGENT_URL is not set. Tutor triggering will not work.")
    
    port = int(os.getenv("PORT", 8004)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
