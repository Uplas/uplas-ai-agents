# uplas-ai-agents/personalized_tutor_nlp_llm/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import httpx # For fetching NLP content via backend and potentially other internal calls
import os
import time
from enum import Enum
import logging
import json # For parsing LLM's JSON output and NLP agent's output

# GCP Clients
from google.cloud import aiplatform
import google.auth
import google.auth.transport.requests

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") #
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") #
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-001") #

# InnovateAI Enhancement: URL to fetch processed NLP content (from an internal backend endpoint)
NLP_CONTENT_SERVICE_URL = os.getenv("NLP_CONTENT_SERVICE_URL", "http://your-backend-service/api/internal/get-processed-nlp-content")


# Initialize Vertex AI
if not GCP_PROJECT_ID: #
    logging.warning("InnovateAI Warning: GCP_PROJECT_ID environment variable not set. LLM calls may fail.")
else:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION) #

# Supported languages (BCP-47 codes)
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] #
DEFAULT_LANGUAGE = "en-US" #

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO) #
logger = logging.getLogger(__name__) #

# --- Pydantic Models for NLP Agent's Output (as consumed by AI Tutor) ---
# These models define the structure of the content processed by the NLP_Content_Agent
class NlpTopicProcessed(BaseModel):
    topic_id: str # Added to identify the topic
    topic_title: str
    key_concepts: List[str]
    content_with_tags: str # This contains the text with <analogy>, <example>, <visual_aid_suggestion> etc. tags

class NlpLessonProcessed(BaseModel):
    lesson_id: str # Added
    lesson_title: str
    lesson_summary: Optional[str] = None
    topics: List[NlpTopicProcessed]

class NlpModuleProcessed(BaseModel): # This is what the AI Tutor will fetch
    module_id: str
    module_title: Optional[str] = None
    language_code: str
    lessons: List[NlpLessonProcessed]


# --- Pydantic Models for AI Tutor API Contract (Existing models from main (3).py, with InnovateAI refinements) ---

class UserProfileSnapshot(BaseModel): #
    industry: Optional[str] = Field(None, examples=["Technology"])
    profession: Optional[str] = Field(None, examples=["Software Engineer"])
    country: Optional[str] = Field(None, examples=["Kenya"])
    city: Optional[str] = Field(None, examples=["Nairobi"])
    preferred_tutor_persona: Optional[str] = Field("Friendly and encouraging", examples=["Socratic", "Technical"])
    learning_style_preference: Optional[Dict[str, float]] = Field(default_factory=dict, examples=[{"visual": 0.7, "kinesthetic": 0.3}])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "TopicX": "Beginner"}])
    career_interest: Optional[str] = Field(None, examples=["AI Specialist"])
    learning_goals: Optional[str] = Field(None, examples=["Master machine learning concepts."])

class TutorRequestContext(BaseModel): #
    course_id: Optional[str] = None
    module_id: Optional[str] = None # InnovateAI: Key to fetch processed NLP module
    topic_id: Optional[str] = None  # InnovateAI: Key to focus on a specific processed topic within the module
    project_id: Optional[str] = None
    current_topic_title: Optional[str] = Field(None, examples=["Introduction to Python Lists"]) # Can be auto-populated from NLP content
    current_course_title: Optional[str] = Field(None, examples=["Python for Beginners"]) # Can be auto-populated
    current_project_title: Optional[str] = Field(None, examples=["Data Analysis Mini-Project"])
    project_assessment_feedback: Optional[str] = Field(None, description="Feedback from a failed project assessment to guide the tutor.")

class ConversationRole(str, Enum): #
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationTurn(BaseModel): #
    role: ConversationRole
    content: str

class AiTutorQueryRequest(BaseModel): #
    user_id: str = Field(..., examples=["user_uuid_123"])
    session_id: Optional[str] = Field(None, examples=["session_uuid_abc"])
    query_text: str = Field(..., min_length=1, examples=["Can you explain Python decorators?"])
    context: Optional[TutorRequestContext] = None
    user_profile_snapshot: UserProfileSnapshot
    conversation_history: Optional[List[ConversationTurn]] = Field(default_factory=list)
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    max_tokens_response: Optional[int] = Field(1024, ge=50, le=8192)

    @validator('language_code')
    def validate_language_code(cls, v): #
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"InnovateAI: Unsupported language_code '{v}'. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class GeneratedAnalogy(BaseModel): #
    analogy: str
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="InnovateAI: Relevance score might be hard for LLM to provide accurately.")

# InnovateAI Enhancement: Structured output model for LLM responses
class StructuredLLMOutput(BaseModel):
    main_answer_text: str = Field(..., description="The primary textual answer to the user's query.")
    suggested_follow_ups: List[str] = Field(default_factory=list, description="List of relevant follow-up questions.")
    generated_analogies_for_answer: List[str] = Field(default_factory=list, description="List of analogies generated within the answer.")
    answer_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="InnovateAI: LLM's confidence in the answer (0.0-1.0), if inferable or provided.")
    # Potentially add other structured fields the LLM could populate, e.g., key terms identified

class DebugInfo(BaseModel): #
    prompt_sent_to_llm_sample: Optional[str] = None
    llm_model_name_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    language_used: Optional[str] = None

class AiTutorQueryResponse(BaseModel): #
    answer_text: str # Will be populated from StructuredLLMOutput.main_answer_text
    suggested_follow_up_questions: List[str] # From StructuredLLMOutput
    generated_analogies: List[GeneratedAnalogy] # Mapped from StructuredLLMOutput
    confidence_score: Optional[float] # From StructuredLLMOutput
    debug_info: Optional[DebugInfo] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Tutor Agent (Vertex AI Integrated) - InnovateAI Enhanced",
    description="Provides hyper-personalized explanations and guidance using Google Cloud Vertex AI (Gemini), powered by NLP-structured content and advanced prompting.",
    version="0.3.0" # Incremented version reflecting enhancements
)

# --- Vertex AI LLM Client Logic (InnovateAI Enhanced for Structured JSON Output) ---
class VertexAITutorLLMClient: # Renamed for clarity from generic VertexAILLMClient
    def __init__(self, model_name: str):
        self.model_name = model_name #

    async def generate_structured_response( # Renamed from generate_response
        self,
        system_prompt: str,
        conversation_turns: List[Dict[str, Any]],
        max_output_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        # InnovateAI Enhancement: Expect a Pydantic model schema for JSON output
        response_pydantic_model: Optional[Any] = None # e.g., StructuredLLMOutput
    ) -> Dict[str, Any]:
        if not GCP_PROJECT_ID: #
            raise EnvironmentError("InnovateAI Critical: GCP_PROJECT_ID is not configured for VertexAITutorLLMClient.")

        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold #

        model = GenerativeModel(
            self.model_name,
            system_instruction=[Part.from_text(system_prompt)] if system_prompt else None #
        )
        
        generation_config = GenerationConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ) #

        # InnovateAI Enhancement: Configure Gemini for JSON output if schema provided
        if response_pydantic_model:
            try:
                generation_config.response_mime_type = "application/json"
                generation_config.response_schema = response_pydantic_model.model_json_schema()
                logger.info(f"InnovateAI: Configured Gemini for JSON output using schema for {response_pydantic_model.__name__}")
            except Exception as e_schema:
                logger.error(f"InnovateAI Error: Failed to set JSON schema for model {response_pydantic_model.__name__}. Error: {e_schema}", exc_info=True)
                # Fallback to text if schema setup fails, or raise error
                generation_config.response_mime_type = "text/plain"


        history_for_model = [] #
        for turn in conversation_turns:
            role = 'user' if turn['role'] == ConversationRole.USER.value else 'model'
            history_for_model.append({'role': role, 'parts': [Part.from_text(turn['content'])]})

        safety_settings = { #
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        if len(history_for_model) > 1: #
            chat = model.start_chat(history=history_for_model[:-1])
            current_user_message = history_for_model[-1]['parts'][0].text
            llm_response = await chat.send_message_async(
                [Part.from_text(current_user_message)],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        elif history_for_model: #
            current_user_message = history_for_model[0]['parts'][0].text
            llm_response = await model.generate_content_async(
                [Part.from_text(current_user_message)],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        else:
            return {"raw_response_text": "InnovateAI Info: No input provided to LLM.", "prompt_token_count": 0, "response_token_count": 0}

        raw_response_text = "" # This will be the JSON string if schema was successfully used
        prompt_tokens = 0
        response_tokens = 0

        if llm_response.candidates: #
            candidate = llm_response.candidates[0]
            if candidate.content and candidate.content.parts:
                raw_response_text = "".join([part.text for part in candidate.content.parts if part.text])
            
            if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata: #
                prompt_tokens = llm_response.usage_metadata.prompt_token_count
                response_tokens = llm_response.usage_metadata.candidates_token_count
        
        return {
            "raw_response_text": raw_response_text, # Caller will parse this if it's JSON
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

llm_client = VertexAITutorLLMClient(model_name=LLM_MODEL_NAME) #

# --- Helper Functions (InnovateAI Enhanced) ---

async def fetch_processed_nlp_content(
    module_id: Optional[str],
    topic_id_to_focus: Optional[str], # Specific topic to find within the module
    language_code: str
) -> Optional[NlpModuleProcessed]:
    """
    InnovateAI Enhanced: Fetches NLP-processed structured content for a module.
    This function would ideally call an internal Uplas backend API endpoint
    that retrieves the processed JSON from GCS (where NLP_Content_Agent stored it)
    based on module_id and language_code.
    """
    if not module_id:
        logger.info("InnovateAI: No module_id provided to fetch NLP content.")
        return None
    
    # Example: GET NLP_CONTENT_SERVICE_URL/?module_id=<module_id>&lang=<language_code>
    # The response would be the JSON content of the NlpModuleProcessed.
    request_url = f"{NLP_CONTENT_SERVICE_URL}?module_id={module_id}&language_code={language_code}"
    logger.info(f"InnovateAI: Attempting to fetch NLP-processed content from: {request_url}")

    # **** Placeholder for actual HTTP call to internal service - Mugambi to implement ****
    # This service would in turn fetch the JSON file from GCS.
    # async with httpx.AsyncClient(timeout=15.0) as client:
    #     try:
    #         response = await client.get(request_url)
    #         response.raise_for_status() # Raise an exception for bad status codes
    #         nlp_data_dict = response.json()
    #         # Validate with Pydantic model
    #         processed_module = NlpModuleProcessed(**nlp_data_dict)
    #         logger.info(f"InnovateAI: Successfully fetched and parsed NLP content for module_id: {module_id}")
    #         return processed_module
    #     except httpx.HTTPStatusError as e:
    #         logger.error(f"InnovateAI HTTP Error fetching NLP content for module {module_id}: {e.response.status_code} - {e.response.text}", exc_info=True)
    #         return None
    #     except (json.JSONDecodeError, Exception) as e: # Includes Pydantic validation errors
    #         logger.error(f"InnovateAI Error parsing/validating NLP content for module {module_id}: {e}", exc_info=True)
    #         return None
    # Mocked response for InnovateAI framework:
    if module_id == "course101_module3_processed" and language_code == "en-US": # Example known ID
        logger.info(f"InnovateAI Placeholder: Using mock NLP content for module_id: {module_id}")
        mock_nlp_json = {
          "module_id": "course101_module3_processed", "module_title": "Intro to Quantum (Processed)", "language_code": "en-US",
          "lessons": [{
              "lesson_id": "L1_proc", "lesson_title": "What is a Qubit? (Processed)", "lesson_summary": "Qubits are fundamental.",
              "topics": [{
                  "topic_id": "L1_T1_proc", "topic_title": "Understanding Superposition (Processed)",
                  "key_concepts": ["Qubits can be 0, 1, or both.", "Superposition enables power."],
                  "content_with_tags": "Classical bits are 0 or 1. <analogy type=\"comparison_to_classical_needed\" /> Qubits use superposition. <visual_aid_suggestion type=\"animated_diagram\" description=\"Show Bloch sphere\" /> <interactive_question_opportunity text_suggestion=\"How is a qubit different?\" /> This is <difficulty type=\"foundational_info\">key</difficulty>."
                },{
                  "topic_id": "L1_T2_proc", "topic_title": "Entanglement Fun (Processed)",
                  "key_concepts": ["Spooky action.", "Instant connection."],
                  "content_with_tags": "Entanglement links qubits. <example domain=\"physics_analogy_needed\" /> <difficulty type=\"advanced_detail\">Tricky concept</difficulty>!"
                }]
            }]
        }
        return NlpModuleProcessed(**mock_nlp_json)
    # **** END Placeholder ****
    logger.warning(f"InnovateAI: No NLP content (mock or real) found for module_id: {module_id}, lang: {language_code}")
    return None


def construct_innovateai_system_prompt( # Renamed from construct_system_prompt_for_llm
    user_profile: UserProfileSnapshot,
    language_code: str,
    nlp_module_content: Optional[NlpModuleProcessed] = None, # InnovateAI: Expects structured, tagged content
    current_topic_id_in_focus: Optional[str] = None, # To pinpoint specific section in nlp_module_content
    request_context: Optional[TutorRequestContext] = None, # For project feedback etc.
) -> str:
    system_message_parts = [
        "You are Uplas AI Tutor, an exceptionally skilled, empathetic, and personalized AI learning partner. Your persona is dynamic, guided by the user's preference.",
        f"Your primary mission is to illuminate complex concepts with clarity, tailoring explanations and examples to the user's unique profile (profession, industry, interests, learning goals). Respond ONLY in [{language_code}].",
        f"Embody the user's preferred communication style: '{user_profile.preferred_tutor_persona}'.",
        "Break down information into digestible chunks. Be patient, encouraging, and deeply supportive.",
        "If a question is unclear, gently ask for clarification in [{language_code}].",
        "If a question is off-topic (personal advice, harmful, unrelated), politely state in [{language_code}] your focus on Uplas educational content and guide them back.",
    ]

    # User Profile Block (similar to existing)
    profile_str = "... (InnovateAI: User profile details like profession, industry, career interest would be dynamically inserted here) ..."
    system_message_parts.append(f"\n--- User Profile Context ---\n{profile_str}")

    # InnovateAI Enhancement: Utilizing NLP-processed and tagged content
    relevant_content_snippet = "No specific course material was found for this query."
    if nlp_module_content and nlp_module_content.lessons:
        target_topic_content = None
        # Try to find the specific topic in focus
        if current_topic_id_in_focus:
            for lesson in nlp_module_content.lessons:
                for topic in lesson.topics:
                    if topic.topic_id == current_topic_id_in_focus:
                        target_topic_content = f"Topic: {topic.topic_title}\nKey Concepts: {', '.join(topic.key_concepts)}\nMaterial: {topic.content_with_tags}"
                        system_message_parts.append(f"\n--- Focused Learning Material ({topic.topic_title}) ---")
                        system_message_parts.append(target_topic_content[:3500]) # Truncate for prompt size
                        break
                if target_topic_content: break
        
        if not target_topic_content: # Fallback if specific topic not found or ID not given
            # Provide a general summary or first lesson's overview
            first_lesson = nlp_module_content.lessons[0]
            lesson_summary = f"Module: {nlp_module_content.module_title or 'Current Module'}\nLesson: {first_lesson.lesson_title}\nSummary: {first_lesson.lesson_summary or 'Refer to topics.'}"
            system_message_parts.append(f"\n--- General Module Context ---\n{lesson_summary}")
            # You might append the first topic's content_with_tags as well if space allows.

        system_message_parts.append(
            "\n--- InnovateAI Content Interaction Instructions ---"
            "The 'Material' or 'Content' above may contain special tags: "
            "- `<analogy type=\"...\" />` or `<example domain=\"...\" />`: When you see these, proactively generate a relevant analogy/example based on the user's profile and the tag's suggestion."
            "- `<interactive_question_opportunity text_suggestion=\"...\" />`: Consider asking the suggested question or a similar one to engage the user."
            "- `<visual_aid_suggestion type=\"...\" description=\"...\" />`: Acknowledge if a visual would be helpful; you can describe what it might show."
            "- `<difficulty type=\"foundational_info|advanced_detail\" />`: Use this to tailor the depth of your explanation. If user seems to be a beginner (from profile or query), focus on foundational, otherwise cover advanced details if relevant."
            "Seamlessly integrate these actions into your response."
        )

    # InnovateAI Enhancement: Empathetic Error Attribution for Project Feedback
    if request_context and request_context.project_assessment_feedback:
        system_message_parts.append("\n--- InnovateAI Guidance for Project Remediation ---")
        system_message_parts.append(
            "The user is seeking help after a project assessment. "
            f"Project: '{request_context.current_project_title or 'N/A'}'. "
            f"Assessment Feedback: '{request_context.project_assessment_feedback}'. "
            "Adopt an **Empathetic Mentor** role. Do NOT just repeat the feedback. "
            "1. Acknowledge their effort and the challenge."
            "2. Gently help them diagnose *why* they might have struggled, linking to concepts from the course material (provided above, if relevant)."
            "3. Clearly explain the core concepts they need to revisit or solidify."
            "4. Offer concrete, actionable steps, simpler examples, or prerequisite knowledge they can review to improve."
            "5. Maintain a highly supportive, encouraging, and patient tone. Focus on building their confidence."
        )

    # InnovateAI Enhancement: Instructing LLM to return JSON
    system_message_parts.append(
        "\n--- InnovateAI Output Format Mandate ---"
        "Your entire response MUST be a single, valid JSON object. Do not add any text before or after this JSON object. "
        "The JSON object must strictly adhere to the following schema (Pydantic model: StructuredLLMOutput): "
        f"{StructuredLLMOutput.model_json_schema(indent=2)}" # Provide the schema
        f"All text content within this JSON (main_answer_text, analogies, follow-ups) MUST be in [{language_code}]."
    )
    return "\n".join(system_message_parts)


# --- API Endpoint (InnovateAI Enhanced) ---
@app.post("/v1/ask-tutor", response_model=AiTutorQueryResponse, summary="Get a personalized answer from the AI Tutor (InnovateAI Enhanced)")
async def ask_tutor_endpoint(
    request_data: AiTutorQueryRequest,
): #
    processing_start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE #

    nlp_processed_module: Optional[NlpModuleProcessed] = None
    current_topic_id: Optional[str] = None

    if request_data.context and request_data.context.module_id:
        current_topic_id = request_data.context.topic_id # May be None
        # InnovateAI: Fetch structured NLP content
        nlp_processed_module = await fetch_processed_nlp_content(
            request_data.context.module_id,
            current_topic_id, # Pass topic_id for potential focusing within fetch or for logging
            effective_language_code
        )
        if nlp_processed_module and not request_data.context.current_topic_title and current_topic_id:
            # Auto-populate current_topic_title from fetched NLP content if not provided in request
            for lesson in nlp_processed_module.lessons:
                for topic in lesson.topics:
                    if topic.topic_id == current_topic_id:
                        request_data.context.current_topic_title = topic.topic_title
                        if nlp_processed_module.module_title:
                             request_data.context.current_course_title = nlp_processed_module.module_title # Assuming module title as course title for now
                        break
                if request_data.context.current_topic_title: break
    
    system_prompt_for_llm = construct_innovateai_system_prompt( #
        user_profile=request_data.user_profile_snapshot,
        language_code=effective_language_code,
        nlp_module_content=nlp_processed_module,
        current_topic_id_in_focus=current_topic_id,
        request_context=request_data.context
    )

    llm_conversation_history: List[Dict[str, Any]] = [] #
    for turn in request_data.conversation_history:
        role = 'model' if turn.role == ConversationRole.ASSISTANT else turn.role.value
        llm_conversation_history.append({'role': role, 'content': turn.content})
    llm_conversation_history.append({'role': ConversationRole.USER.value, 'content': request_data.query_text})

    structured_llm_output_obj: Optional[StructuredLLMOutput] = None
    llm_response_metrics = {}

    try:
        llm_response_metrics = await llm_client.generate_structured_response( #
            system_prompt=system_prompt_for_llm,
            conversation_turns=llm_conversation_history,
            max_output_tokens=request_data.max_tokens_response,
            response_pydantic_model=StructuredLLMOutput # InnovateAI: Pass the schema model
        )
        
        raw_json_string = llm_response_metrics.get("raw_response_text", "{}")
        try:
            llm_output_dict = json.loads(raw_json_string)
            # InnovateAI: Validate and parse the JSON into our Pydantic model
            structured_llm_output_obj = StructuredLLMOutput(**llm_output_dict)
        except json.JSONDecodeError as e_json:
            logger.error(f"InnovateAI Error: LLM returned invalid JSON for structured output: {raw_json_string[:500]}. Error: {e_json}", exc_info=True)
            # Fallback: Treat the whole thing as plain text if JSON parsing fails, and log the error.
            structured_llm_output_obj = StructuredLLMOutput(main_answer_text=f"InnovateAI Apology: I couldn't structure my thoughts perfectly, but here's what I found: {raw_json_string}")
        except Exception as e_pydantic: # Catch Pydantic validation errors
            logger.error(f"InnovateAI Error: Pydantic validation failed for LLM JSON output: {raw_json_string[:500]}. Error: {e_pydantic}", exc_info=True)
            structured_llm_output_obj = StructuredLLMOutput(main_answer_text=f"InnovateAI Alert: My response structure was a bit off. Here's the raw info: {raw_json_string}")

    except EnvironmentError as e_env: #
        logger.error(f"InnovateAI Config Error for LLM: {e_env}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI service configuration error: {e_env}")
    except Exception as e_llm: #
        logger.error(f"InnovateAI LLM Communication/Processing Error: {e_llm}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="There was an issue communicating with the AI assistance service.")

    if not structured_llm_output_obj: # Should ideally be handled by fallbacks above
        structured_llm_output_obj = StructuredLLMOutput(main_answer_text="InnovateAI Note: I'm sorry, I couldn't formulate a structured response at this moment.")

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000 #

    debug_information = DebugInfo( #
        llm_model_name_used=llm_client.model_name,
        processing_time_ms=round(processing_time_ms, 2),
        prompt_token_count=llm_response_metrics.get("prompt_token_count"),
        response_token_count=llm_response_metrics.get("response_token_count"),
        language_used=effective_language_code
    )
    if os.getenv("UPLAS_DEBUG_MODE", "false").lower() == "true": #
        # Log more extensive prompt if needed for debugging.
        debug_information.prompt_sent_to_llm_sample = (system_prompt_for_llm + "\n" + \
            "\n".join([f"Role: {turn['role']}, Content: {turn['content'][:100]}..." for turn in llm_conversation_history]))[:1500] + "..."

    return AiTutorQueryResponse(
        answer_text=structured_llm_output_obj.main_answer_text,
        suggested_follow_up_questions=structured_llm_output_obj.suggested_follow_ups,
        generated_analogies=[GeneratedAnalogy(analogy=text) for text in structured_llm_output_obj.generated_analogies_for_answer],
        confidence_score=structured_llm_output_obj.answer_confidence_score,
        debug_info=debug_information
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK) #
async def health_check():
    # InnovateAI Enhancement: Add check for NLP_CONTENT_SERVICE_URL if critical
    if not GCP_PROJECT_ID:
        return {"status": "unhealthy", "reason": "GCP_PROJECT_ID not configured.", "service": "PersonalizedTutorNLP_LLM_Agent_InnovateAI"}
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL: # Check if it's default
         logger.warning("InnovateAI Health Warning: NLP_CONTENT_SERVICE_URL is not configured or using default. NLP content fetching will be mocked/fail.")
         # Don't mark as unhealthy for this yet unless it's absolutely critical for basic operation
    return {"status": "healthy", "service": "PersonalizedTutorNLP_LLM_Agent_InnovateAI", "innovate_ai_enhancements_active": True}


if __name__ == "__main__": #
    import uvicorn
    logger.info("InnovateAI: Starting Personalized AI Tutor Agent for local development...")
    if not GCP_PROJECT_ID:
        print("InnovateAI Warning: GCP_PROJECT_ID is not set. LLM calls will fail.")
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL:
        print("InnovateAI Warning: NLP_CONTENT_SERVICE_URL is not set or is default. NLP content fetching will be mocked or fail.")
    
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
