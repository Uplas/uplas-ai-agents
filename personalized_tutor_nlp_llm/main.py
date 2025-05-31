# uplas-ai-agents/personalized_tutor_nlp_llm/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import httpx # For internal calls to fetch NLP-processed content via backend
import os
import time
from enum import Enum
import logging
import json # For parsing NLP agent's JSON output

# GCP Clients
from google.cloud import aiplatform
import google.auth
import google.auth.transport.requests

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-001")

# Assuming nlp_content_agent output models are defined elsewhere or simplified here
# For this example, we'll assume the structure from the NLP agent's ProcessedModule
# This might live in a shared Pydantic models library eventually.
class NlpTopicProcessed(BaseModel): # Simplified for tutor's consumption context
    topic_title: str
    key_concepts: List[str]
    content_with_tags: str

class NlpLessonProcessed(BaseModel):
    lesson_title: str
    topics: List[NlpTopicProcessed]

class NlpModuleProcessed(BaseModel): # What the tutor will fetch
    module_id: str
    language_code: str
    lessons: List[NlpLessonProcessed]


# Initialize Vertex AI (from existing code)
if not GCP_PROJECT_ID:
    logging.warning("GCP_PROJECT_ID environment variable not set. LLM calls may fail.")
else:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models (largely from existing code, with minor notes) ---

class UserProfileSnapshot(BaseModel): # As per existing
    industry: Optional[str] = Field(None, examples=["Technology"])
    profession: Optional[str] = Field(None, examples=["Software Engineer"])
    country: Optional[str] = Field(None, examples=["Kenya"])
    city: Optional[str] = Field(None, examples=["Nairobi"])
    preferred_tutor_persona: Optional[str] = Field("Friendly and encouraging", examples=["Socratic", "Technical"])
    learning_style_preference: Optional[Dict[str, float]] = Field(default_factory=dict, examples=[{"visual": 0.7, "kinesthetic": 0.3}])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "TopicX": "Beginner"}])
    career_interest: Optional[str] = Field(None, examples=["AI Specialist"])
    learning_goals: Optional[str] = Field(None, examples=["Master machine learning concepts."])

class TutorRequestContext(BaseModel): # As per existing, will be used to fetch NLP content
    course_id: Optional[str] = None
    topic_id: Optional[str] = None # Key to fetch specific NLP-processed topic
    module_id: Optional[str] = None # Alternative key to fetch whole NLP-processed module
    project_id: Optional[str] = None
    current_topic_title: Optional[str] = Field(None, examples=["Introduction to Python Lists"]) # Can be derived from NLP content
    current_course_title: Optional[str] = Field(None, examples=["Python for Beginners"])
    current_project_title: Optional[str] = Field(None, examples=["Data Analysis Mini-Project"])
    project_assessment_feedback: Optional[str] = Field(None, description="Feedback from a failed project assessment to guide the tutor.")

class ConversationRole(str, Enum): # As per existing
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ConversationTurn(BaseModel): # As per existing
    role: ConversationRole
    content: str

class AiTutorQueryRequest(BaseModel): # As per existing
    user_id: str = Field(..., examples=["user_uuid_123"])
    session_id: Optional[str] = Field(None, examples=["session_uuid_abc"])
    query_text: str = Field(..., min_length=1, examples=["Can you explain Python decorators?"])
    context: Optional[TutorRequestContext] = None
    user_profile_snapshot: UserProfileSnapshot
    conversation_history: Optional[List[ConversationTurn]] = Field(default_factory=list)
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    max_tokens_response: Optional[int] = Field(1024, ge=50, le=8192)

    @validator('language_code')
    def validate_language_code(cls, v): # As per existing
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' received. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class GeneratedAnalogy(BaseModel): # As per existing
    analogy: str
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)

# NEW/ENHANCED: Structure for what LLM should return for better parsing
class StructuredLLMOutput(BaseModel):
    main_answer_text: str
    suggested_follow_ups: List[str] = Field(default_factory=list)
    generated_analogies_for_answer: List[str] = Field(default_factory=list)
    # Optional: if LLM can provide a confidence on its answer
    answer_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class DebugInfo(BaseModel): # As per existing
    prompt_sent_to_llm_sample: Optional[str] = None
    llm_model_name_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    language_used: Optional[str] = None

class AiTutorQueryResponse(BaseModel): # Modified to use StructuredLLMOutput's fields
    answer_text: str # This will be main_answer_text
    suggested_follow_up_questions: List[str] = Field(default_factory=list)
    generated_analogies: List[GeneratedAnalogy] = Field(default_factory=list) # Map from List[str]
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    debug_info: Optional[DebugInfo] = None


# --- FastAPI Application (existing structure) ---
app = FastAPI(
    title="Uplas AI Tutor Agent (Vertex AI Integrated) - Enhanced",
    description="Provides personalized explanations and guidance using Google Cloud Vertex AI (Gemini), powered by NLP-structured content.",
    version="0.3.0" # Incremented version
)

# --- Vertex AI LLM Client Logic (Adapted from project_generator_agent for JSON output) ---
class VertexAITutorLLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate_structured_response( # Adapted from project_generator_agent
        self,
        system_prompt: str,
        conversation_turns: List[Dict[str, Any]],
        max_output_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        response_schema: Optional[Dict[str, Any]] = None # For Gemini JSON mode
    ) -> Dict[str, Any]:
        if not GCP_PROJECT_ID:
            raise EnvironmentError("GCP_PROJECT_ID is not configured.")

        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold #

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
        if response_schema: # Requesting JSON output
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = response_schema
        
        history_for_model = []
        for turn in conversation_turns:
            role = 'user' if turn['role'] == ConversationRole.USER.value else 'model'
            history_for_model.append({'role': role, 'parts': [Part.from_text(turn['content'])]})

        # Simplified logic from existing tutor:
        if len(history_for_model) > 1:
            chat = model.start_chat(history=history_for_model[:-1])
            current_user_message = history_for_model[-1]['parts'][0].text
            llm_response = await chat.send_message_async(
                [Part.from_text(current_user_message)],
                generation_config=generation_config
                # safety_settings can be added here as in existing code
            )
        elif history_for_model:
            current_user_message = history_for_model[0]['parts'][0].text
            llm_response = await model.generate_content_async(
                [Part.from_text(current_user_message)],
                generation_config=generation_config
                # safety_settings
            )
        else:
             return {"structured_output_json": "{}", "prompt_token_count": 0, "response_token_count": 0}


        response_text = "" # This will be the JSON string if schema is used
        prompt_tokens = 0
        response_tokens = 0

        if llm_response.candidates:
            candidate = llm_response.candidates[0]
            if candidate.content and candidate.content.parts:
                response_text = "".join([part.text for part in candidate.content.parts if part.text])
            
            if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata: #
                prompt_tokens = llm_response.usage_metadata.prompt_token_count
                response_tokens = llm_response.usage_metadata.candidates_token_count
        
        # The 'response_text' should now be a JSON string conforming to StructuredLLMOutput
        return {
            "structured_output_json": response_text, # LLM returns a JSON string
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

llm_client = VertexAITutorLLMClient(model_name=LLM_MODEL_NAME)


# --- Helper Functions ---

async def fetch_processed_nlp_content(
    module_id: Optional[str],
    topic_id: Optional[str],
    language_code: str
) -> Optional[NlpModuleProcessed]: # Returns the whole module, tutor can pick relevant parts
    """
    ENHANCED: Fetches NLP-processed content.
    In a real scenario, this would call a backend API endpoint that retrieves
    the processed JSON from GCS based on module_id or topic_id.
    e.g., GET /api/internal/processed-content/?module_id=<module_id>&lang=<language_code>
    """
    logger.info(f"Attempting to fetch NLP-processed content for module: {module_id}, topic: {topic_id}, lang: {language_code}")
    # **** Placeholder for actual backend API call to get GCS URI & fetch content ****
    # For now, returning a mock structure if a specific ID is queried.
    if module_id == "course101_module3" or topic_id == "L1_T1_processed": # Example ID
        mock_data_path = f"mock_nlp_output_{language_code}.json" # Simplified mock
        try:
            # This mock implies the NLP agent's output (a JSON file) is somehow available.
            # In reality, you'd use httpx to call an internal API.
            # For this example, let's imagine a simplified direct GCS fetch or local mock.
            # For example, if topic_id == "L1_T1_processed" for "en-US":
            if language_code == "en-US" and (module_id == "course101_module3" or topic_id == "L1_T1_processed"):
                mock_nlp_json = {
                  "moduleId": "course101_module3",
                  "language_code": "en-US",
                  "lessons": [
                    {
                      "lesson_id": "L1",
                      "lesson_title": "What is a Qubit?",
                      "topics": [
                        {
                          "topic_id": "L1_T1_processed",
                          "topic_title": "Understanding Superposition (Processed)",
                          "key_concepts": [
                            "Qubits can represent 0, 1, or a combination of both.",
                            "Superposition allows quantum computers to perform many calculations at once."
                          ],
                          "content_with_tags": "A classical bit is either 0 or 1. <analogy type=\"comparison_to_classical_needed\" /> A qubit, however, can be in a state of superposition. <visual_aid_suggestion type=\"animated_diagram\" description=\"Show a classical bit vs. a qubit sphere (Bloch sphere)\" /> Think of it like a spinning coin. <interactive_question_opportunity text_suggestion=\"How is a qubit different from a classical bit you use every day?\" /> This property is fundamental. <difficulty type=\"foundational_info\" />"
                        }
                      ]
                    }
                  ]
                }
                return NlpModuleProcessed(**mock_nlp_json)
            logger.warning(f"No mock NLP content found for {module_id}/{topic_id} in {language_code}")
            return None
        except Exception as e:
            logger.error(f"Error fetching/parsing mock NLP content: {e}")
            return None
    return None


def construct_enhanced_system_prompt(
    user_profile: UserProfileSnapshot,
    language_code: str,
    nlp_processed_content: Optional[NlpModuleProcessed] = None, # Now expects structured content
    current_topic_id_in_focus: Optional[str] = None,
    context_data: Optional[TutorRequestContext] = None,
) -> str:
    system_message_parts = [ # Largely from existing, with enhancements
        "You are Uplas AI Tutor, a highly proficient, engaging, and personalized AI learning assistant.",
        f"Your primary goal is to explain complex concepts clearly and concisely, tailored to the user's background and learning preferences. Respond ONLY in [{language_code}].",
        f"Adhere to the user's preferred '{user_profile.preferred_tutor_persona}' communication style.",
        "When appropriate, generate analogies and suggest follow-up questions as per the requested JSON output structure.",
    ]

    # User Profile Block (similar to existing)
    profile_context_str_parts = ["\n--- User Profile Summary ---"]
    # ... (Add user profile details as in existing construct_system_prompt_for_llm) ...
    if user_profile.profession: profile_context_str_parts.append(f"- Profession: {user_profile.profession}")
    if user_profile.industry: profile_context_str_parts.append(f"- Industry: {user_profile.industry}")
    if user_profile.learning_goals: profile_context_str_parts.append(f"- Learning Goals: {user_profile.learning_goals}")
    system_message_parts.extend(profile_context_str_parts)

    # ENHANCED: Incorporate NLP-processed content and its tags
    if nlp_processed_content:
        system_message_parts.append("\n--- Relevant Processed Learning Material (Use this as primary reference) ---")
        # Find the specific topic in focus if current_topic_id_in_focus is provided
        content_to_inject = ""
        if current_topic_id_in_focus and nlp_processed_content.lessons:
            for lesson in nlp_processed_content.lessons:
                for topic in lesson.topics:
                    if topic.topic_id == current_topic_id_in_focus:
                        content_to_inject = f"Current Topic: {topic.topic_title}\nKey Concepts: {', '.join(topic.key_concepts)}\nContent: {topic.content_with_tags}"
                        break
                if content_to_inject: break
        
        if not content_to_inject: # Fallback to a general snippet if specific topic not found or ID not given
            # Simplified: just take first lesson's first topic for brevity in example
            if nlp_processed_content.lessons and nlp_processed_content.lessons[0].topics:
                first_topic = nlp_processed_content.lessons[0].topics[0]
                content_to_inject = f"Topic: {first_topic.topic_title}\nContent: {first_topic.content_with_tags}"
        
        system_message_parts.append(content_to_inject[:4000]) # Truncate to manage prompt size
        system_message_parts.append(
            "\nInstruction on using tags from the content: If you see tags like "
            "`<analogy type=\"...needed\" />` or `<example domain=\"...needed\" />`, "
            "proactively generate a suitable analogy or example personalized to the user's profile (listed above). "
            "If you see `<interactive_question_opportunity text_suggestion=\"...\" />`, consider asking that question or a similar one. "
            "The `content_with_tags` is your primary source material for the current context."
        )
    
    # Project Assessment Feedback Context (ENHANCED for empathy)
    if context_data and context_data.project_assessment_feedback:
        system_message_parts.append("\n--- Context: Project Feedback ---")
        system_message_parts.append(
            "The user recently submitted a project and did not pass or needs further guidance. "
            f"Their project was: '{context_data.current_project_title or 'Not specified'}'. "
            f"The assessment feedback was: '{context_data.project_assessment_feedback}'. "
            "Your role now is to be an **Empathetic Guide**. Do NOT simply repeat the feedback. "
            "Help the user understand the core reasons they struggled, "
            "explain the relevant concepts from the course material (provided above, if any) that they might need to revisit, "
            "and offer encouraging, actionable steps or simpler exercises they can take to improve. "
            "Acknowledge their effort and maintain a supportive tone."
        )

    system_message_parts.append(
        "\nYour final response MUST be a single JSON object matching the schema: "
        "{'main_answer_text': 'Your detailed answer here, in [Language].', "
        "'suggested_follow_ups': ['Question 1 in [Language]?', 'Question 2 in [Language]?'], "
        "'generated_analogies_for_answer': ['Analogy 1 in [Language]', 'Analogy 2 in [Language]'], "
        "'answer_confidence_score': 0.0-1.0 (optional)}"
        f" All text in the JSON must be in [{language_code}]."
    )
    return "\n".join(system_message_parts)


# --- API Endpoint (Modified to use enhanced logic) ---
@app.post("/v1/ask-tutor", response_model=AiTutorQueryResponse, summary="Get a personalized answer from the AI Tutor")
async def ask_tutor_endpoint(
    request_data: AiTutorQueryRequest,
):
    processing_start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE

    nlp_material_context: Optional[NlpModuleProcessed] = None
    current_topic_id: Optional[str] = None

    if request_data.context:
        current_topic_id = request_data.context.topic_id
        # Fetch NLP processed content using module_id or topic_id from context
        # The NLP agent should have processed and stored this. This call goes to our backend,
        # which then fetches from GCS or its DB.
        nlp_material_context = await fetch_processed_nlp_content(
            request_data.context.module_id, # Prefer module_id to get broader context
            request_data.context.topic_id,
            effective_language_code
        )
        if not nlp_material_context and request_data.context.course_id: # Fallback if only course_id is there (less specific)
            # This implies a mapping from course_id to some default module_id if needed
            logger.info(f"NLP context not found for module/topic, attempting with course_id: {request_data.context.course_id} as a general context.")
            # Potentially fetch a general "course overview" processed document if available

    system_prompt_for_llm = construct_enhanced_system_prompt(
        user_profile=request_data.user_profile_snapshot,
        language_code=effective_language_code,
        nlp_processed_content=nlp_material_context,
        current_topic_id_in_focus=current_topic_id,
        context_data=request_data.context
    )

    llm_conversation_history: List[Dict[str, Any]] = []
    for turn in request_data.conversation_history: #
        role = 'model' if turn.role == ConversationRole.ASSISTANT else turn.role.value
        llm_conversation_history.append({'role': role, 'content': turn.content})
    llm_conversation_history.append({'role': ConversationRole.USER.value, 'content': request_data.query_text})

    structured_llm_output: Optional[StructuredLLMOutput] = None
    llm_raw_response_data = {}

    try:
        # Define the schema for the expected JSON output from Gemini
        response_json_schema = StructuredLLMOutput.model_json_schema()

        llm_raw_response_data = await llm_client.generate_structured_response(
            system_prompt=system_prompt_for_llm,
            conversation_turns=llm_conversation_history,
            max_output_tokens=request_data.max_tokens_response,
            response_schema=response_json_schema # Requesting JSON output
        )
        
        raw_json_string = llm_raw_response_data.get("structured_output_json", "{}")
        try:
            llm_output_dict = json.loads(raw_json_string)
            # Ensure language of the output parts is also considered if not inherently handled by LLM responding in the target lang
            structured_llm_output = StructuredLLMOutput(**llm_output_dict)
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON for structured output: {raw_json_string[:500]}. Error: {e}")
            # Fallback to treating the whole thing as text if JSON parsing fails
            structured_llm_output = StructuredLLMOutput(main_answer_text=raw_json_string if raw_json_string.strip() else "I apologize, I encountered an issue processing the response format.")
        except Exception as p_exc: # Pydantic validation error
            logger.error(f"Pydantic validation failed for LLM output: {raw_json_string[:500]}. Error: {p_exc}")
            structured_llm_output = StructuredLLMOutput(main_answer_text=raw_json_string if raw_json_string.strip() else "I apologize, I encountered an issue with the response structure.")


    except EnvironmentError as e:
        logger.error(f"Configuration error for LLM: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI service configuration error: {e}")
    except Exception as e:
        logger.error(f"LLM Communication/Processing Error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="There was an issue communicating with the AI assistance service.")

    if not structured_llm_output: # Should not happen if try/except is robust
        structured_llm_output = StructuredLLMOutput(main_answer_text="I'm sorry, I couldn't formulate a response at this moment.")

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    debug_information = DebugInfo( #
        llm_model_name_used=llm_client.model_name,
        processing_time_ms=round(processing_time_ms, 2),
        prompt_token_count=llm_raw_response_data.get("prompt_token_count"),
        response_token_count=llm_raw_response_data.get("response_token_count"),
        language_used=effective_language_code
    )
    if os.getenv("UPLAS_DEBUG_MODE", "false").lower() == "true": #
        debug_information.prompt_sent_to_llm_sample = (system_prompt_for_llm + "\n" + \
            "\n".join([f"{turn['role']}: {turn['content'][:100]}..." for turn in llm_conversation_history]))[:1000] + "..."

    return AiTutorQueryResponse(
        answer_text=structured_llm_output.main_answer_text,
        suggested_follow_up_questions=structured_llm_output.suggested_follow_ups,
        generated_analogies=[GeneratedAnalogy(analogy=text) for text in structured_llm_output.generated_analogies_for_answer],
        confidence_score=structured_llm_output.answer_confidence_score,
        debug_info=debug_information
    )

# --- Health Check Endpoint (existing) ---
@app.get("/health", status_code=status.HTTP_200_OK) #
async def health_check():
    return {"status": "healthy", "service": "PersonalizedTutorNLP_LLM_Agent"}


if __name__ == "__main__": #
    import uvicorn
    if not GCP_PROJECT_ID:
        print("Warning: GCP_PROJECT_ID is not set. Please set this environment variable.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))
