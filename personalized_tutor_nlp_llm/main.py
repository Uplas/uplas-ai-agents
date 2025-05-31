# uplas-ai-agents/personalized_tutor_nlp_llm/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import httpx # For potential future calls to Django to fetch content
import os
import time
from enum import Enum
import logging

# GCP Clients
from google.cloud import aiplatform
import google.auth
import google.auth.transport.requests

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") # Default to us-central1 if not set
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-001") # Using a Gemini model

# Initialize Vertex AI
if not GCP_PROJECT_ID:
    logging.warning("GCP_PROJECT_ID environment variable not set. LLM calls may fail.")
    # In a real deployment, you might want to raise an error or have a fallback
else:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

# Supported languages (BCP-47 codes)
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"]
DEFAULT_LANGUAGE = "en-US"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Contract (Updated) ---

class UserProfileSnapshot(BaseModel):
    industry: Optional[str] = Field(None, examples=["Technology"])
    profession: Optional[str] = Field(None, examples=["Software Engineer"])
    country: Optional[str] = Field(None, examples=["Kenya"])
    city: Optional[str] = Field(None, examples=["Nairobi"])
    preferred_tutor_persona: Optional[str] = Field("Friendly and encouraging", examples=["Socratic", "Technical"])
    learning_style_preference: Optional[Dict[str, float]] = Field(default_factory=dict, examples=[{"visual": 0.7, "kinesthetic": 0.3}])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "TopicX": "Beginner"}])
    career_interest: Optional[str] = Field(None, examples=["AI Specialist"])
    learning_goals: Optional[str] = Field(None, examples=["Master machine learning concepts."])

class TutorRequestContext(BaseModel):
    course_id: Optional[str] = None
    topic_id: Optional[str] = None
    project_id: Optional[str] = None
    current_topic_title: Optional[str] = Field(None, examples=["Introduction to Python Lists"])
    current_course_title: Optional[str] = Field(None, examples=["Python for Beginners"])
    current_project_title: Optional[str] = Field(None, examples=["Data Analysis Mini-Project"])
    project_assessment_feedback: Optional[str] = Field(None, description="Feedback from a failed project assessment to guide the tutor.")

class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant" # LLM's role
    SYSTEM = "system" # For initial instructions, not part of chat history for some models

class ConversationTurn(BaseModel):
    role: ConversationRole
    content: str # For Gemini, 'parts': [{'text': '...'}]

class AiTutorQueryRequest(BaseModel):
    user_id: str = Field(..., examples=["user_uuid_123"])
    session_id: Optional[str] = Field(None, examples=["session_uuid_abc"])
    query_text: str = Field(..., min_length=1, examples=["Can you explain Python decorators?"])
    context: Optional[TutorRequestContext] = None
    user_profile_snapshot: UserProfileSnapshot
    conversation_history: Optional[List[ConversationTurn]] = Field(default_factory=list, description="History of user and assistant messages.")
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    max_tokens_response: Optional[int] = Field(1024, ge=50, le=8192) # Increased for more comprehensive answers

    @validator('language_code')
    def validate_language_code(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language_code '{v}' received. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class GeneratedAnalogy(BaseModel):
    analogy: str
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0) # LLM might not provide this directly

class DebugInfo(BaseModel):
    prompt_sent_to_llm_sample: Optional[str] = None # Sample of the prompt for debugging
    llm_model_name_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    language_used: Optional[str] = None

class AiTutorQueryResponse(BaseModel):
    answer_text: str
    suggested_follow_up_questions: List[str] = Field(default_factory=list)
    generated_analogies: List[GeneratedAnalogy] = Field(default_factory=list) # LLM may embed these in answer_text
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0) # LLM might not provide this
    debug_info: Optional[DebugInfo] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Tutor Agent (Vertex AI Integrated)",
    description="Provides personalized explanations and guidance using Google Cloud Vertex AI (Gemini).",
    version="0.2.0"
)

# --- Vertex AI LLM Client Logic ---
class VertexAILLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # The actual model object is loaded per request or kept if stateful interactions are needed
        # For stateless, a new GenerativeModel instance can be created each time.
        # For chat, you'd initialize `model.start_chat(history=...)`

    async def generate_response(
        self,
        system_prompt: str, # Prepended to the main conversation
        conversation_turns: List[Dict[str, Any]], # List of {'role': 'user'|'model', 'parts': [{'text': '...'}]}
        max_output_tokens: int,
        temperature: float = 0.7, # Default temperature
        top_p: float = 0.95,      # Default top_p
        top_k: int = 40         # Default top_k
    ) -> Dict[str, Any]:
        """
        Generates a response from Vertex AI's Gemini model.
        Handles both single-turn and multi-turn (chat-like) interactions.
        """
        if not GCP_PROJECT_ID:
            raise EnvironmentError("GCP_PROJECT_ID is not configured. Cannot call Vertex AI.")

        from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold

        model = GenerativeModel(
            self.model_name,
            system_instruction=[Part.from_text(system_prompt)] if system_prompt else None
        )
        
        # Construct history for the model
        # Gemini expects roles 'user' and 'model'.
        history_for_model = []
        for turn in conversation_turns:
            role = 'user' if turn['role'] == ConversationRole.USER.value else 'model'
            history_for_model.append({'role': role, 'parts': [Part.from_text(turn['content'])]})

        # If there's history, start a chat session
        if len(history_for_model) > 1: # More than just the current user query
            chat = model.start_chat(history=history_for_model[:-1]) # All but the last user message
            current_user_message = history_for_model[-1]['parts'][0].text
            llm_response = await chat.send_message_async(
                [Part.from_text(current_user_message)],
                generation_config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                },
                safety_settings={ # Example safety settings
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
        elif history_for_model: # Single user message
            current_user_message = history_for_model[0]['parts'][0].text
            llm_response = await model.generate_content_async(
                [Part.from_text(current_user_message)],
                generation_config={
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                },
                 safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
        else: # Should not happen if query_text is mandatory
            return {"answer_text": "No input provided to LLM.", "prompt_token_count": 0, "response_token_count": 0}

        answer_text = ""
        prompt_tokens = 0
        response_tokens = 0

        if llm_response.candidates:
            # Assuming the first candidate is the one we want
            candidate = llm_response.candidates[0]
            if candidate.content and candidate.content.parts:
                answer_text = "".join([part.text for part in candidate.content.parts if part.text])

            # Get token counts if available (usage_metadata might be present)
            if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
                prompt_tokens = llm_response.usage_metadata.prompt_token_count
                response_tokens = llm_response.usage_metadata.candidates_token_count # Sum of tokens for all candidates
                # If only one candidate, this is effectively the response token count.
                # For more precise candidate-specific tokens, one might need to inspect candidate.token_count if available.
            else: # Fallback rough estimation (less accurate)
                prompt_tokens = sum(len(turn['content'].split()) for turn in conversation_turns) + len(system_prompt.split())
                response_tokens = len(answer_text.split())

        # Placeholder for extracting structured analogies and follow-up questions
        # This would ideally be done by prompting the LLM for a JSON output or by parsing its text.
        # For now, we'll keep it simple.
        return {
            "answer_text": answer_text,
            "generated_analogies": [], # To be implemented with better prompting or parsing
            "suggested_follow_up_questions": [], # To be implemented
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

llm_client = VertexAILLMClient(model_name=LLM_MODEL_NAME)

# --- Helper Functions ---

def construct_system_prompt_for_llm(
    user_profile: UserProfileSnapshot,
    language_code: str,
    course_content_snippet: Optional[str] = None,
    context_data: Optional[TutorRequestContext] = None,
) -> str:
    system_message_parts = [
        "You are Uplas AI Tutor, a highly proficient, engaging, and personalized AI learning assistant.",
        "Your primary goal is to explain complex concepts clearly and concisely, tailored to the user's background and learning preferences.",
        f"IMPORTANT: Generate your entire response exclusively in the following language: {language_code}.",
        f"The user indicates a preference for a '{user_profile.preferred_tutor_persona}' communication style. Adhere to this style.",
        "Be encouraging, patient, and supportive. Break down answers into digestible parts if the concept is complex.",
        "If a question is ambiguous, ask for clarification in {language_code} before attempting an answer.",
        "If a question is outside the scope of the platform's educational content (e.g., personal advice, harmful content, unrelated topics), politely state in {language_code} that you cannot answer it and try to guide them back to relevant Uplas topics.",
        "When generating analogies, if possible, explicitly state them using a prefix like 'Analogy:' or 'Think of it like this:' (translated to {language_code}) for clarity. These analogies should be highly relevant to the user's profile.",
        "Structure your answers well. Use bullet points or numbered lists for steps or multiple points if it enhances clarity.",
    ]

    # User Profile Context Block
    profile_context_str_parts = ["\n--- User Profile Summary for Personalization ---"]
    has_profile_info = False
    if user_profile.profession: profile_context_str_parts.append(f"- Profession: {user_profile.profession}"); has_profile_info = True
    if user_profile.industry: profile_context_str_parts.append(f"- Industry: {user_profile.industry}"); has_profile_info = True
    if user_profile.city or user_profile.country:
        location = f"{user_profile.city or ''}, {user_profile.country or ''}".strip(", ")
        if location: profile_context_str_parts.append(f"- Location: {location}"); has_profile_info = True
    if user_profile.career_interest: profile_context_str_parts.append(f"- Career Interest: {user_profile.career_interest}"); has_profile_info = True
    if user_profile.learning_goals: profile_context_str_parts.append(f"- Stated Learning Goals: {user_profile.learning_goals}"); has_profile_info = True
    if user_profile.learning_style_preference:
        styles = ", ".join([f"{k} (preference: {v*100:.0f}%)" for k, v in user_profile.learning_style_preference.items() if v > 0.3])
        if styles: profile_context_str_parts.append(f"- Noted Learning Styles: {styles}"); has_profile_info = True
    if user_profile.current_knowledge_level:
        knowledge = ", ".join([f"{k}: {v}" for k, v in user_profile.current_knowledge_level.items()])
        if knowledge: profile_context_str_parts.append(f"- Self-Assessed Knowledge: {knowledge}"); has_profile_info = True
    
    if has_profile_info:
        system_message_parts.extend(profile_context_str_parts)
    else:
        system_message_parts.append("\nUser profile details for personalization were not extensively provided.")

    # Specific Question Context Block
    if context_data:
        context_info_str_parts = ["\n--- Current Learning Context ---"]
        has_context_info = False
        if context_data.current_course_title: context_info_str_parts.append(f"- Course: {context_data.current_course_title}"); has_context_info = True
        if context_data.current_topic_title: context_info_str_parts.append(f"- Topic: {context_data.current_topic_title}"); has_context_info = True
        if context_data.current_project_title: context_info_str_parts.append(f"- Project: {context_data.current_project_title}"); has_context_info = True
        if context_data.project_assessment_feedback:
            context_info_str_parts.append(f"- Important: The user recently received feedback on a project: '{context_data.project_assessment_feedback}'. If their question relates to this project or similar concepts, address this feedback directly and help them understand the areas they struggled with.")
            has_context_info = True
        if has_context_info:
            system_message_parts.extend(context_info_str_parts)

    # Main Course/Topic Content for RAG (if provided)
    if course_content_snippet:
        system_message_parts.append("\n--- Relevant Information/Course Material Snippet (use this as primary reference) ---")
        system_message_parts.append(course_content_snippet[:3000]) # Truncate

    system_message_parts.append("\nRemember to respond ONLY in {language_code}.")
    return "\n".join(system_message_parts)


async def fetch_course_content_from_django(topic_id: Optional[str], course_id: Optional[str], language_code: str) -> Optional[str]:
    """
    MOCK: Fetches course content from Django.
    In a real scenario, this would be an authenticated HTTP call to an internal Django API endpoint
    that serves content based on topic_id, course_id, and language_code.
    e.g., GET /api/internal/content/?topic_id=<topic_id>&course_id=<course_id>&lang=<language_code>
    """
    logger.info(f"Mock fetching course content for topic: {topic_id}, course: {course_id}, lang: {language_code}")
    if topic_id == "topic_python_lists_id":
        if language_code.startswith("es"):
            return "Las listas de Python son secuencias versátiles, ordenadas y mutables. Métodos clave incluyen append(), extend(), insert(), remove(), pop(), index(), count(), sort(), reverse(). El rebanado también es poderoso."
        return "Python lists are versatile, ordered, and mutable sequences. Key methods include append(), extend(), insert(), remove(), pop(), index(), count(), sort(), reverse(). Slicing is also powerful."
    # Add more mock content for different topics/languages as needed for testing
    return None

# --- API Endpoint ---
@app.post("/v1/ask-tutor", response_model=AiTutorQueryResponse, summary="Get a personalized answer from the AI Tutor")
async def ask_tutor_endpoint(
    request_data: AiTutorQueryRequest,
    # fastapi_request: FastAPIRequest # For accessing raw request if needed
):
    processing_start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE

    course_material_context: Optional[str] = None
    if request_data.context and (request_data.context.topic_id or request_data.context.course_id):
        try:
            course_material_context = await fetch_course_content_from_django(
                request_data.context.topic_id,
                request_data.context.course_id,
                effective_language_code
            )
        except Exception as e:
            logger.error(f"Error fetching course content from Django (mock): {e}")
            # Decide if this should be a fatal error or just proceed without course context

    system_prompt_for_llm = construct_system_prompt_for_llm(
        user_profile=request_data.user_profile_snapshot,
        language_code=effective_language_code,
        course_content_snippet=course_material_context,
        context_data=request_data.context
    )

    # Prepare conversation history for LLM
    # Gemini expects roles 'user' and 'model'
    llm_conversation_history: List[Dict[str, Any]] = []
    for turn in request_data.conversation_history:
        # Map roles: ASSISTANT (our Pydantic model) -> 'model' (for Gemini)
        role = 'model' if turn.role == ConversationRole.ASSISTANT else turn.role.value
        llm_conversation_history.append({'role': role, 'content': turn.content})
    
    # Add current user query to the history for the LLM
    llm_conversation_history.append({'role': ConversationRole.USER.value, 'content': request_data.query_text})

    try:
        llm_response_data = await llm_client.generate_response(
            system_prompt=system_prompt_for_llm,
            conversation_turns=llm_conversation_history,
            max_output_tokens=request_data.max_tokens_response
        )
        answer_text = llm_response_data.get("answer_text", "I apologize, I couldn't generate a response for that.")
        # Analogies and follow-ups are placeholders for now, assuming they are part of answer_text
        # For structured output, the LLM prompt would need to request JSON.

    except EnvironmentError as e: # Catch if GCP_PROJECT_ID is not set
        logger.error(f"Configuration error for LLM: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI service configuration error: {e}")
    except Exception as e:
        logger.error(f"LLM Communication/Processing Error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="There was an issue communicating with the AI assistance service.")

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    debug_information = DebugInfo(
        llm_model_name_used=llm_client.model_name,
        processing_time_ms=round(processing_time_ms, 2),
        prompt_token_count=llm_response_data.get("prompt_token_count"),
        response_token_count=llm_response_data.get("response_token_count"),
        language_used=effective_language_code
    )
    # Only include prompt sample in debug mode (controlled by env var, e.g.)
    if os.getenv("UPLAS_DEBUG_MODE", "false").lower() == "true":
        debug_information.prompt_sent_to_llm_sample = (system_prompt_for_llm + "\n" + \
            "\n".join([f"{turn['role']}: {turn['content'][:100]}..." for turn in llm_conversation_history]))[:1000] + "..."


    return AiTutorQueryResponse(
        answer_text=answer_text,
        suggested_follow_up_questions=llm_response_data.get("suggested_follow_up_questions", []),
        generated_analogies=llm_response_data.get("generated_analogies", []),
        # confidence_score=0.88, # Mocked, LLM might not provide this directly
        debug_info=debug_information
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "PersonalizedTutorNLP_LLM_Agent"}


if __name__ == "__main__":
    import uvicorn
    # Ensure environment variables are set for local testing if needed
    # Example: GCP_PROJECT_ID, GOOGLE_APPLICATION_CREDENTIALS (if not using ADC from gcloud)
    if not GCP_PROJECT_ID:
        print("Warning: GCP_PROJECT_ID is not set. Please set this environment variable.")
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8001)))

# To run this locally (ensure GCP auth is set up, e.g., `gcloud auth application-default login`):
# GCP_PROJECT_ID="your-gcp-project-id" python main.py
# or using uvicorn directly:
# GCP_PROJECT_ID="your-gcp-project-id" uvicorn main:app --reload --port 8001
