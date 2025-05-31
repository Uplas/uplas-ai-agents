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
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") 
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") 
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-001") 

# NovaSpark Enhancement: URL to fetch processed NLP content (from an internal backend endpoint)
NLP_CONTENT_SERVICE_URL = os.getenv("NLP_CONTENT_SERVICE_URL", "http://your-backend-service/api/internal/get-processed-nlp-content")


# Initialize Vertex AI
if not GCP_PROJECT_ID: 
    logging.warning("NovaSpark Warning: GCP_PROJECT_ID environment variable not set. LLM calls may fail.")
else:
    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION) 
        logging.info(f"NovaSpark: Vertex AI SDK initialized for AI Tutor. Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}")
    except Exception as e_init:
        logging.error(f"NovaSpark Critical: Failed to initialize Vertex AI SDK for AI Tutor. Error: {e_init}", exc_info=True)


# Supported languages (BCP-47 codes)
SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] 
DEFAULT_LANGUAGE = "en-US" 

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

# --- Pydantic Models for NLP Agent's Output (as consumed by AI Tutor) ---
# These models define the structure of the content processed by the NLP_Content_Agent
class NlpTopicProcessed(BaseModel):
    topic_id: str 
    topic_title: str
    key_concepts: List[str]
    content_with_tags: str 

class NlpLessonProcessed(BaseModel):
    lesson_id: str 
    lesson_title: str
    lesson_summary: Optional[str] = None
    topics: List[NlpTopicProcessed]

class NlpModuleProcessed(BaseModel): 
    module_id: str
    module_title: Optional[str] = None
    language_code: str
    lessons: List[NlpLessonProcessed]


# --- Pydantic Models for AI Tutor API Contract ---

class UserProfileSnapshot(BaseModel): 
    # Fields from main (7).py, confirmed adequate for personalization task
    industry: Optional[str] = Field(None, examples=["Technology", "Healthcare Management", "Creative Design"])
    profession: Optional[str] = Field(None, examples=["Software Engineer", "Hospital Administrator", "Graphic Designer"])
    country: Optional[str] = Field(None, examples=["Kenya", "Canada", "India"])
    city: Optional[str] = Field(None, examples=["Nairobi", "Vancouver", "Bangalore"])
    preferred_tutor_persona: Optional[str] = Field("Friendly and encouraging", examples=["Socratic and inquisitive", "Direct and technical", "Humorous and engaging"])
    learning_style_preference: Optional[Dict[str, float]] = Field(default_factory=dict, examples=[{"visual": 0.7, "kinesthetic": 0.3, "auditory": 0.5, "reading_writing": 0.6}])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "Calculus": "Beginner", "Quantum Physics": "Novice"}])
    career_interest: Optional[str] = Field(None, examples=["AI Specialist in Medical Diagnosis", "Lead Cloud Architect", "Game Developer"])
    learning_goals: Optional[str] = Field(None, examples=["Master machine learning concepts for my startup.", "Prepare for cloud certification.", "Build a strong portfolio for game design."])
    # NovaSpark Addition: Explicit field for areas of personal interest for analogies
    hobbies_and_interests: Optional[List[str]] = Field(default_factory=list, examples=[["hiking", "classical music", "sci-fi novels"]])


class TutorRequestContext(BaseModel): 
    course_id: Optional[str] = None
    module_id: Optional[str] = None 
    topic_id: Optional[str] = None  
    project_id: Optional[str] = None
    current_topic_title: Optional[str] = Field(None, examples=["Introduction to Python Lists"]) 
    current_course_title: Optional[str] = Field(None, examples=["Python for Beginners"]) 
    current_project_title: Optional[str] = Field(None, examples=["Data Analysis Mini-Project"])
    project_assessment_feedback: Optional[str] = Field(None, description="Feedback from a failed project assessment to guide the tutor.")

class ConversationRole(str, Enum): 
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system" # Retained for completeness, though history usually user/assistant

class ConversationTurn(BaseModel): 
    role: ConversationRole
    content: str

class AiTutorQueryRequest(BaseModel): 
    user_id: str = Field(..., examples=["user_uuid_123"])
    session_id: Optional[str] = Field(None, examples=["session_uuid_abc"])
    query_text: str = Field(..., min_length=1, examples=["Can you explain Python decorators?"])
    context: Optional[TutorRequestContext] = None
    user_profile_snapshot: UserProfileSnapshot # Crucial for personalization
    conversation_history: Optional[List[ConversationTurn]] = Field(default_factory=list)
    language_code: Optional[str] = Field(DEFAULT_LANGUAGE, examples=SUPPORTED_LANGUAGES)
    max_tokens_response: Optional[int] = Field(1024, ge=50, le=8192)

    @validator('language_code')
    def validate_language_code(cls, v): 
        if v not in SUPPORTED_LANGUAGES:
            logger.warning(f"NovaSpark: Unsupported language_code '{v}'. Falling back to default '{DEFAULT_LANGUAGE}'.")
            return DEFAULT_LANGUAGE
        return v

class GeneratedAnalogy(BaseModel): 
    analogy: str
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="NovaSpark: LLM's attempt to score relevance based on profile (0.0-1.0).")

class StructuredLLMOutput(BaseModel):
    main_answer_text: str = Field(..., description="The primary textual answer to the user's query.")
    suggested_follow_ups: List[str] = Field(default_factory=list, description="List of relevant follow-up questions.")
    generated_analogies_for_answer: List[str] = Field(default_factory=list, description="List of analogies generated within the answer, ideally personalized.")
    answer_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="LLM's confidence in the answer (0.0-1.0).")
    # NovaSpark Addition: Explicit field for personalized examples
    personalized_examples_for_answer: List[str] = Field(default_factory=list, description="List of examples generated, aiming for personalization.")


class DebugInfo(BaseModel): 
    prompt_sent_to_llm_sample: Optional[str] = None
    llm_model_name_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    prompt_token_count: Optional[int] = None
    response_token_count: Optional[int] = None
    language_used: Optional[str] = None

class AiTutorQueryResponse(BaseModel): 
    answer_text: str 
    suggested_follow_up_questions: List[str] 
    generated_analogies: List[GeneratedAnalogy] 
    # NovaSpark Addition: To return personalized examples separately if desired
    personalized_examples: List[str]
    confidence_score: Optional[float] 
    debug_info: Optional[DebugInfo] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Tutor Agent (Vertex AI Integrated) - NovaSpark Personalized Edition",
    description="Provides hyper-personalized explanations and guidance using Google Cloud Vertex AI (Gemini), powered by NLP-structured content and advanced prompting techniques developed by NovaSpark.",
    version="0.4.0" # Incremented version
)

# --- Vertex AI LLM Client Logic ---
class VertexAITutorLLMClient: 
    def __init__(self, model_name: str):
        self.model_name = model_name 
        self.retry_config = httpx.Timeout(30.0, connect=10.0) # For GCP calls
        # Consider adding exponential backoff for actual GCP SDK calls if not natively supported by Vertex AI SDK's async methods

    async def generate_structured_response( 
        self,
        system_prompt: str,
        conversation_turns: List[Dict[str, Any]],
        max_output_tokens: int,
        temperature: float = 0.6, # Slightly lower temp for more focused personalization
        top_p: float = 0.9,    # Adjusted for potentially more creative but still grounded responses
        top_k: int = 35,
        response_pydantic_model: Optional[Any] = None 
    ) -> Dict[str, Any]:
        if not GCP_PROJECT_ID: 
            raise EnvironmentError("NovaSpark Critical: GCP_PROJECT_ID is not configured for VertexAITutorLLMClient.")

        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold 

        model = GenerativeModel(
            self.model_name,
            system_instruction=[Part.from_text(system_prompt)] if system_prompt else None 
        )
        
        generation_config_dict = {
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        if response_pydantic_model:
            try:
                generation_config_dict["response_mime_type"] = "application/json"
                generation_config_dict["response_schema"] = response_pydantic_model.model_json_schema()
                logger.info(f"NovaSpark: Configured Gemini for JSON output using schema for {response_pydantic_model.__name__}")
            except Exception as e_schema:
                logger.error(f"NovaSpark Error: Failed to set JSON schema for model {response_pydantic_model.__name__}. Error: {e_schema}", exc_info=True)
                generation_config_dict["response_mime_type"] = "text/plain" # Fallback

        generation_config = GenerationConfig(**generation_config_dict)

        history_for_model = [] 
        for turn in conversation_turns:
            # Ensure role is 'user' or 'model' as per Gemini API requirements
            role = 'user' if turn['role'] == ConversationRole.USER.value else 'model'
            history_for_model.append({'role': role, 'parts': [Part.from_text(turn['content'])]})

        # Safety settings (as per original, good practice)
        safety_settings = { 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        llm_response_text = ""
        prompt_tokens = 0
        response_tokens = 0

        try:
            if len(history_for_model) > 1: 
                chat = model.start_chat(history=history_for_model[:-1])
                current_user_message_content = history_for_model[-1]['parts']
                llm_response_obj = await chat.send_message_async(
                    current_user_message_content,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    # request_options={"timeout": 60} # Example timeout if SDK supports it directly
                )
            elif history_for_model: 
                current_user_message_content = history_for_model[0]['parts']
                llm_response_obj = await model.generate_content_async(
                    current_user_message_content,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    # request_options={"timeout": 60}
                )
            else: # Should not happen if AiTutorQueryRequest validates query_text
                return {"raw_response_text": "NovaSpark Info: No input query provided to LLM.", "prompt_token_count": 0, "response_token_count": 0}

            if llm_response_obj.candidates: 
                candidate = llm_response_obj.candidates[0]
                # Handle potential finish reasons like SAFETY or RECITATION
                if candidate.finish_reason not in [candidate.FinishReason.STOP, candidate.FinishReason.MAX_TOKENS]:
                    logger.warning(f"NovaSpark LLM Warning: Response generation finished with reason: {candidate.finish_reason.name}. Content may be incomplete or blocked.")
                    if candidate.finish_reason == candidate.FinishReason.SAFETY:
                         llm_response_text = "NovaSpark Safety: I'm unable to respond to this specific query due to safety guidelines. Could you please rephrase or ask something else?"
                    # Add handling for other finish reasons if necessary
                
                if candidate.content and candidate.content.parts:
                    # If llm_response_text is not already set by a safety intervention
                    if not llm_response_text:
                        llm_response_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text])
                
                if hasattr(llm_response_obj, 'usage_metadata') and llm_response_obj.usage_metadata: 
                    prompt_tokens = llm_response_obj.usage_metadata.prompt_token_count
                    response_tokens = llm_response_obj.usage_metadata.candidates_token_count
            
            if not llm_response_text and candidate.finish_reason == candidate.FinishReason.STOP: # Successfully stopped but no text
                 llm_response_text = "{}" # Empty JSON if expecting JSON but got nothing
                 logger.warning("NovaSpark LLM Warning: Successful stop but no text content from LLM. Assuming empty JSON.")


        except Exception as e_sdk:
            logger.error(f"NovaSpark SDK Error during LLM call: {e_sdk}", exc_info=True)
            # Reraise as a specific error type or handle by returning a default error JSON.
            # For now, let the endpoint catch it as a general LLM communication issue.
            raise # Reraise for the main endpoint to handle

        return {
            "raw_response_text": llm_response_text, 
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

llm_client = VertexAITutorLLMClient(model_name=LLM_MODEL_NAME) 

# --- Helper Functions (NovaSpark Enhanced) ---

async def fetch_processed_nlp_content(
    module_id: Optional[str],
    topic_id_to_focus: Optional[str], 
    language_code: str
) -> Optional[NlpModuleProcessed]:
    if not module_id:
        logger.info("NovaSpark: No module_id provided to fetch NLP content.")
        return None
    
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL:
        logger.warning(f"NovaSpark: NLP_CONTENT_SERVICE_URL is not configured or is default. Mocking NLP content fetch for module_id: {module_id}.")
        # Fallback to mock if URL is not configured properly
        if module_id == "course101_module3_processed" and language_code == "en-US":
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
            try:
                return NlpModuleProcessed(**mock_nlp_json)
            except Exception as e_mock_parse:
                logger.error(f"NovaSpark Error: Failed to parse mock NLP content. Error: {e_mock_parse}")
                return None
        return None # No mock for other IDs

    request_url = f"{NLP_CONTENT_SERVICE_URL}?module_id={module_id}&language_code={language_code}"
    logger.info(f"NovaSpark: Attempting to fetch NLP-processed content from: {request_url}")

    # NovaSpark: Implementing retry logic for fetching NLP content
    # Using httpx for retries
    async with httpx.AsyncClient(
        timeout=llm_client.retry_config, # Using the same timeout config as LLM client for consistency
        transport=httpx.AsyncHTTPTransport(retries=3) # httpx native retries
    ) as client:
        try:
            response = await client.get(request_url)
            response.raise_for_status() 
            nlp_data_dict = response.json()
            processed_module = NlpModuleProcessed(**nlp_data_dict)
            logger.info(f"NovaSpark: Successfully fetched and parsed NLP content for module_id: {module_id}")
            return processed_module
        except httpx.HTTPStatusError as e:
            logger.error(f"NovaSpark HTTP Error fetching NLP content for module {module_id}: {e.response.status_code} - {e.response.text}", exc_info=True)
            # Consider specific handling for 404 (content not found) vs 5xx (service error)
            if e.response.status_code == 404:
                logger.warning(f"NovaSpark: NLP content not found (404) for module {module_id}, lang {language_code}.")
            return None
        except (json.JSONDecodeError, Exception) as e: 
            logger.error(f"NovaSpark Error parsing/validating NLP content for module {module_id}: {e}", exc_info=True)
            return None

# NovaSpark: Enhanced System Prompt Construction for Deep Personalization
def construct_innovateai_system_prompt(
    user_profile: UserProfileSnapshot,
    language_code: str,
    nlp_module_content: Optional[NlpModuleProcessed] = None,
    current_topic_id_in_focus: Optional[str] = None,
    request_context: Optional[TutorRequestContext] = None,
) -> str:
    system_message_parts = [
        "You are Uplas AI Tutor, an exceptionally skilled, empathetic, and personalized AI learning partner. Your persona is dynamic, guided by the user's preference.",
        f"Your primary mission is to illuminate complex concepts with clarity, tailoring explanations, examples, and analogies to the user's unique profile and stated learning goals. Respond ONLY in [{language_code}].",
        f"Embody the user's preferred communication style: '{user_profile.preferred_tutor_persona}'.",
        "Break down information into digestible chunks. Be patient, encouraging, and deeply supportive.",
        "If a question is unclear, gently ask for clarification in [{language_code}].",
        "If a question is off-topic (personal advice, harmful, unrelated), politely state in [{language_code}] your focus on Uplas educational content and guide them back.",
    ]

    # --- NovaSpark: Dynamic User Profile Injection & Guidance ---
    profile_details_for_prompt = []
    if user_profile.profession:
        profile_details_for_prompt.append(f"- Profession: {user_profile.profession}")
    if user_profile.industry:
        profile_details_for_prompt.append(f"- Industry: {user_profile.industry}")
    if user_profile.career_interest:
        profile_details_for_prompt.append(f"- Stated Career Interest: {user_profile.career_interest}")
    if user_profile.learning_goals:
        profile_details_for_prompt.append(f"- Learning Goals: \"{user_profile.learning_goals}\"")
    if user_profile.current_knowledge_level:
        knowledge_str = ", ".join([f"{topic}: {level}" for topic, level in user_profile.current_knowledge_level.items()])
        profile_details_for_prompt.append(f"- Current Knowledge (self-assessed): {knowledge_str if knowledge_str else 'Not specified'}")
    if user_profile.hobbies_and_interests:
        profile_details_for_prompt.append(f"- Personal Hobbies/Interests: {', '.join(user_profile.hobbies_and_interests)}")

    if profile_details_for_prompt:
        system_message_parts.append("\n--- User Profile Context for Personalization ---")
        system_message_parts.extend(profile_details_for_prompt)
        system_message_parts.append(
            "\n--- NovaSpark Personalization Instructions for LLM ---"
            "1. Deeply analyze the User Profile Context provided above."
            "2. When explaining concepts, GENERATE analogies and examples that directly relate to the user's Profession, Industry, Career Interest, or Hobbies/Interests. Prioritize relevance to their professional context first."
            "   Example: If the user is a 'Software Engineer' in 'Technology', an analogy for 'recursion' could be 'a function calling itself, similar to how a build script might recursively process directories'."
            "   Example: If the user is interested in 'Creative Design' and 'hiking', an analogy for 'color theory' could relate to 'observing color palettes in nature during a hike'."
            "3. Adjust the DEPTH and TECHNICALITY of your explanation based on their 'Current Knowledge (self-assessed)'. If they are a 'Beginner' in a relevant prerequisite, simplify complex terms and build up from fundamentals. If 'Advanced', you can be more technical."
            "4. Frame answers and suggestions, where appropriate, to align with their 'Learning Goals' and 'Career Interest'."
            "5. If the user's query relates to a project or problem-solving, leverage their background to suggest approaches or tools they might be familiar with or find intuitive."
        )
    else:
        system_message_parts.append("\nUser profile details for deep personalization were not extensively provided. Use general best practices for clear explanations.")

    # Utilizing NLP-processed and tagged content (from existing file)
    if nlp_module_content and nlp_module_content.lessons:
        target_topic_content_str = ""
        # ... (rest of the NLP content injection logic from main (7).py, seems okay)
        # NovaSpark: Ensure this section clearly tells the LLM to USE the user profile when fulfilling generic tags.
        # (Adding this to the existing "InnovateAI Content Interaction Instructions")

        focused_material_header = "\n--- Focused Learning Material from Uplas NLP Engine ---"
        general_material_header = "\n--- General Module Context from Uplas NLP Engine ---"
        material_to_include = ""

        if current_topic_id_in_focus:
            for lesson in nlp_module_content.lessons:
                for topic in lesson.topics:
                    if topic.topic_id == current_topic_id_in_focus:
                        material_to_include = f"{focused_material_header}\nTopic: {topic.topic_title}\nKey Concepts: {', '.join(topic.key_concepts)}\nMaterial with Semantic Tags: {topic.content_with_tags}"
                        break
                if material_to_include: break
        
        if not material_to_include and nlp_module_content.lessons: # Fallback
            first_lesson = nlp_module_content.lessons[0]
            material_to_include = f"{general_material_header}\nModule: {nlp_module_content.module_title or 'Current Module'}\nLesson: {first_lesson.lesson_title}\nSummary: {first_lesson.lesson_summary or 'Refer to topics.'}"
            if first_lesson.topics: # Add first topic's content as well
                 material_to_include += f"\nFirst Topic Preview ({first_lesson.topics[0].topic_title}): {first_lesson.topics[0].content_with_tags[:1000]}"


        if material_to_include:
            system_message_parts.append(material_to_include[:4000]) # Truncate for prompt size
            system_message_parts.append(
                "\n--- NovaSpark Content Interaction & Personalization Instructions ---"
                "The 'Material with Semantic Tags' above may contain special XML-like tags:"
                "- `<analogy type=\"...\" />` or `<example domain=\"...\" />`: When you see these, ALWAYS try to generate a RELEVANT analogy/example based on the USER'S PROFILE (Profession, Industry, Hobbies) AND the tag's suggestion. Make it specific!"
                "  Example: If tag is `<analogy type=\"comparison_to_classical_needed\"/>` and user is a 'Chef', compare a quantum concept to a classical cooking technique."
                "- `<interactive_question_opportunity text_suggestion=\"...\" />`: Consider asking the suggested question or a similar one to engage the user."
                "- `<visual_aid_suggestion type=\"...\" description=\"...\" />`: Acknowledge if a visual would be helpful; you can describe what it might show."
                "- `<difficulty type=\"foundational_info|intermediate_detail|advanced_detail\" />`: Use this to tailor the depth of your explanation, ALSO considering the user's 'Current Knowledge' from their profile."
                "Seamlessly integrate these actions into your response. The goal is a rich, personalized dialogue."
            )
        else:
             system_message_parts.append("\nNo specific pre-processed learning material was found for this query's context. Rely on your general knowledge and the user's profile for personalization.")


    # Empathetic Error Attribution for Project Feedback (from existing file, seems okay)
    if request_context and request_context.project_assessment_feedback:
        system_message_parts.append("\n--- NovaSpark Guidance for Project Remediation ---")
        # ... (rest of the project remediation logic from main (7).py)
        system_message_parts.append(
            "The user is seeking help after a project assessment. "
            f"Project: '{request_context.current_project_title or 'N/A'}'. "
            f"Assessment Feedback: '{request_context.project_assessment_feedback}'. "
            "Adopt an **Empathetic Mentor** role. Do NOT just repeat the feedback. "
            "1. Acknowledge their effort and the challenge, referencing their preferred persona for tone."
            "2. Gently help them diagnose *why* they might have struggled, linking to concepts. If their 'Current Knowledge' in a relevant area (from their profile) is 'Beginner' or 'Novice', this might be a factor to consider in your explanation."
            "3. Clearly explain the core concepts they need to revisit or solidify."
            "4. Offer concrete, actionable steps, simpler examples, or prerequisite knowledge they can review to improve. Suggest resources if appropriate."
            "5. Maintain a highly supportive, encouraging, and patient tone. Focus on building their confidence."
        )

    # Instructing LLM to return JSON (from existing file, schema reference is good)
    system_message_parts.append(
        "\n--- NovaSpark Output Format Mandate ---"
        "Your entire response MUST be a single, valid JSON object. Do not add any text before or after this JSON object. "
        "The JSON object must strictly adhere to the following Pydantic schema (StructuredLLMOutput): "
        f"{StructuredLLMOutput.model_json_schema(indent=2)}" 
        f"All text content within this JSON (main_answer_text, analogies, examples, follow-ups) MUST be in [{language_code}]."
        "Prioritize generating useful 'main_answer_text', 'personalized_examples_for_answer', and 'generated_analogies_for_answer' based on all available context and user profile."
    )
    
    final_prompt = "\n".join(system_message_parts)
    # NovaSpark Token Management Note: This prompt can become quite long.
    # Monitor token counts closely. If prompts become too large for Gemini Flash's context window
    # with conversation history, consider summarizing older history or the NLP content snippet further.
    # For now, relying on truncation of NLP content.
    logger.debug(f"NovaSpark Generated System Prompt (first 500 chars): {final_prompt[:500]}...")
    return final_prompt


# --- API Endpoint (NovaSpark Enhanced) ---
@app.post("/v1/ask-tutor", response_model=AiTutorQueryResponse, summary="Get a personalized answer from the AI Tutor (NovaSpark Personalized Edition)")
async def ask_tutor_endpoint(
    request_data: AiTutorQueryRequest,
): 
    processing_start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE 

    nlp_processed_module: Optional[NlpModuleProcessed] = None
    current_topic_id: Optional[str] = None

    # Fetch and prepare context (logic from main (7).py is largely sound, with minor logging additions)
    if request_data.context and request_data.context.module_id:
        current_topic_id = request_data.context.topic_id 
        logger.info(f"NovaSpark: Module context provided: module_id='{request_data.context.module_id}', topic_id='{current_topic_id}'. Fetching NLP content.")
        nlp_processed_module = await fetch_processed_nlp_content(
            request_data.context.module_id,
            current_topic_id, 
            effective_language_code
        )
        if nlp_processed_module:
            if not request_data.context.current_topic_title and current_topic_id:
                for lesson in nlp_processed_module.lessons:
                    for topic_item in lesson.topics: # Renamed from topic to topic_item to avoid conflict
                        if topic_item.topic_id == current_topic_id:
                            request_data.context.current_topic_title = topic_item.topic_title
                            logger.info(f"NovaSpark: Auto-populated topic title: '{topic_item.topic_title}'")
                            if nlp_processed_module.module_title and not request_data.context.current_course_title :
                                 request_data.context.current_course_title = nlp_processed_module.module_title 
                                 logger.info(f"NovaSpark: Auto-populated course title: '{nlp_processed_module.module_title}'")
                            break
                    if request_data.context.current_topic_title: break
            if not nlp_processed_module.module_title and request_data.context.current_course_title:
                 # If NLP module doesn't have a title but request context does, log it
                 logger.info(f"NovaSpark: Using course title from request context: '{request_data.context.current_course_title}' as NLP module had no title.")
        else:
            logger.warning(f"NovaSpark: Failed to fetch or parse NLP content for module_id: {request_data.context.module_id}")
    
    system_prompt_for_llm = construct_innovateai_system_prompt( 
        user_profile=request_data.user_profile_snapshot,
        language_code=effective_language_code,
        nlp_module_content=nlp_processed_module,
        current_topic_id_in_focus=current_topic_id,
        request_context=request_data.context
    )

    llm_conversation_history: List[Dict[str, Any]] = [] 
    for turn in request_data.conversation_history:
        # Ensure role is 'user' or 'model' as per Gemini API requirements
        role = 'user' if turn.role == ConversationRole.USER.value else 'model'
        llm_conversation_history.append({'role': role, 'content': turn.content})
    # Add current user query as the last turn
    llm_conversation_history.append({'role': ConversationRole.USER.value, 'content': request_data.query_text})

    structured_llm_output_obj: Optional[StructuredLLMOutput] = None
    llm_response_metrics = {}
    raw_json_string_from_llm = "{}" # Initialize for fallback

    try:
        logger.info(f"NovaSpark: Sending request to LLM for user {request_data.user_id}, query: \"{request_data.query_text[:100]}...\"")
        llm_response_metrics = await llm_client.generate_structured_response( 
            system_prompt=system_prompt_for_llm,
            conversation_turns=llm_conversation_history,
            max_output_tokens=request_data.max_tokens_response,
            response_pydantic_model=StructuredLLMOutput 
        )
        
        raw_json_string_from_llm = llm_response_metrics.get("raw_response_text", "{}")
        
        if not raw_json_string_from_llm.strip(): # Handle completely empty response from LLM
            logger.error("NovaSpark Error: LLM returned a completely empty response string.")
            structured_llm_output_obj = StructuredLLMOutput(main_answer_text="I'm sorry, I encountered an issue and couldn't generate a response. Please try again.")
        else:
            try:
                llm_output_dict = json.loads(raw_json_string_from_llm)
                structured_llm_output_obj = StructuredLLMOutput(**llm_output_dict)
                logger.info(f"NovaSpark: Successfully parsed structured LLM output for user {request_data.user_id}.")
            except json.JSONDecodeError as e_json:
                logger.error(f"NovaSpark Error: LLM returned invalid JSON for structured output: {raw_json_string_from_llm[:500]}. Error: {e_json}", exc_info=True)
                structured_llm_output_obj = StructuredLLMOutput(main_answer_text=f"I seem to have gotten my thoughts a bit tangled! Here's the raw information I was processing: \"{raw_json_string_from_llm}\". Could you try rephrasing your question?", suggested_follow_ups=[], generated_analogies_for_answer=[], personalized_examples_for_answer=[])
            except Exception as e_pydantic: 
                logger.error(f"NovaSpark Error: Pydantic validation failed for LLM JSON output: {raw_json_string_from_llm[:500]}. Error: {e_pydantic}", exc_info=True)
                structured_llm_output_obj = StructuredLLMOutput(main_answer_text=f"My response structure was a little unusual this time. Here's what I gathered: \"{raw_json_string_from_llm}\". Perhaps we can try a different angle on your question?", suggested_follow_ups=[], generated_analogies_for_answer=[], personalized_examples_for_answer=[])

    except EnvironmentError as e_env: 
        logger.error(f"NovaSpark Config Error for LLM: {e_env}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI service configuration error: {e_env}")
    except Exception as e_llm: 
        logger.error(f"NovaSpark LLM Communication/Processing Error: {e_llm}", exc_info=True)
        # Provide a more user-friendly error if it's a general comms issue
        user_friendly_detail = "I'm having a bit of trouble connecting to my knowledge circuits right now. Please try again in a few moments."
        if "timeout" in str(e_llm).lower(): # Be more specific for timeouts
            user_friendly_detail = "It's taking me a little longer than usual to process that. Could you try again shortly?"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=user_friendly_detail)

    if not structured_llm_output_obj: 
        # This case should be rare given the fallbacks above, but as a final safety net:
        logger.error("NovaSpark Critical: structured_llm_output_obj is None after LLM call and fallbacks. This should not happen.")
        structured_llm_output_obj = StructuredLLMOutput(
            main_answer_text="I'm sorry, I was unable to formulate a response. Please try asking in a different way or check back soon.",
            suggested_follow_ups=[], generated_analogies_for_answer=[], personalized_examples_for_answer=[]
            )

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000 

    debug_information = DebugInfo( 
        llm_model_name_used=llm_client.model_name,
        processing_time_ms=round(processing_time_ms, 2),
        prompt_token_count=llm_response_metrics.get("prompt_token_count"),
        response_token_count=llm_response_metrics.get("response_token_count"),
        language_used=effective_language_code
    )
    # Enhanced debug logging for prompt, only if a specific ENV VAR is set (e.g., for staging/dev)
    if os.getenv("UPLAS_AI_TUTOR_DEBUG_PROMPT", "false").lower() == "true": 
        full_prompt_for_debug = system_prompt_for_llm + "\n--- Conversation History ---\n" + \
            "\n".join([f"Role: {turn['role']}, Content Snippet: {turn['content'][:200]}..." for turn in llm_conversation_history])
        debug_information.prompt_sent_to_llm_sample = full_prompt_for_debug[:2000] + ("..." if len(full_prompt_for_debug) > 2000 else "")
        logger.info(f"NovaSpark DEBUG Full Prompt Sample (User {request_data.user_id}): {debug_information.prompt_sent_to_llm_sample}")


    return AiTutorQueryResponse(
        answer_text=structured_llm_output_obj.main_answer_text,
        suggested_follow_up_questions=structured_llm_output_obj.suggested_follow_ups,
        generated_analogies=[GeneratedAnalogy(analogy=text) for text in structured_llm_output_obj.generated_analogies_for_answer],
        personalized_examples=structured_llm_output_obj.personalized_examples_for_answer, # NovaSpark: Added
        confidence_score=structured_llm_output_obj.answer_confidence_score,
        debug_info=debug_information
    )

# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK) 
async def health_check():
    if not GCP_PROJECT_ID:
        return {"status": "unhealthy", "reason": "GCP_PROJECT_ID not configured.", "service": "PersonalizedTutorNLP_LLM_Agent_NovaSpark"}
    
    nlp_service_ok = True
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL: 
         logger.warning("NovaSpark Health Warning: NLP_CONTENT_SERVICE_URL is not configured or using default. NLP content fetching will be mocked/fail.")
         nlp_service_ok = False # Consider this a degraded state for full functionality
    
    # NovaSpark: Check Vertex AI client initialization status if possible (conceptual)
    # For now, presence of GCP_PROJECT_ID implies an attempt was made.
    # A more robust check might involve a tiny, non-costly call or status check if Vertex AI SDK offers one.

    service_name = "PersonalizedTutorNLP_LLM_Agent_NovaSpark"
    if not nlp_service_ok:
        return {"status": "degraded", "reason": "NLP_CONTENT_SERVICE_URL not optimally configured. Personalization from course content may be limited.", "service": service_name, "innovate_ai_enhancements_active": True}

    return {"status": "healthy", "service": service_name, "innovate_ai_enhancements_active": True}


if __name__ == "__main__": 
    import uvicorn
    logger.info("NovaSpark: Starting Personalized AI Tutor Agent (NovaSpark Edition) for local development...")
    if not GCP_PROJECT_ID:
        print("NovaSpark Warning: GCP_PROJECT_ID is not set. LLM calls will fail.")
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL:
        print("NovaSpark Warning: NLP_CONTENT_SERVICE_URL is not set or is default. NLP content fetching will be mocked or fail, limiting personalization.")
    
    port = int(os.getenv("PORT", 8001)) # Default from original file
    uvicorn.run(app, host="0.0.0.0", port=port)
