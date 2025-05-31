# uplas-ai-agents/personalized_tutor_nlp_llm/main.py
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest, status
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import httpx 
import os
import time
from enum import Enum
import logging
import json 

from google.cloud import aiplatform
import google.auth
import google.auth.transport.requests

# --- Configuration ---
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID") 
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1") 
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-001") 
NLP_CONTENT_SERVICE_URL = os.getenv("NLP_CONTENT_SERVICE_URL", "http://your-backend-service/api/internal/get-processed-nlp-content")

if not GCP_PROJECT_ID: 
    logging.warning("NovaSpark Warning: GCP_PROJECT_ID environment variable not set. LLM calls may fail.")
else:
    try:
        aiplatform.init(project=GCP_PROJECT_ID, location=GCP_LOCATION) 
        logging.info(f"NovaSpark: Vertex AI SDK initialized for AI Tutor. Project: {GCP_PROJECT_ID}, Location: {GCP_LOCATION}")
    except Exception as e_init:
        logging.error(f"NovaSpark Critical: Failed to initialize Vertex AI SDK for AI Tutor. Error: {e_init}", exc_info=True)

SUPPORTED_LANGUAGES = ["en-US", "fr-FR", "es-ES", "de-DE", "pt-BR", "zh-CN", "hi-IN"] 
DEFAULT_LANGUAGE = "en-US" 

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

# --- Persona Snippet Library (Task 4) ---
PERSONA_LIBRARY = {
    "uncle_trevor_style": {
        "description": "Wise, patient, encouraging, uses relatable stories and analogies, a bit folksy and avuncular.",
        "snippets": [
            "Alright there, young trailblazer! Don't you worry your head too much about this tricky bit. Old Uncle Trevor has seen a fair few of these in his day. The secret is often just looking at it from a slightly different angle, like turning a puzzle piece until it clicks. Let's walk through it together, nice and easy.",
            "Think of it like baking a good ol' apple pie. You need the right ingredients (that's your data), a solid recipe (that's your algorithm), and a bit of patience while it bakes (that's your processing time). Get one part wrong, and the pie might not turn out quite right, but we can always figure out what to adjust!",
            "Every expert was once a beginner, remember that! This might seem like a steep hill now, but with each step, you're getting stronger and the view from the top is worth it. We'll get you there."
        ]
    },
    "susan_style": {
        "description": "Clear, articulate, professional, modern, efficient, encouraging but direct, values precision and practical application.",
        "snippets": [
            "Excellent question. Let's approach this systematically. The core principle here is X, which leads to Y. Understanding this is key to unlocking Z. We can break this down into a few manageable components.",
            "From a practical standpoint, especially in your field of [user_profession], this concept allows you to [achieve specific outcome]. For instance, consider a scenario where you need to [user_industry_relevant_task]. This is where this knowledge becomes highly valuable.",
            "That's a very insightful observation. You're definitely on the right track. To solidify this, let's consider how you would apply this to [a small, concrete problem/example]. What would be your first step?"
        ]
    },
    "empathetic_mentor": {
        "description": "Deeply understanding, supportive, patient, focuses on building confidence, good at diagnosing underlying issues gently.",
        "snippets": [
            "I hear you, and it's completely understandable to feel that way when you're grappling with something new and challenging. Many people find this particular area a bit tricky at first. You're not alone in this, and we can definitely work through it.",
            "Remember that learning is a journey with ups and downs. The fact that you're asking these questions shows you're engaging deeply, and that's fantastic! Let's try a different approach. Sometimes, just a small shift in perspective can make all the difference.",
            "It sounds like the core of the confusion might be around [specific concept]. Is that right? Or is there another part that feels like the main hurdle? No worries if it's hard to pinpoint; we can explore it together. What if we start with a simpler version of this idea?"
        ]
    }
}

# --- Pydantic Models ---
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

class UserProfileSnapshot(BaseModel): 
    industry: Optional[str] = Field(None, examples=["Technology", "Healthcare Management", "Creative Design"])
    profession: Optional[str] = Field(None, examples=["Software Engineer", "Hospital Administrator", "Graphic Designer"])
    country: Optional[str] = Field(None, examples=["Kenya", "Canada", "India"])
    city: Optional[str] = Field(None, examples=["Nairobi", "Vancouver", "Bangalore"])
    preferred_tutor_persona: Optional[str] = Field("Friendly and encouraging", examples=["Socratic and inquisitive", "Direct and technical", "Humorous and engaging", "Uncle Trevor Style", "Susan Style", "Empathetic Mentor"]) # Added new persona examples
    learning_style_preference: Optional[Dict[str, float]] = Field(default_factory=dict, examples=[{"visual": 0.7, "kinesthetic": 0.3, "auditory": 0.5, "reading_writing": 0.6}])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "Calculus": "Beginner", "Quantum Physics": "Novice"}])
    career_interest: Optional[str] = Field(None, examples=["AI Specialist in Medical Diagnosis", "Lead Cloud Architect", "Game Developer"])
    learning_goals: Optional[str] = Field(None, examples=["Master machine learning concepts for my startup.", "Prepare for cloud certification.", "Build a strong portfolio for game design."])
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
    SYSTEM = "system"

class ConversationTurn(BaseModel): 
    role: ConversationRole
    content: str

class AiTutorQueryRequest(BaseModel): 
    user_id: str = Field(..., examples=["user_uuid_123"])
    session_id: Optional[str] = Field(None, examples=["session_uuid_abc"])
    query_text: str = Field(..., min_length=1, examples=["Can you explain Python decorators?"])
    context: Optional[TutorRequestContext] = None
    user_profile_snapshot: UserProfileSnapshot
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
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class StructuredLLMOutput(BaseModel):
    main_answer_text: str = Field(..., description="The primary textual answer to the user's query.")
    suggested_follow_ups: List[str] = Field(default_factory=list)
    generated_analogies_for_answer: List[str] = Field(default_factory=list)
    answer_confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    personalized_examples_for_answer: List[str] = Field(default_factory=list)

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
    personalized_examples: List[str]
    confidence_score: Optional[float] 
    debug_info: Optional[DebugInfo] = None

app = FastAPI(
    title="Uplas AI Tutor Agent (Vertex AI) - NovaSpark Empathetic Edition",
    description="Provides hyper-personalized, empathetic, and engaging guidance using Google Cloud Vertex AI (Gemini), enhanced by NovaSpark's advanced interaction design.",
    version="0.5.0" # Incremented for Task 4
)

class VertexAITutorLLMClient: 
    def __init__(self, model_name: str):
        self.model_name = model_name 
        self.retry_config = httpx.Timeout(30.0, connect=10.0) 

    async def generate_structured_response( 
        self, system_prompt: str, conversation_turns: List[Dict[str, Any]],
        max_output_tokens: int, temperature: float = 0.6, top_p: float = 0.9, top_k: int = 35,
        response_pydantic_model: Optional[Any] = None 
    ) -> Dict[str, Any]:
        if not GCP_PROJECT_ID: 
            raise EnvironmentError("NovaSpark Critical: GCP_PROJECT_ID is not configured for VertexAITutorLLMClient.")

        from vertexai.generative_models import GenerativeModel, Part, GenerationConfig, HarmCategory, HarmBlockThreshold 

        model = GenerativeModel(self.model_name, system_instruction=[Part.from_text(system_prompt)] if system_prompt else None)
        
        generation_config_dict = {"max_output_tokens": max_output_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}
        if response_pydantic_model:
            try:
                generation_config_dict["response_mime_type"] = "application/json"
                generation_config_dict["response_schema"] = response_pydantic_model.model_json_schema()
                logger.info(f"NovaSpark: Configured Gemini for JSON output using schema for {response_pydantic_model.__name__}")
            except Exception as e_schema:
                logger.error(f"NovaSpark Error: Failed to set JSON schema for model {response_pydantic_model.__name__}. Error: {e_schema}", exc_info=True)
                generation_config_dict["response_mime_type"] = "text/plain" 
        generation_config = GenerationConfig(**generation_config_dict)

        history_for_model = [{'role': 'user' if turn['role'] == ConversationRole.USER.value else 'model', 
                              'parts': [Part.from_text(turn['content'])]} for turn in conversation_turns]

        safety_settings = { 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        llm_response_text = ""
        prompt_tokens = 0
        response_tokens = 0
        candidate = None # Initialize candidate

        try:
            if len(history_for_model) > 1: 
                chat = model.start_chat(history=history_for_model[:-1])
                current_user_message_content = history_for_model[-1]['parts']
                llm_response_obj = await chat.send_message_async(
                    current_user_message_content, generation_config=generation_config, safety_settings=safety_settings)
            elif history_for_model: 
                current_user_message_content = history_for_model[0]['parts']
                llm_response_obj = await model.generate_content_async(
                    current_user_message_content, generation_config=generation_config, safety_settings=safety_settings)
            else: 
                return {"raw_response_text": "NovaSpark Info: No input query provided to LLM.", "prompt_token_count": 0, "response_token_count": 0}

            if llm_response_obj.candidates: 
                candidate = llm_response_obj.candidates[0]
                if candidate.finish_reason not in [candidate.FinishReason.STOP, candidate.FinishReason.MAX_TOKENS]:
                    logger.warning(f"NovaSpark LLM Warning: Response generation finished with reason: {candidate.finish_reason.name}. Content may be incomplete or blocked.")
                    if candidate.finish_reason == candidate.FinishReason.SAFETY:
                         llm_response_text = json.dumps({"main_answer_text": "NovaSpark Safety: I'm unable to respond to this specific query due to safety guidelines. Could you please rephrase or ask something else?", "suggested_follow_ups": [], "generated_analogies_for_answer": [], "personalized_examples_for_answer": []}) # Safety fallback JSON
                
                if not llm_response_text and candidate.content and candidate.content.parts: # Only populate if not set by safety
                    llm_response_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text])
                
                if hasattr(llm_response_obj, 'usage_metadata') and llm_response_obj.usage_metadata: 
                    prompt_tokens = llm_response_obj.usage_metadata.prompt_token_count
                    response_tokens = llm_response_obj.usage_metadata.candidates_token_count
            
            if not llm_response_text and candidate and candidate.finish_reason == candidate.FinishReason.STOP:
                 llm_response_text = "{}" 
                 logger.warning("NovaSpark LLM Warning: Successful stop but no text content from LLM. Assuming empty JSON.")

        except Exception as e_sdk:
            logger.error(f"NovaSpark SDK Error during LLM call: {e_sdk}", exc_info=True)
            raise 
        return {"raw_response_text": llm_response_text, "prompt_token_count": prompt_tokens, "response_token_count": response_tokens}

llm_client = VertexAITutorLLMClient(model_name=LLM_MODEL_NAME) 

async def fetch_processed_nlp_content(
    module_id: Optional[str], topic_id_to_focus: Optional[str], language_code: str
) -> Optional[NlpModuleProcessed]:
    # ... (fetch_processed_nlp_content from main (10).py - no changes needed for Task 4)
    if not module_id:
        logger.info("NovaSpark: No module_id provided to fetch NLP content.")
        return None
    
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL:
        logger.warning(f"NovaSpark: NLP_CONTENT_SERVICE_URL is not configured or is default. Mocking NLP content fetch for module_id: {module_id}.")
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
            try: return NlpModuleProcessed(**mock_nlp_json)
            except Exception as e_mock_parse: logger.error(f"NovaSpark Error: Failed to parse mock NLP content. Error: {e_mock_parse}"); return None
        return None 

    request_url = f"{NLP_CONTENT_SERVICE_URL}?module_id={module_id}&language_code={language_code}"
    logger.info(f"NovaSpark: Attempting to fetch NLP-processed content from: {request_url}")
    async with httpx.AsyncClient(timeout=llm_client.retry_config, transport=httpx.AsyncHTTPTransport(retries=3)) as client:
        try:
            response = await client.get(request_url)
            response.raise_for_status() 
            processed_module = NlpModuleProcessed(**response.json())
            logger.info(f"NovaSpark: Successfully fetched and parsed NLP content for module_id: {module_id}")
            return processed_module
        except httpx.HTTPStatusError as e:
            logger.error(f"NovaSpark HTTP Error fetching NLP content for module {module_id}: {e.response.status_code} - {e.response.text}", exc_info=True)
            if e.response.status_code == 404: logger.warning(f"NovaSpark: NLP content not found (404) for module {module_id}, lang {language_code}.")
            return None
        except Exception as e: 
            logger.error(f"NovaSpark Error parsing/validating NLP content for module {module_id}: {e}", exc_info=True)
            return None

# NovaSpark: Enhanced System Prompt Construction for Human-Like Interaction (Task 4)
def construct_innovateai_system_prompt(
    user_profile: UserProfileSnapshot, language_code: str,
    nlp_module_content: Optional[NlpModuleProcessed] = None,
    current_topic_id_in_focus: Optional[str] = None,
    request_context: Optional[TutorRequestContext] = None,
    is_user_frustrated: bool = False # Task 4: For sentiment adaptation
) -> str:
    system_message_parts = [
        f"You are Uplas AI Tutor, an exceptionally skilled, empathetic, and personalized AI learning partner. Your primary mission is to illuminate complex concepts with clarity, tailoring explanations, examples, and analogies to the user's unique profile and stated learning goals. Respond ONLY in [{language_code}].",
        # Task 4: Initial persona instruction based on user preference
        f"Your communication style should generally embody the user's preference: '{user_profile.preferred_tutor_persona}'."
    ]
    
    # Task 4: Inject specific persona snippets if preferred_tutor_persona matches a library key
    preferred_persona_key = user_profile.preferred_tutor_persona.lower().replace(" ", "_").replace("-", "_") # Normalize key
    if preferred_persona_key in PERSONA_LIBRARY:
        persona_data = PERSONA_LIBRARY[preferred_persona_key]
        system_message_parts.append(f"\n--- Embodying: {user_profile.preferred_tutor_persona} ---")
        system_message_parts.append(f"Key Characteristics: {persona_data['description']}")
        system_message_parts.append("A few ways to express this (use as inspiration, not verbatim):")
        for snippet in random.sample(persona_data['snippets'], min(len(persona_data['snippets']), 2)): # Add 1-2 random snippets
            system_message_parts.append(f"- \"{snippet}\"")
    else: # Default empathetic instruction if no specific library persona matches
        system_message_parts.append(
             "\nGenerally, be patient, encouraging, and deeply supportive. Break down information into digestible chunks. "
             "If a question is unclear, gently ask for clarification. If off-topic, politely guide back to Uplas educational content."
        )

    # Task 4: Conceptual: Sentiment-Adaptive Logic Instruction
    if is_user_frustrated:
        system_message_parts.append(
            "\n--- NovaSpark Adaptive Response: User May Be Frustrated ---"
            "IMPORTANT: The user's last message may indicate frustration. It is CRITICAL to respond with EXTRA empathy, patience, and encouragement. "
            "Acknowledge their feelings directly (e.g., 'I understand this can be frustrating...'). "
            "Offer to break the problem down into even smaller steps, or to try a completely different type of explanation or analogy. "
            "Reassure them that it's okay to struggle and that you're there to help them succeed. Prioritize their emotional state and rebuilding confidence. "
            "Lean heavily into the 'Empathetic Mentor' characteristics for this interaction, even if their general preference is different."
        )
    
    # Dynamic User Profile Injection & Personalization Instructions (from Task 2 enhancements)
    profile_details_for_prompt = []
    # ... (profile injection logic from previous version - main (10).py)
    if user_profile.profession: profile_details_for_prompt.append(f"- Profession: {user_profile.profession}")
    if user_profile.industry: profile_details_for_prompt.append(f"- Industry: {user_profile.industry}")
    if user_profile.career_interest: profile_details_for_prompt.append(f"- Stated Career Interest: {user_profile.career_interest}")
    if user_profile.learning_goals: profile_details_for_prompt.append(f"- Learning Goals: \"{user_profile.learning_goals}\"")
    if user_profile.current_knowledge_level:
        knowledge_str = ", ".join([f"{topic}: {level}" for topic, level in user_profile.current_knowledge_level.items()])
        profile_details_for_prompt.append(f"- Current Knowledge (self-assessed): {knowledge_str if knowledge_str else 'Not specified'}")
    if user_profile.hobbies_and_interests: profile_details_for_prompt.append(f"- Personal Hobbies/Interests: {', '.join(user_profile.hobbies_and_interests)}")

    if profile_details_for_prompt:
        system_message_parts.append("\n--- User Profile Context for Personalization ---")
        system_message_parts.extend(profile_details_for_prompt)
        system_message_parts.append(
            "\n--- NovaSpark Personalization Instructions (Task 2) ---"
            "1. Deeply analyze the User Profile Context."
            "2. GENERATE analogies and examples that directly relate to the user's Profession, Industry, Career Interest, or Hobbies/Interests."
            "3. Adjust explanation DEPTH based on their 'Current Knowledge'. Simplify for 'Beginner', be more technical for 'Advanced'."
            "4. Align answers with their 'Learning Goals' and 'Career Interest'."
        )
    else:
        system_message_parts.append("\nUser profile details were not extensively provided. Use general best practices for clear explanations.")

    # Utilizing NLP-processed content & instructions (from Task 2 enhancements)
    # ... (NLP content injection logic from main (10).py - seems fine)
    focused_material_header = "\n--- Focused Learning Material from Uplas NLP Engine ---"
    general_material_header = "\n--- General Module Context from Uplas NLP Engine ---"
    material_to_include = ""
    if nlp_module_content and nlp_module_content.lessons:
        if current_topic_id_in_focus:
            for lesson in nlp_module_content.lessons:
                for topic in lesson.topics:
                    if topic.topic_id == current_topic_id_in_focus:
                        material_to_include = f"{focused_material_header}\nTopic: {topic.topic_title}\nKey Concepts: {', '.join(topic.key_concepts)}\nMaterial with Semantic Tags: {topic.content_with_tags}"
                        break
                if material_to_include: break
        if not material_to_include and nlp_module_content.lessons: 
            first_lesson = nlp_module_content.lessons[0]
            material_to_include = f"{general_material_header}\nModule: {nlp_module_content.module_title or 'Current Module'}\nLesson: {first_lesson.lesson_title}\nSummary: {first_lesson.lesson_summary or 'Refer to topics.'}"
            if first_lesson.topics: material_to_include += f"\nFirst Topic Preview ({first_lesson.topics[0].topic_title}): {first_lesson.topics[0].content_with_tags[:1000]}"
        if material_to_include:
            system_message_parts.append(material_to_include[:4000]) 
            system_message_parts.append(
                "\n--- NovaSpark Content Interaction & Personalization Instructions (Task 2) ---"
                "The 'Material with Semantic Tags' may contain XML-like tags like `<analogy type=\"...\" />` or `<example domain=\"...\" />`. When you see these, ALWAYS try to generate a RELEVANT analogy/example based on the USER'S PROFILE AND the tag's suggestion."
                "Use `<difficulty ... />` tags to tailor depth, ALSO considering user's 'Current Knowledge'."
            )
        else:
             system_message_parts.append("\nNo specific pre-processed learning material was found for this query's context. Rely on your general knowledge and the user's profile for personalization.")
    
    # Project Remediation Guidance (from Task 2 enhancements)
    # ... (Project remediation logic from main (10).py - seems fine)
    if request_context and request_context.project_assessment_feedback:
        system_message_parts.append("\n--- NovaSpark Guidance for Project Remediation (Task 2) ---")
        system_message_parts.append(
            f"The user is seeking help after a project assessment. Project: '{request_context.current_project_title or 'N/A'}'. Assessment Feedback: '{request_context.project_assessment_feedback}'. "
            "Adopt an **Empathetic Mentor** role. Acknowledge their effort. Gently help them diagnose *why* they struggled, linking to concepts and their 'Current Knowledge'. "
            "Offer concrete, actionable steps and simpler examples. Maintain a supportive, encouraging, patient tone."
        )

    # Task 4: Few-shot examples for empathetic and engaging responses
    system_message_parts.append(
        "\n--- NovaSpark Examples of Empathetic & Engaging Interaction (Task 4) ---"
        "1. Example of Empathetic Response to Frustration:"
        "   User: \"I'm so lost with this, I'll never understand it.\""
        "   Tutor (Empathetic Mentor Style): \"It's completely okay to feel that way when you're tackling something new and complex! Many learners find this part challenging. You're not alone. Let's try breaking it down into smaller pieces. What's one specific part that feels the most confusing right now? We'll figure it out together.\""
        "2. Example of Engaging Follow-up (after explaining a concept):"
        "   Tutor (Susan Style): \"So, that's the core idea behind [concept]. Now, thinking about your work as a [user_profession], can you imagine a scenario where understanding this could be particularly useful or change how you approach a task?\""
    )
    
    # Output Format Mandate (from Task 2 enhancements)
    system_message_parts.append(
        "\n--- NovaSpark Output Format Mandate ---"
        "Your entire response MUST be a single, valid JSON object. Do not add any text before or after this JSON object. "
        "The JSON object must strictly adhere to the following Pydantic schema (StructuredLLMOutput): "
        f"{StructuredLLMOutput.model_json_schema(indent=2)}" 
        f"All text content within this JSON MUST be in [{language_code}]."
        "Prioritize 'main_answer_text', 'personalized_examples_for_answer', and 'generated_analogies_for_answer'."
    )
    
    final_prompt = "\n".join(system_message_parts)
    logger.debug(f"NovaSpark Generated System Prompt (first 500 chars for user {user_profile.preferred_tutor_persona if user_profile else 'N/A'}): {final_prompt[:500]}...")
    return final_prompt

# --- API Endpoint (NovaSpark Enhanced) ---
@app.post("/v1/ask-tutor", response_model=AiTutorQueryResponse, summary="Get a personalized, empathetic answer from the AI Tutor (NovaSpark Enhanced)")
async def ask_tutor_endpoint(request_data: AiTutorQueryRequest): 
    processing_start_time = time.perf_counter()
    effective_language_code = request_data.language_code or DEFAULT_LANGUAGE 

    nlp_processed_module: Optional[NlpModuleProcessed] = None
    current_topic_id: Optional[str] = None
    
    # Task 4: Conceptual: Simple check for user frustration keywords
    is_user_frustrated_detected = False
    negative_sentiment_keywords = ["stuck", "hate this", "confused", "impossible", "don't understand", "frustrated", "lost", "hard time"]
    if any(keyword in request_data.query_text.lower() for keyword in negative_sentiment_keywords):
        is_user_frustrated_detected = True
        logger.info(f"NovaSpark: Potential user frustration detected in query for user {request_data.user_id}.")

    if request_data.context and request_data.context.module_id:
        current_topic_id = request_data.context.topic_id 
        logger.info(f"NovaSpark: Module context: module_id='{request_data.context.module_id}', topic_id='{current_topic_id}'. Fetching NLP content.")
        nlp_processed_module = await fetch_processed_nlp_content(
            request_data.context.module_id, current_topic_id, effective_language_code)
        if nlp_processed_module:
            if not request_data.context.current_topic_title and current_topic_id:
                for lesson in nlp_processed_module.lessons:
                    for topic_item in lesson.topics: 
                        if topic_item.topic_id == current_topic_id:
                            request_data.context.current_topic_title = topic_item.topic_title
                            logger.info(f"NovaSpark: Auto-populated topic title: '{topic_item.topic_title}'")
                            if nlp_processed_module.module_title and not request_data.context.current_course_title :
                                 request_data.context.current_course_title = nlp_processed_module.module_title 
                                 logger.info(f"NovaSpark: Auto-populated course title: '{nlp_processed_module.module_title}'")
                            break
                    if request_data.context.current_topic_title: break
            if not nlp_processed_module.module_title and request_data.context.current_course_title:
                 logger.info(f"NovaSpark: Using course title from request context: '{request_data.context.current_course_title}' as NLP module had no title.")
        else:
            logger.warning(f"NovaSpark: Failed to fetch or parse NLP content for module_id: {request_data.context.module_id}")
    
    system_prompt_for_llm = construct_innovateai_system_prompt( 
        user_profile=request_data.user_profile_snapshot, language_code=effective_language_code,
        nlp_module_content=nlp_processed_module, current_topic_id_in_focus=current_topic_id,
        request_context=request_data.context, is_user_frustrated=is_user_frustrated_detected # Task 4: Pass frustration flag
    )

    llm_conversation_history: List[Dict[str, Any]] = [] 
    for turn in request_data.conversation_history:
        role = 'user' if turn.role == ConversationRole.USER.value else 'model'
        llm_conversation_history.append({'role': role, 'content': turn.content})
    llm_conversation_history.append({'role': ConversationRole.USER.value, 'content': request_data.query_text})

    structured_llm_output_obj: Optional[StructuredLLMOutput] = None
    llm_response_metrics = {}
    raw_json_string_from_llm = "{}" 

    try:
        logger.info(f"NovaSpark: Sending request to LLM for user {request_data.user_id}, query snippet: \"{request_data.query_text[:100]}...\"")
        llm_response_metrics = await llm_client.generate_structured_response( 
            system_prompt=system_prompt_for_llm, conversation_turns=llm_conversation_history,
            max_output_tokens=request_data.max_tokens_response, response_pydantic_model=StructuredLLMOutput)
        
        raw_json_string_from_llm = llm_response_metrics.get("raw_response_text", "{}")
        
        if not raw_json_string_from_llm.strip(): 
            logger.error("NovaSpark Error: LLM returned a completely empty response string.")
            # Task 4: More empathetic error message if LLM returns nothing
            fallback_message = "I'm sorry, I seem to be having a moment of silence. Could you try asking that again, perhaps in a different way?"
            structured_llm_output_obj = StructuredLLMOutput(main_answer_text=fallback_message, suggested_follow_ups=[], generated_analogies_for_answer=[], personalized_examples_for_answer=[])
        else:
            try:
                llm_output_dict = json.loads(raw_json_string_from_llm)
                structured_llm_output_obj = StructuredLLMOutput(**llm_output_dict)
                logger.info(f"NovaSpark: Successfully parsed structured LLM output for user {request_data.user_id}.")
            except json.JSONDecodeError as e_json:
                logger.error(f"NovaSpark Error: LLM returned invalid JSON: {raw_json_string_from_llm[:500]}. Error: {e_json}", exc_info=True)
                structured_llm_output_obj = StructuredLLMOutput(main_answer_text=f"My thoughts got a bit jumbled translating into our usual format! Here's the raw information I was processing: \"{raw_json_string_from_llm}\". Can we try approaching your question from another angle?", suggested_follow_ups=[], generated_analogies_for_answer=[], personalized_examples_for_answer=[])
            except Exception as e_pydantic: 
                logger.error(f"NovaSpark Error: Pydantic validation failed for LLM JSON: {raw_json_string_from_llm[:500]}. Error: {e_pydantic}", exc_info=True)
                structured_llm_output_obj = StructuredLLMOutput(main_answer_text=f"My response format was a little off this time. Here's what I gathered: \"{raw_json_string_from_llm}\". Perhaps we can try a different way to look at your query?", suggested_follow_ups=[], generated_analogies_for_answer=[], personalized_examples_for_answer=[])

    except EnvironmentError as e_env: 
        logger.error(f"NovaSpark Config Error for LLM: {e_env}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI service configuration error: {e_env}")
    except Exception as e_llm: 
        logger.error(f"NovaSpark LLM Communication/Processing Error: {e_llm}", exc_info=True)
        user_friendly_detail = "I'm having a bit of trouble connecting to my knowledge circuits right now. Please try again in a few moments."
        if "timeout" in str(e_llm).lower(): 
            user_friendly_detail = "It's taking me a little longer than usual to process that. Could you try again shortly?"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=user_friendly_detail)

    if not structured_llm_output_obj: 
        logger.error("NovaSpark Critical: structured_llm_output_obj is None after LLM call and fallbacks.")
        structured_llm_output_obj = StructuredLLMOutput(
            main_answer_text="I'm truly sorry, I was unable to formulate a response this time. Please try asking in a different way or check back soon.",
            suggested_follow_ups=[], generated_analogies_for_answer=[], personalized_examples_for_answer=[]
        )

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000 

    debug_information = DebugInfo( 
        llm_model_name_used=llm_client.model_name, processing_time_ms=round(processing_time_ms, 2),
        prompt_token_count=llm_response_metrics.get("prompt_token_count"), 
        response_token_count=llm_response_metrics.get("response_token_count"),
        language_used=effective_language_code
    )
    if os.getenv("UPLAS_AI_TUTOR_DEBUG_PROMPT", "false").lower() == "true": 
        full_prompt_for_debug = system_prompt_for_llm + "\n--- Conversation History ---\n" + \
            "\n".join([f"Role: {turn['role']}, Content Snippet: {turn['content'][:200]}..." for turn in llm_conversation_history])
        debug_information.prompt_sent_to_llm_sample = full_prompt_for_debug[:2000] + ("..." if len(full_prompt_for_debug) > 2000 else "")
        logger.info(f"NovaSpark DEBUG Full Prompt Sample (User {request_data.user_id}): {debug_information.prompt_sent_to_llm_sample}")

    return AiTutorQueryResponse(
        answer_text=structured_llm_output_obj.main_answer_text,
        suggested_follow_up_questions=structured_llm_output_obj.suggested_follow_ups,
        generated_analogies=[GeneratedAnalogy(analogy=text) for text in structured_llm_output_obj.generated_analogies_for_answer],
        personalized_examples=structured_llm_output_obj.personalized_examples_for_answer,
        confidence_score=structured_llm_output_obj.answer_confidence_score,
        debug_info=debug_information
    )

@app.get("/health", status_code=status.HTTP_200_OK) 
async def health_check():
    # ... (health check logic from main (10).py - no changes needed for Task 4)
    service_name = "PersonalizedTutorNLP_LLM_Agent_NovaSpark_Empathetic" # Updated service name
    if not GCP_PROJECT_ID:
        return {"status": "unhealthy", "reason": "GCP_PROJECT_ID not configured.", "service": service_name}
    nlp_service_ok = True
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL: 
         logger.warning("NovaSpark Health Warning: NLP_CONTENT_SERVICE_URL not configured or using default.")
         nlp_service_ok = False 
    if not nlp_service_ok:
        return {"status": "degraded", "reason": "NLP_CONTENT_SERVICE_URL not optimally configured.", "service": service_name, "innovate_ai_enhancements_active": True}
    return {"status": "healthy", "service": service_name, "innovate_ai_enhancements_active": True}

if __name__ == "__main__": 
    import uvicorn
    logger.info(f"NovaSpark: Starting {app.title} v{app.version} for local development...")
    if not GCP_PROJECT_ID: print("NovaSpark Warning: GCP_PROJECT_ID is not set.")
    if not NLP_CONTENT_SERVICE_URL or "your-backend-service" in NLP_CONTENT_SERVICE_URL:
        print("NovaSpark Warning: NLP_CONTENT_SERVICE_URL not optimally configured.")
    port = int(os.getenv("PORT", 8001)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
