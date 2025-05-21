from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import httpx # For potential future calls to Django to fetch content
import os
from enum import Enum
from unittest.mock import MagicMock # For mocking the LLM call
import time # For simulating processing time
# from datetime import datetime # FastAPIRequest uses datetime from pydantic/python's datetime

# --- Pydantic Models for API Contract ---

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
    # Add any other fields from Django User/UserProfile that are relevant for personalization


class TutorRequestContext(BaseModel):
    course_id: Optional[str] = None
    topic_id: Optional[str] = None
    project_id: Optional[str] = None
    previous_assessment_feedback: Optional[str] = None
    current_topic_title: Optional[str] = Field(None, examples=["Introduction to Python Lists"])
    current_course_title: Optional[str] = Field(None, examples=["Python for Beginners"])
    current_project_title: Optional[str] = Field(None, examples=["Data Analysis Mini-Project"])


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
    query_text: str = Field(..., min_length=3, examples=["Can you explain Python decorators?"])
    context: Optional[TutorRequestContext] = None
    user_profile_snapshot: UserProfileSnapshot
    conversation_history: Optional[List[ConversationTurn]] = Field(default_factory=list)
    max_tokens_response: Optional[int] = Field(300, ge=50, le=2048) # Adjusted max for more comprehensive answers


class GeneratedAnalogy(BaseModel):
    analogy: str
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class DebugInfo(BaseModel):
    prompt_sent_to_llm: Optional[str] = None # Be cautious about logging full prompts with PII in production
    llm_model_name_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    prompt_token_count: Optional[int] = None # If LLM API provides it
    response_token_count: Optional[int] = None # If LLM API provides it


class AiTutorQueryResponse(BaseModel):
    answer_text: str
    suggested_follow_up_questions: List[str] = Field(default_factory=list)
    generated_analogies: List[GeneratedAnalogy] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    debug_info: Optional[DebugInfo] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Tutor Agent",
    description="Provides personalized explanations and guidance using a (mocked) LLM.",
    version="0.1.1"
)

# --- Mock LLM (Vertex AI Gemini) Client ---
MOCKED_LLM_NAME = "mocked-gemini-pro-vLocal"

class MockLLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # No MagicMock here, we'll implement simple logic directly in generate_response for now

    async def generate_response(self, prompt: str, max_tokens: int, user_profile: UserProfileSnapshot, context: Optional[TutorRequestContext]) -> Dict[str, Any]:
        """
        Simulates LLM response based on prompt content.
        This logic needs to be expanded for more varied and realistic mock responses.
        """
        print(f"\n--- Mock LLM ({self.model_name}) Received Prompt (first 300 chars): ---")
        print(f"{prompt[:300]}...")
        print("---------------------------------------------------------------------\n")

        answer = f"This is a mocked, personalized answer to your query about '{prompt.split('--- User\\'s Question ---')[-1].strip()[:50]}...' based on your profile as a {user_profile.profession or 'learner'} in {user_profile.industry or 'your field'}."
        analogies_list = []
        follow_ups = ["Can you give me an example?", "How does this relate to X?"]

        query_lower = prompt.lower()

        if "python lists" in query_lower:
            if user_profile.profession and "chef" in user_profile.profession.lower():
                analogy_text = "Think of a Python list like your recipe ingredient list - ordered, and you can change items (mutable)!"
                answer = f"{analogy_text} You can add ('append'), remove ('remove'/'pop'), or change ingredients, just like items in a Python list."
                analogies_list.append({"analogy": analogy_text, "relevance_score": 0.92})
            else:
                answer = "In Python, a list is an ordered and mutable collection of items. They are very versatile! For instance, `my_friends = ['Alice', 'Bob', 'Charlie']`."
            follow_ups = ["How do I access items in a list?", "What are list methods?"]
        elif "decorator" in query_lower:
            answer = "Python decorators are a powerful way to modify or enhance functions or methods. They are a form of metaprogramming. A decorator is a function that takes another function and extends its behavior without explicitly modifying it."
            if user_profile.profession and "software engineer" in user_profile.profession.lower():
                analogy_text = "Imagine a decorator as a wrapper you put around a gift (your function). The gift itself doesn't change, but the wrapper adds something extra, like a ribbon or a card!"
                answer += f" Analogy: {analogy_text}"
                analogies_list.append({"analogy": analogy_text, "relevance_score": 0.88})
            follow_ups = ["Can you show a code example of a decorator?", "What are common use cases?"]
        elif context and context.previous_assessment_feedback:
            answer = f"I see you received feedback: '{context.previous_assessment_feedback}'. Let's break that down. For the part about [mocked specific point from feedback], it means..."
            follow_ups = ["Can we go over [specific concept] again?", "What resources can help me with this?"]
        
        # Simulate token counts for debug info
        prompt_tokens = len(prompt.split()) # Very rough estimate
        response_tokens = len(answer.split())

        # Simulate some processing time
        # await asyncio.sleep(random.uniform(0.1, 0.5)) # If using asyncio

        return {
            "answer_text": answer,
            "generated_analogies": analogies_list,
            "suggested_follow_up_questions": follow_ups, # Added
            "prompt_token_count": prompt_tokens,
            "response_token_count": response_tokens
        }

mock_llm_client = MockLLMClient(model_name=MOCKED_LLM_NAME)

# --- Helper Functions ---

def construct_llm_prompt(
    query: str,
    user_profile: UserProfileSnapshot,
    course_content: Optional[str] = None,
    context_data: Optional[TutorRequestContext] = None,
    conversation_history: Optional[List[ConversationTurn]] = None
) -> str:
    system_message_parts = [
        "You are Uplas AI Tutor, a highly proficient, engaging, and personalized AI learning assistant.",
        "Your primary goal is to explain complex concepts clearly and concisely, tailored to the user's background and learning preferences.",
        "Always strive to use analogies and real-world examples relevant to their provided profile (profession, industry, location, interests).",
        f"The user indicates a preference for a '{user_profile.preferred_tutor_persona}' communication style. Adhere to this style.",
        "Be encouraging, patient, and supportive. Break down answers into digestible parts if the concept is complex.",
        "If a question is ambiguous, ask for clarification before attempting an answer.",
        "If a question is outside the scope of the platform's educational content (e.g., personal advice, harmful content, unrelated topics), politely state that you cannot answer it and try to guide them back to relevant Uplas topics.",
        "When generating analogies, if possible, explicitly state them using a prefix like 'Analogy:' or 'Think of it like this:' for clarity.",
        "Structure your answers well. Use bullet points or numbered lists for steps or multiple points if it enhances clarity.",
        "If relevant information or course content is provided, base your answer on that primarily, supplementing with your general knowledge."
    ]
    system_message = " ".join(system_message_parts)

    prompt_sections = [f"SYSTEM: {system_message}"]

    # User Profile Context Block
    profile_context_str = "\n--- User Profile Summary for Personalization ---"
    has_profile_info = False
    if user_profile.profession: profile_context_str += f"\n- Profession: {user_profile.profession}"; has_profile_info = True
    if user_profile.industry: profile_context_str += f"\n- Industry: {user_profile.industry}"; has_profile_info = True
    if user_profile.city or user_profile.country:
        location = f"{user_profile.city or ''}, {user_profile.country or ''}".strip(", ")
        if location: profile_context_str += f"\n- Location: {location}"; has_profile_info = True
    if user_profile.career_interest: profile_context_str += f"\n- Career Interest: {user_profile.career_interest}"; has_profile_info = True
    if user_profile.learning_goals: profile_context_str += f"\n- Stated Learning Goals: {user_profile.learning_goals}"; has_profile_info = True
    if user_profile.learning_style_preference:
        styles = ", ".join([f"{k} (preference: {v*100:.0f}%)" for k, v in user_profile.learning_style_preference.items() if v > 0.3]) # Only mention prominent styles
        if styles: profile_context_str += f"\n- Noted Learning Styles: {styles}"; has_profile_info = True
    if user_profile.current_knowledge_level:
        knowledge = ", ".join([f"{k}: {v}" for k, v in user_profile.current_knowledge_level.items()])
        if knowledge: profile_context_str += f"\n- Self-Assessed Knowledge: {knowledge}"; has_profile_info = True
    
    if has_profile_info:
        prompt_sections.append(profile_context_str)
    else:
        prompt_sections.append("\nUser profile details for personalization were not extensively provided.")

    # Specific Question Context Block
    if context_data:
        context_info_str = "\n--- Current Learning Context ---"
        has_context_info = False
        if context_data.current_course_title: context_info_str += f"\n- Course: {context_data.current_course_title}"; has_context_info = True
        if context_data.current_topic_title: context_info_str += f"\n- Topic: {context_data.current_topic_title}"; has_context_info = True
        if context_data.current_project_title: context_info_str += f"\n- Project: {context_data.current_project_title}"; has_context_info = True
        if context_data.previous_assessment_feedback:
            context_info_str += f"\n- Note: The user previously received this feedback on a related project: '{context_data.previous_assessment_feedback}' Address this if relevant to their query."; has_context_info = True
        if has_context_info:
            prompt_sections.append(context_info_str)

    # Main Course/Topic Content for RAG
    if course_content:
        prompt_sections.append("\n--- Relevant Information/Course Material Snippet ---")
        prompt_sections.append(course_content[:3000]) # Truncate to avoid overly long prompts (adjust limit)

    # Conversation History
    if conversation_history:
        prompt_sections.append("\n--- Recent Conversation History (User and Assistant) ---")
        for turn in conversation_history[-6:]: # Last 3 user/assistant pairs
            prompt_sections.append(f"{turn.role.value.upper()}: {turn.content}")

    prompt_sections.append(f"\n--- User's Current Question ---")
    prompt_sections.append(f"USER: {query}")
    prompt_sections.append("\n--- Your Personalized Answer ---")
    prompt_sections.append("ASSISTANT:")

    return "\n".join(prompt_sections)


async def fetch_course_content_from_django(topic_id: Optional[str], course_id: Optional[str]) -> Optional[str]:
    # This function remains mocked for GitHub-centric development.
    # In production, it would make an authenticated HTTP call to an internal Django API endpoint.
    # e.g., GET /api/internal/content/?topic_id=<topic_id>&course_id=<course_id>
    if topic_id == "topic_python_lists_id":
        return "Python lists are versatile, ordered, and mutable sequences. Key methods include append(), extend(), insert(), remove(), pop(), index(), count(), sort(), reverse(). Slicing is also powerful."
    if topic_id == "topic_django_models_id":
        return "In Django, a model is the single, definitive source of information about your data. It contains the essential fields and behaviors of the data you’re storing. Generally, each model maps to a single database table."
    if course_id == "course_python_basics_id" and not topic_id:
        return "This is the Python for Beginners course. It covers fundamental concepts like variables, data types (integers, strings, lists, dictionaries), loops, conditionals, functions, and basic object-oriented principles."
    
    # Generic mock content if no specific match
    mock_content = "Placeholder content: "
    if topic_id: mock_content += f"Detailed information about topic '{topic_id}'. "
    if course_id: mock_content += f"Context from course '{course_id}'. "
    return mock_content if topic_id or course_id else None

# --- API Endpoint ---
@app.post("/v1/ask-tutor", response_model=AiTutorQueryResponse, summary="Get a personalized answer from the AI Tutor")
async def ask_tutor_endpoint(
    request_data: AiTutorQueryRequest,
    # fastapi_request: FastAPIRequest # For accessing raw request if needed, e.g. for client IP
):
    processing_start_time = time.perf_counter()

    course_material_context: Optional[str] = None
    if request_data.context:
        course_material_context = await fetch_course_content_from_django(
            request_data.context.topic_id, 
            request_data.context.course_id
        )

    prompt = construct_llm_prompt(
        query=request_data.query_text,
        user_profile=request_data.user_profile_snapshot,
        course_content=course_material_context,
        context_data=request_data.context,
        conversation_history=request_data.conversation_history
    )

    try:
        llm_response_data = await mock_llm_client.generate_response(
            prompt=prompt,
            max_tokens=request_data.max_tokens_response,
            user_profile=request_data.user_profile_snapshot,
            context=request_data.context
        )
        answer_text = llm_response_data.get("answer_text", "I apologize, I couldn't generate a response for that.")
        raw_analogies = llm_response_data.get("generated_analogies", [])
        suggested_f_ups = llm_response_data.get("suggested_follow_up_questions", [])

        parsed_analogies = []
        if isinstance(raw_analogies, list):
            for analogy_item in raw_analogies:
                if isinstance(analogy_item, dict) and "analogy" in analogy_item:
                    parsed_analogies.append(GeneratedAnalogy(**analogy_item))
                elif isinstance(analogy_item, str):
                     parsed_analogies.append(GeneratedAnalogy(analogy=analogy_item))

    except Exception as e:
        print(f"LLM Communication/Processing Error: {e}") # Log full error
        raise HTTPException(status_code=503, detail="There was an issue communicating with the AI assistance service.")

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    # Build debug info
    debug_information = DebugInfo(
        llm_model_name_used=mock_llm_client.model_name,
        processing_time_ms=round(processing_time_ms, 2),
        prompt_token_count=llm_response_data.get("prompt_token_count"),
        response_token_count=llm_response_data.get("response_token_count")
    )
    if os.getenv("UPLAS_DEBUG_MODE", "false").lower() == "true": # Only include full prompt in debug mode
        debug_information.prompt_sent_to_llm = prompt


    return AiTutorQueryResponse(
        answer_text=answer_text,
        suggested_follow_up_questions=suggested_f_ups,
        generated_analogies=parsed_analogies,
        confidence_score=0.88, # Mocked, could be derived from LLM if it provides logits/confidence
        debug_info=debug_information
    )

# Example of how to run this locally:
# cd uplas-ai-agents/personalized_tutor_nlp_llm
# uvicorn main:app --reload --port 8001
