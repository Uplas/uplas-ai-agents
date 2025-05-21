from fastapi import FastAPI, HTTPException, Depends, Request as FastAPIRequest
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import httpx # For making API calls (e.g., back to Django to fetch course content if needed)
import os
from enum import Enum
from unittest.mock import MagicMock # For mocking the LLM call

# --- Pydantic Models for API Contract ---

class UserProfileSnapshot(BaseModel):
    industry: Optional[str] = Field(None, examples=["Technology"])
    profession: Optional[str] = Field(None, examples=["Software Engineer"])
    country: Optional[str] = Field(None, examples=["Kenya"])
    city: Optional[str] = Field(None, examples=["Nairobi"])
    preferred_tutor_persona: Optional[str] = Field("Friendly and encouraging", examples=["Socratic", "Technical"])
    # Example: learning_style_preference: {"visual": 0.7, "kinesthetic": 0.3}
    learning_style_preference: Optional[Dict[str, float]] = Field(default_factory=dict)
    # Example: current_knowledge_level: {"Python": "Intermediate", "TopicX": "Beginner"}
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict)
    career_interest: Optional[str] = Field(None, examples=["AI Specialist"])
    learning_goals: Optional[str] = Field(None, examples=["Master machine learning concepts."])


class TutorRequestContext(BaseModel):
    course_id: Optional[str] = None
    topic_id: Optional[str] = None
    project_id: Optional[str] = None
    previous_assessment_feedback: Optional[str] = None
    # Add any other contextual info from Django, e.g., current module/topic title
    current_topic_title: Optional[str] = None
    current_course_title: Optional[str] = None
    current_project_title: Optional[str] = None


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system" # For initial instructions to the LLM


class ConversationTurn(BaseModel):
    role: ConversationRole
    content: str


class AiTutorQueryRequest(BaseModel):
    user_id: str = Field(..., examples=["user_uuid_123"])
    session_id: Optional[str] = Field(None, examples=["session_uuid_abc"]) # For conversation tracking
    query_text: str = Field(..., examples=["Can you explain Python decorators?"])
    context: Optional[TutorRequestContext] = None
    user_profile_snapshot: UserProfileSnapshot
    conversation_history: Optional[List[ConversationTurn]] = Field(default_factory=list, examples=[
        [{"role": "user", "content": "What are lists?"}, {"role": "assistant", "content": "Lists are ordered collections..."}]
    ])
    max_tokens_response: Optional[int] = Field(250, ge=50, le=1024) # Limit LLM response length


class GeneratedAnalogy(BaseModel):
    analogy: str
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class DebugInfo(BaseModel):
    prompt_sent_to_llm: Optional[str] = None
    llm_model_name_used: Optional[str] = None
    processing_time_ms: Optional[float] = None


class AiTutorQueryResponse(BaseModel):
    answer_text: str
    suggested_follow_up_questions: Optional[List[str]] = Field(default_factory=list)
    generated_analogies: Optional[List[GeneratedAnalogy]] = Field(default_factory=list)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0) # Agent's confidence in the answer
    debug_info: Optional[DebugInfo] = None


# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Tutor Agent",
    description="Provides personalized explanations and guidance using an LLM.",
    version="0.1.0"
)

# --- Mock LLM (Vertex AI Gemini) Client ---
# In a real scenario, this would be an actual client for Vertex AI
MOCKED_LLM_NAME = "mocked-gemini-pro"

class MockLLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.llm_mock = MagicMock()
        # Configure the mock's behavior
        self.llm_mock.predict.side_effect = self._mock_predict_logic

    def _mock_predict_logic(self, prompt: str, max_tokens: int, user_profile: UserProfileSnapshot, context: Optional[TutorRequestContext]):
        """Simulates LLM response based on prompt content."""
        answer = f"Mocked answer for: '{prompt[:50]}...' based on your profile as a {user_profile.profession or 'learner'} in {user_profile.industry or 'your field'}."
        if context and context.current_topic_title:
            answer += f" This relates to '{context.current_topic_title}'."
        
        if "explain python lists" in prompt.lower():
            if user_profile.profession and "chef" in user_profile.profession.lower():
                analogy = "Think of a Python list like your recipe ingredient list - ordered and you can change items!"
                answer = f"{analogy} You can add, remove, or change ingredients just like list items."
                return {"text": answer, "analogies": [{"analogy": analogy, "relevance_score": 0.9}]}
            answer = "A Python list is an ordered, mutable collection of items. For example, `my_list = [1, 'apple', True]`."
            return {"text": answer, "analogies": []}

        if "analogy for" in prompt.lower():
            concept = prompt.split("analogy for")[-1].split("for a")[0].strip()
            analogy_text = f"An analogy for '{concept}' for a {user_profile.profession or 'learner'} in {user_profile.industry or 'general terms'} would be like [mocked specific analogy here]."
            return {"text": analogy_text, "analogies": [{"analogy": analogy_text, "relevance_score": 0.75}]}

        return {"text": answer, "analogies": []}


    async def generate_response(self, prompt: str, max_tokens: int, user_profile: UserProfileSnapshot, context: Optional[TutorRequestContext]) -> Dict[str, Any]:
        # In a real implementation, this would make an async call to Vertex AI
        # For now, use the mock
        print(f"\n--- Sending to Mock LLM ({self.model_name}) ---")
        print(f"Prompt (first 100 chars): {prompt[:100]}...")
        print(f"Max Tokens: {max_tokens}")
        print(f"User Profile Profession: {user_profile.profession}")
        print(f"Context Topic: {context.current_topic_title if context else 'N/A'}")
        print("--------------------------------------------\n")
        
        # Simulate some processing time
        # await asyncio.sleep(0.1) 
        
        response_data = self.llm_mock.predict(prompt=prompt, max_tokens=max_tokens, user_profile=user_profile, context=context)
        return {
            "answer_text": response_data.get("text", "Sorry, I couldn't generate a response."),
            "generated_analogies": response_data.get("analogies", [])
        }

# Initialize the mock client (dependency for the endpoint)
mock_llm_client = MockLLMClient(model_name=MOCKED_LLM_NAME)

# --- Helper Functions ---

def construct_llm_prompt(
    query: str,
    user_profile: UserProfileSnapshot,
    course_content: Optional[str] = None,
    context_data: Optional[TutorRequestContext] = None,
    conversation_history: Optional[List[ConversationTurn]] = None
) -> str:
    """
    Constructs a detailed prompt for the LLM based on all available information.
    """
    system_message = (
        "You are Uplas AI Tutor, a highly proficient, engaging, and personalized AI learning assistant. "
        "Your goal is to explain complex concepts clearly and concisely, tailored to the user's background and learning preferences. "
        "Use analogies and real-world examples relevant to their profile. "
        f"The user prefers a '{user_profile.preferred_tutor_persona}' communication style. "
        "Be encouraging and supportive. If a question is outside the scope of the platform's educational content or AI, "
        "politely state that you cannot answer it and try to guide them back to relevant topics. "
        "If you generate analogies, try to make them explicit."
    )

    prompt_parts = [system_message]

    # User Profile Context
    profile_context_parts = ["\n--- User Context ---"]
    if user_profile.profession: profile_context_parts.append(f"Profession: {user_profile.profession}")
    if user_profile.industry: profile_context_parts.append(f"Industry: {user_profile.industry}")
    if user_profile.city or user_profile.country:
        location = f"{user_profile.city}, {user_profile.country}".strip(", ")
        if location: profile_context_parts.append(f"Location: {location}")
    if user_profile.career_interest: profile_context_parts.append(f"Career Interest: {user_profile.career_interest}")
    if user_profile.learning_goals: profile_context_parts.append(f"Learning Goals: {user_profile.learning_goals}")
    if user_profile.learning_style_preference:
        styles = ", ".join([f"{k} ({v*100:.0f}%)" for k, v in user_profile.learning_style_preference.items()])
        if styles: profile_context_parts.append(f"Learning Style Preferences: {styles}")
    if user_profile.current_knowledge_level:
        knowledge = ", ".join([f"{k}: {v}" for k, v in user_profile.current_knowledge_level.items()])
        if knowledge: profile_context_parts.append(f"Current Knowledge: {knowledge}")
    
    if len(profile_context_parts) > 1:
        prompt_parts.extend(profile_context_parts)

    # Specific Question Context
    if context_data:
        context_info_parts = ["\n--- Question Context ---"]
        if context_data.current_course_title: context_info_parts.append(f"Course: {context_data.current_course_title}")
        if context_data.current_topic_title: context_info_parts.append(f"Topic: {context_data.current_topic_title}")
        if context_data.current_project_title: context_info_parts.append(f"Project: {context_data.current_project_title}")
        if context_data.previous_assessment_feedback:
            context_info_parts.append(f"Feedback from previous project attempt: {context_data.previous_assessment_feedback}")
        if len(context_info_parts) > 1:
            prompt_parts.extend(context_info_parts)

    # Main Course/Topic Content for RAG (Retrieval Augmented Generation)
    if course_content:
        prompt_parts.append("\n--- Relevant Information ---")
        prompt_parts.append(course_content[:2000]) # Truncate to avoid overly long prompts

    # Conversation History
    if conversation_history:
        prompt_parts.append("\n--- Conversation History ---")
        for turn in conversation_history[-5:]: # Include last 5 turns for context
            prompt_parts.append(f"{turn.role.capitalize()}: {turn.content}")

    # The User's Actual Question
    prompt_parts.append("\n--- User's Question ---")
    prompt_parts.append(query)
    prompt_parts.append("\n--- Your Answer ---\n(Provide a clear, personalized explanation. If appropriate, explicitly state an analogy using 'Analogy:' prefix and then explain it.)")

    return "\n".join(prompt_parts)


async def fetch_course_content_from_django(topic_id: Optional[str]) -> Optional[str]:
    """
    (Mockable) Fetches relevant course content from Django backend.
    In a real setup, this would make an authenticated HTTP call.
    """
    if not topic_id:
        return None
    
    # UPLAS_BACKEND_API_URL = os.getenv("UPLAS_BACKEND_API_URL", "http://localhost:8000/api/internal")
    # UPLAS_INTERNAL_API_KEY = os.getenv("UPLAS_INTERNAL_API_KEY")
    # headers = {"Authorization": f"InternalAPIKey {UPLAS_INTERNAL_API_KEY}"}
    # async with httpx.AsyncClient() as client:
    #     try:
    #         response = await client.get(f"{UPLAS_BACKEND_API_URL}/topics/{topic_id}/content/", headers=headers)
    #         response.raise_for_status()
    #         return response.json().get("text_content_html") # Assuming this structure
    #     except httpx.HTTPStatusError as e:
    #         print(f"Error fetching content for topic {topic_id} from Django: {e.response.status_code}")
    #         return None
    #     except Exception as e:
    #         print(f"Network error fetching content for topic {topic_id}: {e}")
    #         return None

    # Mocked content for now
    if topic_id == "topic_python_lists":
        return "Python lists are ordered collections that are mutable (changeable) and allow duplicate members. Example: my_list = ['apple', 'banana', 'cherry']"
    elif topic_id == "topic_django_models":
        return "Django models are Python classes that represent database tables. Each attribute of the model represents a database field."
    return f"Mocked detailed content for topic ID: {topic_id}. This content would be used by the LLM for context."


# --- API Endpoint ---

@app.post("/v1/ask-tutor", response_model=AiTutorQueryResponse)
async def ask_tutor_endpoint(
    request_data: AiTutorQueryRequest,
    # llm_client: MockLLMClient = Depends(lambda: mock_llm_client) # Example of dependency injection
):
    """
    Handles a user's query, constructs a personalized prompt,
    gets a response from the LLM, and formats the output.
    """
    start_time = timezone.now() # For processing time calculation

    # 1. Fetch relevant course content (RAG part) - MOCKED
    # In a real system, this might involve semantic search over course materials based on query_text
    # or directly fetching content if topic_id is precise.
    course_material_context: Optional[str] = None
    if request_data.context and request_data.context.topic_id:
        course_material_context = await fetch_course_content_from_django(request_data.context.topic_id)
    elif request_data.context and request_data.context.course_id: # Less specific
        course_material_context = f"General context for course ID {request_data.context.course_id} might be fetched here."


    # 2. Construct the prompt for the LLM
    prompt = construct_llm_prompt(
        query=request_data.query_text,
        user_profile=request_data.user_profile_snapshot,
        course_content=course_material_context,
        context_data=request_data.context,
        conversation_history=request_data.conversation_history
    )

    # 3. Get response from the (Mocked) LLM client
    try:
        llm_response_data = await mock_llm_client.generate_response(
            prompt=prompt,
            max_tokens=request_data.max_tokens_response,
            user_profile=request_data.user_profile_snapshot,
            context=request_data.context # Pass full context for mock logic
        )
        answer_text = llm_response_data.get("answer_text", "I am unable to answer this at the moment.")
        raw_analogies = llm_response_data.get("generated_analogies", [])
        
        # Convert raw analogies to Pydantic models
        parsed_analogies = []
        if isinstance(raw_analogies, list):
            for analogy_item in raw_analogies:
                if isinstance(analogy_item, dict) and "analogy" in analogy_item:
                    parsed_analogies.append(GeneratedAnalogy(**analogy_item))
                elif isinstance(analogy_item, str): # If LLM just returns a string list
                     parsed_analogies.append(GeneratedAnalogy(analogy=analogy_item))


    except Exception as e:
        # Log the exception properly in a real application
        print(f"Error calling LLM or processing its response: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with the AI model.")

    end_time = timezone.now()
    processing_time_ms = (end_time - start_time).total_seconds() * 1000

    # 4. Prepare and return the final response
    # (Could add logic here to extract follow-up questions or confidence from LLM output if designed in prompt)
    
    # Simple follow-up suggestions based on keywords for mock
    follow_ups = []
    if "python lists" in request_data.query_text.lower():
        follow_ups = ["How do I add items to a list?", "Tell me about list comprehensions."]
    elif "decorator" in request_data.query_text.lower():
        follow_ups = ["Can you show an example of a decorator?", "What are common use cases for decorators?"]


    return AiTutorQueryResponse(
        answer_text=answer_text,
        suggested_follow_up_questions=follow_ups,
        generated_analogies=parsed_analogies,
        confidence_score=0.85, # Mocked confidence
        debug_info=DebugInfo(
            prompt_sent_to_llm=prompt if os.getenv("DEBUG_MODE") else None, # Only send full prompt in debug
            llm_model_name_used=mock_llm_client.model_name,
            processing_time_ms=round(processing_time_ms, 2)
        )
    )

# To run this FastAPI app locally (from within uplas-ai-agents/personalized_tutor_nlp_llm/ directory):
# uvicorn main:app --reload
