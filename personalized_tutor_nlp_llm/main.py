from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import httpx # For making API calls to Django backend & Vertex AI
import os
from google.cloud import aiplatform # Vertex AI SDK

# Initialize Vertex AI (example, actual initialization depends on environment)
# PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
# LOCATION = os.environ.get("GCP_REGION", "us-central1")
# aiplatform.init(project=PROJECT_ID, location=LOCATION)
# # Assuming a fine-tuned model endpoint on Vertex AI
# LLM_ENDPOINT_ID = os.environ.get("LLM_TUTOR_ENDPOINT_ID") 
# llm_endpoint = aiplatform.Endpoint(LLM_ENDPOINT_ID)

# UPLAS_BACKEND_API_URL = os.environ.get("UPLAS_BACKEND_API_URL") # e.g., https://api.uplas.me/api
# UPLAS_INTERNAL_API_KEY = os.environ.get("UPLAS_INTERNAL_API_KEY") # For service-to-service auth

app = FastAPI()

class TutorQuery(BaseModel):
    user_id: int
    query: str
    context_course_id: str | None = None
    context_topic_id: str | None = None
    # context_project_id from frontend guide 
    context_project_id: str | None = None


class UserProfileData(BaseModel): # Simplified for example
    career_interest: str | None = None
    industry: str | None = None
    profession: str | None = None
    country: str | None = None
    city: str | None = None
    preferred_tutor_persona: str | None = "Friendly and encouraging"
    # ... other fields from UserProfile ...

async def get_user_profile_from_backend(user_id: int) -> UserProfileData | None:
    # This function would make an authenticated API call to the Django backend
    # headers = {"Authorization": f"InternalAPIKey {UPLAS_INTERNAL_API_KEY}"}
    # async with httpx.AsyncClient() as client:
    #     response = await client.get(f"{UPLAS_BACKEND_API_URL}/internal/users/{user_id}/profile/", headers=headers)
    #     if response.status_code == 200:
    #         return UserProfileData(**response.json())
    # return None
    print(f"SIMULATED: Fetching profile for user {user_id}")
    # Simulate fetching data
    if user_id == 1: # Example user
         return UserProfileData(career_interest="AI Specialist", industry="Technology", profession="Software Engineer", country="Kenya", city="Nairobi")
    return UserProfileData() # Default empty profile

def generate_personalized_prompt(user_query: str, user_profile: UserProfileData, course_context: str = "") -> str:
    prompt = f"You are Uplas AI Tutor, a helpful and engaging AI learning assistant. Your persona should be: {user_profile.preferred_tutor_persona}.\n\n"
    prompt += f"User Profile Context:\n- Career Interest: {user_profile.career_interest or 'Not specified'}\n"
    prompt += f"- Industry: {user_profile.industry or 'Not specified'}\n"
    prompt += f"- Profession: {user_profile.profession or 'Not specified'}\n"
    prompt += f"- Location: {user_profile.city or ''}, {user_profile.country or 'Not specified'}\n\n"

    if course_context:
        prompt += f"Relevant Course Context:\n{course_context}\n\n"

    prompt += f"User's Question: \"{user_query}\"\n\n"
    prompt += "Please answer the user's question. Personalize your explanation using analogies and examples that would resonate with someone in their specified industry, profession, and location. If details are not specified, use general but engaging examples. Ensure your answer is clear, concise, and directly addresses the question. If the question is outside the scope of AI or education, politely decline to answer.\n\nAnswer:"
    return prompt

@app.post("/ask")
async def ask_tutor(query: TutorQuery):
    print(f"AI Tutor Agent received query: {query.query} for user {query.user_id}")
    user_profile = await get_user_profile_from_backend(query.user_id)
    if not user_profile:
        # Fallback if profile can't be fetched
        user_profile = UserProfileData()

    # TODO: Fetch relevant course_context based on query.context_course_id and query.context_topic_id
    # from Django backend or a dedicated content store.
    course_context_text = f"This question is related to course {query.context_course_id}, topic {query.context_topic_id}."
    if query.context_project_id: # Add project context if available 
         course_context_text += f" It is also related to project {query.context_project_id}."


    personalized_prompt = generate_personalized_prompt(query.query, user_profile, course_context_text)
    print(f"\n--- Generated Prompt for LLM ---\n{personalized_prompt}\n-------------------------------\n")

    try:
        # Simulate LLM call
        # response = llm_endpoint.predict(instances=[{"prompt": personalized_prompt}])
        # ai_answer = response.predictions[0]['content']

        # Simulated response for now
        await httpx.AsyncClient().get("https://jsonplaceholder.typicode.com/todos/1") # Simulate network delay
        simulated_llm_response = (
            f"Okay, as a {user_profile.profession or 'learner'} in {user_profile.industry or 'your field'} "
            f"from {user_profile.city or user_profile.country or 'your location'}, "
            f"let's talk about '{query.query[:30]}...'. "
            f"Imagine it's like [relevant analogy for {user_profile.industry}]. For example, in {user_profile.city}, "
            f"you might see [example relevant to {user_profile.city} and {user_profile.profession}]. "
            f"This should help you understand the concept better! (This is a personalized, simulated response from the AI Tutor based on your data.)"
        )
        if "thank you" in query.query.lower():
            simulated_llm_response = "You're very welcome! I'm glad I could help. Do you have any other questions?"

        ai_answer = simulated_llm_response

        return {"answer": ai_answer}
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise HTTPException(status_code=500, detail="Error processing your request with the AI model.")

# This FastAPI app would then be containerized (Dockerfile) and deployed to Cloud Run.
# The Django backend's /api/ai-tutor/ask/ view would make an HTTP request to this Cloud Run service.
