from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import os
import uuid
import time # For processing time
from enum import Enum
# from unittest.mock import MagicMock # We'll create a direct mock for now

# --- Pydantic Models for API Contract ---

class UserProfileSnapshotForProjects(BaseModel):
    # Fields from User & UserProfile relevant for project suggestions
    user_id: str # To link suggestions back if needed, or for logging
    industry: Optional[str] = Field(None, examples=["Finance", "Healthcare", "Creative Arts"])
    profession: Optional[str] = Field(None, examples=["Data Analyst", "UX Designer", "Student"])
    career_interest: Optional[str] = Field(None, examples=["Machine Learning Engineer", "Game Developer"])
    # Example: {"Python": "Intermediate", "Data Analysis": "Beginner", "Web Development": "Novice"}
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict) 
    areas_of_interest: Optional[List[str]] = Field(default_factory=list, examples=[["NLP", "Ethical AI"], ["Sustainable Tech"]])
    learning_goals: Optional[str] = Field(None, examples=["Build a portfolio piece for data visualization.", "Understand full-stack development."])
    # Potentially add completed_course_ids or topics to inform suggestions
    # completed_course_ids: Optional[List[str]] = Field(default_factory=list)


class ProjectPreferences(BaseModel):
    difficulty_level: Optional[str] = Field("intermediate", examples=["beginner", "intermediate", "advanced"]) # Align with Project.DIFFICULTY_CHOICES
    # Example: ["Python", "JavaScript", "FastAPI"] or "Any"
    preferred_technologies: Optional[List[str]] = Field(default_factory=list) 
    # Example: "Data Analysis", "Web Application", "AI Model"
    project_type_focus: Optional[str] = Field(None, examples=["Portfolio Piece", "Skill Development", "Exploratory"]) 
    time_commitment_hours_estimate: Optional[int] = Field(None, examples=[20, 40], ge=5) # Estimated hours user can commit


class ProjectIdeaGenerationRequest(BaseModel):
    user_profile_snapshot: UserProfileSnapshotForProjects
    preferences: Optional[ProjectPreferences] = Field(default_factory=ProjectPreferences)
    # Number of project ideas to generate
    number_of_ideas: Optional[int] = Field(1, ge=1, le=5) 


class GeneratedProjectTask(BaseModel):
    task_id: int
    description: str
    estimated_sub_duration: Optional[str] = None # e.g., "2-3 hours", "1 day"


class GeneratedProjectIdea(BaseModel): # This structure should align with Project.ai_generated_spec_json
    # Also matches conceptual Project model fields from Uplas Backend Integration Guide 
    request_id: str # For tracking, matches an ID from the request if provided, or generated
    project_idea_id: str = Field(default_factory=lambda: f"idea_{uuid.uuid4().hex[:8]}")
    
    title: str = Field(..., examples=["Ethical AI Chatbot Analyzer"])
    subtitle: Optional[str] = Field(None, examples=["Analyze chatbot conversations for potential biases."])
    description_html: str = Field(..., examples=["<p>This project involves collecting a dataset of chatbot interactions...</p>"])
    difficulty_level: str = Field(..., examples=["intermediate"]) # beginner, intermediate, advanced
    estimated_duration: Optional[str] = Field(None, examples=["15-20 hours"]) # e.g., "15 hours", "2-3 weeks"
    
    learning_objectives_html: List[str] = Field(default_factory=list, examples=[["Understand bias detection techniques."], ["Apply NLP for text analysis."]])
    requirements_html: Optional[List[str]] = Field(default_factory=list, examples=[["Basic Python knowledge."], ["Familiarity with REST APIs."]])
    target_audience_html: Optional[List[str]] = Field(default_factory=list, examples=[["Students learning AI ethics."], ["Developers building chatbots."]])
    
    # key_tasks_html or a structured list of tasks
    key_tasks: List[GeneratedProjectTask] = Field(default_factory=list)
    suggested_technologies: List[str] = Field(default_factory=list, examples=[["Python", "Pandas", "NLTK", "FastAPI"]])
    
    # Optional fields from AI
    personalization_rationale: Optional[str] = None # Why this project is good for *this* user
    potential_challenges: Optional[List[str]] = Field(default_factory=list)
    # A very basic rubric preview, more detailed rubric would be for assessment phase
    assessment_rubric_preview_html: Optional[List[str]] = Field(default_factory=list, examples=[["Completion of all key tasks (40%)"], ["Code quality and clarity (30%)"]])
    real_world_application_examples: Optional[List[str]] = Field(default_factory=list)


class ProjectIdeaGenerationResponse(BaseModel):
    generated_ideas: List[GeneratedProjectIdea]
    debug_info: Optional[Dict[str, Any]] = None


# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Project Idea Generator",
    description="Generates personalized real-world project ideas using a (mocked) LLM.",
    version="0.1.0"
)

# --- Mock LLM Client for Project Idea Generation ---
MOCKED_PROJECT_LLM_NAME = "mocked-gemini-pro-project-gen"

class MockProjectLLMClient:
    async def generate_ideas(
        self,
        user_profile: UserProfileSnapshotForProjects,
        preferences: ProjectPreferences,
        num_ideas: int
    ) -> List[Dict[str, Any]]: # Returns a list of dicts, each conforming to GeneratedProjectIdea structure
        """Simulates LLM generating project ideas."""
        print(f"\n--- Mock Project LLM ({MOCKED_PROJECT_LLM_NAME}) Received Request ---")
        print(f"User Profile: Industry - {user_profile.industry}, Profession - {user_profile.profession}")
        print(f"Preferences: Difficulty - {preferences.difficulty_level}, Tech - {preferences.preferred_technologies}")
        print(f"Number of Ideas Requested: {num_ideas}")
        print("---------------------------------------------------------------------\n")

        generated_project_ideas = []
        for i in range(num_ideas):
            idea_id_suffix = uuid.uuid4().hex[:4]
            difficulty = preferences.difficulty_level or random.choice(["beginner", "intermediate", "advanced"])
            
            # Basic personalization in mock based on industry/profession
            project_title_prefix = "Personalized Project"
            if user_profile.industry:
                project_title_prefix = f"{user_profile.industry}-Focused Project"
            if user_profile.profession: # Profession might be more specific
                project_title_prefix = f"Project for a {user_profile.profession}"

            tech_stack = preferences.preferred_technologies or ["Python", "Flask/FastAPI", "Basic HTML/CSS"]
            if "Data" in project_title_prefix or (user_profile.areas_of_interest and any("data" in interest.lower() for interest in user_profile.areas_of_interest)):
                tech_stack.extend(["Pandas", "Matplotlib/Seaborn"])
                tech_stack = list(set(tech_stack)) # Remove duplicates

            idea = {
                "request_id": f"req_{uuid.uuid4().hex[:6]}", # Simulate a request ID if we had one
                "project_idea_id": f"idea_{idea_id_suffix}_{i+1}",
                "title": f"{project_title_prefix}: Interactive Dashboard {idea_id_suffix}",
                "subtitle": f"A {difficulty} project to build an interactive dashboard using {tech_stack[0]}.",
                "description_html": f"<p>Develop an engaging dashboard for '{user_profile.industry or 'a selected domain'}' data. This project will enhance your skills in {', '.join(tech_stack)}.</p>",
                "difficulty_level": difficulty,
                "estimated_duration": f"{random.randint(10, 30)} hours",
                "learning_objectives_html": [
                    f"Master {tech_stack[0]} for backend logic.",
                    "Learn data visualization techniques.",
                    "Understand user interaction for dashboards."
                ],
                "requirements_html": [f"Basic understanding of {tech_stack[0]}.", "Access to a code editor."],
                "key_tasks": [
                    {"task_id": 1, "description": "Setup project environment and install dependencies."},
                    {"task_id": 2, "description": "Design the dashboard layout and components."},
                    {"task_id": 3, "description": "Implement backend API endpoints (if needed)."},
                    {"task_id": 4, "description": "Develop frontend components and interactivity."},
                    {"task_id": 5, "description": "Test and deploy the dashboard (simulation)."}
                ],
                "suggested_technologies": tech_stack,
                "personalization_rationale": f"Tailored for your interest in {user_profile.areas_of_interest[0] if user_profile.areas_of_interest else 'practical applications'} and experience as a {user_profile.profession or 'learner'}.",
                "assessment_rubric_preview_html": ["Functionality (50%)", "Code Quality (30%)", "UI/UX (20%)"]
            }
            generated_project_ideas.append(idea)
        
        # await asyncio.sleep(random.uniform(0.5, 2.0)) # Simulate LLM processing time
        return generated_project_ideas

mock_project_llm = MockProjectLLMClient()

# --- Helper for Prompt Construction (Conceptual) ---
def construct_project_gen_prompt(
    user_profile: UserProfileSnapshotForProjects,
    preferences: ProjectPreferences,
    num_ideas: int
) -> str:
    """
    Constructs a prompt for an LLM to generate project ideas.
    This is more conceptual for this agent, as the mock LLM doesn't use a detailed text prompt.
    In a real scenario, this would be very detailed.
    """
    prompt = f"Generate {num_ideas} personalized real-world project idea(s) for a user with the following profile:\n"
    prompt += f"- User ID (for reference): {user_profile.user_id}\n"
    if user_profile.industry: prompt += f"- Industry: {user_profile.industry}\n"
    if user_profile.profession: prompt += f"- Profession/Role: {user_profile.profession}\n"
    if user_profile.career_interest: prompt += f"- Career Interests: {user_profile.career_interest}\n"
    if user_profile.current_knowledge_level:
        knowledge_str = ", ".join([f"{k}: {v}" for k,v in user_profile.current_knowledge_level.items()])
        prompt += f"- Current Knowledge: {knowledge_str}\n"
    if user_profile.areas_of_interest:
        prompt += f"- Areas of Interest: {', '.join(user_profile.areas_of_interest)}\n"
    if user_profile.learning_goals:
        prompt += f"- Learning Goals: {user_profile.learning_goals}\n"

    prompt += "\nUser's Project Preferences:\n"
    if preferences.difficulty_level: prompt += f"- Difficulty: {preferences.difficulty_level}\n"
    if preferences.preferred_technologies:
        prompt += f"- Preferred Technologies: {', '.join(preferences.preferred_technologies)} (or suggest suitable alternatives)\n"
    if preferences.project_type_focus: prompt += f"- Project Focus: {preferences.project_type_focus}\n"
    if preferences.time_commitment_hours_estimate:
        prompt += f"- Estimated Time Commitment by User: {preferences.time_commitment_hours_estimate} hours\n"

    prompt += "\nFor each project idea, provide the following in a structured format (e.g., JSON-like):\n"
    prompt += "- title (string, catchy and descriptive)\n"
    prompt += "- subtitle (string, short summary)\n"
    prompt += "- description_html (string, detailed, can include HTML for paragraphs)\n"
    prompt += "- difficulty_level (string: 'beginner', 'intermediate', or 'advanced')\n"
    prompt += "- estimated_duration (string, e.g., '10-15 hours', '2 weeks')\n"
    prompt += "- learning_objectives_html (list of strings, what the user will learn)\n"
    prompt += "- requirements_html (list of strings, prerequisites)\n"
    prompt += "- key_tasks (list of objects, each with 'task_id' and 'description')\n"
    prompt += "- suggested_technologies (list of strings)\n"
    prompt += "- personalization_rationale (string, why this project is good for *this* user based on their profile)\n"
    prompt += "- assessment_rubric_preview_html (list of strings, high-level criteria for success)\n"
    prompt += "\nEnsure the projects are practical, engaging, and can be realistically tackled. Focus on real-world applicability."
    return prompt

# --- API Endpoint ---

@app.post("/v1/generate-project-ideas", response_model=ProjectIdeaGenerationResponse, summary="Generate Personalized Project Ideas")
async def generate_project_ideas_endpoint(request_data: ProjectIdeaGenerationRequest):
    processing_start_time = time.perf_counter()

    # In a real app, you might construct a detailed text prompt here using construct_project_gen_prompt
    # For the mock, we pass structured data directly to the mock client.
    # conceptual_prompt = construct_project_gen_prompt(
    #     user_profile=request_data.user_profile_snapshot,
    #     preferences=request_data.preferences,
    #     num_ideas=request_data.number_of_ideas
    # )
    # print(f"Conceptual LLM Prompt for Project Gen:\n{conceptual_prompt[:500]}...\n")


    try:
        raw_ideas_from_llm = await mock_project_llm.generate_ideas(
            user_profile=request_data.user_profile_snapshot,
            preferences=request_data.preferences,
            num_ideas=request_data.number_of_ideas
        )
        
        # Validate and convert raw ideas to Pydantic models
        validated_ideas: List[GeneratedProjectIdea] = []
        for raw_idea in raw_ideas_from_llm:
            try:
                # Ensure all required fields for GeneratedProjectIdea are present in raw_idea
                # Pydantic will raise validation error if not.
                validated_ideas.append(GeneratedProjectIdea(**raw_idea))
            except Exception as e_val: # Catch Pydantic validation error for one idea
                print(f"Warning: Failed to validate a raw project idea from LLM mock: {raw_idea}. Error: {e_val}")
                # Optionally, skip this idea or try to fix it

        if not validated_ideas and raw_ideas_from_llm: # If all failed validation but some were returned
             raise HTTPException(status_code=500, detail="AI service returned ideas in an unexpected format.")
        if not validated_ideas: # If LLM returned nothing or all failed validation
             raise HTTPException(status_code=404, detail="Could not generate suitable project ideas based on the provided profile and preferences.")


    except Exception as e:
        print(f"Error during project idea generation or LLM call: {e}")
        raise HTTPException(status_code=503, detail="Error communicating with the AI project generation service.")

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    debug_info = {
        "llm_model_name_used": MOCKED_PROJECT_LLM_NAME,
        "processing_time_ms": round(processing_time_ms, 2),
        "num_ideas_requested": request_data.number_of_ideas,
        "num_ideas_generated_valid": len(validated_ideas)
    }
    if os.getenv("UPLAS_DEBUG_MODE", "false").lower() == "true":
        # In debug mode, you might include the conceptual_prompt or other details
        # For now, conceptual_prompt is not directly used by the mock, so not adding.
        pass

    return ProjectIdeaGenerationResponse(
        generated_ideas=validated_ideas,
        debug_info=debug_info
    )

# To run this FastAPI app locally (from within uplas-ai-agents/project_generator_agent/ directory):
# uvicorn main:app --reload --port 8004
