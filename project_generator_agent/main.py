from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
import os
import uuid
import time # For processing time
from enum import Enum
import random # For mock elements

# --- Pydantic Models for API Contract ---

class UserProfileSnapshotForProjects(BaseModel):
    user_id: str 
    industry: Optional[str] = Field(None, examples=["Finance", "Healthcare", "Creative Arts"])
    profession: Optional[str] = Field(None, examples=["Data Analyst", "UX Designer", "Student"])
    career_interest: Optional[str] = Field(None, examples=["Machine Learning Engineer", "Game Developer"])
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "Data Analysis": "Beginner"}]) 
    areas_of_interest: Optional[List[str]] = Field(default_factory=list, examples=[["NLP", "Ethical AI"], ["Sustainable Tech"]])
    learning_goals: Optional[str] = Field(None, examples=["Build a portfolio piece for data visualization."])

class ProjectPreferences(BaseModel):
    difficulty_level: Optional[str] = Field("intermediate", examples=["beginner", "intermediate", "advanced"])
    preferred_technologies: Optional[List[str]] = Field(default_factory=list, examples=[["Python", "JavaScript", "FastAPI"]]) 
    project_type_focus: Optional[str] = Field(None, examples=["Portfolio Piece", "Skill Development"]) 
    time_commitment_hours_estimate: Optional[int] = Field(None, examples=[20, 40], ge=5)

class ProjectIdeaGenerationRequest(BaseModel):
    user_profile_snapshot: UserProfileSnapshotForProjects
    preferences: Optional[ProjectPreferences] = Field(default_factory=ProjectPreferences)
    number_of_ideas: Optional[int] = Field(1, ge=1, le=3) # Limit to 3 ideas for mock diversity

class GeneratedProjectTask(BaseModel):
    task_id: int
    description: str
    estimated_sub_duration: Optional[str] = None

class GeneratedProjectIdea(BaseModel):
    request_id: Optional[str] = None # Optional, can be added by the agent
    project_idea_id: str = Field(default_factory=lambda: f"idea_{uuid.uuid4().hex[:8]}")
    title: str = Field(..., examples=["Ethical AI Chatbot Analyzer"])
    subtitle: Optional[str] = Field(None, examples=["Analyze chatbot conversations for potential biases."])
    description_html: str = Field(..., examples=["<p>This project involves collecting a dataset...</p>"])
    difficulty_level: str = Field(..., examples=["intermediate"])
    estimated_duration: Optional[str] = Field(None, examples=["15-20 hours"])
    learning_objectives_html: List[str] = Field(default_factory=list, examples=[["Understand bias detection techniques."]])
    requirements_html: Optional[List[str]] = Field(default_factory=list, examples=[["Basic Python knowledge."]])
    target_audience_html: Optional[List[str]] = Field(default_factory=list, examples=[["Students learning AI ethics."]])
    key_tasks: List[GeneratedProjectTask] = Field(default_factory=list)
    suggested_technologies: List[str] = Field(default_factory=list, examples=[["Python", "Pandas"]])
    personalization_rationale: Optional[str] = None
    potential_challenges: Optional[List[str]] = Field(default_factory=list)
    assessment_rubric_preview_html: Optional[List[str]] = Field(default_factory=list, examples=[["Functionality (50%)"]])
    real_world_application_examples: Optional[List[str]] = Field(default_factory=list)

class ProjectIdeaGenerationResponse(BaseModel):
    generated_ideas: List[GeneratedProjectIdea]
    debug_info: Optional[Dict[str, Any]] = None

# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Project Idea Generator",
    description="Generates personalized real-world project ideas using a (mocked) LLM.",
    version="0.1.1" # Incremented version
)

# --- Mock LLM Client for Project Idea Generation ---
MOCKED_PROJECT_LLM_NAME = "mocked-gemini-pro-project-gen-v2"

class MockProjectLLMClient:
    async def generate_ideas(
        self,
        user_profile: UserProfileSnapshotForProjects,
        preferences: ProjectPreferences,
        num_ideas: int
    ) -> List[Dict[str, Any]]:
        print(f"\n--- Mock Project LLM ({MOCKED_PROJECT_LLM_NAME}) Refining Ideas ---")
        print(f"User Profile: Industry='{user_profile.industry}', Profession='{user_profile.profession}', Interests='{user_profile.areas_of_interest}', Knowledge='{user_profile.current_knowledge_level}'")
        print(f"Preferences: Difficulty='{preferences.difficulty_level}', Tech='{preferences.preferred_technologies}', Focus='{preferences.project_type_focus}'")
        print(f"Number of Ideas Requested: {num_ideas}")
        print("---------------------------------------------------------------------\n")

        generated_project_ideas = []
        base_titles_and_domains = [
            ("Interactive Data Visualization Dashboard", "Data Analytics"), ("Real-Time Stock Monitoring App", "FinTech"),
            ("E-commerce Product Recommendation System", "E-commerce/AI"), ("Personalized News Aggregator", "Web/Content"),
            ("AI-Powered Symptom Checker Mockup", "HealthTech/AI"), ("Sustainable Recipe Planner", "Lifestyle/Web"),
            ("Local Event Discovery Platform", "Social/Web"), ("Fitness Challenge Tracker", "Health/Mobile"),
            ("Educational Quiz Game Builder", "EdTech/GameDev"), ("Community Skill-Share Network API", "Social/API")
        ]
        
        random.shuffle(base_titles_and_domains)

        for i in range(min(num_ideas, len(base_titles_and_domains))):
            idea_id_suffix = uuid.uuid4().hex[:4]
            base_title, domain = base_titles_and_domains[i]
            difficulty = preferences.difficulty_level or random.choice(["beginner", "intermediate", "advanced"])
            
            project_title = f"{base_title}"
            personalization_notes_list = []

            # Personalize title and description based on profile
            current_industry = user_profile.industry or domain # Fallback to domain if industry not specified
            if current_industry:
                project_title = f"{current_industry}-Focused: {base_title}"
                personalization_notes_list.append(f"Tailored for the {current_industry} sector.")
            
            if user_profile.profession:
                project_title = f"{base_title} for a {user_profile.profession}" # Profession often more specific
                personalization_notes_list.append(f"Designed with a {user_profile.profession}'s typical skillset or learning path in mind.")
            
            user_interests = user_profile.areas_of_interest or []
            if user_interests:
                interest = random.choice(user_interests)
                project_title += f" (Exploring {interest.title()})"
                personalization_notes_list.append(f"Allows exploration of your interest in {interest}.")
            else:
                personalization_notes_list.append("A generally useful project to build foundational skills.")


            tech_stack = ["Python"] # Default
            if preferences.preferred_technologies:
                tech_stack.extend(preferences.preferred_technologies)
            else: # Suggest based on domain/title if no preference
                if "Web" in domain or "Dashboard" in domain or "Platform" in domain or "Aggregator" in domain:
                    tech_stack.extend(random.choice([["FastAPI", "React"], ["Flask", "Vue.js"], ["Django", "HTML/CSS/JS"]]))
                if "Data" in domain or "AI" in domain or "System" in domain or "Monitoring" in domain:
                    tech_stack.extend(["Pandas", random.choice(["Scikit-learn", "TensorFlow", "PyTorch", "NLTK"])])
            
            tech_stack = sorted(list(set(tech_stack))) # Unique and sorted

            description = f"<p>This is a '{difficulty}' level project designed to help you build an innovative '{project_title}'. "
            description += f"You will get hands-on experience with technologies such as {', '.join(tech_stack)}. "
            if user_profile.learning_goals:
                description += f"This project directly addresses your learning goal: '{user_profile.learning_goals[:70]}...'.</p>"
            else:
                description += "It offers a fantastic opportunity to create a practical application and significantly expand your skillset.</p>"

            estimated_hours = preferences.time_commitment_hours_estimate or random.randint(15, 50)

            idea = {
                "request_id": f"req_{uuid.uuid4().hex[:6]}",
                "project_idea_id": f"idea_{idea_id_suffix}_{i+1}",
                "title": project_title,
                "subtitle": f"A {difficulty} project focusing on {domain}, utilizing {tech_stack[0] if tech_stack else 'key technologies'}.",
                "description_html": description,
                "difficulty_level": difficulty,
                "estimated_duration": f"Approx. {estimated_hours} - {estimated_hours + random.randint(5,15)} hours",
                "learning_objectives_html": [
                    f"<li>Master practical application of {', '.join(tech_stack)}.</li>",
                    f"<li>Develop a tangible project for your portfolio focusing on {preferences.project_type_focus or 'real-world problem solving'}.</li>",
                    f"<li>Enhance your skills in {random.choice(['API design', 'data manipulation', 'frontend interaction', 'system architecture'])}.</li>"
                ],
                "requirements_html": [
                    f"<li>{user_profile.current_knowledge_level.get(tech_stack[0].split('/')[0].strip(), 'Basic understanding of') if tech_stack else 'Fundamental programming concepts'}.</li>", # Check knowledge for primary tech
                    "<li>Access to a computer with internet and a code editor.</li>",
                    "<li>Proactive learning attitude and problem-solving mindset.</li>"
                ],
                "key_tasks": [
                    {"task_id": 1, "description": "Project Planning & Setup: Define scope, choose tools, initialize project (Git, virtual environment).", "estimated_sub_duration": "2-3 hours"},
                    {"task_id": 2, "description": f"Core Feature Development: Implement the primary logic using {tech_stack[0] if tech_stack else 'chosen technologies'}."},
                    {"task_id": 3, "description": "Data Integration (if applicable): Source, clean, and manage necessary data."},
                    {"task_id": 4, "description": "User Interface/Interaction (if applicable): Design and build basic UI or API endpoints."},
                    {"task_id": 5, "description": "Testing & Documentation: Write basic tests, document your code and project structure."}
                ],
                "suggested_technologies": tech_stack,
                "personalization_rationale": " ".join(personalization_notes_list),
                "assessment_rubric_preview_html": [
                    "<li>Core Functionality & Task Completion (50%)</li>", 
                    f"<li>Code Quality & Use of {tech_stack[0] if tech_stack else ''} Best Practices (30%)</li>",
                    "<li>Project Structure, Documentation & (Optional) Presentation (20%)</li>"
                ],
                "real_world_application_examples": [
                    f"This project's concepts are applicable in {current_industry} for tasks like [example task].",
                    "Could serve as a foundational component for a larger, more complex system."
                ]
            }
            generated_project_ideas.append(idea)
        
        # await asyncio.sleep(random.uniform(0.3, 1.2)) # Simulate LLM processing time
        return generated_project_ideas

mock_project_llm = MockProjectLLMClient()

# --- Helper for Prompt Construction (Conceptual for Real LLM) ---
def construct_project_gen_prompt(
    user_profile: UserProfileSnapshotForProjects,
    preferences: ProjectPreferences,
    num_ideas: int
) -> str:
    # System Preamble
    prompt = (
        "You are an AI assistant specialized in generating personalized, real-world project ideas "
        "for users looking to enhance their skills and build their portfolios. For each idea, you must provide "
        "all specified fields in a structured JSON format. Ensure the projects are engaging, practical, "
        "and offer clear learning objectives. Personalize heavily based on user's industry, profession, interests, and knowledge.\n\n"
    )

    # User Profile Section
    prompt += "USER PROFILE:\n"
    prompt += f"- User ID (for reference only): {user_profile.user_id}\n"
    if user_profile.industry: prompt += f"- Current or Target Industry: {user_profile.industry}\n"
    if user_profile.profession: prompt += f"- Current or Target Profession/Role: {user_profile.profession}\n"
    if user_profile.career_interest: prompt += f"- Stated Career Interests: {user_profile.career_interest}\n"
    if user_profile.current_knowledge_level:
        knowledge_str = ", ".join([f"{k}: {v}" for k, v in user_profile.current_knowledge_level.items()])
        if knowledge_str: prompt += f"- Self-Assessed Knowledge Levels: {knowledge_str}\n"
    if user_profile.areas_of_interest:
        prompt += f"- Specific Areas of Interest: {', '.join(user_profile.areas_of_interest)}\n"
    if user_profile.learning_goals:
        prompt += f"- Stated Learning Goals: {user_profile.learning_goals}\n"

    # Project Preferences Section
    prompt += "\nUSER'S PROJECT PREFERENCES:\n"
    if preferences.difficulty_level: prompt += f"- Desired Difficulty Level: {preferences.difficulty_level} (Suggest projects matching this, or state if not possible and suggest alternatives).\n"
    if preferences.preferred_technologies:
        prompt += f"- Preferred Technologies: {', '.join(preferences.preferred_technologies)}. If not suitable for the idea, suggest alternatives but try to incorporate if possible.\n"
    else:
        prompt += "- Preferred Technologies: User is open to suggestions; choose technologies appropriate for the project and user's profile (especially considering their current knowledge).\n"
    if preferences.project_type_focus: prompt += f"- Desired Project Focus: {preferences.project_type_focus}\n"
    if preferences.time_commitment_hours_estimate:
        prompt += f"- User's Estimated Time Commitment: Around {preferences.time_commitment_hours_estimate} hours. Tailor project scope accordingly.\n"

    # Request for Ideas and Output Structure
    prompt += f"\nTASK: Generate {num_ideas} distinct project idea(s) based on the user profile and preferences above.\n"
    prompt += "For EACH project idea, provide the output as a SINGLE, VALID JSON object with the following EXACT fields and value types (use HTML for list items in appropriate fields):\n"
    prompt += """
{
  "project_idea_id": "string (a unique identifier you generate for this idea, e.g., idea_xxxx)",
  "title": "string (catchy, descriptive, and highly personalized project title incorporating user's industry/profession/interests)",
  "subtitle": "string (a brief, engaging one-sentence summary of the project, highlighting personalization)",
  "description_html": "string (detailed project description, 2-4 paragraphs, using <p> and <ul> for formatting. Explain what the project is about and its value *to this specific user*.)",
  "difficulty_level": "string (choose one: 'beginner', 'intermediate', 'advanced', matching user preference or justifying if different, considering their stated knowledge)",
  "estimated_duration": "string (e.g., '15-20 hours', 'Approx. 3 weeks part-time', aligning with user's time commitment if provided)",
  "learning_objectives_html": ["string (e.g., '<li>Master asynchronous programming in Python, relevant for your interest in building scalable web services.</li>')", "string (e.g., '<li>Learn to integrate third-party APIs effectively, a key skill for a {user_profile.profession or 'developer'} in the {user_profile.industry or 'tech'} field.</li>')"],
  "requirements_html": ["string (e.g., '<li>{user_profile.current_knowledge_level.get("Python", "Basic Python proficiency")} (variables, loops, functions).</li>')", "string (e.g., '<li>Familiarity with JSON data structures, useful for your interest in {user_profile.areas_of_interest[0] if user_profile.areas_of_interest else 'APIs'}.</li>')"],
  "target_audience_html": ["string (e.g., '<li>A {user_profile.profession or 'learner'} aiming to strengthen their {user_profile.areas_of_interest[0] if user_profile.areas_of_interest else 'skills'} in the {user_profile.industry or 'chosen'} domain.</li>')", "string (e.g., '<li>Individuals looking to build a portfolio piece demonstrating [specific skill].</li>')"],
  "key_tasks": [
    {"task_id": 1, "description": "string (Clear, actionable step 1, e.g., 'Project Planning & Setup: Define detailed scope based on your learning goals, choose specific libraries from suggested tech, initialize project (Git, virtual environment).')", "estimated_sub_duration": "string (e.g., '1-2 hours')"},
    {"task_id": 2, "description": "string (Clear, actionable step 2, e.g., 'Core Logic for [Main Feature]: Implement the primary functionality using {preferences.preferred_technologies[0] if preferences.preferred_technologies else 'Python'}.')", "estimated_sub_duration": "string (e.g., '5-8 hours')"}
    // ... add 3-5 detailed, actionable tasks in total
  ],
  "suggested_technologies": ["string (e.g., 'Python 3.9+')", "string (e.g., 'FastAPI 0.100+')" /* , ... more specific versions if applicable */ ],
  "personalization_rationale": "string (Crucial: Explain in 1-2 sentences EXACTLY WHY this project and these technologies are a good fit for *this specific user*, referencing their profile: industry, profession, current knowledge, interests, and learning goals. Be very specific in the connection.)",
  "potential_challenges": ["string (e.g., 'Debugging asynchronous code if using FastAPI/asyncio for the first time.')", "string (e.g., 'Finding a suitable, free dataset for {domain related to project}.')"],
  "assessment_rubric_preview_html": ["string (e.g., '<li>Successful implementation of all key tasks and core functionality (50%)</li>')", "string (e.g., '<li>Code quality, organization, and adherence to best practices for suggested technologies (30%)</li>')", "string (e.g., '<li>Clear documentation and (if applicable) a brief video presentation of the project (20%)</li>')"],
  "real_world_application_examples": ["string (e.g., 'This type of system is used by companies like [Example Company] for [their use case], relevant to your interest in {user_profile.industry or 'the field'}.')", "string (e.g., 'The skills gained can be directly applied to roles such as {user_profile.career_interest or 'a relevant job title'}.')"]
}
"""
    prompt += "\nIMPORTANT: Ensure the entire response is a list of these valid JSON objects. Do not include any introductory or concluding text outside the JSON list structure. Personalize deeply.\n"
    return prompt

# --- API Endpoint ---
@app.post("/v1/generate-project-ideas", response_model=ProjectIdeaGenerationResponse, summary="Generate Personalized Project Ideas")
async def generate_project_ideas_endpoint(request_data: ProjectIdeaGenerationRequest):
    processing_start_time = time.perf_counter()

    # This conceptual prompt would be used if calling a real LLM service.
    # For the mock, we pass structured data to mock_project_llm.generate_ideas.
    conceptual_llm_prompt = construct_project_gen_prompt(
        user_profile=request_data.user_profile_snapshot,
        preferences=request_data.preferences,
        num_ideas=request_data.number_of_ideas
    )
    # print(f"DEBUG: Conceptual LLM Prompt for Project Gen:\n{conceptual_llm_prompt[:1000]}...\n") # For debugging prompt construction

    try:
        raw_ideas_from_llm = await mock_project_llm.generate_ideas(
            user_profile=request_data.user_profile_snapshot,
            preferences=request_data.preferences,
            num_ideas=request_data.number_of_ideas
        )
        
        validated_ideas: List[GeneratedProjectIdea] = []
        for raw_idea in raw_ideas_from_llm:
            try:
                # Add request_id to each idea if not already present from LLM (though mock adds it)
                raw_idea.setdefault("request_id", f"req_{uuid.uuid4().hex[:6]}")
                validated_ideas.append(GeneratedProjectIdea(**raw_idea))
            except Exception as e_val:
                print(f"Warning: Failed to validate a raw project idea from LLM mock: {raw_idea}. Error: {e_val}")
                # Optionally, collect validation errors to return in debug info

        if not validated_ideas and raw_ideas_from_llm: # If all failed validation but some were returned
             raise HTTPException(status_code=500, detail="AI service returned project ideas in an unexpected or incomplete format.")
        if not validated_ideas: # If LLM returned nothing or all failed validation and raw was empty
             raise HTTPException(status_code=404, detail="Could not generate suitable project ideas based on the provided profile and preferences. Please try adjusting your preferences or profile details.")

    except Exception as e:
        print(f"Error during project idea generation or LLM call: {e}") # Log full error
        raise HTTPException(status_code=503, detail="Error communicating with the AI project generation service. Please try again later.")

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    debug_info_dict = {
        "llm_model_name_used": MOCKED_PROJECT_LLM_NAME,
        "processing_time_ms": round(processing_time_ms, 2),
        "num_ideas_requested": request_data.number_of_ideas,
        "num_ideas_generated_valid": len(validated_ideas)
    }
    if os.getenv("UPLAS_DEBUG_MODE", "false").lower() == "true":
        # Only include potentially large/sensitive prompt in debug mode
        debug_info_dict["conceptual_llm_prompt_sent_sample"] = conceptual_llm_prompt[:1500] + "..." # Sample

    return ProjectIdeaGenerationResponse(
        generated_ideas=validated_ideas,
        debug_info=debug_info_dict
    )

# To run this FastAPI app locally (from within uplas-ai-agents/project_generator_agent/ directory):
# uvicorn main:app --reload --port 8004
