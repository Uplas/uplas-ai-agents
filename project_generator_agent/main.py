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
    number_of_ideas: Optional[int] = Field(1, ge=1, le=3) # Capped at 3 for mock diversity and performance


class GeneratedProjectTask(BaseModel):
    task_id: int
    description: str
    estimated_sub_duration: Optional[str] = None # e.g., "2-3 hours", "1 day"


class GeneratedProjectIdea(BaseModel): # This structure should align with Project.ai_generated_spec_json
    # Also matches conceptual Project model fields from Uplas Backend Integration Guide
    request_id: Optional[str] = None # For tracking, can be passed in request or generated
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
    version="0.1.1" # Incremented version after refinements
)

# --- Mock LLM Client for Project Idea Generation ---
MOCKED_PROJECT_LLM_NAME = "mocked-gemini-pro-project-gen-v2.1" # Updated mock version

class MockProjectLLMClient:
    async def generate_ideas(
        self,
        user_profile: UserProfileSnapshotForProjects,
        preferences: ProjectPreferences,
        num_ideas: int
    ) -> List[Dict[str, Any]]: # Returns a list of dicts, each conforming to GeneratedProjectIdea structure
        print(f"\n--- Mock Project LLM ({MOCKED_PROJECT_LLM_NAME}) Generating Ideas ---")
        print(f"User Profile: Industry='{user_profile.industry}', Profession='{user_profile.profession}', Interests='{user_profile.areas_of_interest}', Knowledge='{user_profile.current_knowledge_level}'")
        print(f"Preferences: Difficulty='{preferences.difficulty_level}', Tech='{preferences.preferred_technologies}', Focus='{preferences.project_type_focus}'")
        print(f"Number of Ideas Requested: {num_ideas}")
        print("---------------------------------------------------------------------\n")

        generated_project_ideas = []
        
        # More diverse base project ideas with associated domains/keywords
        base_project_templates = [
            {"title_template": "Interactive {domain} Data Visualization Dashboard", "domain": "Data Analytics", "keywords": ["dashboard", "visualization"], "default_tech": ["Plotly Dash", "Streamlit"]},
            {"title_template": "Real-Time {asset_type} Monitoring App", "domain": "FinTech/IoT", "keywords": ["real-time", "monitoring", "api"], "default_tech": ["WebSocket", "FastAPI", "TimescaleDB"]},
            {"title_template": "{commerce_type} Product Recommendation Engine", "domain": "E-commerce/AI", "keywords": ["recommendation", "machine learning", "personalization"], "default_tech": ["Scikit-learn", "Collaborative Filtering"]},
            {"title_template": "Personalized {content_type} News Aggregator", "domain": "Web/Content", "keywords": ["aggregator", "nlp", "web scraping"], "default_tech": ["BeautifulSoup", "RSS feeds", "Flask"]},
            {"title_template": "AI-Powered {health_area} Symptom Checker (Conceptual)", "domain": "HealthTech/AI", "keywords": ["ai", "chatbot", "health"], "default_tech": ["Rasa/Dialogflow (Conceptual)", "Knowledge Base"]},
            {"title_template": "Sustainable {lifestyle_aspect} Planner", "domain": "Lifestyle/Web", "keywords": ["sustainability", "planning", "user data"], "default_tech": ["React", "Firebase"]},
            {"title_template": "Local {interest_type} Discovery Platform API", "domain": "Social/API", "keywords": ["social", "geo-location", "events", "api design"], "default_tech": ["Node.js/Express", "MongoDB Atlas"]},
            {"title_template": "{activity_type} Challenge Tracker & Leaderboard", "domain": "Health/Social", "keywords": ["gamification", "tracking", "mobile-friendly web"], "default_tech": ["Flutter/React Native (Conceptual for Web)", "Supabase"]},
            {"title_template": "Educational {subject} Quiz Game Builder", "domain": "EdTech/GameDev", "keywords": ["education", "game", "interactive content"], "default_tech": ["Phaser.js/Godot (Conceptual)", "JSON for quizzes"]},
            {"title_template": "Community {skill_type} Skill-Share Network Backend", "domain": "Social/Backend", "keywords": ["community", "skill sharing", "database design"], "default_tech": ["Django REST Framework", "PostgreSQL"]}
        ]
        
        random.shuffle(base_titles_and_domains) # Corrected variable name

        for i in range(min(num_ideas, len(base_project_templates))): # Corrected variable name
            idea_id_suffix = uuid.uuid4().hex[:4]
            template = base_project_templates[i] # Corrected variable name
            base_title = template["title_template"]
            domain = template["domain"]
            
            difficulty = preferences.difficulty_level or random.choice(["beginner", "intermediate", "advanced"])
            
            # Substitute placeholders in title template
            # This is very basic, a real template engine or more complex logic would be better
            title_fillers = {
                "{domain}": user_profile.industry or domain,
                "{asset_type}": random.choice(["Stock", "Crypto", "Sensor"]),
                "{commerce_type}": random.choice(["E-commerce", "Service Booking"]),
                "{content_type}": random.choice(["Tech", "Financial", "Local"]),
                "{health_area}": random.choice(["General Wellness", "Fitness Query"]),
                "{lifestyle_aspect}": random.choice(["Meal", "Travel", "Energy Use"]),
                "{interest_type}": random.choice(["Event", "Hobby Group", "Local Business"]),
                "{activity_type}": random.choice(["Fitness", "Coding", "Learning"]),
                "{subject}": random.choice(["Python", "History", "Math"]),
                "{skill_type}": random.choice(["Coding", "Language", "Crafting"])
            }
            current_project_title = base_title
            for placeholder, value in title_fillers.items():
                current_project_title = current_project_title.replace(placeholder, value)
            
            project_title = f"{current_project_title}"
            personalization_notes_list = []

            current_industry = user_profile.industry or domain
            if user_profile.industry: # Prioritize user's stated industry
                project_title = f"{user_profile.industry}-Focused: {current_project_title}"
                personalization_notes_list.append(f"Directly applicable to the {user_profile.industry} sector.")
            elif domain: # Fallback to template's domain
                 project_title = f"{domain}-Related: {current_project_title}"
                 personalization_notes_list.append(f"Relevant to the {domain} domain.")


            if user_profile.profession:
                project_title += f" (for a {user_profile.profession})"
                personalization_notes_list.append(f"Considers the perspective and potential skill application for a {user_profile.profession}.")
            
            user_interests = user_profile.areas_of_interest or []
            if user_interests:
                interest = random.choice(user_interests) if user_interests else template["keywords"][0] if template["keywords"] else "a key topic"
                project_title += f" involving {interest.title()}"
                personalization_notes_list.append(f"Allows exploration of your stated interest in {interest}.")
            else:
                personalization_notes_list.append("A generally useful project to build foundational and practical skills.")


            tech_stack = []
            if preferences.preferred_technologies:
                tech_stack.extend(preferences.preferred_technologies)
            else: # Suggest based on domain/title if no preference
                tech_stack.extend(template.get("default_tech", ["Python"])) # Start with template default

            # Add more techs based on keywords or knowledge, ensuring some relevance
            if any(kw in project_title.lower() for kw in ["web", "dashboard", "platform", "aggregator", "network"]):
                if not any(t in tech_stack for t in ["FastAPI", "Flask", "Django", "React", "Vue.js", "Node.js"]):
                    tech_stack.append(random.choice(["FastAPI", "Flask", "Node.js/Express"]))
            if any(kw in project_title.lower() for kw in ["data", "ai", "system", "monitoring", "recommendation", "analyzer"]):
                 if not any(t in tech_stack for t in ["Pandas", "Scikit-learn", "TensorFlow", "PyTorch"]):
                    tech_stack.append("Pandas")
                    if "AI" in project_title or "Recommendation" in project_title:
                         tech_stack.append(random.choice(["Scikit-learn", "NLTK"]))
            
            tech_stack = sorted(list(set(tech_stack))) # Unique and sorted

            description = f"<p>This is a '{difficulty}' level project designed to help you build an innovative '{project_title}'. "
            description += f"You will get hands-on experience with technologies such as {', '.join(tech_stack)}. "
            if user_profile.learning_goals:
                description += f"This project directly addresses your learning goal: '{user_profile.learning_goals[:70]}...'.</p>"
            else:
                description += "It offers a fantastic opportunity to create a practical application and significantly expand your skillset by tackling a real-world inspired challenge.</p>"

            estimated_hours = preferences.time_commitment_hours_estimate or random.randint(15, 50)
            learning_objectives = [
                f"<li>Gain practical, hands-on experience with {', '.join(tech_stack)}.</li>",
                f"<li>Develop a tangible project suitable for a portfolio, focusing on {preferences.project_type_focus or 'real-world problem solving and application development'}.</li>",
                f"<li>Enhance problem-solving skills by building a project from concept to (simulated) deployment.</li>"
            ]
            if user_interests:
                learning_objectives.append(f"<li>Deepen your understanding of concepts related to your interest in {random.choice(user_interests)}.</li>")
            else:
                learning_objectives.append(f"<li>Explore key concepts within the {domain} domain through practical application.</li>")


            idea = {
                "request_id": f"req_{uuid.uuid4().hex[:6]}",
                "project_idea_id": f"idea_{idea_id_suffix}_{i+1}",
                "title": project_title,
                "subtitle": f"A {difficulty} project focusing on {domain}, utilizing {tech_stack[0] if tech_stack else 'key technologies'} to address a real-world inspired scenario.",
                "description_html": description,
                "difficulty_level": difficulty,
                "estimated_duration": f"Approx. {estimated_hours} - {estimated_hours + random.randint(5,15)} hours",
                "learning_objectives_html": learning_objectives,
                "requirements_html": [
                    f"<li>{user_profile.current_knowledge_level.get(tech_stack[0].split('/')[0].strip().split(' ')[0], 'Basic to intermediate understanding of') if tech_stack else 'Fundamental programming concepts'}.</li>",
                    "<li>Access to a computer with internet and a suitable code editor/IDE.</li>",
                    "<li>A proactive learning attitude and the ability to research solutions independently.</li>"
                ],
                "key_tasks": [
                    {"task_id": 1, "description": "Project Initialization: Define clear project scope, choose specific tools/libraries from suggestions, set up version control (Git), and create a virtual environment.", "estimated_sub_duration": "2-4 hours"},
                    {"task_id": 2, "description": f"Core Logic Implementation: Develop the primary backend functionalities using {tech_stack[0] if tech_stack else 'chosen primary language/framework'}. Focus on clean, modular code."},
                    {"task_id": 3, "description": "Data Management (if applicable): Design schema, implement data storage (e.g., local files, simple DB, or mock API), and data processing logic."},
                    {"task_id": 4, "description": "API/UI Development (if applicable): Build necessary API endpoints for interaction or a basic user interface to demonstrate functionality."},
                    {"task_id": 5, "description": "Testing and Refinement: Write unit/integration tests for core components. Debug and refine based on test results and self-review."},
                    {"task_id": 6, "description": "Documentation & Deployment Prep: Create a README with setup instructions, project overview, and (simulated) deployment steps."}
                ],
                "suggested_technologies": tech_stack,
                "personalization_rationale": " ".join(personalization_notes_list) if personalization_notes_list else f"A generally useful project to build skills in {tech_stack[0] if tech_stack else 'key areas of software development'}.",
                "assessment_rubric_preview_html": [
                    "<li>Successful Implementation of Core Functionality & Key Tasks (50%)</li>", 
                    f"<li>Code Quality, Readability, and Adherence to Best Practices for {tech_stack[0] if tech_stack else 'Selected Technologies'} (30%)</li>",
                    "<li>Project Documentation (README, Code Comments) and (Optional) Brief Presentation/Demo (20%)</li>"
                ],
                "real_world_application_examples": [
                    f"The principles used in this project are common in {current_industry} for developing tools related to [example real-world task in that industry].",
                    f"This project can serve as a strong foundation for a more complex system or a specialized tool, enhancing your portfolio for roles like a {user_profile.career_interest or 'Software Developer'}."
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
    if preferences.difficulty_level: prompt += f"- Desired Difficulty Level: {preferences.difficulty_level} (Suggest projects matching this, or state if not possible and suggest alternatives based on their knowledge level).\n"
    if preferences.preferred_technologies:
        prompt += f"- Preferred Technologies: {', '.join(preferences.preferred_technologies)}. If not suitable for the idea, suggest alternatives but try to incorporate if possible, or explain why an alternative is better for their goals/knowledge.\n"
    else:
        prompt += "- Preferred Technologies: User is open to suggestions; choose technologies appropriate for the project and user's profile (especially considering their current knowledge and interests).\n"
    if preferences.project_type_focus: prompt += f"- Desired Project Focus: {preferences.project_type_focus}\n"
    if preferences.time_commitment_hours_estimate:
        prompt += f"- User's Estimated Time Commitment: Around {preferences.time_commitment_hours_estimate} hours. Tailor project scope and number of key tasks accordingly.\n"

    # Request for Ideas and Output Structure
    prompt += f"\nTASK: Generate {num_ideas} distinct project idea(s) based on the user profile and preferences above.\n"
    prompt += "For EACH project idea, provide the output as a SINGLE, VALID JSON object with the following EXACT fields and value types (use HTML for list items in appropriate fields, e.g., '<li>Objective 1</li>'):\n"
    prompt += """
{
  "project_idea_id": "string (a unique identifier you generate for this idea, e.g., idea_xxxx)",
  "title": "string (catchy, descriptive, and highly personalized project title incorporating user's industry/profession/interests; avoid generic titles)",
  "subtitle": "string (a brief, engaging one-sentence summary of the project, highlighting personalization and key tech/skill)",
  "description_html": "string (detailed project description, 2-4 paragraphs, using <p> and <ul> for formatting. Explain what the project is about, its core challenge, and its value *to this specific user* considering their profile.)",
  "difficulty_level": "string (choose one: 'beginner', 'intermediate', 'advanced', matching user preference OR justifying if different based on their stated knowledge level and the project's complexity)",
  "estimated_duration": "string (e.g., '15-20 hours', 'Approx. 3 weeks part-time', aligning with user's time commitment if provided, otherwise a reasonable estimate for the scope)",
  "learning_objectives_html": [
    "string (HTML list item: e.g., '<li>Master asynchronous programming in Python, which is highly relevant for your interest in building scalable web services for the {user_profile.industry or 'tech'} sector.</li>')",
    "string (HTML list item: e.g., '<li>Learn to integrate third-party APIs effectively, a key skill for a {user_profile.profession or 'developer'} and useful for your goal of {user_profile.learning_goals or 'building X'}.</li>')"
    // ... 3-5 specific, measurable, achievable, relevant, time-bound (SMART-like) objectives
  ],
  "requirements_html": [
    "string (HTML list item: e.g., '<li>{user_profile.current_knowledge_level.get("Python", "Basic Python proficiency")} (variables, data types, loops, functions). If 'Beginner', suggest starting with foundational concepts.</li>')",
    "string (HTML list item: e.g., '<li>Familiarity with JSON data structures, which will be useful for your interest in {user_profile.areas_of_interest[0] if user_profile.areas_of_interest else 'API interaction'}.</li>')",
    "string (HTML list item: e.g., '<li>Access to a computer with internet, a code editor (like VS Code), and Git for version control.</li>')"
  ],
  "target_audience_html": [ // Who is this project for, described in relation to the user
    "string (HTML list item: e.g., 'A {user_profile.profession or 'learner'} like you, aiming to strengthen their {user_profile.areas_of_interest[0] if user_profile.areas_of_interest else 'core development skills'} within the {user_profile.industry or 'chosen'} domain.')",
    "string (HTML list item: e.g., 'Individuals looking to build a compelling portfolio piece that demonstrates practical application of [key technology/skill from project].')"
  ],
  "key_tasks": [ // 3-6 clear, actionable, and somewhat sequential tasks
    {"task_id": 1, "description": "string (e.g., 'Project Planning & Environment Setup: Clearly define the project scope based on your selected idea. Choose specific libraries from the suggested tech stack. Initialize your project repository with Git and set up a virtual environment.')", "estimated_sub_duration": "string (e.g., '1-3 hours')"},
    {"task_id": 2, "description": "string (e.g., 'Core Feature Implementation - Part 1: Develop the primary backend logic for [main feature A] using {preferences.preferred_technologies[0] if preferences.preferred_technologies else 'Python'}. Focus on creating modular and testable functions/classes.')", "estimated_sub_duration": "string (e.g., '5-8 hours')"},
    {"task_id": 3, "description": "string (e.g., 'Data Handling & Integration: Implement data sourcing (e.g., from a mock API, CSV, or a simple database schema if needed) and any necessary data transformation logic.')", "estimated_sub_duration": "string (e.g., '4-6 hours')"}
    // ... more tasks covering UI (if any), testing, basic documentation.
  ],
  "suggested_technologies": [ // Be specific, include versions if important, and justify briefly if not in user's preferences
    "string (e.g., 'Python 3.9+ (for its extensive libraries and readability)')",
    "string (e.g., 'FastAPI 0.100+ (for building efficient APIs, aligning with your interest in web services)')",
    "string (e.g., 'Pandas (if data manipulation is involved, essential for data tasks)')"
    // ... list other key frameworks, libraries, or tools.
  ],
  "personalization_rationale": "string (CRUCIAL: 2-3 sentences explaining *precisely* why this project, its domain, and its technologies are an excellent match for *this specific user*. Reference their industry, profession, current knowledge levels (e.g., 'builds on your Intermediate Python'), areas of interest, and learning goals. Avoid generic statements. Make the connection explicit and compelling.)",
  "potential_challenges": [ // 2-3 potential hurdles user might face
    "string (e.g., 'Debugging asynchronous code if using FastAPI/asyncio for the first time, requiring careful study of async concepts.')",
    "string (e.g., 'Finding or generating a suitable and diverse dataset if the project involves data analysis or ML.')",
    "string (e.g., 'Scope creep: It might be tempting to add too many features; focus on the core tasks first.')"
  ],
  "assessment_rubric_preview_html": [ // High-level criteria for success, 3-4 points
    "string (HTML list item: e.g., '<li>Successful implementation and demonstration of all key tasks and core project functionality (50%)</li>')",
    "string (HTML list item: e.g., '<li>Code quality, organization, readability, and adherence to best practices for the suggested technologies (30%)</li>')",
    "string (HTML list item: e.g., '<li>Clear project documentation (README with setup, usage, and key design decisions) and (if applicable) a brief video presentation of the working project (20%)</li>')"
  ],
  "real_world_application_examples": [ // 1-2 examples connecting project to real world
      "string (e.g., 'The data processing pipeline developed here is similar to those used in {user_profile.industry or 'many industries'} for ETL tasks and business intelligence.')",
      "string (e.g., 'The API design principles learned can be directly applied to building microservices, a common pattern for roles like a {user_profile.career_interest or 'Backend Developer'}.')"
  ]
}
"""
    prompt += "\nIMPORTANT: Respond with a valid JSON list containing the specified number of project idea objects. Do not include any introductory or concluding text outside this JSON list. Each field must be populated appropriately. The personalization_rationale is key.\n"
    return prompt

# --- API Endpoint ---
@app.post("/v1/generate-project-ideas", response_model=ProjectIdeaGenerationResponse, summary="Generate Personalized Project Ideas")
async def generate_project_ideas_endpoint(request_data: ProjectIdeaGenerationRequest):
    processing_start_time = time.perf_counter()

    # This conceptual prompt would be used if calling a real LLM service.
    # For the mock, we pass structured data directly to mock_project_llm.generate_ideas.
    conceptual_llm_prompt = construct_project_gen_prompt(
        user_profile=request_data.user_profile_snapshot,
        preferences=request_data.preferences,
        num_ideas=request_data.number_of_ideas
    )
    # For debugging prompt construction:
    # print(f"DEBUG: Conceptual LLM Prompt for Project Gen (First 1000 chars):\n{conceptual_llm_prompt[:1000]}...\n")

    try:
        raw_ideas_from_llm = await mock_project_llm.generate_ideas(
            user_profile=request_data.user_profile_snapshot,
            preferences=request_data.preferences,
            num_ideas=request_data.number_of_ideas
        )
        
        validated_ideas: List[GeneratedProjectIdea] = []
        validation_errors_for_ideas = []
        for i, raw_idea in enumerate(raw_ideas_from_llm):
            try:
                # Add request_id to each idea if not already present from LLM
                # (though our mock now includes it conceptually)
                raw_idea.setdefault("request_id", f"req_from_user_{request_data.user_profile_snapshot.user_id}_{i}")
                validated_ideas.append(GeneratedProjectIdea(**raw_idea))
            except Exception as e_val: # Catch Pydantic validation error for one idea
                print(f"Warning: Failed to validate raw project idea #{i+1} from LLM mock: {raw_idea}. Error: {e_val}")
                validation_errors_for_ideas.append({"idea_index": i, "raw_idea_title_sample": str(raw_idea.get("title", "N/A"))[:50], "error": str(e_val)})
                # Optionally, skip this idea or try to fix it based on error type

        if not validated_ideas and raw_ideas_from_llm: # If all failed validation but some were returned
             # Log more details about why validation failed
             print(f"Error: All {len(raw_ideas_from_llm)} ideas from LLM failed Pydantic validation. Errors: {validation_errors_for_ideas}")
             raise HTTPException(status_code=500, detail=f"AI service returned project ideas, but they were in an unexpected or incomplete format. {len(validation_errors_for_ideas)} ideas had validation issues.")
        if not validated_ideas: # If LLM returned nothing or all failed validation and raw was empty
             raise HTTPException(status_code=404, detail="Could not generate suitable project ideas based on the provided profile and preferences. Please try adjusting your preferences or profile details, or try again later.")

    except Exception as e:
        print(f"Error during project idea generation or LLM call: {e}") # Log full error
        raise HTTPException(status_code=503, detail="Error communicating with the AI project generation service. Please try again later.")

    processing_end_time = time.perf_counter()
    processing_time_ms = (processing_end_time - processing_start_time) * 1000

    debug_info_dict = {
        "llm_model_name_used": MOCKED_PROJECT_LLM_NAME,
        "processing_time_ms": round(processing_time_ms, 2),
        "num_ideas_requested": request_data.number_of_ideas,
        "num_ideas_generated_valid": len(validated_ideas),
        "ideas_validation_failures": len(validation_errors_for_ideas)
    }
    if os.getenv("UPLAS_DEBUG_MODE", "false").lower() == "true":
        # Only include potentially large/sensitive prompt in debug mode
        debug_info_dict["conceptual_llm_prompt_sent_sample"] = conceptual_llm_prompt[:2000] + "..." # Sample
        if validation_errors_for_ideas:
             debug_info_dict["validation_error_details_sample"] = validation_errors_for_ideas[:2] # Sample of errors


    return ProjectIdeaGenerationResponse(
        generated_ideas=validated_ideas,
        debug_info=debug_info_dict
    )

# To run this FastAPI app locally (from within uplas-ai-agents/project_generator_agent/ directory):
# uvicorn main:app --reload --port 8004
