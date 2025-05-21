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
    user_id: str = Field(..., examples=["user_uuid_abc123"])
    industry: Optional[str] = Field(None, examples=["Finance", "Healthcare", "Creative Arts", "Education"])
    profession: Optional[str] = Field(None, examples=["Data Analyst", "UX Designer", "Student", "Software Developer"])
    career_interest: Optional[str] = Field(None, examples=["Machine Learning Engineer", "Game Developer", "Cloud Architect"])
    # Example: {"Python": "Intermediate", "Data Analysis": "Beginner", "Web Development": "Novice"}
    current_knowledge_level: Optional[Dict[str, str]] = Field(default_factory=dict, examples=[{"Python": "Intermediate", "SQL": "Intermediate"}])
    areas_of_interest: Optional[List[str]] = Field(default_factory=list, examples=[["NLP", "Ethical AI"], ["Sustainable Tech", "Data Visualization"]])
    learning_goals: Optional[str] = Field(None, examples=["Build a portfolio piece for data visualization.", "Understand full-stack development principles."])
    completed_course_ids: Optional[List[str]] = Field(default_factory=list, examples=["course_uuid_1", "course_uuid_2"])


class ProjectPreferences(BaseModel):
    difficulty_level: Optional[str] = Field("intermediate", examples=["beginner", "intermediate", "advanced"])
    preferred_technologies: Optional[List[str]] = Field(default_factory=list, examples=[["Python", "FastAPI"], ["JavaScript", "React"]])
    project_type_focus: Optional[str] = Field(None, examples=["Portfolio Piece", "Skill Enhancement", "Exploratory Research"])
    time_commitment_hours_estimate: Optional[int] = Field(None, examples=[20, 40], ge=5, le=100)


class ProjectIdeaGenerationRequest(BaseModel):
    user_profile_snapshot: UserProfileSnapshotForProjects
    preferences: Optional[ProjectPreferences] = Field(default_factory=ProjectPreferences)
    number_of_ideas: Optional[int] = Field(1, ge=1, le=3) # Max 3 ideas per request for this agent


class GeneratedProjectTask(BaseModel):
    task_id: int = Field(..., ge=1)
    description: str = Field(..., min_length=10)
    estimated_sub_duration: Optional[str] = Field(None, examples=["2-4 hours", "1 day"])


class GeneratedProjectIdea(BaseModel):
    request_id: Optional[str] = None # Can be linked to an initial request ID if provided by caller
    project_idea_id: str = Field(default_factory=lambda: f"proj_idea_{uuid.uuid4().hex[:10]}")
    
    title: str = Field(..., min_length=5, examples=["Personalized Financial Health Dashboard for Millennials"])
    subtitle: Optional[str] = Field(None, min_length=10, examples=["An interactive web app to track spending, set budgets, and visualize financial progress using Plaid API integration."])
    description_html: str = Field(..., min_length=50, examples=["<p>This project aims to develop a comprehensive financial health dashboard tailored for young adults...</p>"])
    difficulty_level: str = Field(..., examples=["intermediate"])
    estimated_duration: Optional[str] = Field(None, examples=["25-35 hours", "Approx. 2-3 weeks part-time"])
    
    learning_objectives_html: List[str] = Field(default_factory=list, min_items=2, examples=[["<li>Understand and implement secure API integration (e.g., Plaid).</li>"], ["<li>Master data visualization techniques for financial data using Plotly Dash.</li>"]])
    requirements_html: Optional[List[str]] = Field(default_factory=list, min_items=1, examples=[["<li>Intermediate Python skills (FastAPI/Flask preferred).</li>"], ["<li>Basic understanding of REST APIs and JSON.</li>"]])
    target_audience_html: Optional[List[str]] = Field(default_factory=list, examples=[["<li>Individuals in their 20s-30s looking to manage personal finances.</li>"], ["<li>Developers interested in FinTech applications.</li>"]])
    
    key_tasks: List[GeneratedProjectTask] = Field(default_factory=list, min_items=3)
    suggested_technologies: List[str] = Field(default_factory=list, min_items=1, examples=[["Python", "FastAPI", "Plotly Dash", "Plaid API"]])
    
    personalization_rationale: Optional[str] = Field(None, min_length=20, examples=["Given your interest in FinTech and 'Intermediate' Python knowledge, this project offers a practical way to apply API skills and data visualization."])
    potential_challenges: Optional[List[str]] = Field(default_factory=list, examples=[["Managing Plaid API rate limits and sandbox data variability."], ["Ensuring data security and privacy for financial information."]])
    assessment_rubric_preview_html: Optional[List[str]] = Field(default_factory=list, min_items=2, examples=[["<li>Core functionality (Plaid integration, dashboard display): 50%</li>"], ["<li>Code quality and API design: 30%</li>"]])
    real_world_application_examples: Optional[List[str]] = Field(default_factory=list, examples=[["Serves as a base for a personal finance SaaS product."], ["Demonstrates skills valuable for FinTech developer roles."]])


class ProjectIdeaGenerationResponse(BaseModel):
    generated_ideas: List[GeneratedProjectIdea]
    debug_info: Optional[Dict[str, Any]] = None


# --- FastAPI Application ---
app = FastAPI(
    title="Uplas AI Project Idea Generator",
    description="Generates personalized real-world project ideas using a (mocked) LLM for idea generation.",
    version="0.2.0" # Version update for more refined mock
)

# --- Mock LLM Client for Project Idea Generation (Refined as per previous step) ---
MOCKED_PROJECT_LLM_NAME = "mocked-gemini-pro-project-gen-v2.2" # From previous refinement

class MockProjectLLMClient:
    async def generate_ideas(
        self,
        user_profile: UserProfileSnapshotForProjects,
        preferences: ProjectPreferences,
        num_ideas: int
    ) -> List[Dict[str, Any]]:
        print(f"\n--- Mock Project LLM ({MOCKED_PROJECT_LLM_NAME}) Generating {num_ideas} Ideas ---")
        print(f"User Profile: Industry='{user_profile.industry}', Profession='{user_profile.profession}', Interests='{user_profile.areas_of_interest}', Knowledge='{user_profile.current_knowledge_level}'")
        print(f"Preferences: Difficulty='{preferences.difficulty_level}', Tech='{preferences.preferred_technologies}', Focus='{preferences.project_type_focus}'")
        print("---------------------------------------------------------------------\n")

        generated_project_ideas = []
        base_project_templates = [
            {"title_template": "Interactive {domain} Data Visualization Dashboard", "domain": "Data Analytics", "keywords": ["dashboard", "visualization"], "default_tech": ["Plotly Dash", "Streamlit", "Python"]},
            {"title_template": "Real-Time {asset_type} Monitoring App", "domain": "FinTech/IoT", "keywords": ["real-time", "monitoring", "api"], "default_tech": ["WebSocket", "FastAPI", "Python"]},
            {"title_template": "{commerce_type} Product Recommendation Engine", "domain": "E-commerce/AI", "keywords": ["recommendation", "machine learning"], "default_tech": ["Scikit-learn", "Python", "Pandas"]},
            {"title_template": "Personalized {content_type} News Aggregator", "domain": "Web/Content", "keywords": ["aggregator", "nlp", "scraping"], "default_tech": ["BeautifulSoup", "Flask", "Python"]},
            {"title_template": "AI-Powered {health_area} Symptom Checker (Conceptual)", "domain": "HealthTech/AI", "keywords": ["ai", "chatbot", "health"], "default_tech": ["Rasa/Dialogflow (Conceptual)", "Knowledge Base"]},
            {"title_template": "Sustainable {lifestyle_aspect} Planner", "domain": "Lifestyle/Web", "keywords": ["sustainability", "planning", "user data"], "default_tech": ["React", "Firebase"]},
            {"title_template": "Local {interest_type} Discovery Platform API", "domain": "Social/API", "keywords": ["social", "geo-location", "events", "api design"], "default_tech": ["Node.js/Express", "MongoDB Atlas"]},
            {"title_template": "{activity_type} Challenge Tracker & Leaderboard", "domain": "Health/Social", "keywords": ["gamification", "tracking", "mobile-friendly web"], "default_tech": ["Flutter/React Native (Conceptual for Web)", "Supabase"]},
            {"title_template": "Educational {subject} Quiz Game Builder", "domain": "EdTech/GameDev", "keywords": ["education", "game", "interactive content"], "default_tech": ["Phaser.js/Godot (Conceptual)", "JSON for quizzes"]},
            {"title_template": "Community {skill_type} Skill-Share Network Backend", "domain": "Social/Backend", "keywords": ["community", "skill sharing", "database design"], "default_tech": ["Django REST Framework", "PostgreSQL"]}
        ]
        
        random.shuffle(base_project_templates)

        for i in range(min(num_ideas, len(base_project_templates))):
            idea_id_suffix = uuid.uuid4().hex[:4]
            template = base_project_templates[i]
            base_title = template["title_template"]
            domain = template["domain"]
            
            difficulty = preferences.difficulty_level or random.choice(["beginner", "intermediate", "advanced"])
            
            title_fillers = {
                "{domain}": user_profile.industry or domain,
                "{asset_type}": random.choice(["Stock", "Crypto", "Sensor Data"]),
                "{commerce_type}": random.choice(["Retail", "Service Booking", "Digital Product"]),
                "{content_type}": random.choice(["Tech News", "Financial Insights", "Local Happenings"]),
                "{health_area}": random.choice(["General Wellness Query", "Fitness Activity Log", "Mental Health Check-in"]),
                "{lifestyle_aspect}": random.choice(["Meal Planning", "Sustainable Travel", "Home Energy Use"]),
                "{interest_type}": random.choice(["Community Event", "Hobby Group Finder", "Local Business Showcase"]),
                "{activity_type}": random.choice(["Fitness Steps", "Coding Hours", "Language Learning"]),
                "{subject}": random.choice(["Python Programming", "World History", "Calculus Concepts"]),
                "{skill_type}": random.choice(["Coding Language", "Foreign Language", "Crafting Skill"])
            }
            current_project_title = base_title
            for placeholder, value in title_fillers.items():
                current_project_title = current_project_title.replace(placeholder, value)
            
            project_title = f"{current_project_title}"
            personalization_notes_list = []

            current_industry = user_profile.industry or domain
            if user_profile.industry:
                project_title = f"{user_profile.industry}-Focused: {current_project_title}"
                personalization_notes_list.append(f"Directly applicable to the {user_profile.industry} sector.")
            elif domain and domain not in project_title :
                 project_title = f"{domain}-Related: {current_project_title}"
                 personalization_notes_list.append(f"Relevant to the {domain} domain.")

            if user_profile.profession:
                project_title += f" (Designed for a {user_profile.profession})"
                personalization_notes_list.append(f"Considers the perspective and potential skill application for a {user_profile.profession}.")
            
            user_interests = user_profile.areas_of_interest or []
            if user_interests:
                interest = random.choice(user_interests) if user_interests else template.get("keywords",["a key topic"])[0]
                project_title += f" (incorporating {interest.title()})"
                personalization_notes_list.append(f"Allows exploration of your stated interest in {interest}.")
            elif template.get("keywords"):
                 personalization_notes_list.append(f"Touches upon themes like {', '.join(template['keywords'])}.")
            else:
                personalization_notes_list.append("A generally useful project to build foundational and practical skills.")

            tech_stack = []
            if preferences.preferred_technologies:
                tech_stack.extend(preferences.preferred_technologies)
            else: 
                tech_stack.extend(template.get("default_tech", ["Python", "Basic Web Tech"]))

            if any(kw in project_title.lower() for kw in ["web", "dashboard", "platform", "aggregator", "network", "api", "app"]):
                if not any(t.lower() in map(str.lower, tech_stack) for t in ["FastAPI", "Flask", "Django", "React", "Vue.js", "Node.js", "Express", "HTML", "CSS"]):
                    tech_stack.append(random.choice(["FastAPI", "Flask", "Node.js/Express", "React"]))
            if any(kw in project_title.lower() for kw in ["data", "ai", "system", "monitoring", "recommendation", "analyzer", "engine", "machine learning", "nlp"]):
                 if not any(t.lower() in map(str.lower, tech_stack) for t in ["Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "NLTK"]):
                    tech_stack.append("Pandas")
                    if "ai" in project_title.lower() or "recommendation" in project_title.lower() or "nlp" in project_title.lower() or "machine learning" in project_title.lower():
                         tech_stack.append(random.choice(["Scikit-learn", "NLTK", "spaCy"]))
            
            tech_stack = sorted(list(set(tech_stack)))

            description = f"<p>Embark on a '{difficulty}' level project to develop: '{project_title}'. "
            description += f"This endeavor will provide you with substantial hands-on experience with a modern tech stack including {', '.join(tech_stack)}. "
            if user_profile.learning_goals:
                description += f"This project is designed to help you achieve your learning goal: '{user_profile.learning_goals[:70]}...'.</p>"
            else:
                description += "It's an excellent opportunity to build a practical, portfolio-worthy application while deepening your technical expertise by tackling a real-world inspired challenge.</p>"

            estimated_hours = preferences.time_commitment_hours_estimate or random.randint(15, 50)
            learning_objectives = [
                f"<li>Gain practical, hands-on experience designing and implementing solutions with {', '.join(tech_stack)}.</li>",
                f"<li>Develop a tangible project suitable for showcasing in your portfolio, with a focus on {preferences.project_type_focus or 'real-world problem solving and application development'}.</li>",
                f"<li>Enhance your skills in critical areas such as {random.choice(['API development and integration', 'data processing and analysis', 'interactive frontend design', 'backend system architecture', 'algorithmic problem-solving'])}.</li>"
            ]
            if user_interests:
                learning_objectives.append(f"<li>Deepen your understanding and application of concepts related to your interest in {random.choice(user_interests)}.</li>")
            else:
                learning_objectives.append(f"<li>Explore key concepts within the {domain} domain through direct, practical application and development.</li>")

            idea = {
                "request_id": f"req_{user_profile.user_id}_{uuid.uuid4().hex[:4]}",
                "project_idea_id": f"idea_{idea_id_suffix}_{i+1}",
                "title": project_title,
                "subtitle": f"A {difficulty} project focusing on {domain}, utilizing {tech_stack[0] if tech_stack else 'key technologies'} to address a real-world inspired scenario.",
                "description_html": description,
                "difficulty_level": difficulty,
                "estimated_duration": f"Approx. {estimated_hours} - {estimated_hours + random.randint(5,15)} hours ({random.choice(['1-2 weeks', '2-3 weeks', '3-4 weeks'])} part-time)",
                "learning_objectives_html": learning_objectives,
                "requirements_html": [
                    f"<li>{user_profile.current_knowledge_level.get(tech_stack[0].split('/')[0].strip().split(' ')[0], 'Basic to intermediate understanding of') if tech_stack and user_profile.current_knowledge_level else 'Fundamental programming concepts are recommended'}.</li>", 
                    "<li>A personal computer with internet access and a suitable code editor/IDE (e.g., VS Code).</li>",
                    "<li>A proactive learning attitude and the ability to research solutions independently using documentation and online resources.</li>"
                ],
                "key_tasks": [
                    {"task_id": 1, "description": "Phase 1: Project Initialization & Detailed Planning - Refine project scope, select specific tools/libraries from suggestions, outline core features, set up Git repository, and configure your development environment (virtual environment, linters).", "estimated_sub_duration": "2-4 hours"},
                    {"task_id": 2, "description": f"Phase 2: Core Logic Implementation - Develop the primary backend functionalities and algorithms using {tech_stack[0] if tech_stack else 'your chosen primary language/framework'}. Focus on creating clean, modular, and well-commented code for main functionalities."},
                    {"task_id": 3, "description": "Phase 3: Data Handling & Integration (if applicable) - Design necessary data schemas (if using a database). Implement data sourcing (e.g., from external APIs, CSV files, or creating mock data) and any required data transformation, cleaning, or storage logic."},
                    {"task_id": 4, "description": "Phase 4: API / User Interface Development (if applicable) - Build the necessary API endpoints for interaction if it's a backend-focused project, or develop a basic, functional user interface to demonstrate the project's capabilities if it's full-stack or frontend-focused."},
                    {"task_id": 5, "description": "Phase 5: Testing, Refinement & Basic Documentation - Write unit or integration tests for critical components to ensure correctness. Debug issues, refine features based on self-review, and create a README.md with setup, usage, and key design decisions."}
                ],
                "suggested_technologies": tech_stack,
                "personalization_rationale": " ".join(personalization_notes_list),
                "assessment_rubric_preview_html": [
                    "<li>Successful Implementation of Core Functionality & All Key Tasks (50%)</li>", 
                    f"<li>Code Quality (Readability, Organization, Modularity) & Adherence to Best Practices for {tech_stack[0] if tech_stack else 'Selected Technologies'} (30%)</li>",
                    "<li>Project Documentation (Comprehensive README.md, Clear Code Comments) and (Optional but Highly Recommended) a Brief Video Presentation or Live Demo of the Working Project (20%)</li>"
                ],
                "real_world_application_examples": [
                    f"The principles and technologies used in this project are directly applicable in the {current_industry} sector for developing innovative tools related to [example real-world task relevant to the project domain and industry].",
                    f"Completing this project can significantly enhance your portfolio, demonstrating practical skills for roles such as a {user_profile.career_interest or 'Software Developer/Engineer'}, especially if you aim to work with {', '.join(tech_stack)}."
                ]
            }
            generated_project_ideas.append(idea)
        
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
  "learning_objectives_html": ["string (e.g., '<li>Master asynchronous programming in Python, which is highly relevant for your interest in building scalable web services for the {user_profile.industry or 'tech'} sector.</li>')", "string (e.g., '<li>Learn to integrate third-party APIs effectively, a key skill for a {user_profile.profession or 'developer'} and useful for your goal of {user_profile.learning_goals or 'building X'}.</li>')"],
  "requirements_html": ["string (e.g., '<li>{user_profile.current_knowledge_level.get("Python", "Basic Python proficiency")} (variables, data types, loops, functions). If 'Beginner', suggest starting with foundational concepts.</li>')", "string (e.g., '<li>Familiarity with JSON data structures, which will be useful for your interest in {user_profile.areas_of_interest[0] if user_profile.areas_of_interest else 'API interaction'}.</li>')", "string (e.g., '<li>Access to a computer with internet, a code editor (like VS Code), and Git for version control.</li>')"],
  "target_audience_html": [ // Who is this project for, described in relation to the user
    "string (HTML list item: e.g., 'A {user_profile.profession or 'learner'} like you, aiming to strengthen their {user_profile.areas_of_interest[0] if user_profile.areas_of_interest else 'core development skills'} within the {user_profile.industry or 'chosen'} domain.')",
    "string (HTML list item: e.g., 'Individuals looking to build a compelling portfolio piece that demonstrates practical application of [key technology/skill from project].')"
  ],
  "key_tasks": [ // 3-6 clear, actionable, and somewhat sequential tasks
    {"task_id": 1, "description": "string (e.g., 'Project Planning & Environment Setup: Clearly define the project scope based on this idea. Choose specific tools/libraries from the suggestions. Initialize your project repository with Git and set up a dedicated virtual environment.')", "estimated_sub_duration": "string (e.g., '1-3 hours')"},
    {"task_id": 2, "description": "string (e.g., 'Core Logic Implementation - Part 1: Develop the primary backend functionalities and algorithms using {preferences.preferred_technologies[0] if preferences.preferred_technologies else 'Python'}. Focus on creating modular and testable functions/classes.')", "estimated_sub_duration": "string (e.g., '5-8 hours')"},
    {"task_id": 3, "description": "string (e.g., 'Data Handling & Integration: Implement data sourcing (e.g., from a mock API, CSV, or a simple database schema if needed) and any necessary data transformation or cleaning logic.')", "estimated_sub_duration": "string (e.g., '4-6 hours')"}
    // ... more tasks covering UI (if any), testing, basic documentation.
  ],
  "suggested_technologies": [ // Be specific, include versions if important, and justify briefly if not in user's preferences
    "string (e.g., 'Python 3.9+ (for its extensive libraries and readability)')",
    "string (e.g., 'FastAPI 0.100+ (for building efficient APIs, aligning with your interest in web services)')",
    "string (e.g., 'Pandas (if data manipulation is involved, essential for data tasks)')"
    // ... list other key frameworks, libraries, or tools.
  ],
  "personalization_rationale": "string (CRUCIAL: 2-3 sentences explaining *precisely* why this project, its domain, and its technologies are an excellent match for *this specific user*. Reference their industry, profession, current knowledge levels (e.g., 'builds on your Intermediate Python'), areas of interest, and learning goals. Be very specific in the connection.)",
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
      "string (e.g., 'This type of system is used by companies like [Example Company] for [their use case], relevant to your interest in {user_profile.industry or 'the field'}.')",
      "string (e.g., 'The skills gained can be directly applied to roles such as {user_profile.career_interest or 'a relevant job title'}.')"
  ]
}
"""
    prompt += "\nIMPORTANT: Ensure the entire response is a list of these valid JSON objects. Do not include any introductory or concluding text outside this JSON list structure. Each field must be populated appropriately. The personalization_rationale is key.\n"
    return prompt # This is a truncated representation; the full prompt from earlier is intended here.


# --- API Endpoint ---
@app.post("/v1/generate-project-ideas", response_model=ProjectIdeaGenerationResponse, summary="Generate Personalized Project Ideas")
async def generate_project_ideas_endpoint(request_data: ProjectIdeaGenerationRequest):
    processing_start_time = time.perf_counter()

    conceptual_llm_prompt = construct_project_gen_prompt( # Construct conceptual prompt for debugging/logging
        user_profile=request_data.user_profile_snapshot,
        preferences=request_data.preferences,
        num_ideas=request_data.number_of_ideas
    )
    
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
                raw_idea.setdefault("request_id", f"req_user_{request_data.user_profile_snapshot.user_id}_{i}")
                validated_ideas.append(GeneratedProjectIdea(**raw_idea))
            except Exception as e_val:
                print(f"Warning: Pydantic validation failed for raw project idea #{i+1} from LLM mock: {raw_idea}. Error: {e_val}")
                validation_errors_for_ideas.append({"idea_index": i, "raw_idea_title_sample": str(raw_idea.get("title", "N/A"))[:50], "error_summary": str(e_val)[:200]})

        if not validated_ideas and raw_ideas_from_llm:
             print(f"Critical Error: All {len(raw_ideas_from_llm)} ideas from LLM failed Pydantic validation. Errors: {validation_errors_for_ideas}")
             raise HTTPException(status_code=500, detail=f"AI service returned project ideas, but they were in an unexpected or incomplete format. {len(validation_errors_for_ideas)} idea(s) had validation issues. Please check agent logs.")
        if not validated_ideas:
             raise HTTPException(status_code=404, detail="Could not generate suitable project ideas based on the provided profile and preferences. Please try adjusting your input or try again later.")

    except Exception as e:
        print(f"Critical Error during project idea generation pipeline or LLM call: {e}")
        raise HTTPException(status_code=503, detail="An unexpected error occurred with the AI project generation service. Please try again later.")

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
        debug_info_dict["conceptual_llm_prompt_sent_sample"] = conceptual_llm_prompt[:2500] + ("..." if len(conceptual_llm_prompt) > 2500 else "")
        if validation_errors_for_ideas:
             debug_info_dict["validation_error_details_sample"] = validation_errors_for_ideas[:3] # Sample of errors

    return ProjectIdeaGenerationResponse(
        generated_ideas=validated_ideas,
        debug_info=debug_info_dict
    )

# To run this FastAPI app locally:
# Ensure you are in the `uplas-ai-agents/project_generator_agent/` directory.
# Then run: uvicorn main:app --reload --port 8004
